"""The world loader enforces every reference, and the screener's installment view
follows the cents that stand after reversals.

Tests marked reloads_mysql drop and reload the configured database (see tests/conftest.py).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from core import ledger, world
from core.config import db_settings

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "mini_world"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "db" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _orphan(tables: dict[str, pd.DataFrame], column: str) -> dict[str, pd.DataFrame]:
    """The mini world with its first approved order pointing at a nonexistent entity."""
    changed = {name: frame.copy() for name, frame in tables.items()}
    orders = changed["order_attempts"]
    first = orders.index[orders["processor_result"] == "approved"][0]
    orders.loc[first, column] = 999_999
    return changed


REFERENCES = ["device_id", "card_id", "ship_address_id", "user_id", "merchant_id"]


@pytest.mark.parametrize("column", REFERENCES)
def test_the_loader_refuses_a_world_with_a_missing_reference(column) -> None:
    tables = _orphan(world.read_world(FIXTURE), column)
    with pytest.raises(world.WorldError) as raised:
        _module("load_world").load_tables(tables)  # refused before connecting
    assert "missing_reference" in raised.value.checks


def test_the_loader_refuses_a_payment_for_a_missing_plan() -> None:
    tables = world.read_world(FIXTURE)
    tables["payment_attempts"] = tables["payment_attempts"].copy()
    tables["payment_attempts"].loc[0, "plan_id"] = 999_999
    with pytest.raises(world.WorldError) as raised:
        _module("load_world").load_tables(tables)
    assert "missing_reference" in raised.value.checks


def _count(table: str) -> int:
    import pymysql

    connection = pymysql.connect(**db_settings().pymysql_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            return int(cursor.fetchone()[0])
    finally:
        connection.close()


@pytest.mark.reloads_mysql
@pytest.mark.parametrize("column", ["device_id", "card_id", "ship_address_id"])
def test_the_database_itself_refuses_orphaned_references(column) -> None:
    """With the validator bypassed, foreign keys still refuse the orphan and the
    whole load rolls back."""
    import pymysql

    loader = _module("load_world")
    tables = _orphan(world.read_world(FIXTURE), column)
    try:
        with pytest.raises(pymysql.err.IntegrityError, match="foreign key constraint"):
            loader.load_tables(tables, validate=False)
        assert _count("accounts") == 0
        assert _count("order_attempts") == 0
    finally:
        loader.load_world(FIXTURE)
    assert _count("order_attempts") == len(world.read_world(FIXTURE)["order_attempts"])


def _fay_plan(tables: dict[str, pd.DataFrame]) -> tuple[int, pd.Timestamp, int]:
    """fay's first plan (written off), its second installment's due date and amount."""
    orders, plans = tables["order_attempts"], tables["plans"]
    order = orders.loc[orders["amount_cents"].eq(52000), "order_id"].iloc[0]
    plan = int(plans.loc[plans["order_id"] == order, "plan_id"].iloc[0])
    schedule = tables["installment_schedule"]
    row = schedule[(schedule["plan_id"] == plan) & (schedule["seq"] == 2)].iloc[0]
    return plan, row["due_at"], int(row["amount_cents"])


def _with_moves(moves: list[tuple[str, int, int]]) -> tuple[dict[str, pd.DataFrame], int]:
    """fay's first plan, second installment, with dated moves: ("pay", hours after due,
    cents) or ("reverse", hours after due, index of the payment reversed)."""
    builder = _module_at(FIXTURE / "build.py", "mini_world_build")
    tables = world.read_world(FIXTURE)
    plan, due, _ = _fay_plan(tables)
    attempts, reversals = tables["payment_attempts"], tables["payment_reversals"]
    template = attempts[attempts["plan_id"] == plan].iloc[[0]]
    paid: list[tuple[int, int]] = []
    new_payments, new_reversals = [], []
    for n, (kind, hours, value) in enumerate(moves):
        at = due + pd.Timedelta(hours=hours)
        if kind == "pay":
            paid.append((10**6 + n, value))
            new_payments.append(template.assign(event_id=10**6 + n, occurred_at=at, known_at=at,
                                                seq=2, attempt_no=n + 2, amount_cents=value,
                                                result="success"))
        else:
            event, cents = paid[value]
            new_reversals.append(reversals.iloc[[0]].assign(
                event_id=10**6 + n, occurred_at=at, known_at=at, payment_event_id=event,
                plan_id=plan, amount_cents=cents))
    tables["payment_attempts"] = pd.concat([attempts, *new_payments], ignore_index=True)
    tables["payment_reversals"] = pd.concat([reversals, *new_reversals], ignore_index=True)
    tables["cash_events"] = ledger.derive_cash_events(tables, builder.TERMS)
    tables = world.renumber_events(tables)
    tables["labels"] = world.adjudicate(tables, horizon_days=builder.HORIZON_DAYS,
                                        observed_until=builder.OBSERVED_UNTIL,
                                        **builder.LABEL_RULES)
    return {name: world.coerce(name, frame) for name, frame in tables.items()}, plan


def _module_at(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FULL = _fay_plan(world.read_world(FIXTURE))[2]


@pytest.mark.reloads_mysql
@pytest.mark.parametrize(("moves", "outcome"), [
    ([("pay", 0, 100)], "written_off"),  # a part payment is not a paid installment
    ([("pay", 0, FULL)], "paid"),
    ([("pay", 0, FULL), ("reverse", 24, 0)], "written_off"),  # bounced and never paid again
    # paid on time, paid twice by mistake, the first bounces: never short, so paid
    ([("pay", 0, FULL), ("pay", 1, FULL), ("reverse", 2, 0)], "paid"),
    # bounced, then paid again a day later: paid late
    ([("pay", 0, FULL), ("reverse", 2, 0), ("pay", 24, FULL)], "late"),
])
def test_installment_status_follows_standing_cents_over_time(moves, outcome) -> None:
    import pymysql

    tables, plan = _with_moves(moves)
    loader = _module("load_world")
    try:
        loader.load_tables(tables)
        connection = pymysql.connect(**db_settings().pymysql_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT outcome FROM installments WHERE plan_id = %s AND seq = 2",
                               (plan,))
                ((got,),) = cursor.fetchall()
        finally:
            connection.close()
        assert got == outcome
    finally:
        loader.load_world(FIXTURE)
