"""The current simulator's loader and the world loader can replace each other's schema.

These tests drop and reload the configured database (see tests/conftest.py).
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


def _kind(connection, name: str) -> str | None:
    connection.commit()
    with connection.cursor() as cursor:
        cursor.execute("SELECT table_type FROM information_schema.tables "
                       "WHERE table_schema = DATABASE() AND table_name = %s", (name,))
        row = cursor.fetchone()
    return row[0] if row else None


@pytest.mark.reloads_mysql
def test_each_loader_replaces_either_schema() -> None:
    import pymysql

    old, new = _module("load"), _module("load_world")
    connection = pymysql.connect(**db_settings().pymysql_kwargs())
    try:
        old.create_schema(connection)
        old.create_schema(connection)  # old over old
        assert _kind(connection, "users") == "BASE TABLE"
        new.load_world(FIXTURE)  # new over old
        assert _kind(connection, "users") == "VIEW"
        assert _kind(connection, "user_devices") is None
        old.create_schema(connection)  # old over new
        assert _kind(connection, "users") == "BASE TABLE"
        assert _kind(connection, "order_attempts") is None
    finally:
        connection.close()
        new.load_world(FIXTURE)


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
