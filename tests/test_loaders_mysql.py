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


def _with_partial_payment() -> tuple[dict[str, pd.DataFrame], int]:
    """fay pays 100 cents of her first plan's second installment, then nothing more."""
    builder = _module_at(FIXTURE / "build.py", "mini_world_build")
    tables = world.read_world(FIXTURE)
    orders, plans = tables["order_attempts"], tables["plans"]
    order = orders.loc[orders["amount_cents"].eq(52000), "order_id"].iloc[0]
    plan = int(plans.loc[plans["order_id"] == order, "plan_id"].iloc[0])
    schedule = tables["installment_schedule"]
    due = schedule.loc[(schedule["plan_id"] == plan) & (schedule["seq"] == 2), "due_at"].iloc[0]
    attempts = tables["payment_attempts"]
    partial = attempts[attempts["plan_id"] == plan].iloc[[0]].assign(
        event_id=10**6, occurred_at=due, known_at=due, seq=2, attempt_no=9,
        amount_cents=100, result="success")
    tables["payment_attempts"] = pd.concat([attempts, partial], ignore_index=True)
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


@pytest.mark.reloads_mysql
def test_a_partial_payment_does_not_make_an_installment_paid() -> None:
    import pymysql

    tables, plan = _with_partial_payment()
    loader = _module("load_world")
    try:
        loader.load_tables(tables)
        connection = pymysql.connect(**db_settings().pymysql_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT seq, outcome FROM installments WHERE plan_id = %s "
                               "ORDER BY seq", (plan,))
                outcomes = dict(cursor.fetchall())
        finally:
            connection.close()
        assert outcomes == {1: "written_off", 2: "written_off", 3: "written_off"}
    finally:
        loader.load_world(FIXTURE)
