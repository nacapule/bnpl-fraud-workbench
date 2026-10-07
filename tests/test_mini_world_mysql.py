"""The mini world loads into MySQL with foreign keys enforced, and the
merchant-risk-screener compatibility surface keeps the columns and vocabularies it reads.

These tests drop and reload the configured database (see tests/conftest.py).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from core import world
from core.config import db_settings

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "mini_world"
SCREENER_COLUMNS = {
    "users": ["user_id", "signup_ts"],
    "orders": ["order_id", "user_id", "merchant_id", "ts", "amount", "status"],
    "plans": ["plan_id", "order_id"],
    "installments": ["plan_id", "due_ts", "outcome"],
    "chargebacks": ["order_id", "opened_ts"],
}


def _loader():
    spec = importlib.util.spec_from_file_location("load_world", REPO / "db" / "load_world.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_file_matches_the_table_specifications() -> None:
    assert (REPO / "db" / "schema.sql").read_text() == _loader().schema_sql()


@pytest.fixture(scope="module")
def loaded():
    import pymysql

    counts = _loader().load_world(FIXTURE)
    connection = pymysql.connect(**db_settings().pymysql_kwargs())
    yield counts, connection
    connection.close()


def _query(connection, sql: str, *args) -> list[tuple]:
    with connection.cursor() as cursor:
        cursor.execute(sql, args)
        return list(cursor.fetchall())


@pytest.mark.reloads_mysql
def test_every_table_loads_with_its_rows(loaded) -> None:
    counts, connection = loaded
    tables = world.read_world(FIXTURE)
    assert counts == {name: len(frame) for name, frame in tables.items()}
    columns = _query(
        connection,
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = DATABASE() ORDER BY table_name, ordinal_position",
    )
    by_table: dict[str, list[str]] = {}
    for table, column in columns:
        by_table.setdefault(table, []).append(column)
    for name, spec in world.TABLES.items():
        assert by_table[name] == list(spec.column_names), name


@pytest.mark.reloads_mysql
def test_screener_columns_exist(loaded) -> None:
    _, connection = loaded
    for table, columns in SCREENER_COLUMNS.items():
        found = {c for (c,) in _query(
            connection,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s", table)}
        assert set(columns) <= found, table


# The screener's daily merchant rollup, as it reads these five relations.
SCREENER_ROLLUP = """
WITH orders_day AS (
  SELECT o.merchant_id, DATE(o.ts) AS d, COUNT(*) AS n_orders,
         ROUND(SUM(o.amount), 2) AS gmv, ROUND(AVG(o.amount), 2) AS avg_ticket,
         ROUND(AVG(u.signup_ts > o.ts - INTERVAL 30 DAY), 3) AS new_buyer_share
  FROM orders o JOIN users u ON u.user_id = o.user_id
  WHERE o.status = 'approved'
  GROUP BY o.merchant_id, DATE(o.ts)
),
cbs_day AS (
  SELECT o.merchant_id, DATE(cb.opened_ts) AS d, COUNT(*) AS n_cbs_opened
  FROM chargebacks cb JOIN orders o ON o.order_id = cb.order_id
  GROUP BY o.merchant_id, DATE(cb.opened_ts)
),
fails_day AS (
  SELECT o.merchant_id, DATE(i.due_ts) AS d, COUNT(*) AS n_inst_failed
  FROM installments i JOIN plans p ON p.plan_id = i.plan_id
  JOIN orders o ON o.order_id = p.order_id
  WHERE i.outcome IN ('failed', 'written_off')
  GROUP BY o.merchant_id, DATE(i.due_ts)
)
SELECT SUM(od.n_orders), SUM(od.gmv),
       (SELECT SUM(n_cbs_opened) FROM cbs_day), (SELECT SUM(n_inst_failed) FROM fails_day)
FROM orders_day od
"""


@pytest.mark.reloads_mysql
def test_screener_rollup_runs_on_the_compatibility_views(loaded) -> None:
    _, connection = loaded
    ((orders, gmv, disputes, failed),) = _query(connection, SCREENER_ROLLUP)
    tables = world.read_world(FIXTURE)
    orders_table = tables["order_attempts"]
    approved = orders_table[orders_table["processor_result"] == "approved"]
    assert orders == len(approved) == 24
    assert round(float(gmv) * 100) == int(approved["amount_cents"].sum())
    assert disputes == 5
    assert failed == 24  # eight written-off plans, three installments each


@pytest.mark.reloads_mysql
def test_screener_status_vocabularies(loaded) -> None:
    """A renamed status value would silently drop rows from the screener's filters."""
    _, connection = loaded
    statuses = {s for (s,) in _query(connection, "SELECT DISTINCT status FROM orders")}
    assert statuses == {"approved", "declined"}
    outcomes = {o for (o,) in _query(connection, "SELECT DISTINCT outcome FROM installments")}
    assert outcomes <= {"pending", "paid", "late", "failed", "written_off"}
    assert outcomes == {"pending", "paid", "late", "written_off"}


@pytest.mark.reloads_mysql
def test_merchant_day_aggregation_is_derivable_without_loss(loaded) -> None:
    _, connection = loaded
    # one dated opening row per dispute, each joined to its order and merchant
    ((disputes,),) = _query(connection, "SELECT COUNT(*) FROM dispute_openings")
    ((rows,),) = _query(connection, "SELECT COUNT(*) FROM chargebacks c JOIN orders o "
                                    "USING (order_id)")
    assert rows == disputes == 5
    # every approved order has an amount and a signup no later than the first order
    ((bad_signup,),) = _query(
        connection,
        "SELECT COUNT(*) FROM users u JOIN (SELECT user_id, MIN(ts) AS first_ts FROM orders "
        "GROUP BY user_id) f USING (user_id) WHERE u.signup_ts > f.first_ts")
    assert bad_signup == 0
    ((missing_amount,),) = _query(
        connection, "SELECT COUNT(*) FROM orders WHERE status = 'approved' AND amount IS NULL")
    assert missing_amount == 0
    # post-closure disputes stay attributable to the closed merchant
    ((after_close,),) = _query(
        connection,
        "SELECT COUNT(*) FROM chargebacks c JOIN orders o USING (order_id) "
        "JOIN merchants m USING (merchant_id) WHERE m.closed_at < c.opened_ts")
    assert after_close == 1


@pytest.mark.reloads_mysql
def test_installment_outcomes_follow_payment_events(loaded) -> None:
    _, connection = loaded
    rows = dict(_query(
        connection,
        "SELECT outcome, COUNT(*) FROM installments GROUP BY outcome"))
    # 8 written-off plans x 3 installments; mo's last two are still pending; ana's bounced
    # installment was paid on retry after its due date
    assert rows["written_off"] == 24
    assert rows["pending"] == 2
    assert rows["late"] == 1


@pytest.mark.reloads_mysql
def test_foreign_keys_reject_an_unknown_card(loaded) -> None:
    import pymysql

    _, connection = loaded
    with pytest.raises(pymysql.err.IntegrityError):
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO order_attempts SELECT event_id + 100000, occurred_at, known_at, "
                "order_id + 100000, user_id, merchant_id, device_id, 999999, ship_address_id, "
                "amount_cents, promo_id, promo_discount_cents, ip, ip_country, avs_result, "
                "cvv_result, processor_result FROM order_attempts LIMIT 1")
    connection.rollback()


@pytest.mark.reloads_mysql
def test_loader_refuses_an_invalid_world() -> None:
    tables = world.read_world(FIXTURE)
    early = pd.Timedelta(days=900)  # every order before its account existed
    orders = tables["order_attempts"]
    tables["order_attempts"] = orders.assign(occurred_at=orders["occurred_at"] - early,
                                             known_at=orders["known_at"] - early)
    with pytest.raises(world.WorldError):
        _loader().load_tables(tables)
