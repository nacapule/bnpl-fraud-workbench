"""A generated world loads into MySQL and serves the merchant-risk-screener's tables.

The screener reads orders, users, chargebacks, installments and plans; a
merchant/day rollup over them must lose no dispute, including those opened after a
bust-out merchant closed. These tests drop and reload the configured database.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core import world
from core.config import db_settings
from simulator.generate import generate_world

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "mini_world"

# The screener's daily rollup (monitor/metrics.sql), restricted to the columns it reads.
ROLLUP_DISPUTES = """
SELECT o.merchant_id, DATE(cb.opened_ts) AS d, COUNT(*) AS n
FROM chargebacks cb JOIN orders o ON o.order_id = cb.order_id
GROUP BY o.merchant_id, DATE(cb.opened_ts)
"""


def _loader():
    spec = importlib.util.spec_from_file_location("load_world", REPO / "db" / "load_world.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    import pymysql

    directory = tmp_path_factory.mktemp("world")
    manifest = generate_world(416, "baseline", directory, scale=0.02)
    loader = _loader()
    counts = loader.load_world(directory)
    connection = pymysql.connect(**db_settings().pymysql_kwargs())
    yield world.read_world(directory), manifest, counts, connection
    connection.close()
    loader.load_world(FIXTURE)


def _rows(connection, sql: str, *args) -> list[tuple]:
    with connection.cursor() as cursor:
        cursor.execute(sql, args)
        return list(cursor.fetchall())


@pytest.mark.reloads_mysql
def test_a_generated_world_loads_completely(loaded) -> None:
    tables, _, counts, _ = loaded
    assert counts == {name: len(frame) for name, frame in tables.items()}


@pytest.mark.reloads_mysql
def test_screener_views_keep_every_dispute_and_order(loaded) -> None:
    tables, _, _, connection = loaded
    ((orders,),) = _rows(connection, "SELECT COUNT(*) FROM orders WHERE status = 'approved'")
    assert orders == len(tables["plans"])
    ((disputes,),) = _rows(connection, "SELECT COUNT(*) FROM chargebacks")
    assert disputes == len(tables["dispute_openings"])
    assert sum(n for _, _, n in _rows(connection, ROLLUP_DISPUTES)) == disputes
    ((early,),) = _rows(connection, """
        SELECT COUNT(*) FROM orders o JOIN users u ON u.user_id = o.user_id
        WHERE u.signup_ts > o.ts""")
    assert early == 0
    statuses = {s for (s,) in _rows(connection, "SELECT DISTINCT outcome FROM installments")}
    assert statuses <= {"paid", "late", "failed", "written_off", "pending"}


@pytest.mark.reloads_mysql
def test_bustout_merchants_have_disputes_after_closing(loaded) -> None:
    tables, manifest, _, connection = loaded
    bustouts = manifest["evaluation_only"]["bustout_merchant_ids"]
    closed = tables["merchants"].dropna(subset=["closed_at"])
    assert sorted(closed["merchant_id"]) == bustouts
    rows = _rows(connection, """
        SELECT o.merchant_id, COUNT(*) FROM chargebacks cb
        JOIN orders o ON o.order_id = cb.order_id
        JOIN merchants m ON m.merchant_id = o.merchant_id
        WHERE m.closed_at IS NOT NULL AND cb.opened_ts > m.closed_at
        GROUP BY o.merchant_id""")
    assert {merchant for merchant, _ in rows} == set(bustouts)
