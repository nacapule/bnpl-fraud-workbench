"""Packet hygiene: no ground-truth leakage into what the model sees."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from llm.eval.select_cases import assert_provenance, select_case_metadata
from llm.packet import FORBIDDEN_KEYS, _assert_no_forbidden

REPO = Path(__file__).resolve().parent.parent


def _db_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 3306), timeout=2):
            return True
    except OSError:
        return False


def test_forbidden_walker_catches_planted_label() -> None:
    with pytest.raises(AssertionError):
        _assert_no_forbidden({"account": {"nested": [{"pattern_id": "P-ATO"}]}})
    _assert_no_forbidden({"account": {"ok_field": 1}})  # clean passes


def test_forbidden_keys_cover_truth_tables() -> None:
    assert {"pattern_id", "labels", "story_id"} <= FORBIDDEN_KEYS


@pytest.mark.skipif(not _db_up(), reason="mysql not running")
def test_live_packet_has_no_forbidden_keys() -> None:
    import pandas as pd
    import sqlalchemy as sa

    from llm.packet import build_packet, get_engine

    engine = get_engine()
    try:
        with engine.connect() as c:
            alerts = pd.read_sql(sa.text("SELECT alert_id FROM alerts LIMIT 3"), c)
    except sa.exc.ProgrammingError:
        pytest.skip("alerts table not built yet (rules engine hasn't run)")
    if alerts.empty:
        pytest.skip("no alerts")
    for aid in alerts.alert_id:
        packet = build_packet(int(aid), engine)
        _assert_no_forbidden(packet)  # raises on violation
        assert packet["alert"]["fired_rules"], "packet must carry fired rules"
        json.dumps(packet)  # must be serializable

    # Appending facts after the alert is equivalent to comparing a world
    # truncated at the alert timestamp with the full world.  The packet must
    # be invariant to those future facts.
    alert_id = int(alerts.iloc[0]["alert_id"])
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            before = build_packet(alert_id, connection)
            alert = before["alert"]
            future_ts = pd.Timestamp(alert["ts"]) + pd.Timedelta(days=400)
            new_user_id = int(connection.scalar(sa.text("SELECT MAX(user_id) + 1 FROM users")))
            new_order_id = int(connection.scalar(sa.text("SELECT MAX(order_id) + 1 FROM orders")))
            new_plan_id = int(connection.scalar(sa.text("SELECT MAX(plan_id) + 1 FROM plans")))
            new_installment_id = int(
                connection.scalar(sa.text("SELECT MAX(installment_id) + 1 FROM installments"))
            )
            connection.execute(
                sa.text(
                    """INSERT INTO users
                       SELECT :new_uid, :future_ts, email, email_domain, kyc_country, dob_year
                       FROM users WHERE user_id = :uid"""
                ),
                {
                    "new_uid": new_user_id,
                    "future_ts": future_ts,
                    "uid": alert["user_id"],
                },
            )
            connection.execute(
                sa.text(
                    """INSERT INTO orders
                       SELECT :new_oid, :new_uid, merchant_id, :future_ts, amount, ip,
                              ip_country, device_id, card_id, ship_address_id, avs_result,
                              cvv_result, 'approved'
                       FROM orders WHERE order_id = :oid"""
                ),
                {
                    "new_oid": new_order_id,
                    "new_uid": new_user_id,
                    "future_ts": future_ts,
                    "oid": alert["order_id"],
                },
            )
            connection.execute(
                sa.text(
                    """INSERT INTO plans
                       VALUES (:plan_id, :order_id, 100.00, 25.00, 3)"""
                ),
                {"plan_id": new_plan_id, "order_id": new_order_id},
            )
            connection.execute(
                sa.text(
                    """INSERT INTO installments
                       VALUES (:installment_id, :plan_id, 1, :due_ts, :paid_ts,
                               25.00, 'paid')"""
                ),
                {
                    "installment_id": new_installment_id,
                    "plan_id": new_plan_id,
                    "due_ts": future_ts + pd.Timedelta(days=14),
                    "paid_ts": future_ts + pd.Timedelta(days=14),
                },
            )
            after = build_packet(alert_id, connection)
            assert after == before
        finally:
            transaction.rollback()


def test_frozen_eval_packets_clean_if_present() -> None:
    pdir = REPO / "llm" / "eval" / "packets"
    files = list(pdir.glob("*.json")) if pdir.exists() else []
    for f in files[:50]:
        _assert_no_forbidden(json.loads(f.read_text()))


def test_case_metadata_regeneration_matches_frozen_file() -> None:
    assert_provenance()
    regenerated, _ = select_case_metadata()
    committed = json.loads((REPO / "llm" / "eval" / "cases.json").read_text())
    assert regenerated == committed
