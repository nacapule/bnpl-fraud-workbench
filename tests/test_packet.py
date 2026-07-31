"""Packet hygiene: no ground-truth leakage into what the model sees."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

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


def test_frozen_eval_packets_clean_if_present() -> None:
    pdir = REPO / "llm" / "eval" / "packets"
    files = list(pdir.glob("*.json")) if pdir.exists() else []
    for f in files[:50]:
        _assert_no_forbidden(json.loads(f.read_text()))
