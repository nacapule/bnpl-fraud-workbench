"""Packets carry the as-of facts at the decision time, placeholders instead of ids, and
nothing from labels or simulation truth."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from core.actions import Check, CheckOutcome
from core.evidence import CheckResult
from llm.packet import (
    CONTEXT_FIELDS,
    ENTITIES,
    assert_no_forbidden,
    build_packet,
    build_packets,
    checks_of,
    context_row,
    placeholders,
)

REPO = Path(__file__).resolve().parents[1]
CASE = json.loads((REPO / "tests" / "fixtures" / "llm" / "quiet_case.json").read_text())


def packet(**changes):
    row = {**CASE["context_row"], **changes.pop("context", {})}
    return build_packet(row, CASE["order"], merchant_category=CASE["merchant_category"],
                        card_bin_country=CASE["card_bin_country"],
                        home_country=CASE["home_country"], **changes)


def test_packet_carries_every_context_fact_at_the_decision() -> None:
    built = packet()
    assert list(built["context"]) == list(CONTEXT_FIELDS)
    assert built["decision"] == {"point": "review", "decision_at": "2025-07-03 14:05:00",
                                 "checks": []}
    assert built["order"]["placed_at"] == "2025-07-03 13:40:12"
    assert built["order"]["promotion_used"] is False
    json.dumps(built)  # serializable


def test_entity_ids_never_reach_the_packet() -> None:
    text = json.dumps(packet())
    for key in ("order_id", "user_id", "merchant_id", "device_id", "card_id", "ship_address_id"):
        assert f'"{key}"' not in text
    for raw in ("5012", "377", "9001", "4410", "8120", "10.1.2.3"):
        assert raw not in text
    assert {packet()["order"][entity] for entity in ENTITIES} == {
        "O1", "U1", "M1", "D1", "C1", "A1"}


def test_fresh_placeholders_are_distinct_and_reproducible() -> None:
    first, again, other = placeholders(11), placeholders(11), placeholders(12)
    assert first == again != other
    assert len(set(first.values())) == len(ENTITIES)
    renamed = packet(names=first)
    assert renamed["context"] == packet()["context"]
    assert renamed["order"]["order"] == first["order"]


def test_values_keep_full_precision() -> None:
    # rounding 72.0001 hours to 72 would make R01's "within 72 h" hold on the packet
    # while it does not hold on the context row
    built = packet(context={"device_link_age_hours": 72.0001})
    assert built["context"]["device_link_age_hours"] == 72.0001
    assert isinstance(built["context"]["attempts_user_24h"], int)


def test_missing_values_stay_missing() -> None:
    built = packet(context={"home_address_age_days": float("nan")})
    assert built["context"]["home_address_age_days"] is None
    assert pd.isna(context_row(built)["home_address_age_days"])


def test_checks_are_recorded_in_completion_order() -> None:
    checks = (
        CheckResult(Check.ID_CHECK, CheckOutcome.PASSED, datetime(2025, 7, 3, 14, 0)),
        CheckResult(Check.CONTACT, CheckOutcome.NO_RESPONSE, datetime(2025, 7, 3, 13, 50)),
    )
    built = packet(checks=checks)
    assert built["decision"]["point"] == "check_completed"
    assert [item["check"] for item in built["decision"]["checks"]] == ["contact", "id_check"]
    assert {(r.check, r.outcome) for r in checks_of(built)} == {
        (r.check, r.outcome) for r in checks}


@pytest.mark.parametrize("problem", ["decision_before_order", "check_after_decision",
                                     "check_twice", "other_order", "missing_fact"])
def test_inconsistent_inputs_are_refused(problem: str) -> None:
    late = CheckResult(Check.CONTACT, CheckOutcome.PASSED, datetime(2025, 7, 3, 15, 0))
    early = CheckResult(Check.CONTACT, CheckOutcome.PASSED, datetime(2025, 7, 3, 13, 45))
    with pytest.raises((ValueError, KeyError)):
        if problem == "decision_before_order":
            packet(context={"decision_at": "2025-07-03 13:00:00"})
        elif problem == "check_after_decision":
            packet(checks=(late,))
        elif problem == "check_twice":
            packet(checks=(early, early))
        elif problem == "other_order":
            packet(context={"order_id": 1})
        else:
            row = dict(CASE["context_row"])
            del row["accounts_on_device_30d"]
            build_packet(row, CASE["order"], merchant_category="x", card_bin_country="US",
                         home_country="US")


def test_forbidden_walker_catches_planted_truth() -> None:
    with pytest.raises(AssertionError):
        assert_no_forbidden({"context": {"nested": [{"pattern_id": "P-ATO"}]}})
    with pytest.raises(AssertionError):
        assert_no_forbidden({"label_known_at": "2025-01-01"})
    assert_no_forbidden(packet())


def test_packets_for_context_rows_read_only_the_order_and_its_entities() -> None:
    order = CASE["order"]
    tables = {
        "order_attempts": pd.DataFrame([order]),
        "accounts": pd.DataFrame([{"user_id": 377, "home_country": "CA"}]),
        "cards": pd.DataFrame([{"card_id": 4410, "bin_country": "GB"}]),
        "merchants": pd.DataFrame([{"merchant_id": 12, "category": "travel"}]),
        "labels": None,  # never read
    }
    context = pd.DataFrame([CASE["context_row"]])
    built = build_packets(tables, context)
    (only,) = built.values()
    assert only["order"]["account_home_country"] == "CA"
    assert only["order"]["card_bin_country"] == "GB"
    assert only["order"]["merchant_category"] == "travel"


def test_packets_from_a_generated_world_carry_its_as_of_context(tmp_path: Path) -> None:
    # a small simulated world, its as-of context at checkout and the next morning, and
    # the packets: the referee's reading of each packet (after a JSON round trip) holds
    # the same policy conditions as the evidence module's reading of the context row
    from core import asof, evidence
    from core.world import read_world
    from llm import referee
    from simulator.generate import generate_world

    generate_world(416, "baseline", tmp_path, scale=0.02)
    tables = read_world(tmp_path)
    attempts = tables["order_attempts"]
    approved = attempts[attempts["processor_result"].eq("approved")]
    fraud = approved["order_id"].isin(
        tables["latent_orders"].dropna(subset=["pattern_id"])["order_id"])
    chosen = pd.concat([approved[fraud].head(20), approved[~fraud].sample(40, random_state=1)])
    placed = pd.to_datetime(chosen["known_at"])
    decisions = pd.concat([
        pd.DataFrame({"order_id": chosen["order_id"], "decision_at": placed}),
        pd.DataFrame({"order_id": chosen["order_id"],
                      "decision_at": (placed + pd.Timedelta(days=1)).dt.floor("D")}),
    ], ignore_index=True)
    context = asof.build_context(tables, decisions)
    built = build_packets(tables, context)
    assert len(built) == len(context) == len(decisions)
    held = evidence.conditions(context)
    rules = list(evidence.RULE_FAMILY)
    assert held[rules].to_numpy().any()  # some condition holds, so the comparison bites
    for index, row in enumerate(context.to_dict("records")):
        shown = json.loads(json.dumps(built[(int(row["order_id"]),
                                             pd.Timestamp(row["decision_at"]))]))
        assert_no_forbidden(shown)
        assert set(shown["context"]) == set(CONTEXT_FIELDS)
        expected = {rule for rule in rules if bool(held.iloc[index][rule])}
        assert set(referee.view(shown).rules) == expected
