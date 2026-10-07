"""The unmatched-token check: concrete tokens in memo text looked up in the packet."""

from __future__ import annotations

import pytest

from llm.eval.tokens import (
    FP1_RULE_IDS,
    FP2_RULE_IDS,
    archived_check_texts,
    check_citations,
    check_text,
    check_texts,
    rule_ids,
    summarize,
)

PACKET = {
    "alert": {
        "order_id": 501,
        "amount": 899.99,
        "ts": "2026-02-11 03:40:00",
        "fired_rules": [
            {"id": "R08", "rationale": "ship-to address a_2899019 already used by 14 accounts"}
        ],
    },
    "account": {"tenure_days": 412, "n_orders_prior": 9, "avg_amount_prior": 101.11857},
    "last_orders": [
        {"order_id": 101, "status": "approved", "amount": 20.0},
        {"order_id": 102, "status": "approved", "amount": 30.5},
        {"order_id": 103, "status": "declined", "amount": 12.0},
    ],
}


# Identifier boundaries -------------------------------------------------------------
def test_an_id_does_not_match_inside_a_longer_identifier():
    check = check_text("device DEV_42", {"device_id": "OTHERDEV_42X"})
    assert check.classification == "unmatched"
    assert check.missing == ("DEV_42",)


@pytest.mark.parametrize("value", ["DEV_420", "XDEV_42", "DEV_42_B"])
def test_an_id_needs_a_whole_identifier_on_both_sides(value):
    assert check_text("device DEV_42", {"device_id": value}).classification == "unmatched"


def test_a_whole_identifier_matches():
    assert check_text("device DEV_42", {"device_id": "DEV_42"}).classification == "verbatim"
    rationale = PACKET["alert"]["fired_rules"][0]["rationale"]
    assert "a_2899019" in rationale
    assert check_text("address a_2899019, shared", PACKET).classification == "verbatim"


# Signs ------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["order amount -$99", "refund $-99.00", "amount −$99"])
def test_a_negative_amount_does_not_match_its_magnitude(text):
    check = check_text(text, {"amount": 99})
    assert check.classification == "unmatched"


@pytest.mark.parametrize("text", ["order amount -$99", "refund $-99.00", "amount −$99"])
def test_a_negative_amount_matches_a_negative_packet_value(text):
    assert check_text(text, {"amount": -99}).classification == "verbatim"


def test_a_signed_number_needs_the_sign_in_the_packet():
    assert check_text("tenure_days=-14", {"tenure_days": 14}).classification == "unmatched"
    assert check_text("tenure_days=-14", {"tenure_days": -14}).classification == "verbatim"
    assert check_text("tenure of 14 days", {"tenure_days": -14}).classification == "verbatim"


def test_a_signed_number_does_not_match_a_hyphen_inside_a_date():
    check = check_text("balance -07", {"ts": "2025-11-07 10:00:00"})
    assert check.classification == "unmatched"


@pytest.mark.parametrize(
    ("text", "packet"),
    [
        ("orders 1-3 approved", {"first": 1, "last": 3}),
        ("between 2025-12-09 03:03:03-03:15:03", {"ts": "2025-12-09 03:03:03", "end": "03:15:03"}),
        ("rules R03-R06 fired", {"rules": ["R03", "R06"]}),
    ],
)
def test_a_hyphen_after_a_letter_digit_or_token_is_not_a_sign(text, packet):
    assert check_text(text, packet).classification == "verbatim"


# What the check kept --------------------------------------------------------------
def test_a_number_rounded_to_the_texts_precision_matches():
    assert check_text("average amount was $101.12", PACKET).classification == "verbatim"
    assert check_text("average amount was $101.1", PACKET).classification == "verbatim"
    assert check_text("average amount was $101.13", PACKET).classification == "unmatched"


def test_trailing_zeros_keep_the_precision_the_text_claims():
    assert check_text("amount $99.00", {"amount": 99.49}).classification == "unmatched"
    assert check_text("amount -$99.00", {"amount": -99.49}).classification == "unmatched"
    assert check_text("amount $99.00", {"amount": 99}).classification == "verbatim"
    assert check_text("amount $99.00", {"amount": 99.004}).classification == "verbatim"


def test_a_minus_after_a_bracket_in_the_packet_is_not_a_sign():
    assert check_text("balance -99", {"range": "[1]-99"}).classification == "unmatched"


def test_digits_inside_a_longer_number_do_not_match():
    for invented in ("7 linked accounts", "12 prior orders", "99 days"):
        assert check_text(invented, {"amount": 1370.55, "tenure": 412, "x": 899.99}).missing


def test_money_with_thousands_separators_matches():
    assert check_text("aggregate $1,370.55", {"amount": 1370.55}).classification == "verbatim"


def test_a_count_over_a_named_list_is_derived():
    check = check_text("last_orders holds 2 approved orders", PACKET)
    assert check.classification == "derived"
    assert check.derived == ("2",)
    assert check.missing == ()


def test_a_sum_over_a_named_list_is_derived_and_other_numbers_are_not():
    assert check_text("last_orders total amount 62.5", PACKET).classification == "derived"
    assert check_text("last_orders total amount 63", PACKET).classification == "unmatched"


def test_timestamps_are_one_token():
    check = check_text("order at 2026-02-11 03:40:00", PACKET)
    assert check.tokens == ("2026-02-11 03:40:00",)
    assert check.classification == "verbatim"


def test_text_without_tokens_is_verbatim_and_unchecked():
    check = check_text("no prior chargebacks and a stable device", PACKET)
    assert check.tokens == ()
    assert check.classification == "verbatim"


def test_policy_references_are_not_packet_tokens():
    check = check_text("permitted under FP-2 §6.6(b) and FP-1 §6.1", {"amount": 5})
    assert check.tokens == ()


def test_a_real_number_in_a_false_relationship_still_passes():
    # The scope sentence: a token check, not semantic truth.
    assert check_text("2 prior chargebacks", {"tenure_days": 2}).classification == "verbatim"


def test_non_text_is_rejected():
    with pytest.raises(TypeError):
        check_text(9, PACKET)  # type: ignore[arg-type]


def test_summarize_counts_texts():
    checks = check_texts(
        ["tenure 412 days", "no tokens here", "last_orders holds 2 approved orders", "7 devices"],
        PACKET,
    )
    assert summarize(checks) == {"n_texts": 4, "n_with_tokens": 3, "n_derived": 1, "n_unmatched": 1}


def test_the_archived_check_keeps_the_archived_behaviour():
    # Used only to reproduce the archived numbers.
    old = archived_check_texts(
        ["device DEV_42", "order amount -$99", "amount $99.00"],
        {"d": "OTHERDEV_42X", "a": 99, "b": 98.6},
    )
    assert [check.classification for check in old] == ["verbatim", "verbatim", "verbatim"]


# Citations --------------------------------------------------------------------------
def test_rule_ids_in_a_citation():
    assert rule_ids("R03 and R06 (FP-1 §5)") == ("R03", "R06")
    assert rule_ids("FP-1 §6.1") == ()


def test_a_citation_without_a_rule_id_is_neither_valid_nor_invalid():
    [entry] = check_citations(["FP-1 §999"], FP1_RULE_IDS, {"R03"})
    assert entry["status"] == "no rule id"
    assert entry["valid"] is None and entry["fired"] is None


def test_r12_is_valid_for_archived_memos_and_not_under_fp2():
    fired = {"R03"}
    assert check_citations(["R12"], FP1_RULE_IDS, fired)[0]["valid"] is True
    assert check_citations(["R12"], FP2_RULE_IDS, fired)[0]["valid"] is False


def test_each_rule_id_is_checked_and_marked_fired():
    entries = check_citations(["R03, R99"], FP1_RULE_IDS, {"R03"})
    assert [(e["rule_id"], e["status"], e["fired"]) for e in entries] == [
        ("R03", "valid", True),
        ("R99", "invalid", False),
    ]
