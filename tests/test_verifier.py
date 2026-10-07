"""Structured claims are checked against the packet field they name; derived claims are
recomputed from their declared inputs."""

from __future__ import annotations

import pytest

from llm.eval.verifier import MISSING, check_claim, recompute, resolve, value_matches, verify_memo

PACKET = {
    "decision": {"decision_at": "2025-07-03 14:05:00",
                 "checks": [{"check": "id_check", "outcome": "passed"}]},
    "order": {"order": "O1", "ip_country": "US", "promotion_used": False},
    "context": {"amount_cents": 89999, "device_link_age_hours": 71.99972222,
                "installments_due_user": 9, "installments_paid_user": 7,
                "accounts_on_device_30d": 3, "home_address_age_days": None,
                "amount_over_category_p95": 1.2345},
}


def claim(field, value, derived=None):
    return {"field": field, "value": value, "statement": "s", "derived": derived}


def test_paths_resolve_into_nested_fields_and_lists() -> None:
    assert resolve(PACKET, "context.amount_cents") == 89999
    assert resolve(PACKET, "decision.checks[0].outcome") == "passed"
    for path in ("context.nope", "decision.checks[3].outcome", "context", "order.ip_country.x",
                 "context.amount cents"):
        assert resolve(PACKET, path) is MISSING or isinstance(resolve(PACKET, path), dict)


@pytest.mark.parametrize("field,value,status", [
    ("context.amount_cents", "89999", "verified"),
    ("context.amount_cents", "89,999", "verified"),
    ("context.amount_cents", 89999, "verified"),
    ("context.amount_cents", "89998", "wrong_value"),
    ("context.amount_cents", "-89999", "wrong_value"),       # a sign is part of the value
    ("context.device_link_age_hours", "72.0", "verified"),     # rounded at the claim's precision
    ("context.device_link_age_hours", "72", "verified"),
    ("context.device_link_age_hours", "71.9", "wrong_value"),
    ("context.amount_over_category_p95", "1.23", "verified"),
    ("context.amount_over_category_p95", "1.24", "wrong_value"),
    ("context.home_address_age_days", "null", "verified"),
    ("context.home_address_age_days", "0", "wrong_value"),
    ("order.ip_country", "us", "verified"),
    ("order.ip_country", "CA", "wrong_value"),
    ("order.promotion_used", "false", "verified"),
    ("order.promotion_used", "true", "wrong_value"),
    ("decision.checks[0].outcome", "passed", "verified"),
    ("context.accounts_on_device_30d", "yes", "wrong_value"),
    ("context.devices_on_account", "3", "unknown_field"),     # an invented field
    ("context", "{}", "unknown_field"),                         # not a single fact
    ("decision.checks", [{"check": "id_check", "outcome": "passed"}], "verified"),
    ("decision.checks", [], "wrong_value"),
    ("decision.checks", '[{"check": "id_check", "outcome": "passed"}]', "verified"),
    ("decision.checks", "[]", "wrong_value"),
])
def test_plain_claims(field, value, status) -> None:
    assert check_claim(claim(field, value), PACKET).status == status


@pytest.mark.parametrize("operation,inputs,value,status", [
    ("difference", ["context.installments_due_user", "context.installments_paid_user"], "2",
     "derived_ok"),
    ("difference", ["context.installments_due_user", "context.installments_paid_user"], "3",
     "derived_wrong"),
    ("sum", ["context.installments_due_user", "context.installments_paid_user"], "16",
     "derived_ok"),
    ("ratio", ["context.installments_paid_user", "context.installments_due_user"], "0.78",
     "derived_ok"),
    ("ratio", ["context.installments_paid_user", "context.installments_due_user"], "0.7",
     "derived_wrong"),
    ("max", ["context.installments_due_user", "context.accounts_on_device_30d"], "9",
     "derived_ok"),
    ("count", ["decision.checks"], "1", "derived_ok"),
    ("difference", ["context.installments_due_user", "context.no_such_field"], "2",
     "derived_invalid"),
    ("ratio", ["context.amount_cents", "order.ip_country"], "1", "derived_invalid"),
    ("difference", ["context.installments_due_user"], "9", "derived_invalid"),
])
def test_derived_claims_are_recomputed(operation, inputs, value, status) -> None:
    derived = {"operation": operation, "inputs": inputs}
    assert check_claim(claim("result", value, derived), PACKET).status == status


def test_value_matching_edge_cases() -> None:
    assert value_matches("$1,234.50", 1234.5)
    assert not value_matches("1234.5", float("nan"))
    assert value_matches(None, None)
    assert not value_matches(["1"], 1)
    assert recompute("ratio", [1, 0]) is None


def test_memo_report_counts_claim_errors_and_unmatched_tokens() -> None:
    memo = {
        "claims": [claim("context.amount_cents", "89999"),
                   claim("context.accounts_on_device_30d", "4")],
        "hypotheses": [{"explanation": "household", "likelihood": "low",
                        "reasoning": "Shared by 3 accounts."}],
        "memo": "Device shared by 41 accounts; amount 89999 cents.",
    }
    report = verify_memo(memo, PACKET)
    assert report["n_claims"] == 2 and report["n_claim_errors"] == 1
    assert report["claim_errors"][0]["status"] == "wrong_value"
    assert report["has_unmatched_token"]  # the memo's "41" appears nowhere in the packet
