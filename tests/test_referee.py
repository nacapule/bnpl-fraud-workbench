"""The memo referee applies the fraud policy (FP-2) to the packet alone, citing a clause
for every constraint.

The expected sets come from the policy text, not from the code under test:
:func:`policy_sets` restates the §6.6 decision table and the §5.3 check outcomes, and
each case below names the facts that make its conditions hold.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from core.actions import Check, CheckOutcome
from core.evidence import CheckResult
from llm.packet import build_packet
from llm.referee import cited_ids, policy_ids, score, view

REPO = Path(__file__).resolve().parents[1]
CASE = json.loads((REPO / "tests" / "fixtures" / "llm" / "quiet_case.json").read_text())
DISPOSITIONS = {"clear", "hold", "decline", "escalate", "needs_check"}
AT = datetime(2025, 7, 3, 14, 0)


def packet(checks=(), **context):
    row = {**CASE["context_row"], **context}
    return build_packet(row, CASE["order"], merchant_category="electronics",
                        card_bin_country="US", home_country="US", checks=checks)


def policy_sets(*, settled, families, linkage, required, passed=0, failed=0, ran=0):
    """(standard, permitted, prohibited) as §6.6 and §5.3 read."""
    if settled or failed:
        if linkage:
            return {"escalate"}, {"decline"}, {"clear", "hold", "needs_check"}
        return {"decline"}, set(), {"clear", "hold", "needs_check", "escalate"}
    if families == 0 or passed == required:
        return {"clear"}, set(), {"hold", "decline", "escalate", "needs_check"}
    permitted, prohibited = set(), {"clear", "escalate"}
    (permitted if ran < required else prohibited).add("needs_check")
    (permitted if families >= 2 and passed == 0 else prohibited).add("decline")
    return {"hold"}, permitted, prohibited


R01 = {"hours_since_password_change": 10.0, "device_link_age_hours": 20.0}  # 400-day account
R03 = {"bin_ip_country_mismatch": 1, "avs_mismatch": 1}
R02_SHARED = {"accounts_on_device_30d": 5}  # more than the household exception's 4
SETTLED = {"unauthorized_disputes_lost_user": 1}

CASES = {
    # name: (context changes, checks, rules, families, settled, linkage, required checks)
    "nothing holds": ({}, (), set(), 0, False, False, set()),
    "card only": (R03, (), {"R03"}, 1, False, False, {"id_check"}),
    "account access and card": ({**R01, **R03}, (), {"R01", "R03"}, 2, False, False,
                                {"contact", "id_check"}),
    "shared device, household": ({"accounts_on_device_30d": 3}, (), {"R02"}, 0, False, False,
                                 set()),
    "shared device, no household": (R02_SHARED, (), {"R02"}, 1, False, True, {"id_check"}),
    "shared home address, household": ({"accounts_on_address_30d": 3}, (), {"R08"}, 0, False,
                                       False, set()),
    "context conditions only": ({"inr_disputes_opened_user": 2, "email_domain_class": 2}, (),
                                {"R09", "R06(a)"}, 0, False, False, set()),
    "settled": ({**R03, **SETTLED}, (), {"R03"}, 1, True, False, set()),
    "settled with linkage": ({**R02_SHARED, **SETTLED}, (), {"R02"}, 1, True, True, set()),
    "never-pay hypothesis, first order": (
        {"is_first_attempt_user": 1, "installments_due_user": 0, "installments_paid_user": 0,
         "installments_paid_share_user": 0.0, "approved_orders_user_ever": 0,
         "account_age_days": 2.0}, (), set(), 0, False, False, set()),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_referee_sets_follow_the_decision_table(name: str) -> None:
    changes, checks, rules, families, settled, linkage, required = CASES[name]
    result = view(packet(checks, **changes))
    standard, permitted, prohibited = policy_sets(
        settled=settled, families=families, linkage=linkage, required=len(required) or 1)
    assert set(result.rules) == rules
    assert len(result.families) == families
    assert (result.standard, result.permitted, result.prohibited) == (
        standard, permitted, prohibited)
    assert result.required_checks == (required if not settled and families else set())
    assert result.standard | result.permitted | result.prohibited == DISPOSITIONS
    # every constraint names its clause
    assert all(str(c).startswith(f"{c.kind} {c.subject} (FP-2 §") for c in result.constraints)
    assert {c.subject for c in result.constraints if c.kind in (
        "standard", "permitted", "prohibited")} == DISPOSITIONS


@pytest.mark.parametrize("outcome,expected_standard", [
    (CheckOutcome.FAILED, {"decline"}),
    (CheckOutcome.PASSED, {"clear"}),
    (CheckOutcome.NO_RESPONSE, {"hold"}),
])
def test_check_outcomes_decide_or_continue_the_hold(outcome, expected_standard) -> None:
    result = view(packet((CheckResult(Check.ID_CHECK, outcome, AT),), **R03))
    standard, permitted, prohibited = policy_sets(
        settled=False, families=1, linkage=False, required=1,
        passed=int(outcome == CheckOutcome.PASSED), failed=int(outcome == CheckOutcome.FAILED),
        ran=1)
    assert result.standard == standard == expected_standard
    assert (result.permitted, result.prohibited) == (permitted, prohibited)
    assert "needs_check" in result.prohibited  # the required check has run


def test_a_missing_fact_never_makes_a_condition_hold() -> None:
    result = view(packet(**{**R03, "bin_ip_country_mismatch": float("nan")}))
    assert "R03" not in result.rules
    assert result.standard == {"clear"}


def test_merchant_or_never_pay_suspicion_alone_cannot_require_decline_or_escalation() -> None:
    # A first order with nothing due (the old referee's never-pay packets) and an order with
    # only card evidence (its merchant bust-out packets) are judged on their facts.
    first_order = view(packet(**CASES["never-pay hypothesis, first order"][0]))
    assert first_order.standard == {"clear"} and first_order.no_installment_due
    card_only = view(packet(**R03))
    assert {"decline", "escalate"} <= card_only.prohibited


def memo(disposition, citations, next_check="none"):
    return {"disposition": disposition, "citations": citations, "next_check": next_check}


def test_scores_an_acceptable_hold_with_its_clause_and_check() -> None:
    referee = view(packet(**R03))
    result = score(memo("hold", ["FP-2 §6.6(b), R03"], "id_check"), referee)
    assert result.acceptable and result.standard and result.citation_ok
    assert result.next_check_ok and not result.problems


def test_needs_check_is_wrong_where_the_evidence_decides() -> None:
    decided = view(packet())
    result = score(memo("needs_check", ["FP-2 §6.6(b)"], "contact"), decided)
    assert decided.decisive and result.prohibited and not result.acceptable
    assert not result.next_check_ok
    open_case = view(packet(**R03))
    ok = score(memo("needs_check", ["FP-2 §6.6(b)", "R03"], "id_check"), open_case)
    assert ok.acceptable and ok.needs_check and ok.next_check_ok


def test_a_check_the_policy_does_not_require_is_not_the_next_check() -> None:
    result = score(memo("hold", ["§6.6(b)", "R03"], "contact"), view(packet(**R03)))
    assert result.acceptable and not result.next_check_ok


def test_prohibited_decline_on_one_family_is_wrong() -> None:
    result = score(memo("decline", ["FP-2 §6.6(b)", "R03"]), view(packet(**R03)))
    assert result.prohibited and not result.acceptable


def test_adverse_disposition_must_cite_its_clause_and_rules_that_hold() -> None:
    referee = view(packet(**R03))
    assert not score(memo("hold", ["R03"], "id_check"), referee).citation_ok
    assert not score(memo("hold", ["§6.6(b)", "R01"], "id_check"), referee).citation_ok


def test_retired_and_unknown_ids_are_invalid_citations() -> None:
    referee = view(packet(**R03))
    for citation in ("R12", "FP-1 §999", "§6.7"):
        result = score(memo("hold", ["§6.6(b)", "R03", citation], "id_check"), referee)
        assert not result.citation_ok, citation
    assert {"R06(a)", "R06(b)", "§6.6(b)", "§9(f)", "§5.3(c)"} <= policy_ids()
    assert not {"R12", "§6.7", "§4.3(b)", "§6.3(d)"} & policy_ids()


def test_non_payment_grounds_without_an_installment_due_are_a_violation() -> None:
    referee = view(packet(**CASES["never-pay hypothesis, first order"][0]))
    cited = score(memo("clear", ["§6.6(c)", "§8.3"]), referee)
    assert cited.nonpayment_violation
    assert not score(memo("clear", ["§6.6(c)"]), referee).nonpayment_violation


def test_citation_ids_are_read_from_free_text() -> None:
    assert cited_ids(["FP-2 §6.6(b), R03", "R06(b) and §5.2"]) == [
        "§6.6(b)", "R03", "R06(b)", "§5.2"]
