"""Evidence classification and the decision table against the fraud policy's own text.

The expected results are written from policy/fraud-policy.md (FP-2 §5, §6), not from
core.evidence: each case states the facts, and the disposition the policy requires.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import asof, evidence
from core.actions import Check, CheckOutcome
from core.evidence import CheckResult, Family

REPO = Path(__file__).resolve().parents[1]
AT = datetime(2025, 3, 1, 12)

# An order on which no condition, exception, settlement or benign explanation holds: a
# two-year-old account ordering from its home country, on its old device, to its home.
QUIET = {
    "hours_since_password_change": 10_000.0, "hours_since_password_reset": 10_000.0,
    "hours_since_email_change": 10_000.0, "hours_since_phone_change": 10_000.0,
    "hours_since_credential_change": 10_000.0,
    "account_age_days": 700.0, "device_link_age_hours": 9_000.0,
    "geo_kmh_from_previous_attempt": 0.0, "bin_ip_country_mismatch": 0, "avs_mismatch": 0,
    "cvv_mismatch": 0, "processor_declines_card_24h": 0, "processor_declines_device_24h": 0,
    "attempts_user_24h": 1, "attempts_device_24h": 1, "accounts_on_device_30d": 1,
    "accounts_on_address_30d": 1, "email_root_other_accounts": 0,
    "promo_uses_linked_accounts": 0, "email_domain_class": 0, "is_first_attempt_user": 0,
    "amount_over_category_p95": 0.4, "inr_disputes_opened_user": 0, "ship_to_home": 1,
    "home_address_age_days": 600.0, "ship_address_link_age_hours": 14_000.0,
    "ip_country_not_home": 0, "unauthorized_disputes_lost_user": 0, "victim_reports_user": 0,
    "never_pay_determined_user": 0, "inr_claims_rejected_user": 0,
    "installments_due_user": 0,
}


def row(**changes: object) -> dict[str, object]:
    unknown = set(changes) - set(QUIET)
    assert not unknown, unknown
    return {**QUIET, **changes}


def rules(values: dict[str, object]) -> set[str]:
    return {condition.rule for condition in evidence.classify(values, ()).conditions}


def check(name: Check, outcome: CheckOutcome) -> CheckResult:
    return CheckResult(name, outcome, AT)


# ---------------------------------------------------------------- §6.2 conditions


@pytest.mark.parametrize(("changes", "holds"), [
    # R01: a credential change in the 48 h, account at least 90 days, device new in 72 h
    ({"hours_since_password_reset": 48.0, "account_age_days": 90.0,
      "device_link_age_hours": 72.0}, {"R01"}),
    ({"hours_since_email_change": 30.0, "account_age_days": 400.0,
      "device_link_age_hours": 5.0}, {"R01"}),
    ({"hours_since_password_change": 48.5, "account_age_days": 400.0,
      "device_link_age_hours": 5.0}, set()),
    ({"hours_since_password_change": 2.0, "account_age_days": 89.9,
      "device_link_age_hours": 5.0}, set()),
    ({"hours_since_password_change": 2.0, "account_age_days": 400.0,
      "device_link_age_hours": 72.5}, set()),
    # R11 over 900 km/h; R03 geography with AVS or CVV; R07 three declines
    ({"geo_kmh_from_previous_attempt": 900.5}, {"R11"}),
    ({"geo_kmh_from_previous_attempt": 900.0}, set()),
    ({"bin_ip_country_mismatch": 1, "cvv_mismatch": 1}, {"R03"}),
    ({"bin_ip_country_mismatch": 1}, set()),
    ({"processor_declines_device_24h": 3}, {"R07"}),
    ({"processor_declines_card_24h": 2, "processor_declines_device_24h": 2}, set()),
    # R05: more than 3 on the account or more than 5 on the device
    ({"attempts_user_24h": 4}, {"R05"}),
    ({"attempts_user_24h": 3, "attempts_device_24h": 5}, set()),
    ({"attempts_device_24h": 6}, {"R05"}),
    # Linkage: R02, R08 at 3 accounts; R06(b) any other holder; R10 at 3 accounts
    ({"accounts_on_device_30d": 3}, {"R02"}),
    ({"accounts_on_address_30d": 3, "ship_to_home": 0}, {"R08"}),
    ({"accounts_on_device_30d": 2, "accounts_on_address_30d": 2}, set()),
    ({"email_root_other_accounts": 1}, {"R06(b)"}),
    ({"promo_uses_linked_accounts": 3}, {"R10"}),
    ({"promo_uses_linked_accounts": 2}, set()),
    # Context: R06(a), R04 (all three parts), R09
    ({"email_domain_class": 2}, {"R06(a)"}),
    ({"email_domain_class": 1}, set()),
    ({"is_first_attempt_user": 1, "amount_over_category_p95": 1.2,
      "account_age_days": 6.9}, {"R04"}),
    ({"is_first_attempt_user": 1, "amount_over_category_p95": 1.2,
      "account_age_days": 7.0}, set()),
    ({"inr_disputes_opened_user": 2}, {"R09"}),
])
def test_each_condition_holds_exactly_as_the_policy_states(
    changes: dict[str, object], holds: set[str]
) -> None:
    assert rules(row(**changes)) == holds


def test_conditions_carry_their_family_and_the_columns_they_rest_on() -> None:
    ev = evidence.classify(row(bin_ip_country_mismatch=1, avs_mismatch=1,
                               accounts_on_device_30d=5), ())
    by_rule = {condition.rule: condition for condition in ev.conditions}
    assert by_rule["R03"].family is Family.CARD
    assert by_rule["R02"].family is Family.LINKAGE
    assert "bin_ip_country_mismatch" in by_rule["R03"].columns
    assert ev.families_present == {Family.CARD, Family.LINKAGE}


# ---------------------------------------------------------------- §6.4 households


def test_old_shared_device_is_a_household_up_to_four_accounts() -> None:
    old = 24 * 90.0
    for accounts, counted in ((3, False), (4, False), (5, True)):
        ev = evidence.classify(row(accounts_on_device_30d=accounts,
                                   device_link_age_hours=old), ())
        assert {c.rule for c in ev.conditions} == {"R02"}  # priority still sees it (§7.1)
        assert (Family.LINKAGE in ev.families_present) is counted
    young = evidence.classify(row(accounts_on_device_30d=3, device_link_age_hours=old - 1), ())
    assert Family.LINKAGE in young.families_present


def test_shared_home_address_is_a_household_only_when_it_is_home_and_old() -> None:
    shared = {"accounts_on_address_30d": 4}
    home = evidence.classify(row(**shared), ())
    assert [c.clause for c in home.household_exceptions] == ["§6.4(b)"]
    assert home.families_present == frozenset()
    for changes in ({"ship_to_home": 0}, {"home_address_age_days": 89.0},
                    {"accounts_on_address_30d": 5}):
        ev = evidence.classify(row(**{**shared, **changes}), ())
        assert ev.families_present == {Family.LINKAGE}, changes


# ---------------------------------------------------------------- §6.3 and §6.5


def test_earlier_outcomes_that_settle_the_order() -> None:
    def settled(**changes: object) -> list[str]:
        return [c.clause for c in evidence.classify(row(**changes), ()).settlements]

    assert settled(unauthorized_disputes_lost_user=1) == ["§6.3(a)"]
    # the holder reported a takeover: the lost dispute was the victim's, not theirs
    assert settled(unauthorized_disputes_lost_user=1, victim_reports_user=1) == []
    assert settled(never_pay_determined_user=1) == ["§6.3(b)"]
    assert settled(inr_claims_rejected_user=2) == ["§6.3(c)"]
    assert settled(inr_claims_rejected_user=1, inr_disputes_opened_user=3) == []


def test_benign_explanations_are_cited_but_never_count() -> None:
    traveller = evidence.classify(row(ip_country_not_home=1, avs_mismatch=1), ())
    assert {c.clause for c in traveller.benign} == {"§6.5(a)", "§6.5(b)"}
    assert traveller.families_present == frozenset()
    reset_after_new_phone = evidence.classify(
        row(hours_since_password_reset=10.0, hours_since_phone_change=30.0), ())
    assert {c.clause for c in reset_after_new_phone.benign} == {"§6.5(b)"}
    large_first = evidence.classify(row(is_first_attempt_user=1, amount_over_category_p95=3.0,
                                        account_age_days=40.0), ())
    assert {c.clause for c in large_first.benign} == {"§6.5(c)"}
    r04 = evidence.classify(row(is_first_attempt_user=1, amount_over_category_p95=3.0,
                                account_age_days=1.0), ())
    assert {c.clause for c in r04.benign} == set()  # R04 itself covers it
    history = evidence.classify(row(installments_due_user=3), ())
    assert {c.clause for c in history.benign} == {"§6.5(f)"}
    assert permitted(row(installments_due_user=3, ip_country_not_home=1)).standard == {"clear"}


# ---------------------------------------------------------------- §6.6 and §5.3


def permitted(values: dict[str, object], *checks: CheckResult) -> evidence.PermittedActions:
    return evidence.permitted_actions(evidence.classify(values, checks))


CARD = {"bin_ip_country_mismatch": 1, "avs_mismatch": 1}
ACCESS = {"hours_since_password_reset": 3.0, "device_link_age_hours": 3.0}
LINKAGE = {"accounts_on_device_30d": 6}


def test_no_adverse_family_clears_and_context_never_holds() -> None:
    for values in (row(), row(email_domain_class=2, inr_disputes_opened_user=4),
                   row(is_first_attempt_user=1, amount_over_category_p95=5.0,
                       account_age_days=0.5)):
        result = permitted(values)
        assert result.row == "§6.6(c)" and result.standard == {"clear"}
        assert result.prohibited == {"hold", "decline", "escalate", "needs_check"}


def test_one_family_holds_for_its_check_and_may_not_decline() -> None:
    result = permitted(row(**CARD))
    assert result == evidence.EXAMPLE_ACTIONS
    access = permitted(row(**ACCESS))
    assert access.required_checks == {Check.CONTACT}
    assert {"decline", "escalate", "clear"} <= access.prohibited


def test_two_families_may_decline_without_a_check_and_need_both_checks() -> None:
    result = permitted(row(**CARD, **ACCESS))
    assert result.standard == {"hold"}
    assert result.permitted == {"needs_check", "decline"}
    assert result.required_checks == {Check.CONTACT, Check.ID_CHECK}
    assert "escalate" in result.prohibited


def test_checks_decide_in_the_policy_order() -> None:
    values = row(**CARD, **ACCESS)
    halfway = permitted(values, check(Check.CONTACT, CheckOutcome.PASSED))
    assert halfway.standard == {"hold"} and halfway.required_checks == {Check.ID_CHECK}
    assert "needs_check" in halfway.permitted
    passed = permitted(values, check(Check.CONTACT, CheckOutcome.PASSED),
                       check(Check.ID_CHECK, CheckOutcome.PASSED))
    assert passed.standard == {"clear"} and {"hold", "decline"} <= passed.prohibited
    waiting = permitted(values, check(Check.CONTACT, CheckOutcome.NO_RESPONSE),
                        check(Check.ID_CHECK, CheckOutcome.PASSED))
    assert waiting.standard == {"hold"} and not waiting.required_checks
    assert "needs_check" in waiting.prohibited  # every required check has run (§4.3)
    failed = permitted(values, check(Check.CONTACT, CheckOutcome.PASSED),
                       check(Check.ID_CHECK, CheckOutcome.FAILED))
    assert failed.standard == {"decline"} and failed.clauses["decline"] == "§5.3(b)"
    assert {"clear", "hold", "needs_check", "escalate"} <= failed.prohibited


def test_a_failed_check_with_linkage_escalates_or_declines() -> None:
    failed = permitted(row(**LINKAGE), check(Check.ID_CHECK, CheckOutcome.FAILED))
    assert failed.standard == {"escalate"} and failed.permitted == {"decline"}
    assert failed.prohibited == {"clear", "hold", "needs_check"}


def test_two_families_may_not_decline_once_a_check_has_passed() -> None:
    values = row(**CARD, **ACCESS)
    halfway = permitted(values, check(Check.CONTACT, CheckOutcome.PASSED))
    assert halfway.standard == {"hold"} and "decline" in halfway.prohibited
    unanswered = permitted(values, check(Check.CONTACT, CheckOutcome.NO_RESPONSE))
    assert "decline" in unanswered.permitted


def test_a_failed_check_decides_even_after_its_evidence_lapses() -> None:
    """The 30-day linkage window moved on while the check ran: the failure still decides."""
    lapsed = permitted(row(), check(Check.ID_CHECK, CheckOutcome.FAILED))
    assert lapsed.row == "§6.6(b)" and lapsed.standard == {"decline"}


def test_a_settling_outcome_overrides_passed_checks() -> None:
    values = row(**CARD, unauthorized_disputes_lost_user=1)
    result = permitted(values, check(Check.ID_CHECK, CheckOutcome.PASSED))
    assert result.row == "§6.6(a)" and result.standard == {"decline"}
    assert evidence.permitted_actions(evidence.EXAMPLE_SETTLED_EVIDENCE) == \
        evidence.EXAMPLE_SETTLED_ACTIONS
    linked = permitted(row(**LINKAGE, never_pay_determined_user=1))
    assert linked.standard == {"escalate"} and linked.permitted == {"decline"}
    assert linked.prohibited == {"clear", "hold", "needs_check"}


@pytest.mark.parametrize(("case", "values", "standard", "check_needed"), [
    # the September counterexample classes, before any check and with no settling outcome
    ("never-pay first order, Context only", row(is_first_attempt_user=1,
     amount_over_category_p95=2.0, account_age_days=0.2, email_domain_class=2), "clear", None),
    ("never-pay first order with R03", row(**CARD, is_first_attempt_user=1), "hold",
     Check.ID_CHECK),
    ("merchant bust-out customer with R03", row(**CARD), "hold", Check.ID_CHECK),
    ("two open item-not-received disputes", row(inr_disputes_opened_user=2), "clear", None),
    ("address shared with 3 other accounts", row(accounts_on_address_30d=4, ship_to_home=0),
     "hold", Check.ID_CHECK),
    ("the same address as an old home", row(accounts_on_address_30d=4), "clear", None),
])
def test_dispositions_rest_on_evidence_not_on_pattern(
    case: str, values: dict[str, object], standard: str, check_needed: Check | None
) -> None:
    result = permitted(values)
    assert result.standard == {standard}, case
    assert result.required_checks == ({check_needed} if check_needed else set()), case
    assert "escalate" in result.prohibited, case


def _expected(settled: bool, families: int, linkage: bool, required: set[str],
              outcomes: dict[str, str]) -> tuple[set[str], set[str], set[str]]:
    """The sets as FP-2 §5.3 and §6.6 word them, written independently of core.evidence."""
    if settled or "failed" in outcomes.values():  # §6.6(a), or §5.3(b) at once
        if linkage:
            return {"escalate"}, {"decline"}, {"clear", "hold", "needs_check"}
        return {"decline"}, set(), {"clear", "hold", "needs_check", "escalate"}
    if families == 0:  # §6.6(c)
        return {"clear"}, set(), {"hold", "decline", "escalate", "needs_check"}
    if all(outcomes.get(c) == "passed" for c in required):  # §5.3(a)
        return {"clear"}, set(), {"hold", "decline", "escalate", "needs_check"}
    allowed, ruled_out = set(), {"clear", "escalate"}  # §6.6(b)
    (allowed if set(required) - set(outcomes) else ruled_out).add("needs_check")
    no_pass = "passed" not in outcomes.values()
    (allowed if families >= 2 and no_pass else ruled_out).add("decline")
    return {"hold"}, allowed, ruled_out


FAMILY_ROWS = {"Account access": ACCESS, "Card": CARD,
               "Velocity": {"attempts_user_24h": 9}, "Linkage": LINKAGE}


@pytest.mark.parametrize("settled", [False, True])
def test_every_evidence_state_gets_the_policy_sets(settled: bool) -> None:
    """All family combinations x check outcomes: each disposition in exactly one set."""
    import itertools

    names = sorted(FAMILY_ROWS)
    results = (None, "passed", "failed", "no_response")
    states = 0
    for k in range(len(names) + 1):
        for chosen in itertools.combinations(names, k):
            values = row(unauthorized_disputes_lost_user=int(settled))
            for name in chosen:
                values.update(FAMILY_ROWS[name])
            required = {evidence.REQUIRED_CHECK[Family(name)].value for name in chosen}
            for contact, id_check in itertools.product(results, results):
                outcomes = {c: o for c, o in (("contact", contact), ("id_check", id_check))
                            if o is not None}
                checks = [check(Check(c), CheckOutcome(o)) for c, o in outcomes.items()]
                got = permitted(values, *checks)
                want = _expected(settled, len(chosen), "Linkage" in chosen, required,
                                 outcomes)
                assert (got.standard, got.permitted, got.prohibited) == want, (chosen, outcomes)
                assert got.standard | got.permitted | got.prohibited == {
                    "clear", "hold", "decline", "escalate", "needs_check"}
                states += 1
    assert states == 16 * 16


# ---------------------------------------------------------------- one definition


def test_vectorized_conditions_agree_with_classify() -> None:
    rng = np.random.default_rng(416)
    n = 400
    frame = pd.DataFrame({column: np.full(n, value) for column, value in QUIET.items()})
    for column in QUIET:
        values = np.asarray(QUIET[column], dtype=float)
        noise = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 48.0, 72.0, 90.0, 900.5,
                            2200.0, np.nan], size=n)
        frame[column] = np.where(rng.random(n) < 0.3, noise, values)
    table = evidence.conditions(frame)
    for i in range(n):
        ev = evidence.classify(frame.iloc[i], ())
        held = {c.rule for c in ev.conditions}
        assert held == {r for r in evidence.RULE_FAMILY if table.iloc[i][r]}
        assert {c.clause for c in ev.household_exceptions} == {
            clause for clause in evidence.HOUSEHOLD_EXCEPTIONS if table.iloc[i][clause]}
        assert {c.clause for c in ev.settlements} == {
            clause for clause in evidence.SETTLEMENTS if table.iloc[i][clause]}
        assert {f.value for f in ev.families_present} == {
            f.value for f in evidence.ADVERSE_FAMILIES if table.iloc[i][f.value]}
        assert table.iloc[i]["families"] == len(ev.families_present)


def test_classify_reads_only_reviewer_columns_never_labels_or_truth() -> None:
    """The reviewer and the referee see context columns and check results only."""
    spec = {column.name: column for column in asof.COLUMNS}
    for column in evidence.CLASSIFY_COLUMNS:
        assert column in spec, column
        assert {"reviewer", "referee"} <= set(spec[column].used_by), column

    class Recording(dict):
        def __getitem__(self, key: str) -> object:
            read.add(key)
            return super().__getitem__(key)

    read: set[str] = set()
    leaked = Recording(row(**CARD), label=1, pattern_id="P-STOLEN", intent="fraud",
                       actor="fraudster", basis="third_party_fraud")
    with_truth = evidence.permitted_actions(evidence.classify(leaked, ()))
    assert read <= set(evidence.CLASSIFY_COLUMNS)
    clean = permitted(row(**CARD))
    assert with_truth == clean
    flipped = Recording(row(**CARD), label=0, pattern_id=None, intent="legitimate")
    assert evidence.permitted_actions(evidence.classify(flipped, ())) == clean


def test_missing_values_never_make_a_condition_hold() -> None:
    frame = pd.DataFrame([row(**CARD), row(**CARD)])
    frame["accounts_on_device_30d"] = pd.array([pd.NA, 7], dtype="Int64")
    frame["bin_ip_country_mismatch"] = pd.array([1, pd.NA], dtype="Int64")
    table = evidence.conditions(frame)
    assert table["R02"].tolist() == [False, True]
    assert table["R03"].tolist() == [True, False]
    assert rules(frame.iloc[0].to_dict()) == {"R03"}
    assert rules(frame.iloc[1].to_dict()) == {"R02"}


def test_missing_context_column_is_an_error_not_a_quiet_order() -> None:
    values = row()
    del values["accounts_on_device_30d"]
    with pytest.raises(KeyError, match="accounts_on_device_30d"):
        evidence.classify(values, ())
    with pytest.raises(KeyError, match="accounts_on_device_30d"):
        evidence.conditions(pd.DataFrame([values]))


# ---------------------------------------------------------------- the policy's text


def _policy() -> str:
    return (REPO / "policy" / "fraud-policy.md").read_text()


def _numbers(text: str) -> set[float]:
    """Distinct numbers stated in ``text``, leaving out rule ids and clause references."""
    text = re.sub(r"R\d\d|§[\d.]+(\([a-f]\))?", "", text)
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}


def test_rule_table_matches_the_policy_text() -> None:
    text = _policy()
    section = text[text.index("**6.2**"):text.index("**6.3**")]
    rows = re.findall(r"^\| (R\d\d(?:\([ab]\))?) \| (.+?) \| (.+?) \| .+? \|$", section, re.M)
    assert len(rows) == 12
    assert {rule: family for rule, _, family in rows} == {
        rule: family.value for rule, family in evidence.RULE_FAMILY.items()}
    for rule, condition, _ in rows:
        assert _numbers(condition) == set(evidence.THRESHOLDS[rule].values()), rule


def test_household_numbers_match_the_policy_text() -> None:
    text = _policy()
    section = text[text.index("**6.4**"):text.index("**6.5**")]
    for clause, letter in (("§6.4(a)", "(a)"), ("§6.4(b)", "(b)")):
        part = section[section.index(letter):]
        part = part[:part.index(".", part.index("accounts"))]
        assert _numbers(part) == set(evidence.THRESHOLDS[clause].values()), clause
