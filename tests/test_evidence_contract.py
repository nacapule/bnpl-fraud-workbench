"""The evidence contract: the fraud policy's vocabulary, the decision-table result, an example."""

from __future__ import annotations

from datetime import datetime

import pytest

from core import asof, evidence
from core.actions import MEMO_DISPOSITIONS, Check, CheckOutcome
from core.evidence import CheckResult, Citation, Condition, Evidence, Family, PermittedActions


def test_families_rules_and_checks_are_the_policys() -> None:
    assert [f.value for f in Family] == ["Account access", "Card", "Velocity", "Linkage",
                                         "Context"]
    assert {f.value for f in evidence.ADVERSE_FAMILIES} == {"Account access", "Card",
                                                             "Velocity", "Linkage"}
    assert {rule: f.value for rule, f in evidence.RULE_FAMILY.items()} == {
        "R01": "Account access", "R11": "Account access", "R03": "Card", "R07": "Card",
        "R05": "Velocity", "R02": "Linkage", "R08": "Linkage", "R06(b)": "Linkage",
        "R10": "Linkage", "R06(a)": "Context", "R04": "Context", "R09": "Context",
    }
    assert {f.value: c.value for f, c in evidence.REQUIRED_CHECK.items()} == {
        "Account access": "contact", "Card": "id_check", "Velocity": "id_check",
        "Linkage": "id_check",
    }
    assert evidence.SETTLEMENTS == ("§6.3(a)", "§6.3(b)", "§6.3(c)")
    assert dict(evidence.HOUSEHOLD_EXCEPTIONS) == {"§6.4(a)": "R02", "§6.4(b)": "R08"}
    assert evidence.BENIGN == ("§6.5(a)", "§6.5(b)", "§6.5(c)", "§6.5(d)", "§6.5(e)",
                               "§6.5(f)")
    assert evidence.TABLE_ROWS == ("§6.6(a)", "§6.6(b)", "§6.6(c)")


def test_example_result_uses_memo_dispositions_in_disjoint_sets() -> None:
    result = evidence.EXAMPLE_ACTIONS
    sets = (result.standard, result.permitted, result.prohibited)
    named = set().union(*sets)
    assert named <= set(MEMO_DISPOSITIONS)
    assert sum(len(s) for s in sets) == len(named)
    assert set(result.clauses) == named
    assert result.row in evidence.TABLE_ROWS
    assert all(clause.startswith("§") for clause in result.clauses.values())


def test_example_is_one_card_condition_before_its_check() -> None:
    """One adverse family and no check: hold for the check §5.2 requires; decline needs two."""
    ev, result = evidence.EXAMPLE_EVIDENCE, evidence.EXAMPLE_ACTIONS
    assert ev.families_present == {Family.CARD}
    assert not ev.checks and not ev.settlements
    assert result.row == "§6.6(b)"
    assert result.standard == {"hold"}
    assert {"clear", "decline", "escalate"} <= result.prohibited
    assert result.required_checks == {evidence.REQUIRED_CHECK[f] for f in ev.families_present}


def test_example_conditions_rest_on_asof_columns() -> None:
    for ev in (evidence.EXAMPLE_EVIDENCE, evidence.EXAMPLE_SETTLED_EVIDENCE):
        for item in (*ev.conditions, *ev.household_exceptions, *ev.benign, *ev.settlements):
            assert item.columns
            assert set(item.columns) <= set(asof.COLUMN_NAMES)


def test_a_settlement_known_at_review_decides_the_waiting_order() -> None:
    """R03 rests on the attempt (order-anchored); the settlement is decision-anchored, so
    a re-evaluation at review sees it, and row (a) overrides the pending check."""
    ev, result = evidence.EXAMPLE_SETTLED_EVIDENCE, evidence.EXAMPLE_SETTLED_ACTIONS
    anchor = {column.name: column.anchor for column in asof.COLUMNS}
    assert ev.conditions == evidence.EXAMPLE_EVIDENCE.conditions
    assert {anchor[c] for c in ev.conditions[0].columns} == {asof.ORDER}
    assert {anchor[c] for c in ev.settlements[0].columns} == {asof.DECISION}
    assert result.row == "§6.6(a)" and result.standard == {"decline"}
    assert {"clear", "hold", "needs_check"} <= result.prohibited
    assert not result.required_checks


def test_household_exception_stops_the_rule_counting_but_not_holding() -> None:
    shared_device = Condition("R02", Family.LINKAGE, ("accounts_on_device_30d",))
    household = Citation("§6.4(a)", ("device_link_age_hours", "accounts_on_device_30d"))
    ev = Evidence(conditions=(shared_device,), household_exceptions=(household,))
    assert ev.families_present == frozenset()
    assert ev.conditions == (shared_device,)  # queue priority still sees R02 (§7.1)


def test_contract_rejects_what_the_policy_rules_out() -> None:
    at = datetime(2025, 3, 1, 12)
    with pytest.raises(ValueError):
        Condition("R12", Family.CONTEXT, ("email_domain_class",))  # retired
    with pytest.raises(ValueError):
        Condition("R03", Family.LINKAGE, ("bin_ip_country_mismatch",))
    with pytest.raises(ValueError):
        Evidence(household_exceptions=(Citation("§6.4(b)", ("ship_to_home",)),))
    with pytest.raises(ValueError):
        Evidence(checks=(CheckResult(Check.CONTACT, CheckOutcome.NO_RESPONSE, at),
                         CheckResult(Check.CONTACT, CheckOutcome.PASSED, at)))
    with pytest.raises(ValueError):
        PermittedActions("§6.6(c)", frozenset({"approve"}), frozenset(), frozenset(),
                         {"approve": "§6.6(c)"}, frozenset())
    with pytest.raises(ValueError):
        PermittedActions("§6.6(c)", frozenset({"clear"}), frozenset(), frozenset({"clear"}),
                         {"clear": "§6.6(c)"}, frozenset())


def test_classification_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        evidence.classify({}, ())
    with pytest.raises(NotImplementedError):
        evidence.permitted_actions(evidence.EXAMPLE_EVIDENCE)
