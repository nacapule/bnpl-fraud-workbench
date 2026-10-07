"""Evidence classification shared by the simulated reviewer and the LLM referee (contract).

One written standard drives both: the fraud policy (``policy/fraud-policy.md``,
:data:`POLICY`). Classification reads the order's as-of context row at the
evaluation time (core.asof: each column at its anchor, so what the policy measures
before the order stays fixed while linkage, earlier outcomes and shipment status
are as known at the evaluation time) plus the outcomes of at most two
verification checks (core.actions.Check), each run at most once (§5.1). It never reads labels or
latent tables. Latent truth may generate check outcomes, at stated per-pattern
rates keyed to the order's stable id, but never chooses an action; sparse
evidence stays sparse.

The policy is evaluated at review and again when each check completes, each time
on the evidence known then (§6.6). One evaluation has two steps:

* :func:`classify` turns the context row and the completed checks into
  :class:`Evidence`: every §6.2 rule condition that holds (as the rules hold,
  which queue priority also uses, §7.1), the §6.4 household exceptions that stop
  R02 or R08 from counting, the §6.5 benign explanations a memo weighs, the §6.3
  earlier outcomes that settle the order, and the checks run.
* :func:`permitted_actions` applies the §6.6 decision table to it: the row that
  applies; the standard, also-permitted and prohibited dispositions (strings of
  core.actions.MEMO_DISPOSITIONS), each with the clause it cites; and the §5.2
  checks still to run.

Clause ids are strings in the policy's own form (``"§6.2"``, ``"§6.6(b)"``); rule
ids are the §6.2 table's (``"R01"``, ``"R06(b)"``; R12 is retired). A citation
reads ``f"{POLICY} {clause}, {rule}"``, as in "FP-2 §6.6(b), R03".
:data:`EXAMPLE_EVIDENCE` and :data:`EXAMPLE_ACTIONS` are one hand-checked case;
:data:`EXAMPLE_SETTLED_EVIDENCE` and :data:`EXAMPLE_SETTLED_ACTIONS` are the same order
re-evaluated after an earlier outcome that settles it became known while it waited.

The reviewer's procedure (which check to run next, which disposition to take) is
built on these two functions and frozen with the generator parameters; if it
outgrows about 250 lines or needs more than two checks, a stated confusion table
replaces it (never-pay limited to clear or hold) and the switch is recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from core.actions import MEMO_DISPOSITIONS, Check, CheckOutcome

POLICY = "FP-2"


class Family(StrEnum):
    """The §6.2 evidence families."""

    ACCOUNT_ACCESS = "Account access"
    CARD = "Card"
    VELOCITY = "Velocity"
    LINKAGE = "Linkage"
    CONTEXT = "Context"


# The four adverse families can support hold or decline; Context cannot, alone or together.
ADVERSE_FAMILIES = frozenset(set(Family) - {Family.CONTEXT})

# §6.2: each rule is an evidence condition in one family.
RULE_FAMILY: Mapping[str, Family] = {
    "R01": Family.ACCOUNT_ACCESS, "R11": Family.ACCOUNT_ACCESS,
    "R03": Family.CARD, "R07": Family.CARD,
    "R05": Family.VELOCITY,
    "R02": Family.LINKAGE, "R08": Family.LINKAGE, "R06(b)": Family.LINKAGE,
    "R10": Family.LINKAGE,
    "R06(a)": Family.CONTEXT, "R04": Family.CONTEXT, "R09": Family.CONTEXT,
}

# §5.2: the check each adverse family present requires (both when both are required).
REQUIRED_CHECK: Mapping[Family, Check] = {
    Family.ACCOUNT_ACCESS: Check.CONTACT,
    Family.CARD: Check.ID_CHECK,
    Family.VELOCITY: Check.ID_CHECK,
    Family.LINKAGE: Check.ID_CHECK,
}

SETTLEMENTS = ("§6.3(a)", "§6.3(b)", "§6.3(c)")  # earlier outcomes that settle the order
HOUSEHOLD_EXCEPTIONS: Mapping[str, str] = {"§6.4(a)": "R02", "§6.4(b)": "R08"}  # clause: rule
BENIGN = tuple(f"§6.5({item})" for item in "abcdef")  # never support hold, decline, escalate
TABLE_ROWS = ("§6.6(a)", "§6.6(b)", "§6.6(c)")


@dataclass(frozen=True)
class Condition:
    """One §6.2 rule condition that holds at the evaluation time."""

    rule: str  # a key of RULE_FAMILY
    family: Family
    columns: tuple[str, ...]  # the core.asof columns it rests on, cited as facts (§6.1)
    clause: str = "§6.2"

    def __post_init__(self) -> None:
        if RULE_FAMILY.get(self.rule) != self.family:
            raise ValueError(f"{self.rule} is not a §6.2 condition in {self.family}")


@dataclass(frozen=True)
class Citation:
    """A §6.3 settlement, §6.4 household exception or §6.5 benign explanation that applies."""

    clause: str
    columns: tuple[str, ...]  # the core.asof columns it rests on (empty when none)


@dataclass(frozen=True)
class CheckResult:
    """A completed §5.1 verification check."""

    check: Check
    outcome: CheckOutcome
    completed_at: datetime


@dataclass(frozen=True)
class Evidence:
    """What the policy evaluates for one order at one evaluation time."""

    conditions: tuple[Condition, ...] = ()  # every §6.2 condition that holds, excepted or not
    household_exceptions: tuple[Citation, ...] = ()  # §6.4(a) for R02, §6.4(b) for R08
    benign: tuple[Citation, ...] = ()  # §6.5 items a memo weighs
    settlements: tuple[Citation, ...] = ()  # §6.3 items known at the evaluation time
    checks: tuple[CheckResult, ...] = ()  # completed by the evaluation time

    def __post_init__(self) -> None:
        rules = {condition.rule for condition in self.conditions}
        for group, known in ((self.household_exceptions, HOUSEHOLD_EXCEPTIONS),
                             (self.benign, BENIGN), (self.settlements, SETTLEMENTS)):
            unknown = {citation.clause for citation in group} - set(known)
            if unknown:
                raise ValueError(f"not a clause of this kind: {sorted(unknown)}")
        for citation in self.household_exceptions:
            if HOUSEHOLD_EXCEPTIONS[citation.clause] not in rules:
                raise ValueError(f"{citation.clause} excepts a rule that does not hold")
        if len({result.check for result in self.checks}) != len(self.checks):
            raise ValueError("each check runs at most once (§5.1)")

    @property
    def families_present(self) -> frozenset[Family]:
        """Adverse families with a condition that holds without a §6.4 exception (§6.6)."""
        excepted = {HOUSEHOLD_EXCEPTIONS[c.clause] for c in self.household_exceptions}
        return frozenset(c.family for c in self.conditions
                         if c.family in ADVERSE_FAMILIES and c.rule not in excepted)


@dataclass(frozen=True)
class PermittedActions:
    """The §6.6 decision table applied to one :class:`Evidence`.

    ``standard`` is the review procedure's disposition; a memo's disposition must be
    in ``standard | permitted`` (§10). ``prohibited`` is what the policy rules out
    explicitly. The three sets are disjoint, and ``clauses`` maps each disposition in
    them to the clause it cites.
    """

    row: str  # one of TABLE_ROWS
    standard: frozenset[str]
    permitted: frozenset[str]
    prohibited: frozenset[str]
    clauses: Mapping[str, str]
    required_checks: frozenset[Check]  # §5.2 checks not yet run

    def __post_init__(self) -> None:
        sets = (self.standard, self.permitted, self.prohibited)
        named = set().union(*sets)
        if self.row not in TABLE_ROWS:
            raise ValueError(f"not a §6.6 row: {self.row}")
        if not named <= set(MEMO_DISPOSITIONS):
            raise ValueError(f"not memo dispositions: {sorted(named - set(MEMO_DISPOSITIONS))}")
        if sum(len(s) for s in sets) != len(named):
            raise ValueError("standard, permitted and prohibited overlap")
        if set(self.clauses) != named:
            raise ValueError("every disposition named needs exactly one clause")


# --------------------------------------------------------------------------- conditions
# Each condition of the policy is one predicate over the order's context row. A predicate
# uses only comparisons and the & and | operators (never ~, which negates a Python bool
# arithmetically), so the same definition evaluates one
# row (a mapping of scalars, in classify) and a whole frame (columns as Series, in
# conditions): one definition of each clause serves the reviewer, the referee and the
# queue's priority.

# The numbers each clause states, as the policy states them (FP-2 §6.2-§6.4); windows that
# live in a column's own definition are listed too, so a test can compare them with the
# policy text.
THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    "R01": {"credential_change_hours": 48, "account_age_days": 90, "device_first_use_hours": 72},
    "R11": {"previous_attempt_hours": 12, "kmh": 900},
    "R03": {},
    "R07": {"declines": 3, "window_hours": 24},
    "R05": {"attempts_user_over": 3, "attempts_device_over": 5, "window_hours": 24},
    "R02": {"accounts": 3, "window_days": 30},
    "R08": {"accounts": 3, "window_days": 30},
    "R06(b)": {},
    "R10": {"accounts": 3},
    "R06(a)": {},
    "R04": {"percentile": 95, "account_age_days": 7},
    "R09": {"inr_disputes": 2},
    "§6.4(a)": {"device_age_days": 90, "max_accounts": 4},
    "§6.4(b)": {"home_age_days": 90, "max_accounts": 4},
    "§6.3(c)": {"inr_claims_rejected": 2},
}
_T = THRESHOLDS


@dataclass(frozen=True)
class Predicate:
    """One clause's condition: its id, the context columns it reads, and the test."""

    clause: str
    columns: tuple[str, ...]
    test: Callable[[Mapping[str, Any]], Any]


def _rule(rule: str, columns: tuple[str, ...], test: Callable[[Mapping[str, Any]], Any]
          ) -> Predicate:
    return Predicate(rule, columns, test)


# §6.2, in the table's order.
RULES: tuple[Predicate, ...] = (
    # a password change or reset or an email change in the 48 h (the three columns whose
    # minimum is hours_since_credential_change, which the reviewer and referee do not read)
    _rule("R01", ("hours_since_password_change", "hours_since_password_reset",
                  "hours_since_email_change", "account_age_days", "device_link_age_hours"),
          lambda c: ((c["hours_since_password_change"] <= _T["R01"]["credential_change_hours"])
                     | (c["hours_since_password_reset"] <= _T["R01"]["credential_change_hours"])
                     | (c["hours_since_email_change"] <= _T["R01"]["credential_change_hours"]))
          & (c["account_age_days"] >= _T["R01"]["account_age_days"])
          & (c["device_link_age_hours"] <= _T["R01"]["device_first_use_hours"])),
    _rule("R11", ("geo_kmh_from_previous_attempt",),
          lambda c: c["geo_kmh_from_previous_attempt"] > _T["R11"]["kmh"]),
    _rule("R03", ("bin_ip_country_mismatch", "avs_mismatch", "cvv_mismatch"),
          lambda c: (c["bin_ip_country_mismatch"] == 1)
          & ((c["avs_mismatch"] == 1) | (c["cvv_mismatch"] == 1))),
    _rule("R07", ("processor_declines_card_24h", "processor_declines_device_24h"),
          lambda c: (c["processor_declines_card_24h"] >= _T["R07"]["declines"])
          | (c["processor_declines_device_24h"] >= _T["R07"]["declines"])),
    _rule("R05", ("attempts_user_24h", "attempts_device_24h"),
          lambda c: (c["attempts_user_24h"] > _T["R05"]["attempts_user_over"])
          | (c["attempts_device_24h"] > _T["R05"]["attempts_device_over"])),
    _rule("R02", ("accounts_on_device_30d",),
          lambda c: c["accounts_on_device_30d"] >= _T["R02"]["accounts"]),
    _rule("R08", ("accounts_on_address_30d",),
          lambda c: c["accounts_on_address_30d"] >= _T["R08"]["accounts"]),
    _rule("R06(b)", ("email_root_other_accounts",),
          lambda c: c["email_root_other_accounts"] >= 1),
    _rule("R10", ("promo_uses_linked_accounts",),
          lambda c: c["promo_uses_linked_accounts"] >= _T["R10"]["accounts"]),
    _rule("R06(a)", ("email_domain_class",), lambda c: c["email_domain_class"] == 2),
    _rule("R04", ("is_first_attempt_user", "amount_over_category_p95", "account_age_days"),
          lambda c: (c["is_first_attempt_user"] == 1) & (c["amount_over_category_p95"] > 1)
          & (c["account_age_days"] < _T["R04"]["account_age_days"])),
    _rule("R09", ("inr_disputes_opened_user",),
          lambda c: c["inr_disputes_opened_user"] >= _T["R09"]["inr_disputes"]),
)

# §6.4: households. Each excepts its rule from the family count when the rule holds.
EXCEPTIONS: tuple[Predicate, ...] = (
    Predicate("§6.4(a)", ("device_link_age_hours", "accounts_on_device_30d"),
              lambda c: (c["device_link_age_hours"] >= 24 * _T["§6.4(a)"]["device_age_days"])
              & (c["accounts_on_device_30d"] <= _T["§6.4(a)"]["max_accounts"])),
    Predicate("§6.4(b)", ("ship_to_home", "home_address_age_days", "accounts_on_address_30d"),
              lambda c: (c["ship_to_home"] == 1)
              & (c["home_address_age_days"] >= _T["§6.4(b)"]["home_age_days"])
              & (c["accounts_on_address_30d"] <= _T["§6.4(b)"]["max_accounts"])),
)

# §6.3: earlier outcomes known at the evaluation time that settle the order.
SETTLING: tuple[Predicate, ...] = (
    Predicate("§6.3(a)", ("unauthorized_disputes_lost_user", "victim_reports_user"),
              lambda c: (c["unauthorized_disputes_lost_user"] >= 1)
              & (c["victim_reports_user"] == 0)),
    Predicate("§6.3(b)", ("never_pay_determined_user",),
              lambda c: c["never_pay_determined_user"] == 1),
    Predicate("§6.3(c)", ("inr_claims_rejected_user",),
              lambda c: c["inr_claims_rejected_user"] >= _T["§6.3(c)"]["inr_claims_rejected"]),
)

# §6.5: benign explanations a memo weighs. They never support hold, decline or escalate.
# (d) scores and resemblance and (e) merchant evidence are prohibitions on what may be
# cited, with no fact in the order's row to cite, so classify never lists them.
BENIGN_EXPLANATIONS: tuple[Predicate, ...] = (
    Predicate("§6.5(a)", ("avs_mismatch", "cvv_mismatch", "bin_ip_country_mismatch"),
              lambda c: ((c["avs_mismatch"] == 1) | (c["cvv_mismatch"] == 1))
              & (c["bin_ip_country_mismatch"] == 0)),
    # new device or shipping address (inside R01's 72 h), travel, a password reset after a
    # new phone (the reset inside R01's 48 h, the phone change at most 72 h before it), or
    # shipping away from home
    Predicate("§6.5(b)", ("device_link_age_hours", "ship_address_link_age_hours",
                          "ip_country_not_home", "ship_to_home", "hours_since_password_reset",
                          "hours_since_phone_change"),
              lambda c: (c["device_link_age_hours"] <= 72)
              | (c["ship_address_link_age_hours"] <= 72) | (c["ip_country_not_home"] == 1)
              | (c["ship_to_home"] == 0)
              | ((c["hours_since_password_reset"] <= 48)
                 & (c["hours_since_phone_change"] >= c["hours_since_password_reset"])
                 & (c["hours_since_phone_change"] <= c["hours_since_password_reset"] + 72))),
    # a first or large order that R04 does not cover
    Predicate("§6.5(c)", ("is_first_attempt_user", "amount_over_category_p95",
                          "account_age_days"),
              lambda c: ((c["is_first_attempt_user"] == 1) | (c["amount_over_category_p95"] > 1))
              & ((c["is_first_attempt_user"] == 0) | (c["amount_over_category_p95"] <= 1)
                 | (c["account_age_days"] >= _T["R04"]["account_age_days"]))),
    # repayment history on other plans, good or bad (§6.3(b) is the only exception)
    Predicate("§6.5(f)", ("installments_due_user", "never_pay_determined_user"),
              lambda c: (c["installments_due_user"] > 0) & (c["never_pay_determined_user"] == 0)),
)

CLASSIFY_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys(
    column for group in (RULES, EXCEPTIONS, SETTLING, BENIGN_EXPLANATIONS)
    for predicate in group for column in predicate.columns))


def _holds(predicate: Predicate, values: Mapping[str, Any]) -> bool:
    """Whether the condition holds; a missing value (NaN or NA) never makes it hold."""
    try:
        result = predicate.test(values)
    except KeyError as missing:
        raise KeyError(f"{predicate.clause} needs context column {missing}") from None
    return False if result is pd.NA else bool(result)


def conditions(context: pd.DataFrame) -> pd.DataFrame:
    """Every clause's condition for each row of ``context``, as booleans.

    Columns: the §6.2 rules (``"R01"`` ... ``"R09"``, as they hold, which queue priority
    uses), ``"§6.4(a)"``/``"§6.4(b)"`` (the household exception applies: its rule holds
    and the exception's test passes), the §6.3 settlements, one column per adverse
    family (present: a condition holds without an exception) and ``"families"`` (their
    number, the evidence strength). Same definitions as :func:`classify`.
    """
    missing = [column for column in CLASSIFY_COLUMNS if column not in context.columns]
    if missing:
        raise KeyError(f"context lacks columns {missing}")
    # plain float arrays: a missing value is NaN, and NaN compares false
    values = {column: pd.to_numeric(context[column], errors="raise").to_numpy(
        dtype=float, na_value=np.nan) for column in CLASSIFY_COLUMNS}
    out: dict[str, np.ndarray] = {}
    for predicate in RULES + SETTLING:
        out[predicate.clause] = np.asarray(predicate.test(values), dtype=bool)
    for predicate in EXCEPTIONS:
        rule = HOUSEHOLD_EXCEPTIONS[predicate.clause]
        out[predicate.clause] = out[rule] & np.asarray(predicate.test(values), dtype=bool)
    excepted = {rule: out[clause] for clause, rule in HOUSEHOLD_EXCEPTIONS.items()}
    families = sorted(ADVERSE_FAMILIES)
    for family in families:
        present = np.zeros(len(context), dtype=bool)
        for rule, rule_family in RULE_FAMILY.items():
            if rule_family == family:
                present |= out[rule] & ~excepted[rule] if rule in excepted else out[rule]
        out[family.value] = present
    out["families"] = np.sum([out[f.value] for f in families], axis=0).astype(int) \
        if len(context) else np.zeros(0, dtype=int)
    return pd.DataFrame(out, index=context.index)


def classify(context_row: Mapping[str, Any], checks: Sequence[CheckResult]) -> Evidence:
    """The evidence in an order's context row at an evaluation time and the checks so far.

    ``context_row`` is the order's as-of context row (core.asof KEY_COLUMNS plus
    COLUMN_NAMES) with ``decision_at`` the evaluation time (the review, or a check's
    completion): order-anchored columns as at the order, decision-anchored ones
    (linkage, the current email, outcome-derived columns under the policy being
    replayed) as known by then, so accounts that join the device while the order
    waits count for R02 and an earlier outcome that settles the order (§6.3) counts
    as soon as it is known. It is a pandas Series or
    a mapping. ``checks`` are the results completed by then, at most one per check.
    Nothing else is read: no labels, latent tables or world.
    """
    try:
        values = {column: context_row[column] for column in CLASSIFY_COLUMNS}
    except KeyError as missing:
        raise KeyError(f"context row lacks column {missing}") from None
    # a missing value (None, NaN or NA) compares false, so it never makes a condition hold
    values = {column: np.nan if value is None or value is pd.NA else value
              for column, value in values.items()}
    held = [p for p in RULES if _holds(p, values)]
    rules = {p.clause for p in held}
    return Evidence(
        conditions=tuple(Condition(p.clause, RULE_FAMILY[p.clause], p.columns) for p in held),
        household_exceptions=tuple(
            Citation(p.clause, p.columns) for p in EXCEPTIONS
            if HOUSEHOLD_EXCEPTIONS[p.clause] in rules and _holds(p, values)),
        benign=tuple(Citation(p.clause, p.columns) for p in BENIGN_EXPLANATIONS
                     if _holds(p, values)),
        settlements=tuple(Citation(p.clause, p.columns) for p in SETTLING if _holds(p, values)),
        checks=tuple(checks),
    )


def _result(row: str, standard: Mapping[str, str], permitted: Mapping[str, str],
            prohibited: Mapping[str, str], required: frozenset[Check]) -> PermittedActions:
    return PermittedActions(row, frozenset(standard), frozenset(permitted),
                            frozenset(prohibited), {**standard, **permitted, **prohibited},
                            required)


def permitted_actions(evidence: Evidence) -> PermittedActions:
    """Apply the §6.6 decision table and the §5.3 check outcomes to ``evidence``.

    Precedence, as §6.6 words it: (a) a §6.3 settlement applies, whatever the checks
    returned; otherwise a failed check gives the sets of §5.3(b) at once, even if no
    adverse family is present any longer (a 30-day linkage window moved on while the
    check ran); otherwise (c) when no adverse family is present; otherwise, once every
    required check has passed, §5.3(a); otherwise row (b), where the hold continues.
    Outcomes of §5.3 are reported under row (b), citing §5.3. ``needs_check`` is
    permitted only under row (b) while a required check has not run (``no_response``
    counts as run, §4.3); ``decline`` without a failed check only with two or more
    families and before any check has passed; ``escalate`` only with Linkage present at
    this evaluation. The cancellation after 48 hours without a response (§5.3(c)) is the
    replay's timing, not a disposition, so the result keeps the hold until then.
    Every disposition lands in exactly one set.
    """
    families = evidence.families_present
    linkage = Family.LINKAGE in families
    outcomes = {result.check: result.outcome for result in evidence.checks}
    settled = bool(evidence.settlements)
    if settled or CheckOutcome.FAILED in outcomes.values():
        clause = "§6.6(a)" if settled else "§5.3(b)"
        if linkage:
            standard, permitted, extra = {"escalate": clause}, {"decline": clause}, {}
        else:
            standard, permitted, extra = {"decline": clause}, {}, {"escalate": clause}
        prohibited = {"clear": clause, "hold": clause, "needs_check": clause, **extra}
        return _result("§6.6(a)" if settled else "§6.6(b)", standard, permitted, prohibited,
                       frozenset())
    if not families:
        return _result("§6.6(c)", {"clear": "§6.6(c)"}, {},
                       {d: "§6.6(c)" for d in ("hold", "decline", "escalate", "needs_check")},
                       frozenset())
    required = frozenset(REQUIRED_CHECK[family] for family in families)
    if all(outcomes.get(check) == CheckOutcome.PASSED for check in required):
        return _result("§6.6(b)", {"clear": "§5.3(a)"}, {},
                       {d: "§5.3(a)" for d in ("hold", "decline", "escalate", "needs_check")},
                       frozenset())
    not_run = frozenset(check for check in required if check not in outcomes)
    any_passed = CheckOutcome.PASSED in outcomes.values()
    permitted: dict[str, str] = {}
    prohibited = {"clear": "§6.6(b)", "escalate": "§6.6(b)"}
    (permitted if not_run else prohibited)["needs_check"] = "§6.6(b)"
    (permitted if len(families) >= 2 and not any_passed else prohibited)["decline"] = "§6.6(b)"
    return _result("§6.6(b)", {"hold": "§6.6(b)"}, permitted, prohibited, not_run)


# One adverse family (Card: the card's issuing and IP countries differ and AVS failed)
# and no check run yet. Row (b): hold for §5.2's id_check; needs_check is permitted while
# it has not run; clear, escalate and (with one family) decline are prohibited.
EXAMPLE_EVIDENCE = Evidence(
    conditions=(Condition("R03", Family.CARD, ("bin_ip_country_mismatch", "avs_mismatch")),),
)
EXAMPLE_ACTIONS = PermittedActions(
    row="§6.6(b)",
    standard=frozenset({"hold"}),
    permitted=frozenset({"needs_check"}),
    prohibited=frozenset({"clear", "decline", "escalate"}),
    clauses={"hold": "§6.6(b)", "needs_check": "§6.6(b)", "clear": "§6.6(b)",
             "decline": "§6.6(b)", "escalate": "§6.6(b)"},
    required_checks=frozenset({Check.ID_CHECK}),
)

# The same order at its review: meanwhile an unauthorized dispute on an earlier order of
# the account was resolved lost, and the holder reported no takeover (§6.3(a)). Row (a)
# overrides the checks: decline; clear, hold, needs_check and (without Linkage) escalate
# are prohibited.
EXAMPLE_SETTLED_EVIDENCE = Evidence(
    conditions=EXAMPLE_EVIDENCE.conditions,
    settlements=(Citation("§6.3(a)", ("unauthorized_disputes_lost_user",
                                      "victim_reports_user")),),
)
EXAMPLE_SETTLED_ACTIONS = PermittedActions(
    row="§6.6(a)",
    standard=frozenset({"decline"}),
    permitted=frozenset(),
    prohibited=frozenset({"clear", "hold", "needs_check", "escalate"}),
    clauses={"decline": "§6.6(a)", "clear": "§6.6(a)", "hold": "§6.6(a)",
             "needs_check": "§6.6(a)", "escalate": "§6.6(a)"},
    required_checks=frozenset(),
)
