"""Evidence classification shared by the simulated reviewer and the LLM referee (contract).

One written standard drives both: the fraud policy (``policy/fraud-policy.md``,
:data:`POLICY`). Classification reads the order's as-of context row at the
evaluation time (core.asof: rule columns anchored at the order, earlier outcomes
and shipment status as known at the evaluation time) plus the outcomes of at
most two verification checks
(core.actions.Check), each run at most once (§5.1). It never reads labels or
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

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


def classify(context_row: Mapping[str, Any], checks: Sequence[CheckResult]) -> Evidence:
    """The evidence in an order's context row at an evaluation time and the checks so far.

    ``context_row`` is the order's as-of context row (core.asof KEY_COLUMNS plus
    COLUMN_NAMES) with ``decision_at`` the evaluation time (the review, or a check's
    completion): attempt-derived columns anchored at the order, outcome-derived ones
    under the policy being replayed and known by then, so an earlier outcome that
    settles the order (§6.3) counts as soon as it is known. It is a pandas Series or
    a mapping. ``checks`` are the results completed by then, at most one per check.
    Nothing else is read: no labels, latent tables or world.
    """
    raise NotImplementedError("evidence classification is implemented with the reviewer")


def permitted_actions(evidence: Evidence) -> PermittedActions:
    """Apply the §6.6 decision table to ``evidence``, using the first applicable row.

    (a) a §6.3 settlement applies, whatever the checks returned; (b) an adverse
    family is present; (c) otherwise. Under (b) a failed check decides at once
    (§5.3(b)), every required check passed clears (§5.3(a)), and until then the hold
    continues; ``needs_check`` is correct only under (b) while no check has failed
    and a required check has not run (§4.3); ``escalate`` requires Linkage (§6.6).
    The cancellation after 48 hours without a response (§5.3(c)) is the replay's
    timing, not a disposition, so the result keeps the hold until then.
    """
    raise NotImplementedError("the decision table is implemented with the reviewer")


# One adverse family (Card: the card's issuing and IP countries differ and AVS failed)
# and no check run yet. Row (b): hold for §5.2's id_check; needs_check is correct while
# it has not run; clear before it passes, decline with one family and escalate before a
# check fails are prohibited.
EXAMPLE_EVIDENCE = Evidence(
    conditions=(Condition("R03", Family.CARD, ("bin_ip_country_mismatch", "avs_mismatch")),),
)
EXAMPLE_ACTIONS = PermittedActions(
    row="§6.6(b)",
    standard=frozenset({"hold"}),
    permitted=frozenset({"needs_check"}),
    prohibited=frozenset({"clear", "decline", "escalate"}),
    clauses={"hold": "§6.6(b)", "needs_check": "§4.3", "clear": "§6.6(b)",
             "decline": "§6.6(b)", "escalate": "§6.6(b)"},
    required_checks=frozenset({Check.ID_CHECK}),
)

# The same order at its review: meanwhile an unauthorized dispute on an earlier order of
# the account was resolved lost, and the holder reported no takeover (§6.3(a)). Row (a)
# overrides the checks: decline (escalate would need Linkage); clear, hold and
# needs_check are prohibited.
EXAMPLE_SETTLED_EVIDENCE = Evidence(
    conditions=EXAMPLE_EVIDENCE.conditions,
    settlements=(Citation("§6.3(a)", ("unauthorized_disputes_lost_user",
                                      "victim_reports_user")),),
)
EXAMPLE_SETTLED_ACTIONS = PermittedActions(
    row="§6.6(a)",
    standard=frozenset({"decline"}),
    permitted=frozenset(),
    prohibited=frozenset({"clear", "hold", "needs_check"}),
    clauses={"decline": "§6.6(a)", "clear": "§6.6(a)", "hold": "§6.6(a)",
             "needs_check": "§6.6(a)"},
    required_checks=frozenset(),
)
