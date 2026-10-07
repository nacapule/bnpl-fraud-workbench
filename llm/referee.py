"""The memo referee: what the fraud policy permits, requires and prohibits for one packet.

The referee reads only the packet and the policy (FP-2). It classifies the packet's
facts with the same evidence module the simulated reviewer uses
(:func:`core.evidence.classify` and :func:`core.evidence.permitted_actions`), so one
written standard drives both, and states every constraint with the clause it rests
on. It never reads labels, latent truth or a model's answer, and it is fixed before
any memo it scores exists.

A memo is scored on:

* its disposition: acceptable when the policy's standard or also-permitted set holds
  it (§10, §6.6, §5.3); wrong when it is prohibited. ``needs_check`` is correct only
  where §6.6(b) permits it, so it is wrong wherever the evidence already decides;
* its next check: a check the policy still requires when one remains, ``none`` when
  the evidence decides (§10, §4.3, §5.2);
* its citations: an adverse disposition or ``needs_check`` cites the clause that
  permits it (§6.1), every cited rule holds, and every cited clause exists;
* non-payment: with no installment due before the decision, it may not rest on
  non-payment or never-pay (§8.1).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from core.evidence import POLICY, classify, permitted_actions
from llm.packet import checks_of, context_row

REPO = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO / "policy" / "fraud-policy.md"
ADVERSE = ("hold", "decline", "escalate")
CHECKS = ("contact", "id_check")
NONPAYMENT_CLAUSES = frozenset({"§6.3(b)", "§8.2", "§8.3", "§9(c)"})  # never-pay grounds
CITATION = re.compile(r"§\d+(?:\.\d+)?(?:\([a-z]\))?|\bR\d{2}(?:\([ab]\))?")


@cache
def policy_text() -> str:
    return POLICY_PATH.read_text()


@cache
def policy_ids() -> frozenset[str]:
    """Every clause and rule id the policy defines: sections, numbered clauses, their
    lettered items, the lettered rows of the §6.6 and §9 tables, and the §6.2 rules."""
    ids: set[str] = set()
    section = None
    clause = None
    for line in policy_text().splitlines():
        heading = re.match(r"## (\d+)\.", line)
        if heading:
            section, clause = heading.group(1), None
            ids.add(f"§{section}")
            continue
        numbered = re.match(r"\*\*(\d+\.\d+)\*\*|\*\*(\d+\.\d+) ", line)
        if numbered:
            clause = numbered.group(1) or numbered.group(2)
            ids.add(f"§{clause}")
        row = re.match(r"\| \(([a-z])\) \|", line)
        if row and section in ("6", "9"):
            ids.add(f"§{'6.6' if section == '6' else '9'}({row.group(1)})")
            continue
        rule = re.match(r"\| (R\d{2})(\([ab]\))? \|", line)
        if rule:
            ids.update({rule.group(1), "".join(part for part in rule.groups() if part)})
            continue
        if clause and section and clause.startswith(f"{section}."):
            # lettered items, not references such as "§6.6(b)" inside the text
            ids.update(f"§{clause}({letter})"
                       for letter in re.findall(r"(?<![\d)])\(([a-f])\)", line))
    return frozenset(ids)


def cited_ids(citations: Sequence[str]) -> list[str]:
    """Clause and rule ids in a memo's citations, in order, as the policy writes them."""
    return [match for citation in citations for match in CITATION.findall(str(citation))]


@dataclass(frozen=True)
class Constraint:
    kind: str  # standard, permitted, prohibited, required_check, no_nonpayment_grounds
    subject: str  # a disposition, a check, or "citations"
    clause: str

    def __str__(self) -> str:
        return f"{self.kind} {self.subject} ({POLICY} {self.clause})"


@dataclass(frozen=True)
class RefereeView:
    """The policy's answer for one packet."""

    row: str
    standard: frozenset[str]
    permitted: frozenset[str]
    prohibited: frozenset[str]
    clauses: Mapping[str, str]
    required_checks: frozenset[str]
    rules: tuple[str, ...]
    families: tuple[str, ...]
    settlements: tuple[str, ...]
    household_exceptions: tuple[str, ...]
    benign: tuple[str, ...]
    no_installment_due: bool
    constraints: tuple[Constraint, ...] = field(default=())

    @property
    def decisive(self) -> bool:
        """The evidence decides the order: no check would change the disposition."""
        return "needs_check" in self.prohibited

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row, "standard": sorted(self.standard),
            "permitted": sorted(self.permitted), "prohibited": sorted(self.prohibited),
            "clauses": dict(sorted(self.clauses.items())),
            "required_checks": sorted(self.required_checks), "rules": list(self.rules),
            "families": list(self.families), "settlements": list(self.settlements),
            "household_exceptions": list(self.household_exceptions),
            "benign": list(self.benign), "no_installment_due": self.no_installment_due,
            "decisive": self.decisive, "constraints": [str(c) for c in self.constraints],
        }


def view(packet: Mapping[str, Any]) -> RefereeView:
    """What FP-2 permits, requires and prohibits for this packet, each with its clause."""
    evidence = classify(context_row(packet), checks_of(packet))
    actions = permitted_actions(evidence)
    required = frozenset(str(check) for check in actions.required_checks)
    installments_due = packet["context"].get("installments_due_user")
    no_installment_due = installments_due is not None and installments_due == 0
    constraints = [Constraint(kind, disposition, actions.clauses[disposition])
                   for kind, group in (("standard", actions.standard),
                                       ("permitted", actions.permitted),
                                       ("prohibited", actions.prohibited))
                   for disposition in sorted(group)]
    constraints += [Constraint("required_check", check, "§5.2") for check in sorted(required)]
    if no_installment_due:
        constraints.append(Constraint("no_nonpayment_grounds", "citations", "§8.1"))
    return RefereeView(
        row=actions.row, standard=actions.standard, permitted=actions.permitted,
        prohibited=actions.prohibited, clauses=dict(actions.clauses),
        required_checks=required,
        rules=tuple(condition.rule for condition in evidence.conditions),
        families=tuple(sorted(str(family) for family in evidence.families_present)),
        settlements=tuple(c.clause for c in evidence.settlements),
        household_exceptions=tuple(c.clause for c in evidence.household_exceptions),
        benign=tuple(c.clause for c in evidence.benign),
        no_installment_due=no_installment_due,
        constraints=tuple(constraints),
    )


@dataclass(frozen=True)
class Score:
    disposition: str
    acceptable: bool  # in the standard or also-permitted set
    standard: bool  # the review procedure's own disposition
    prohibited: bool
    needs_check: bool
    next_check_ok: bool
    citation_ok: bool
    nonpayment_violation: bool
    problems: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "problems": list(self.problems)}


def score(memo: Mapping[str, Any], referee: RefereeView) -> Score:
    """Score a validated memo (llm.memo) against the referee's view of its packet."""
    disposition = memo["disposition"]
    problems: list[str] = []
    acceptable = disposition in referee.standard | referee.permitted
    if not acceptable:
        clause = referee.clauses.get(disposition, "§10")
        problems.append(f"disposition {disposition} is not permitted ({POLICY} {clause})")

    next_check = memo["next_check"]
    if referee.decisive or not referee.required_checks:
        next_check_ok = next_check == "none"
    else:
        next_check_ok = next_check in referee.required_checks
    if not next_check_ok:
        expected = sorted(referee.required_checks) or ["none"]
        problems.append(f"next check {next_check} (policy: {' or '.join(expected)})")

    ids = cited_ids(memo["citations"])
    unknown = [item for item in ids if item not in policy_ids()]
    rules_not_holding = [item for item in ids if re.fullmatch(r"R\d{2}(\([ab]\))?", item)
                         and not any(rule == item or rule.startswith(f"{item}(")
                                     for rule in referee.rules)]
    citation_ok = not unknown and not rules_not_holding
    if disposition in (*ADVERSE, "needs_check") and acceptable:
        needed = referee.clauses[disposition]
        if needed not in ids:
            citation_ok = False
            problems.append(f"{disposition} does not cite {needed} (§6.1)")
    if unknown:
        problems.append(f"cites ids the policy does not define: {unknown}")
    if rules_not_holding:
        problems.append(f"cites rules that do not hold: {rules_not_holding}")

    nonpayment = referee.no_installment_due and bool(NONPAYMENT_CLAUSES & set(ids))
    if nonpayment:
        problems.append("rests on non-payment with no installment due (§8.1)")

    return Score(
        disposition=disposition, acceptable=acceptable,
        standard=disposition in referee.standard,
        prohibited=disposition in referee.prohibited,
        needs_check=disposition == "needs_check", next_check_ok=next_check_ok,
        citation_ok=citation_ok, nonpayment_violation=nonpayment, problems=tuple(problems),
    )
