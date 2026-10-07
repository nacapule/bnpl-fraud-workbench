"""The simulated analyst: an evidence-based procedure, and how check outcomes are drawn.

The procedure follows the fraud policy through ``core.evidence`` and nothing else. At
the review, and again whenever a check completes, it classifies the order's as-of
context row with the checks completed so far and takes the decision table's standard
disposition: ``clear``, ``decline`` or ``escalate`` decide the order; ``hold`` runs the
checks the policy requires that have not run yet (FP-2 §5.2, at most one of each). It
never sees labels or simulation truth.

Check outcomes are observations the simulated world supplies. :class:`Verification`
draws them per order from who placed it (an actor class mapped from the latent
pattern), at the rates in ``config/policy.yaml`` ``reviewer``, keyed by the order's id
so every policy sees the same outcome for the same order and check. Truth generates
the observation; it never chooses the action: first-party fraud passes the checks at
the legitimate rate, because the checks establish who is ordering, never intent.

:class:`PerfectReviewer` is the labelled upper bound: it declines every order whose
latent intent is fraud or abuse and clears the rest, without checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core import config, evidence
from core.actions import Check, CheckOutcome, Disposition
from core.evidence import CheckResult
from queue_sim.draws import keyed_uniform, lognormal

LEGITIMATE = "legitimate"
ACTOR_CLASSES = ("legitimate", "first_party", "takeover", "third_party")


@dataclass(frozen=True)
class Decision:
    """One evaluation: the disposition taken and what it rested on."""

    disposition: Disposition
    row: str  # the §6.6 row (core.evidence.TABLE_ROWS)
    clause: str  # the clause the disposition cites
    families: int  # adverse families present (the evidence strength)
    rules: tuple[str, ...]  # §6.2 conditions that held
    checks_to_run: tuple[Check, ...] = ()


class Reviewer:
    """The evidence-based procedure (reads the context row and completed checks only)."""

    name = "evidence"

    def decide(self, order_id: int, row: Mapping[str, Any],
               checks: Sequence[CheckResult]) -> Decision:
        found = evidence.classify(row, checks)
        allowed = evidence.permitted_actions(found)
        (standard,) = allowed.standard
        disposition = Disposition(standard)
        return Decision(
            disposition=disposition, row=allowed.row, clause=allowed.clauses[standard],
            families=len(found.families_present),
            rules=tuple(c.rule for c in found.conditions),
            checks_to_run=tuple(sorted(allowed.required_checks))
            if disposition is Disposition.HOLD else (),
        )


@dataclass(frozen=True)
class PerfectReviewer:
    """The labelled upper bound: decides from simulation truth, with no checks."""

    fraud_orders: frozenset[int]
    name: str = "perfect"

    @classmethod
    def from_latent(cls, latent_orders: pd.DataFrame) -> PerfectReviewer:
        bad = latent_orders.loc[latent_orders["intent"] != "legitimate", "order_id"]
        return cls(frozenset(int(order) for order in bad))

    def decide(self, order_id: int, row: Mapping[str, Any],
               checks: Sequence[CheckResult]) -> Decision:
        if order_id in self.fraud_orders:
            return Decision(Disposition.DECLINE, "§6.6(a)", "upper bound", 0, ())
        return Decision(Disposition.CLEAR, "§6.6(c)", "upper bound", 0, ())


@dataclass(frozen=True)
class Verification:
    """Check outcomes and response delays, drawn per order and check."""

    seed: int
    actor_class: Mapping[int, str] = field(repr=False)  # order id -> actor class
    rates: Mapping[str, Mapping[str, Mapping[str, float]]]
    response_median_hours: float
    response_sigma: float

    def __post_init__(self) -> None:
        for actor, checks in self.rates.items():
            if actor not in ACTOR_CLASSES:
                raise ValueError(f"unknown actor class {actor!r}")
            for check, probs in checks.items():
                Check(check)
                passed, failed = float(probs["passed"]), float(probs["failed"])
                if passed < 0 or failed < 0 or passed + failed > 1:
                    raise ValueError(f"{actor}/{check}: rates must be a distribution")
        missing = set(ACTOR_CLASSES) - set(self.rates)
        if missing:
            raise ValueError(f"no verification rates for {sorted(missing)}")

    @classmethod
    def from_config(cls, seed: int, latent_orders: pd.DataFrame,
                    policy: Mapping[str, Any] | None = None, *,
                    rates: str = "verification") -> Verification:
        """Rates from ``config/policy.yaml`` ``reviewer`` (``verification_weak`` for the
        sensitivity); actor classes from the latent order patterns."""
        reviewer = (config.load("policy") if policy is None else policy)["reviewer"]
        mapping = reviewer["actor_class"]
        unknown = set(latent_orders["pattern_id"].dropna()) - set(mapping)
        if unknown:
            raise ValueError(f"patterns without an actor class: {sorted(unknown)}")
        classes = latent_orders["pattern_id"].map(mapping).fillna(LEGITIMATE)
        return cls(
            seed=seed,
            actor_class=dict(zip(latent_orders["order_id"].astype(int), classes, strict=True)),
            rates=reviewer[rates],
            response_median_hours=float(reviewer["response_hours"]["median"]),
            response_sigma=float(reviewer["response_hours"]["sigma"]),
        )

    def draw(self, order_id: int, check: Check) -> tuple[CheckOutcome, int]:
        """The check's outcome for this order and the seconds until the answer.

        ``no_response`` has no answer time of its own: it completes when the hold
        times out, which the replay handles.
        """
        actor = self.actor_class.get(int(order_id), LEGITIMATE)
        probs = self.rates[actor][check.value]
        u = float(keyed_uniform(self.seed, f"check-outcome-{check.value}", order_id)[0])
        if u < probs["passed"]:
            outcome = CheckOutcome.PASSED
        elif u < probs["passed"] + probs["failed"]:
            outcome = CheckOutcome.FAILED
        else:
            outcome = CheckOutcome.NO_RESPONSE
        v = keyed_uniform(self.seed, f"check-delay-{check.value}", order_id)
        delay = lognormal(v, median=self.response_median_hours, sigma=self.response_sigma)
        return outcome, int(round(float(delay[0]) * 3600))


def service_seconds(seed: int, order_ids: np.ndarray, mean_minutes: float,
                    sigma: float) -> np.ndarray:
    """Each order's review time (lognormal with the configured arithmetic mean)."""
    u = keyed_uniform(seed, "review-service", order_ids)
    return np.maximum(np.rint(lognormal(u, mean=mean_minutes, sigma=sigma) * 60), 1).astype(
        np.int64)
