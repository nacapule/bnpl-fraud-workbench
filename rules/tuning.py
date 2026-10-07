"""Threshold tuning through the replay, on the validation window.

Each policy's review and decline thresholds are chosen by replaying the validation
window at the base capacity for every point of a grid of score cut-points the policy
can attain, and taking the best feasible point by the rule in ``config/policy.yaml``
``tuning``:

* cut-points: the policy's score at each listed review rate and decline rate (the
  share of validation orders at or above it, scored at checkout with the world-level
  context); rate 0 switches the route off, so "no review" and "no decline" are always
  in the grid;
* feasible: the review minutes the point offers (every routed order's review time)
  fit in the analyst minutes available over the window; a point whose checkout routing
  alone (world-level context, before any block) offers more than
  :data:`SKIP_MARGIN` times the available minutes is reported infeasible without a
  replay (``replayed`` false);
* best: the highest net contribution after the friction cost (the LTV proxy for each
  legitimate order declined or cancelled); ties go to fewer reviews, then fewer
  declines.

The result is the best point *in the searched grid*, reported with the whole frontier,
whether it lies on the grid's boundary (the highest review or decline rate searched,
where a wider grid might do better), and explicitly when no point is feasible (no
policy is chosen then; nothing falls back silently). The chosen thresholds are the
policy the replay evaluates, identified by its version; no other copy of the bands
exists.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core import config
from queue_sim.policies import Policy


@dataclass(frozen=True)
class Grid:
    review_rates: tuple[float, ...]
    decline_rates: tuple[float, ...]

    @classmethod
    def from_config(cls, policy: Mapping[str, Any] | None = None) -> Grid:
        tuning = (config.load("policy") if policy is None else policy)["tuning"]
        return cls(tuple(float(r) for r in tuning["review_rates"]),
                   tuple(float(r) for r in tuning["decline_rates"]))


def cut_point(scores: np.ndarray, rate: float) -> float | None:
    """The score at or above which ``rate`` of the orders lie (None for rate 0)."""
    if rate <= 0:
        return None
    values = np.sort(np.asarray(scores, dtype=float)[~np.isnan(scores)])[::-1]
    if not len(values):
        return None
    k = min(max(math.ceil(rate * len(values)), 1), len(values))
    return float(values[k - 1])


@dataclass(frozen=True)
class Tuned:
    policy: str
    chosen: Policy | None  # None when no grid point is feasible
    frontier: pd.DataFrame  # one row per grid point
    on_boundary: dict[str, bool]  # the chosen point is at the highest rate searched
    rule: str

    @property
    def feasible_points(self) -> int:
        return int(self.frontier["feasible"].sum())


RULE = ("best in the searched grid: highest net contribution after the friction cost "
        "among points whose offered review minutes fit the minutes available; ties to "
        "fewer reviews, then fewer declines; points whose checkout routing alone offers "
        "more than 1.25 times the available minutes are infeasible without a replay")
SKIP_MARGIN = 1.25
_METRICS = ("net_cents", "net_vs_approve_all_cents", "friction_cost_cents",
            "prevented_loss_cents", "reviews", "review_minutes_offered", "legitimate_declined",
            "legitimate_held", "decided_after_shipping")


def tune(
    policy: Policy,
    scores: Mapping[str, np.ndarray],
    evaluate: Callable[[Sequence[Policy]], Sequence[Mapping[str, Any]]],
    grid: Grid,
    *,
    routed_minutes: Callable[[Policy], float] | None = None,
    available_minutes: float | None = None,
) -> Tuned:
    """Search ``grid`` for ``policy``.

    ``scores`` holds the validation orders' checkout scores for the review and the
    decline signal (keys ``review`` and ``decline``); ``evaluate`` replays a list of
    candidates and returns their outcome rows (queue_sim.outcomes.outcome_row) in the
    same order. With ``routed_minutes`` (a candidate's review minutes from checkout
    routing alone) and ``available_minutes``, points far over capacity are not replayed.
    """
    review_points = _points("review", policy, scores, grid.review_rates)
    decline_points = _points("decline", policy, scores, grid.decline_rates)
    points = []
    for (r_rate, r_cut) in review_points:
        for (d_rate, d_cut) in decline_points:
            points.append((r_rate, d_rate, policy.with_thresholds(r_cut, d_cut)))
    routed: dict[str, float] = {}
    to_run: dict[str, Policy] = {}
    for _, _, candidate in points:
        if candidate.version in routed or candidate.version in to_run:
            continue
        if routed_minutes is not None and available_minutes is not None:
            routed[candidate.version] = float(routed_minutes(candidate))
            if routed[candidate.version] > SKIP_MARGIN * available_minutes:
                continue
        to_run[candidate.version] = candidate
    rows = dict(zip(to_run, evaluate(list(to_run.values())), strict=True))

    records = []
    for r_rate, d_rate, candidate in points:
        row = rows.get(candidate.version)
        record: dict[str, Any] = {
            "review_rate": r_rate, "decline_rate": d_rate,
            "review_threshold": candidate.review_threshold,
            "decline_threshold": candidate.decline_threshold,
            "policy_version": candidate.version, "replayed": row is not None,
            "review_minutes_routed": routed.get(candidate.version),
        }
        if row is None:
            record.update({"objective_cents": None, "feasible": False,
                           "available_minutes": available_minutes, "declines": None,
                           **{k: None for k in _METRICS}})
        else:
            record.update({
                "objective_cents": int(row["net_cents"]) - int(row["friction_cost_cents"]),
                "feasible": int(row["review_minutes_offered"]) <= int(row["available_minutes"]),
                "available_minutes": row["available_minutes"],
                "declines": int(row["fraud_declined_checkout"])
                + int(row["legitimate_declined_checkout"]),
                **{k: row[k] for k in _METRICS},
            })
        records.append(record)
    frontier = pd.DataFrame(records)
    feasible = frontier.loc[frontier["feasible"]]
    if feasible.empty:
        return Tuned(policy.name, None, frontier, {"review": False, "decline": False}, RULE)
    best = feasible.sort_values(["objective_cents", "reviews", "declines"],
                                ascending=[False, True, True], kind="stable").iloc[0]
    chosen = policy.with_thresholds(_none(best["review_threshold"]),
                                    _none(best["decline_threshold"]))
    boundary = {
        "review": len(review_points) > 1 and best["review_rate"] == review_points[-1][0],
        "decline": len(decline_points) > 1 and best["decline_rate"] == decline_points[-1][0],
    }
    return Tuned(policy.name, chosen, frontier, boundary, RULE)


def _points(which: str, policy: Policy, scores: Mapping[str, np.ndarray],
            rates: Sequence[float]) -> list[tuple[float, float | None]]:
    """(rate, threshold) pairs for one route; a fixed route keeps its own threshold."""
    signal = policy.review if which == "review" else policy.decline
    if signal is None:
        return [(0.0, None)]
    if which not in policy.tunable:
        fixed = policy.review_threshold if which == "review" else policy.decline_threshold
        return [(float("nan"), fixed)]
    return [(rate, cut_point(scores[which], rate)) for rate in rates]


def _none(value: Any) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) \
        else float(value)
