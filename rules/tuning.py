"""Threshold tuning through the replay, on the validation window.

Each policy's review and decline thresholds are chosen by replaying the validation
window at the base capacity for every point of a grid of score cut-points the policy
can attain, and taking the best feasible point by the rule in ``config/policy.yaml``
``tuning``:

* cut-points: for each listed review rate and decline rate, the lowest score the
  policy attains on the validation orders (scored at checkout with the world-level
  context) whose share of orders at or above it does not exceed the rate. With tied
  scores (the rule score takes few values) the share reached can be well below the
  rate, and several rates can share one cut-point (one point then); never does a rate
  send every tied order over the line. Rate 0 switches the route off, so "no review" and
  "no decline" are always in the grid. The frontier gives the share each cut-point
  reaches;
* feasible: the review minutes the point offers in the replay (every order that
  reached the queue, with its review time) fit in the analyst minutes available over
  the window. Every point is replayed: checkout routing alone can offer far more than
  the replay does, because a decline blocks the account's later orders;
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
    """The lowest attained score with at most ``rate`` of the orders at or above it
    (None for rate 0; the highest score when even it is shared by more)."""
    if rate <= 0:
        return None
    values = np.asarray(scores, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return None
    distinct, counts = np.unique(values, return_counts=True)  # ascending
    share = np.cumsum(counts[::-1])[::-1] / len(values)  # at or above each value
    within = np.flatnonzero(share <= rate + 1e-12)
    return float(distinct[within[0]] if len(within) else distinct[-1])


def share_at(scores: np.ndarray, threshold: float | None) -> float:
    """The share of scored orders at or above ``threshold`` (0 when the route is off)."""
    values = np.asarray(scores, dtype=float)
    values = values[~np.isnan(values)]
    if threshold is None or not len(values):
        return 0.0
    return float((values >= threshold).mean())


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
        "fewer reviews, then fewer declines")
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
) -> Tuned:
    """Search ``grid`` for ``policy``.

    ``scores`` holds the validation orders' checkout scores for the review and the
    decline signal (keys ``review`` and ``decline``); ``evaluate`` replays a list of
    candidates and returns their outcome rows (queue_sim.outcomes.outcome_row) in the
    same order. ``routed_minutes`` (optional) gives a candidate's review minutes from
    checkout routing alone, reported beside the replay's.
    """
    review_points = _points("review", policy, scores, grid.review_rates)
    decline_points = _points("decline", policy, scores, grid.decline_rates)
    points = []
    for (r_rate, r_cut) in review_points:
        for (d_rate, d_cut) in decline_points:
            points.append((r_rate, d_rate, policy.with_thresholds(r_cut, d_cut)))
    to_run: dict[str, Policy] = {}
    for _, _, candidate in points:
        to_run.setdefault(candidate.version, candidate)
    rows = dict(zip(to_run, evaluate(list(to_run.values())), strict=True))
    routed = ({version: float(routed_minutes(c)) for version, c in to_run.items()}
              if routed_minutes is not None else {})

    records = []
    for r_rate, d_rate, candidate in points:
        row = rows[candidate.version]
        records.append({
            "review_rate": r_rate, "decline_rate": d_rate,
            "review_threshold": candidate.review_threshold,
            "decline_threshold": candidate.decline_threshold,
            "review_share": share_at(scores.get("review", np.array([])),
                                     candidate.review_threshold),
            "decline_share": share_at(scores.get("decline", np.array([])),
                                      candidate.decline_threshold),
            "policy_version": candidate.version,
            "objective_cents": int(row["net_cents"]) - int(row["friction_cost_cents"]),
            "feasible": int(row["review_minutes_offered"]) <= int(row["available_minutes"]),
            "available_minutes": row["available_minutes"],
            "review_minutes_routed": routed.get(candidate.version),
            "declines": int(row["fraud_declined_checkout"])
            + int(row["legitimate_declined_checkout"]),
            **{k: row[k] for k in _METRICS},
        })
    frontier = pd.DataFrame(records)
    feasible = frontier.loc[frontier["feasible"]]
    if feasible.empty:
        return Tuned(policy.name, None, frontier, {"review": False, "decline": False}, RULE)
    best = feasible.sort_values(["objective_cents", "reviews", "declines"],
                                ascending=[False, True, True], kind="stable").iloc[0]
    chosen = policy.with_thresholds(_none(best["review_threshold"]),
                                    _none(best["decline_threshold"]))
    boundary = {  # the chosen threshold is the one at the highest rate searched
        "review": len(review_points) > 1
        and _same(chosen.review_threshold, review_points[-1][1]),
        "decline": len(decline_points) > 1
        and _same(chosen.decline_threshold, decline_points[-1][1]),
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


def _same(a: float | None, b: float | None) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and float(a) == float(b))


def _none(value: Any) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) \
        else float(value)
