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
* within the guardrail: legitimate orders held, and legitimate orders declined, per
  10,000 legitimate orders, at most the caps in ``tuning.friction_guardrail``;
* best: the highest net contribution after the friction cost (the LTV proxy for each
  legitimate order declined or cancelled) among points that are feasible and within
  the guardrail; ties go to fewer reviews, then fewer declines.

:func:`tune` replays the whole grid with one history. :func:`tune_shortlist` is the
protocol's procedure: the whole grid with frozen approve-all history, then a shortlist
fixed by that screen (the frozen winner, the points one grid step from it on each
threshold, and the best ``tuning.shortlist_best_k`` qualifying frozen points) with
policy-specific history, choosing the best qualifying point of the shortlist. The
frontier keeps every replay, labelled by history.

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
    chosen: Policy | None  # None when no point qualifies
    frontier: pd.DataFrame  # one row per grid point and history replayed
    on_boundary: dict[str, bool]  # the chosen point is at the highest rate searched
    rule: str
    screened: Policy | None = None  # the frozen-history winner (shortlist tuning)
    shortlist: tuple[str, ...] = ()  # versions replayed with policy history

    @property
    def feasible_points(self) -> int:
        return int(self.frontier["feasible"].sum())


RULE = ("best in the searched grid: highest net contribution after the friction cost "
        "among points whose offered review minutes fit the minutes available and whose "
        "legitimate friction meets the guardrail; ties to fewer reviews, then fewer "
        "declines")
SHORTLIST_RULE = (
    "every grid point replayed with frozen approve-all history; the frozen winner, its "
    "neighbouring threshold pairs (one grid step on each threshold) and the best {k} "
    "frozen points that qualify replayed with policy-specific history; the choice is the "
    "best qualifying policy-history point by the same rule")
_METRICS = ("net_cents", "net_vs_approve_all_cents", "friction_cost_cents",
            "prevented_loss_cents", "reviews", "review_minutes_offered", "legitimate_orders",
            "legitimate_declined", "legitimate_held", "decided_after_shipping")


@dataclass(frozen=True)
class _Point:
    review_index: int
    decline_index: int
    review_rate: float
    decline_rate: float
    candidate: Policy


def _grid(policy: Policy, scores: Mapping[str, np.ndarray], grid: Grid
          ) -> tuple[list[_Point], list[tuple[float, float | None]],
                     list[tuple[float, float | None]]]:
    review_points = _points("review", policy, scores, grid.review_rates)
    decline_points = _points("decline", policy, scores, grid.decline_rates)
    points = [_Point(i, j, r_rate, d_rate, policy.with_thresholds(r_cut, d_cut))
              for i, (r_rate, r_cut) in enumerate(review_points)
              for j, (d_rate, d_cut) in enumerate(decline_points)]
    return points, review_points, decline_points


@dataclass(frozen=True)
class Guardrail:
    """Caps on legitimate friction, per 10,000 legitimate orders (None: no cap)."""
    held_per_10000: float | None = None
    declined_per_10000: float | None = None

    @classmethod
    def from_config(cls, policy: Mapping[str, Any] | None = None) -> Guardrail:
        tuning = (config.load("policy") if policy is None else policy)["tuning"]
        caps = tuning.get("friction_guardrail") or {}
        return cls(*(None if caps.get(k) is None else float(caps[k])
                     for k in ("legitimate_held_per_10000",
                               "legitimate_declined_per_10000")))

    def rates(self, row: Mapping[str, Any]) -> tuple[float, float]:
        legitimate = int(row["legitimate_orders"])
        if legitimate == 0:
            return 0.0, 0.0
        return (10_000 * int(row["legitimate_held"]) / legitimate,
                10_000 * int(row["legitimate_declined"]) / legitimate)

    def met(self, row: Mapping[str, Any]) -> bool:
        held, declined = self.rates(row)
        return ((self.held_per_10000 is None or held <= self.held_per_10000)
                and (self.declined_per_10000 is None
                     or declined <= self.declined_per_10000))


def _records(points: Sequence[_Point], rows: Mapping[str, Mapping[str, Any]],
             scores: Mapping[str, np.ndarray], history: str, guardrail: Guardrail,
             routed: Mapping[str, float]) -> list[dict[str, Any]]:
    out = []
    for point in points:
        candidate = point.candidate
        row = rows[candidate.version]
        held, declined = guardrail.rates(row)
        out.append({
            "history": history,
            "review_rate": point.review_rate, "decline_rate": point.decline_rate,
            "review_threshold": candidate.review_threshold,
            "decline_threshold": candidate.decline_threshold,
            "review_share": share_at(scores.get("review", np.array([])),
                                     candidate.review_threshold),
            "decline_share": share_at(scores.get("decline", np.array([])),
                                      candidate.decline_threshold),
            "policy_version": candidate.version,
            "objective_cents": int(row["net_cents"]) - int(row["friction_cost_cents"]),
            "feasible": int(row["review_minutes_offered"]) <= int(row["available_minutes"]),
            "legitimate_held_per_10000": held,
            "legitimate_declined_per_10000": declined,
            "within_guardrail": guardrail.met(row),
            "available_minutes": row["available_minutes"],
            "review_minutes_routed": routed.get(candidate.version),
            "declines": int(row["fraud_declined_checkout"])
            + int(row["legitimate_declined_checkout"]),
            **{k: row[k] for k in _METRICS},
        })
    return out


def _ranked(frontier: pd.DataFrame) -> pd.DataFrame:
    """Qualifying points, best first: feasible, within the guardrail, by the rule."""
    eligible = frontier.loc[frontier["feasible"] & frontier["within_guardrail"]]
    return eligible.sort_values(["objective_cents", "reviews", "declines"],
                                ascending=[False, True, True], kind="stable")


def _evaluate(points: Sequence[_Point], evaluate: Callable[[Sequence[Policy]],
              Sequence[Mapping[str, Any]]]) -> dict[str, Mapping[str, Any]]:
    to_run: dict[str, Policy] = {}
    for point in points:
        to_run.setdefault(point.candidate.version, point.candidate)
    return dict(zip(to_run, evaluate(list(to_run.values())), strict=True))


def _choice(policy: Policy, frontier: pd.DataFrame, review_points, decline_points
            ) -> tuple[Policy | None, dict[str, bool]]:
    ranked = _ranked(frontier)
    if ranked.empty:
        return None, {"review": False, "decline": False}
    best = ranked.iloc[0]
    chosen = policy.with_thresholds(_none(best["review_threshold"]),
                                    _none(best["decline_threshold"]))
    boundary = {  # the chosen threshold is the one at the highest rate searched
        "review": len(review_points) > 1
        and _same(chosen.review_threshold, review_points[-1][1]),
        "decline": len(decline_points) > 1
        and _same(chosen.decline_threshold, decline_points[-1][1]),
    }
    return chosen, boundary


def tune(
    policy: Policy,
    scores: Mapping[str, np.ndarray],
    evaluate: Callable[[Sequence[Policy]], Sequence[Mapping[str, Any]]],
    grid: Grid,
    *,
    routed_minutes: Callable[[Policy], float] | None = None,
    guardrail: Guardrail | None = None,
    history: str = "",
) -> Tuned:
    """Search ``grid`` for ``policy`` with one replay history.

    ``scores`` holds the validation orders' checkout scores for the review and the
    decline signal (keys ``review`` and ``decline``); ``evaluate`` replays a list of
    candidates and returns their outcome rows (queue_sim.outcomes.outcome_row) in the
    same order. ``guardrail`` caps legitimate friction (None: no caps).
    ``routed_minutes`` (optional) gives a candidate's review minutes from checkout
    routing alone, reported beside the replay's. ``history`` labels the frontier rows.
    """
    points, review_points, decline_points = _grid(policy, scores, grid)
    rows = _evaluate(points, evaluate)
    routed = ({version: float(routed_minutes(policy_of))
               for version, policy_of in {p.candidate.version: p.candidate
                                          for p in points}.items()}
              if routed_minutes is not None else {})
    frontier = pd.DataFrame(_records(points, rows, scores, history,
                                     guardrail or Guardrail(), routed))
    chosen, boundary = _choice(policy, frontier, review_points, decline_points)
    return Tuned(policy.name, chosen, frontier, boundary, RULE)


def tune_shortlist(
    policy: Policy,
    scores: Mapping[str, np.ndarray],
    frozen: Callable[[Sequence[Policy]], Sequence[Mapping[str, Any]]],
    rebuilt: Callable[[Sequence[Policy]], Sequence[Mapping[str, Any]]],
    grid: Grid,
    *,
    k: int = 5,
    guardrail: Guardrail | None = None,
    routed_minutes: Callable[[Policy], float] | None = None,
) -> Tuned:
    """Screen ``grid`` with frozen history, choose with policy-specific history.

    ``frozen`` and ``rebuilt`` replay candidates with frozen approve-all and with
    policy-specific history. Every grid point is replayed by ``frozen``; the shortlist is
    the frozen winner, the points one grid step from it on each threshold, and the best
    ``k`` qualifying frozen points; ``rebuilt`` replays the shortlist and the choice is
    its best qualifying point. The frontier holds both histories (column ``history``).
    """
    screen = tune(policy, scores, frozen, grid, routed_minutes=routed_minutes,
                  guardrail=guardrail, history="frozen")
    points, review_points, decline_points = _grid(policy, scores, grid)
    first = {}  # each threshold pair at its first grid position
    for point in points:
        first.setdefault(point.candidate.version, point)
    ranked = _ranked(screen.frontier)
    picked: list[_Point] = []
    if screen.chosen is not None:
        # tied scores can put the winner at several grid positions: every one of them
        # brings its neighbours
        spots = [(p.review_index, p.decline_index) for p in points
                 if p.candidate.version == screen.chosen.version]
        picked += [p for p in points
                   if any(abs(p.review_index - i) <= 1 and abs(p.decline_index - j) <= 1
                          for i, j in spots)]
    picked += [first[v] for v in ranked["policy_version"].drop_duplicates().head(k)]
    shortlist = list({p.candidate.version: p for p in picked}.values())
    rows = _evaluate(shortlist, rebuilt) if shortlist else {}
    routed = dict(zip(screen.frontier["policy_version"],
                      screen.frontier["review_minutes_routed"], strict=True))
    second = pd.DataFrame(_records(shortlist, rows, scores, "policy",
                                   guardrail or Guardrail(), routed),
                          columns=screen.frontier.columns)
    frontier = pd.concat([screen.frontier, second], ignore_index=True)
    chosen, boundary = (_choice(policy, second, review_points, decline_points)
                        if len(second) else (None, {"review": False, "decline": False}))
    return Tuned(policy.name, chosen, frontier, boundary,
                 SHORTLIST_RULE.format(k=k) + "; " + RULE, screened=screen.chosen,
                 shortlist=tuple(p.candidate.version for p in shortlist))


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
