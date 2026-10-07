"""Routing with fixed thresholds, stable alert ids, and tuning through the replay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from replay_support import (
    CALENDAR,
    SETTINGS,
    StubContext,
    StubScorer,
    always_on_roster,
    frozen,
    mini_tables,
    scored_policy,
    stub_world,
    verification,
)

from queue_sim import outcomes, policies
from queue_sim.replay import replay
from queue_sim.reviewer import Reviewer
from queue_sim.roster import Roster, Shift
from rules.tuning import Grid, cut_point, tune

T = pd.Timestamp
WHOLE = (T("2024-12-01"), T("2025-06-15"))


def _rows(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"order_id": np.arange(1, len(scores) + 1), "test_score": scores})


# ------------------------------------------------------------------ fixed thresholds


def _daily_top_k(rows: pd.DataFrame, k: int) -> set[int]:
    """The old selection: the k highest scores of the day go to review."""
    return set(rows.nlargest(k, "test_score")["order_id"])


def test_an_orders_route_does_not_depend_on_the_other_orders() -> None:
    quiet_day = _rows([0.2, 0.9, 0.4])
    busy_day = _rows([0.2, 0.9, 0.4, 0.95, 0.97])  # two riskier orders arrive later
    # a daily top-2 sends order 2 to review on the quiet day but not on the busy one
    assert 2 in _daily_top_k(quiet_day, 2) and 2 not in _daily_top_k(busy_day, 2)
    policy = scored_policy(review=0.5)
    quiet, busy = policy.route(quiet_day), policy.route(busy_day)
    assert quiet.set_index("order_id")["route"].to_dict() == \
        busy.set_index("order_id")["route"].iloc[:3].to_dict()
    one_by_one = pd.concat([policy.route(busy_day.iloc[[i]]) for i in range(5)])
    assert one_by_one["route"].tolist() == busy["route"].tolist()


def test_decline_comes_before_review_and_none_switches_a_route_off() -> None:
    rows = _rows([0.1, 0.6, 0.95])
    assert scored_policy(0.5, 0.9).route(rows)["route"].tolist() == [
        "approve", "review", "auto_decline"]
    assert scored_policy(None, None).route(rows)["route"].tolist() == ["approve"] * 3
    assert policies.approve_all().route(rows)["route"].tolist() == ["approve"] * 3


def test_alert_ids_name_the_order_and_the_policy_version() -> None:
    """Changing a band never makes an alert id point at a different order."""
    rows = _rows([0.6, 0.2, 0.7, 0.8, 0.3])
    low, high = scored_policy(0.5), scored_policy(0.65)
    ids = {}
    for policy in (low, high):
        routed = policy.route(rows)
        flagged = routed.loc[routed["route"] == "review", "order_id"]
        old_style = dict(zip(range(1, len(flagged) + 1), flagged, strict=True))  # numbered
        ids[policy.version] = (dict(zip(policies.alert_id(flagged, policy.version), flagged,
                                        strict=True)), old_style)
    (new_low, old_low), (new_high, old_high) = ids[low.version], ids[high.version]
    assert old_low[1] != old_high[1]  # sequential numbering: alert 1 changed orders
    assert set(new_low).isdisjoint(new_high)
    for mapping in (new_low, new_high):
        assert all(int(alert.split(":")[0]) == order for alert, order in mapping.items())
    assert low.version != high.version and low.version == scored_policy(0.5).version


def test_expected_loss_declines_when_expected_loss_beats_the_false_decline_cost() -> None:
    from core import ledger

    terms = ledger.ProductTerms.from_config()
    policy = policies.expected_loss(StubScorer(), terms, 1500).with_thresholds(None, 0.0)
    rows = pd.DataFrame({"order_id": [1, 2], "test_score": [0.5, 0.01],
                         "order_exposure_cents": [10_000, 10_000],
                         "amount_cents": [12_000, 12_000]})
    # 0.5 x 10,000 > 0.5 x (600 fee + 1,500 LTV); 0.01 x 10,000 < 0.99 x 2,100
    assert policy.route(rows)["route"].tolist() == ["auto_decline", "approve"]
    assert policy.tunable == ("review",)


# ------------------------------------------------------------------ tuning


def test_cut_points_are_scores_the_policy_attains() -> None:
    scores = np.array([0.1, 0.4, 0.4, 0.8, 0.9, np.nan])
    assert cut_point(scores, 0.0) is None
    assert cut_point(scores, 0.2) == 0.9
    assert cut_point(scores, 0.5) == 0.4
    assert cut_point(scores, 1.0) == 0.1
    assert all(cut_point(scores, r) in set(scores[~np.isnan(scores)]) for r in (0.3, 0.7))


def _tuning(tables, roster, grid, *, routed=False):
    stub = StubContext(tables, scores={o: float(o) for o in tables["order_attempts"]["order_id"]})
    world = stub_world(tables, stub)
    attempts = world.orders(*WHOLE)
    scores = attempts["order_id"].astype(float).to_numpy()
    replayed = []

    def evaluate(candidates):
        rows = []
        for candidate in candidates:
            replayed.append(candidate.version)
            result = replay(world, candidate, window=WHOLE, roster=roster, calendar=CALENDAR,
                            reviewer=Reviewer(), verification=verification(),
                            history=frozen(world, stub), settings=SETTINGS)
            rows.append(outcomes.outcome_row(result, world, keys={}, ltv_cents=1500))
        return rows

    extra = {}
    if routed:  # ten minutes per order the checkout routing sends to review
        extra = {"routed_minutes": lambda c: 10.0 * int(
                     (c.route(world.context)["route"] == "review").sum()),
                 "available_minutes": roster.available_minutes(
                     int(WHOLE[0].timestamp()), int(WHOLE[1].timestamp()))}
    tuned = tune(scored_policy(None, None), {"review": scores, "decline": scores}, evaluate,
                 grid, **extra)
    assert len(replayed) == len(set(replayed))  # each candidate replayed once
    return tuned, replayed


def test_tuning_reports_the_frontier_and_a_boundary_optimum() -> None:
    tables = mini_tables()
    tuned, _ = _tuning(tables, always_on_roster(), Grid((0.0, 0.25, 0.5), (0.0, 0.1, 0.2)))
    frontier = tuned.frontier
    assert len(frontier) == 9 and frontier["feasible"].all()
    best = frontier.sort_values("objective_cents", ascending=False).iloc[0]
    assert tuned.chosen is not None
    assert tuned.chosen.decline_threshold == best["decline_threshold"]
    # declining the highest-id orders (the never-pay ring, promotion farmers and others)
    # wins here, and the best decline rate is the highest searched: say so
    assert best["decline_rate"] == 0.2 and tuned.on_boundary["decline"]
    assert "searched grid" in tuned.rule


def test_when_no_point_fits_the_capacity_nothing_is_chosen() -> None:
    tables = mini_tables()
    nobody = Roster((Shift("none", (0,), 0, 1),), {"none": 0})
    tuned, replayed = _tuning(tables, nobody, Grid((0.25, 0.5), (0.1,)))
    assert tuned.chosen is None and tuned.feasible_points == 0
    assert not tuned.frontier["feasible"].any() and len(replayed) == 2
    # with the checkout routing known, points far over capacity are not replayed
    tuned, replayed = _tuning(tables, nobody, Grid((0.25, 0.5), (0.1,)), routed=True)
    assert tuned.chosen is None and replayed == []
    assert not tuned.frontier["replayed"].any() and tuned.frontier["objective_cents"].isna().all()


def test_only_points_far_over_capacity_are_skipped() -> None:
    tables = mini_tables()
    # one analyst, three minutes on Mondays: 84 minutes over the window; checkout routing
    # offers 60 minutes at a 25% review rate and 140 at 50%
    roster = Roster((Shift("mon", (0,), 9 * 60, 3),), {"mon": 1})
    tuned, replayed = _tuning(tables, roster, Grid((0.0, 0.25, 0.5), (0.0,)), routed=True)
    frontier = tuned.frontier.set_index("review_rate")
    assert frontier["available_minutes"].iloc[0] == 84
    assert frontier["review_minutes_routed"].tolist() == [0, 60, 140]
    assert frontier["replayed"].tolist() == [True, True, False]  # 140 > 1.25 x 84
    assert len(replayed) == 2


def test_the_tuned_policy_is_the_one_the_replay_evaluates() -> None:
    tables = mini_tables()
    tuned, _ = _tuning(tables, always_on_roster(), Grid((0.0, 0.25), (0.0, 0.1)))
    chosen_row = tuned.frontier.loc[tuned.frontier["policy_version"] == tuned.chosen.version]
    assert len(chosen_row) == 1
    assert float(chosen_row["decline_threshold"].iloc[0]) == pytest.approx(
        tuned.chosen.decline_threshold)


# ------------------------------------------------------------------ a follow-up as a policy variant


def test_a_rule_change_is_measured_by_the_replay_not_by_its_triggers() -> None:
    """A proposed change (review orders scoring 20 or more instead of nothing) is judged
    by what the replay's actions change: prevented loss in cents and friction counts,
    not by how many fraud orders the new trigger touches."""
    tables = mini_tables()
    scores = {o: float(o) for o in tables["order_attempts"]["order_id"]}
    stub = StubContext(tables, scores=scores)
    world = stub_world(tables, stub)

    def measure(policy):
        result = replay(world, policy, window=WHOLE, roster=always_on_roster(),
                        calendar=CALENDAR, reviewer=Reviewer(), verification=verification(),
                        history=frozen(world, stub), settings=SETTINGS)
        return result, outcomes.outcome_row(result, world, keys={}, ltv_cents=1500)

    _, base = measure(scored_policy(None))
    result, variant = measure(scored_policy(20.0))
    truth = outcomes.truth(tables, world.observed_until).set_index("order_id")["truth"]
    triggered_fraud = int((result.reviews["order_id"].map(truth) == "fraud").sum())
    assert variant["reviews"] == 7 and triggered_fraud == 6  # orders 20-25 are fraud
    # quiet evidence: every review clears, so the trigger prevents nothing and costs time
    assert variant["prevented_loss_cents"] - base["prevented_loss_cents"] == 0
    assert variant["review_minutes_used"] == 70
