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
from rules.tuning import Grid, Guardrail, cut_point, tune, tune_shortlist

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
    assert cut_point(scores, 0.5) == 0.8  # 0.4 would send 4 of 5
    assert cut_point(scores, 0.8) == 0.4
    assert cut_point(scores, 1.0) == 0.1
    assert cut_point(scores, 0.1) == 0.9  # even the top score is 1 in 5: it is the floor
    assert all(cut_point(scores, r) in set(scores[~np.isnan(scores)]) for r in (0.3, 0.7))


def test_tied_scores_never_send_every_tied_order_over_the_line() -> None:
    """A rule score is 0 for most orders: the old cut-point at a 5% rate was 0, which
    routed every order to review (23,446 of 23,801 on a development world)."""
    scores = np.array([0.0] * 96 + [30.0, 30.0, 60.0, 90.0])
    old_style = np.sort(scores)[::-1][int(np.ceil(0.05 * len(scores))) - 1]
    assert old_style == 0.0 and (scores >= old_style).mean() == 1.0
    assert cut_point(scores, 0.05) == 30.0 and (scores >= 30.0).mean() == 0.04
    assert cut_point(scores, 0.12) == cut_point(scores, 0.05)  # one point, not two
    assert cut_point(scores, 0.02) == 60.0


def _tuning(tables, roster, grid, *, stub=None, window=WHOLE, settings=SETTINGS):
    stub = stub or StubContext(tables, scores={o: float(o)
                                               for o in tables["order_attempts"]["order_id"]})
    world = stub_world(tables, stub)
    scores = world.context.set_index("order_id")["test_score"].reindex(
        world.orders(*window)["order_id"]).to_numpy(dtype=float)
    replayed = []

    def evaluate(candidates):
        rows = []
        for candidate in candidates:
            replayed.append(candidate.version)
            result = replay(world, candidate, window=window, roster=roster, calendar=CALENDAR,
                            reviewer=Reviewer(), verification=verification(),
                            history=frozen(world, stub), settings=settings)
            rows.append(outcomes.outcome_row(result, world, keys={}, ltv_cents=1500))
        return rows

    def routed(candidate):  # review minutes from checkout routing alone
        rows = world.context.set_index("order_id").reindex(
            world.orders(*window)["order_id"]).reset_index()
        return settings.service_mean_minutes * int(
            (candidate.route(rows)["route"] == "review").sum())

    tuned = tune(scored_policy(None, None), {"review": scores, "decline": scores}, evaluate,
                 grid, routed_minutes=routed)
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


def test_a_point_whose_routing_exceeds_capacity_can_be_feasible_through_blocks() -> None:
    """Orders 11 and 13 of one account would take 20 review minutes against 10; the
    first review declines on settled evidence and blocks the account, so order 13 never
    reaches the queue and the point fits. Judging feasibility by checkout routing alone
    would have discarded it."""
    tables = mini_tables()
    stub = StubContext(tables, scores={11: 5.0, 13: 5.0},
                       overrides={11: {"unauthorized_disputes_lost_user": 1}})
    saturday = Roster((Shift("sat", (5,), 12 * 60 + 30, 10),), {"sat": 1})  # 12:30-12:40
    window = (T("2025-02-01"), T("2025-02-05"))
    tuned, replayed = _tuning(tables, saturday, Grid((0.0, 0.5), (0.0,)), stub=stub,
                              window=window)
    point = tuned.frontier.set_index("review_rate").loc[0.5]
    assert point["review_minutes_routed"] == 20 and point["available_minutes"] == 10
    assert point["review_minutes_offered"] == 10 and point["feasible"]
    assert len(replayed) == 2


def test_a_boundary_is_flagged_when_the_widest_rate_gives_the_chosen_threshold() -> None:
    """Tied scores make several rates share one cut-point: the chosen threshold is also
    the widest searched, so the optimum may lie beyond the grid."""
    scores = np.array([9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0])
    base = {"net_cents": 0, "friction_cost_cents": 0, "review_minutes_offered": 0,
            "available_minutes": 100, "fraud_declined_checkout": 0,
            "legitimate_declined_checkout": 0, "net_vs_approve_all_cents": 0,
            "prevented_loss_cents": 0, "reviews": 0, "legitimate_orders": 8,
            "legitimate_declined": 0, "legitimate_held": 0, "decided_after_shipping": 0}

    def evaluate(candidates):  # reviewing at 9 or more earns the most
        return [{**base, "net_cents": 100 if c.review_threshold == 9.0 else 0,
                 "reviews": 4 if c.review_threshold == 9.0 else 0} for c in candidates]

    tuned = tune(scored_policy(None, None), {"review": scores, "decline": scores}, evaluate,
                 Grid((0.0, 0.125, 0.5), (0.0,)))
    frontier = tuned.frontier.set_index("review_rate")
    assert frontier.loc[0.125, "review_threshold"] == frontier.loc[0.5, "review_threshold"]
    assert tuned.chosen.review_threshold == 9.0 and tuned.on_boundary["review"]


def test_the_tuned_policy_is_the_one_the_replay_evaluates() -> None:
    tables = mini_tables()
    tuned, _ = _tuning(tables, always_on_roster(), Grid((0.0, 0.25), (0.0, 0.1)))
    chosen_row = tuned.frontier.loc[tuned.frontier["policy_version"] == tuned.chosen.version]
    assert len(chosen_row) == 1
    assert float(chosen_row["decline_threshold"].iloc[0]) == pytest.approx(
        tuned.chosen.decline_threshold)


ROW = {"net_cents": 0, "friction_cost_cents": 0, "review_minutes_offered": 0,
       "available_minutes": 100, "fraud_declined_checkout": 0,
       "legitimate_declined_checkout": 0, "net_vs_approve_all_cents": 0,
       "prevented_loss_cents": 0, "reviews": 0, "legitimate_orders": 10_000,
       "legitimate_declined": 0, "legitimate_held": 0, "decided_after_shipping": 0}
SCORES = np.arange(100) / 100  # cut-points 0.99, 0.98, 0.95, 0.90, 0.80
GRID = Grid((0.0, 0.01, 0.02, 0.05, 0.1, 0.2), (0.0, 0.01, 0.02, 0.05))


def _index(threshold, cuts):
    return 0 if threshold is None else cuts.index(round(threshold, 2))


def _synthetic(net, *, held=lambda i, j: 0):
    """Outcome rows from a net contribution and legitimate holds by grid position."""
    review_cuts = [None, 0.99, 0.98, 0.95, 0.9, 0.8]
    decline_cuts = [None, 0.99, 0.98, 0.95]
    calls = []

    def evaluate(candidates):
        calls.append([c.version for c in candidates])
        out = []
        for c in candidates:
            i = _index(c.review_threshold, review_cuts)
            j = _index(c.decline_threshold, decline_cuts)
            out.append({**ROW, "net_cents": net(i, j), "reviews": i,
                        "legitimate_held": held(i, j)})
        return out
    return evaluate, calls


def _position(frontier, version):
    row = frontier.loc[frontier["policy_version"] == version].iloc[0]
    return GRID.review_rates.index(row["review_rate"]), GRID.decline_rates.index(
        row["decline_rate"])


def test_the_shortlist_is_fixed_by_the_frozen_screen_and_the_choice_by_policy_history():
    """Frozen history peaks at (2, 1); policy history (a policy that leans on repayment
    history) prefers one more review step. The shortlist is the frozen winner, its
    neighbours one step away on each threshold and the best k frozen points; only those
    are replayed with policy history, and the choice is the best of them."""
    frozen_eval, frozen_calls = _synthetic(lambda i, j: 100 - 10 * abs(i - 2) - 7 * abs(j - 1))
    policy_eval, policy_calls = _synthetic(lambda i, j: 100 - 10 * abs(i - 3) - 7 * abs(j - 1))
    scores = {"review": SCORES, "decline": SCORES}
    tuned = tune_shortlist(scored_policy(None, None), scores, frozen_eval, policy_eval, GRID,
                           k=5)
    frozen_rows = tuned.frontier.loc[tuned.frontier["history"] == "frozen"]
    policy_rows = tuned.frontier.loc[tuned.frontier["history"] == "policy"]
    assert len(frozen_rows) == 24 and len(frozen_calls) == 1 and len(policy_calls) == 1
    assert _position(frozen_rows, tuned.screened.version) == (2, 1)
    shortlisted = {_position(frozen_rows, v) for v in tuned.shortlist}
    neighbours = {(i, j) for i in (1, 2, 3) for j in (0, 1, 2)}
    assert neighbours <= shortlisted  # the winner and its eight neighbours
    ranked = frozen_rows.sort_values(["objective_cents", "reviews", "declines"],
                                     ascending=[False, True, True], kind="stable")
    best_five = {_position(frozen_rows, v) for v in ranked["policy_version"].head(5)}
    assert shortlisted == neighbours | best_five
    assert sorted(policy_calls[0]) == sorted(tuned.shortlist)
    assert set(policy_rows["policy_version"]) == set(tuned.shortlist)
    assert _position(policy_rows, tuned.chosen.version) == (3, 1)
    assert "frozen approve-all" in tuned.rule and "best 5" in tuned.rule


def test_a_winner_at_several_grid_positions_brings_every_positions_neighbours():
    """Four orders in a hundred tie at the top score, so every positive rate cuts at 9:
    the frozen winner (review and decline at 9) sits at review steps 1-5 and decline
    steps 1-3. Its neighbours at the grid's low corner (review off, decline at 9) are
    shortlisted, and policy history prefers one of them."""
    scores = np.array([9.0] * 4 + [1.0] * 96)

    def rows(net):
        def evaluate(candidates):
            return [{**ROW, "net_cents": net(c.review_threshold, c.decline_threshold),
                     "reviews": 0 if c.review_threshold is None else 1}
                    for c in candidates]
        return evaluate

    frozen_net = {(9.0, 9.0): 100, (None, 9.0): 50, (9.0, None): 40, (None, None): 30}
    policy_net = {(9.0, 9.0): 100, (None, 9.0): 2_000, (9.0, None): 40, (None, None): 30}
    tuned = tune_shortlist(scored_policy(None, None), {"review": scores, "decline": scores},
                           rows(lambda r, d: frozen_net[(r, d)]),
                           rows(lambda r, d: policy_net[(r, d)]),
                           Grid((0.0, 0.01, 0.02, 0.05, 0.1, 0.2), (0.0, 0.01, 0.02, 0.05)),
                           k=1)
    assert (tuned.screened.review_threshold, tuned.screened.decline_threshold) == (9.0, 9.0)
    assert len(tuned.shortlist) == 4  # every distinct pair neighbours some winner position
    assert (tuned.chosen.review_threshold, tuned.chosen.decline_threshold) == (None, 9.0)


def test_the_guardrail_keeps_points_with_too_much_friction_out_of_the_choice():
    """Holding legitimate customers earns more here, but past 30 per 10,000 the point is
    out: of the shortlist's best k and of the choice."""
    def held(i, j):
        return 10 * i  # 10 legitimate holds per review step, of 10,000

    frozen_eval, _ = _synthetic(lambda i, j: 10 * i - j, held=held)
    policy_eval, _ = _synthetic(lambda i, j: 10 * i - j, held=held)
    scores = {"review": SCORES, "decline": SCORES}
    cap = Guardrail(held_per_10000=30)
    tuned = tune_shortlist(scored_policy(None, None), scores, frozen_eval, policy_eval, GRID,
                           k=2, guardrail=cap)
    frontier = tuned.frontier
    assert (frontier["within_guardrail"] == (frontier["legitimate_held_per_10000"] <= 30)).all()
    assert _position(frontier, tuned.screened.version) == (3, 0)
    assert _position(frontier, tuned.chosen.version) == (3, 0)
    over = frontier.loc[~frontier["within_guardrail"]]
    shortlisted = set(tuned.shortlist)
    # beyond the winner's neighbours (review step 4), no point over the cap is replayed
    assert not {v for v in over["policy_version"]
                if _position(frontier, v)[0] > 4} & shortlisted
    unbounded = tune(scored_policy(None, None), scores, frozen_eval, GRID)
    assert _position(unbounded.frontier, unbounded.chosen.version) == (5, 0)


def test_the_guardrail_reads_its_caps_from_the_tuning_settings() -> None:
    assert Guardrail.from_config({"tuning": {}}) == Guardrail()
    caps = {"legitimate_held_per_10000": 40, "legitimate_declined_per_10000": None}
    assert Guardrail.from_config({"tuning": {"friction_guardrail": caps}}) == Guardrail(40.0)
    row = {**ROW, "legitimate_held": 41, "legitimate_declined": 500}
    assert Guardrail.from_config({"tuning": {"friction_guardrail": caps}}).rates(row) == (
        41.0, 500.0)
    assert not Guardrail(40.0).met(row) and Guardrail(41.0).met(row)
    assert not Guardrail(None, 499.0).met(row)


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
