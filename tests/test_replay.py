"""The replay's mechanics on the mini world: routing, the queue, the reviewer's actions,
blocks, capacity, the SLA clock, the backlog and the history each policy sees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from replay_support import (
    CALENDAR,
    CARD,
    SETTINGS,
    StubContext,
    always_on_roster,
    frozen,
    mini_tables,
    scored_policy,
    stub_world,
    verification,
)

from core import asof
from core.actions import CheckoutRoute
from queue_sim import outcomes
from queue_sim.replay import PolicyHistory, Settings, replay
from queue_sim.reviewer import Reviewer
from queue_sim.roster import Roster, Shift

T = pd.Timestamp
WHOLE = (T("2024-12-01"), T("2025-06-15"))
FAIL_ID_CHECK = {"third_party": {"contact": {"passed": 1.0, "failed": 0.0},
                                 "id_check": {"passed": 0.0, "failed": 1.0}}}
SETTLED = {"unauthorized_disputes_lost_user": 1}  # FP-2 §6.3(a): row (a), decline at once


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return mini_tables()


def run(tables, *, overrides=None, scores=None, review=1.0, decline=None, roster=None,
        checks=None, classes=None, window=WHOLE, settings=SETTINGS, history=None,
        linked=None, median_hours=2.0):
    stub = StubContext(tables, overrides=overrides or {}, scores=scores or {})
    world = stub_world(tables, stub)
    result = replay(
        world, scored_policy(review, decline), window=window,
        roster=roster or always_on_roster(), calendar=CALENDAR, reviewer=Reviewer(),
        verification=verification(checks, classes=classes, median_hours=median_hours),
        history=history(world, stub) if history else frozen(world, stub),
        settings=settings, linked_accounts=linked)
    return result, world


def row_of(result, world):
    return outcomes.outcome_row(result, world, keys={}, ltv_cents=1500)


def fate(result, order):
    return result.fates.set_index("order_id").loc[order]


# ------------------------------------------------------------------ edge cases (C2-16)


def test_an_empty_window_replays_to_empty_results(tables) -> None:
    result, world = run(tables, window=(T("2025-04-01"), T("2025-05-01")))
    assert result.fates.empty and result.reviews.empty and result.log.empty
    row = row_of(result, world)
    assert row["orders"] == 0 and row["net_vs_approve_all_cents"] == 0
    assert row["max_backlog"] == 0 and row["wait_p90_minutes"] == 0


def test_quiet_orders_reviewed_by_policy_are_cleared_and_change_nothing(tables) -> None:
    """Every order reviewed, none with adverse evidence: all cleared, no friction, no cash."""
    scores = {order: 5.0 for order in tables["order_attempts"]["order_id"]}
    result, world = run(tables, scores=scores)
    assert set(result.reviews["final"]) == {"clear"}
    row = row_of(result, world)
    assert row["reviews"] == row["reviews_decided"] == 24
    assert row["net_vs_approve_all_cents"] == 0 and row["holds"] == 0
    assert row["legitimate_declined"] == row["friction_cost_cents"] == 0


def test_an_overnight_shift_works_across_midnight(tables) -> None:
    # one analyst 22:00-06:00 every night; 30-minute reviews
    night = Roster((Shift("night", tuple(range(7)), 22 * 60, 8 * 60),), {"night": 1})
    settings = Settings(48.0, 30.0, 0.0, 20.0)
    result, _ = run(tables, scores={6: 5.0, 12: 5.0}, roster=night, settings=settings)
    reviews = result.reviews.set_index("order_id")
    # order 12 at 02:20 is reviewed on shift at once; order 6 at 11:00 waits for 22:00
    assert reviews.loc[12, "started_at"] == T("2025-02-03 02:20")
    assert reviews.loc[12, "decided_at"] == T("2025-02-03 02:50")
    assert reviews.loc[6, "started_at"] == T("2025-01-10 22:00")


def test_a_review_reaching_the_end_of_a_shift_resumes_on_the_next(tables) -> None:
    short = Roster((Shift("day", tuple(range(7)), 13 * 60 + 55, 60),), {"day": 1})
    result, _ = run(tables, scores={3: 5.0}, roster=short)  # order 3 at 13:00, 10 minutes
    review = result.reviews.set_index("order_id").loc[3]
    assert review["started_at"] == T("2024-12-20 13:55")
    assert review["decided_at"] == T("2024-12-20 14:05")
    late = Roster((Shift("day", tuple(range(7)), 13 * 60 + 55, 5),), {"day": 1})
    result, _ = run(tables, scores={3: 5.0}, roster=late)  # five minutes a day
    review = result.reviews.set_index("order_id").loc[3]
    assert review["started_at"] == T("2024-12-20 13:55")
    assert review["decided_at"] == T("2024-12-21 14:00")  # 5 min on the 20th, 5 on the 21st


# ------------------------------------------------------------------ capacity and the queue


def test_analysts_never_work_beyond_their_shifts_or_two_reviews_at_once(tables) -> None:
    shifts = Roster((Shift("a", (0, 2, 4), 9 * 60, 120), Shift("b", (1, 3, 5, 6), 15 * 60, 90)),
                    {"a": 1, "b": 2})
    settings = Settings(48.0, 45.0, 0.5, 20.0)
    scores = {order: float(order) for order in tables["order_attempts"]["order_id"]}
    result, world = run(tables, scores=scores, roster=shifts, settings=settings)
    reviews = result.reviews.dropna(subset=["started_at"])
    start, end = WHOLE[0], reviews["decided_at"].max() + pd.Timedelta(days=1)
    t0, t1 = int(start.timestamp()), int(end.timestamp())
    for name, shift in shifts.analysts:
        mine = reviews.loc[reviews["analyst"] == name].sort_values("started_at")
        windows = shifts.windows(shift, t0, t1)
        previous_end = None
        for review in mine.itertuples():
            a, b = int(review.started_at.timestamp()), int(review.decided_at.timestamp())
            assert previous_end is None or a >= previous_end  # one review at a time
            previous_end = b
            worked = sum(max(0, min(b, we) - max(a, ws)) for ws, we in windows)
            assert worked == review.service_seconds  # all of it on shift
    row = row_of(result, world)
    assert row["review_minutes_used"] <= row["available_minutes"]


def test_backlog_is_counted_at_each_event_in_time_order(tables) -> None:
    # orders 23, 24 and 25 arrive a day apart; one analyst who works 09:00-09:30 only,
    # 20-minute reviews: the queue builds up and drains
    roster = Roster((Shift("am", tuple(range(7)), 9 * 60, 30),), {"am": 1})
    settings = Settings(48.0, 20.0, 0.0, 20.0)
    result, world = run(tables, scores={23: 5.0, 24: 5.0, 25: 5.0, 26: 5.0}, roster=roster,
                        settings=settings)
    log = result.log
    assert log["at"].is_monotonic_increasing
    enters = (log["kind"] == "enter").cumsum()
    starts = (log["kind"] == "start").cumsum()
    assert (log["backlog"] == enters - starts).all()
    assert log.loc[log["kind"] == "enter", "at"].tolist() == [
        T("2025-03-10 20:20"), T("2025-03-11 20:50"), T("2025-03-12 21:20"),
        T("2025-06-10 10:00")]
    assert row_of(result, world)["max_backlog"] == 1


def test_priority_beats_score_and_arrival(tables) -> None:
    # orders 23-25 wait overnight; 25 matches R05 (P0), 24 is over $500? no: 23 has R02 (P1)
    roster = Roster((Shift("am", tuple(range(7)), 9 * 60, 8 * 60),), {"am": 1})
    overrides = {25: {"attempts_user_24h": 9}, 23: {"accounts_on_device_30d": 3}}
    window = (T("2025-03-12 00:00"), T("2025-03-13 00:00"))
    result, _ = run(tables, scores={25: 1.0, 24: 9.0}, overrides=overrides, roster=roster,
                    window=window)
    assert result.reviews.set_index("order_id").loc[25, "priority"] == "P0"
    window = (T("2025-03-10 00:00"), T("2025-03-13 00:00"))
    result, _ = run(tables, scores={23: 2.0, 24: 9.0, 25: 1.0}, overrides=overrides,
                    roster=roster, window=window)
    reviews = result.reviews.set_index("order_id")
    assert reviews.loc[[23, 24, 25], "priority"].tolist() == ["P1", "P2", "P0"]
    order = reviews.sort_values("started_at").index.tolist()
    assert order == [23, 24, 25] or order.index(25) < order.index(24)


def test_utilization_counts_every_analyst_and_coverage_counts_hours_once(tables) -> None:
    roster = Roster((Shift("day", tuple(range(7)), 9 * 60, 8 * 60),), {"day": 3})
    result, world = run(tables, scores={3: 5.0}, roster=roster)
    assert result.available_minutes == pytest.approx(3 * result.coverage_minutes)


def test_the_sla_clock_does_not_move_with_the_roster(tables) -> None:
    scores = {order: 5.0 for order in (3, 6, 12, 19)}
    early = Roster((Shift("early", tuple(range(7)), 6 * 60, 4 * 60),), {"early": 1})
    late = Roster((Shift("late", tuple(range(7)), 18 * 60, 4 * 60),), {"late": 1})
    for roster in (early, late):
        result, _ = run(tables, scores=scores, roster=roster)
        r = result.reviews
        expected = [CALENDAR.service_hours(int(a.timestamp()), int(b.timestamp()))
                    for a, b in zip(r["entered_at"], r["decided_at"], strict=True)]
        assert np.allclose(r["service_hours_to_decision"], expected)
    # a decision outside service hours the next morning counts only service hours
    assert CALENDAR.service_hours(int(T("2025-03-01 19:00").timestamp()),
                                  int(T("2025-03-02 09:00").timestamp())) == pytest.approx(2.0)


# ------------------------------------------------------------------ actions and their effects


def test_fraud_stopped_before_shipping_nets_zero(tables) -> None:
    """Order 18 (stolen card, ships 09:40): held at review, id_check fails, voided."""
    result, world = run(tables, scores={18: 5.0}, overrides={18: CARD}, checks=FAIL_ID_CHECK,
                        classes={18: "third_party"})
    f = fate(result, 18)
    assert f["hold_before_shipment"] and f["void_cause"] == "decline"
    assert f["void_at"] < T("2025-02-14 09:40")
    row = row_of(result, world)
    assert row["prevented_loss_cents"] == 53_500  # the world's loss on order 18
    assert row["fraud_stopped_before_shipping"] == 1
    assert row["accounts_blocked"] == 1


def test_one_minute_after_shipping_the_loss_stands_and_the_next_order_is_declined(
        tables) -> None:
    """Fay's order 11 ships 2025-02-02 06:30; the analyst decides at 06:31 (row (a))."""
    roster = Roster((Shift("dawn", tuple(range(7)), 6 * 60 + 30, 60),), {"dawn": 1})
    settings = Settings(48.0, 1.0, 0.0, 20.0)
    result, world = run(tables, scores={11: 5.0}, overrides={11: SETTLED}, roster=roster,
                        settings=settings)
    review = result.reviews.set_index("order_id").loc[11]
    assert review["decided_at"] == T("2025-02-02 06:31")
    assert review["shipped_at_decision"] and review["final"] == "decline"
    assert pd.isna(fate(result, 11)["void_at"])  # the order's cash is the world's
    assert fate(result, 13)["route"] == "blocked"  # her next order, 02-04
    row = row_of(result, world)
    assert row["fraud_declined_after_shipping"] == 1 and row["decided_after_shipping"] == 1
    world_cash = world.tables["cash_events"]
    net_13 = int(world_cash.loc[world_cash["order_id"] == 13, "amount_cents"].sum())
    assert row["prevented_loss_cents"] == -net_13  # only the blocked order is prevented


def test_a_legitimate_customer_who_verifies_late_is_released_and_ships_later(tables) -> None:
    """Order 1 (ana, legitimate) held for R03; she answers after about 30 hours."""
    result, world = run(tables, scores={1: 5.0}, overrides={1: CARD}, median_hours=30.0)
    f = fate(result, 1)
    assert f["hold_outcome"] == "cleared" and f["hold_before_shipment"]
    assert f["released_at"] - f["hold_at"] == pd.Timedelta(hours=30)
    row = row_of(result, world)
    assert row["legitimate_held"] == 1 and row["legitimate_cancelled"] == 0
    assert row["net_vs_approve_all_cents"] == 0  # the same cash, later


def test_a_legitimate_customer_who_never_answers_is_cancelled_without_a_block(tables) -> None:
    silent = {"legitimate": {"contact": {"passed": 0.0, "failed": 0.0},
                             "id_check": {"passed": 0.0, "failed": 0.0}}}
    result, world = run(tables, scores={1: 5.0}, overrides={1: CARD}, checks=silent)
    f = fate(result, 1)
    assert f["hold_outcome"] == "cancelled" and f["void_cause"] == "hold_cancelled"
    assert f["void_at"] - f["hold_at"] == pd.Timedelta(hours=48)
    assert result.blocks.empty and fate(result, 10)["route"] == "approve"  # her next order
    row = row_of(result, world)
    assert row["legitimate_cancelled"] == 1 and row["friction_cost_cents"] == 1500
    world_cash = world.tables["cash_events"]
    assert row["net_vs_approve_all_cents"] == -int(
        world_cash.loc[world_cash["order_id"] == 1, "amount_cents"].sum())


def test_escalation_blocks_the_linked_accounts(tables) -> None:
    """Order 23 shares device 16 with orders 24 and 25 (pf2, pf3): escalated, both blocked."""
    def linked(world_tables, decisions):
        assert decisions["order_id"].tolist() == [23]
        return pd.DataFrame({"order_id": [23, 23], "decision_at": decisions["decision_at"][0],
                             "user_id": [17, 18]})

    shared = {"accounts_on_device_30d": 3, "device_link_age_hours": 10.0}  # not a household
    result, world = run(tables, scores={23: 5.0}, overrides={23: shared},
                        checks=FAIL_ID_CHECK, classes={23: "third_party"}, linked=linked)
    assert result.reviews.set_index("order_id").loc[23, "final"] == "escalate"
    assert set(result.blocks["user_id"]) == {16, 17, 18}
    assert fate(result, 24)["route"] == fate(result, 25)["route"] == "blocked"
    assert row_of(result, world)["escalations"] == 1


def test_an_auto_decline_removes_the_order_and_blocks_nobody(tables) -> None:
    result, world = run(tables, scores={1: 9.0}, review=5.0, decline=8.0)
    assert fate(result, 1)["route"] == CheckoutRoute.AUTO_DECLINE.value
    assert result.blocks.empty and result.reviews.empty
    row = row_of(result, world)
    assert row["legitimate_declined_checkout"] == 1 and row["friction_cost_cents"] == 1500
    assert row["net_vs_approve_all_cents"] == -600  # ana's $120 order earned the 5% fee


def test_the_decline_threshold_changes_cash(tables) -> None:
    """A decline band is an action with a ledger effect, not a label on a caught order."""
    scores = {18: 9.0, 1: 9.0}
    off, world = run(tables, scores=scores, review=None, decline=None)
    on, _ = run(tables, scores=scores, review=None, decline=8.0)
    a, b = row_of(off, world), row_of(on, world)
    assert b["prevented_loss_cents"] - a["prevented_loss_cents"] == 53_500
    assert b["legitimate_declined_checkout"] == 1 and b["friction_cost_cents"] == 1500
    assert b["net_vs_approve_all_cents"] == 53_500 - 600


# ------------------------------------------------------------------ what the reviewer reads


TRUTH_KEYS = {"label", "basis", "label_known_at", "pattern_id", "intent", "actor", "episode_id",
              "mimic", "profile"}


def test_the_reviewer_reads_context_and_checks_never_truth(tables) -> None:
    seen = []

    class Watching(Reviewer):
        def decide(self, order_id, row, checks):
            seen.append(set(row))
            return super().decide(order_id, row, checks)

    stub = StubContext(tables, overrides={18: CARD}, scores={18: 5.0})
    world = stub_world(tables, stub)
    firsts = []
    for classes in ({18: "third_party"}, {18: "legitimate"}):
        result = replay(world, scored_policy(1.0), window=WHOLE, roster=always_on_roster(),
                        calendar=CALENDAR, reviewer=Watching(),
                        verification=verification(FAIL_ID_CHECK, classes=classes),
                        history=frozen(world, stub), settings=SETTINGS)
        firsts.append(result.reviews.set_index("order_id").loc[18, "first_disposition"])
    assert seen and all(not (keys & TRUTH_KEYS) for keys in seen)
    assert firsts == ["hold", "hold"]  # truth changed only the check's answer


# ------------------------------------------------------------------ policy-specific history


class Spy:
    """Stands in for core.asof.policy_rows: records its inputs, returns the frozen rows."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, world_rows, realized, state, decisions, **kwargs):
        self.calls.append((realized, state, decisions.copy()))
        return world_rows.reset_index(drop=True)


def _policy_history(spy):
    return lambda world, stub: PolicyHistory(world, policy_rows=spy,
                                             frozen=frozen(world, stub))


def test_a_declined_first_order_leaves_no_history_for_the_next(tables) -> None:
    """Ana's order 1 auto-declined: when her order 10 is scored, the realized tables the
    context is rebuilt from hold no plan, schedule or payment of order 1."""
    spy = Spy()
    run(tables, scores={1: 9.0}, review=None, decline=8.0, history=_policy_history(spy))
    asked = [(r, s, d) for r, s, d in spy.calls if 10 in set(d["order_id"])]
    assert len(asked) == 1
    realized, state, _ = asked[0]
    assert 1 not in set(realized["plans"]["order_id"])
    assert 1 not in set(realized["installment_schedule"]["plan_id"])
    assert 1 not in set(realized["payment_attempts"]["plan_id"])
    assert 1 not in set(state.approved["order_id"])
    assert 10 not in set(realized["plans"]["order_id"])  # not yet decided when scored


def test_a_held_then_cancelled_order_leaves_no_history_for_the_next(tables) -> None:
    spy = Spy()
    silent = {"legitimate": {"contact": {"passed": 0.0, "failed": 0.0},
                             "id_check": {"passed": 0.0, "failed": 0.0}}}
    run(tables, scores={1: 5.0}, overrides={1: CARD}, checks=silent,
        history=_policy_history(spy))
    realized, state, _ = next((r, s, d) for r, s, d in spy.calls
                              if 10 in set(d["order_id"]))
    # order 1's plan is plan 1: only the checkout payment and installment remain
    assert realized["payment_attempts"].query("plan_id == 1")["seq"].tolist() == [0]
    assert realized["installment_schedule"].query("plan_id == 1")["seq"].tolist() == [0]
    assert set(state.voided["order_id"]) == {1} and 1 not in set(state.approved["order_id"])
    refunds = realized["cash_events"].query("order_id == 1 and kind == 'refund'")
    assert refunds["amount_cents"].tolist() == [-3000]


def test_a_hold_moves_the_repayment_schedule_the_next_order_sees(tables) -> None:
    spy = Spy()
    run(tables, scores={1: 5.0}, overrides={1: CARD}, median_hours=30.0,
        history=_policy_history(spy))
    realized, state, _ = next((r, s, d) for r, s, d in spy.calls
                              if 10 in set(d["order_id"]))
    due = realized["installment_schedule"].query("plan_id == 1").set_index("seq")["due_at"]
    released = state.held.set_index("order_id").loc[1, "released_at"]
    assert due[1] == released + pd.Timedelta(days=14)  # was checkout + 14 days
    assert due[0] == T("2024-12-05 12:00")
    assert state.approved.set_index("order_id").loc[1, "approved_at"] == released


def test_untouched_policies_use_the_approve_all_rows(tables) -> None:
    spy = Spy()
    scores = {order: 5.0 for order in tables["order_attempts"]["order_id"]}
    run(tables, scores=scores, history=_policy_history(spy))  # every review clears
    assert spy.calls == []


def test_outcome_columns_are_the_ones_rebuilt() -> None:
    assert {"installments_due_user", "approved_orders_user_ever",
            "never_pay_determined_user"} <= set(asof.OUTCOME_COLUMNS)
