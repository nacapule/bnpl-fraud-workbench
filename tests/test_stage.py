"""The tune and replay stages on the mini world, and the history a later order sees.

The first part runs the stages through a stand-in pipeline run object with the
stand-in context (``replay_support``). The second part rebuilds a later order's context
with the real as-of context (``core.asof``) after the policy declined or cancelled an
earlier order of the same account; it is skipped until that context is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from replay_support import ColumnScorer, StubContext, StubScorer, mini_tables

from core import actions, asof, ledger
from queue_sim import outcomes, stage
from queue_sim.replay import PolicyHistory, World
from rules import tuning

T = pd.Timestamp


@dataclass(frozen=True)
class Ref:
    seed: int
    family: str


class FakeRun:
    """What the stages read from the pipeline's run object."""

    def __init__(self, tables: dict[str, pd.DataFrame], context: pd.DataFrame,
                 scorer: type = StubScorer) -> None:
        self.ref = Ref(416, "baseline")
        self._tables = tables
        self.worlds = [self.ref]
        self.all_worlds = [self.ref]
        window = SimpleNamespace
        self.protocol = SimpleNamespace(
            windows={"fit": window(start=T("2024-12-01"), end=T("2025-03-01")),
                     "validation": window(start=T("2024-12-01"), end=T("2025-03-01")),
                     "test": window(start=T("2025-03-01"), end=T("2025-06-15"))},
            observed_until=T("2025-06-30 23:59:59"),
            order_start=T("2024-12-01"), order_end=T("2025-06-15"))
        scorers = {name: scorer(name=name) for name in ("rules", "tree", "logistic",
                                                        "boosting")}
        self.memory: dict[str, Any] = {"scorers": {416: scorers},
                                       "context": {self.ref: context}}

    def tables(self, ref: Ref) -> dict[str, pd.DataFrame]:
        assert ref == self.ref
        return self._tables


@pytest.fixture(scope="module")
def staged() -> tuple[FakeRun, stage.StageOutput, stage.StageOutput]:
    tables = mini_tables()
    scores = {order: order / 30 for order in tables["order_attempts"]["order_id"]}
    stub = StubContext(tables, scores=scores)
    patch = pytest.MonkeyPatch()
    patch.setattr(asof, "build_context", lambda t, d=None, **_: stub.build(t, d))
    patch.setattr(asof, "policy_rows",
                  lambda world_rows, *_, **__: world_rows.reset_index(drop=True),
                  raising=False)
    patch.setattr(tuning.Grid, "from_config",
                  classmethod(lambda cls, cfg=None: cls((0.0, 0.2), (0.0, 0.1))))
    try:
        run = FakeRun(tables, stub.build(tables))
        tuned = stage.tune(run)
        replayed = stage.replay(run)
        run.memory["frames"] = (stage.routing_frame(run, run.ref),
                                stage.review_decisions(run, run.ref))
    finally:
        patch.undo()
    return run, tuned, replayed


def test_tuning_keeps_a_chosen_policy_for_each_of_the_seven(staged) -> None:
    run, tuned, _ = staged
    found = run.memory["tuned"][416]
    assert set(found) == {"approve_all", "incumbent_rules", "tree_depth3", "logistic",
                          "boosting", "hybrid", "expected_loss"}
    chosen = pd.DataFrame(tuned.tables["tune.chosen"])
    assert len(chosen) == 6  # approve-all has nothing to tune
    frontier = pd.DataFrame(tuned.tables["tune.frontier"])
    assert set(frontier.groupby("policy").size()) == {2, 4}  # expected loss tunes review only
    assert frontier["feasible"].all()


def test_the_replay_stage_writes_one_integer_row_per_policy_and_variant(staged) -> None:
    _, _, replayed = staged
    rows = pd.DataFrame(replayed.tables["replay.outcomes"])
    assert len(rows) == 7 * len(stage.VARIANTS)  # one configured level on the mini world
    assert not rows.duplicated(["seed", "family", "policy", "capacity_level", "layout",
                                "history", "reviewer", "verification"]).any()
    for column in outcomes.OUTCOME_COLUMNS:
        values = rows[column]
        assert values.map(lambda v: isinstance(v, int | np.integer)).all(), column


def test_the_routing_frame_and_review_decisions_have_their_columns(staged) -> None:
    run, _, _ = staged
    alerts, decisions = run.memory["frames"]
    assert list(alerts.columns) == ["alert_id", "order_id", "user_id", "ts", "score", "band",
                                    "fired_rules", "policy", "policy_version"]
    assert set(alerts["band"]) <= {"review", "auto_decline"}
    assert (alerts["alert_id"] == [f"{o}:{v}" for o, v in
                                   zip(alerts["order_id"], alerts["policy_version"],
                                       strict=True)]).all()
    assert list(decisions.columns[:len(asof.KEY_COLUMNS)]) == list(asof.KEY_COLUMNS)
    assert {"taken_up_at", "decided_at", "checks", "checks_later", "disposition",
            "final"} <= set(decisions.columns)
    assert (decisions["policy"] == "incumbent_rules").all()
    assert (decisions["decision_at"] <= decisions["taken_up_at"]).all()
    assert (decisions["taken_up_at"] <= decisions["decided_at"]).all()
    window = run.protocol.windows["test"]
    for frame, column in ((alerts, "ts"), (decisions, "taken_up_at")):  # the test window
        assert frame[column].between(window.start, window.end).all()
    assert run.ref in run.memory["incumbent"]  # kept from the replay stage's own run
    assert len(alerts)  # (the tuned stand-in incumbent declines; it reviews nothing here)


def test_the_capacity_base_is_todays_queue_over_eighty_percent() -> None:
    from queue_sim.reviewer import service_seconds

    tables = mini_tables()
    # today's bands review at 30 and decline at 90: orders 3, 6 and 12 go to review in
    # the fit window, order 15 is declined, order 19 is after the window
    stub = StubContext(tables, scores={3: 40.0, 6: 31.0, 12: 30.0, 15: 95.0, 19: 50.0})
    run = FakeRun(tables, stub.build(tables))
    base = stage.capacity_base(run, [run.ref])
    offered = service_seconds(416, np.array([3, 6, 12]), 7.0, 0.6).sum() / 60
    assert base["offered_minutes"] == pytest.approx(offered)
    assert base["needed_minutes"] == pytest.approx(offered / 0.8)
    days = pd.date_range("2024-12-01", "2025-02-28")
    shifts = int((days.dayofweek <= 4).sum() + (days.dayofweek >= 2).sum())  # early, late
    assert base["shifts_per_world"] == shifts
    per_shift = base["minutes_per_shift"]
    assert per_shift["base"]["review_minutes_per_shift"] == {
        "early": int(np.ceil(offered / 0.8 / shifts)), "late": int(np.ceil(offered / 0.8 / shifts))}
    assert base["whole_analysts"]["base"]["analysts_per_shift"] == {"early": 1, "late": 1}
    assert not base["whole_analysts"]["low"]["binds"]  # one analyst per shift is far too many
    assert base["whole_analysts"]["low"]["minutes"] == shifts * 390


# ------------------------------------------------------------------ with the real context


def _real_context() -> bool:
    try:
        asof.build_context(mini_tables(), pd.DataFrame(
            {"order_id": pd.Series([1], dtype="int64"),
             "decision_at": pd.Series([T("2024-12-05 12:00")], dtype="datetime64[s]")}))
    except NotImplementedError:
        return False
    return True


needs_context = pytest.mark.skipif(not _real_context(),
                                   reason="the as-of context is not built yet")


@dataclass
class SoFar:
    fates_: pd.DataFrame
    blocks_: pd.DataFrame

    def changed_users(self) -> np.ndarray:
        changed = self.fates_["route"].ne("approve") | self.fates_["hold_at"].notna()
        return np.unique(self.fates_.loc[changed, "user_id"].to_numpy(np.int64))

    def fates(self, users: np.ndarray | None = None) -> pd.DataFrame:
        if users is None:
            return self.fates_
        return self.fates_.loc[self.fates_["user_id"].isin(users)].reset_index(drop=True)

    def blocks(self) -> pd.DataFrame:
        return self.blocks_


def _ana_order_10(change) -> tuple[pd.Series, pd.Series]:
    """Order 10 (Ana's second order) at checkout: approve-all and with order 1 changed."""
    tables = mini_tables()
    world = World(tables=tables, context=asof.build_context(tables), seed=416,
                  observed_until=T("2025-06-30 23:59:59"),
                  terms=ledger.ProductTerms.from_config())
    history = PolicyHistory(world, neighbours=asof.Neighbours.of(tables))
    at = T("2025-01-25")
    earlier = tables["order_attempts"].loc[tables["order_attempts"]["known_at"] < at]
    fates = actions.approve_all_fates(earlier).set_index("order_id")
    change(fates)
    so_far = SoFar(actions.typed_fates(fates.reset_index()),
                   pd.DataFrame({"user_id": pd.Series(dtype="int64"),
                                 "at": pd.Series(dtype="datetime64[s]")}))
    decision = pd.DataFrame({"order_id": [10], "decision_at": [T("2025-01-25 21:00")]})
    decision["decision_at"] = decision["decision_at"].astype("datetime64[s]")
    frozen = history.frozen.rows(decision, None, at).iloc[0]
    rebuilt = history.rows(decision, so_far, at)
    assert history.rebuilt.tolist() == [True]
    return frozen, rebuilt.iloc[0]


@needs_context
def test_a_declined_first_order_leaves_nothing_due_for_the_next() -> None:
    def decline(fates: pd.DataFrame) -> None:
        fates.loc[1, "route"] = "auto_decline"

    frozen, rebuilt = _ana_order_10(decline)
    assert frozen["installments_due_user"] == 3 and frozen["approved_orders_user_ever"] == 1
    assert rebuilt["installments_due_user"] == 0
    assert rebuilt["approved_orders_user_ever"] == 0


@needs_context
def test_a_hold_cancelled_before_shipment_leaves_nothing_due_for_the_next() -> None:
    def cancel(fates: pd.DataFrame) -> None:
        held, ended = T("2024-12-05 12:10"), T("2024-12-07 12:10")
        fates.loc[1, ["route", "hold_at", "hold_before_shipment", "hold_outcome",
                      "hold_ended_at", "void_at", "void_cause"]] = [
            "review", held, True, "cancelled", ended, ended, "hold_cancelled"]

    frozen, rebuilt = _ana_order_10(cancel)
    assert frozen["installments_due_user"] == 3
    assert rebuilt["installments_due_user"] == 0
    assert rebuilt["approved_orders_user_ever"] == 0


@needs_context
def test_a_released_hold_counts_the_order_from_its_release() -> None:
    def release(fates: pd.DataFrame) -> None:
        held, released = T("2024-12-05 12:10"), T("2024-12-06 18:00")
        fates.loc[1, ["route", "hold_at", "hold_before_shipment", "hold_outcome",
                      "hold_ended_at", "released_at"]] = [
            "review", held, True, "cleared", released, released]

    frozen, rebuilt = _ana_order_10(release)
    assert rebuilt["approved_orders_user_ever"] == frozen["approved_orders_user_ever"] == 1
    assert rebuilt["installments_due_user"] == frozen["installments_due_user"] == 3


@needs_context
def test_worker_processes_give_the_same_rows_as_one_process() -> None:
    tables = mini_tables()
    context = asof.build_context(tables)
    patch = pytest.MonkeyPatch()
    patch.setattr(tuning.Grid, "from_config",
                  classmethod(lambda cls, cfg=None: cls((0.0, 0.2), (0.0, 0.1))))
    outputs = []
    try:
        for count in (1, 2):
            run = FakeRun(tables, context, scorer=ColumnScorer)
            run.workers = count
            tuned = stage.tune(run)
            replayed = stage.replay(run)
            outputs.append((tuned.tables, replayed.tables))
    finally:
        patch.undo()
    (tuned_one, replayed_one), (tuned_two, replayed_two) = outputs
    assert tuned_one == tuned_two
    assert replayed_one == replayed_two
    assert len(replayed_one["replay.outcomes"]) == 7 * len(stage.VARIANTS)
