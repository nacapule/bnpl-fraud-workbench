"""The pipeline's tune and replay stages, and the frames other stages keep from them.

The pipeline (``pipeline.py``) passes its run object, which gives each world's tables
(``run.tables(ref)``), the fitted scorers per seed (``run.memory["scorers"]``), the
protocol (``run.protocol``) and the worlds to evaluate (``run.worlds``). The world-level
context is shared through ``run.memory["context"]`` as the pipeline caches it.

* :func:`tune`: per seed, on the baseline world's validation window at the base
  capacity, each policy's thresholds (``rules.tuning``); kept in
  ``run.memory["tuned"]``.
* :func:`replay`: per evaluation world, every tuned policy and approve-all on the test
  window at each capacity level and layout, plus three variants at the base level
  (approve-all history frozen, the perfect reviewer, weaker verification): one row of
  integer outcomes each (``queue_sim.outcomes``), and per-world tables (the reviewer's
  confusion matrix, prevented loss by pattern).
* :func:`review_decisions`: the incumbent's review decisions at base capacity on one
  world's test window, with the context rows and checks behind them, for case and
  packet selection (kept from the replay stage's main run of the incumbent).
* :func:`routing_frame`: the incumbent's routing at checkout on one world's test window
  (the MySQL ``alerts`` table), from the same run.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib.externals import cloudpickle

from core import asof, config, evidence, ledger
from core.actions import CheckoutRoute
from queue_sim import outcomes, policies
from queue_sim.replay import FrozenHistory, PolicyHistory, ReplayResult, Settings, World
from queue_sim.replay import replay as run_replay
from queue_sim.reviewer import PerfectReviewer, Reviewer, Verification, service_seconds
from queue_sim.roster import Roster, ServiceCalendar, Shift, to_seconds
from rules import tuning

BASELINE = "baseline"
INCUMBENT = "incumbent_rules"


@dataclass
class StageOutput:
    """The fields the pipeline's stage outputs carry."""

    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    inputs: dict[str, Path] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class Staffing:
    """One staffing variant: a capacity level on a shift layout."""

    level: str
    layout: str
    analysts_per_shift: Mapping[str, int]
    review_minutes_per_shift: Mapping[str, int] | None = None  # None: the roster's setting

    @classmethod
    def from_config(cls, level: str, layout: str, entry: Mapping[str, Any]) -> Staffing:
        """One ``capacity.levels`` or ``capacity.redesigned`` entry."""
        minutes = entry.get("review_minutes_per_shift")
        return cls(level, layout, dict(entry["analysts_per_shift"]),
                   None if minutes is None else dict(minutes))

    def roster(self, policy_cfg: Mapping[str, Any]) -> Roster:
        return Roster.from_config(policy_cfg, layout=self.layout,
                                  analysts_per_shift=self.analysts_per_shift,
                                  review_minutes_per_shift=self.review_minutes_per_shift)


def staffing(policy_cfg: Mapping[str, Any] | None = None) -> list[Staffing]:
    """Capacity levels on the current layout, then the redesigned layout, if configured.

    Before the capacity base is fixed only the configured roster exists ("configured").
    """
    policy_cfg = config.load("policy") if policy_cfg is None else policy_cfg
    current = policy_cfg["roster"]["layout"]
    levels = policy_cfg["capacity"].get("levels") or {}
    variants = [Staffing.from_config(level, current, entry) for level, entry in levels.items()
                if entry is not None]
    if not variants:
        variants = [Staffing("configured", current, policy_cfg["roster"]["analysts_per_shift"])]
    redesigned = policy_cfg["capacity"].get("redesigned")
    if redesigned:
        variants.append(Staffing.from_config("base", redesigned["layout"], redesigned))
    return variants


def base_staffing(policy_cfg: Mapping[str, Any]) -> Staffing:
    variants = staffing(policy_cfg)
    current = policy_cfg["roster"]["layout"]
    for item in variants:
        if item.level in ("base", "configured") and item.layout == current:
            return item
    raise ValueError("no base staffing on the current layout")


STAFFED_FOR = 0.8  # the base is staffed so today's queue uses 80% of its minutes


def capacity_base(run: Any, refs: Sequence[Any]) -> dict[str, Any]:
    """The capacity base from development worlds, fixed before any policy comparison.

    Today's queue is the fraud-queue minutes today's rules at today's bands
    (``config/policy.yaml`` ``rules.bands``) offer over each world's fit window, replayed
    with policy history at the configured staffing (``roster.analysts_per_shift``, whole
    productive shifts, far more than the queue needs): review time and escalations'
    senior review. The minutes checkout routing alone would send are reported beside
    them. :func:`capacity_levels` sizes the levels on the current layout from today's
    queue.
    """
    policy_cfg = config.load("policy")
    bands = policy_cfg["rules"]["bands"]
    window = _window(run.protocol, "fit")
    layout = policy_cfg["roster"]["layout"]
    shifts = Roster.from_config(policy_cfg, layout=layout, analysts_per_shift={
        item["name"]: 1 for item in policy_cfg["roster"]["layouts"][layout]},
        review_minutes_per_shift={}).shifts
    configured = Staffing("configured", layout, policy_cfg["roster"]["analysts_per_shift"])
    per_world = []
    for ref in refs:
        bench = _bench(run, ref)
        today = policies_for(run.memory["scorers"][ref.seed], bench)[INCUMBENT].with_thresholds(
            float(bands["review"]), float(bands["decline"]))
        result = bench.run(today, window, configured, history="policy")
        reviews = result.reviews
        senior = float(reviews["senior_minutes"].sum())
        per_world.append({"seed": ref.seed, "family": ref.family,
                          "orders": len(bench.world.orders(*window)),
                          "offered_minutes": float(reviews["service_seconds"].sum()) / 60
                          + senior,
                          "senior_minutes": senior, "reviews": len(reviews),
                          "routed_minutes": bench.routed_minutes(today, window),
                          "configured_minutes": result.available_minutes})
    levels = capacity_levels(shifts, tuple(int(to_seconds(t)) for t in window),
                             [w["offered_minutes"] for w in per_world])
    return {"window": [str(window[0]), str(window[1])], "bands": dict(bands),
            "layout": layout, "worlds": per_world, **levels}


def capacity_levels(shifts: Sequence[Shift], window: tuple[int, int],
                    offered: Sequence[float]) -> dict[str, Any]:
    """Low, base and high staffing for today's queue (``offered`` minutes per world).

    The base must provide today's minutes per world, averaged over the worlds, divided
    by :data:`STAFFED_FOR`, over ``window`` (seconds); it is the smallest staffing whose
    available minutes (shifts clipped to the window) reach that target; low and high are
    the smallest reaching half and 1.5 times the base's minutes. Two readings:

    * ``whole_analysts``: the same number of analysts on every shift, at least one;
    * ``minutes_per_shift``: the fewest analysts per shift (at least one) and then the
      fewest whole review minutes per shift, the same on every shift, that reach the
      target (more analysts only when the productive shift is not enough).

    Each level reports its staffing, its minutes per world, its target, and ``binds``:
    today's minutes exceed the level's on every world. With no review minutes at all the
    levels are the minimum staffing (one analyst, one minute).
    """
    t0, t1 = window
    names = [shift.name for shift in shifts]
    productive = min(shift.productive_minutes for shift in shifts)
    average = sum(offered) / max(len(offered), 1)
    needed = average / STAFFED_FOR

    def roster(n: int, m: int | None = None) -> Roster:
        return Roster(tuple(shifts), {name: n for name in names},
                      None if m is None else {name: m for name in names})

    def minutes(r: Roster) -> float:
        return r.available_minutes(t0, t1)

    one = minutes(roster(1))
    if one <= 0:
        raise ValueError("no shift falls in the window")

    def whole(target: float) -> Roster:
        n = max(1, int(target // one))
        while minutes(roster(n)) < target:
            n += 1
        return roster(n)

    def budget(target: float) -> Roster:
        n = max(1, int(target // one))
        while minutes(roster(n, productive)) < target:  # the productive shift is not enough
            n += 1
        lo, hi = 1, productive  # the fewest whole minutes per shift that reach the target
        while lo < hi:
            mid = (lo + hi) // 2
            if minutes(roster(n, mid)) >= target:
                hi = mid
            else:
                lo = mid + 1
        return roster(n, lo)

    def level(r: Roster, target: float) -> dict[str, Any]:
        out = {"analysts_per_shift": dict(r.analysts_per_shift), "minutes": minutes(r),
               "target_minutes": target,
               "binds": bool(offered) and all(o > minutes(r) for o in offered)}
        if r.review_minutes_per_shift is not None:
            out["review_minutes_per_shift"] = dict(r.review_minutes_per_shift)
        return out

    readings = {}
    for name, size in (("whole_analysts", whole), ("minutes_per_shift", budget)):
        base = size(needed)
        readings[name] = {
            "low": level(size(minutes(base) / 2), minutes(base) / 2),
            "base": level(base, needed),
            "high": level(size(1.5 * minutes(base)), 1.5 * minutes(base)),
        }
    return {"staffed_for": STAFFED_FOR, "offered_minutes_per_world": average,
            "needed_minutes_per_world": needed,
            "shifts_per_world": sum(len(roster(1).windows(s, t0, t1)) for s in shifts),
            "minutes_one_analyst_per_shift": one, **readings}


# ------------------------------------------------------------------------- one world


@dataclass
class Bench:
    """Everything a replay of one world needs, built once per world."""

    world: World
    latent_orders: pd.DataFrame
    neighbours: Any
    policy_cfg: Mapping[str, Any]
    frozen: FrozenHistory

    @classmethod
    def of(cls, tables: Mapping[str, pd.DataFrame], context: pd.DataFrame, *, seed: int,
           observed_until: pd.Timestamp, policy_cfg: Mapping[str, Any] | None = None) -> Bench:
        policy_cfg = config.load("policy") if policy_cfg is None else policy_cfg
        world = World(tables=tables, context=context, seed=seed,
                      observed_until=pd.Timestamp(observed_until),
                      terms=ledger.ProductTerms.from_config())
        neighbours = asof.Neighbours.of(tables) if hasattr(asof, "Neighbours") else None
        return cls(world, tables["latent_orders"], neighbours, policy_cfg,
                   FrozenHistory(world, neighbours=neighbours))

    def verification(self, rates: str = "verification") -> Verification:
        cache = self.__dict__.setdefault("_verification", {})
        if rates not in cache:
            cache[rates] = Verification.from_config(self.world.seed, self.latent_orders,
                                                    self.policy_cfg, rates=rates)
        return cache[rates]

    def run(self, policy: policies.Policy, window: tuple[pd.Timestamp, pd.Timestamp],
            staff: Staffing, *, history: str = "policy", reviewer: str = "evidence",
            rates: str = "verification") -> ReplayResult:
        chosen = PolicyHistory(self.world, neighbours=self.neighbours, frozen=self.frozen) \
            if history == "policy" else self.frozen
        judge = (PerfectReviewer.from_latent(self.latent_orders) if reviewer == "perfect"
                 else Reviewer())
        return run_replay(
            self.world, policy, window=window, roster=staff.roster(self.policy_cfg),
            calendar=ServiceCalendar.from_config(self.policy_cfg), reviewer=judge,
            verification=self.verification(rates), history=chosen,
            settings=Settings.from_config(self.policy_cfg))

    def ltv_cents(self) -> int:
        return int(round(float(self.policy_cfg["costs"]["false_decline_ltv_usd"]) * 100))

    def routed_minutes(self, policy: policies.Policy,
                       window: tuple[pd.Timestamp, pd.Timestamp]) -> float:
        """Review minutes the policy's checkout routing sends to the queue over ``window``
        on the world-level context (before any block), with each order's review time.
        Escalations' senior minutes are not here: they follow decisions (the replay)."""
        orders = self.world.orders(*window)
        rows = self.world.context.set_index("order_id").reindex(orders["order_id"])
        routed = policy.route(rows.reset_index(), known=self.world.scores)
        ids = routed.loc[routed["route"] == CheckoutRoute.REVIEW.value, "order_id"]
        settings = Settings.from_config(self.policy_cfg)
        seconds = service_seconds(self.world.seed, ids.to_numpy(np.int64),
                                  settings.service_mean_minutes, settings.service_sigma)
        return float(seconds.sum()) / 60

    def checkout_scores(self, policy: policies.Policy,
                        window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, np.ndarray]:
        orders = self.world.orders(*window)
        rows = self.world.context.set_index("order_id").reindex(orders["order_id"])
        rows = rows.reset_index()
        return {"review": policy.review(rows) if policy.review is not None else np.array([]),
                "decline": policy.decline(rows) if policy.decline is not None
                else np.array([])}


def _window(protocol: Any, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    window = protocol.windows[name]
    return window.start, window.end


def _bench(run: Any, ref: Any, observed_until: pd.Timestamp | None = None) -> Bench:
    """One world's bench, observed until ``observed_until`` (default: the end of
    observation)."""
    cut = pd.Timestamp(run.protocol.observed_until if observed_until is None
                       else observed_until)
    cache = run.memory.setdefault("benches", {})
    key = ref if cut == pd.Timestamp(run.protocol.observed_until) else (ref, cut)
    if key not in cache:
        tables = run.tables(ref)
        contexts = run.memory.setdefault("context", {})
        if ref not in contexts:
            contexts[ref] = asof.build_context(tables)
        cache[key] = Bench.of(tables, contexts[ref], seed=ref.seed, observed_until=cut)
    return cache[key]


def tuning_cut(protocol: Any) -> pd.Timestamp:
    """What tuning may know: labels and cash known before the policy freeze (the replay
    of the validation window stops there too)."""
    return pd.Timestamp(protocol.freezes["policy"]) - pd.Timedelta(seconds=1)


def _baseline(run: Any, seed: int) -> Any:
    ref = next((r for r in run.all_worlds if r.seed == seed and r.family == BASELINE), None)
    if ref is None:  # build one of the same type
        sample = run.all_worlds[0]
        ref = replace(sample, seed=seed, family=BASELINE)
    return ref


def policies_for(scorers: Mapping[str, Any], bench: Bench) -> dict[str, policies.Policy]:
    return policies.policy_set(scorers, bench.world.terms, bench.ltv_cents())


# ------------------------------------------------------------------------- many replays


@dataclass(frozen=True)
class Task:
    """One replay of a world: a policy (by name and thresholds), a staffing, a variant."""

    policy: str
    review_threshold: float | None
    decline_threshold: float | None
    window: tuple[pd.Timestamp, pd.Timestamp]
    staff: Staffing
    history: str = "policy"
    reviewer: str = "evidence"
    rates: str = "verification"
    keys: tuple[tuple[str, Any], ...] = ()  # leading columns of the outcome row
    detail: bool = False  # also the confusion tables and prevented loss by pattern

    @classmethod
    def of(cls, policy: policies.Policy, window: tuple[pd.Timestamp, pd.Timestamp],
           staff: Staffing, **kwargs: Any) -> Task:
        keys = kwargs.pop("keys", {})
        return cls(policy.name, policy.review_threshold, policy.decline_threshold, window,
                   staff, keys=tuple(dict(keys).items()), **kwargs)


@dataclass
class Worker:
    """Replays of one world: its bench, the seed's untuned policies and the outcome truth."""

    bench: Bench
    policies: dict[str, policies.Policy]
    classes: pd.DataFrame
    latent_classes: pd.DataFrame
    legitimate_truth: pd.Index

    @classmethod
    def of(cls, bench: Bench, scorers: Mapping[str, Any]) -> Worker:
        latent = bench.latent_orders[["order_id", "pattern_id"]].assign(
            pattern_id=lambda f: f["pattern_id"].fillna("legitimate"))
        return cls(bench, policies_for(scorers, bench),
                   outcomes.truth(bench.world.tables, bench.world.observed_until), latent,
                   outcomes.latent_legitimate(bench.latent_orders))

    def run(self, task: Task) -> dict[str, Any]:
        policy = self.policies[task.policy].with_thresholds(task.review_threshold,
                                                             task.decline_threshold)
        result = self.bench.run(policy, task.window, task.staff, history=task.history,
                                reviewer=task.reviewer, rates=task.rates)
        out: dict[str, Any] = {"row": outcomes.outcome_row(
            result, self.bench.world, keys={**dict(task.keys), "policy_version": policy.version},
            ltv_cents=self.bench.ltv_cents(), classes=self.classes,
            legitimate_truth=self.legitimate_truth)}
        if task.detail and task.policy == INCUMBENT:
            out["incumbent"] = {"decisions": decision_rows(policy, result),
                                "alerts": alerts(result, policy)}
        if task.detail:
            out["confusion"] = _records(outcomes.confusion(result, self.classes))
            out["confusion_latent"] = _records(outcomes.confusion(
                result, self.latent_classes, by="pattern_id"))
            out["prevented"] = _records(outcomes.prevented_by_pattern(
                result, self.bench.world, self.classes))
        return out


_WORKER: Worker | None = None


def _start_worker(tables: Mapping[str, pd.DataFrame], context: pd.DataFrame,
                  scorers: bytes, seed: int, observed_until: pd.Timestamp,
                  policy_cfg: Mapping[str, Any]) -> None:
    global _WORKER
    bench = Bench.of(tables, context, seed=seed, observed_until=observed_until,
                     policy_cfg=policy_cfg)
    _WORKER = Worker.of(bench, pickle.loads(scorers))


def _run_task(task: Task) -> dict[str, Any]:
    assert _WORKER is not None
    return _WORKER.run(task)


def workers(run: Any) -> int:
    """Worker processes for replays: ``run.workers``, else ``BNPL_REPLAY_WORKERS``, else 1."""
    count = getattr(run, "workers", None) or os.environ.get("BNPL_REPLAY_WORKERS") or 1
    return max(int(count), 1)


class Replays:
    """Replays of one world, in this process or in a pool of worker processes.

    Every draw is keyed by seed, stream and id, so a task's result does not depend on
    the process that ran it; results come back in task order.
    """

    def __init__(self, run: Any, ref: Any, seed: int, count: int | None = None, *,
                 observed_until: pd.Timestamp | None = None) -> None:
        self.bench = _bench(run, ref, observed_until)
        self.scorers = run.memory["scorers"][seed]
        self.count = workers(run) if count is None else count
        self.local: Worker | None = None
        self.pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Replays:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.pool is not None:
            self.pool.shutdown()

    def run(self, tasks: Sequence[Task]) -> list[dict[str, Any]]:
        if self.count <= 1 or len(tasks) <= 1:
            if self.local is None:
                self.local = Worker.of(self.bench, self.scorers)
            return [self.local.run(task) for task in tasks]
        if self.pool is None:
            world = self.bench.world
            self.pool = ProcessPoolExecutor(
                max_workers=self.count, mp_context=multiprocessing.get_context("spawn"),
                initializer=_start_worker,
                initargs=(world.tables, world.context,
                          cloudpickle.dumps(self.scorers),  # fitted scorers may hold closures
                          world.seed, world.observed_until, self.bench.policy_cfg))
        return list(self.pool.map(_run_task, tasks))


# ------------------------------------------------------------------------- stages


TUNING_HISTORIES = ("shortlist", "frozen", "policy")  # "shortlist": the protocol's procedure


def tune(run: Any, *, history: str = "shortlist") -> StageOutput:
    """Each seed's policies tuned on its baseline world's validation window (base level).

    Tuning knows what was known before the policy freeze (:func:`tuning_cut`): the
    replay stops there, and outcome rows count only labels and cash known by then.
    ``history`` is the tuning procedure: ``shortlist`` (the protocol's: the whole grid
    with frozen approve-all history, the shortlist with policy-specific history),
    ``frozen`` or ``policy`` (the whole grid with that history alone).
    """
    if history not in TUNING_HISTORIES:
        raise ValueError(f"tuning history {history!r} is not one of {TUNING_HISTORIES}")
    policy_cfg = config.load("policy")
    grid = tuning.Grid.from_config(policy_cfg)
    guardrail = tuning.Guardrail.from_config(policy_cfg)
    best_k = int(policy_cfg["tuning"]["shortlist_best_k"])
    staff = base_staffing(policy_cfg)
    window = _window(run.protocol, "validation")
    frontier_rows, chosen_rows = [], []
    for seed in sorted({ref.seed for ref in run.all_worlds}):
        found = run.memory.setdefault("tuned", {}).setdefault(seed, {})
        with Replays(run, _baseline(run, seed), seed,
                     observed_until=tuning_cut(run.protocol)) as replays:
            bench = replays.bench

            def evaluate(with_history: str, replays: Replays = replays):
                def run_all(candidates: Sequence[policies.Policy]) -> list[dict[str, Any]]:
                    tasks = [Task.of(c, window, staff, history=with_history)
                             for c in candidates]
                    return [out["row"] for out in replays.run(tasks)]
                return run_all

            def routed(c: policies.Policy, bench: Bench = bench) -> float:
                return bench.routed_minutes(c, window)

            for name, policy in policies_for(run.memory["scorers"][seed], bench).items():
                if not policy.tunable:
                    found[name] = policy
                    continue
                scores = bench.checkout_scores(policy, window)
                if history == "shortlist":
                    tuned = tuning.tune_shortlist(
                        policy, scores, evaluate("frozen"), evaluate("policy"), grid,
                        k=best_k, guardrail=guardrail, routed_minutes=routed)
                else:
                    tuned = tuning.tune(policy, scores, evaluate(history), grid,
                                        routed_minutes=routed, guardrail=guardrail,
                                        history=history)
                found[name] = tuned.chosen
                frontier_rows += [{"seed": seed, "policy": name, "tuning": history, **row}
                                  for row in _records(tuned.frontier)]
                chosen_rows.append({
                    "seed": seed, "policy": name, "tuning": history,
                    "history": "policy" if history == "shortlist" else history,
                    "feasible_points": tuned.feasible_points, "points": len(tuned.frontier),
                    "chosen_version": None if tuned.chosen is None else tuned.chosen.version,
                    "review_threshold": None if tuned.chosen is None
                    else tuned.chosen.review_threshold,
                    "decline_threshold": None if tuned.chosen is None
                    else tuned.chosen.decline_threshold,
                    "review_on_boundary": tuned.on_boundary["review"],
                    "decline_on_boundary": tuned.on_boundary["decline"],
                    "screened_version": None if tuned.screened is None
                    else tuned.screened.version,
                    "shortlist": len(tuned.shortlist),
                })
    return StageOutput(tables={"tune.frontier": frontier_rows, "tune.chosen": chosen_rows},
                       notes=[f"tuning rule: {tuning.SHORTLIST_RULE.format(k=best_k)}; "
                              f"{tuning.RULE}" if history == "shortlist"
                              else f"tuning rule ({history} history): {tuning.RULE}"])


VARIANTS = (  # (history, reviewer, verification rates) at the base level
    ("policy", "evidence", "verification"),
    ("frozen", "evidence", "verification"),
    ("policy", "perfect", "verification"),
    ("policy", "evidence", "verification_weak"),
)


def replay(run: Any) -> StageOutput:
    """Every tuned policy and approve-all on every evaluation world's test window.

    Each world is replayed at every staffing (capacity levels, the redesigned layout)
    and with the variants at the base level; a world of a family in
    ``run.base_only_families`` (sensitivity families) only at the base level on the
    current layout, main variant.
    """
    policy_cfg = config.load("policy")
    window = _window(run.protocol, "test")
    current = policy_cfg["roster"]["layout"]
    base_only = set(getattr(run, "base_only_families", ()) or ())
    rows, confusion, latent, prevented = [], [], [], []
    for ref in run.worlds:
        tuned = run.memory["tuned"][ref.seed]
        tasks: list[Task] = []
        for name, policy in tuned.items():
            keys = {"seed": ref.seed, "family": ref.family, "policy": name}
            if policy is None:
                rows.append({**keys, "capacity_level": "base", "evaluated": False})
                continue
            for staff in staffing(policy_cfg):
                main = staff.level in ("base", "configured") and staff.layout == current
                if ref.family in base_only and not main:
                    continue
                variants = VARIANTS if main and ref.family not in base_only else VARIANTS[:1]
                for history, reviewer, rates in variants:
                    tasks.append(Task.of(
                        policy, window, staff, history=history, reviewer=reviewer,
                        rates=rates, detail=main and (history, reviewer, rates) == VARIANTS[0],
                        keys={**keys, "capacity_level": staff.level, "layout": staff.layout,
                              "history": history, "reviewer": reviewer,
                              "verification": rates, "evaluated": True}))
        with Replays(run, ref, ref.seed) as replays:
            results = replays.run(tasks)
        for task, out in zip(tasks, results, strict=True):
            rows.append(out["row"])
            if "incumbent" in out:
                run.memory.setdefault("incumbent", {})[ref] = out["incumbent"]
            if task.detail:
                keys = {"seed": ref.seed, "family": ref.family, "policy": task.policy}
                confusion += [{**keys, **r} for r in out["confusion"]]
                latent += [{**keys, **r} for r in out["confusion_latent"]]
                prevented += [{**keys, **r} for r in out["prevented"]]
    return StageOutput(tables={"replay.outcomes": rows, "replay.confusion": confusion,
                               "replay.confusion_latent": latent,
                               "replay.prevented_by_pattern": prevented})


def _incumbent(run: Any, ref: Any) -> dict[str, pd.DataFrame]:
    """The incumbent's review decisions and alerts on one world's test window at base
    capacity: kept by :func:`replay` from its main run, else replayed here."""
    kept = run.memory.setdefault("incumbent", {})
    if ref not in kept:
        policy = run.memory["tuned"][ref.seed][INCUMBENT]
        if policy is None:
            raise ValueError(f"the incumbent has no feasible operating point for seed {ref.seed}")
        bench = _bench(run, ref)
        result = bench.run(policy, _window(run.protocol, "test"),
                           base_staffing(config.load("policy")))
        kept[ref] = {"decisions": decision_rows(policy, result),
                     "alerts": alerts(result, policy)}
    return kept[ref]


def review_decisions(run: Any, ref: Any) -> pd.DataFrame:
    """The incumbent's reviews on one world's test window at base capacity.

    One row per order an analyst took up and decided: the context columns (core.asof
    KEY_COLUMNS + COLUMN_NAMES) of the row the reviewer's first decision read, exactly
    as the replay had it (``decision_at`` is when that evidence was assembled: the
    checkout, or the start of the day the review was taken up or decided, rebuilt under
    the incumbent's decisions before that day); ``taken_up_at`` and ``decided_at``;
    ``checks``, the verification checks completed when the first decision was made
    (none: checks start with a hold), and ``checks_later`` those completed afterwards,
    each ``(check, outcome, completed_at)``; ``disposition`` and ``final``, the
    reviewer's first and last decisions.
    """
    return _incumbent(run, ref)["decisions"]


def decision_rows(policy: policies.Policy, result: ReplayResult) -> pd.DataFrame:
    columns = [*asof.KEY_COLUMNS, *asof.COLUMN_NAMES]
    rows = result.review_rows
    reviews = result.reviews.set_index("order_id")
    out = rows[columns].reset_index(drop=True)
    out["taken_up_at"] = rows["started_at"].to_numpy()
    out["decided_at"] = rows["decided_at"].to_numpy()
    out["policy"] = policy.name
    out["policy_version"] = policy.version
    out["checks"] = [[] for _ in range(len(out))]
    order = out["order_id"].to_numpy()
    out["checks_later"] = list(reviews["check_results"].reindex(order)) if len(out) else []
    out["disposition"] = reviews["first_disposition"].reindex(order).to_numpy()
    out["final"] = reviews["final"].reindex(order).to_numpy()
    return out


def routing_frame(run: Any, ref: Any) -> pd.DataFrame:
    """The incumbent's routing at checkout on one world's test window: the MySQL
    ``alerts`` table.

    One row per order the incumbent routed to review or auto-declined at checkout
    (orders of blocked accounts were declined by the block, not alerted): ``alert_id``
    (order id plus policy version), ``order_id``, ``user_id``, ``ts`` (checkout),
    ``score``, ``band`` (``review`` or ``auto_decline``), ``fired_rules`` (FP-2 §6.2
    conditions that held at checkout, ``R06(a)``/``R06(b)`` named apart), ``policy``,
    ``policy_version``.
    """
    return _incumbent(run, ref)["alerts"]


def alerts(result: ReplayResult, policy: policies.Policy) -> pd.DataFrame:
    routes = result.routes
    flagged = routes.loc[routes["route"].isin([CheckoutRoute.REVIEW.value,
                                              CheckoutRoute.AUTO_DECLINE.value])]
    rows = result.alert_rows.set_index("order_id").reindex(flagged["order_id"]).reset_index()
    held = evidence.conditions(rows)
    rules = list(evidence.RULE_FAMILY)
    fired = [[rule for rule in rules if flags[rule]] for flags in held[rules].to_dict("records")]
    score = np.where(flagged["route"] == CheckoutRoute.AUTO_DECLINE.value,
                     flagged["decline_score"], flagged["review_score"])
    return pd.DataFrame({
        "alert_id": flagged["alert_id"].to_numpy(), "order_id": flagged["order_id"].to_numpy(),
        "user_id": rows["user_id"].to_numpy(np.int64),
        "ts": pd.to_datetime(rows["decision_at"]).to_numpy(),
        "score": score.astype(float), "band": flagged["route"].to_numpy(),
        "fired_rules": fired, "policy": policy.name, "policy_version": policy.version,
    })


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for record in frame.to_dict("records"):
        out.append({key: _plain(value) for key, value in record.items()})
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
