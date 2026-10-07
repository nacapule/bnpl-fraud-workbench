"""The replay: one policy acting on one world's orders, with a fixed review staff.

Run by :func:`replay`. Orders whose checkout falls in the window arrive in time order.
Each is routed at its checkout by the policy (``queue_sim.policies``) from its as-of
context row; an account blocked by an earlier decline or escalation has its later
orders declined at checkout. Orders routed to review wait in one queue, by FP-2 §7.1
priority and then by the policy's review score. Analysts on the roster
(``queue_sim.roster``) take the next order whenever they are free and on shift; a
review lasts a service time drawn for that order. When it ends, the reviewer
(``queue_sim.reviewer``) decides on the evidence known then; a hold runs the checks,
whose answers arrive later and are evaluated as they come, and a hold with no
answer after ``actions.hold_max_hours`` is cancelled (before shipment) or changes
nothing (after it). Each decision's effect on the order follows ``core.actions``:
before shipment a decline voids the order and a hold pauses it; after shipment the
loss stands and only the block applies. The merchant ships at the world's fulfilment
time unless a hold paused the order first.

Simulation is a heap of timed events (arrivals, review completions, check answers,
hold time-outs, analysts coming on shift), processed in time order; ties go to
arrivals first, so a free analyst always sees every order that has arrived. Random
draws (review times, check outcomes and delays) are keyed by order id
(``queue_sim.draws``), so every policy sees the same draws.

Evidence and history. The replay advances one day at a time. At the start of each
day it asks its :class:`History` for the context rows of that day's arrivals (at
their checkout) and of every order still waiting for review or held (as of the day's
start): a reviewer working on a later day sees evidence as known at the start of that
day, so decision-anchored columns, linkage and earlier outcomes included, are at most
one replay day stale and never from the future. :class:`PolicyHistory` rebuilds the
outcome-derived columns from this policy's own decisions so far (``core.asof``);
:class:`FrozenHistory` keeps the approve-all values, for the comparison that shows how
much policy-specific history matters.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from core import actions, asof, config, ledger
from core.actions import Check, CheckOutcome, CheckoutRoute, Disposition
from core.evidence import CheckResult
from queue_sim import policies
from queue_sim.reviewer import Decision, Verification, service_seconds
from queue_sim.roster import DAY, Roster, ServiceCalendar, to_seconds

# Event kinds, in the order simultaneous events are processed.
ARRIVAL, REVIEW_DONE, CHECK_DONE, HOLD_EXPIRE, WAKE = range(5)
PRIORITY_RANK = {name: rank for rank, name in enumerate(policies.PRIORITIES)}


@dataclass(frozen=True)
class World:
    """What a replay reads from one generated world.

    ``tables`` are the world's tables (core.world); ``context`` is its world-level
    as-of context (``core.asof.build_context``: one row per order attempt at checkout).
    Latent truth is not here: it reaches the replay only through
    :class:`queue_sim.reviewer.Verification`, which draws check outcomes from it.
    """

    tables: Mapping[str, pd.DataFrame]
    context: pd.DataFrame
    seed: int
    observed_until: pd.Timestamp
    terms: ledger.ProductTerms
    scores: dict[tuple[str, str], pd.Series] = field(default_factory=dict, compare=False,
                                                     repr=False)

    def checkout_scores(self, signal: policies.Signal | None) -> None:
        """Score every world-level checkout row once per signal (kept for every replay)."""
        if signal is not None and signal.key not in self.scores:
            self.scores[signal.key] = pd.Series(
                signal(self.context), index=self.context["order_id"].to_numpy())

    def orders(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Processor-approved attempts with checkout in [start, end), in arrival order."""
        attempts = self.tables["order_attempts"]
        mask = (attempts["processor_result"] == "approved") & (attempts["known_at"] >= start) \
            & (attempts["known_at"] < end)
        out = attempts.loc[mask, ["order_id", "user_id", "known_at", "amount_cents"]].rename(
            columns={"known_at": "checkout_at"})
        ships = self.tables["fulfilments"].groupby("order_id")["occurred_at"].min()
        out["shipped_at"] = out["order_id"].map(ships)
        return out.sort_values(["checkout_at", "order_id"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class Settings:
    hold_max_hours: float
    service_mean_minutes: float
    service_sigma: float
    senior_minutes: float

    @classmethod
    def from_config(cls, policy: Mapping[str, Any] | None = None) -> Settings:
        policy = config.load("policy") if policy is None else policy
        return cls(
            hold_max_hours=float(policy["actions"]["hold_max_hours"]),
            service_mean_minutes=float(policy["roster"]["service_time_arithmetic_mean_min"]),
            service_sigma=float(policy["roster"]["service_time_sigma"]),
            senior_minutes=float(policy["actions"]["senior_review_minutes"]),
        )


class SoFar(Protocol):
    """What the policy has decided before a moment, built only when asked for."""

    def changed_users(self) -> np.ndarray:
        """Accounts whose orders it declined at checkout, held or voided, or that it blocked."""
        ...

    def fates(self, users: np.ndarray | None = None) -> pd.DataFrame:
        """core.actions fates of the orders checked out before the moment, as decided then
        (only the orders of ``users`` when given)."""
        ...

    def blocks(self) -> pd.DataFrame:
        """Blocks (user_id, at) placed before the moment."""
        ...


class History(Protocol):
    """Context rows for decisions, under what the policy has decided so far."""

    def rows(self, decisions: pd.DataFrame, so_far: SoFar, at: pd.Timestamp) -> pd.DataFrame:
        """One row per decision (``order_id``, ``decision_at``), in the same order.

        ``so_far`` is the policy's decisions made before ``at``; every decision is at or
        after ``at``.
        """
        ...


def _checkout_rows(world: World, decisions: pd.DataFrame) -> pd.Series:
    checkout = world.context.set_index("order_id")["decision_at"]
    return decisions["decision_at"].to_numpy() == checkout.reindex(
        decisions["order_id"]).to_numpy()


@dataclass
class FrozenHistory:
    """Approve-all evidence: outcome-derived columns as if every order were approved."""

    world: World
    build: Callable[..., pd.DataFrame] | None = None  # default core.asof.build_context
    neighbours: Any = None
    rebuilt: np.ndarray | None = None  # rows of the last call that differ from approve-all
    _cache: dict[tuple[int, Any], dict[str, Any]] = field(default_factory=dict, repr=False)

    def rows(self, decisions: pd.DataFrame, so_far: SoFar | None,
             at: pd.Timestamp) -> pd.DataFrame:
        at_checkout = _checkout_rows(self.world, decisions)
        context = self.world.context
        by_order = context.set_index("order_id", drop=False)
        parts = [by_order.reindex(decisions.loc[at_checkout, "order_id"]).set_index(
            decisions.index[at_checkout])]
        later = decisions.loc[~at_checkout]
        if len(later):
            keys = [(int(order), pd.Timestamp(at)) for order, at in
                    zip(later["order_id"], later["decision_at"], strict=True)]
            missing = [key not in self._cache for key in keys]
            if any(missing):
                build = self.build or asof.build_context
                extra = {} if self.neighbours is None else {
                    "world_context": self.world.context, "neighbours": self.neighbours}
                built = build(self.world.tables, later.loc[missing].reset_index(drop=True),
                              **extra)
                for record in built.to_dict("records"):
                    key = (int(record["order_id"]), pd.Timestamp(record["decision_at"]))
                    self._cache[key] = record
            parts.append(pd.DataFrame([self._cache[key] for key in keys], index=later.index))
        out = pd.concat(parts).loc[decisions.index, list(context.columns)]
        self.rebuilt = np.zeros(len(out), dtype=bool)
        return out.reset_index(drop=True)


@dataclass
class PolicyHistory:
    """Outcome-derived columns rebuilt from the policy's own decisions (core.asof).

    Each day's rows start as the approve-all rows at the same decision times. Rows of
    accounts the policy has touched (declined, held, voided or blocked), or that ever
    shared a device, an address or an email with one (``neighbours``,
    ``core.asof.Neighbours``), then get their outcome-derived columns from
    ``core.asof.policy_rows`` on the realized tables and state as decided before the
    day. Other accounts' outcomes are exactly approve-all's, so their rows stay.
    """

    world: World
    neighbours: Any = None
    policy_rows: Callable[..., pd.DataFrame] | None = None  # default core.asof.policy_rows
    frozen: FrozenHistory | None = None
    calls: int = 0
    rebuilt: np.ndarray | None = None  # rows of the last call rebuilt under the policy
    _memo: dict[Any, pd.DataFrame] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.frozen = self.frozen or FrozenHistory(self.world, neighbours=self.neighbours)

    def rows(self, decisions: pd.DataFrame, so_far: SoFar, at: pd.Timestamp) -> pd.DataFrame:
        world_rows = self.frozen.rows(decisions, so_far, at)
        self.rebuilt = np.zeros(len(world_rows), dtype=bool)
        changed = so_far.changed_users()
        if not len(changed):
            return world_rows
        touched = (self.neighbours.around(changed) if self.neighbours is not None
                   else world_rows["user_id"].to_numpy())
        mask = np.isin(world_rows["user_id"].to_numpy(np.int64), touched)
        if not mask.any():
            return world_rows
        self.rebuilt = mask
        users = None
        if self.neighbours is not None:  # what core.asof.outcome_columns reads for them
            users = self.neighbours.around(
                np.unique(world_rows.loc[mask, "user_id"].to_numpy(np.int64)))
        fates, blocks = so_far.fates(users), so_far.blocks()
        view = _policy_view(self.world.tables, users, at)
        realized = actions.realize(view, fates, self.world.terms, memo=self._memo)
        state = actions.policy_state(fates, blocks)
        kwargs = {} if self.neighbours is None else {"neighbours": self.neighbours}
        self.calls += 1
        policy_rows = self.policy_rows or asof.policy_rows
        rebuilt = policy_rows(world_rows.loc[mask], realized, state,
                              decisions.loc[mask].reset_index(drop=True), **kwargs)
        out = world_rows.copy()
        for column in asof.OUTCOME_COLUMNS:
            out.loc[mask, column] = rebuilt[column].to_numpy()
        return out


_ACCOUNT_TABLES = ("accounts", "account_events", "device_links", "address_links", "cards",
                   "order_attempts")


def _accounts_view(tables: Mapping[str, pd.DataFrame],
                   users: np.ndarray | None) -> dict[str, pd.DataFrame]:
    """The world with only ``users``' account rows and attempts (all when None)."""
    out = dict(tables)
    if users is not None:
        for name in _ACCOUNT_TABLES:
            frame = tables[name]
            out[name] = frame.loc[np.isin(frame["user_id"].to_numpy(np.int64), users)]
    return out


def _policy_view(tables: Mapping[str, pd.DataFrame], users: np.ndarray | None,
                 at: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """The world's rows the policy's history at ``at`` is built from.

    With ``users``, only those accounts' rows (they include every account whose rows
    the outcome columns of the decided accounts read). Order-bound rows (plans,
    shipments, payments, disputes, reports, cash) only for orders checked out before
    ``at``; the attempts themselves whole.
    """
    out = _accounts_view(tables, users)
    attempts = out["order_attempts"]
    early = attempts.loc[attempts["known_at"] < at, "order_id"].to_numpy(np.int64)
    plans = tables["plans"]
    early_plans = plans.loc[np.isin(plans["order_id"].to_numpy(np.int64), early),
                            "plan_id"].to_numpy(np.int64)
    disputes = tables["dispute_openings"]
    early_disputes = disputes.loc[np.isin(disputes["order_id"].to_numpy(np.int64), early),
                                  "dispute_id"].to_numpy(np.int64)
    for name in ("plans", "fulfilments", "deliveries", "dispute_openings", "victim_reports",
                 "cash_events"):
        out[name] = tables[name].loc[np.isin(tables[name]["order_id"].to_numpy(np.int64), early)]
    for name in ("installment_schedule", "payment_attempts", "payment_reversals",
                 "plan_writeoffs"):
        out[name] = tables[name].loc[np.isin(tables[name]["plan_id"].to_numpy(np.int64),
                                             early_plans)]
    out["dispute_resolutions"] = tables["dispute_resolutions"].loc[np.isin(
        tables["dispute_resolutions"]["dispute_id"].to_numpy(np.int64), early_disputes)]
    return out


# ------------------------------------------------------------------------- the replay


@dataclass
class _Order:
    order_id: int
    user_id: int
    checkout: int
    ship: int | None  # the world's shipment time
    route: str = CheckoutRoute.APPROVE.value
    priority: str | None = None
    queue_score: float = 0.0
    row: dict[str, Any] | None = None  # context row the reviewer reads now
    started: int | None = None
    decided: int | None = None
    analyst: int | None = None
    first: Decision | None = None
    shipped_at_decision: bool | None = None
    hold_at: int | None = None
    hold_before_shipment: bool = False
    hold_expires: int | None = None
    checks: dict[Check, CheckResult] = field(default_factory=dict)
    started_checks: set[Check] = field(default_factory=set)
    released: int | None = None
    hold_outcome: str | None = None
    hold_ended: int | None = None
    void_at: int | None = None
    void_cause: str | None = None
    final: str | None = None
    final_at: int | None = None
    senior_minutes: float = 0.0

    @property
    def holding(self) -> bool:
        return self.hold_at is not None and self.hold_outcome is None

    def shipped(self, t: int) -> bool:
        """Whether the merchant has shipped by ``t`` (a pending pre-shipment hold pauses it)."""
        if self.ship is None or (self.holding and self.hold_before_shipment):
            return False
        if self.released is not None:
            return self.ship + (self.released - self.hold_at) <= t
        return self.ship <= t


@dataclass
class ReplayResult:
    policy: str
    policy_version: str
    window: tuple[pd.Timestamp, pd.Timestamp]
    fates: pd.DataFrame  # core.actions.FATE_COLUMNS, one row per order in the window
    blocks: pd.DataFrame  # user_id, at, cause, order_id
    routes: pd.DataFrame  # order_id, route, review_score, decline_score, alert_id
    alert_rows: pd.DataFrame  # context rows at checkout of orders routed to review or declined
    reviews: pd.DataFrame  # one row per order routed to review
    log: pd.DataFrame  # at, kind, order_id, backlog
    available_minutes: float  # analysts' review minutes inside the window
    coverage_minutes: float  # minutes inside the window with someone on shift
    roster: Roster
    reviewer: str
    history: str


def replay(
    world: World,
    policy: policies.Policy,
    *,
    window: tuple[pd.Timestamp, pd.Timestamp],
    roster: Roster,
    calendar: ServiceCalendar,
    reviewer: Any,
    verification: Verification,
    history: History,
    settings: Settings,
    linked_accounts: Callable[..., pd.DataFrame] | None = None,
) -> ReplayResult:
    """Replay ``policy`` on the world's orders with checkout in ``window``.

    Orders before the window keep their approve-all history (the policy starts at the
    window). Outcomes keep arriving until ``world.observed_until``; reviews still
    waiting then are left undecided.
    """
    start, end = (pd.Timestamp(t) for t in window)
    orders = world.orders(start, end)
    horizon = int(to_seconds(world.observed_until))
    t0, t1 = int(to_seconds(start)), int(to_seconds(end))
    linked = linked_accounts or getattr(asof, "linked_accounts", None)
    hold_s = int(round(settings.hold_max_hours * 3600))
    sim = _Simulation(world, policy, roster, reviewer, verification, settings, linked,
                      hold_s, t0, horizon)
    sim.neighbours = getattr(history, "neighbours", None)  # core.asof.Neighbours, if any
    for record in orders.itertuples(index=False):
        ship = None if pd.isna(record.shipped_at) else int(to_seconds(record.shipped_at))
        sim.orders[int(record.order_id)] = _Order(int(record.order_id), int(record.user_id),
                                                  int(to_seconds(record.checkout_at)), ship)
    arrivals = list(sim.orders.values())
    needs_rows = policy.review is not None or policy.decline is not None
    world.checkout_scores(policy.review)
    world.checkout_scores(policy.decline)
    routes: list[pd.DataFrame] = []
    alert_rows: list[pd.DataFrame] = []
    cursor = 0
    day = t0 // DAY
    while True:
        day_start, day_end = day * DAY, (day + 1) * DAY
        if day_start >= horizon:
            break
        todays = []
        while cursor < len(arrivals) and arrivals[cursor].checkout < day_end:
            todays.append(arrivals[cursor])
            cursor += 1
        waiting = sim.waiting()
        if not todays and not sim.events and cursor >= len(arrivals) and (
                not waiting or not sim.can_work(day_start)):
            break
        if needs_rows and (todays or waiting):
            at = pd.Timestamp(max(day_start, t0), unit="s")
            decisions = pd.DataFrame({
                "order_id": [o.order_id for o in todays] + [o.order_id for o in waiting],
                "decision_at": np.array([o.checkout for o in todays] + [max(day_start, o.checkout)
                                        for o in waiting], dtype="datetime64[s]"),
            })
            rows = history.rows(decisions, _SoFar(sim, at), at)
            records = rows.to_dict("records")
            for o, record in zip(todays + waiting, records, strict=True):
                o.row = record
            if todays:
                rebuilt = getattr(history, "rebuilt", None)
                fresh = None if rebuilt is None else np.asarray(rebuilt[:len(todays)], bool)
                routed = sim.route(todays, rows.iloc[:len(todays)], fresh)
                routes.append(routed)
                flagged = routed["route"].to_numpy() != CheckoutRoute.APPROVE.value
                if flagged.any():
                    alert_rows.append(rows.iloc[:len(todays)].loc[flagged])
        elif todays:
            routes.append(sim.route(todays, None))
        for o in todays:
            sim.push(o.checkout, ARRIVAL, o.order_id)
        sim.run(until=day_end)
        day += 1

    route_frame = (pd.concat(routes, ignore_index=True) if routes else pd.DataFrame(
        columns=["order_id", "route", "review_score", "decline_score"]))
    route_frame["alert_id"] = policies.alert_id(route_frame["order_id"], policy.version) \
        if len(route_frame) else pd.Series(dtype=object)
    return ReplayResult(
        policy=policy.name, policy_version=policy.version, window=(start, end),
        fates=sim.fates(), blocks=sim.blocks_frame(), routes=route_frame,
        alert_rows=(pd.concat(alert_rows, ignore_index=True) if alert_rows
                    else world.context.iloc[0:0]),
        reviews=sim.reviews_frame(calendar), log=sim.log_frame(),
        available_minutes=roster.available_minutes(t0, t1),
        coverage_minutes=roster.coverage_minutes(t0, t1), roster=roster,
        reviewer=getattr(reviewer, "name", type(reviewer).__name__),
        history=type(history).__name__,
    )


@dataclass
class _SoFar:
    sim: _Simulation
    at: pd.Timestamp

    def changed_users(self) -> np.ndarray:
        cut = int(to_seconds(self.at))
        return np.array(sorted(u for u, t in self.sim.changed.items() if t < cut), dtype=np.int64)

    def fates(self, users: np.ndarray | None = None) -> pd.DataFrame:
        return self.sim.fates(before=self.at, users=users)

    def blocks(self) -> pd.DataFrame:
        return self.sim.blocks_frame(before=self.at)


class _Simulation:
    def __init__(self, world: World, policy: policies.Policy, roster: Roster, reviewer: Any,
                 verification: Verification, settings: Settings,
                 linked: Callable[..., pd.DataFrame] | None, hold_s: int, t0: int,
                 horizon: int) -> None:
        self.world, self.policy, self.reviewer = world, policy, reviewer
        self.verification, self.settings, self.linked = verification, settings, linked
        self.hold_s, self.horizon = hold_s, horizon
        self.orders: dict[int, _Order] = {}
        self.events: list[tuple[int, int, int, int, int]] = []
        self.seq = 0
        self.queue: list[tuple[int, float, int, int]] = []
        self.blocked: dict[int, tuple[int, str, int]] = {}  # user -> (at, cause, order)
        self.log: list[tuple[int, int, str, int]] = []  # at, seq, kind, order
        names = roster.analysts
        self.clocks = [roster.clock(shift, t0, horizon) for _, shift in names]
        self.analyst_names = [name for name, _ in names]
        self.busy = [False] * len(names)
        self.wake_at: list[int | None] = [None] * len(names)
        self.service: dict[int, int] = {}
        self.changed: dict[int, int] = {}  # user -> first time the policy acted on it
        self.neighbours: Any = None

    def touch(self, user: int, at: int) -> None:
        if user not in self.changed or self.changed[user] > at:
            self.changed[user] = at

    # -- bookkeeping
    def push(self, at: int, kind: int, a: int, b: int = 0) -> None:
        self.seq += 1
        heapq.heappush(self.events, (at, kind, self.seq, a, b))

    def note(self, at: int, kind: str, order: int) -> None:
        self.seq += 1
        self.log.append((at, self.seq, kind, order))

    def can_work(self, now: int) -> bool:
        """Whether some analyst still has a shift ahead."""
        return any(clock.next_on_shift(now) is not None for clock in self.clocks)

    def waiting(self) -> list[_Order]:
        return [o for o in self.orders.values()
                if o.route == CheckoutRoute.REVIEW.value and o.final is None
                and o.priority is not None and (o.decided is None or o.holding)]

    def route(self, todays: list[_Order], rows: pd.DataFrame | None,
              fresh: np.ndarray | None = None) -> pd.DataFrame:
        if rows is None:
            return pd.DataFrame({"order_id": [o.order_id for o in todays],
                                 "route": CheckoutRoute.APPROVE.value,
                                 "review_score": np.nan, "decline_score": np.nan})
        routed = self.policy.route(rows, known=self.world.scores, fresh=fresh)
        review = routed["route"].to_numpy() == CheckoutRoute.REVIEW.value
        if review.any():
            entered = np.array([o.checkout for o in todays], dtype="datetime64[s]")[review]
            prio = policies.priority(rows.loc[review], entered, entered)
            ids = rows.loc[review, "order_id"].to_numpy(dtype=np.int64)
            seconds = service_seconds(self.world.seed, ids, self.settings.service_mean_minutes,
                                      self.settings.service_sigma)
            for order, p, s in zip(ids, prio, seconds, strict=True):
                self.orders[int(order)].priority = str(p)
                self.service[int(order)] = int(s)
        for o, r, score in zip(todays, routed["route"], routed["review_score"], strict=True):
            o.route = str(r)
            o.queue_score = 0.0 if np.isnan(score) else float(score)
        return routed.reset_index(drop=True)

    # -- the event loop
    def run(self, until: int) -> None:
        while self.events and self.events[0][0] < until:
            at, kind, _, a, b = heapq.heappop(self.events)
            if at >= self.horizon:
                self.events.clear()
                return
            if kind == ARRIVAL:
                self.arrive(self.orders[a], at)
            elif kind == REVIEW_DONE:
                self.busy[b] = False
                self.review_done(self.orders[a], at)
            elif kind == CHECK_DONE:
                self.check_done(self.orders[a], Check(list(Check)[b]), at)
            elif kind == HOLD_EXPIRE:
                self.hold_expired(self.orders[a], at)
            elif kind == WAKE:
                self.wake_at[a] = None
            self.dispatch(at)

    def arrive(self, o: _Order, at: int) -> None:
        block = self.blocked.get(o.user_id)
        if block is not None and block[0] <= at:
            o.route = actions.BLOCKED
        if o.route != CheckoutRoute.REVIEW.value:
            o.final, o.final_at = o.route, at
            if o.route != CheckoutRoute.APPROVE.value:
                self.touch(o.user_id, at)
            return
        heapq.heappush(self.queue, (PRIORITY_RANK[o.priority], -o.queue_score, o.checkout,
                                    o.order_id))
        self.note(at, "enter", o.order_id)

    def dispatch(self, now: int) -> None:
        for i, clock in enumerate(self.clocks):
            if not self.queue:
                return
            if self.busy[i]:
                continue
            on = clock.next_on_shift(now)
            if on is None:
                continue
            if on > now:
                if self.wake_at[i] != on:
                    self.wake_at[i] = on
                    self.push(on, WAKE, i)
                continue
            _, _, _, order = heapq.heappop(self.queue)
            o = self.orders[order]
            finish = clock.take(now, self.service[order])
            if finish is None:  # no shift left to finish it: it stays undecided
                heapq.heappush(self.queue, (PRIORITY_RANK[o.priority], -o.queue_score,
                                            o.checkout, o.order_id))
                continue
            self.busy[i] = True
            o.started, o.analyst = now, i
            self.note(now, "start", order)
            self.push(finish, REVIEW_DONE, order, i)

    def review_done(self, o: _Order, at: int) -> None:
        o.decided = at
        o.shipped_at_decision = o.shipped(at)
        decision = self.reviewer.decide(o.order_id, o.row, ())
        o.first = decision
        self.note(at, "decide", o.order_id)
        self.apply(o, decision, at)

    def check_done(self, o: _Order, check: Check, at: int) -> None:
        if not o.holding:
            return
        outcome, _ = self.verification.draw(o.order_id, check)
        o.checks[check] = CheckResult(check, outcome, pd.Timestamp(at, unit="s").to_pydatetime())
        self.apply(o, self.reviewer.decide(o.order_id, o.row, tuple(o.checks.values())), at)

    def hold_expired(self, o: _Order, at: int) -> None:
        if not o.holding:
            return
        o.hold_ended = at
        self.touch(o.user_id, at)
        if o.hold_before_shipment:
            o.hold_outcome, o.void_at, o.void_cause = "cancelled", at, "hold_cancelled"
            o.final, o.final_at = "cancelled", at
        else:  # after shipment no answer changes nothing (FP-2 §5.3(c))
            o.hold_outcome, o.final, o.final_at = "cleared", "unchanged", at

    def apply(self, o: _Order, decision: Decision, at: int) -> None:
        disposition = decision.disposition
        if disposition is Disposition.CLEAR:
            if o.holding:
                o.hold_outcome, o.hold_ended = "cleared", at
                if o.hold_before_shipment:
                    o.released = at
            o.final, o.final_at = "clear", at
        elif disposition in (Disposition.DECLINE, Disposition.ESCALATE):
            if not o.shipped(at):
                o.void_at, o.void_cause = at, disposition.value
                self.touch(o.user_id, at)
            if o.holding:
                o.hold_outcome, o.hold_ended = "declined", at
            self.block(o.user_id, at, disposition.value, o.order_id)
            if disposition is Disposition.ESCALATE:
                o.senior_minutes += self.settings.senior_minutes
                for user in self.linked_users(o, at):
                    self.block(user, at, "linked", o.order_id)
            o.final, o.final_at = disposition.value, at
        else:  # hold: run the checks the policy requires that have not run
            if o.hold_at is None:
                self.touch(o.user_id, at)
                o.hold_at, o.hold_before_shipment = at, not o.shipped(at)
                o.hold_expires = at + self.hold_s
                self.push(o.hold_expires, HOLD_EXPIRE, o.order_id)
            for check in decision.checks_to_run:
                if check in o.started_checks:
                    continue
                o.started_checks.add(check)
                outcome, delay = self.verification.draw(o.order_id, check)
                if outcome is not CheckOutcome.NO_RESPONSE and at + delay < o.hold_expires:
                    self.push(at + delay, CHECK_DONE, o.order_id, list(Check).index(check))

    def linked_users(self, o: _Order, at: int) -> list[int]:
        if self.linked is None:
            raise RuntimeError("escalation needs core.asof.linked_accounts")
        tables = self.world.tables
        if self.neighbours is not None:  # linked accounts share a device or an address
            tables = _accounts_view(tables, self.neighbours.around(np.array([o.user_id])))
        found = self.linked(tables, pd.DataFrame({
            "order_id": [o.order_id], "decision_at": np.array([at], dtype="datetime64[s]")}))
        return sorted({int(u) for u in found["user_id"]} - {o.user_id})

    def block(self, user: int, at: int, cause: str, order: int) -> None:
        self.touch(user, at)
        if user not in self.blocked or self.blocked[user][0] > at:
            self.blocked[user] = (at, cause, order)

    # -- results
    def fates(self, before: pd.Timestamp | None = None,
              users: np.ndarray | None = None) -> pd.DataFrame:
        cut = None if before is None else int(to_seconds(before))
        wanted = None if users is None else {int(u) for u in users}
        rows = []
        for o in self.orders.values():
            if (cut is not None and o.checkout >= cut) or (
                    wanted is not None and o.user_id not in wanted):
                continue
            rows.append({
                "order_id": o.order_id, "user_id": o.user_id,
                "checkout_at": o.checkout, "route": o.route,
                "hold_at": o.hold_at, "hold_before_shipment": o.hold_before_shipment,
                "released_at": o.released, "hold_outcome": o.hold_outcome,
                "hold_ended_at": o.hold_ended,
                "void_at": o.void_at, "void_cause": o.void_cause,
            })
        frame = pd.DataFrame(rows, columns=list(actions.FATE_COLUMNS))
        for column in ("checkout_at", "hold_at", "released_at", "hold_ended_at", "void_at"):
            frame[column] = _stamps(frame[column])
        frame["hold_before_shipment"] = frame["hold_before_shipment"].astype(bool)
        fates = actions.typed_fates(frame)
        return fates if before is None else actions.fates_as_of(fates, before)

    def blocks_frame(self, before: pd.Timestamp | None = None) -> pd.DataFrame:
        cut = None if before is None else int(to_seconds(before))
        rows = [(user, at, cause, order) for user, (at, cause, order) in self.blocked.items()
                if cut is None or at < cut]
        frame = pd.DataFrame(rows, columns=["user_id", "at", "cause", "order_id"])
        frame["at"] = _stamps(frame["at"])
        return frame.astype({"user_id": "int64", "order_id": "int64"}).sort_values(
            ["at", "user_id"]).reset_index(drop=True)

    def reviews_frame(self, calendar: ServiceCalendar) -> pd.DataFrame:
        reviewed = [o for o in self.orders.values() if o.route == CheckoutRoute.REVIEW.value]
        rows = []
        for o in reviewed:
            first = o.first
            rows.append({
                "order_id": o.order_id, "priority": o.priority, "entered_at": o.checkout,
                "started_at": o.started, "decided_at": o.decided,
                "analyst": None if o.analyst is None else self.analyst_names[o.analyst],
                "service_seconds": self.service.get(o.order_id),
                "first_disposition": None if first is None else first.disposition.value,
                "table_row": None if first is None else first.row,
                "families": None if first is None else first.families,
                "rules": None if first is None else ",".join(first.rules),
                "shipped_at_decision": o.shipped_at_decision,
                "checks": ",".join(f"{c.check.value}:{c.outcome.value}"
                                   for c in o.checks.values()),
                "check_results": [(c.check.value, c.outcome.value, pd.Timestamp(c.completed_at))
                                  for c in o.checks.values()],
                "checks_started": len(o.started_checks),
                "hold_before_shipment": o.hold_at is not None and o.hold_before_shipment,
                "final": o.final, "final_at": o.final_at, "senior_minutes": o.senior_minutes,
            })
        frame = pd.DataFrame(rows, columns=[
            "order_id", "priority", "entered_at", "started_at", "decided_at", "analyst",
            "service_seconds", "first_disposition", "table_row", "families", "rules",
            "shipped_at_decision", "checks", "check_results", "checks_started",
            "hold_before_shipment",
            "final", "final_at", "senior_minutes"])
        decided = frame["decided_at"].notna()
        service = np.full(len(frame), np.nan)
        if decided.any():
            service[decided.to_numpy()] = calendar.service_hours(
                frame.loc[decided, "entered_at"].to_numpy(dtype=np.int64),
                frame.loc[decided, "decided_at"].to_numpy(dtype=np.int64))
        frame["service_hours_to_decision"] = service
        for column in ("entered_at", "started_at", "decided_at", "final_at"):
            frame[column] = _stamps(frame[column])
        return frame

    def log_frame(self) -> pd.DataFrame:
        rows = sorted(self.log)
        backlog, out = 0, []
        for at, _, kind, order in rows:
            backlog += 1 if kind == "enter" else -1 if kind == "start" else 0
            out.append((at, kind, order, backlog))
        frame = pd.DataFrame(out, columns=["at", "kind", "order_id", "backlog"])
        frame["at"] = _stamps(frame["at"])
        return frame


def _stamps(values: pd.Series) -> pd.Series:
    seconds = pd.array(values, dtype="Int64")
    stamps = np.full(len(seconds), np.datetime64("NaT", "s"), dtype="datetime64[s]")
    present = ~pd.isna(seconds)
    stamps[np.asarray(present)] = np.asarray(seconds[present], dtype=np.int64).astype(
        "datetime64[s]")
    return pd.Series(stamps, index=values.index)
