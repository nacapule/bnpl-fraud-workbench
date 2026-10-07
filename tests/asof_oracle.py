"""The as-of context computed row by row from its written specification.

For each decision, every column of ``core.asof.COLUMNS`` is computed by the most
literal reading of its definition: filter the decision's account, device,
address, card or orders by the visibility rule of the ``core.asof`` module
docstring, then count, sum or take the minimum, in plain Python. Nothing is
shared with ``core.asof`` beyond the email rule (``normalize_email``), the column
list and the reference values its definitions name (the domain lists, country
centroids and minimum category sample); never-pay determinations come from
``core.world.adjudicate``'s rule, as the definition says; the email holdings and
account sharing are written here from their definitions. Speed does not matter:
the worlds are small.

What a decision sees (the module docstring): a checkout sees the events before
its attempt in the order ``(known_at, kind rank, event_id)``; a later decision
at ``T`` sees the events known by ``T``; entities from their creation, also in
the same second; derived times (due dates, determinations, the policy's
approvals at a release, voids, holds and blocks) at a later decision when at or
before ``T``, at a checkout only when strictly earlier. An approval at checkout
takes the attempt's own place in the event order. Order-anchored columns are
computed at the checkout whatever the decision time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core import asof, config, ledger, world

HOUR, DAY = 3_600, 86_400
HOURS_CAP = 10_000.0  # "capped at 10,000 (also when none)"
EARTH_RADIUS_KM = 6_371.0
INR = "item_not_received"


def _seconds(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(pd.Timestamp(value).value // 10**9)


def _records(frame: pd.DataFrame, table: str | None = None) -> list[SimpleNamespace]:
    """Rows as objects; times in whole seconds, missing values None; event rows get
    their place ``key`` in the event order."""
    timed = {column for column in frame.columns if frame[column].dtype.kind == "M"}
    rows = []
    for values in frame.to_dict("records"):
        row = {}
        for column, value in values.items():
            if column in timed:
                row[column] = _seconds(value)
            elif not isinstance(value, str) and pd.isna(value):
                row[column] = None
            else:
                row[column] = value
        if table in world.EVENT_RANK:
            row["key"] = (row["known_at"], world.EVENT_RANK[table], row["event_id"])
        rows.append(SimpleNamespace(**row))
    return rows


def _group(rows: list, column: str) -> defaultdict:
    groups = defaultdict(list)
    for row in rows:
        groups[getattr(row, column)].append(row)
    return groups


@dataclass(frozen=True)
class View:
    """What one decision sees: a checkout (``at`` is the attempt's second and ``place``
    its key) or a later decision at the second ``at``."""

    at: int
    checkout: bool
    place: tuple

    def event(self, row, current: bool = False) -> bool:
        """An event row; ``current`` also admits the decided attempt itself."""
        if self.checkout:
            return row.key < self.place or (current and row.key == self.place)
        return row.known_at <= self.at

    def entity(self, created: int | None) -> bool:
        return created is not None and created <= self.at

    def derived(self, time: int | None) -> bool:
        if time is None:
            return False
        return time < self.at if self.checkout else time <= self.at


def _quantile(values: list[int], q: float) -> float:
    """Linear interpolation between the closest ranks."""
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    low = math.floor(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def _km(first: str, second: str) -> float:
    (lat1, lon1), (lat2, lon2) = ((math.radians(x) for x in asof.CENTROIDS[c])
                                  for c in (first, second))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _overlap_start(start1: int, end1: int | None, start2: int, end2: int | None) -> int | None:
    """Start of the overlap of [start1, end1) and [start2, end2) (None: open), if any."""
    start = max(start1, start2)
    ends = [end for end in (end1, end2) if end is not None]
    return start if not ends or start < min(ends) else None


def never_pay_determinations(tables, went_through, stops) -> dict[int, int]:
    """order_id -> when its never-pay determination is known, by core.world.adjudicate's
    rule on these tables with the orders that went through as the approved ones, each from
    when it went through (``went_through``: order_id -> that time), so an order released
    from a hold shares its shipping address from the release. A plan
    the policy stopped (``stops``: order_id -> when it was voided or cancelled) owes
    nothing after that, so it defaults, and marks others, only if its default came
    strictly before the stop. Only the never-pay rule's inputs are passed, so every
    never-pay determination shows, also on an order whose first label has another
    basis."""
    rules = config.load("world")["labels"]
    orders = tables["order_attempts"]
    approved = orders["order_id"].isin(list(went_through))
    schedule = tables["installment_schedule"]
    first_due = {plan: _seconds(due) for plan, seq, due in zip(
        schedule["plan_id"], schedule["seq"], schedule["due_at"], strict=True) if seq == 1}
    plans = tables["plans"]
    owed = [order not in stops or (plan in first_due and first_due[plan]
                                   + rules["default_grace_days"] * DAY < stops[order])
            for plan, order in zip(plans["plan_id"], plans["order_id"], strict=True)]
    inputs = {
        **tables,
        "plans": plans[owed],
        "order_attempts": orders.assign(
            processor_result=np.where(approved, "approved", "declined"),
            occurred_at=[pd.Timestamp(went_through[o], unit="s") if o in went_through else at
                         for o, at in zip(orders["order_id"], orders["occurred_at"],
                                          strict=True)]),
        "promotions": tables["promotions"].assign(first_purchase_only=False),
        **{name: world.empty(name)
           for name in ("dispute_openings", "dispute_resolutions", "victim_reports")},
    }
    labels = world.adjudicate(inputs, horizon_days=0, observed_until="2100-01-01", **rules)
    labels = labels[labels["basis"] == "never_pay"]
    return {int(o): _seconds(at)
            for o, at in zip(labels["order_id"], labels["label_known_at"], strict=True)}


class Oracle:
    """Context rows for decisions on ``tables`` under ``state`` (approve-all when None:
    every processor-approved order let through at its checkout)."""

    def __init__(self, tables, state: asof.PolicyState | None = None,
                 terms: ledger.ProductTerms | None = None) -> None:
        self.terms = terms or ledger.ProductTerms.from_config()
        t = {name: _records(tables[name], name) for name in asof.INPUT_TABLES if name in tables}
        self.accounts = {a.user_id: a for a in t["accounts"]}
        self.merchants = {m.merchant_id: m for m in t["merchants"]}
        self.cards = {c.card_id: c for c in t["cards"]}
        self.promotions = {p.promo_id: p for p in t["promotions"]}
        self.attempts = t["order_attempts"]
        self.orders = {a.order_id: a for a in self.attempts}
        self.by_user = _group(self.attempts, "user_id")
        self.by_device = _group(self.attempts, "device_id")
        self.by_card = _group(self.attempts, "card_id")
        self.by_ship = _group(self.attempts, "ship_address_id")
        self.events = _group(t["account_events"], "user_id")
        self.events_on_device = _group(t["account_events"], "device_id")
        self.device_links = _group(t["device_links"], "user_id")
        self.address_links = _group(t["address_links"], "user_id")
        self.links_to_address = _group(t["address_links"], "address_id")
        self.plans = _group(t["plans"], "order_id")
        self.schedule = _group(t["installment_schedule"], "plan_id")
        self.payments = _group(t["payment_attempts"], "plan_id")
        self.reversals = _group(t["payment_reversals"], "payment_event_id")
        self.writeoffs = _group(t["plan_writeoffs"], "plan_id")
        self.fulfilments = _group(t["fulfilments"], "order_id")
        self.deliveries = _group(t["deliveries"], "order_id")
        self.disputes = _group(t["dispute_openings"], "order_id")
        self.resolutions = _group(t["dispute_resolutions"], "dispute_id")
        self.reports = _group(t["victim_reports"], "order_id")
        self.holdings = {user: self._holdings(user) for user in self.accounts}

        # what the policy did; rows for orders that are not in the tables do not exist yet
        self.approved_at: dict[int, int] = {}
        self.voided_at = defaultdict(list)
        self.cancelled_at = defaultdict(list)
        self.paused_at = defaultdict(list)
        self.resumed_at = defaultdict(list)
        self.blocked_at = defaultdict(list)
        if state is None:
            for a in self.attempts:
                if a.processor_result == "approved":
                    self.approved_at[a.order_id] = a.known_at
        else:
            for row in _records(state.approved):
                if row.order_id in self.orders:
                    previous = self.approved_at.get(row.order_id, row.approved_at)
                    self.approved_at[row.order_id] = min(previous, row.approved_at)
            for row in _records(state.voided):
                self.voided_at[row.order_id].append(row.at)
            for row in _records(state.held):
                if row.order_id not in self.orders:
                    continue
                # a held order went through at its checkout
                checkout = self.orders[row.order_id].known_at
                self.approved_at[row.order_id] = min(
                    self.approved_at.get(row.order_id, checkout), checkout)
                if not row.before_shipment:
                    continue  # a hold after shipment pauses nothing
                self.paused_at[row.order_id].append(row.held_at)
                if row.released_at is not None and row.outcome == "cleared":
                    self.resumed_at[row.order_id].append(row.released_at)
                elif row.released_at is not None and row.outcome in ("cancelled", "declined"):
                    self.cancelled_at[row.order_id].append(row.released_at)
            for row in _records(state.blocked):
                self.blocked_at[row.user_id].append(row.at)
        stops = {o: min(self.voided_at.get(o, []) + self.cancelled_at.get(o, []))
                 for o in set(self.voided_at) | set(self.cancelled_at)
                 if self.voided_at.get(o) or self.cancelled_at.get(o)}
        self.never_pay = never_pay_determinations(tables, self.approved_at, stops)

    # -------------------------------------------------------------------- emails
    def _holdings(self, user: int) -> list[tuple[str, int, int | None]]:
        """(root, since, until): the signup address from the account's creation, then each
        email change's address from when it is known (changes in one second in event id
        order) until the next one."""
        account = self.accounts[user]
        changes = sorted((e for e in self.events[user] if e.kind == "email_change"),
                         key=lambda e: e.key)
        starts = [(account.email, account.created_at)] + [(e.email, e.known_at)
                                                          for e in changes]
        return [(asof.normalize_email(email), since,
                 starts[i + 1][1] if i + 1 < len(starts) else None)
                for i, (email, since) in enumerate(starts)]

    def email_at(self, user: int, at: int) -> str | None:
        """The root of the address the account holds at ``at`` (None before it exists)."""
        held = None
        for root, since, _ in self.holdings[user]:
            if since <= at:
                held = root
        return held

    def shared_since(self, user: int, other: int) -> int | None:
        """When two accounts first shared a device (overlapping links) or a normalized
        email (overlapping holdings)."""
        starts = []
        for a in self.device_links[user]:
            for b in self.device_links[other]:
                if a.device_id == b.device_id:
                    starts.append(_overlap_start(a.created_at, a.removed_at,
                                                 b.created_at, b.removed_at))
        for root, since, until in self.holdings[user]:
            for other_root, other_since, other_until in self.holdings[other]:
                if root == other_root:
                    starts.append(_overlap_start(since, until, other_since, other_until))
        return min((s for s in starts if s is not None), default=None)

    # --------------------------------------------------------------------- rows
    def frame(self, decisions: pd.DataFrame | None = None) -> pd.DataFrame:
        """Rows in the layout of core.asof.build_context; by default every attempt at its
        checkout, in the event order."""
        if decisions is None:
            pairs = [(a.order_id, a.known_at) for a in sorted(self.attempts, key=lambda a: a.key)]
        else:
            pairs = [(int(o), _seconds(at)) for o, at in
                     zip(decisions["order_id"], decisions["decision_at"], strict=True)]
        rows = [self.row(order_id, at) for order_id, at in pairs]
        out = pd.DataFrame(rows, columns=[*asof.KEY_COLUMNS, *asof.COLUMN_NAMES])
        out["decision_at"] = np.array([at for _, at in pairs], dtype="datetime64[s]")
        return out.astype({"order_id": "int64", "user_id": "int64", "merchant_id": "int64",
                           **{c.name: c.dtype for c in asof.COLUMNS}})

    def row(self, order_id: int, decision_at: int) -> dict:
        o = self.orders[order_id]
        if decision_at < o.known_at:
            raise ValueError("a decision before the checkout")
        at_order = View(o.known_at, True, o.key)
        at_decision = View(decision_at, decision_at == o.known_at, o.key)
        return {"order_id": o.order_id, "user_id": o.user_id, "merchant_id": o.merchant_id,
                **self.as_placed(o, at_order), **self.linkage(o, at_decision),
                **self.outcomes(o, at_decision)}

    # ------------------------------------------------- the attempt as it was placed
    def as_placed(self, o, v: View) -> dict:
        t = o.known_at
        account = self.accounts[o.user_id]
        merchant = self.merchants[o.merchant_id]
        c = {}
        c["amount_cents"] = o.amount_cents
        down = ledger.split_principal(o.amount_cents - o.promo_discount_cents, self.terms)[0]
        c["order_exposure_cents"] = (ledger.settlement_cents(o.amount_cents,
                                                             o.promo_discount_cents, self.terms)
                                     + o.promo_discount_cents - down)
        c["account_age_days"] = (t - account.created_at) / DAY
        c["merchant_age_days"] = (t - merchant.created_at) / DAY
        c["merchant_fulfilment_median_hours"] = merchant.fulfilment_median_hours
        c["avs_mismatch"] = int(o.avs_result == "N")
        c["cvv_mismatch"] = int(o.cvv_result == "N")
        c["bin_ip_country_mismatch"] = int(self.cards[o.card_id].bin_country != o.ip_country)
        c["ip_country_not_home"] = int(account.home_country != o.ip_country)
        c["night_order"] = int(pd.Timestamp(o.occurred_at, unit="s").hour < 6)

        earlier = [a.amount_cents for a in self.attempts if v.event(a)
                   and a.processor_result == "approved"
                   and self.merchants[a.merchant_id].category == merchant.category]
        for name, q in (("amount_over_category_median", 0.5),
                        ("amount_over_category_p95", 0.95)):
            enough = len(earlier) >= asof.CATEGORY_MIN_SAMPLE
            c[name] = o.amount_cents / _quantile(earlier, q) if enough else math.nan

        def recent(rows, seconds, current):
            return [a for a in rows if v.event(a, current) and a.known_at >= t - seconds]

        mine = self.by_user[o.user_id]
        c["attempts_user_1h"] = len(recent(mine, HOUR, True))
        c["attempts_user_24h"] = len(recent(mine, DAY, True))
        c["attempts_user_7d"] = len(recent(mine, 7 * DAY, True))
        c["amount_attempted_user_24h"] = sum(a.amount_cents for a in recent(mine, DAY, True))
        c["amount_attempted_user_7d"] = sum(a.amount_cents for a in recent(mine, 7 * DAY, True))
        c["attempts_device_24h"] = len(recent(self.by_device[o.device_id], DAY, True))
        c["processor_declines_card_24h"] = sum(
            a.processor_result == "declined" for a in recent(self.by_card[o.card_id], DAY, False))
        c["processor_declines_device_24h"] = sum(
            a.processor_result == "declined"
            for a in recent(self.by_device[o.device_id], DAY, False))
        before = [a for a in mine if v.event(a)]
        c["is_first_attempt_user"] = int(not before)
        previous = max(before, key=lambda a: a.key, default=None)
        c["geo_kmh_from_previous_attempt"] = 0.0
        if previous is not None and previous.ip_country != o.ip_country:
            gap = (t - previous.known_at) / HOUR
            if gap < 12:
                c["geo_kmh_from_previous_attempt"] = (_km(previous.ip_country, o.ip_country)
                                                      / max(gap, 0.02))

        linked = [link.created_at for link in self.device_links[o.user_id]
                  if link.device_id == o.device_id and v.entity(link.created_at)]
        c["device_link_age_hours"] = (t - min(linked)) / HOUR if linked else math.nan
        used = recent(mine, 30 * DAY, True) + [e for e in self.events[o.user_id]
                                               if v.event(e) and e.known_at >= t - 30 * DAY]
        c["distinct_devices_user_30d"] = len({x.device_id for x in used})
        c["card_link_age_hours"] = (t - self.cards[o.card_id].created_at) / HOUR
        c["card_first_use_age_hours"] = (t - min(
            a.known_at for a in mine if a.card_id == o.card_id and v.event(a, True))) / HOUR
        c["ship_address_first_use_age_hours"] = (t - min(
            a.known_at for a in mine
            if a.ship_address_id == o.ship_address_id and v.event(a, True))) / HOUR
        address_links = self.address_links[o.user_id]
        linked = [link.created_at for link in address_links
                  if link.address_id == o.ship_address_id and v.entity(link.created_at)]
        c["ship_address_link_age_hours"] = (t - min(linked)) / HOUR if linked else math.nan
        homes = [link for link in address_links if link.role == "home"
                 and link.created_at <= t and (link.removed_at is None or t < link.removed_at)]
        c["ship_to_home"] = int(any(h.address_id == o.ship_address_id for h in homes))
        c["home_address_age_days"] = ((t - max(h.created_at for h in homes)) / DAY
                                      if homes else math.nan)
        c["distinct_ship_addresses_user_ever"] = len(
            {a.ship_address_id for a in mine if v.event(a, True)})

        for name, kind in (("hours_since_password_change", "password_change"),
                           ("hours_since_password_reset", "password_reset"),
                           ("hours_since_email_change", "email_change"),
                           ("hours_since_phone_change", "phone_change")):
            last = max((e.known_at for e in self.events[o.user_id]
                        if e.kind == kind and v.event(e)), default=None)
            c[name] = min((t - last) / HOUR, HOURS_CAP) if last is not None else HOURS_CAP
        c["hours_since_credential_change"] = min(
            c["hours_since_password_change"], c["hours_since_password_reset"],
            c["hours_since_email_change"])
        return c

    # ---------------------------------------------------- linkage at the decision
    def linkage(self, o, v: View) -> dict:
        low = v.at - 30 * DAY

        def accounts(rows) -> set[int]:
            return {x.user_id for x in rows if v.event(x, True) and x.known_at >= low}

        on_device = self.by_device[o.device_id] + self.events_on_device[o.device_id]
        ever = {a.user_id for a in self.by_ship[o.ship_address_id] if v.event(a, True)} | {
            link.user_id for link in self.links_to_address[o.ship_address_id]
            if v.entity(link.created_at)}
        root = self.email_at(o.user_id, v.at)
        domain = root.rpartition("@")[2]  # of the normalized address: googlemail is gmail
        return {
            "accounts_on_device_30d": len(accounts(on_device)),
            "accounts_on_address_30d": len(accounts(self.by_ship[o.ship_address_id])),
            "accounts_on_address_ever": len(ever),
            "email_domain_class": 2 if domain in asof.DISPOSABLE_DOMAINS
            else 0 if domain in asof.COMMON_DOMAINS else 1,
            "email_root_other_accounts": sum(
                1 for user in self.accounts
                if user != o.user_id and self.email_at(user, v.at) == root),
        }

    # ----------------------------------------------------------------- outcomes
    def let_through(self, a, v: View) -> bool:
        """Let through as far as the decision knows (a later void does not undo it)."""
        at = self.approved_at.get(a.order_id)
        if at is None:
            return False
        return v.event(a) if at == a.known_at else v.derived(at)  # at checkout: in place

    def resumed(self, a, v: View) -> int | None:
        """When a hold before shipment released the order, if the decision knows it."""
        return next((x for x in self.resumed_at[a.order_id] if v.derived(x)), None)

    def live(self, a, v: View) -> bool:
        """Let through as far as the decision knows, not paused by a hold placed before
        shipment that has not been released, and neither voided nor cancelled."""
        paused = any(v.derived(x) for x in self.paused_at[a.order_id]) and (
            self.resumed(a, v) is None)
        stopped = any(v.derived(x) for x in self.voided_at[a.order_id]
                      + self.cancelled_at[a.order_id])
        return self.let_through(a, v) and not paused and not stopped

    def approved_last(self, a, v: View) -> int:
        """When the order was last approved as the decision knows: at the release of
        its hold, or at checkout."""
        released = self.resumed(a, v)
        return self.approved_at[a.order_id] if released is None else released

    def outcomes(self, o, v: View) -> dict:
        user = o.user_id
        others = [a for a in self.by_user[user] if a.order_id != o.order_id]
        live = [a for a in others if self.live(a, v)]
        through = [a for a in others if self.let_through(a, v)]
        c = {"approved_orders_user_ever": len(live),
             "approved_orders_user_24h": sum(self.approved_last(a, v) >= v.at - DAY
                                             for a in live),
             "promo_redemptions_user": sum(a.promo_id is not None for a in live)}

        due = paid = failed = balance = 0
        for a in live:
            for plan in self.plans[a.order_id]:
                payments = [p for p in self.payments[plan.plan_id] if v.event(p)]

                def standing(rows):
                    good = [p for p in rows if p.result == "success"]
                    returned = [r for p in good for r in self.reversals[p.event_id]
                                if v.event(r)]
                    return sum(p.amount_cents for p in good) - sum(
                        r.amount_cents for r in returned)

                for entry in self.schedule[plan.plan_id]:
                    if entry.seq < 1 or not v.derived(entry.due_at):
                        continue
                    on_entry = [p for p in payments if p.seq == entry.seq]
                    in_full = standing(on_entry) >= entry.amount_cents
                    due += 1
                    paid += in_full
                    failed += not in_full and any(p.result == "failed" for p in on_entry)
                if not any(v.event(w) for w in self.writeoffs[plan.plan_id]):
                    balance += plan.principal_cents - standing(payments)
        c.update(installments_due_user=due, installments_paid_user=paid,
                 installments_failed_user=failed, open_balance_user_cents=balance,
                 installments_paid_share_user=paid / due if due else 0.0)

        disputes = [(a, d) for a in through for d in self.disputes[a.order_id]]
        resolved = [(a, d, r) for a, d in disputes for r in self.resolutions[d.dispute_id]
                    if v.event(r)]
        on_card = [d for a in self.by_card[o.card_id]
                   if a.order_id != o.order_id and self.let_through(a, v)
                   for d in self.disputes[a.order_id]]
        c["unauthorized_disputes_on_card"] = sum(
            d.reason == "unauthorized" and v.event(d) for d in on_card)
        c["unauthorized_disputes_lost_user"] = sum(
            d.reason == "unauthorized" and r.outcome == "lost" for _, d, r in resolved)
        c["victim_reports_user"] = sum(
            r.user_id == user and v.event(r) for a in through for r in self.reports[a.order_id])
        c["inr_disputes_opened_user"] = sum(d.reason == INR and v.event(d) for _, d in disputes)
        c["inr_claims_rejected_user"] = sum(
            d.reason == INR and r.outcome == "won"
            and any(v.event(x) for x in self.deliveries[a.order_id]) for a, d, r in resolved)
        c["never_pay_determined_user"] = int(any(
            v.derived(self.never_pay.get(a.order_id)) for a in through))
        c["promo_uses_linked_accounts"] = self.promo_uses(o, v)

        c["shipped_at_decision"] = int(any(v.event(f) for f in self.fulfilments[o.order_id]))
        c["cancelled_at_decision"] = int(any(
            v.derived(x) for x in self.voided_at[o.order_id] + self.cancelled_at[o.order_id]))
        c["account_blocked"] = int(any(v.derived(x) for x in self.blocked_at[user]))
        near = (self.by_device[o.device_id] + self.events_on_device[o.device_id]
                + self.by_ship[o.ship_address_id])
        linked = {x.user_id for x in near if v.event(x) and x.known_at >= v.at - 30 * DAY}
        c["linked_account_blocked_30d"] = int(any(
            v.derived(x) for other in linked - {user} for x in self.blocked_at[other]))
        return c

    def promo_uses(self, o, v: View) -> int:
        promo = self.promotions.get(o.promo_id)
        if promo is None or not promo.first_purchase_only:
            return 0
        users = {o.user_id}
        for other, orders in self.by_user.items():
            if other in users:
                continue
            since = self.shared_since(o.user_id, other)
            if since is not None and v.entity(since) and any(
                    a.promo_id == o.promo_id and self.live(a, v) for a in orders):
                users.add(other)
        return len(users)
