"""Shared primitives every actor uses, and the assembly of the world's tables.

Legitimate customers, fraudsters and merchants all act through the same
methods: open an account, attach a device, add an address or a card, log in or
change a credential, place an order, pay or miss an installment, have a
payment reversed, ship, deliver, dispute, report a takeover. Each method checks
that what it uses exists at that moment (the account, a device linked to it, an
address and a card on it, an open merchant), so every event is generated from
the account state available at its time; :func:`core.world.validate_world`
checks the assembled world again.

Times are integer epoch seconds until assembly. Every row carries a random
tiebreak drawn from its actor's own random stream; ids are assigned after the
world is sorted by time, with the tiebreak deciding exact ties, so an id
carries the time order and nothing about who created the row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core import ledger, world

DAY = 86_400
TIE_MAX = 2**62


class GenerationError(RuntimeError):
    """A primitive was asked to do something the account state does not allow."""


@dataclass
class Actor:
    """One actor's private random stream (a SeedSequence child keyed by component
    and index), so adding or removing other actors never changes its draws."""

    rng: np.random.Generator

    @classmethod
    def of(cls, seed: int, *key: int) -> Actor:
        return cls(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key)))

    def tie(self) -> int:
        return int(self.rng.integers(1, TIE_MAX))


@dataclass
class Link:
    owner: int
    item: int
    created: int
    removed: int | None = None
    role: str | None = None

    def active(self, t: int) -> bool:
        return self.created <= t and (self.removed is None or t < self.removed)


@dataclass
class Order:
    pk: int
    tie: int
    t: int
    user: int
    merchant: int
    device: int
    card: int
    address: int
    amount: int
    promo: int | None
    discount: int
    ip: str
    ip_country: str
    avs: str
    cvv: str
    approved: bool
    # plan
    principal: int = 0
    schedule: list[tuple[int, int, int]] = field(default_factory=list)  # (seq, due, cents)
    # successful collections: [t, cents, tie, reversed_at]
    payments: list[list[Any]] = field(default_factory=list)
    shipped: int | None = None
    delivered: int | None = None
    # latent truth
    pattern: str | None = None
    episode: int | None = None
    intent: str = "legitimate"
    mimic: set[str] = field(default_factory=set)

    def due(self, seq: int) -> int:
        return self.schedule[seq][1]

    def collected(self, at: int) -> int:
        """Payments standing at ``at`` (collected and not reversed by then)."""
        return sum(cents for paid, cents, _, back in self.payments
                   if paid <= at and not (back is not None and back <= at))


class Builder:
    """Accumulates entities and events; :meth:`tables` assembles the world."""

    def __init__(self, terms: ledger.ProductTerms) -> None:
        self.terms = terms
        self._pk = 0
        self.accounts: dict[int, list[Any]] = {}
        self.devices: dict[int, list[Any]] = {}
        self.addresses: dict[int, list[Any]] = {}
        self.cards: dict[int, list[Any]] = {}  # [user, created, removed, bin, net, last4, tie]
        self.merchants: dict[int, dict[str, Any]] = {}
        self.promotions: dict[int, dict[str, Any]] = {}
        self.device_links: list[Link] = []
        self.address_links: list[Link] = []
        self.user_devices: dict[int, list[Link]] = {}
        self.user_addresses: dict[int, list[Link]] = {}
        self.user_cards: dict[int, list[int]] = {}
        self.user_orders: dict[int, list[Order]] = {}
        self.orders: list[Order] = []
        self.account_events: list[tuple] = []
        self.payment_attempts: list[tuple] = []
        self.reversals: list[tuple] = []
        self.fulfilments: list[tuple] = []
        self.deliveries: list[tuple] = []
        self.openings: list[tuple] = []
        self.resolutions: list[tuple] = []
        self.victim_reports: list[tuple] = []
        self.writeoffs: list[tuple] = []
        self.episodes: dict[int, list[Any]] = {}  # pk -> [pattern, started, ended, tie]
        self.latent_accounts: dict[int, list[Any]] = {}  # user -> [actor, episode, tags]

    def _next(self) -> int:
        self._pk += 1
        return self._pk

    # ============================================================ entities
    def account(self, a: Actor, t: int, email: str, home_country: str, dob_year: int,
                *, actor: str = "legitimate", episode: int | None = None,
                tags: tuple[str, ...] = ()) -> int:
        pk = self._next()
        self.accounts[pk] = [t, email, email.split("@")[1].lower(), home_country, dob_year,
                             a.tie()]
        self.user_devices[pk], self.user_addresses[pk] = [], []
        self.user_cards[pk], self.user_orders[pk] = [], []
        self.latent_accounts[pk] = [actor, episode, set(tags)]
        return pk

    def tag(self, user: int, *tags: str) -> None:
        self.latent_accounts[user][2].update(tags)

    def signup(self, user: int) -> int:
        return self.accounts[user][0]

    def device(self, a: Actor, t: int, ua: str, fingerprint: str) -> int:
        pk = self._next()
        self.devices[pk] = [t, fingerprint, ua, a.tie()]
        return pk

    def link_device(self, user: int, device: int, t: int) -> Link:
        self._exists(user, t)
        if self.devices[device][0] > t:
            raise GenerationError(f"device {device} linked before it was first seen")
        link = Link(user, device, t)
        self.device_links.append(link)
        self.user_devices[user].append(link)
        return link

    def address(self, a: Actor, t: int, line_hash: str, city: str, region: str,
                country: str) -> int:
        pk = self._next()
        self.addresses[pk] = [t, line_hash, city, region, country, a.tie()]
        return pk

    def link_address(self, user: int, address: int, t: int, role: str) -> Link:
        self._exists(user, t)
        if self.addresses[address][0] > t:
            raise GenerationError(f"address {address} linked before it existed")
        link = Link(user, address, t, role=role)
        self.address_links.append(link)
        self.user_addresses[user].append(link)
        return link

    def unlink(self, link: Link, t: int) -> None:
        if link.removed is not None or t < link.created:
            raise GenerationError("a link ends once, after it starts")
        link.removed = t

    def card(self, a: Actor, user: int, t: int, bin_country: str, network: str,
             last4: str) -> int:
        self._exists(user, t)
        pk = self._next()
        self.cards[pk] = [user, t, None, bin_country, network, last4, a.tie()]
        self.user_cards[user].append(pk)
        return pk

    def remove_card(self, card: int, t: int) -> None:
        row = self.cards[card]
        if row[2] is not None or t < row[1]:
            raise GenerationError("a card is removed once, after it is added")
        row[2] = t

    def merchant(self, a: Actor, t: int, name: str, category: str, risk_tier: int,
                 fulfilment_median_hours: float, closed_at: int | None = None) -> int:
        pk = self._next()
        self.merchants[pk] = {"created": t, "name": name, "category": category,
                              "risk_tier": risk_tier, "median": fulfilment_median_hours,
                              "closed": closed_at, "tie": a.tie(), "bustout_from": None}
        return pk

    def promotion(self, a: Actor, code: str, discount_bps: int, first_purchase_only: bool,
                  valid_from: int, valid_to: int) -> int:
        pk = self._next()
        self.promotions[pk] = {"code": code, "bps": discount_bps, "first": first_purchase_only,
                               "from": valid_from, "to": valid_to, "tie": a.tie()}
        return pk

    def episode(self, a: Actor, pattern: str, started: int) -> int:
        pk = self._next()
        self.episodes[pk] = [pattern, started, started, a.tie()]
        return pk

    # ---------------------------------------------------------- state
    def _exists(self, user: int, t: int) -> None:
        if self.accounts[user][0] > t:
            raise GenerationError(f"account {user} used before it was opened")

    def devices_at(self, user: int, t: int) -> list[int]:
        return [link.item for link in self.user_devices[user] if link.active(t)]

    def addresses_at(self, user: int, t: int, role: str | None = None) -> list[int]:
        return [link.item for link in self.user_addresses[user]
                if link.active(t) and (role is None or link.role == role)]

    def cards_at(self, user: int, t: int) -> list[int]:
        out = []
        for card in self.user_cards[user]:
            _, created, removed = self.cards[card][:3]
            if created <= t and (removed is None or t < removed):
                out.append(card)
        return out

    def merchant_open(self, merchant: int, t: int) -> bool:
        m = self.merchants[merchant]
        return m["created"] <= t and (m["closed"] is None or t < m["closed"])

    def approved_before(self, user: int, t: int) -> bool:
        return any(o.approved and o.t < t for o in self.user_orders[user])

    # ============================================================ events
    def account_event(self, a: Actor, user: int, t: int, kind: str, device: int, ip: str,
                      ip_country: str, email: str | None = None) -> None:
        if device not in self.devices_at(user, t):
            raise GenerationError(f"account event on device {device} not linked to {user}")
        self.account_events.append((a.tie(), t, t, user, kind, device, ip, ip_country, email))

    def login_before(self, a: Actor, user: int, t: int, device: int, ip: str, ip_country: str,
                     p: float) -> None:
        """With probability ``p``, a login on the ordering device minutes before checkout."""
        if a.rng.random() < p:
            login = t - int(a.rng.uniform(60, 600))
            if device in self.devices_at(user, login):
                self.account_event(a, user, login, "login", device, ip, ip_country)

    def order(self, a: Actor, user: int, merchant: int, t: int, amount: int, *, device: int,
              card: int, address: int, ip: str, ip_country: str, avs: str = "Y",
              cvv: str = "M", approved: bool = True, promo: int | None = None) -> Order:
        self._exists(user, t)
        if device not in self.devices_at(user, t):
            raise GenerationError(f"order on device {device} not linked to account {user}")
        if address not in self.addresses_at(user, t):
            raise GenerationError(f"order to address {address} not on account {user}")
        if card not in self.cards_at(user, t):
            raise GenerationError(f"order with card {card} not on account {user}")
        if not self.merchant_open(merchant, t):
            raise GenerationError(f"order at merchant {merchant} outside its trading period")
        discount = 0
        if promo is not None:
            p = self.promotions[promo]
            if not p["from"] <= t < p["to"]:
                raise GenerationError("promotion used outside its validity")
            if p["first"] and self.approved_before(user, t):
                raise GenerationError("first-purchase promotion on a later order")
            discount = ledger.round_bps(amount, p["bps"])
        o = Order(self._next(), a.tie(), t, user, merchant, device, card, address, amount,
                  promo, discount, ip, ip_country, avs, cvv, approved)
        self.orders.append(o)
        self.user_orders[user].append(o)
        if approved:
            o.principal = amount - discount
            parts = ledger.split_principal(o.principal, self.terms)
            step = self.terms.installment_interval_days * DAY
            o.schedule = [(seq, t + seq * step, cents) for seq, cents in enumerate(parts)]
            self.pay(a, o, 0, t)  # the checkout payment
        return o

    def pay(self, a: Actor, o: Order, seq: int, t: int, *, success: bool = True,
            attempt_no: int = 1) -> list[Any] | None:
        if not o.approved or t < o.t:
            raise GenerationError("payment on an unapproved order or before it")
        tie = a.tie()
        cents = o.schedule[seq][2]
        self.payment_attempts.append((tie, t, t, o.pk, seq, attempt_no, cents,
                                      "success" if success else "failed"))
        if not success:
            return None
        payment = [t, cents, tie, None]
        o.payments.append(payment)
        return payment

    def reverse(self, a: Actor, o: Order, payment: list[Any], t: int,
                reason: str = "bank_return") -> None:
        if t < payment[0] or payment[3] is not None:
            raise GenerationError("a payment is reversed once, after it is collected")
        payment[3] = t
        self.reversals.append((a.tie(), t, t, payment[2], o.pk, payment[1], reason))

    def ship(self, a: Actor, o: Order, t: int) -> None:
        if not o.approved or t < o.t or o.shipped is not None:
            raise GenerationError("an approved order ships once, after it is placed")
        o.shipped = t
        self.fulfilments.append((a.tie(), t, t, o.pk))

    def deliver(self, a: Actor, o: Order, t: int) -> None:
        if o.shipped is None or t < o.shipped or o.delivered is not None:
            raise GenerationError("delivery needs one earlier shipment")
        o.delivered = t
        self.deliveries.append((a.tie(), t, t, o.pk))

    def dispute(self, a: Actor, o: Order, reason: str, filed: int, notified: int,
                outcome: str | None = None, decided: int | None = None,
                decided_known: int | None = None) -> None:
        if not o.approved or filed < o.t or notified < filed:
            raise GenerationError("dispute filed before its order or known before filing")
        amount = o.collected(notified)
        if amount <= 0:
            return  # nothing collected to dispute
        dispute = self._next()
        self.openings.append((a.tie(), filed, notified, dispute, o.pk, reason, amount))
        if outcome is not None:
            if decided is None or decided_known is None or decided < filed \
                    or decided_known < max(decided, notified):
                raise GenerationError("a dispute is resolved after it is filed and known")
            self.resolutions.append((a.tie(), decided, decided_known, dispute, outcome))

    def victim_report(self, a: Actor, o: Order, t: int) -> None:
        if t < o.t:
            raise GenerationError("an order is disowned after it is placed")
        self.victim_reports.append((a.tie(), t, t, o.user, o.pk))

    def write_off(self, a: Actor, o: Order) -> int:
        """Write the plan off ``writeoff_after_days`` after its last due date, if unpaid."""
        at = o.schedule[-1][1] + self.terms.writeoff_after_days * DAY
        outstanding = o.principal - o.collected(at)
        if outstanding > 0:
            self.writeoffs.append((a.tie(), at, at, o.pk, outstanding))
        return outstanding

    def writeoff_time(self, o: Order) -> int:
        return o.schedule[-1][1] + self.terms.writeoff_after_days * DAY

    # ============================================================ assembly
    def tables(self, observed_until: int) -> dict[str, pd.DataFrame]:
        """Every table of the world contract (cash events and labels excluded)."""
        ids: dict[int, int] = {}
        out: dict[str, pd.DataFrame] = {}

        def assign(rows: dict[int, list[Any]], created: int, tie: int) -> list[int]:
            keys = sorted(rows, key=lambda pk: (rows[pk][created], rows[pk][tie]))
            for n, pk in enumerate(keys, start=1):
                ids[pk] = n
            return keys

        keys = assign(self.accounts, 0, 5)
        acc = [self.accounts[k] for k in keys]
        out["accounts"] = pd.DataFrame({
            "user_id": [ids[k] for k in keys], "created_at": _ts([r[0] for r in acc]),
            "email": [r[1] for r in acc], "email_domain": [r[2] for r in acc],
            "home_country": [r[3] for r in acc], "dob_year": [r[4] for r in acc],
        })
        merchant_rows = {pk: [m["created"], m["tie"]] for pk, m in self.merchants.items()}
        keys = assign(merchant_rows, 0, 1)
        ms = [self.merchants[k] for k in keys]
        out["merchants"] = pd.DataFrame({
            "merchant_id": [ids[k] for k in keys], "created_at": _ts([m["created"] for m in ms]),
            "name": [m["name"] for m in ms], "category": [m["category"] for m in ms],
            "risk_tier": [m["risk_tier"] for m in ms],
            "fulfilment_median_hours": [float(m["median"]) for m in ms],
            "closed_at": _ts([m["closed"] for m in ms]),
        })
        keys = assign(self.devices, 0, 3)
        dv = [self.devices[k] for k in keys]
        out["devices"] = pd.DataFrame({
            "device_id": [ids[k] for k in keys], "created_at": _ts([r[0] for r in dv]),
            "fingerprint": [r[1] for r in dv], "ua_family": [r[2] for r in dv],
        })
        keys = assign(self.addresses, 0, 5)
        ad = [self.addresses[k] for k in keys]
        out["addresses"] = pd.DataFrame({
            "address_id": [ids[k] for k in keys], "created_at": _ts([r[0] for r in ad]),
            "line_hash": [r[1] for r in ad], "city": [r[2] for r in ad],
            "region": [r[3] for r in ad], "country": [r[4] for r in ad],
        })
        card_rows = {pk: [row[1], row[6]] for pk, row in self.cards.items()}
        keys = assign(card_rows, 0, 1)
        cd = [self.cards[k] for k in keys]
        out["cards"] = pd.DataFrame({
            "card_id": [ids[k] for k in keys], "user_id": [ids[r[0]] for r in cd],
            "created_at": _ts([r[1] for r in cd]), "removed_at": _ts([r[2] for r in cd]),
            "bin_country": [r[3] for r in cd], "network": [r[4] for r in cd],
            "last4": [r[5] for r in cd],
        })
        promo_rows = {pk: [p["from"], p["tie"]] for pk, p in self.promotions.items()}
        keys = assign(promo_rows, 0, 1)
        ps = [self.promotions[k] for k in keys]
        out["promotions"] = pd.DataFrame({
            "promo_id": [ids[k] for k in keys], "code": [p["code"] for p in ps],
            "discount_bps": [p["bps"] for p in ps],
            "first_purchase_only": [p["first"] for p in ps],
            "valid_from": _ts([p["from"] for p in ps]), "valid_to": _ts([p["to"] for p in ps]),
        })
        out["device_links"] = pd.DataFrame({
            "user_id": [ids[x.owner] for x in self.device_links],
            "device_id": [ids[x.item] for x in self.device_links],
            "created_at": _ts([x.created for x in self.device_links]),
            "removed_at": _ts([x.removed for x in self.device_links]),
        })
        out["address_links"] = pd.DataFrame({
            "user_id": [ids[x.owner] for x in self.address_links],
            "address_id": [ids[x.item] for x in self.address_links],
            "created_at": _ts([x.created for x in self.address_links]),
            "removed_at": _ts([x.removed for x in self.address_links]),
            "role": [x.role for x in self.address_links],
        })

        orders = sorted(self.orders, key=lambda o: (o.t, o.tie))
        for n, o in enumerate(orders, start=1):
            ids[o.pk] = n
        out["order_attempts"] = pd.DataFrame({
            "event_id": [o.tie for o in orders], "occurred_at": _ts([o.t for o in orders]),
            "known_at": _ts([o.t for o in orders]), "order_id": [ids[o.pk] for o in orders],
            "user_id": [ids[o.user] for o in orders],
            "merchant_id": [ids[o.merchant] for o in orders],
            "device_id": [ids[o.device] for o in orders], "card_id": [ids[o.card] for o in orders],
            "ship_address_id": [ids[o.address] for o in orders],
            "amount_cents": [o.amount for o in orders],
            "promo_id": pd.array([None if o.promo is None else ids[o.promo] for o in orders],
                                 dtype="Int64"),
            "promo_discount_cents": [o.discount for o in orders],
            "ip": [o.ip for o in orders], "ip_country": [o.ip_country for o in orders],
            "avs_result": [o.avs for o in orders], "cvv_result": [o.cvv for o in orders],
            "processor_result": ["approved" if o.approved else "declined" for o in orders],
        })
        approved = [o for o in orders if o.approved]
        plan_of = {o.pk: n for n, o in enumerate(approved, start=1)}
        out["plans"] = pd.DataFrame({
            "plan_id": [plan_of[o.pk] for o in approved], "order_id": [ids[o.pk] for o in approved],
            "created_at": _ts([o.t for o in approved]),
            "principal_cents": [o.principal for o in approved],
            "down_payment_cents": [o.schedule[0][2] for o in approved],
            "n_installments": [len(o.schedule) - 1 for o in approved],
        })
        sched = [(plan_of[o.pk], seq, due, cents)
                 for o in approved for seq, due, cents in o.schedule]
        out["installment_schedule"] = pd.DataFrame({
            "plan_id": [s[0] for s in sched], "seq": [s[1] for s in sched],
            "due_at": _ts([s[2] for s in sched]), "amount_cents": [s[3] for s in sched],
        })

        def events(rows: list[tuple], columns: list[str], mapper) -> pd.DataFrame:
            kept = [r for r in rows if r[2] <= observed_until]
            frame = pd.DataFrame({
                "event_id": [r[0] for r in kept],
                "occurred_at": _ts([r[1] for r in kept]),
                "known_at": _ts([r[2] for r in kept]),
            })
            for name, values in mapper(kept).items():
                frame[name] = values
            return frame[["event_id", "occurred_at", "known_at", *columns]]

        out["account_events"] = events(self.account_events, [
            "user_id", "kind", "device_id", "ip", "ip_country", "email"],
            lambda rs: {"user_id": [ids[r[3]] for r in rs], "kind": [r[4] for r in rs],
                        "device_id": [ids[r[5]] for r in rs], "ip": [r[6] for r in rs],
                        "ip_country": [r[7] for r in rs], "email": [r[8] for r in rs]})
        out["payment_attempts"] = events(self.payment_attempts, [
            "plan_id", "seq", "attempt_no", "amount_cents", "result"],
            lambda rs: {"plan_id": [plan_of[r[3]] for r in rs], "seq": [r[4] for r in rs],
                        "attempt_no": [r[5] for r in rs], "amount_cents": [r[6] for r in rs],
                        "result": [r[7] for r in rs]})
        out["payment_reversals"] = events(self.reversals, [
            "payment_event_id", "plan_id", "amount_cents", "reason"],
            lambda rs: {"payment_event_id": [r[3] for r in rs],
                        "plan_id": [plan_of[r[4]] for r in rs],
                        "amount_cents": [r[5] for r in rs], "reason": [r[6] for r in rs]})
        for name, rows in (("fulfilments", self.fulfilments), ("deliveries", self.deliveries)):
            out[name] = events(rows, ["order_id"], lambda rs: {"order_id": [ids[r[3]] for r in rs]})
        openings = sorted((r for r in self.openings if r[2] <= observed_until),
                          key=lambda r: (r[2], r[0]))
        for n, r in enumerate(openings, start=1):
            ids[r[3]] = n
        out["dispute_openings"] = events(openings, [
            "dispute_id", "order_id", "reason", "amount_cents"],
            lambda rs: {"dispute_id": [ids[r[3]] for r in rs], "order_id": [ids[r[4]] for r in rs],
                        "reason": [r[5] for r in rs], "amount_cents": [r[6] for r in rs]})
        resolutions = [r for r in self.resolutions if r[3] in ids]
        out["dispute_resolutions"] = events(resolutions, ["dispute_id", "outcome"],
                                            lambda rs: {"dispute_id": [ids[r[3]] for r in rs],
                                                        "outcome": [r[4] for r in rs]})
        out["victim_reports"] = events(self.victim_reports, ["user_id", "order_id"],
                                       lambda rs: {"user_id": [ids[r[3]] for r in rs],
                                                   "order_id": [ids[r[4]] for r in rs]})
        out["plan_writeoffs"] = events(self.writeoffs, ["plan_id", "outstanding_cents"],
                                       lambda rs: {"plan_id": [plan_of[r[3]] for r in rs],
                                                   "outstanding_cents": [r[4] for r in rs]})

        # latent truth
        ep_keys = sorted(self.episodes, key=lambda pk: (self.episodes[pk][1], self.episodes[pk][3]))
        for n, pk in enumerate(ep_keys, start=1):
            ids[pk] = n
        ends: dict[int, int] = {}
        for o in orders:
            if o.episode is not None:
                ends[o.episode] = max(ends.get(o.episode, o.t), o.t)
        out["latent_episodes"] = pd.DataFrame({
            "episode_id": [ids[pk] for pk in ep_keys],
            "pattern_id": [self.episodes[pk][0] for pk in ep_keys],
            "started_at": _ts([self.episodes[pk][1] for pk in ep_keys]),
            "ended_at": _ts([max(ends.get(pk, self.episodes[pk][2]), self.episodes[pk][2])
                             for pk in ep_keys]),
        })
        users = sorted(self.latent_accounts, key=lambda pk: ids[pk])
        rows = [self.latent_accounts[pk] for pk in users]
        out["latent_accounts"] = pd.DataFrame({
            "user_id": [ids[pk] for pk in users], "actor": [r[0] for r in rows],
            "episode_id": pd.array([None if r[1] is None else ids[r[1]] for r in rows],
                                   dtype="Int64"),
            "profile": ["+".join(sorted(r[2])) or None for r in rows],
        })
        out["latent_orders"] = pd.DataFrame({
            "order_id": [ids[o.pk] for o in orders], "pattern_id": [o.pattern for o in orders],
            "episode_id": pd.array([None if o.episode is None else ids[o.episode]
                                    for o in orders], dtype="Int64"),
            "intent": [o.intent for o in orders],
            "mimic": ["+".join(sorted(o.mimic)) or None for o in orders],
        })
        self.ids = ids
        return {name: world.coerce(name, frame) for name, frame in out.items()}


def _ts(values: list[int | None]) -> pd.Series:
    """Epoch seconds (None for missing) as datetime64[s]."""
    array = np.array([np.iinfo(np.int64).min if v is None else v for v in values],
                     dtype=np.int64)
    out = array.astype("datetime64[s]")
    out[array == np.iinfo(np.int64).min] = np.datetime64("NaT", "s")
    return pd.Series(out, dtype="datetime64[s]")
