"""Fraud patterns as parameter sets over the shared primitives.

Each pattern's episodes start evenly over the order horizon (one in each equal
slice of time, at a random day within it), so every evaluation window receives
its share of fresh episodes. Within an episode, every action takes its hour
from the same time-of-day profile as everyone else's, its IP from the same
country blocks and its email from the same generator. Fraudsters act through
the same primitives as customers: an order needs a device linked to the
account, an address and a card on it, all existing at that moment.

The generator models what each actor does, never the label: outcomes come from
behaviour (a cardholder disputes, a victim reports, a never-pay customer stops
paying), and :func:`core.world.adjudicate` decides later which of them the
policy can call fraud.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import numpy as np

from simulator import population as pop
from simulator.builder import DAY, Actor, Builder, Order
from simulator.legit import Customers, Household
from simulator.merchants import Market
from simulator.outcomes import Outcomes
from simulator.timing import HOUR, Clock, session_gap

# Random-stream components (SeedSequence spawn keys); never renumber them.
ATO, STOLEN, SLEEPER_POOL, SYNTH, NEVERPAY, PROMO, INR, MERCH = 10, 11, 12, 13, 14, 15, 16, 17
MERCH_BUYERS = 18
FAMILY_ATO, FAMILY_SLEEPER = 30, 31


@dataclass
class Identity:
    user: int
    device: int
    address: int
    card: int
    ip: str
    ip_country: str


class Fraud:
    def __init__(self, b: Builder, clock: Clock, market: Market, outcomes: Outcomes,
                 customers: Customers, cfg: dict[str, Any], seed: int,
                 promotions: dict[str, int], scale: float) -> None:
        self.b, self.clock, self.market, self.out = b, clock, market, outcomes
        self.customers = customers
        self.f = cfg["fraud"]
        self.pay = cfg["payments"]
        self.seed = seed
        self.promotions = promotions
        self.start, self.end = clock.start, customers.order_end
        self.volume = cfg["volume"]["target_orders"] * scale / 100_000
        self.login_before_order = cfg["customers"]["login_before_order"]
        self.victims: set[int] = set()
        self.sleeper_pool: list[tuple[Actor, int, Identity]] = []
        self._victim_index: list[tuple[int, int, int]] | None = None

    def count(self, pattern: str, minimum: int = 2) -> int:
        return max(minimum, int(round(self.f[pattern]["episodes_per_100k_orders"] * self.volume)))

    def _schedule(self, component: int, n: int, lo: int, hi: int) -> list[tuple[Actor, int]]:
        planner = Actor.of(self.seed, component, 0)
        starts = self.clock.stratified_starts(planner.rng, n, lo, hi)
        return [(Actor.of(self.seed, component, k + 1), t) for k, t in enumerate(starts)]

    # ============================================================ helpers
    def _ip(self, rng: np.random.Generator, country: str, foreign_p: float) -> tuple[str, str]:
        if rng.random() < foreign_p:
            country = pop.pick(rng, pop.FOREIGN_FRAUD_COUNTRIES)
        return pop.ip_address(rng, country), country

    def _identity(self, a: Actor, t: int, *, actor: str, episode: int, tags: tuple[str, ...],
                  disposable_p: float, foreign_ip_p: float, card_bin: str | None = None,
                  device: int | None = None, address: int | None = None,
                  email: str | None = None, country: str | None = None,
                  dob: tuple[int, int] = (1955, 2007), mobile_p: float = 1.0) -> Identity:
        """A new account opened at ``t`` the way a customer opens one: a device and a
        home address at signup, a card within the first minute."""
        b, rng = self.b, a.rng
        country = country or pop.pick(rng, pop.HOME_COUNTRIES)
        if email is None:
            first, last = pop.person_name(rng)
            email = pop.email_address(rng, first, last, disposable=rng.random() < disposable_p)
        user = b.account(a, t, email, country, int(rng.integers(*dob)), actor=actor,
                         episode=episode, tags=tags)
        if device is None:
            device = b.device(a, t, pop.device_ua(rng, mobile=rng.random() < mobile_p),
                              pop.fingerprint(rng))
        b.link_device(user, device, t)
        if address is None:
            city, region = pop.home_city(rng, country)
            address = b.address(a, t, pop.line_hash(rng), city, region, country)
        b.link_address(user, address, t, "home")
        card = b.card(a, user, t + int(rng.uniform(0, 60)), card_bin or country,
                      pop.card_network(rng), pop.last4(rng))
        ip, ip_country = self._ip(rng, country, foreign_ip_p)
        return Identity(user, device, address, card, ip, ip_country)

    def _order(self, a: Actor, who: Identity, t: int, *, pattern: str, episode: int,
               intent: str = "fraud", categories: tuple[str, ...] | None = None,
               multiplier: tuple[float, float] = (1.0, 1.0), avs_fail: float | None = None,
               cvv_fail: float | None = None, approved: bool = True, promo: int | None = None,
               card: int | None = None, address: int | None = None, merchant: int | None = None,
               amount: int | None = None) -> Order:
        rng = a.rng
        if merchant is None:
            merchant = self.market.choose(rng, t, categories)
        if amount is None:
            category = self.b.merchants[merchant]["category"]
            amount = pop.order_amount_cents(rng, category, rng.uniform(*multiplier), sigma=0.4)
        avs_fail = self.pay["avs_fail"] if avs_fail is None else avs_fail
        cvv_fail = self.pay["cvv_fail"] if cvv_fail is None else cvv_fail
        self.b.login_before(a, who.user, t, who.device, who.ip, who.ip_country,
                            self.login_before_order)
        o = self.b.order(a, who.user, merchant, t, amount, device=who.device,
                         card=who.card if card is None else card,
                         address=who.address if address is None else address,
                         ip=who.ip, ip_country=who.ip_country,
                         avs="N" if rng.random() < avs_fail else "Y",
                         cvv="N" if rng.random() < cvv_fail else "M",
                         approved=approved, promo=promo)
        o.pattern, o.episode, o.intent = pattern, episode, intent
        return o

    def _times(self, a: Actor, t: int, n: int, mean_minutes: float) -> list[int]:
        """``n`` times in one sitting starting at ``t``."""
        out = [t]
        for _ in range(n - 1):
            out.append(out[-1] + session_gap(a.rng, mean_minutes))
        return out

    def _in_horizon(self, t: int) -> bool:
        return self.start <= t < self.end

    # ===================================================== account takeover
    def _victim(self, a: Actor, t: int) -> int | None:
        """An established customer with an order before ``t`` and a usable card."""
        if self._victim_index is None:
            members = [(m.signup, m.user, m.active_until)
                       for h in self.customers.base for m in h.members]
            self._victim_index = sorted(members)
        index = self._victim_index
        n = bisect.bisect_right(index, (t - 14 * DAY, 2**62, 0))
        b, rng = self.b, a.rng
        tenured = bisect.bisect_right(index, (t - self.f["P-ATO"]["tenure_days"] * DAY, 2**62, 0))
        for _ in range(400):
            limit = tenured if (tenured and rng.random() < self.f["P-ATO"]["tenured_share"]) else n
            if limit == 0:
                return None
            _, user, until = index[int(rng.integers(0, limit))]
            if user in self.victims or until < t:
                continue
            if not b.approved_before(user, t) or not b.cards_at(user, t) \
                    or not b.devices_at(user, t) or not b.addresses_at(user, t, "home"):
                continue
            return user
        return None

    def account_takeover(self, a: Actor, t: int) -> None:
        b, rng, p = self.b, a.rng, self.f["P-ATO"]
        victim = self._victim(a, t)
        if victim is None:
            return
        self.victims.add(victim)
        b.tag(victim, "takeover_victim")
        country = b.accounts[victim][3]
        ep = b.episode(a, "P-ATO", t)
        device = b.device(a, t, pop.device_ua(rng, mobile=rng.random() < 0.4),
                          pop.fingerprint(rng))
        dev_link = b.link_device(victim, device, t)
        ip, ip_country = self._ip(rng, country, p["foreign_ip"])
        b.account_event(a, victim, t, "login", device, ip, ip_country)
        when = t + session_gap(rng, 4)
        credential = pop.pick(rng, tuple(p["credential_change"].items()))
        new_email = None
        if credential != "none":
            if credential == "email_change":
                first, last = pop.person_name(rng)
                new_email = pop.email_address(rng, first, last, disposable=rng.random() < 0.3)
            b.account_event(a, victim, when, credential, device, ip, ip_country, new_email)
            when += session_gap(rng, 4)
        drop_link = None
        address = b.addresses_at(victim, when, "home")[0]
        use_drop = rng.random() < p["drop_address"]
        card = b.cards_at(victim, when)[-1]
        stolen_card = rng.random() < p["new_card"]
        who = Identity(victim, device, address, card, ip, ip_country)
        orders = []
        n = pop.pick(rng, tuple(p["orders"].items()))
        for k in range(int(n)):
            when += session_gap(rng, p["order_gap_minutes"]) if k else session_gap(rng, 6)
            if not self._in_horizon(when):
                break
            if who.card not in b.cards_at(victim, when) or (
                    drop_link is None and not use_drop
                    and who.address not in b.addresses_at(victim, when)):
                break
            if k == 0 and use_drop:  # the drop address is entered at checkout
                city, region = pop.home_city(rng, country)
                who.address = b.address(a, when, pop.line_hash(rng), city, region, country)
                drop_link = b.link_address(victim, who.address, when, "shipping")
            if stolen_card and k == 0:
                who.card = b.card(a, victim, when, pop.pick(rng, pop.STOLEN_CARD_ISSUERS),
                                  pop.card_network(rng), pop.last4(rng))
            o = self._order(a, who, when, pattern="P-ATO", episode=ep,
                            categories=pop.RESALE_CATEGORIES if rng.random() < p["resale"]
                            else None, multiplier=tuple(p["amount_multiplier"]),
                            avs_fail=p["avs_fail"] if stolen_card else None,
                            cvv_fail=p["cvv_fail"] if stolen_card else None)
            orders.append(o)
        if not orders:
            return
        noticed = pop.pick(rng, tuple(p["noticed"].items()))
        report_at = self.clock.after(rng, orders[-1].t, *p["report_days"])
        for o in orders:
            self.out.fulfil(a, o)
            self.out.collect(a, o, "paid" if noticed == "unnoticed" else "zero_effort")
            if noticed == "report":
                b.victim_report(a, o, report_at)
            elif noticed == "dispute":
                outcome = "lost" if rng.random() < p["dispute_lost"] else "won"
                self.out.dispute(a, o, "unauthorized",
                                 self.clock.after(rng, o.t, *p["dispute_days"]), outcome)
        if noticed == "report":
            secured = report_at + session_gap(rng, 10)
            b.unlink(dev_link, secured)
            if drop_link is not None:
                b.unlink(drop_link, secured)
            if new_email is not None:
                own = [d for d in b.devices_at(victim, secured) if d != device]
                if own:
                    b.account_event(a, victim, secured, "email_change", own[0],
                                    pop.ip_address(rng, country), country,
                                    b.accounts[victim][1])

    # ===================================================== stolen cards
    def stolen_card(self, a: Actor, t: int, *, sleeper_identity: Identity | None = None,
                    episode: int | None = None) -> None:
        b, rng, p = self.b, a.rng, self.f["P-STOLEN"]
        if sleeper_identity is None:
            ep = b.episode(a, "P-STOLEN", t)
            who = self._identity(a, t, actor="fraudster", episode=ep, tags=("stolen_card",),
                                 disposable_p=p["disposable_email"], foreign_ip_p=p["foreign_ip"],
                                 card_bin=pop.pick(rng, pop.STOLEN_CARD_ISSUERS),
                                 mobile_p=p["mobile_device"])
            when = t + session_gap(rng, 4)
        else:
            ep, who = episode, sleeper_identity
            who.ip, who.ip_country = self._ip(rng, b.accounts[who.user][3], p["foreign_ip"])
            when = t
            who.card = b.card(a, who.user, when, pop.pick(rng, pop.STOLEN_CARD_ISSUERS),
                              pop.card_network(rng), pop.last4(rng))
            when += session_gap(rng, 3)
        if rng.random() < p["card_testing"]:
            for _ in range(int(rng.integers(*p["testing_attempts"]))):
                card = b.card(a, who.user, when, pop.pick(rng, pop.STOLEN_CARD_ISSUERS),
                              pop.card_network(rng), pop.last4(rng))
                self._order(a, who, when, pattern="P-STOLEN", episode=ep, card=card,
                            categories=("beauty", "accessories", "toys", "health"),
                            multiplier=(0.3, 0.6), avs_fail=0.5, cvv_fail=0.5, approved=False)
                when += session_gap(rng, 2)
            who.card = b.card(a, who.user, when, pop.pick(rng, pop.STOLEN_CARD_ISSUERS),
                              pop.card_network(rng), pop.last4(rng))
            when += session_gap(rng, 2)
        orders = []
        n = int(pop.pick(rng, tuple(p["orders"].items())))
        for k in range(n):
            if k:
                when = (when + session_gap(rng, 25) if rng.random() < 0.6
                        else self.clock.after(rng, when, 0.3, 2.0))
            if not self._in_horizon(when):
                break
            orders.append(self._order(
                a, who, when, pattern="P-STOLEN", episode=ep,
                categories=pop.RESALE_CATEGORIES if rng.random() < p["resale"] else None,
                multiplier=tuple(p["amount_multiplier"]), avs_fail=p["avs_fail"],
                cvv_fail=p["cvv_fail"]))
        for o in orders:
            self.out.fulfil(a, o)
            self.out.collect(a, o, "zero_effort")
            if rng.random() < p["disputed"]:
                outcome = "lost" if rng.random() < p["dispute_lost"] else "won"
                self.out.dispute(a, o, "unauthorized",
                                 self.clock.after(rng, o.t, *p["dispute_days"]), outcome)

    def sleeper(self, a: Actor, activation: int, *, created: int | None = None,
                activate: bool = True) -> tuple[Identity, int]:
        """An account opened months ahead under a stolen identity, aged, then used."""
        b, rng, p = self.b, a.rng, self.f["P-STOLEN"]
        if created is None:
            created = activation - int(rng.uniform(*p["sleeper_age_days"]) * DAY)
        created = self.clock.at_profile_hour(rng, created)
        ep = b.episode(a, "P-STOLEN", created)
        who = self._identity(a, created, actor="fraudster", episode=ep, tags=("sleeper",),
                             disposable_p=0.0, foreign_ip_p=0.0)
        who.ip = pop.ip_address(rng, b.accounts[who.user][3])
        self._keep_warm(a, who, created, activation)
        if rng.random() < p["sleeper_warm_up"]:
            t = self.clock.after(rng, created, 5, 40)
            if self._in_horizon(t) and t < activation:
                o = self._order(a, who, t, pattern="P-STOLEN", episode=ep,
                                categories=("apparel", "beauty", "home", "accessories"),
                                multiplier=(0.5, 0.9))
                self.out.fulfil(a, o)
                self.out.collect(a, o, "paid")
        if activate:
            self.stolen_card(a, activation, sleeper_identity=who, episode=ep)
        return who, ep

    def _keep_warm(self, a: Actor, who: Identity, lo: int, hi: int) -> None:
        """Occasional logins on an account kept for months, as customers have."""
        rng, rate = a.rng, self.f["P-STOLEN"]["logins_per_year"]
        t = max(lo, self.start)
        while True:
            t = self.clock.after(rng, t, 0.1, 2 * 365 / rate)
            if t >= min(hi, self.end):
                break
            self.b.account_event(a, who.user, t, "login", who.device, who.ip, who.ip_country)

    # ===================================================== synthetic rings
    def synthetic_ring(self, a: Actor, burst: int) -> None:
        b, rng, p = self.b, a.rng, self.f["P-SYNTH"]
        size = int(rng.integers(*p["ring_size"]))
        opened = burst - int(rng.uniform(*p["lead_days"]) * DAY)
        ep = b.episode(a, "P-SYNTH", opened)
        sharing = pop.pick(rng, tuple(p["sharing"].items()))
        country = pop.pick(rng, pop.HOME_COUNTRIES)
        devices = [b.device(a, opened, pop.device_ua(rng), pop.fingerprint(rng))
                   for _ in range(int(rng.integers(1, 3)))] if sharing == "device" else []
        drops = []
        if sharing == "address":
            for _ in range(int(rng.integers(1, 3))):
                city, region = pop.home_city(rng, country)
                drops.append(b.address(a, opened, pop.line_hash(rng), city, region, country))
        root = None
        if rng.random() < p["email_variants"]:
            first, last = pop.person_name(rng)
            root = pop.email_address(rng, first, last, domain="gmail.com")
        ring_ips = [pop.ip_address(rng, country) for _ in range(int(rng.integers(1, 4)))]
        members = []
        for k in range(size):
            t = opened + int(rng.uniform(0, max(1.0, (burst - opened) / DAY - 20)) * DAY)
            t = self.clock.at_profile_hour(rng, t)
            who = self._identity(
                a, t, actor="synthetic_identity", episode=ep, tags=("ring_member",),
                disposable_p=p["disposable_email"], foreign_ip_p=0.0, country=country,
                device=devices[k % len(devices)] if devices else None,
                address=drops[k % len(drops)] if drops else None,
                email=pop.email_variant(rng, root, k) if root else None, dob=(1986, 2003),
                mobile_p=p["mobile_device"])
            members.append(who)
        for who in members:
            who.ip = ring_ips[int(rng.integers(0, len(ring_ips)))] if rng.random() < 0.7 \
                else pop.ip_address(rng, country)
            signup = b.signup(who.user)
            for _ in range(int(rng.integers(*p["warm_up_orders"]))):
                t = self.clock.after(rng, signup, 3, max(4, (burst - signup) / DAY - 4))
                if not self._in_horizon(t) or t >= burst:
                    continue
                o = self._order(a, who, t, pattern="P-SYNTH", episode=ep,
                                multiplier=tuple(p["warm_up_multiplier"]))
                self.out.fulfil(a, o)
                self.out.collect(a, o, "paid")
        for who in members:
            t = burst + int(rng.uniform(0, p["burst_hours"]) * HOUR)
            t = self.clock.at_profile_hour(rng, t) if t - burst > 6 * HOUR else t
            for k in range(int(rng.integers(*p["burst_orders"]))):
                when = t + k * session_gap(rng, 30)
                if not self._in_horizon(when) or when < b.signup(who.user):
                    continue
                o = self._order(a, who, when, pattern="P-SYNTH", episode=ep,
                                categories=pop.RESALE_CATEGORIES,
                                multiplier=tuple(p["burst_multiplier"]))
                self.out.fulfil(a, o)
                self.out.collect(a, o, "zero_effort")

    # ===================================================== first-party never-pay
    def never_pay(self, a: Actor, t: int) -> None:
        b, rng, p = self.b, a.rng, self.f["P-NEVERPAY"]
        mode = pop.pick(rng, tuple(p["modes"].items()))
        ep = b.episode(a, "P-NEVERPAY", t)
        tags = ("never_pay", mode)
        if mode in ("single", "burst"):
            who = self._identity(a, t, actor="fraudster", episode=ep, tags=tags,
                                 disposable_p=p["disposable_email"], foreign_ip_p=0.0)
            who.ip = pop.ip_address(rng, b.accounts[who.user][3])
            n = 1 if mode == "single" else int(rng.integers(*p["burst_orders"]))
            first = t + session_gap(rng, 8) if rng.random() < p["order_at_signup"] \
                else self.clock.after(rng, t, 0.3, 5)
            times = [first] + sorted(self.clock.after(rng, first, 0.05, p["burst_days"])
                                     for _ in range(n - 1))
            self._never_pay_orders(a, who, times, ep, p)
            return
        country = pop.pick(rng, pop.HOME_COUNTRIES)
        sharing = pop.pick(rng, tuple(p["group_sharing"].items()))
        device = b.device(a, t, pop.device_ua(rng), pop.fingerprint(rng)) \
            if sharing == "device" else None
        address = None
        if sharing == "address":
            city, region = pop.home_city(rng, country)
            address = b.address(a, t, pop.line_hash(rng), city, region, country)
        root = None
        if sharing == "email":
            first, last = pop.person_name(rng)
            root = pop.email_address(rng, first, last)
        for k in range(int(rng.integers(*p["group_size"]))):
            signup = t if k == 0 else self.clock.after(rng, t, 0.2, p["group_days"])
            who = self._identity(a, signup, actor="fraudster", episode=ep, tags=tags,
                                 disposable_p=0.0, foreign_ip_p=0.0, country=country,
                                 device=device, address=address,
                                 email=pop.email_variant(rng, root, k) if root else None)
            who.ip = pop.ip_address(rng, country)
            first = signup + session_gap(rng, 8)
            times = [first] + [self.clock.after(rng, first, 0.1, 3)
                               for _ in range(int(rng.integers(0, 2)))]
            self._never_pay_orders(a, who, sorted(times), ep, p)

    def _never_pay_orders(self, a: Actor, who: Identity, times: list[int], ep: int,
                          p: dict[str, Any]) -> None:
        rng = a.rng
        for when in times:
            if not self._in_horizon(when):
                continue
            o = self._order(a, who, when, pattern="P-NEVERPAY", episode=ep,
                            categories=pop.RESALE_CATEGORIES if rng.random() < p["resale"]
                            else None, multiplier=tuple(p["amount_multiplier"]))
            self.out.fulfil(a, o)
            self.out.collect(a, o, "zero_effort")

    # ===================================================== promotion farming
    def promo_farm(self, a: Actor, t: int) -> None:
        b, rng, p = self.b, a.rng, self.f["P-PROMO"]
        promo = self.promotions["FIRST10"]
        ep = b.episode(a, "P-PROMO", t)
        sharing = pop.pick(rng, tuple(p["sharing"].items()))
        country = pop.pick(rng, pop.HOME_COUNTRIES)
        devices = [b.device(a, t, pop.device_ua(rng), pop.fingerprint(rng))
                   for _ in range(int(rng.integers(1, 3)))] if sharing == "device" else []
        address = None
        if sharing == "address":
            city, region = pop.home_city(rng, country)
            address = b.address(a, t, pop.line_hash(rng), city, region, country)
        root = None
        if sharing == "email":
            first, last = pop.person_name(rng)
            root = pop.email_address(rng, first, last, domain="gmail.com")
        home_ip = pop.ip_address(rng, country)
        for k in range(int(rng.integers(*p["cluster_size"]))):
            signup = t if k == 0 else self.clock.after(rng, t, 0.05, p["spread_days"])
            who = self._identity(a, signup, actor="fraudster", episode=ep, tags=("promo_farm",),
                                 disposable_p=p["disposable_email"], foreign_ip_p=0.0,
                                 country=country,
                                 device=devices[k % len(devices)] if devices else None,
                                 address=address,
                                 email=pop.email_variant(rng, root, k) if root else None)
            who.ip = home_ip if rng.random() < 0.6 else pop.ip_address(rng, country)
            when = signup + session_gap(rng, 8)
            if not self._in_horizon(when) or when >= b.promotions[promo]["to"]:
                continue
            o = self._order(a, who, when, pattern="P-PROMO", episode=ep, intent="abuse",
                            multiplier=tuple(p["amount_multiplier"]), promo=promo)
            self.out.fulfil(a, o)
            self.out.collect(a, o, "paid" if rng.random() < p["repays"] else "zero_effort")

    # ===================================================== item-not-received abuse
    def inr_abuse(self, a: Actor, t: int) -> None:
        b, rng, p = self.b, a.rng, self.f["P-INR-ABUSE"]
        opened = self.clock.at_profile_hour(rng, t - int(rng.uniform(*p["tenure_days"]) * DAY))
        ep = b.episode(a, "P-INR-ABUSE", opened)
        who = self._identity(a, opened, actor="fraudster", episode=ep, tags=("inr_abuse",),
                             disposable_p=0.0, foreign_ip_p=0.0)
        who.ip = pop.ip_address(rng, b.accounts[who.user][3])
        n = int(rng.integers(*p["orders"]))
        claims = int(pop.pick(rng, tuple(p["claims"].items())))
        times = sorted(self.clock.after(rng, opened, 1, p["active_days"]) for _ in range(n))
        orders = []
        for when in times:
            if not self._in_horizon(when):
                continue
            o = self._order(a, who, when, pattern="P-INR-ABUSE", episode=ep, intent="abuse",
                            categories=("electronics", "apparel", "shoes", "gaming"))
            self.out.fulfil(a, o)
            self.out.collect(a, o, "paid")
            orders.append(o)
        claimed = [o for o in orders if o.delivered is not None][-claims:]
        for o in orders:
            if o in claimed:
                outcome = "won" if rng.random() < p["claim_rejected"] else "lost"
                self.out.dispute(a, o, "item_not_received",
                                 self.clock.after(rng, o.delivered, *p["claim_days"]), outcome)
            else:
                o.pattern, o.episode, o.intent = None, None, "legitimate"

    # ===================================================== merchant bust-out
    def merchant_bustout(self, a: Actor, closed: int, buyers_from: int) -> int:
        """A merchant that ramps up sales, stops delivering and disappears.

        Returns the merchant; its customers (new accounts acquired at its
        checkout) are households added to ``customers`` for the second pass."""
        b, rng, p = self.b, a.rng, self.f["P-MERCH"]
        bust_days = rng.uniform(*p["bustout_days"])
        ramp_days = rng.uniform(*p["ramp_days"])
        bust_from = closed - int(bust_days * DAY)
        onboard = self.clock.at_profile_hour(rng, bust_from - int(ramp_days * DAY))
        category = pop.RESALE_CATEGORIES[int(rng.integers(0, len(pop.RESALE_CATEGORIES)))]
        merchant = b.merchant(a, onboard, self.market.name(rng), category,
                              int(rng.choice([2, 3], p=[0.4, 0.6])),
                              round(float(rng.uniform(8, 24)), 1), closed_at=closed)
        ep = b.episode(a, "P-MERCH", onboard)
        b.merchants[merchant]["bustout_from"] = bust_from
        b.merchants[merchant]["episode"] = ep
        self.market.add(merchant, p["general_traffic_weight"],
                        ramp=(onboard, onboard + int(ramp_days * DAY)))
        # buyers acquired at the merchant's own checkout, more as its promotion ramps up
        t, k = onboard, 0
        last_order = closed - int(p["last_order_hours_before_close"] * HOUR)
        while True:
            share = min(1.0, (t - onboard) / max(1, bust_from - onboard))
            low, high = p["buyers_per_day"]
            rate = low + share * (high - low)
            t += int(rng.exponential(DAY / rate))
            if t >= last_order:
                break
            if not self._in_horizon(t):
                continue
            k += 1
            buyer = Actor.of(self.seed, MERCH_BUYERS, buyers_from + k)
            h = Household(buyers_from + k, buyer, self.clock.at_profile_hour(buyer.rng, t),
                          pop.pick(buyer.rng, pop.HOME_COUNTRIES))
            if h.created >= last_order or h.created < onboard:
                h.created = t
            ticket = 1.0 + (p["ticket_lift"] - 1.0) * share
            self.customers.skeleton(h, 1, existing=False)
            for m in h.members:
                m.first_order, m.forced_first = "at_signup", (merchant, ticket)
                m.tags.add("merchant_acquired")
            self.customers.extra.append(h)
        return merchant

    # ===================================================== base world
    def run(self) -> None:
        """Every pattern's episodes over the order horizon (bust-outs run earlier)."""
        lo, hi = self.start, self.end
        for a, t in self._schedule(ATO, self.count("P-ATO"), lo + 30 * DAY, hi - DAY):
            self.account_takeover(a, t)
        n = self.count("P-STOLEN")
        for a, t in self._schedule(STOLEN, n, lo, hi - DAY):
            if a.rng.random() < self.f["P-STOLEN"]["sleeper_share"]:
                self.sleeper(a, t)
            else:
                self.stolen_card(a, t)
        p = self.f["P-SYNTH"]
        for a, t in self._schedule(SYNTH, self.count("P-SYNTH"), lo + 21 * DAY,
                                   hi - int(p["burst_hours"] * HOUR) - DAY):
            self.synthetic_ring(a, t)
        for a, t in self._schedule(NEVERPAY, self.count("P-NEVERPAY"), lo, hi - 7 * DAY):
            self.never_pay(a, t)
        for a, t in self._schedule(PROMO, self.count("P-PROMO"), lo, hi - 7 * DAY):
            self.promo_farm(a, t)
        p = self.f["P-INR-ABUSE"]
        for a, t in self._schedule(INR, self.count("P-INR-ABUSE"), lo + 30 * DAY,
                                   hi - int(p["active_days"] * DAY)):
            self.inr_abuse(a, t)

    def sleeper_reserve(self, n: int, created_lo: int, created_hi: int) -> None:
        """Aged sleeper accounts opened before the test window in every family; only
        the fraud-mix family activates them."""
        for a, t in self._schedule(SLEEPER_POOL, n, created_lo, created_hi):
            who, ep = self.sleeper(a, created_hi, created=t, activate=False)
            self.sleeper_pool.append((a, ep, who))

    def activate_sleepers(self, lo: int, hi: int) -> None:
        planner = Actor.of(self.seed, FAMILY_SLEEPER, 0)
        starts = self.clock.stratified_starts(planner.rng, len(self.sleeper_pool), lo, hi)
        for k, ((_, ep, who), t) in enumerate(zip(self.sleeper_pool, starts, strict=True)):
            a = Actor.of(self.seed, FAMILY_SLEEPER, k + 1)
            self.stolen_card(a, t, sleeper_identity=who, episode=ep)

    def extra_takeovers(self, n: int, lo: int, hi: int) -> None:
        for a, t in self._schedule(FAMILY_ATO, n, lo, hi):
            self.account_takeover(a, t)

