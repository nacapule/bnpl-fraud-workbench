"""Legitimate customers: households and the life of each account.

A household is one or more accounts sharing a home address (and sometimes a
tablet). Each account opens with its own phone and card, then lives through
the ordinary events that make fraud detection hard: new phones (sometimes
followed by a password reset), extra devices and cards, moves, trips abroad,
gifts shipped elsewhere, promotions, late payments, hardship and defaults,
lost parcels and the occasional dispute. Home country decides home IPs.

Generation runs in two passes so that gifts can ship to other households'
homes: :meth:`Customers.skeleton` creates each household's accounts and home
addresses, then :meth:`Customers.live` plays out each account's events in time
order through the shared primitives, using the state each account has at that
moment.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulator import population as pop
from simulator.builder import DAY, Actor, Builder, Order
from simulator.merchants import Market
from simulator.outcomes import Outcomes
from simulator.timing import MINUTE, Clock, session_gap

YEAR = 365 * DAY


@dataclass
class Member:
    user: int
    signup: int
    active_until: int
    first_order: str  # at_signup | soon | none | existing
    rate: float  # relative order rate (gamma draw)
    late_prone: bool
    fragile: bool
    first_name: str
    last_name: str
    forced_first: tuple[int, float] | None = None  # (merchant, amount multiplier)
    promo_take: float | None = None
    tags: set[str] = field(default_factory=set)


@dataclass
class Household:
    index: int
    actor: Actor
    created: int
    country: str
    members: list[Member] = field(default_factory=list)
    # (address, from, to, home IP)
    homes: list[tuple[int, int, int | None, str]] = field(default_factory=list)
    tablet: int | None = None


class Customers:
    def __init__(self, b: Builder, clock: Clock, market: Market, outcomes: Outcomes,
                 cfg: dict[str, Any], promotions: dict[str, int], order_end: int) -> None:
        self.b, self.clock, self.market, self.out = b, clock, market, outcomes
        self.c = cfg["customers"]
        self.pay = cfg["payments"]
        self.disp = cfg["legit_disputes"]
        self.promotions = promotions
        self.order_end = order_end
        self.households: list[Household] = []  # every household, in creation order
        self.base: list[Household] = []  # the arrival process's households
        self.extra: list[Household] = []  # acquired by a merchant or a family's campaign
        self.home_registry: list[tuple[int, int]] = []  # (created, address) sorted later
        self.mu: float | None = None

    # ============================================================ pass 1
    def skeleton(self, h: Household, size: int, *, existing: bool) -> None:
        """Accounts, home addresses (with moves) and the shared tablet of one household."""
        b, rng, c = self.b, h.actor.rng, self.c
        country = h.country
        city, region = pop.home_city(rng, country)
        home = b.address(h.actor, h.created, pop.line_hash(rng), city, region, country)
        h.homes.append((home, h.created, None, pop.ip_address(rng, country)))
        self.home_registry.append((h.created, home))
        # moves: the household changes home address
        t = h.created
        while True:
            t += int(rng.exponential(YEAR / c["move_rate_per_year"]))
            if t >= self.order_end:
                break
            city, region = pop.home_city(rng, country)
            new = b.address(h.actor, t, pop.line_hash(rng), city, region, country)
            old = h.homes[-1]
            h.homes[-1] = (old[0], old[1], t, old[3])
            h.homes.append((new, t, None, pop.ip_address(rng, country)))
        last = None
        for k in range(size):
            if k == 0:
                signup = h.created
            elif existing:
                signup = h.created + int(rng.uniform(0, 400) * DAY)
            else:
                signup = h.created + int(rng.exponential(c["member_join_mean_days"]) * DAY)
            if signup >= self.order_end:
                continue
            if k:
                signup = self.clock.around(rng, signup, after=h.created)
            first, family = pop.person_name(rng)
            if last is not None and rng.random() < 0.6:
                family = last
            last = family
            email = pop.email_address(rng, first, family,
                                      disposable=rng.random() < c["disposable_email_share"])
            user = b.account(h.actor, signup, email, country, pop.birth_year(rng, signup))
            m = self._member(h, user, signup, first, family, existing=existing and k == 0
                             or signup < self.clock.start)
            if size > 1:
                m.tags.add("household")
            h.members.append(m)
            for addr, start, end, _ in h.homes:
                if end is not None and end <= signup:
                    continue
                link = b.link_address(user, addr, max(start, signup), "home")
                if end is not None:
                    b.unlink(link, end)
            if len(h.homes) > 1:
                m.tags.add("mover")
        if len(h.members) > 1 and rng.random() < c["shared_tablet_share"]:
            second = h.members[1].signup
            h.tablet = b.device(h.actor, second, "Android" if rng.random() < 0.5 else "iOS",
                                pop.fingerprint(rng))
            for m in h.members:
                b.link_device(m.user, h.tablet, max(second, m.signup))
                m.tags.add("shared_device")
        self.households.append(h)

    def _member(self, h: Household, user: int, signup: int, first: str, last: str,
                *, existing: bool) -> Member:
        rng, c = h.actor.rng, self.c
        life = int(rng.exponential(c["lifetime_mean_days"]) * DAY)
        if existing:
            kind = "existing"
        else:
            u = rng.random()
            share = c["first_order"]
            kind = ("at_signup" if u < share["at_signup"] else
                    "soon" if u < share["at_signup"] + share["soon"] else "none")
        rate = float(rng.gamma(c["order_rate_shape"], 1.0 / c["order_rate_shape"]))
        if kind == "none" or (existing and rng.random() < c["dormant_share"]):
            rate = 0.0
        m = Member(user, signup, signup + life, kind, rate,
                   late_prone=rng.random() < self.pay["late_prone_share"],
                   fragile=rng.random() < self.pay["fragile_share"],
                   first_name=first, last_name=last)
        m.tags.add("existing" if existing else "new_customer")
        if rate == 0.0:
            m.tags.add("dormant")
        if m.late_prone:
            m.tags.add("late_payer")
        if m.fragile:
            m.tags.add("fragile")
        return m

    def exposure(self, m: Member) -> tuple[float, float]:
        """(certain first orders, seasonal exposure of the repeat-order process)."""
        if m.first_order == "none":
            return 0.0, 0.0
        if m.first_order == "existing":
            lo = max(m.signup, self.clock.start)
            return 0.0, m.rate * self.clock.exposure(lo, min(m.active_until, self.order_end))
        lo = m.signup + (0 if m.first_order == "at_signup" else
                         int(self.c["first_order"]["soon_mean_days"] * DAY))
        first = 1.0 if self.clock.start <= lo < self.order_end else 0.0
        return first, m.rate * self.clock.exposure(lo, min(m.active_until, self.order_end))

    def calibrate(self, target_orders: float) -> None:
        firsts, exposure = 0.0, 0.0
        for h in self.base:
            for m in h.members:
                f, e = self.exposure(m)
                firsts += f
                exposure += e
        self.mu = max(0.0, target_orders - firsts) / max(exposure, 1e-9)

    def finish_registry(self) -> None:
        self.home_registry.sort()
        self._home_times = [t for t, _ in self.home_registry]

    def gift_address(self, rng: np.random.Generator, t: int) -> int | None:
        n = bisect.bisect_right(self._home_times, t)
        return self.home_registry[int(rng.integers(0, n))][1] if n else None

    # ============================================================ pass 2
    def live(self, h: Household) -> None:
        for m in h.members:
            self._life(h, m)
            self.b.tag(m.user, *m.tags)

    def _life(self, h: Household, m: Member) -> None:
        b, a, rng, c, clock = self.b, h.actor, h.actor.rng, self.c, self.clock
        start, end = clock.start, self.order_end
        country = h.country
        phone = b.device(a, m.signup, pop.device_ua(rng), pop.fingerprint(rng))
        phone_link = b.link_device(m.user, phone, m.signup)
        bin_country = country if rng.random() > c["foreign_card_share"] else pop.pick(
            rng, pop.TRAVEL_COUNTRIES)
        b.card(a, m.user, m.signup + int(rng.uniform(0, MINUTE)), bin_country,
               pop.card_network(rng), pop.last4(rng))

        actions: list[tuple[int, int, str, Any]] = []  # (time, seq, kind, payload)
        counter = itertools.count()

        def push(t: int, kind: str, payload: Any = None) -> None:
            heapq.heappush(actions, (t, next(counter), kind, payload))

        def schedule(kind: str, rate_per_year: float, lo: int, hi: int) -> None:
            t = lo
            while rate_per_year > 0:
                t += int(rng.exponential(YEAR / rate_per_year))
                if t >= hi:
                    break
                when = clock.around(rng, t, after=lo)
                if when < hi:
                    push(when, kind)

        schedule("new_phone", c["new_phone_per_year"], m.signup, end)
        schedule("card_added", c["card_added_per_year"], m.signup, end)
        schedule("card_replaced", c["card_replaced_per_year"], m.signup, end)
        schedule("password_change", c["password_change_per_year"], m.signup, end)
        schedule("email_change", c["email_change_per_year"], m.signup, end)
        schedule("phone_change", c["phone_change_per_year"], m.signup, end)
        schedule("login", c["logins_per_year"], max(m.signup, start), min(m.active_until, end))
        if rng.random() < c["extra_device_share"] and end - m.signup > DAY:
            push(clock.between(rng, m.signup + 1, end), "extra_device")
        trips: list[tuple[int, int, str]] = []
        t = max(m.signup, start)
        while True:
            t += int(rng.exponential(YEAR / c["trips_per_year"]))
            if t >= end:
                break
            back = t + int(rng.uniform(*c["trip_days"]) * DAY)
            trips.append((t, back, pop.travel_destination(rng, country,
                                                           c["trip_neighbour_share"])))
            t = back
        if trips:
            m.tags.add("traveller")
        for t in self._order_times(m, rng):
            push(int(t), "order")

        state = {"phone": phone_link, "reset_at": None, "churn_after": None}
        while actions:
            t, _, kind, payload = heapq.heappop(actions)
            if t >= end:
                break
            if kind == "new_phone":
                new = b.device(a, t, pop.device_ua(rng), pop.fingerprint(rng))
                link = b.link_device(m.user, new, t)
                old = state["phone"]
                b.unlink(old, t + int(rng.uniform(0, 3) * DAY))
                state["phone"] = link
                if rng.random() < c["reset_after_new_phone"] and t >= start:
                    push(t + int(rng.uniform(5 * MINUTE, DAY)), "reset", new)
            elif kind == "reset":
                if payload in b.devices_at(m.user, t):
                    self._event(a, m, h, t, "password_reset", payload, trips)
                    state["reset_at"] = t
                    m.tags.add("new_phone_reset")
            elif kind == "extra_device":
                device = b.device(a, t, pop.device_ua(rng, mobile=False), pop.fingerprint(rng))
                b.link_device(m.user, device, t)
            elif kind == "card_added":
                b.card(a, m.user, t, bin_country, pop.card_network(rng), pop.last4(rng))
            elif kind == "card_replaced":
                cards = b.cards_at(m.user, t)
                b.card(a, m.user, t, bin_country, pop.card_network(rng), pop.last4(rng))
                if cards:
                    b.remove_card(cards[0], t)
            elif t < start:
                continue
            elif kind in ("password_change", "phone_change", "login"):
                self._event(a, m, h, t, kind, state["phone"].item, trips)
            elif kind == "email_change":
                email = pop.email_address(rng, m.first_name, m.last_name)
                self._event(a, m, h, t, kind, state["phone"].item, trips, email=email)
            elif kind == "order":
                if state["churn_after"] is not None and t > state["churn_after"]:
                    continue
                self._shop(h, m, t, state, trips, push)
            elif kind == "retry":
                self._retry(h, m, t, state, trips, payload)

    def _order_times(self, m: Member, rng: np.random.Generator) -> list[int]:
        clock, start, end = self.clock, self.clock.start, self.order_end
        times: list[int] = []
        if m.first_order == "none" and m.forced_first is None:
            return times
        if m.first_order == "existing":
            lo = max(m.signup, start)
        else:
            if m.forced_first is not None or m.first_order == "at_signup":
                first = m.signup + session_gap(rng, 8)
            else:  # days later, at a profile time around an exponential delay
                earliest = m.signup + 30 * MINUTE
                delay = int(rng.exponential(self.c["first_order"]["soon_mean_days"]) * DAY)
                first = clock.around(rng, max(earliest, m.signup + delay), after=earliest)
            if not start <= first < end:
                return times
            times.append(first)
            lo = first + 1
        hi = min(m.active_until, end)
        if hi > lo and m.rate > 0:
            n = int(rng.poisson(self.mu * m.rate * clock.exposure(lo, hi)))
            times.extend(int(x) for x in clock.seasonal_times(rng, n, lo, hi))
        return times

    def _where(self, h: Household, t: int, trips: list[tuple[int, int, str]],
               rng: np.random.Generator) -> tuple[str, str, bool]:
        """(ip, ip_country, travelling) for an action at ``t``."""
        for leave, back, country in trips:
            if leave <= t < back:
                return pop.ip_address(rng, country), country, True
        if rng.random() < self.c["mobile_ip_share"]:
            return pop.ip_address(rng, h.country), h.country, False
        for _, start, stop, ip in h.homes:
            if start <= t and (stop is None or t < stop):
                return ip, h.country, False
        return h.homes[-1][3], h.country, False

    def _event(self, a: Actor, m: Member, h: Household, t: int, kind: str, device: int,
               trips: list, email: str | None = None) -> None:
        ip, country, _ = self._where(h, t, trips, a.rng)
        self.b.account_event(a, m.user, t, kind, device, ip, country, email)

    # ------------------------------------------------------------ orders
    def _shop(self, h: Household, m: Member, t: int, state: dict[str, Any],
              trips: list[tuple[int, int, str]], push) -> None:
        b, a, rng, c = self.b, h.actor, h.actor.rng, self.c
        devices = b.devices_at(m.user, t)
        homes = b.addresses_at(m.user, t, "home")
        cards = b.cards_at(m.user, t)
        if not devices or not homes or not cards:
            return
        phone = state["phone"].item
        device = phone if (phone in devices and rng.random() < 0.85) else \
            devices[int(rng.integers(0, len(devices)))]
        card = cards[-1] if rng.random() < 0.8 else cards[int(rng.integers(0, len(cards)))]
        first = not b.approved_before(m.user, t)
        mimic: set[str] = set()
        if m.forced_first is not None and not state.get("shopped") \
                and not self.b.merchant_open(m.forced_first[0], t):
            return
        if m.forced_first is not None and not state.get("shopped"):
            merchant, multiplier = m.forced_first
        else:
            merchant, multiplier = self.market.choose(rng, t), 1.0
        category = b.merchants[merchant]["category"]
        if first and t - m.signup < 7 * DAY:
            mimic.add("new_customer_first_order")
            if rng.random() < c["large_first_order_share"]:
                multiplier *= rng.uniform(*c["large_first_order_multiplier"])
                mimic.add("large_first_order")
        amount = pop.order_amount_cents(rng, category, multiplier)
        address = homes[0]
        if rng.random() < c["gift_share"]:
            target = self.gift_address(rng, t) if rng.random() < 0.5 else None
            if target is None or target in homes:
                country = h.country
                city, region = pop.home_city(rng, country)
                target = b.address(a, t, pop.line_hash(rng), city, region, country)
            if target not in b.addresses_at(m.user, t):
                b.link_address(m.user, target, t, "shipping")
            address = target
            mimic.add("gift")
        ip, ip_country, travelling = self._where(h, t, trips, rng)
        if travelling:
            mimic.add("travel")
        if h.tablet is not None and device == h.tablet:
            mimic.add("shared_device")
        if state["reset_at"] is not None and 0 <= t - state["reset_at"] < 3 * DAY:
            mimic.add("new_phone_reset")
        promo = self.promotion(rng, m.user, t, m.promo_take)
        state["shopped"] = True
        b.login_before(a, m.user, t, device, ip, ip_country, c["login_before_order"])
        avs = "N" if rng.random() < self.pay["avs_fail"] else "Y"
        cvv = "N" if rng.random() < self.pay["cvv_fail"] else "M"
        declined = rng.random() < self.pay["processor_decline"]
        o = b.order(a, m.user, merchant, t, amount, device=device, card=card, address=address,
                    ip=ip, ip_country=ip_country, avs=avs, cvv=cvv, approved=not declined,
                    promo=None if declined else promo)
        o.mimic |= mimic
        if declined:
            o.mimic.add("processor_decline")
            if rng.random() < c["retry_after_decline"]:
                push(t + session_gap(rng, 6), "retry",
                     {"merchant": merchant, "amount": amount, "device": device, "card": card,
                      "address": address, "first": first, "mimic": mimic})
            return
        self.settle(a, m, o, state, first)

    def _retry(self, h: Household, m: Member, t: int, state: dict[str, Any],
               trips: list[tuple[int, int, str]], spec: dict[str, Any]) -> None:
        """The same purchase tried again minutes after a processor decline, with what
        the account holds at the retry."""
        b, a, rng = self.b, h.actor, h.actor.rng
        devices = b.devices_at(m.user, t)
        cards = b.cards_at(m.user, t)
        if not devices or not cards or not b.merchant_open(spec["merchant"], t):
            return
        device = spec["device"] if spec["device"] in devices else devices[-1]
        others = [x for x in cards if x != spec["card"]] or cards
        address = spec["address"]
        if address not in b.addresses_at(m.user, t):
            homes = b.addresses_at(m.user, t, "home")
            if not homes:
                return
            address = homes[0]
        ip, ip_country, _ = self._where(h, t, trips, rng)
        o = b.order(a, m.user, spec["merchant"], t, spec["amount"], device=device,
                    card=others[-1], address=address, ip=ip, ip_country=ip_country,
                    avs="N" if rng.random() < self.pay["avs_fail"] else "Y",
                    cvv="N" if rng.random() < self.pay["cvv_fail"] else "M",
                    promo=self.promotion(rng, m.user, t, m.promo_take))
        o.mimic |= spec["mimic"]
        self.settle(a, m, o, state, spec["first"])

    def promotion(self, rng: np.random.Generator, user: int, t: int,
                  first_take: float | None = None) -> int | None:
        """The ordinary use of promotions, for any account: the first-purchase offer on
        an account's first approved order, otherwise a seasonal code while one runs."""
        if not self.b.approved_before(user, t):
            take = self.c["first_purchase_take"] if first_take is None else first_take
            promo = self.promotions["FIRST10"]
            p = self.b.promotions[promo]
            if p["from"] <= t < p["to"] and rng.random() < take:
                return promo
            return None
        for promo in self.promotions.values():
            p = self.b.promotions[promo]
            if not p["first"] and p["from"] <= t < p["to"]:
                return promo if rng.random() < self.c["seasonal_promo_take"] else None
        return None

    def settle(self, a: Actor, m: Member, o: Order, state: dict[str, Any], first: bool) -> None:
        """Repayment, delivery and disputes of a legitimate customer's order."""
        rng, p, d, out = a.rng, self.pay, self.disp, self.out
        new_customer = first and o.t - m.signup < 30 * DAY
        default_p = p["default_first_plan_new"] if new_customer else p["default_other"]
        if m.fragile:
            default_p *= p["fragile_multiplier"]
        plan, stop = "paid", 1
        if rng.random() < default_p:
            zero = p["zero_effort_share_first"] if new_customer else p["zero_effort_share_other"]
            plan = "zero_effort" if rng.random() < zero else "partial"
            stop = int(rng.integers(2, len(o.schedule)))
            o.mimic.add("hardship_default")
            m.tags.add("hardship")
            if rng.random() < p["stop_ordering_after_default"]:
                state["churn_after"] = o.t
        merchant = self.b.merchants[o.merchant]
        bustout = out.vanishing(o)
        lost = not bustout and rng.random() < d["lost_parcel"]
        unscanned = not bustout and not lost and rng.random() < d["unscanned_delivery"]
        out.fulfil(a, o, deliver=not (lost or unscanned))
        if lost:
            o.mimic.add("genuine_non_delivery")
        out.collect(a, o, plan, late_p=p["late_p_prone"] if m.late_prone else p["late_p"],
                    reversal_p=p["reversal"], stop_seq=stop)
        if bustout:
            o.pattern, o.episode, o.intent = "P-MERCH", merchant["episode"], "fraud"
            if rng.random() < d["bustout_claim"]:
                filed = out.file_time(a, max(o.t, merchant["closed"]), 1, 30)
                out.dispute(a, o, "item_not_received", filed, "lost")
            return
        if lost and rng.random() < d["lost_parcel_claim"]:
            out.dispute(a, o, "item_not_received", out.file_time(a, o.t, 10, 25), "lost")
        elif o.delivered is not None:
            u = rng.random()
            if u < d["porch_claim"]:
                outcome = "won" if rng.random() < d["porch_claim_won"] else "lost"
                out.dispute(a, o, "item_not_received", out.file_time(a, o.delivered, 2, 10),
                            outcome)
            elif u < d["porch_claim"] + d["not_as_described"]:
                outcome = "won" if rng.random() < 0.5 else "lost"
                out.dispute(a, o, "not_as_described", out.file_time(a, o.delivered, 5, 25),
                            outcome)
            elif u < d["porch_claim"] + d["not_as_described"] + d["friendly_unauthorized"]:
                outcome = "lost" if rng.random() < 0.5 else "won"
                out.dispute(a, o, "unauthorized", out.file_time(a, o.t, 10, 40), outcome)
