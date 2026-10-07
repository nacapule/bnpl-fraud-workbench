"""Seeded random worlds for the as-of context tests, and what the tests do to them.

:func:`random_world` builds a small world in the core.world schema (about forty
accounts and three hundred order attempts) full of what the context has to get
right: several attempts in one second (also across accounts and event kinds),
attempts and decisions exactly a window's width after earlier activity,
knowledge that arrives after the event, devices and addresses shared by
households, rings and promotion farms, links that end (and a device taken over
at the second its old link ends), email changes (back again, twice in one
second, +tag, dot and googlemail variants), processor declines, first-purchase
and other promotions, failed, retried and partly reversed payments, write-offs,
disputes, victim reports, and merchant categories with and without enough
approved amounts for a reference. ``scale`` multiplies the population
(``scale=425`` gives about 120,000 attempts in about half a minute). Every
reference exists, orders use a device, card and address of their account active
at the time, events follow the entities they use and ``known_at >= occurred_at``;
the world need not pass core.world.validate_world. Event ids are shuffled, so
only the tie order ``(known_at, kind rank, event_id)`` orders events in one second.

:func:`known_by` cuts a world to what is known at a time or at a place in the
event order; :func:`random_acts`, :func:`random_blocks` and :func:`realize` give a
policy's realized tables and :class:`core.asof.PolicyState`; :func:`later_decisions`
picks review-like decision times.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import asof, config, ledger, world

TERMS = ledger.ProductTerms.from_config()
MINUTE, HOUR, DAY = 60, 3_600, 86_400
START = int(pd.Timestamp("2025-01-01").value // 10**9)
ORDER_END = START + 120 * DAY
OBSERVED_UNTIL = START + 240 * DAY
HORIZON_DAYS = 60

# category: share of orders, amount range in cents, merchants at scale 1
CATEGORIES = {"electronics": (0.6, 2_000, 150_000, 3), "apparel": (0.28, 1_500, 30_000, 2),
              "jewelry": (0.12, 5_000, 200_000, 1)}
FOREIGN = ("CA", "GB", "DE", "FR", "MX", "BR", "IN", "RO", "NG")
DOMAINS = ("gmail.com",) * 5 + ("outlook.com", "outlook.com", "yahoo.com", "proton.me",
                                "fastmail.com", "mailinator.com")
FIRST_NAMES = ("ann", "bo", "cy", "dee", "eli", "fox", "gia", "hu", "ida", "jo")
LAST_NAMES = ("lee", "kim", "ng", "ortiz", "park", "quinn", "roy", "sato")
BEHAVIOURS = {"payer": 55, "late": 10, "partial": 7, "never": 8, "inr": 5, "fraud": 5,
              "farmer": 5, "takeover": 5}
FIRST_BEHAVIOURS = ("takeover", "fraud", "farmer", "farmer", "farmer", "inr", "never", "never",
                    "late", "partial", "takeover")
PAYS = {"late": "late", "partial": "partial", "never": "never", "fraud": "never",
        "farmer": "partial"}
WINDOWS = (HOUR, 12 * HOUR, DAY, 7 * DAY, 30 * DAY)  # the context's windows, to hit exactly
PATTERNS = {"fraud": "P-STOLEN", "takeover": "P-ATO", "farmer": "P-PROMO",
            "never": "P-NEVERPAY", "inr": "P-INR-ABUSE"}
CREATED = {"accounts": "created_at", "merchants": "created_at", "devices": "created_at",
           "device_links": "created_at", "addresses": "created_at",
           "address_links": "created_at", "cards": "created_at", "promotions": "valid_from"}
ENDS = {"device_links": "removed_at", "address_links": "removed_at", "cards": "removed_at",
        "merchants": "closed_at"}


def seconds(values) -> np.ndarray:
    """Whole seconds since the epoch of timestamps (NaT becomes the int64 minimum)."""
    return pd.Series(values).to_numpy().astype("datetime64[s]").astype(np.int64)


def times(values) -> np.ndarray:
    """datetime64[s] from seconds since the epoch (None or NaN becomes NaT)."""
    return np.array([None if v is None or v != v else int(v) for v in values],
                    dtype="datetime64[s]")


def active(rows: list[dict], at: int) -> list[dict]:
    """Link or card rows in force at ``at``: created by then, not yet removed."""
    return [r for r in rows if r["created_at"] <= at
            and (r["removed_at"] is None or at < r["removed_at"])]


@dataclass
class _Account:
    user_id: int
    created: int
    country: str
    behaviour: str
    devices: list[dict] = field(default_factory=list)
    addresses: list[dict] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    due_dates: list[int] = field(default_factory=list)
    activity: list[tuple[int, int]] = field(default_factory=list)  # (time, device) used
    joined: int | None = None  # when a farmer joined its farm's device
    approved: bool = False

    @property
    def first(self) -> int:  # earliest time it may order
        return max(START, self.created + MINUTE)


class _Builder:
    def __init__(self, seed: int, scale: float) -> None:
        self.rng = random.Random(seed)
        self.rows: dict[str, list[dict]] = defaultdict(list)
        self.last_id: dict[str, int] = defaultdict(int)
        self.busy: list[int] = []  # known_at of events so far, for ties across kinds
        self.on_device: dict[int, list[int]] = defaultdict(list)  # attempt times per device
        self.device_created: dict[int, int] = {}
        self.links_to: dict[int, list[dict]] = defaultdict(list)  # device links per device
        self.homes: list[tuple[int, int]] = []  # (address, a resident's device)
        self.profile: dict[int, str] = {}
        self.stolen: set[int] = set()
        accounts = max(12, round(40 * scale))
        self.add_merchants(math.ceil(math.sqrt(scale)))
        self.add_promotions()
        self.shared = [self.device(START - self.days(10, 400))
                       for _ in range(max(2, accounts // 6))]
        self.drops = [self.address(START - self.days(10, 400))
                      for _ in range(max(2, accounts // 8))]
        # fewer names than accounts in a small world, so that emails are shared
        self.names = [self.name(i)
                      for i in range(max(4, round(0.75 * accounts * math.sqrt(scale))))]
        self.farm_of: dict[int, int] = {}  # farmer -> its farm's device (four per farm)
        self.farm = 0  # the device of the farm being filled
        weights = list(BEHAVIOURS.values())
        for i in range(accounts):
            self.account(FIRST_BEHAVIOURS[i] if i < len(FIRST_BEHAVIOURS)
                         else self.rng.choices(list(BEHAVIOURS), weights)[0])

    # ------------------------------------------------------------------ helpers
    def new_id(self, column: str) -> int:
        self.last_id[column] += 1
        return self.last_id[column]

    def days(self, low: float, high: float) -> int:
        return self.rng.randint(int(low * DAY), int(high * DAY))

    def late(self, at: int, share: float, low: int, high: int) -> int:
        """``at``, or with probability ``share`` a known_at ``low``..``high`` seconds later."""
        return at + self.rng.randint(low, high) if self.rng.random() < share else at

    def event(self, table: str, occurred: int, known: int | None = None, **values) -> int:
        known = occurred if known is None else known
        event_id = self.new_id("event_id")
        self.rows[table].append({"event_id": event_id, "occurred_at": occurred,
                                 "known_at": known, **values})
        self.busy.append(known)
        return event_id

    def name(self, i: int) -> tuple[str, str]:
        first = FIRST_NAMES[i % len(FIRST_NAMES)]
        last = LAST_NAMES[i // len(FIRST_NAMES) % len(LAST_NAMES)]
        number = i // (len(FIRST_NAMES) * len(LAST_NAMES)) or ""
        return f"{first}.{last}{number}", self.rng.choice(DOMAINS)

    def email(self, name: tuple[str, str]) -> str:
        """An address of ``name`` as someone might type it: Gmail dots moved or dropped,
        googlemail, a +tag, capitals; elsewhere dropped dots make another address."""
        rng = self.rng
        local, domain = name
        if domain == "gmail.com" and rng.random() < 0.5:
            plain = local.replace(".", "")
            spot = rng.randint(1, len(plain) - 1)
            local = plain if rng.random() < 0.5 else f"{plain[:spot]}.{plain[spot:]}"
            domain = "googlemail.com" if rng.random() < 0.5 else domain
        elif rng.random() < 0.15:
            local = local.replace(".", "")
        if rng.random() < 0.25:
            local += "+" + rng.choice(("shop", "x", "promo", "2"))
        if rng.random() < 0.2:
            local, domain = local.capitalize(), domain.upper()
        return f"{local}@{domain}"

    # ----------------------------------------------------------------- entities
    def add_merchants(self, per: int) -> None:
        for category, (_, _, _, count) in CATEGORIES.items():
            for i in range(count * per):
                merchant_id = self.new_id("merchant_id")
                created = START - self.days(30, 900)
                if category == "electronics" and i == 0:
                    created = START + self.days(5, 30)  # onboarded during the orders
                closed = START + self.days(70, 100) if category == "jewelry" and i == 0 else None
                self.rows["merchants"].append({
                    "merchant_id": merchant_id, "created_at": created, "name": f"m{merchant_id}",
                    "category": category, "risk_tier": self.rng.randint(1, 3),
                    "fulfilment_median_hours": float(self.rng.choice((2, 6, 12, 24, 48))),
                    "closed_at": closed})

    def add_promotions(self) -> None:
        for code, bps, first, valid_from, valid_to in (
                ("FIRST10", 1000, True, -30, 200), ("HELLO5", 500, True, 20, 90),
                ("SALE15", 1500, False, 40, 70)):
            self.rows["promotions"].append({
                "promo_id": self.new_id("promo_id"), "code": code, "discount_bps": bps,
                "first_purchase_only": first, "valid_from": START + valid_from * DAY,
                "valid_to": START + valid_to * DAY})

    def device(self, created: int) -> int:
        device_id = self.new_id("device_id")
        self.device_created[device_id] = created
        self.rows["devices"].append({"device_id": device_id, "created_at": created,
                                     "fingerprint": f"fp{device_id}",
                                     "ua_family": self.rng.choice(("iOS", "Android", "Chrome"))})
        return device_id

    def address(self, created: int, country: str = "US") -> int:
        address_id = self.new_id("address_id")
        self.rows["addresses"].append({"address_id": address_id, "created_at": created,
                                       "line_hash": f"h{address_id}", "city": "Springfield",
                                       "region": "IL", "country": country})
        return address_id

    def link_device(self, a: _Account, device: int, start: int) -> dict:
        row = {"user_id": a.user_id, "device_id": device, "created_at": start,
               "removed_at": None}
        self.rows["device_links"].append(row)
        self.links_to[device].append(row)
        a.devices.append(row)
        return row

    def link_address(self, a: _Account, address: int, start: int, role: str) -> dict:
        row = {"user_id": a.user_id, "address_id": address, "created_at": start,
               "removed_at": None, "role": role}
        self.rows["address_links"].append(row)
        a.addresses.append(row)
        return row

    def card(self, a: _Account, created: int, foreign: bool = False) -> dict:
        row = {"card_id": self.new_id("card_id"), "user_id": a.user_id, "created_at": created,
               "removed_at": None,
               "bin_country": self.rng.choice(FOREIGN) if foreign else a.country,
               "network": self.rng.choice(("visa", "mastercard")), "last4": "4242"}
        self.rows["cards"].append(row)
        a.cards.append(row)
        return row

    # ----------------------------------------------------------------- accounts
    def account(self, behaviour: str) -> None:
        rng = self.rng
        user_id = self.new_id("user_id")
        self.profile[user_id] = behaviour
        if behaviour == "takeover" or (behaviour not in ("fraud", "farmer")
                                       and rng.random() < 0.3):
            created = START - self.days(100, 900)
        elif behaviour == "farmer":
            created = START + self.days(30, 40)
        else:
            created = START + self.days(-60, 100)
        country = "US" if rng.random() < 0.85 else rng.choice(FOREIGN)
        email = self.email(rng.choice(self.names))
        self.rows["accounts"].append({
            "user_id": user_id, "created_at": created, "email": email,
            "email_domain": email.rpartition("@")[2].lower(), "home_country": country,
            "dob_year": rng.randint(1950, 2005)})
        a = _Account(user_id, created, country, behaviour)

        # devices: its own (a new phone may replace it at the same second), maybe a shared one
        own = self.link_device(a, self.device(created - self.days(0, 30)), created)
        if rng.random() < 0.3:
            at = rng.randint(created + HOUR, ORDER_END)
            if rng.random() < 0.7:
                own["removed_at"] = at
            self.link_device(a, self.device(at - rng.randint(0, DAY)), at)
        if behaviour == "farmer" or rng.random() < 0.3:
            if behaviour == "farmer":
                if len(self.farm_of) % 4 == 0:
                    self.farm = self.device(START - self.days(10, 400))
                device = self.farm_of[user_id] = self.farm
            else:
                device = rng.choice(self.shared)
            start = max(created, self.device_created[device]) + self.days(0, 60)
            ended = [r["removed_at"] for r in self.links_to[device]
                     if r["removed_at"] is not None and r["removed_at"] >= created]
            if behaviour == "farmer":
                start = a.joined = created + rng.randint(5 * MINUTE, 2 * HOUR)
            elif ended and rng.random() < 0.8:
                start = rng.choice(ended)  # taken over at the second the other link ends
            row = self.link_device(a, device, start)
            if behaviour != "farmer" and rng.random() < 0.5:
                row["removed_at"] = start + self.days(1, 60)

        # home: its own or a household's (with a resident's device), maybe a move
        if self.homes and rng.random() < 0.15:
            address, partner_device = rng.choice(self.homes)
            if rng.random() < 0.5:
                self.link_device(a, partner_device, created + self.days(0, 20))
        else:
            address = self.address(created, country)
        home = self.link_address(a, address, created, "home")
        self.homes.append((address, own["device_id"]))
        if rng.random() < 0.15:
            moved = rng.randint(created + DAY, ORDER_END)
            home["removed_at"] = moved
            new_home = self.link_address(a, self.address(moved, country), moved, "home")
            if rng.random() < 0.3:  # and back again later
                back = rng.randint(moved, ORDER_END + DAY)
                new_home["removed_at"] = back
                self.link_address(a, address, back, "home")
        if rng.random() < (0.6 if behaviour == "fraud" else 0.2):
            row = self.link_address(a, rng.choice(self.drops), created + self.days(0, 30),
                                    "shipping")
            if rng.random() < 0.3:
                row["removed_at"] = row["created_at"] + self.days(1, 60)
        if rng.random() < 0.15:
            self.link_address(a, self.address(created + self.days(0, 50)),
                              created + self.days(50, 60), "shipping")

        # cards: one from signup, maybe more (removed sometimes); stolen ones for fraud
        self.card(a, created + rng.randint(1, 50))
        for _ in range(4 if behaviour == "fraud" else rng.choice((0, 0, 1, 2))):
            row = self.card(a, created + rng.randint(51, 50 * DAY if behaviour != "fraud" else 120),
                            foreign=behaviour == "fraud" or rng.random() < 0.1)
            if behaviour != "fraud" and rng.random() < 0.2:
                row["removed_at"] = row["created_at"] + self.days(1, 90)

        self.account_events(a)
        if behaviour == "takeover":
            self.takeover(a)
        self.shop(a)

    def account_event(self, a: _Account, kind: str, at: int, device: int | None = None,
                      country: str | None = None, email: str | None = None) -> None:
        if device is None:
            device = self.rng.choice(active(a.devices, at))["device_id"]
        if country is None:
            country = a.country if self.rng.random() < 0.9 else self.rng.choice(FOREIGN)
        a.activity.append((at, device))
        self.event("account_events", at, user_id=a.user_id, kind=kind, device_id=device,
                   ip=f"10.0.{a.user_id % 250}.{self.rng.randint(1, 250)}", ip_country=country,
                   email=email)

    def account_events(self, a: _Account) -> None:
        rng = self.rng

        def when() -> int:
            at = rng.randint(a.created + 1, ORDER_END)
            if self.busy and rng.random() < 0.1:
                tie = rng.choice(self.busy)
                at = tie if a.created < tie < ORDER_END else at
            return at

        for _ in range(rng.randint(0, 5)):
            self.account_event(a, "login", when())
        for kind, share in (("password_change", 0.3), ("password_reset", 0.2),
                            ("phone_change", 0.15)):
            if rng.random() < share:
                self.account_event(a, kind, when())
        if rng.random() < 0.35:
            at = when()
            held = [self.rows["accounts"][-1]["email"]]
            for _ in range(rng.randint(1, 3)):
                back = len(held) > 1 and rng.random() < 0.4
                held.append(rng.choice(held[:-1]) if back else self.email(rng.choice(self.names)))
                self.account_event(a, "email_change", at, email=held[-1])
                if rng.random() < 0.3:  # a second change in the same second
                    held.append(self.email(rng.choice(self.names)))
                    self.account_event(a, "email_change", at, email=held[-1])
                at = rng.randint(at, ORDER_END)

    def takeover(self, a: _Account) -> None:
        """Someone else gets in: a new device, a reset, a new drop address and two orders
        that are never paid and that the owner reports."""
        rng = self.rng
        at = START + self.days(1, 45)
        device = self.device(at - 10 * MINUTE)
        link = self.link_device(a, device, at)
        country = rng.choice(FOREIGN)
        self.account_event(a, "login", at + MINUTE, device, country)
        self.account_event(a, "password_reset", at + 3 * MINUTE, device, country)
        if rng.random() < 0.5:
            self.account_event(a, "email_change", at + 4 * MINUTE, device, country,
                               self.email(rng.choice(self.names)))
        drop = self.address(at + 5 * MINUTE)
        self.link_address(a, drop, at + 5 * MINUTE, "shipping")
        for order_at in (at + 10 * MINUTE, at + 10 * MINUTE + self.days(1, 5)):
            self.session(a, order_at, device=device, address=drop, country=country,
                         pays="never", stolen=True)
        if rng.random() < 0.5:
            link["removed_at"] = at + self.days(10, 20)

    # ------------------------------------------------------------------- orders
    def shop(self, a: _Account) -> None:
        rng = self.rng
        if a.first >= ORDER_END:
            return
        count = {"never": rng.randint(1, 3), "fraud": rng.randint(1, 2),
                 "farmer": 1 + (rng.random() < 0.3)}.get(a.behaviour, rng.randint(1, 10))
        moments = sorted(rng.randint(a.first, ORDER_END - 1) for _ in range(count))
        soon = a.created + rng.randint(5 * MINUTE, 2 * HOUR)
        if a.created >= START and soon < ORDER_END and rng.random() < (
                0.9 if a.behaviour in ("fraud", "farmer") else 0.3):
            moments[0] = soon
        # a farmer's first order may come in the second it joins the farm device
        joining = a.joined is not None and a.joined < ORDER_END and rng.random() < 0.6
        for i, at in enumerate(moments):
            country = avoid = None
            if joining and i == 0:
                self.session(a, a.joined, device=self.farm_of[a.user_id])
                continue
            if rng.random() < 0.12:
                tie = rng.choice(self.busy)  # same second as some other event
                at = tie if a.first <= tie < ORDER_END else at
            elif a.due_dates and rng.random() < 0.1:
                due = rng.choice(a.due_dates)  # at an earlier plan's due date
                at = due if due < ORDER_END else at
            elif a.activity and rng.random() < 0.2:
                width = rng.choice(WINDOWS)  # exactly a window after its own activity
                then, avoid = rng.choice(a.activity)  # on another device if it has one
                at = then + width if a.first <= then + width < ORDER_END else at
                if width == 12 * HOUR and rng.random() < 0.7:
                    country = rng.choice(FOREIGN)  # far away exactly 12 hours later
            self.session(a, at, country=country, avoid=avoid)

    def session(self, a: _Account, at: int, device: int | None = None,
                address: int | None = None, country: str | None = None,
                pays: str | None = None, stolen: bool = False, avoid: int | None = None
                ) -> None:
        rng = self.rng
        fraud = a.behaviour == "fraud"
        if device is None:
            links = active(a.devices, at)
            link = rng.choice([r for r in links if r["device_id"] != avoid] or links)
            device = link["device_id"]
            if self.on_device[device] and rng.random() < 0.4:
                # the same second as an attempt on this device, or a day or 30 days later
                moment = rng.choice(self.on_device[device]) + rng.choice((0, 0, DAY, 30 * DAY))
                if (a.first <= moment < ORDER_END and link in active([link], moment)
                        and active(a.cards, moment)):
                    at = moment
        if address is None:
            homes = [r for r in active(a.addresses, at) if r["role"] == "home"]
            others = [r for r in active(a.addresses, at) if r["role"] == "shipping"]
            address = (rng.choice(others) if others and (rng.random() < 0.25 or fraud)
                       else homes[0])["address_id"]
        merchant = self.merchant(at)
        if merchant is None:
            return
        low, high = CATEGORIES[merchant["category"]][1:3]
        amount = rng.randint(low, high)
        promo = self.promotion(a, at)
        country = country or (a.country if rng.random() < 0.88 and not fraud
                              else rng.choice(FOREIGN))
        tries = rng.randint(3, 5) if fraud else (1 if rng.random() < 0.8 else rng.randint(2, 4))
        same_second = rng.random() < 0.5
        risky = fraud or stolen
        for k in range(tries):
            moment = at if k == 0 or same_second else at + rng.randint(1, 900)
            cards = active(a.cards, moment)
            if not cards or moment >= ORDER_END or not any(
                    r["device_id"] == device for r in active(a.devices, moment)):
                break
            last = k == tries - 1
            self.attempt(
                a, moment, merchant, device, cards[k % len(cards)] if fraud else rng.choice(cards),
                address, amount, promo, country,
                avs="N" if rng.random() < (0.4 if risky else 0.04) else "Y",
                cvv="N" if rng.random() < (0.3 if risky else 0.02) else "M",
                declined=(not last and rng.random() < 0.75) or rng.random() < 0.04,
                pays=pays, stolen=stolen)
        if rng.random() < 0.15:
            self.account_event(a, "login", at, device)  # an account event in the same second

    def merchant(self, at: int) -> dict | None:
        open_now = [m for m in self.rows["merchants"] if m["created_at"] <= at
                    and (m["closed_at"] is None or at < m["closed_at"])]
        if not open_now:
            return None
        weights = [CATEGORIES[m["category"]][0] for m in open_now]
        return self.rng.choices(open_now, weights)[0]

    def promotion(self, a: _Account, at: int) -> dict | None:
        rng = self.rng
        valid = [p for p in self.rows["promotions"] if p["valid_from"] <= at < p["valid_to"]]
        first = [p for p in valid if p["first_purchase_only"]]
        if a.behaviour == "farmer" and first and not a.approved:
            return first[0]
        if first and ((not a.approved and rng.random() < 0.3) or rng.random() < 0.03):
            return rng.choice(first)
        other = [p for p in valid if not p["first_purchase_only"]]
        return other[0] if other and rng.random() < 0.15 else None

    def attempt(self, a: _Account, at: int, merchant: dict, device: int, card: dict,
                address: int, amount: int, promo: dict | None, country: str, avs: str,
                cvv: str, declined: bool, pays: str | None, stolen: bool) -> None:
        order_id = self.new_id("order_id")
        discount = ledger.round_bps(amount, promo["discount_bps"]) if promo else 0
        self.event("order_attempts", at, order_id=order_id, user_id=a.user_id,
                   merchant_id=merchant["merchant_id"], device_id=device,
                   card_id=card["card_id"], ship_address_id=address, amount_cents=amount,
                   promo_id=promo["promo_id"] if promo else None, promo_discount_cents=discount,
                   ip=f"10.1.{a.user_id % 250}.{order_id % 250}", ip_country=country,
                   avs_result=avs, cvv_result=cvv,
                   processor_result="declined" if declined else "approved")
        self.on_device[device].append(at)
        a.activity.append((at, device))
        if stolen:
            self.stolen.add(order_id)
        if declined:
            return
        a.approved = True
        self.plan(a, order_id, at, merchant, amount - discount,
                  pays or PAYS.get(a.behaviour, "pays"), stolen)

    def plan(self, a: _Account, order_id: int, at: int, merchant: dict, principal: int,
             pays: str, stolen: bool) -> None:
        rng = self.rng
        plan_id = self.new_id("plan_id")
        parts = ledger.split_principal(principal, TERMS)
        self.rows["plans"].append({"plan_id": plan_id, "order_id": order_id, "created_at": at,
                                   "principal_cents": principal, "down_payment_cents": parts[0],
                                   "n_installments": TERMS.n_installments})
        standing: list[tuple[int, int]] = []  # (known_at, cents): payments and reversals
        for seq, amount in enumerate(parts):
            due = at + seq * TERMS.installment_interval_days * DAY
            self.rows["installment_schedule"].append(
                {"plan_id": plan_id, "seq": seq, "due_at": due, "amount_cents": amount})
            if seq:
                a.due_dates.append(due)
            tries = self.tries(pays, seq)
            for number, (offset, success) in enumerate(tries, 1):
                occurred = due + offset
                known = self.late(occurred, 0.1, MINUTE, DAY)
                payment = self.event("payment_attempts", occurred, known, plan_id=plan_id,
                                     seq=seq, attempt_no=number, amount_cents=amount,
                                     result="success" if success else "failed")
                if not success:
                    continue
                standing.append((known, amount))
                if rng.random() < 0.08:
                    whole = amount < 2 or rng.random() < 0.5
                    back = amount if whole else rng.randint(1, amount - 1)
                    returned = known + rng.randint(HOUR, 5 * DAY)
                    returned_known = self.late(returned, 0.7, HOUR, 4 * DAY)
                    self.event("payment_reversals", returned, returned_known,
                               payment_event_id=payment, plan_id=plan_id, amount_cents=back,
                               reason=rng.choice(("bank_return", "card_reversal")))
                    standing.append((returned_known, -back))
                    if rng.random() < 0.5:  # collected again
                        again = returned_known + 2 * DAY
                        self.event("payment_attempts", again, again, plan_id=plan_id, seq=seq,
                                   attempt_no=len(tries) + 1, amount_cents=back,
                                   result="success")
                        standing.append((again, back))
        written_off = at + (TERMS.n_installments * TERMS.installment_interval_days
                            + TERMS.writeoff_after_days) * DAY
        collected = sum(cents for known, cents in standing if known <= written_off)
        if collected < principal:
            self.event("plan_writeoffs", written_off, plan_id=plan_id,
                       outstanding_cents=principal - collected)
        self.after_checkout(a, order_id, at, merchant, parts[0], pays, stolen)

    def tries(self, pays: str, seq: int) -> list[tuple[int, bool]]:
        """(seconds after the due date, success) of each collection attempt."""
        rng = self.rng
        if seq == 0:
            return [(0, True)]
        if pays == "late":
            return [(0, False), (self.days(2, 6), True)]
        if pays == "never" or (pays == "partial" and seq > 1):
            return [(0, False), (3 * DAY, False)]
        return [(0, True)] if rng.random() < 0.9 else [(0, False), (3 * DAY, True)]

    def after_checkout(self, a: _Account, order_id: int, at: int, merchant: dict,
                       collected: int, pays: str, stolen: bool) -> None:
        rng = self.rng
        closes = merchant["closed_at"]
        shipped = at + int(merchant["fulfilment_median_hours"] * HOUR * rng.uniform(0.3, 2.0))
        unshipped = 0.2 if pays == "never" else 0.03
        if rng.random() < unshipped or (closes is not None and shipped >= closes):
            shipped = None
        else:
            self.event("fulfilments", shipped, self.late(shipped, 0.25, MINUTE, 2 * DAY),
                       order_id=order_id)
        delivered = None
        closing = closes is not None and closes - at < 25 * DAY
        if shipped is not None and rng.random() < (0.4 if closing else 0.92):
            delivered = shipped + self.days(1, 6)
            self.event("deliveries", delivered, self.late(delivered, 0.4, HOUR, 3 * DAY),
                       order_id=order_id)
        reason = None
        if (a.behaviour == "fraud" and rng.random() < 0.7) or (stolen and rng.random() < 0.3):
            reason = "unauthorized"
        elif a.behaviour == "inr" and delivered is not None and rng.random() < 0.6:
            reason = "item_not_received"
        elif rng.random() < 0.06:
            reason = rng.choice(world.DISPUTE_REASONS)
        if reason is not None:
            filed = (delivered or shipped or at) + self.days(1, 30)
            notified = filed + rng.randint(0, 5 * DAY)
            dispute_id = self.new_id("dispute_id")
            self.event("dispute_openings", filed, notified, dispute_id=dispute_id,
                       order_id=order_id, reason=reason, amount_cents=collected)
            if rng.random() < 0.85:
                upheld = {"unauthorized": 0.75}.get(reason, 0.15 if a.behaviour == "inr" else 0.5)
                decided = notified + self.days(7, 45)
                self.event("dispute_resolutions", decided, decided + rng.randint(0, 3 * DAY),
                           dispute_id=dispute_id,
                           outcome="lost" if rng.random() < upheld else "won")
        if stolen:
            reported = at + self.days(2, 12)
            self.event("victim_reports", reported, self.late(reported, 0.5, HOUR, 2 * DAY),
                       user_id=a.user_id, order_id=order_id)

    # ------------------------------------------------------------------- tables
    def tables(self) -> dict[str, pd.DataFrame]:
        rng = self.rng
        count = self.last_id["event_id"]
        final = list(range(1, count + 1))
        rng.shuffle(final)  # event ids carry no trace of the order they were made in
        out: dict[str, pd.DataFrame] = {}
        for name in world.OBSERVABLE_TABLES:
            if name == "cash_events":
                continue
            spec = world.TABLES[name]
            frame = pd.DataFrame(self.rows[name], columns=list(spec.column_names))
            for column in ("event_id", "payment_event_id"):
                if column in frame:
                    frame[column] = [final[i - 1] for i in frame[column]]
            for column in spec.columns:
                if column.type == "ts":
                    frame[column.name] = times(frame[column.name])
            out[name] = world.coerce(name, frame)
        cash = ledger.derive_cash_events(out, TERMS)
        cash_ids = list(range(count + 1, count + 1 + len(cash)))
        rng.shuffle(cash_ids)
        out["cash_events"] = world.coerce("cash_events", cash.assign(event_id=cash_ids))
        out["labels"] = world.adjudicate(
            out, horizon_days=HORIZON_DAYS, observed_until=pd.Timestamp(OBSERVED_UNTIL, unit="s"),
            **config.load("world")["labels"])
        out.update(self.latent(out["order_attempts"]))
        return out

    def latent(self, orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Simulator truth: one episode per fraudulent account or farm."""
        episode: dict[object, int] = {}
        episodes, accounts = [], []
        for user_id, behaviour in self.profile.items():
            key = ("farm", self.farm_of[user_id]) if behaviour == "farmer" else user_id
            if behaviour in PATTERNS and key not in episode:
                episode[key] = len(episode) + 1
                episodes.append({"episode_id": episode[key], "pattern_id": PATTERNS[behaviour],
                                 "started_at": START, "ended_at": ORDER_END})
            accounts.append({"user_id": user_id, "episode_id": episode.get(key),
                             "actor": "fraudster" if behaviour in ("fraud", "farmer", "never")
                             else "legitimate", "profile": behaviour})
        rows = []
        for order_id, user_id in zip(orders["order_id"], orders["user_id"], strict=True):
            behaviour = self.profile[user_id]
            bad = behaviour in PATTERNS and (behaviour != "takeover" or order_id in self.stolen)
            key = ("farm", self.farm_of[user_id]) if behaviour == "farmer" else user_id
            rows.append({"order_id": order_id, "pattern_id": PATTERNS[behaviour] if bad else None,
                         "episode_id": episode[key] if bad else None,
                         "intent": ("abuse" if behaviour in ("farmer", "inr") else "fraud")
                         if bad else "legitimate", "mimic": None})
        frames = {"latent_episodes": pd.DataFrame(episodes), "latent_accounts":
                  pd.DataFrame(accounts), "latent_orders": pd.DataFrame(rows)}
        for name, frame in frames.items():
            for column in world.TABLES[name].columns:
                if column.type == "ts":
                    frame[column.name] = times(frame[column.name])
            frames[name] = world.coerce(name, frame[list(world.TABLES[name].column_names)])
        return frames


def random_world(seed: int, scale: float = 1.0) -> dict[str, pd.DataFrame]:
    """A random world in the core.world schema (every table, labels and latent truth
    included); the same seed and scale give the same world."""
    return _Builder(seed, scale).tables()


# ---------------------------------------------------------------- cutting a world
def known_by(tables: dict[str, pd.DataFrame], cut) -> dict[str, pd.DataFrame]:
    """The observable world as known at ``cut``: a time (every event known by then) or a
    place ``(known_at, kind rank, event_id)`` in the event order (the events up to and
    including it). Entities created by then are kept, also within the cut's second, a plan
    only with its order; later removal and closure times are cleared."""
    place = cut if isinstance(cut, tuple) else None
    second = int(seconds([place[0] if place else cut])[0])
    out = {}
    for name in world.OBSERVABLE_TABLES:
        frame = tables[name]
        if name in world.EVENT_RANK:
            known = seconds(frame["known_at"])
            keep = known <= second
            if place is not None:
                rank, event_id = world.EVENT_RANK[name], frame["event_id"].to_numpy()
                keep = (known < second) | ((known == second) & (
                    (rank < place[1]) | ((rank == place[1]) & (event_id <= place[2]))))
        elif name in CREATED:
            keep = seconds(frame[CREATED[name]]) <= second
        else:
            keep = np.ones(len(frame), bool)
        frame = frame[keep].copy()
        if name in ENDS:
            later = seconds(frame[ENDS[name]]) > second
            frame.loc[later, ENDS[name]] = pd.NaT
        out[name] = frame
    plans = out["plans"]
    out["plans"] = plans[plans["order_id"].isin(out["order_attempts"]["order_id"])]
    schedule = out["installment_schedule"]
    out["installment_schedule"] = schedule[schedule["plan_id"].isin(out["plans"]["plan_id"])]
    return {name: frame.reset_index(drop=True) for name, frame in out.items()}


# ------------------------------------------------------------------ a policy's world
@dataclass(frozen=True)
class Act:
    """What a policy did with one processor-approved order; times are seconds after its
    checkout.

    ``decline``: never let through. ``void``: let through at checkout, voided at ``at``.
    ``hold``: held at ``at``, released at ``release`` (None while pending) with
    ``outcome`` cleared, cancelled or declined; a hold before the order's shipment pauses
    it, one after shipment pauses nothing. ``block``: the account is blocked at the void
    or the release.
    """

    kind: str
    at: int = 0
    release: int | None = None
    outcome: str | None = None
    block: bool = False


def _first_shipment(tables: dict[str, pd.DataFrame]) -> dict[int, int]:
    fulfilments = tables["fulfilments"]
    shipped = pd.Series(seconds(fulfilments["occurred_at"]),
                        index=fulfilments["order_id"].to_numpy())
    return shipped.groupby(level=0).min().to_dict()


def random_acts(tables: dict[str, pd.DataFrame], seed: int) -> dict[int, Act]:
    """A random policy: declines, voids (of orders that never ship also around their
    plan's default), holds before and after shipment (cleared, cancelled, declined or
    pending) and blocks; other orders approved at checkout."""
    rng = random.Random(seed)
    orders = tables["order_attempts"]
    orders = orders[orders["processor_result"] == "approved"]
    shipped = _first_shipment(tables)
    default = (TERMS.installment_interval_days
               + config.load("world")["labels"]["default_grace_days"]) * DAY  # after checkout
    acts = {}
    for order_id, checkout in zip(orders["order_id"], seconds(orders["known_at"]), strict=True):
        # seconds to shipment; an order that never ships may be stopped after its default
        before = int(shipped.get(order_id, checkout + 60 * DAY) - checkout)
        draw = rng.random()
        if draw < 0.06:
            acts[order_id] = Act("decline")
        elif order_id not in shipped and draw < 0.5:  # stopped around its default
            acts[order_id] = Act("void", default + rng.choice((-1, 0, 1, DAY, -DAY)))
        elif draw < 0.12 and before > 1:
            acts[order_id] = Act("void", rng.randint(1, before - 1), block=rng.random() < 0.5)
        elif draw < 0.24 and before > 1:
            outcome = rng.choices(("cleared", "cancelled", "declined", None), (6, 2, 1, 1))[0]
            at = rng.randint(1, before - 1)
            release = None if outcome is None else at + rng.randint(HOUR, 48 * HOUR)
            acts[order_id] = Act("hold", at, release, outcome, outcome == "declined")
        elif draw < 0.3 and order_id in shipped:
            at = before + rng.randint(HOUR, 3 * DAY)
            outcome = rng.choice(("cleared", "declined"))
            acts[order_id] = Act("hold", at, at + rng.randint(HOUR, 48 * HOUR), outcome,
                                 outcome == "declined")
    return acts


ORDER_EVENT_TABLES = ("fulfilments", "deliveries", "payment_attempts", "payment_reversals",
                      "plan_writeoffs", "dispute_openings", "dispute_resolutions",
                      "victim_reports", "cash_events")


def random_blocks(tables: dict[str, pd.DataFrame], seed: int, count: int = 15
                  ) -> list[tuple[int, int]]:
    """(user_id, seconds) blocks in the second of an order's checkout (or one second off):
    mostly of an account linked to it by its device or address in the 30 days before,
    sometimes of the ordering account itself."""
    rng = random.Random(seed)
    orders = tables["order_attempts"]
    events = tables["account_events"]
    near = pd.concat([
        pd.DataFrame({"place": "d" + orders["device_id"].astype(str), "user_id": orders["user_id"],
                      "s": seconds(orders["known_at"])}),
        pd.DataFrame({"place": "d" + events["device_id"].astype(str), "user_id": events["user_id"],
                      "s": seconds(events["known_at"])}),
        pd.DataFrame({"place": "a" + orders["ship_address_id"].astype(str),
                      "user_id": orders["user_id"], "s": seconds(orders["known_at"])})])
    blocks = []
    for row in orders.sample(frac=1, random_state=seed).itertuples():
        t = int(seconds([row.known_at])[0])
        places = {f"d{row.device_id}", f"a{row.ship_address_id}"}
        linked = near[near["place"].isin(places) & (near["user_id"] != row.user_id)
                      & (near["s"] >= t - 30 * DAY) & (near["s"] <= t)]["user_id"].unique()
        if len(linked) or rng.random() < 0.2:
            user = int(rng.choice(list(linked))) if len(linked) else int(row.user_id)
            blocks.append((user, t + rng.choice((0, 0, -1, 1))))
        if len(blocks) == count:
            break
    return blocks


def realize(tables: dict[str, pd.DataFrame], acts: dict[int, Act],
            blocks: Iterable[tuple[int, int]] = ()
            ) -> tuple[dict[str, pd.DataFrame], asof.PolicyState]:
    """A policy's realized tables and state under ``acts`` (processor-approved orders
    without an act are approved at checkout), built as the replay would: an order the
    policy never let through keeps its attempt but loses its plan, schedule, payments,
    shipment, delivery, disputes and reports; a voided order loses what happens after
    the void; an order released from a hold before shipment has its later events and
    its schedule moved by the time it was held. ``blocks`` adds (user_id, seconds)
    blocks."""
    orders = tables["order_attempts"]
    checkout = dict(zip(orders["order_id"], seconds(orders["known_at"]), strict=True))
    owner = dict(zip(orders["order_id"], orders["user_id"], strict=True))
    shipped = _first_shipment(tables)
    approved, voided, blocked, held = [], [], list(blocks), []
    gone, void_at, moved = set(), {}, {}
    for order_id in orders.loc[orders["processor_result"] == "approved", "order_id"]:
        act = acts.get(order_id, Act("approve"))
        t = checkout[order_id]
        if act.kind == "approve":
            approved.append((order_id, t))
        elif act.kind == "decline":
            gone.add(order_id)
        elif act.kind == "void":
            approved.append((order_id, t))
            voided.append((order_id, t + act.at))
            void_at[order_id] = t + act.at
        else:
            before = t + act.at < shipped.get(order_id, np.iinfo(np.int64).max)
            release = None if act.release is None else t + act.release
            held.append((order_id, t + act.at, release, act.outcome, before))
            if not before:
                approved.append((order_id, t))
            elif act.outcome == "cleared":
                approved.append((order_id, release))
                moved[order_id] = act.release
            else:
                gone.add(order_id)
        if act.block:
            blocked.append((owner[order_id], t + (act.release if act.kind == "hold" else act.at)))

    out = dict(tables)
    plans = tables["plans"]
    plan_order = dict(zip(plans["plan_id"], plans["order_id"], strict=True))
    openings = tables["dispute_openings"]
    dispute_order = dict(zip(openings["dispute_id"], openings["order_id"], strict=True))
    for name in ("plans", "installment_schedule", *ORDER_EVENT_TABLES):
        frame = tables[name].copy()
        if "order_id" in frame:
            order = frame["order_id"]
        elif "plan_id" in frame:
            order = frame["plan_id"].map(plan_order)
        else:
            order = frame["dispute_id"].map(dispute_order)
        keep = ~order.isin(gone).to_numpy()
        start = order.map(checkout).to_numpy(np.int64)
        shift = order.map(moved).fillna(0).to_numpy(np.int64)
        moves = {}  # column -> rows after checkout, which a released hold moves
        if name in world.EVENT_RANK:
            occurred = seconds(frame["occurred_at"])
            keep &= ~(occurred > order.map(void_at).fillna(np.inf).to_numpy())
            moves = {"occurred_at": occurred > start, "known_at": occurred > start}
        elif name == "installment_schedule":
            moves = {"due_at": seconds(frame["due_at"]) > start}
        for column, after in moves.items():
            frame[column] = (seconds(frame[column]) + np.where(after, shift, 0)).astype(
                "datetime64[s]")
        out[name] = frame[keep].reset_index(drop=True)
    state = asof.PolicyState(
        approved=_frame_of(approved, order_id="int64", approved_at="ts"),
        voided=_frame_of(voided, order_id="int64", at="ts"),
        blocked=_frame_of(blocked, user_id="int64", at="ts"),
        held=_frame_of(held, order_id="int64", held_at="ts", released_at="ts",
                       outcome="object", before_shipment="bool"))
    return out, state


def _frame_of(rows: list[tuple], **columns: str) -> pd.DataFrame:
    """A frame of ``rows`` with these columns ("ts": seconds made datetime64[s])."""
    return pd.DataFrame({
        name: times([row[i] for row in rows]) if dtype == "ts"
        else pd.Series([row[i] for row in rows], dtype=dtype)
        for i, (name, dtype) in enumerate(columns.items())})


def later_decisions(tables: dict[str, pd.DataFrame], seed: int, count: int,
                    acts: dict[int, Act] | None = None) -> pd.DataFrame:
    """Review-like decisions: ``count`` orders, each decided some hours to weeks after
    checkout, often exactly at (or a second before) a time that matters to it (a due date
    or an event of the account's orders, what the policy did, or a day or 30 days after an
    attempt or account event of the account, on the device or to the address, so windows
    end exactly there), sometimes at checkout itself."""
    rng = random.Random(seed)
    orders = tables["order_attempts"]
    user_of = dict(zip(orders["order_id"], orders["user_id"], strict=True))
    moments: dict[int, list[int]] = defaultdict(list)  # per account
    plans = tables["plans"].merge(orders[["order_id", "user_id"]], on="order_id")
    schedule = tables["installment_schedule"].merge(plans[["plan_id", "user_id"]], on="plan_id")
    for user_id, due in zip(schedule["user_id"], seconds(schedule["due_at"]), strict=True):
        moments[user_id].append(due)
    for name in ("fulfilments", "deliveries", "dispute_openings", "victim_reports"):
        frame = tables[name].merge(orders[["order_id"]], on="order_id")
        for order_id, known in zip(frame["order_id"], seconds(frame["known_at"]), strict=True):
            moments[user_of[order_id]].append(known)
    for name in ("payment_attempts", "payment_reversals", "plan_writeoffs"):
        frame = tables[name].merge(plans[["plan_id", "user_id"]], on="plan_id")
        for user_id, known in zip(frame["user_id"], seconds(frame["known_at"]), strict=True):
            moments[user_id].append(known)
    checkout = dict(zip(orders["order_id"], seconds(orders["known_at"]), strict=True))
    starts: dict[tuple, list[int]] = defaultdict(list)  # times windows may start at
    for order_id, act in (acts or {}).items():
        for offset in (act.at, act.at - 1, act.release):
            if offset is not None and offset > 0:
                moments[user_of[order_id]].append(checkout[order_id] + offset)
        if act.release is not None:
            starts["user", user_of[order_id]].append(checkout[order_id] + act.release)
    where = {}
    for order_id, user_id, device, address in zip(orders["order_id"], orders["user_id"],
                                                  orders["device_id"], orders["ship_address_id"],
                                                  strict=True):
        where[order_id] = (("user", user_id), ("device", device), ("address", address))
        for key in where[order_id]:
            starts[key].append(checkout[order_id])
    events = tables["account_events"]
    for device, known in zip(events["device_id"], seconds(events["known_at"]), strict=True):
        starts["device", device].append(known)
    chosen = rng.sample(list(orders["order_id"]), min(count, len(orders)))
    rows = []
    for order_id in chosen:
        t = checkout[order_id]
        near = [m - before for m in moments[user_of[order_id]] for before in (0, 1)
                if t <= m - before <= t + 45 * DAY]  # at the moment or a second before it
        edges = [m + width for key in where[order_id] for m in starts[key]
                 for width in (DAY, 30 * DAY) if t <= m + width <= t + 45 * DAY]
        draw = rng.random()
        if draw < 0.1:
            at = t
        elif draw < 0.3:
            at = t + rng.randint(MINUTE, 72 * HOUR)
        elif draw < 0.75 and near:
            at = rng.choice(near)
        elif draw >= 0.75 and edges:
            at = rng.choice(edges)
        else:
            at = t + rng.randint(3 * DAY, 42 * DAY)
        rows.append((order_id, at))
    rows = sorted(set(rows))
    return pd.DataFrame({"order_id": [r[0] for r in rows],
                         "decision_at": times([r[1] for r in rows])})
