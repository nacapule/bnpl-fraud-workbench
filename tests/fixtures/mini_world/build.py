"""Build the mini world: a handful of short stories in the world contract.

Run ``python tests/fixtures/mini_world/build.py`` to rewrite the CSV files and
``manifest.json`` beside this script. The stories are listed in README.md;
tests check that the files equal a fresh build, pass the validator and load
into MySQL. Cash events come from core.ledger, labels from
core.world.adjudicate, and every id is assigned after sorting by time.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from core import ledger, world  # noqa: E402

T = pd.Timestamp
H = pd.Timedelta
TERMS = ledger.ProductTerms.from_config()
SEED = 0
FAMILY = "baseline"
ORDER_START = "2025-12-01 00:00:00"
ORDER_END = "2026-06-30 00:00:00"
OBSERVED_UNTIL = T("2026-06-30 23:59:59")
HORIZON_DAYS = 60
ENTITY_IDS = {
    "accounts": "user_id",
    "merchants": "merchant_id",
    "devices": "device_id",
    "addresses": "address_id",
    "cards": "card_id",
    "promotions": "promo_id",
}


class MiniWorld:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.next_event = 0
        self.orders: dict[str, dict[str, Any]] = {}
        self.disputes: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- entities
    def entity(self, table: str, key: str, created: str, **values: Any) -> str:
        column = "valid_from" if table == "promotions" else "created_at"
        self.rows[table].append({"_key": key, column: T(created), **values})
        return key

    def account(self, key: str, created: str, country: str = "US", dob: int = 1988) -> str:
        email = f"{key}.{dob % 100}@{'gmail.com' if len(key) % 2 else 'outlook.com'}"
        self.entity("accounts", key, created, email=email, email_domain=email.split("@")[1],
                    home_country=country, dob_year=dob)
        return key

    def device(self, key: str, created: str, ua: str = "iOS") -> str:
        return self.entity("devices", key, created, fingerprint=f"fp-{key}", ua_family=ua)

    def address(self, key: str, created: str, city: str = "Denver", region: str = "CO",
                country: str = "US") -> str:
        return self.entity("addresses", key, created, line_hash=f"h-{key}", city=city,
                           region=region, country=country)

    def card(self, key: str, user: str, created: str, bin_country: str = "US") -> str:
        return self.entity("cards", key, created, user_id=user, removed_at=None,
                           bin_country=bin_country, network="visa", last4=f"{len(key):04d}")

    def link_device(self, user: str, device: str, created: str) -> None:
        self.rows["device_links"].append({"user_id": user, "device_id": device,
                                          "created_at": T(created), "removed_at": None})

    def link_address(self, user: str, address: str, created: str, role: str = "home") -> None:
        self.rows["address_links"].append({"user_id": user, "address_id": address,
                                           "created_at": T(created), "removed_at": None,
                                           "role": role})

    def customer(self, key: str, created: str, country: str = "US", **kw: Any) -> str:
        """An account with its own device, home address and card, all from signup."""
        self.account(key, created, country=country, **kw)
        self.device(f"d_{key}", created)
        self.link_device(key, f"d_{key}", created)
        self.address(f"a_{key}", created)
        self.link_address(key, f"a_{key}", created)
        self.card(f"c_{key}", key, str(T(created) + H(minutes=5)))
        return key

    # ------------------------------------------------------------------ events
    def event(self, table: str, occurred: pd.Timestamp, known: pd.Timestamp | None = None,
              **values: Any) -> int:
        self.next_event += 1
        known = occurred if known is None else known
        self.rows[table].append({"event_id": self.next_event, "occurred_at": occurred,
                                 "known_at": known, **values})
        return self.next_event

    def account_event(self, user: str, kind: str, at: str, device: str | None = None,
                      ip: str = "24.16.4.9", ip_country: str = "US") -> None:
        self.event("account_events", T(at), user_id=user, kind=kind,
                   device_id=device or f"d_{user}", ip=ip, ip_country=ip_country)

    def order(self, key: str, user: str, merchant: str, at: str, amount_cents: int, *,
              device: str | None = None, card: str | None = None, address: str | None = None,
              ip: str = "24.16.4.9", ip_country: str = "US", avs: str = "Y", cvv: str = "M",
              promo: str | None = None, declined: bool = False, ship_hours: float | None = 12,
              deliver_days: float | None = 2) -> None:
        when = T(at)
        discount = ledger.round_bps(amount_cents, 1000) if promo else 0
        self.event("order_attempts", when, _key=key, user_id=user, merchant_id=merchant,
                   device_id=device or f"d_{user}", card_id=card or f"c_{user}",
                   ship_address_id=address or f"a_{user}", amount_cents=amount_cents,
                   promo_id=promo, promo_discount_cents=discount, ip=ip, ip_country=ip_country,
                   avs_result=avs, cvv_result=cvv,
                   processor_result="declined" if declined else "approved")
        if declined:
            return
        principal = amount_cents - discount
        parts = ledger.split_principal(principal, TERMS)
        schedule = [
            {"seq": seq, "due_at": when + H(days=seq * TERMS.installment_interval_days),
             "amount_cents": amount}
            for seq, amount in enumerate(parts)
        ]
        self.orders[key] = {"at": when, "principal": principal, "schedule": schedule,
                            "payments": [], "merchant": merchant}
        self.rows["plans"].append({"_key": key, "order_id": key, "created_at": when,
                                   "principal_cents": principal,
                                   "down_payment_cents": parts[0],
                                   "n_installments": TERMS.n_installments})
        for entry in schedule:
            self.rows["installment_schedule"].append({"plan_id": key, **entry})
        self.pay(key, 0, when)
        if ship_hours is not None:
            shipped = when + H(hours=ship_hours)
            self.event("fulfilments", shipped, order_id=key)
            if deliver_days is not None:
                self.event("deliveries", shipped + H(days=deliver_days), order_id=key)

    def due(self, order: str, seq: int) -> pd.Timestamp:
        return self.orders[order]["schedule"][seq]["due_at"]

    def pay(self, order: str, seq: int, at: pd.Timestamp | str | None = None,
            result: str = "success", attempt_no: int = 1) -> int:
        when = self.due(order, seq) if at is None else T(at)
        amount = self.orders[order]["schedule"][seq]["amount_cents"]
        event_id = self.event("payment_attempts", when, plan_id=order, seq=seq,
                              attempt_no=attempt_no, amount_cents=amount, result=result)
        if result == "success":
            self.orders[order]["payments"].append([event_id, when, amount, None])
        return event_id

    def repay(self, order: str, seqs: range | list[int] | None = None) -> None:
        for seq in seqs or range(1, TERMS.n_installments + 1):
            self.pay(order, seq)

    def miss(self, order: str, seqs: range | list[int] | None = None) -> None:
        """Each installment fails on its due date and again on a retry three days later."""
        for seq in seqs or range(1, TERMS.n_installments + 1):
            self.pay(order, seq, result="failed")
            self.pay(order, seq, self.due(order, seq) + H(days=3), result="failed",
                     attempt_no=2)

    def reverse(self, order: str, payment_event: int, at: str, reason: str) -> None:
        for payment in self.orders[order]["payments"]:
            if payment[0] == payment_event:
                payment[3] = T(at)
                self.event("payment_reversals", T(at), payment_event_id=payment_event,
                           plan_id=order, amount_cents=payment[2], reason=reason)
                return
        raise KeyError(payment_event)

    def collected(self, order: str, by: pd.Timestamp) -> int:
        return sum(amount for _, at, amount, reversed_at in self.orders[order]["payments"]
                   if at <= by and not (reversed_at is not None and reversed_at <= by))

    def write_off(self, order: str) -> None:
        last_due = self.due(order, TERMS.n_installments)
        when = last_due + H(days=TERMS.writeoff_after_days)
        outstanding = self.orders[order]["principal"] - self.collected(order, when)
        self.event("plan_writeoffs", when, plan_id=order, outstanding_cents=outstanding)

    def dispute(self, order: str, reason: str, filed: str, notified: str,
                resolved: tuple[str, str] | None = None, outcome: str | None = None) -> None:
        amount = self.collected(order, T(notified))
        key = f"{order}-{reason}"
        self.event("dispute_openings", T(filed), T(notified), _key=key, dispute_id=key,
                   order_id=order, reason=reason, amount_cents=amount)
        if resolved is not None:
            self.event("dispute_resolutions", T(resolved[0]), T(resolved[1]), dispute_id=key,
                       outcome=outcome)

    # -------------------------------------------------------------- finishing
    def tables(self) -> dict[str, pd.DataFrame]:
        ids: dict[tuple[str, str], int] = {}
        frames: dict[str, pd.DataFrame] = {}
        for table, id_column in ENTITY_IDS.items():
            frame = pd.DataFrame(self.rows[table])
            created = "valid_from" if table == "promotions" else "created_at"
            frame = frame.sort_values([created, "_key"], kind="stable").reset_index(drop=True)
            frame[id_column] = range(1, len(frame) + 1)
            ids.update({(id_column, key): i for key, i in zip(frame["_key"], frame[id_column],
                                                               strict=True)})
            frames[table] = frame.drop(columns="_key")
        orders = pd.DataFrame(self.rows["order_attempts"]).sort_values(
            ["occurred_at", "_key"], kind="stable")
        ids.update({("order_id", k): i for i, k in enumerate(orders["_key"], start=1)})
        plans = pd.DataFrame(self.rows["plans"]).sort_values(["created_at", "_key"],
                                                              kind="stable")
        ids.update({("plan_id", k): i for i, k in enumerate(plans["_key"], start=1)})
        openings = pd.DataFrame(self.rows["dispute_openings"]).sort_values(
            ["known_at", "_key"], kind="stable")
        ids.update({("dispute_id", k): i for i, k in enumerate(openings["_key"], start=1)})
        frames["order_attempts"] = orders.assign(order_id=orders["_key"]).drop(columns="_key")
        frames["plans"] = plans.assign(plan_id=plans["_key"]).drop(columns="_key")
        frames["dispute_openings"] = openings.drop(columns="_key")

        for table in world.TABLES:
            if table in frames or table in ("cash_events", "labels") or world.TABLES[
                table
            ].layer == "latent":
                continue
            frames[table] = pd.DataFrame(self.rows[table])
        references = {
            "user_id": "user_id", "merchant_id": "merchant_id", "device_id": "device_id",
            "card_id": "card_id", "ship_address_id": "address_id", "address_id": "address_id",
            "promo_id": "promo_id", "order_id": "order_id", "plan_id": "plan_id",
            "dispute_id": "dispute_id",
        }
        for frame in frames.values():
            for column, kind in references.items():
                if column in frame.columns:
                    frame[column] = frame[column].map(
                        lambda key, kind=kind: ids[(kind, key)] if isinstance(key, str) else key
                    )
        for table in frames:
            frames[table] = world.coerce(table, frames[table])
        frames["cash_events"] = ledger.derive_cash_events(frames, TERMS)
        frames = world.renumber_events(frames)
        frames["cash_events"] = world.coerce("cash_events", frames["cash_events"])
        frames["labels"] = world.adjudicate(frames, horizon_days=HORIZON_DAYS,
                                            observed_until=OBSERVED_UNTIL)
        frames.update(self.latent(frames, ids))
        return {name: world.coerce(name, frames[name]) for name in world.TABLES}

    def latent(self, frames: dict[str, pd.DataFrame], ids: dict) -> dict[str, pd.DataFrame]:
        episodes = pd.DataFrame(self.rows["latent_episodes"]).sort_values("started_at")
        episodes["episode_id"] = range(1, len(episodes) + 1)
        episode_ids = dict(zip(episodes["_key"], episodes["episode_id"], strict=True))
        accounts = pd.DataFrame(self.rows["latent_accounts"])
        accounts["user_id"] = [ids[("user_id", k)] for k in accounts["user_id"]]
        accounts["episode_id"] = accounts["episode_id"].map(episode_ids)
        orders = pd.DataFrame(self.rows["latent_orders"])
        orders["order_id"] = [ids[("order_id", k)] for k in orders["order_id"]]
        orders["episode_id"] = orders["episode_id"].map(episode_ids)
        return {
            "latent_episodes": episodes.drop(columns="_key"),
            "latent_accounts": accounts.sort_values("user_id"),
            "latent_orders": orders.sort_values("order_id"),
        }

    def truth(self, user: str, actor: str = "legitimate", episode: str | None = None,
              profile: str | None = None) -> None:
        self.rows["latent_accounts"].append({"user_id": user, "actor": actor,
                                             "episode_id": episode, "profile": profile})

    def order_truth(self, order: str, pattern: str | None = None, episode: str | None = None,
                    intent: str = "legitimate", mimic: str | None = None) -> None:
        self.rows["latent_orders"].append({"order_id": order, "pattern_id": pattern,
                                           "episode_id": episode, "intent": intent,
                                           "mimic": mimic})

    def episode(self, key: str, pattern: str, started: str, ended: str) -> str:
        self.rows["latent_episodes"].append({"_key": key, "pattern_id": pattern,
                                             "started_at": T(started), "ended_at": T(ended)})
        return key


def build() -> dict[str, pd.DataFrame]:
    w = MiniWorld()
    w.entity("merchants", "m_home", "2024-03-01 09:00", name="north-goods-14", category="home",
             risk_tier=1, fulfilment_median_hours=12.0, closed_at=None)
    w.entity("merchants", "m_tech", "2024-06-10 09:00", name="volt-tech-31",
             category="electronics", risk_tier=2, fulfilment_median_hours=18.0, closed_at=None)
    w.entity("merchants", "m_flash", "2025-12-15 09:00", name="flash-gems-70",
             category="jewelry", risk_tier=3, fulfilment_median_hours=10.0,
             closed_at=T("2026-02-20 12:00"))
    w.entity("promotions", "first10", "2025-12-01 00:00", code="FIRST10", discount_bps=1000,
             first_purchase_only=True, valid_to=T("2026-12-01 00:00"))

    # ana: long-tenured customer; one installment bounces and is paid on retry.
    w.customer("ana", "2024-05-02 10:00")
    w.account_event("ana", "login", "2026-01-05 19:10")
    w.order("ana1", "ana", "m_home", "2026-01-05 19:20", 8450, ship_hours=14, deliver_days=3)
    w.repay("ana1")
    w.order("ana2", "ana", "m_tech", "2026-02-10 20:05", 43000, ship_hours=20)
    bounced = w.pay("ana2", 1)
    w.reverse("ana2", bounced, "2026-02-26 20:05", "bank_return")
    w.pay("ana2", 1, "2026-02-28 20:05", attempt_no=2)
    w.repay("ana2", [2, 3])
    w.truth("ana")
    w.order_truth("ana1")
    w.order_truth("ana2")

    # cara and ben: a household sharing an address and a tablet; ben is a new
    # customer whose first order is large. Both use the first-purchase promotion.
    w.customer("cara", "2025-08-20 14:00")
    w.device("d_tablet", "2025-08-20 14:20", ua="Android")
    w.link_device("cara", "d_tablet", "2025-08-20 14:20")
    w.account_event("cara", "email_change", "2026-01-02 08:30")
    w.account("ben", "2026-01-12 18:00", dob=1999)
    w.device("d_ben", "2026-01-12 18:00", ua="Android")
    w.link_device("ben", "d_ben", "2026-01-12 18:00")
    w.link_address("ben", "a_cara", "2026-01-12 18:00")
    w.card("c_ben", "ben", "2026-01-12 18:05")
    w.order("ben1", "ben", "m_tech", "2026-01-12 18:40", 129999, address="a_cara",
            promo="first10", ship_hours=16)
    w.repay("ben1")
    w.link_device("ben", "d_tablet", "2026-01-20 09:00")
    w.order("cara1", "cara", "m_home", "2026-01-20 19:00", 6200, device="d_tablet",
            promo="first10", ship_hours=10)
    w.repay("cara1")
    w.truth("cara", profile="household")
    w.truth("ben", profile="new_customer")
    w.order_truth("ben1", mimic="large_first_order")
    w.order_truth("cara1", mimic="household_promotion")

    # dan: account takeover. New device, password reset, new drop address, one order.
    w.episode("e_ato", "P-ATO", "2026-02-03 02:10", "2026-02-03 02:20")
    w.customer("dan", "2023-09-01 08:00", dob=1971)
    w.order("dan1", "dan", "m_home", "2025-12-20 13:00", 4500)
    w.repay("dan1")
    w.device("d_ato", "2026-02-03 02:10", ua="Chrome")
    w.link_device("dan", "d_ato", "2026-02-03 02:10")
    w.account_event("dan", "login", "2026-02-03 02:10", device="d_ato", ip="185.12.9.40",
                    ip_country="RO")
    w.account_event("dan", "password_reset", "2026-02-03 02:12", device="d_ato",
                    ip="185.12.9.40", ip_country="RO")
    w.address("a_drop", "2026-02-03 02:15", city="Houston", region="TX")
    w.link_address("dan", "a_drop", "2026-02-03 02:15", role="shipping")
    w.order("dan2", "dan", "m_tech", "2026-02-03 02:20", 89900, device="d_ato",
            address="a_drop", ip="185.12.9.40", ip_country="RO", ship_hours=6)
    w.event("victim_reports", T("2026-02-09 10:00"), user_id="dan", order_id="dan2")
    w.miss("dan2")
    w.write_off("dan2")
    w.truth("dan")
    w.order_truth("dan1")
    w.order_truth("dan2", "P-ATO", "e_ato", "fraud")

    # eve: new account testing stolen cards; two processor declines, then one approval.
    w.episode("e_stolen", "P-STOLEN", "2026-02-14 01:00", "2026-02-14 01:40")
    w.account("eve", "2026-02-14 01:00", dob=1995)
    w.device("d_eve", "2026-02-14 01:00", ua="Chrome")
    w.link_device("eve", "d_eve", "2026-02-14 01:00")
    w.address("a_eve", "2026-02-14 01:00", city="Miami", region="FL")
    w.link_address("eve", "a_eve", "2026-02-14 01:00")
    for n, minute in ((1, 5), (2, 15), (3, 30)):
        w.card(f"c_eve{n}", "eve", f"2026-02-14 01:{minute:02d}", bin_country="GB")
    for n, minute in ((1, 10), (2, 20)):
        w.order(f"eve{n}", "eve", "m_tech", f"2026-02-14 01:{minute:02d}", 64999,
                card=f"c_eve{n}", ip="91.200.3.7", ip_country="RO", avs="N", cvv="N",
                declined=True)
    w.order("eve3", "eve", "m_tech", "2026-02-14 01:40", 65000, card="c_eve3",
            ip="91.200.3.7", ip_country="RO", avs="N", ship_hours=8)
    w.miss("eve3")
    w.dispute("eve3", "unauthorized", "2026-03-01 09:00", "2026-03-03 09:00",
              ("2026-04-08 09:00", "2026-04-10 09:00"), "lost")
    w.write_off("eve3")
    w.truth("eve", "fraudster", "e_stolen")
    for n in (1, 2, 3):
        w.order_truth(f"eve{n}", "P-STOLEN", "e_stolen", "fraud")

    # fay: new customer who never pays after checkout (never-pay determination).
    w.episode("e_neverpay", "P-NEVERPAY", "2026-02-01 12:00", "2026-03-15 12:30")
    w.customer("fay", "2026-02-01 12:00", dob=2001)
    w.order("fay1", "fay", "m_tech", "2026-02-01 12:30", 52000, ship_hours=18)
    w.miss("fay1")
    w.write_off("fay1")
    w.truth("fay", "fraudster", "e_neverpay")
    w.order_truth("fay1", "P-NEVERPAY", "e_neverpay", "fraud")

    # gus: returning customer in hardship; a zero-effort default after a repaid plan
    # is a credit loss, not never-pay.
    w.customer("gus", "2024-11-11 11:00", dob=1980)
    w.account_event("gus", "phone_change", "2026-01-10 09:00")
    w.order("gus1", "gus", "m_home", "2025-12-05 12:00", 12000)
    w.repay("gus1")
    w.order("gus2", "gus", "m_home", "2026-01-25 21:00", 24000)
    w.miss("gus2")
    w.write_off("gus2")
    w.truth("gus", profile="hardship")
    w.order_truth("gus1")
    w.order_truth("gus2", mimic="hardship_default")

    # hal: repays, then claims two delivered orders never arrived; both claims rejected.
    w.episode("e_inr", "P-INR-ABUSE", "2025-12-10 18:00", "2026-03-10 10:00")
    w.customer("hal", "2025-06-01 09:00", dob=1990)
    w.order("hal1", "hal", "m_home", "2025-12-10 18:00", 7500)
    w.repay("hal1")
    w.dispute("hal1", "item_not_received", "2026-01-02 10:00", "2026-01-04 10:00",
              ("2026-02-01 10:00", "2026-02-02 10:00"), "won")
    w.order("hal2", "hal", "m_home", "2026-01-15 19:00", 9500)
    w.repay("hal2")
    w.dispute("hal2", "item_not_received", "2026-02-10 10:00", "2026-02-12 10:00",
              ("2026-03-09 10:00", "2026-03-10 10:00"), "won")
    w.truth("hal", "fraudster", "e_inr")
    w.order_truth("hal1", "P-INR-ABUSE", "e_inr", "abuse")
    w.order_truth("hal2", "P-INR-ABUSE", "e_inr", "abuse")

    # ivy: parcel lost by the carrier; her claim wins and the merchant reimburses.
    w.customer("ivy", "2025-03-03 16:00", dob=1984)
    w.order("ivy1", "ivy", "m_tech", "2026-01-08 15:00", 21000, ship_hours=20,
            deliver_days=None)
    w.repay("ivy1")
    w.dispute("ivy1", "item_not_received", "2026-02-04 09:00", "2026-02-05 09:00",
              ("2026-03-18 09:00", "2026-03-20 09:00"), "lost")
    w.truth("ivy")
    w.order_truth("ivy1", mimic="genuine_non_delivery")

    # jay and kim: customers of a merchant that stops shipping and disappears.
    w.episode("e_merch", "P-MERCH", "2025-12-15 09:00", "2026-02-20 12:00")
    w.customer("kim", "2025-07-07 07:00", dob=1993)
    w.order("kim1", "kim", "m_flash", "2026-01-10 11:00", 41000, ship_hours=9)
    w.repay("kim1")
    w.customer("jay", "2025-10-10 10:00", dob=1986)
    w.order("jay1", "jay", "m_flash", "2026-02-12 16:00", 78000, ship_hours=10,
            deliver_days=None)
    w.repay("jay1")
    w.dispute("jay1", "item_not_received", "2026-03-05 10:00", "2026-03-06 10:00",
              ("2026-04-03 10:00", "2026-04-05 10:00"), "lost")
    w.truth("kim")
    w.truth("jay")
    w.order_truth("kim1")
    w.order_truth("jay1", "P-MERCH", "e_merch", "fraud")

    # lee: travelling; orders from a French IP to the home address.
    w.customer("lee", "2024-02-02 12:00", dob=1977)
    w.account_event("lee", "login", "2026-03-01 08:00", ip="81.250.1.4", ip_country="FR")
    w.account_event("lee", "password_change", "2026-03-01 08:05", ip="81.250.1.4",
                    ip_country="FR")
    w.order("lee1", "lee", "m_home", "2026-03-02 13:00", 15500, ip="81.250.1.4",
            ip_country="FR")
    w.repay("lee1")
    w.truth("lee", profile="traveller")
    w.order_truth("lee1", mimic="travel")

    # pf1-pf3: three new accounts on one device each take the first-purchase promotion.
    w.episode("e_promo", "P-PROMO", "2026-03-10 20:00", "2026-03-12 21:20")
    w.device("d_farm", "2026-03-10 20:00", ua="Android")
    for n, (signup, ordered, amount) in enumerate(
        (("2026-03-10 20:00", "2026-03-10 20:20", 5500),
         ("2026-03-11 20:30", "2026-03-11 20:50", 5800),
         ("2026-03-12 21:00", "2026-03-12 21:20", 6000)), start=1):
        user = f"pf{n}"
        w.account(user, signup, dob=2000)
        w.link_device(user, "d_farm", signup)
        w.address(f"a_{user}", signup, city="Phoenix", region="AZ")
        w.link_address(user, f"a_{user}", signup)
        w.card(f"c_{user}", user, signup)
        w.order(f"{user}1", user, "m_home", ordered, amount, device="d_farm", promo="first10")
        w.repay(f"{user}1")
        w.truth(user, "fraudster", "e_promo")
        w.order_truth(f"{user}1", "P-PROMO", "e_promo", "abuse")

    # mo: an order too recent for any label by the end of observation.
    w.customer("mo", "2026-05-20 15:00", dob=1997)
    w.order("mo1", "mo", "m_home", "2026-06-10 10:00", 3000)
    w.repay("mo1", [1])
    w.truth("mo")
    w.order_truth("mo1")

    return w.tables()


def manifest(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return world.build_manifest(
        tables,
        generator_version="mini-world-1",
        config={"product": TERMS.__dict__ | {"liability": dict(TERMS.liability)},
                "horizon_days": HORIZON_DAYS},
        seed=SEED,
        family=FAMILY,
        order_start=ORDER_START,
        order_end=ORDER_END,
        observed_until=str(OBSERVED_UNTIL),
        bustout_merchant_ids=[3],
    )


def main() -> None:
    tables = build()
    world.validate_world(tables)
    world.write_world(tables, HERE)
    world.write_manifest(manifest(tables), HERE / "manifest.json")
    for name, frame in tables.items():
        print(f"{name:22s} {len(frame):4d}")


if __name__ == "__main__":
    main()
