"""Fraud pattern injectors. Each emits labeled rows into the World plus a
machine-readable story (data/stories.jsonl) that case files are built from.

Injectors deliberately overlap benign edge cases (travelers/movers/defaulters)
— see the brief: separability would make every downstream metric a lie.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from simulator.behavior import CAT_MEDIAN, World
from simulator.population import DISPOSABLE_DOMAINS, FIRST, LAST

HIGH_VALUE_CATS = ("electronics", "jewelry", "gaming")


# --------------------------------------------------------------------- helpers
def _new_user(world: World, rng: np.random.Generator, signup: datetime, email: str,
              domain: str, country: str = "US") -> int:
    pop = world.pop
    uid = int(pop.users.user_id.max()) + 1
    pop.users.loc[len(pop.users)] = {
        "user_id": uid, "signup_ts": signup, "email": email, "email_domain": domain,
        "kyc_country": country, "dob_year": int(rng.integers(1960, 2005)),
    }
    return uid


def _new_device(world: World, rng: np.random.Generator, ua: str = "Android") -> int:
    pop = world.pop
    did = int(pop.devices.device_id.max()) + 1
    pop.devices.loc[len(pop.devices)] = {
        "device_id": did, "fingerprint": f"{rng.integers(0, 2**63):016x}", "ua_family": ua,
    }
    return did


def _attach_device(world: World, uid: int, did: int, ts: datetime) -> None:
    world.pop.user_devices.loc[len(world.pop.user_devices)] = {
        "user_id": uid, "device_id": did, "first_seen": ts, "last_seen": world.end,
    }


def _new_address(world: World, rng: np.random.Generator, uid: int, ts: datetime,
                 country: str = "US") -> int:
    pop = world.pop
    aid = int(pop.addresses.address_id.max()) + 1
    pop.addresses.loc[len(pop.addresses)] = {
        "address_id": aid, "user_id": uid, "line_hash": f"f{rng.integers(0, 2**48):012x}",
        "city": "Houston", "region": "TX", "country": country, "added_ts": ts,
    }
    return aid


def _new_card(world: World, rng: np.random.Generator, uid: int, bin_country: str) -> int:
    pop = world.pop
    cid = int(pop.cards.card_id.max()) + 1
    pop.cards.loc[len(pop.cards)] = {
        "card_id": cid, "user_id": uid, "bin_country": bin_country,
        "network": str(rng.choice(["visa", "mc"])), "last4": f"{rng.integers(0, 10000):04d}",
    }
    return cid


def _event(world: World, uid: int, ts: datetime, kind: str, ip: str, device_id: int) -> None:
    world.events.append(
        {"event_id": world.next_id("event"), "user_id": uid, "ts": ts, "kind": kind,
         "ip": ip, "device_id": device_id}
    )


def _fraud_order(world: World, rng: np.random.Generator, *, uid: int, mid: int, ts: datetime,
                 amount: float, ip: str, ipc: str, dev: int, card: int, addr: int,
                 avs_bad_p: float, cvv_bad_p: float, pattern: str,
                 pay_installments: int = 0, status: str = "approved") -> int:
    oid = world.add_order(
        user_id=uid, merchant_id=mid, ts=ts, amount=round(amount, 2), ip=ip, ip_country=ipc,
        device_id=dev, card_id=card, ship_address_id=addr,
        avs_result="N" if rng.random() < avs_bad_p else "Y",
        cvv_result="N" if rng.random() < cvv_bad_p else "M", status=status,
    )
    if status == "approved":
        pid = world.add_plan_and_schedule(oid, ts, amount)
        world.pay(pid, None, ts, amount * 0.25, "success")
        insts = [i for i in world.installments if i["plan_id"] == pid]
        for i, inst in enumerate(insts):
            if i < pay_installments:
                inst["outcome"] = "paid"
                inst["paid_ts"] = inst["due_ts"]
                world.pay(pid, inst["installment_id"], inst["due_ts"], inst["amount"], "success")
            else:
                inst["outcome"] = "written_off"
                world.pay(pid, inst["installment_id"], inst["due_ts"], inst["amount"], "fail")
                world.pay(pid, inst["installment_id"],
                          inst["due_ts"] + timedelta(days=3), inst["amount"], "fail")
        world.labels.append({"order_id": oid, "user_id": uid, "pattern_id": pattern})
    return oid


def _rand_ts(rng: np.random.Generator, lo: datetime, hi: datetime) -> datetime:
    span = max(int((hi - lo).total_seconds()), 1)
    return lo + timedelta(seconds=int(rng.integers(0, span)))


def _cb(world: World, oid: int, ts: datetime, rng: np.random.Generator,
        reason: str = "fraud") -> None:
    world.chargebacks.append(
        {"chargeback_id": world.next_id("chargeback"), "order_id": oid, "reason": reason,
         "opened_ts": ts + timedelta(days=int(rng.integers(14, 42))),
         "outcome": str(rng.choice(["lost", "pending"], p=[0.8, 0.2]))}
    )


# --------------------------------------------------------------------- P-ATO
def inject_ato(cfg: dict, rng: np.random.Generator, world: World) -> None:
    n = cfg["simulator"]["patterns"]["ato_accounts"]
    orders_df = pd.DataFrame(world.orders)
    approved = orders_df[orders_df.status == "approved"]
    counts = approved.groupby("user_id").size()
    cb_users = {
        int(orders_df.loc[orders_df.order_id == c["order_id"], "user_id"].iloc[0])
        for c in world.chargebacks
    }
    pop = world.pop
    tenure_ok = pop.users[pop.users.signup_ts < world.start - timedelta(days=60)]
    candidates = [
        int(u) for u in tenure_ok.user_id
        if counts.get(u, 0) >= 2 and u not in cb_users
    ]
    victims = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    hv_merch = world.pop.merchants[world.pop.merchants.category.isin(HIGH_VALUE_CATS)]
    for uid in victims:
        uid = int(uid)
        t0 = _rand_ts(rng, world.start + timedelta(days=45), world.end - timedelta(days=50))
        dev = _new_device(world, rng, ua=str(rng.choice(["Chrome", "Firefox"])))
        _attach_device(world, uid, dev, t0)
        ip, ipc = world.user_ip(90000 + uid, rng, foreign=bool(rng.random() < 0.7))
        _event(world, uid, t0, "password_change", ip, dev)
        if rng.random() < 0.5:
            _event(world, uid, t0 + timedelta(minutes=int(rng.integers(2, 30))),
                   "email_change", ip, dev)
        addr = _new_address(world, rng, uid, t0 + timedelta(hours=1))
        _event(world, uid, t0 + timedelta(hours=1), "address_add", ip, dev)
        oids = []
        for _k in range(int(rng.integers(1, 5))):
            m = hv_merch.iloc[int(rng.integers(0, len(hv_merch)))]
            amount = float(np.clip(np.exp(rng.normal(np.log(CAT_MEDIAN[m.category] * 1.6), 0.3)),
                                   200, 2500))
            ts = t0 + timedelta(hours=float(rng.uniform(1.5, 20)))
            oid = _fraud_order(
                world, rng, uid=uid, mid=int(m.merchant_id), ts=ts, amount=amount, ip=ip,
                ipc=ipc, dev=dev, card=world.pop.user_primary_card[uid], addr=addr,
                avs_bad_p=0.4, cvv_bad_p=0.2, pattern="P-ATO",
            )
            oids.append(oid)
            if rng.random() < 0.6:
                _cb(world, oid, ts, rng)
        tenure = (t0 - pd.Timestamp(pop.users.loc[pop.users.user_id == uid,
                                                  "signup_ts"].iloc[0]).to_pydatetime()).days
        world.stories.append(
            {"story_id": f"ATO-{uid}", "pattern_id": "P-ATO", "user_ids": [uid],
             "order_ids": oids, "merchant_ids": [], "device_ids": [dev],
             "t_start": str(t0), "t_end": str(t0 + timedelta(hours=20)),
             "facts": {"victim_tenure_days": tenure, "new_device_id": dev,
                       "takeover_ip": ip, "takeover_ip_country": ipc,
                       "n_orders": len(oids), "new_address_id": addr}}
        )


# ------------------------------------------------------------------ P-STOLEN
def inject_stolen(cfg: dict, rng: np.random.Generator, world: World) -> None:
    n = cfg["simulator"]["patterns"]["stolen_accounts"]
    merch = world.pop.merchants
    for _i in range(n):
        signup = _rand_ts(rng, world.start, world.end - timedelta(days=45))
        dom = str(rng.choice(["gmail.com", "outlook.com", "yahoo.com"]))
        email = f"{rng.choice(FIRST)}{rng.integers(100, 99999)}@{dom}"
        uid = _new_user(world, rng, signup, email, email.split("@")[1])
        dev = _new_device(world, rng)
        _attach_device(world, uid, dev, signup)
        addr = _new_address(world, rng, uid, signup)
        world.pop.user_primary_device[uid] = dev
        world.pop.user_primary_address[uid] = addr
        bin_cc = str(rng.choice(["US", "GB", "DE", "FR", "BR"], p=[0.25, 0.2, 0.2, 0.2, 0.15]))
        card = _new_card(world, rng, uid, bin_cc)
        world.pop.user_primary_card[uid] = card
        ip, ipc = world.user_ip(70000 + uid, rng, foreign=bool(rng.random() < 0.65))
        t0 = signup + timedelta(hours=float(rng.uniform(0.2, 48)))
        oids: list[int] = []
        tested = bool(rng.random() < 0.4)
        if tested:
            for k in range(int(rng.integers(3, 9))):
                m = merch.iloc[int(rng.integers(0, len(merch)))]
                _fraud_order(
                    world, rng, uid=uid, mid=int(m.merchant_id),
                    ts=t0 + timedelta(minutes=4 * k), amount=float(rng.uniform(15, 40)),
                    ip=ip, ipc=ipc, dev=dev, card=card, addr=addr,
                    avs_bad_p=0.7, cvv_bad_p=0.6, pattern="P-STOLEN", status="declined",
                )
        hv = merch[merch.category.isin(HIGH_VALUE_CATS)]
        for _k in range(int(rng.integers(1, 4))):
            m = hv.iloc[int(rng.integers(0, len(hv)))]
            ts = t0 + timedelta(hours=float(rng.uniform(0.5, 24)))
            amount = float(np.clip(np.exp(rng.normal(np.log(CAT_MEDIAN[m.category] * 1.4), 0.4)),
                                   150, 2200))
            oid = _fraud_order(
                world, rng, uid=uid, mid=int(m.merchant_id), ts=ts, amount=amount, ip=ip,
                ipc=ipc, dev=dev, card=card, addr=addr, avs_bad_p=0.55, cvv_bad_p=0.45,
                pattern="P-STOLEN",
            )
            oids.append(oid)
            if rng.random() < 0.7:
                _cb(world, oid, ts, rng)
        world.stories.append(
            {"story_id": f"STL-{uid}", "pattern_id": "P-STOLEN", "user_ids": [uid],
             "order_ids": oids, "merchant_ids": [], "device_ids": [dev],
             "t_start": str(signup), "t_end": str(t0 + timedelta(days=1)),
             "facts": {"bin_country": bin_cc, "ip_country": ipc, "card_testing": tested,
                       "n_approved_orders": len(oids)}}
        )


# ------------------------------------------------------------------- P-SYNTH
def inject_synth(cfg: dict, rng: np.random.Generator, world: World) -> None:
    p = cfg["simulator"]["patterns"]
    n_rings, n_total = p["synth_rings"], p["synth_accounts"]
    per_ring = n_total // n_rings
    merch = world.pop.merchants
    for ring_i in range(n_rings):
        ring_id = f"SYN-R{ring_i + 1}"
        t_create0 = _rand_ts(rng, world.start + timedelta(days=20),
                             world.end - timedelta(days=120))
        disposable = bool(rng.random() < 0.5)
        root = f"{rng.choice(FIRST)}{rng.choice(LAST)}"
        domain = str(rng.choice(DISPOSABLE_DOMAINS)) if disposable else "gmail.com"
        n_devices = max(2, per_ring // int(rng.integers(4, 8)))
        devices = [_new_device(world, rng) for _ in range(n_devices)]
        subnet = f"91.{rng.integers(10, 250)}.{rng.integers(10, 250)}"
        share_addr_uids: list[int] = []
        uids, all_oids = [], []
        addr_shared: list[int] = []
        for k in range(per_ring):
            signup = t_create0 + timedelta(days=float(rng.uniform(0, 35)))
            email = (f"{root}+{k}@{domain}" if not disposable
                     else f"{root}{k}{rng.integers(10, 99)}@{domain}")
            uid = _new_user(world, rng, signup, email, domain)
            uids.append(uid)
            dev = devices[k % n_devices]
            _attach_device(world, uid, dev, signup)
            world.pop.user_primary_device[uid] = dev
            card = _new_card(world, rng, uid, "US")
            world.pop.user_primary_card[uid] = card
            if k < 2:
                aid = _new_address(world, rng, uid, signup)
                addr_shared.append(aid)
                share_addr_uids.append(uid)
            world.pop.user_primary_address[uid] = addr_shared[k % len(addr_shared)]
        # warm-up phase: 1-3 small repaid orders each
        for uid in uids:
            for _w in range(int(rng.integers(1, 4))):
                m = merch.iloc[int(rng.integers(0, len(merch)))]
                ts = _rand_ts(rng, t_create0 + timedelta(days=3), t_create0 + timedelta(days=50))
                ip = f"{subnet}.{rng.integers(2, 254)}"
                oid = _fraud_order(
                    world, rng, uid=uid, mid=int(m.merchant_id), ts=ts,
                    amount=float(rng.uniform(25, 90)), ip=ip, ipc="US",
                    dev=world.pop.user_primary_device[uid],
                    card=world.pop.user_primary_card[uid],
                    addr=world.pop.user_primary_address[uid],
                    avs_bad_p=0.05, cvv_bad_p=0.03, pattern="P-SYNTH", pay_installments=3,
                )
                all_oids.append(oid)
        # bust-out burst: 48-72h, max-value orders, nothing repaid
        t_burst = t_create0 + timedelta(days=float(rng.uniform(55, 75)))
        hv = merch[merch.category.isin(HIGH_VALUE_CATS)]
        for uid in uids:
            for _b in range(int(rng.integers(1, 3))):
                m = hv.iloc[int(rng.integers(0, len(hv)))]
                ts = t_burst + timedelta(hours=float(rng.uniform(0, 60)))
                ip = f"{subnet}.{rng.integers(2, 254)}"
                oid = _fraud_order(
                    world, rng, uid=uid, mid=int(m.merchant_id), ts=ts,
                    amount=float(rng.uniform(400, 1400)), ip=ip, ipc="US",
                    dev=world.pop.user_primary_device[uid],
                    card=world.pop.user_primary_card[uid],
                    addr=world.pop.user_primary_address[uid],
                    avs_bad_p=0.15, cvv_bad_p=0.1, pattern="P-SYNTH",
                )
                all_oids.append(oid)
        world.stories.append(
            {"story_id": ring_id, "pattern_id": "P-SYNTH", "user_ids": uids,
             "order_ids": all_oids, "merchant_ids": [], "device_ids": devices,
             "t_start": str(t_create0), "t_end": str(t_burst + timedelta(hours=72)),
             "facts": {"n_accounts": len(uids), "n_shared_devices": len(devices),
                       "shared_subnet": subnet, "email_root": root, "email_domain": domain,
                       "disposable_email": disposable,
                       "shared_addresses": addr_shared, "burst_start": str(t_burst)}}
        )


# ---------------------------------------------------------------- P-NEVERPAY
def inject_neverpay(cfg: dict, rng: np.random.Generator, world: World) -> None:
    n = cfg["simulator"]["patterns"]["neverpay_accounts"]
    merch = world.pop.merchants
    uids = []
    for _i in range(n):
        signup = _rand_ts(rng, world.start, world.end - timedelta(days=50))
        dom = str(rng.choice(["gmail.com", "hotmail.com", "yahoo.com", "outlook.com"]))
        email = f"{rng.choice(FIRST)}.{rng.choice(LAST)}{rng.integers(10, 999)}@{dom}"
        uid = _new_user(world, rng, signup, email, dom)
        dev = _new_device(world, rng, ua=str(rng.choice(["iOS", "Android"])))
        _attach_device(world, uid, dev, signup)
        world.pop.user_primary_device[uid] = dev
        card = _new_card(world, rng, uid, "US")
        world.pop.user_primary_card[uid] = card
        addr = _new_address(world, rng, uid, signup)
        world.pop.user_primary_address[uid] = addr
        m = merch.iloc[int(rng.integers(0, len(merch)))]
        amount = CAT_MEDIAN[m.category] * float(rng.uniform(1.3, 1.8))
        ts = signup + timedelta(days=float(rng.uniform(0.1, 10)))
        ip, ipc = world.user_ip(uid, rng)
        looks_benign = bool(rng.random() < 0.2)
        oid = _fraud_order(
            world, rng, uid=uid, mid=int(m.merchant_id), ts=ts, amount=amount, ip=ip, ipc=ipc,
            dev=dev, card=card, addr=addr, avs_bad_p=0.06, cvv_bad_p=0.04,
            pattern="P-NEVERPAY", pay_installments=1 if looks_benign else 0,
        )
        uids.append(uid)
        world.stories.append(
            {"story_id": f"NP-{uid}", "pattern_id": "P-NEVERPAY", "user_ids": [uid],
             "order_ids": [oid], "merchant_ids": [int(m.merchant_id)], "device_ids": [dev],
             "t_start": str(signup), "t_end": str(ts),
             "facts": {"first_order_amount": round(amount, 2),
                       "category_median": CAT_MEDIAN[m.category],
                       "paid_first_installment": looks_benign}}
        )


# --------------------------------------------------------------- P-INR-ABUSE
def inject_inr_abuse(cfg: dict, rng: np.random.Generator, world: World) -> None:
    n = cfg["simulator"]["patterns"]["inr_abuse_accounts"]
    orders_df = pd.DataFrame(world.orders)
    approved = orders_df[orders_df.status == "approved"]
    counts = approved.groupby("user_id").size()
    labeled_users = {label["user_id"] for label in world.labels}
    candidates = [int(u) for u, c in counts.items() if c >= 4 and u not in labeled_users]
    pick = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    for uid in pick:
        uid = int(uid)
        mine = approved[approved.user_id == uid].sort_values("ts")
        k = int(rng.integers(2, min(6, len(mine) + 1)))
        disputed = mine.iloc[:k]
        oids = []
        for row in disputed.itertuples():
            world.chargebacks.append(
                {"chargeback_id": world.next_id("chargeback"), "order_id": int(row.order_id),
                 "reason": "inr",
                 "opened_ts": row.ts + timedelta(days=int(rng.integers(12, 30))),
                 "outcome": str(rng.choice(["lost", "won", "pending"], p=[0.6, 0.25, 0.15]))}
            )
            world.labels.append(
                {"order_id": int(row.order_id), "user_id": uid, "pattern_id": "P-INR-ABUSE"}
            )
            oids.append(int(row.order_id))
        world.stories.append(
            {"story_id": f"INR-{uid}", "pattern_id": "P-INR-ABUSE", "user_ids": [uid],
             "order_ids": oids, "merchant_ids": [], "device_ids": [],
             "t_start": str(disputed.ts.min()), "t_end": str(disputed.ts.max()),
             "facts": {"n_inr_disputes": len(oids),
                       "total_disputed_value": round(float(disputed.amount.sum()), 2)}}
        )


# ------------------------------------------------------------------- P-PROMO
def inject_promo(cfg: dict, rng: np.random.Generator, world: World) -> None:
    p = cfg["simulator"]["patterns"]
    n_clusters, n_total = p["promo_clusters"], p["promo_accounts"]
    per = n_total // n_clusters
    merch = world.pop.merchants
    for ci in range(n_clusters):
        t0 = _rand_ts(rng, world.start + timedelta(days=10), world.end - timedelta(days=40))
        devices = [_new_device(world, rng) for _ in range(max(2, per // 6))]
        root = f"{rng.choice(FIRST)}{rng.integers(1, 99)}"
        shared_addr: list[int] = []
        uids, oids = [], []
        for k in range(per):
            signup = t0 + timedelta(days=float(rng.uniform(0, 20)))
            email = f"{root}+{k}@gmail.com"
            uid = _new_user(world, rng, signup, email, "gmail.com")
            uids.append(uid)
            dev = devices[k % len(devices)]
            _attach_device(world, uid, dev, signup)
            world.pop.user_primary_device[uid] = dev
            card = _new_card(world, rng, uid, "US")
            world.pop.user_primary_card[uid] = card
            if len(shared_addr) < 2:
                shared_addr.append(_new_address(world, rng, uid, signup))
            addr = shared_addr[k % len(shared_addr)]
            world.pop.user_primary_address[uid] = addr
            m = merch.iloc[int(rng.integers(0, len(merch)))]
            ts = signup + timedelta(days=float(rng.uniform(0.1, 5)))
            repay = 3 if rng.random() < 0.5 else 0
            oid = _fraud_order(
                world, rng, uid=uid, mid=int(m.merchant_id), ts=ts,
                amount=float(rng.uniform(40, 150)), ip=f"103.44.{7 + ci}.{rng.integers(2, 254)}",
                ipc="US", dev=dev, card=card, addr=addr, avs_bad_p=0.05, cvv_bad_p=0.03,
                pattern="P-PROMO", pay_installments=repay,
            )
            world.promo_redemptions.append(
                {"redemption_id": world.next_id("redemption"), "promo_id": 1, "user_id": uid,
                 "order_id": oid, "ts": ts}
            )
            oids.append(oid)
        world.stories.append(
            {"story_id": f"PRM-C{ci + 1}", "pattern_id": "P-PROMO", "user_ids": uids,
             "order_ids": oids, "merchant_ids": [], "device_ids": devices,
             "t_start": str(t0), "t_end": str(t0 + timedelta(days=25)),
             "facts": {"n_accounts": len(uids), "email_root": root,
                       "promo_code": "FIRST10", "shared_addresses": shared_addr}}
        )


# ------------------------------------------------------------------- P-MERCH
def inject_merchant_bustout(cfg: dict, rng: np.random.Generator, world: World) -> None:
    n = cfg["simulator"]["patterns"]["merchant_bustouts"]
    pop = world.pop
    for bi in range(n):
        mid = int(pop.merchants.merchant_id.max()) + 1
        onboard = world.start + timedelta(days=60 + 40 * bi)
        cat = "electronics" if bi % 2 == 0 else "jewelry"
        pop.merchants.loc[len(pop.merchants)] = {
            "merchant_id": mid, "name": f"flash-{cat[:4]}-{70 + bi}", "category": cat,
            "risk_tier": 3, "onboarded_ts": onboard,
        }
        n_orders = 90
        uids, oids = [], []
        for k in range(n_orders):
            week = k // 12  # volume ramps across ~8 weeks
            ts = onboard + timedelta(days=7 * week + float(rng.uniform(0, 7)))
            if ts >= world.end:
                continue
            # thin, new buyer accounts
            signup = ts - timedelta(days=float(rng.uniform(0.5, 20)))
            dom = str(rng.choice(["gmail.com", "outlook.com"]))
            email = f"{rng.choice(FIRST)}{rng.integers(1000, 9999)}@{dom}"
            uid = _new_user(world, rng, signup, email, dom)
            dev = _new_device(world, rng)
            _attach_device(world, uid, dev, signup)
            pop.user_primary_device[uid] = dev
            card = _new_card(world, rng, uid, "US")
            pop.user_primary_card[uid] = card
            addr = _new_address(world, rng, uid, signup)
            pop.user_primary_address[uid] = addr
            amount = 150 + week * 90 + float(rng.uniform(-40, 60))  # rising ticket
            ip, ipc = world.user_ip(uid, rng)
            oid = _fraud_order(
                world, rng, uid=uid, mid=mid, ts=ts, amount=amount, ip=ip, ipc=ipc, dev=dev,
                card=card, addr=addr, avs_bad_p=0.1, cvv_bad_p=0.06, pattern="P-MERCH",
                pay_installments=1 if rng.random() < 0.3 else 0,
            )
            uids.append(uid)
            oids.append(oid)
            if rng.random() < 0.45:
                world.chargebacks.append(
                    {"chargeback_id": world.next_id("chargeback"), "order_id": oid,
                     "reason": str(rng.choice(["fraud", "not_as_described"], p=[0.6, 0.4])),
                     "opened_ts": ts + timedelta(days=int(rng.integers(60, 95))),
                     "outcome": str(rng.choice(["lost", "pending"], p=[0.7, 0.3]))}
                )
        world.stories.append(
            {"story_id": f"MB-{mid}", "pattern_id": "P-MERCH", "user_ids": uids,
             "order_ids": oids, "merchant_ids": [mid], "device_ids": [],
             "t_start": str(onboard), "t_end": str(onboard + timedelta(days=70)),
             "facts": {"merchant_id": mid, "category": cat, "n_orders": len(oids),
                       "onboarded_ts": str(onboard), "ticket_ramp": "150→~800 over 8 weeks",
                       "chargeback_lag_days": "60–95"}}
        )


def inject_all(cfg: dict, rng: np.random.Generator, world: World) -> None:
    inject_ato(cfg, rng, world)
    inject_stolen(cfg, rng, world)
    inject_synth(cfg, rng, world)
    inject_neverpay(cfg, rng, world)
    inject_inr_abuse(cfg, rng, world)
    inject_promo(cfg, rng, world)
    inject_merchant_bustout(cfg, rng, world)
