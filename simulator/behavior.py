"""Benign marketplace behavior: orders, pay-in-4 repayment, account events,
promos, and the benign edge cases (travelers, movers, gift buyers, hardship
defaulters) that make fraud detection non-trivial.

Vectorized with numpy/pandas; the only Python-level loops left are small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from simulator.population import CATEGORIES, COUNTRIES_FOREIGN, Population

CAT_MEDIAN = {c[0]: c[1] for c in CATEGORIES}


@dataclass
class World:
    """Mutable accumulator shared by behavior + pattern injectors."""

    pop: Population
    start: datetime
    end: datetime
    orders: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    installments: list[dict] = field(default_factory=list)
    payments: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    promos: list[dict] = field(default_factory=list)
    promo_redemptions: list[dict] = field(default_factory=list)
    chargebacks: list[dict] = field(default_factory=list)
    labels: list[dict] = field(default_factory=list)
    stories: list[dict] = field(default_factory=list)
    _ids: dict[str, int] = field(default_factory=dict)

    def next_id(self, kind: str) -> int:
        self._ids[kind] = self._ids.get(kind, 0) + 1
        return self._ids[kind]

    def user_ip(
        self, user_id: int, rng: np.random.Generator, foreign: bool = False
    ) -> tuple[str, str]:
        """Stable home IP per user (hash-derived), optionally a foreign one."""
        h = (user_id * 2654435761) % (2**32)
        if foreign:
            cc = COUNTRIES_FOREIGN[h % len(COUNTRIES_FOREIGN)]
            return f"77.{(h >> 8) % 256}.{(h >> 16) % 256}.{rng.integers(1, 255)}", cc
        kyc = "US" if user_id % 100 >= 15 else "CA"
        return f"{24 + h % 60}.{(h >> 8) % 256}.{(h >> 16) % 256}.{rng.integers(1, 255)}", kyc

    def add_order(self, **kw) -> int:
        oid = self.next_id("order")
        self.orders.append({"order_id": oid, **kw})
        return oid

    def add_plan_and_schedule(self, order_id: int, order_ts: datetime, amount: float) -> int:
        pid = self.next_id("plan")
        down = round(amount * 0.25, 2)
        inst = round((amount - down) / 3, 2)
        self.plans.append(
            {
                "plan_id": pid,
                "order_id": order_id,
                "principal": round(amount, 2),
                "down_amount": down,
                "n_installments": 3,
            }
        )
        for seq in range(1, 4):
            self.installments.append(
                {
                    "installment_id": self.next_id("installment"),
                    "plan_id": pid,
                    "seq": seq,
                    "due_ts": order_ts + timedelta(days=14 * seq),
                    "paid_ts": None,
                    "amount": inst,
                    "outcome": "pending",
                }
            )
        return pid

    def pay(self, plan_id: int, installment_id: int | None, ts: datetime, amount: float,
            result: str) -> None:
        self.payments.append(
            {
                "payment_id": self.next_id("payment"),
                "plan_id": plan_id,
                "installment_id": installment_id,
                "ts": ts,
                "amount": round(amount, 2),
                "method": "card",
                "result": result,
            }
        )


# ---------------------------------------------------------------------------


def _seasonal_ts(rng: np.random.Generator, n: int, lo: np.ndarray, hi: np.ndarray,
                 start: datetime) -> np.ndarray:
    """Sample n timestamps (as seconds offset from `start`) uniform in [lo,hi)
    per row, then reshape hour-of-day to an evening-peaked curve and bump
    Nov/Dec days by ~20%."""
    span = np.maximum(hi - lo, 1.0)
    t = lo + rng.random(n) * span
    # replace hour with diurnal draw
    w = np.array(
        [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 8, 9, 10, 12, 12, 10, 8, 5, 3], float
    )
    hours = rng.choice(24, n, p=w / w.sum())
    day_sec = 86400
    t = (t // day_sec) * day_sec + hours * 3600 + rng.integers(0, 3600, n)
    # holiday bump: with p=0.06 resample the date into Nov 15 – Dec 23
    nov15 = (datetime(start.year, 11, 15) - start).total_seconds()
    if nov15 < 0:
        nov15 += 365 * day_sec
    bump = rng.random(n) < 0.06
    t[bump] = nov15 + rng.random(bump.sum()) * 38 * day_sec + hours[bump] * 3600
    return np.clip(t, lo, hi - 1)


def run_benign(cfg: dict, rng: np.random.Generator, world: World) -> None:
    pop = world.pop
    start, end = world.start, world.end
    n_users = len(pop.users)
    target = cfg["simulator"]["target_orders"]

    # -- per-user traits ----------------------------------------------------
    weight = rng.gamma(0.85, 1.0, n_users)
    signup = pop.users.signup_ts.values.astype("datetime64[s]")
    start64 = np.datetime64(start, "s")
    end64 = np.datetime64(end, "s")
    active_from = np.maximum(signup, start64)
    active_frac = (end64 - active_from).astype(float) / float((end64 - start64).astype(float))
    active_frac = np.clip(active_frac, 0.02, 1.0)
    exp_orders = target * (weight * active_frac) / (weight * active_frac).sum()
    counts = rng.poisson(exp_orders)

    payer_class = rng.choice(
        ["on_time", "late_recover", "benign_default"], n_users, p=[0.885, 0.09, 0.025]
    )
    is_traveler = rng.random(n_users) < 0.04
    travel_start = start64 + (rng.random(n_users) * 330 * 86400).astype("timedelta64[s]")
    travel_len = rng.integers(7, 15, n_users).astype("timedelta64[D]")
    is_mover = rng.random(n_users) < 0.06
    move_at = start64 + (rng.random(n_users) * 300 * 86400).astype("timedelta64[s]")
    churn_dev = rng.random(n_users) < 0.08

    # favorite merchants per user (affinity)
    n_merch = len(pop.merchants)
    fav = rng.integers(1, n_merch + 1, (n_users, 3))
    merch_pop = rng.dirichlet(np.full(n_merch, 0.6))  # popularity skew

    # -- movers: new address + event; churners: new device + event ----------
    mover_addr: dict[int, int] = {}
    for uid in pop.users.user_id.values[is_mover]:
        aid = int(pop.addresses.address_id.max()) + len(mover_addr) + 1
        ts = pd.Timestamp(move_at[uid - 1]).to_pydatetime()
        mover_addr[int(uid)] = aid
        pop.addresses.loc[len(pop.addresses)] = {
            "address_id": aid, "user_id": int(uid), "line_hash": f"m{uid:012x}",
            "city": "Nashville", "region": "TN", "country": "US", "added_ts": ts,
        }
        ip, cc = world.user_ip(int(uid), rng)
        world.events.append(
            {"event_id": world.next_id("event"), "user_id": int(uid), "ts": ts,
             "kind": "address_add", "ip": ip, "device_id": pop.user_primary_device[int(uid)]}
        )
    churn_new_dev: dict[int, int] = {}
    for uid in pop.users.user_id.values[churn_dev]:
        did = int(pop.devices.device_id.max()) + len(churn_new_dev) + 1
        churn_new_dev[int(uid)] = did
        ts = pd.Timestamp(
            start64 + np.timedelta64(int(rng.integers(0, 330)) * 86400, "s")
        ).to_pydatetime()
        pop.devices.loc[len(pop.devices)] = {
            "device_id": did, "fingerprint": f"{rng.integers(0, 2**63):016x}",
            "ua_family": "iOS",
        }
        pop.user_devices.loc[len(pop.user_devices)] = {
            "user_id": int(uid), "device_id": did, "first_seen": ts, "last_seen": world.end,
        }
        ip, cc = world.user_ip(int(uid), rng)
        world.events.append(
            {"event_id": world.next_id("event"), "user_id": int(uid), "ts": ts,
             "kind": "device_add", "ip": ip, "device_id": did}
        )

    # -- promos -------------------------------------------------------------
    world.promos = [
        {"promo_id": 1, "code": "FIRST10", "discount_pct": 10,
         "valid_from": start, "valid_to": end},
        {"promo_id": 2, "code": "FALL15", "discount_pct": 15,
         "valid_from": datetime(start.year, 10, 1), "valid_to": datetime(start.year, 11, 1)},
        {"promo_id": 3, "code": "SPRING5", "discount_pct": 5,
         "valid_from": datetime(start.year + 1, 3, 1), "valid_to": datetime(start.year + 1, 4, 1)},
    ]

    # -- orders, vectorized draw then row loop ------------------------------
    uid_rep = np.repeat(pop.users.user_id.values, counts)
    n_orders = len(uid_rep)
    lo = (np.repeat(active_from, counts) - start64).astype(float)
    hi = float((end64 - start64).astype(float))
    t_off = _seasonal_ts(rng, n_orders, lo, np.full(n_orders, hi), start)
    order_ts = start64 + t_off.astype("timedelta64[s]")

    use_fav = rng.random(n_orders) < 0.7
    fav_pick = fav[uid_rep - 1, rng.integers(0, 3, n_orders)]
    pop_pick = rng.choice(np.arange(1, n_merch + 1), n_orders, p=merch_pop)
    merch_id = np.where(use_fav, fav_pick, pop_pick)
    m_cat = pop.merchants.set_index("merchant_id").category
    cats = m_cat.loc[merch_id].values
    med = np.array([CAT_MEDIAN[c] for c in cats])
    amounts = np.round(np.exp(rng.normal(np.log(med), 0.65)), 2)
    amounts = np.clip(amounts, 15, 3000)

    gift = rng.random(n_orders) < 0.07
    second_dev = rng.random(n_orders) < 0.2
    avs_bad = rng.random(n_orders) < 0.04
    cvv_bad = rng.random(n_orders) < 0.02
    declined = rng.random(n_orders) < 0.015

    order_sort = np.argsort(order_ts, kind="stable")

    addr_ids = pop.addresses.address_id.values
    addr_users = pop.addresses.user_id.values
    devs_by_user = pop.user_devices.groupby("user_id").device_id.agg(list)

    for j in order_sort:
        uid = int(uid_rep[j])
        ts = pd.Timestamp(order_ts[j]).to_pydatetime()
        traveling = bool(is_traveler[uid - 1]) and (
            travel_start[uid - 1] <= order_ts[j] <= travel_start[uid - 1] + travel_len[uid - 1]
        )
        ip, ipc = world.user_ip(uid, rng, foreign=traveling)
        devs = devs_by_user.get(uid, [pop.user_primary_device[uid]])
        dev = int(devs[-1]) if (second_dev[j] and len(devs) > 1) else int(devs[0])
        if uid in churn_new_dev and rng.random() < 0.5:
            dev = churn_new_dev[uid]
        if gift[j]:
            k = int(rng.integers(0, len(addr_ids)))
            ship, ship_owner = int(addr_ids[k]), int(addr_users[k])
        else:
            ship = mover_addr.get(uid) if (
                uid in mover_addr and order_ts[j] >= move_at[uid - 1]
            ) else pop.user_primary_address[uid]
            ship_owner = uid
        status = "declined" if declined[j] else "approved"
        oid = world.add_order(
            user_id=uid, merchant_id=int(merch_id[j]), ts=ts, amount=float(amounts[j]),
            ip=ip, ip_country=ipc, device_id=dev, card_id=pop.user_primary_card[uid],
            ship_address_id=int(ship), avs_result="N" if avs_bad[j] else "Y",
            cvv_result="N" if cvv_bad[j] else "M", status=status,
        )
        _ = ship_owner
        if status != "approved":
            continue
        pid = world.add_plan_and_schedule(oid, ts, float(amounts[j]))
        world.pay(pid, None, ts, amounts[j] * 0.25, "success")
        # first-purchase promo on ~25% of first orders (cheap approximation:
        # low order ids per user), seasonal promos randomly
        if rng.random() < 0.03:
            promo = world.promos[int(rng.integers(1, 3))]
            if promo["valid_from"] <= ts <= promo["valid_to"]:
                world.promo_redemptions.append(
                    {"redemption_id": world.next_id("redemption"), "promo_id": promo["promo_id"],
                     "user_id": uid, "order_id": oid, "ts": ts}
                )

    # -- repayment on benign plans (vectorized outcomes) --------------------
    inst = pd.DataFrame(world.installments)
    plans = pd.DataFrame(world.plans)
    orders_df = pd.DataFrame(world.orders)
    plan_user = plans.merge(orders_df[["order_id", "user_id"]], on="order_id")
    pu = plan_user.set_index("plan_id").user_id
    cls = pd.Series(payer_class, index=pop.users.user_id.values)
    inst_cls = cls.loc[pu.loc[inst.plan_id].values].values

    r = rng.random(len(inst))
    late_days = rng.integers(2, 21, len(inst))
    outcome = np.where(r < 0.985, "paid", "late").astype(object)
    # late_recover users: more lateness, rare fail-then-recover
    lr = inst_cls == "late_recover"
    outcome[lr] = np.where(r[lr] < 0.60, "paid", np.where(r[lr] < 0.95, "late", "late"))
    # benign defaulters: installment 1 sometimes paid, rest fail
    bd = inst_cls == "benign_default"
    seqs = inst.seq.values
    bd_paid_first = bd & (seqs == 1) & (r < 0.6)
    bd_fail = bd & ~bd_paid_first
    outcome[bd_paid_first] = "paid"
    outcome[bd_fail] = "written_off"

    due = pd.to_datetime(inst.due_ts)
    paid_ts = np.where(
        outcome == "paid",
        due,
        np.where(outcome == "late", due + pd.to_timedelta(late_days, "D"), pd.NaT),
    )
    inst["outcome"] = outcome
    inst["paid_ts"] = paid_ts
    world.installments = inst.to_dict("records")

    # payment rows for paid/late installments; failed attempts for written_off
    for row in world.installments:
        if row["outcome"] in ("paid", "late"):
            world.pay(row["plan_id"], row["installment_id"],
                      pd.Timestamp(row["paid_ts"]).to_pydatetime(), row["amount"], "success")
        elif row["outcome"] == "written_off":
            world.pay(row["plan_id"], row["installment_id"],
                      pd.Timestamp(row["due_ts"]).to_pydatetime(), row["amount"], "fail")

    # -- benign account events: logins + occasional credential changes ------
    n_logins = rng.poisson(6 * active_frac)
    uid_l = np.repeat(pop.users.user_id.values, n_logins)
    t_l = start64 + (rng.random(len(uid_l)) * (hi - 1)).astype("timedelta64[s]")
    for uid, ts in zip(uid_l, t_l, strict=True):
        ip, _ = world.user_ip(int(uid), rng)
        world.events.append(
            {"event_id": world.next_id("event"), "user_id": int(uid),
             "ts": pd.Timestamp(ts).to_pydatetime(), "kind": "login", "ip": ip,
             "device_id": pop.user_primary_device[int(uid)]}
        )
    for kind, p in (("password_change", 0.02), ("email_change", 0.01)):
        pick = pop.users.user_id.values[rng.random(n_users) < p]
        for uid in pick:
            ip, _ = world.user_ip(int(uid), rng)
            ts = pd.Timestamp(
                start64 + np.timedelta64(int(rng.integers(0, int(hi))), "s")
            ).to_pydatetime()
            world.events.append(
                {"event_id": world.next_id("event"), "user_id": int(uid), "ts": ts,
                 "kind": kind, "ip": ip, "device_id": pop.user_primary_device[int(uid)]}
            )

    # -- benign chargebacks (isolated INR / not-as-described) ---------------
    approved = orders_df[orders_df.status == "approved"]
    cb_pick = approved.sample(frac=0.003, random_state=int(rng.integers(0, 2**31)))
    for row in cb_pick.itertuples():
        world.chargebacks.append(
            {"chargeback_id": world.next_id("chargeback"), "order_id": int(row.order_id),
             "reason": "inr" if rng.random() < 0.6 else "not_as_described",
             "opened_ts": row.ts + timedelta(days=int(rng.integers(10, 40))),
             "outcome": rng.choice(["lost", "won", "pending"], p=[0.45, 0.4, 0.15])}
        )
