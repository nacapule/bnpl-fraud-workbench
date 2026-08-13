"""Rules engine: build the point-in-time enriched order frame, fire R01–R12,
score, band, and write alerts (MySQL table + data/alerts.csv).

Run: python -m rules.engine

Point-in-time discipline applies except for two known final-world enrichments:
category P95 uses all approved orders, and email-root counts use the final users
table. Both are slated for the simulator/rules revision. The ``labels`` table is
never read here (FP-1 §2.4) — only ``rules.tuning`` may read it, offline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rules.definitions import RULES

REPO = Path(__file__).resolve().parent.parent
DISPOSABLE = {"mailinator.com", "tempmailo.com", "guerrillamail.com"}
CENTROIDS = {
    "US": (39.8, -98.6), "CA": (56.1, -106.3), "GB": (54.0, -2.9), "DE": (51.2, 10.4),
    "FR": (46.6, 2.4), "ES": (40.3, -3.7), "BR": (-10.8, -52.9), "IN": (22.9, 79.6),
    "NG": (9.6, 8.1), "RO": (45.9, 24.9), "VN": (16.6, 106.3), "CN": (36.5, 103.8),
    "RU": (61.5, 105.3), "ID": (-2.2, 117.3),
}


def _haversine_km(c1: pd.Series, c2: pd.Series) -> np.ndarray:
    get = np.vectorize(lambda c, i: CENTROIDS.get(c, (np.nan, np.nan))[i])
    lat1, lon1 = np.radians(get(c1, 0)), np.radians(get(c1, 1))
    lat2, lon2 = np.radians(get(c2, 0)), np.radians(get(c2, 1))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        (lon2 - lon1) / 2
    ) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _trailing_count(df: pd.DataFrame, key: str, hours: float) -> np.ndarray:
    """Orders per `key` in the trailing window, inclusive of current row.
    df must be sorted by [key, ts]; returns aligned to df order."""
    out = np.empty(len(df), dtype=np.int32)
    ts = df["ts_epoch"].values
    lo = 0
    keys = df[key].values
    for i in range(len(df)):
        if i > 0 and keys[i] != keys[i - 1]:
            lo = i
        j = lo
        bound = ts[i] - hours * 3600
        while ts[j] < bound:
            j += 1
        lo = j
        out[i] = i - j + 1
    return out


def _windowed_distinct_users(ap: pd.DataFrame, col: str, days: int) -> np.ndarray:
    """Distinct user_ids seen on `col` in the trailing `days` window (inclusive),
    aligned to ap's current row order."""
    order = np.argsort(ap[col].values, kind="stable")
    out = np.zeros(len(ap), dtype=np.int32)
    keys = ap[col].values[order]
    ts = ap["ts_epoch"].values[order]
    users = ap["user_id"].values[order]
    window = days * 86400
    i = 0
    n = len(ap)
    while i < n:
        j = i
        while j < n and keys[j] == keys[i]:
            j += 1
        # rows i..j-1 share the key; they are ts-sorted within (stable argsort of
        # a ts-sorted frame). two-pointer distinct count
        seg_ts, seg_users = ts[i:j], users[i:j]
        seg_order = np.argsort(seg_ts, kind="stable")
        seg_ts, seg_users = seg_ts[seg_order], seg_users[seg_order]
        counts: dict[int, int] = {}
        lo = 0
        seg_out = np.zeros(j - i, dtype=np.int32)
        for k in range(j - i):
            u = seg_users[k]
            counts[u] = counts.get(u, 0) + 1
            while seg_ts[lo] < seg_ts[k] - window:
                ul = seg_users[lo]
                counts[ul] -= 1
                if not counts[ul]:
                    del counts[ul]
                lo += 1
            seg_out[k] = len(counts)
        out[order[i:j][seg_order]] = seg_out
        i = j
    return out


def build_enriched(data_dir: Path | None = None) -> pd.DataFrame:
    """One row per APPROVED order with every rule input, point-in-time safe."""
    d = data_dir or (REPO / "data")
    orders = pd.read_csv(d / "orders.csv", parse_dates=["ts"])
    users = pd.read_csv(d / "users.csv", parse_dates=["signup_ts"])
    cards = pd.read_csv(d / "cards.csv")
    user_devices = pd.read_csv(d / "user_devices.csv", parse_dates=["first_seen"])
    events = pd.read_csv(d / "account_events.csv", parse_dates=["ts"])
    chargebacks = pd.read_csv(d / "chargebacks.csv", parse_dates=["opened_ts"])
    merchants = pd.read_csv(d / "merchants.csv")
    redemptions = pd.read_csv(d / "promo_redemptions.csv", parse_dates=["ts"])

    ap = orders[orders.status == "approved"].copy()
    ap["ts_epoch"] = ap.ts.astype("datetime64[s]").astype("int64")

    # account age
    ap = ap.merge(users[["user_id", "signup_ts", "email", "email_domain"]], on="user_id")
    ap["account_age_days"] = (ap.ts - ap.signup_ts).dt.total_seconds() / 86400

    # credential change recency (password/email changes only)
    cred = events[events.kind.isin(["password_change", "email_change"])][
        ["user_id", "ts"]
    ].sort_values("ts").rename(columns={"ts": "cred_ts"})
    ap = ap.sort_values("ts")
    ap = pd.merge_asof(ap, cred, left_on="ts", right_on="cred_ts", by="user_id")
    ap["cred_change_hours"] = (ap.ts - ap.cred_ts).dt.total_seconds() / 3600
    ap["cred_change_hours"] = ap.cred_change_hours.fillna(np.inf)

    # device novelty: first_seen of (user, device) within 72h before order
    ud = user_devices.groupby(["user_id", "device_id"], as_index=False).first_seen.min()
    ap = ap.merge(ud, on=["user_id", "device_id"], how="left")
    ap["device_new"] = (ap.ts - ap.first_seen).dt.total_seconds() / 3600 <= 72
    ap["device_new"] = ap.device_new.fillna(True)  # unseen pairing is new

    # distinct users per device / ship address in the trailing 30 days
    for col, out in (("device_id", "users_on_device"), ("ship_address_id", "users_on_ship_addr")):
        ap[out] = _windowed_distinct_users(ap, col, days=30)

    # first order + amount vs category P95
    ap = ap.merge(merchants[["merchant_id", "category"]], on="merchant_id")
    ap["order_rank"] = ap.sort_values("ts").groupby("user_id").cumcount() + 1
    ap["is_first_order"] = ap.order_rank == 1
    p95 = ap.groupby("category").amount.quantile(0.95).rename("cat_p95")
    ap = ap.merge(p95, on="category")
    ap["amount_vs_p95"] = ap.amount / ap.cat_p95

    # velocity (trailing 24h, inclusive)
    ap = ap.sort_values(["user_id", "ts_epoch"]).reset_index(drop=True)
    ap["n_user_24h"] = _trailing_count(ap, "user_id", 24)
    ap = ap.sort_values(["device_id", "ts_epoch"]).reset_index(drop=True)
    ap["n_dev_24h"] = _trailing_count(ap, "device_id", 24)

    # verification + geography
    cards_cc = cards.set_index("card_id").bin_country
    ap["bin_country"] = ap.card_id.map(cards_cc)
    ap["bin_ip_mismatch"] = ap.bin_country != ap.ip_country
    ap["avs_bad"] = ap.avs_result != "Y"
    ap["cvv_bad"] = ap.cvv_result != "M"

    # email identity
    root = (
        ap.email.str.split("@").str[0].str.split("+").str[0].str.replace(".", "", regex=False)
        + "@" + ap.email_domain
    )
    ap["email_root"] = root
    root_counts = users.assign(
        root=users.email.str.split("@").str[0].str.split("+").str[0]
        .str.replace(".", "", regex=False) + "@" + users.email_domain
    ).groupby("root").user_id.nunique()
    ap["email_root_accounts"] = ap.email_root.map(root_counts).fillna(1)
    ap["email_root_dup"] = ap.email_root_accounts >= 2
    ap["email_disposable"] = ap.email_domain.isin(DISPOSABLE)

    # card testing: declined attempts on same card OR device in prior 24h
    declined = orders[orders.status == "declined"]
    ap["card_test_declines"] = 0
    for col in ("card_id", "device_id"):
        dec = declined[[col, "ts"]].sort_values("ts")
        dec["epoch"] = dec.ts.astype("datetime64[s]").astype("int64")
        grouped = dec.groupby(col).epoch.apply(np.array).to_dict()
        vals = np.zeros(len(ap), dtype=np.int32)
        keys = ap[col].values
        epochs = ap.ts_epoch.values
        for i in range(len(ap)):
            arr = grouped.get(keys[i])
            if arr is not None:
                vals[i] = np.searchsorted(arr, epochs[i]) - np.searchsorted(
                    arr, epochs[i] - 86400
                )
        ap["card_test_declines"] = np.maximum(ap.card_test_declines.values, vals)

    # prior INR chargebacks per user (opened before this order)
    inr = chargebacks[chargebacks.reason == "inr"].merge(
        orders[["order_id", "user_id"]], on="order_id"
    )[["user_id", "opened_ts"]].sort_values("opened_ts")
    grouped = inr.groupby("user_id").opened_ts.apply(
        lambda s: s.astype("datetime64[s]").astype("int64").values
    ).to_dict()
    vals = np.zeros(len(ap), dtype=np.int32)
    uids = ap.user_id.values
    epochs = ap.ts_epoch.values
    for i in range(len(ap)):
        arr = grouped.get(uids[i])
        if arr is not None:
            vals[i] = np.searchsorted(arr, epochs[i])
    ap["prior_inr_cbs"] = vals

    # promo clustering: this order redeemed a promo AND its device or address has
    # >=3 redemptions up to now
    red = redemptions.merge(
        orders[["order_id", "device_id", "ship_address_id"]], on="order_id"
    ).sort_values("ts")
    redeemed_orders = set(red.order_id)
    ap["promo_cluster_size"] = 0
    if len(red):
        for col in ("device_id", "ship_address_id"):
            cum = red.groupby(col).cumcount() + 1
            m = dict(zip(red.order_id, cum, strict=True))
            sizes = ap.order_id.map(m).fillna(0)
            mask = ap.order_id.isin(redeemed_orders)
            ap.loc[mask, "promo_cluster_size"] = np.maximum(
                ap.loc[mask, "promo_cluster_size"], sizes[mask]
            )

    # geo velocity vs previous order of same user
    ap = ap.sort_values(["user_id", "ts_epoch"]).reset_index(drop=True)
    ap["prev_ip_country"] = ap.groupby("user_id").ip_country.shift()
    ap["prev_epoch"] = ap.groupby("user_id").ts_epoch.shift()
    gap_h = (ap.ts_epoch - ap.prev_epoch) / 3600
    km = _haversine_km(ap.prev_ip_country.fillna("US"), ap.ip_country)
    ap["geo_kmh"] = np.where(
        ap.prev_ip_country.notna() & (ap.prev_ip_country != ap.ip_country) & (gap_h < 12),
        km / np.maximum(gap_h, 0.02),
        0.0,
    )
    ap["prev_ip_country"] = ap.prev_ip_country.fillna("")

    # vendor scores (fixtures; absent file -> R12 never fires)
    ap["vendor_email_score"] = np.nan
    ap["vendor_ip_score"] = np.nan
    scores_path = REPO / "vendor" / "fixtures" / "scores.csv"
    if scores_path.exists():
        sc = pd.read_csv(scores_path)
        em = sc[sc.kind == "email"].set_index("value").fraud_score
        ip = sc[sc.kind == "ip"].set_index("value").fraud_score
        ap["vendor_email_score"] = ap.email.map(em)
        ap["vendor_ip_score"] = ap.ip.map(ip)

    return ap.sort_values("ts").reset_index(drop=True)


def run_rules(ap: pd.DataFrame) -> pd.DataFrame:
    """Apply all rules; return the frame with per-rule fire masks + score."""
    score = np.zeros(len(ap), dtype=int)
    fired_lists: list[list[dict]] = [[] for _ in range(len(ap))]
    for rule in RULES:
        mask = rule.fire(ap).fillna(False).values
        if not mask.any():
            continue
        score[mask] += rule.weight
        expl = rule.explain(ap.loc[mask])
        for pos, (_, text) in zip(np.flatnonzero(mask), expl.items(), strict=True):
            fired_lists[pos].append(
                {"id": rule.id, "name": rule.name, "weight": rule.weight, "rationale": text}
            )
        ap[f"fired_{rule.id}"] = mask
    ap["score"] = score
    ap["fired_rules"] = fired_lists
    return ap


def main() -> None:
    import sqlalchemy as sa
    import yaml

    t0 = time.time()
    cfg = yaml.safe_load(open(REPO / "config.yaml"))
    bands = cfg["rules"]["bands"]
    ap = build_enriched()
    ap = run_rules(ap)

    banded = ap[ap.score >= bands["review"]].copy()
    banded["band"] = np.where(banded.score >= bands["decline"], "decline", "review")
    banded = banded.sort_values("ts").reset_index(drop=True)
    banded["alert_id"] = np.arange(1, len(banded) + 1)

    out = banded[["alert_id", "order_id", "user_id", "ts", "score", "band"]].copy()
    out["fired_rules"] = banded.fired_rules.map(json.dumps)
    out.to_csv(REPO / "data" / "alerts.csv", index=False)

    db = cfg["db"]
    eng = sa.create_engine(
        f"mysql+pymysql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
    )
    with eng.begin() as c:
        c.execute(sa.text("DELETE FROM alerts"))
        out.to_sql("alerts", c, if_exists="append", index=False, chunksize=5000)

    days = (ap.ts.max() - ap.ts.min()).days or 1
    print(f"rules engine: {len(ap):,d} approved orders scored in {time.time() - t0:.0f}s")
    print(f"alerts: {len(banded):,d} ({len(banded) / days:.1f}/day) | "
          f"review {sum(banded.band == 'review'):,d} | decline {sum(banded.band == 'decline'):,d}")
    per_rule = {r.id: int(ap[f"fired_{r.id}"].sum()) for r in RULES if f"fired_{r.id}" in ap}
    print("fires:", per_rule)


if __name__ == "__main__":
    main()
