"""Entry point: python -m simulator.generate

Builds the full synthetic world (benign behavior + injected fraud patterns)
and writes CSVs + stories.jsonl to data/. Deterministic under config seed;
SIM_SCALE (float env var) scales volumes for fast test runs.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulator.behavior import World, run_benign
from simulator.patterns import inject_all
from simulator.population import build_population

REPO = Path(__file__).resolve().parent.parent
TS_FMT = "%Y-%m-%d %H:%M:%S"


def load_scaled_config() -> dict:
    with open(REPO / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    scale = float(os.environ.get("SIM_SCALE", "1.0"))
    if scale != 1.0:
        s = cfg["simulator"]
        s["n_users"] = max(500, int(s["n_users"] * scale))
        s["target_orders"] = max(2000, int(s["target_orders"] * scale))
        s["n_merchants"] = max(30, int(s["n_merchants"] * min(1.0, scale * 4)))
        for k in list(s["patterns"]):
            if k.endswith(("_accounts", "_clusters", "_rings", "_bustouts")):
                s["patterns"][k] = max(2, int(s["patterns"][k] * max(scale, 0.05)))
        s["patterns"]["merchant_bustouts"] = min(2, s["patterns"]["merchant_bustouts"])
        s["patterns"]["synth_rings"] = min(3, max(1, s["patterns"]["synth_rings"]))
        s["patterns"]["promo_clusters"] = min(2, max(1, s["patterns"]["promo_clusters"]))
    return cfg


def _fmt_ts(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            df[c] = s.dt.strftime(TS_FMT)
            df[c] = df[c].where(s.notna(), "")
    return df


def main() -> None:
    t0 = time.time()
    cfg = load_scaled_config()
    rng = np.random.default_rng(cfg["seed"])
    start = datetime.fromisoformat(cfg["simulator"]["start_date"])
    end = start + timedelta(days=30 * cfg["simulator"]["months"])

    pop = build_population(cfg, rng)
    world = World(pop=pop, start=start, end=end)
    run_benign(cfg, rng, world)
    inject_all(cfg, rng, world)

    out = Path(os.environ.get("SIM_OUT", REPO / "data"))
    out.mkdir(exist_ok=True)

    frames: dict[str, pd.DataFrame] = {
        "users": _fmt_ts(pop.users, ["signup_ts"]),
        "devices": pop.devices,
        "user_devices": _fmt_ts(pop.user_devices, ["first_seen", "last_seen"]),
        "cards": pop.cards,
        "addresses": _fmt_ts(pop.addresses, ["added_ts"]),
        "merchants": _fmt_ts(pop.merchants, ["onboarded_ts"]),
        "orders": _fmt_ts(pd.DataFrame(world.orders), ["ts"]),
        "plans": pd.DataFrame(world.plans),
        "installments": _fmt_ts(pd.DataFrame(world.installments), ["due_ts", "paid_ts"]),
        "payments": _fmt_ts(pd.DataFrame(world.payments), ["ts"]),
        "account_events": _fmt_ts(pd.DataFrame(world.events), ["ts"]),
        "promos": _fmt_ts(pd.DataFrame(world.promos), ["valid_from", "valid_to"]),
        "promo_redemptions": _fmt_ts(pd.DataFrame(world.promo_redemptions), ["ts"]),
        "chargebacks": _fmt_ts(pd.DataFrame(world.chargebacks), ["opened_ts"]),
        "labels": pd.DataFrame(world.labels),
    }
    for name, df in frames.items():
        df.to_csv(out / f"{name}.csv", index=False)

    with open(out / "stories.jsonl", "w") as f:
        for s in world.stories:
            f.write(json.dumps(s, sort_keys=True) + "\n")

    orders = frames["orders"]
    approved = orders[orders.status == "approved"]
    labeled = frames["labels"].order_id.nunique()
    labeled_approved = frames["labels"][
        frames["labels"].order_id.isin(set(approved.order_id))
    ].order_id.nunique()
    rate = labeled_approved / max(len(approved), 1)
    print(f"generated in {time.time() - t0:.1f}s")
    for name, df in frames.items():
        print(f"  {name:18s} {len(df):>9,d} rows")
    print(f"  stories            {len(world.stories):>9,d}")
    print(f"approved orders: {len(approved):,d} | labeled fraud (approved): "
          f"{labeled_approved:,d} ({rate:.2%}) | labeled total: {labeled:,d}")


if __name__ == "__main__":
    main()
