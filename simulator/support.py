"""Per-pattern support of a world: how many fraud orders and episodes each
evaluation window holds, and how many of each pattern's orders end up labelled.

The protocol's support check (experiments/protocol.yaml ``world_size.support``)
needs every checkout pattern (config/world.yaml ``support``) to have at least a
minimum number of processor-approved orders from at least a minimum number of
independent episodes in the validation and test windows of each development
seed. Latent truth is read here only to report the world's composition; nothing
that decides reads it.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core import config, world
from core.protocol import Protocol, load_protocol

WINDOWS = ("warm_up", "fit", "calibration", "validation", "test")


def support_table(tables: Mapping[str, pd.DataFrame], protocol: Protocol) -> pd.DataFrame:
    """One row per pattern and window: approved orders, attempts, episodes."""
    orders = tables["order_attempts"][["order_id", "occurred_at", "processor_result"]].merge(
        tables["latent_orders"][["order_id", "pattern_id", "episode_id"]], on="order_id")
    orders = orders[orders["pattern_id"].notna()].copy()
    orders["window"] = protocol.window_of(orders["occurred_at"])
    rows = []
    for pattern in world.PATTERNS:
        mine = orders[orders["pattern_id"] == pattern]
        for window in (*WINDOWS, "all"):
            part = mine if window == "all" else mine[mine["window"] == window]
            approved = part[part["processor_result"] == "approved"]
            rows.append({"pattern": pattern, "window": window, "approved": len(approved),
                         "attempts": len(part), "episodes": approved["episode_id"].nunique()})
    return pd.DataFrame(rows)


def labelled_share(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per pattern: approved latent orders, how many carry a positive label by the
    end of observation, and the bases of those labels."""
    orders = tables["order_attempts"]
    approved = orders.loc[orders["processor_result"] == "approved", ["order_id"]].merge(
        tables["latent_orders"][["order_id", "pattern_id"]], on="order_id")
    labels = tables["labels"]
    positive = labels[labels["label"] == 1].drop_duplicates("order_id")
    merged = approved.merge(positive[["order_id", "basis"]], on="order_id", how="left")
    merged["pattern_id"] = merged["pattern_id"].fillna("legitimate")
    rows = []
    for pattern, group in merged.groupby("pattern_id", sort=True):
        bases = group["basis"].value_counts().sort_index()
        rows.append({"pattern": pattern, "orders": len(group),
                     "labelled": int(group["basis"].notna().sum()),
                     "bases": ", ".join(f"{k} {v}" for k, v in bases.items())})
    return pd.DataFrame(rows)


def check_support(tables: Mapping[str, pd.DataFrame], protocol: Protocol | None = None,
                  ) -> list[str]:
    """Shortfalls against the protocol's minimums (an empty list when met)."""
    protocol = protocol or load_protocol()
    spec = protocol.raw["world_size"]["support"]
    patterns = config.load("world")["support"]["checkout_patterns"]
    table = support_table(tables, protocol).set_index(["pattern", "window"])
    problems = []
    for pattern in patterns:
        for window in spec["windows"]:
            row = table.loc[(pattern, window)]
            if row["approved"] < spec["min_test_orders_per_checkout_pattern"]:
                problems.append(f"{pattern} {window}: {row['approved']} orders")
            if row["episodes"] < spec["min_episodes_per_checkout_pattern"]:
                problems.append(f"{pattern} {window}: {row['episodes']} episodes")
    return problems


def support_report(tables: Mapping[str, pd.DataFrame], protocol: Protocol | None = None) -> str:
    protocol = protocol or load_protocol()
    table = support_table(tables, protocol)
    wide = table.pivot(index="pattern", columns="window", values="approved")[[*WINDOWS, "all"]]
    episodes = table.pivot(index="pattern", columns="window", values="episodes")
    lines = ["approved fraud orders (episodes) by window:"]
    lines.append("  " + "pattern".ljust(13) + "".join(w.rjust(14) for w in (*WINDOWS, "all")))
    for pattern in wide.index:
        cells = "".join(f"{wide.loc[pattern, w]:>8d} ({episodes.loc[pattern, w]:>3d})"
                        for w in (*WINDOWS, "all"))
        lines.append("  " + pattern.ljust(13) + cells)
    orders = tables["order_attempts"]
    fraud = tables["latent_orders"]["pattern_id"].notna()
    lines.append(f"fraud share of order attempts: {fraud.mean():.3%} "
                 f"({int(fraud.sum()):,d} of {len(orders):,d})")
    if "labels" in tables:
        lines.append("labelled share of each pattern's approved orders:")
        for row in labelled_share(tables).itertuples():
            lines.append(f"  {row.pattern:13s} {row.labelled:>6,d} of {row.orders:>7,d}"
                         f"  [{row.bases}]")
    problems = check_support(tables, protocol)
    lines.append("support check: " + ("met" if not problems else "SHORT: " + "; ".join(problems)))
    return "\n".join(lines)
