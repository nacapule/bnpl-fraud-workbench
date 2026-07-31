"""Third-party email/IP risk enrichment (IPQualityScore), fixture-first.

Two modes:

* ``--live``: query IPQS for a sample of flagged + benign entities
  (config ``vendor:``), store sanitized raw responses under
  ``vendor/fixtures/raw/`` and an aggregate ``vendor/fixtures/scores.csv``
  with ``source=ipqs_live``. Needs ``IPQS_API_KEY`` (in ``.env`` or env) and
  account credits.
* default (no key / no credits / offline): write **synthetic stand-in scores**
  with ``source=synthetic`` — deterministic, seeded, and deliberately noisy
  (risky entities mostly-high, benign mostly-low, with overlap). These exist so
  rule R12 and the demo run offline. They are NOT vendor data and every report
  that uses them says so.

R12 consumes ``vendor/fixtures/scores.csv`` regardless of source.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "vendor" / "fixtures"
IPQS_BASE = "https://ipqualityscore.com/api/json"


def _load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _config() -> dict:
    with open(REPO / "config.yaml") as f:
        return yaml.safe_load(f)


def sample_targets() -> pd.DataFrame:
    """Sample flagged vs benign (email, ip) pairs from generated data.

    'Flagged' = order appears in data/alerts.csv (rules output) — NOT labels;
    the vendor layer is analyst-facing and never reads ground truth.
    """
    cfg = _config()["vendor"]
    orders = pd.read_csv(REPO / "data" / "orders.csv", usecols=["order_id", "user_id", "ip"])
    users = pd.read_csv(REPO / "data" / "users.csv", usecols=["user_id", "email"])
    alerts = pd.read_csv(REPO / "data" / "alerts.csv", usecols=["order_id"])
    df = orders.merge(users, on="user_id")
    df["flagged"] = df["order_id"].isin(set(alerts["order_id"]))
    rng = np.random.default_rng(_config()["seed"])
    flagged = df[df.flagged].drop_duplicates("user_id")
    benign = df[~df.flagged].drop_duplicates("user_id")
    n_f = min(cfg["ipqs_sample_flagged"], len(flagged))
    n_b = min(cfg["ipqs_sample_benign"], len(benign))
    take_f = flagged.iloc[rng.choice(len(flagged), n_f, replace=False)]
    take_b = benign.iloc[rng.choice(len(benign), n_b, replace=False)]
    return pd.concat([take_f, take_b], ignore_index=True)


def enrich_live(targets: pd.DataFrame) -> list[dict]:
    import requests

    key = os.environ.get("IPQS_API_KEY")
    if not key:
        raise SystemExit("IPQS_API_KEY not set (put it in .env)")
    raw_dir = FIXTURES / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for _, t in targets.iterrows():
        for kind, value in (("ip", t.ip), ("email", t.email)):
            if (kind, value) in seen:
                continue
            seen.add((kind, value))
            url = f"{IPQS_BASE}/{kind}/{key}/{value}"
            r = requests.get(url, params={"strictness": 1}, timeout=20)
            d = r.json()
            if not d.get("success", False):
                raise SystemExit(f"IPQS error on {kind} {value}: {d.get('message')}")
            d.pop("request_id", None)  # sanitize
            (raw_dir / f"{kind}_{abs(hash(value)) % 10**10}.json").write_text(
                json.dumps(d, indent=1, sort_keys=True)
            )
            rows.append(
                {
                    "kind": kind,
                    "value": value,
                    "fraud_score": d.get("fraud_score", 0),
                    "flagged_sample": bool(t.flagged),
                    "source": "ipqs_live",
                }
            )
            time.sleep(0.15)  # stay friendly to rate limits
    return rows


def enrich_synthetic(targets: pd.DataFrame) -> list[dict]:
    """Deterministic stand-in scores. Overlapping distributions on purpose:
    vendor signals in the real world are noisy corroboration, not oracles."""
    rng = np.random.default_rng(_config()["seed"] + 12)
    rows = []
    for _, t in targets.iterrows():
        for kind, value in (("ip", t.ip), ("email", t.email)):
            if t.flagged:
                score = min(100, max(0, int(rng.normal(72, 18))))
            else:
                score = min(100, max(0, int(rng.normal(22, 16))))
            rows.append(
                {
                    "kind": kind,
                    "value": value,
                    "fraud_score": score,
                    "flagged_sample": bool(t.flagged),
                    "source": "synthetic",
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="query IPQS (needs key + credits)")
    args = ap.parse_args()
    _load_env()
    targets = sample_targets()
    rows = enrich_live(targets) if args.live else enrich_synthetic(targets)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / "scores.csv"
    cols = ["kind", "value", "fraud_score", "flagged_sample", "source"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    src = rows[0]["source"] if rows else "none"
    print(f"wrote {len(rows)} scores to {out} (source={src})")
    if src == "synthetic":
        print("NOTE: synthetic stand-in scores — run with --live for real IPQS data")


if __name__ == "__main__":
    main()
