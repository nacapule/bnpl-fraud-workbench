"""Build the stratified eval set: N alerts with ground truth + frozen packets.

This is the ONE consumer of ``labels`` on the LLM side, and it runs offline
before evaluation; the packets it freezes contain no truth fields (the packet
builder enforces that). Selection is deterministic (config seed).

Output (committed to the repo so CI and reviewers reproduce with zero keys):
  llm/eval/cases.json          [{alert_id, truth_pattern, truth_action}, ...]
  llm/eval/packets/<id>.json   frozen packets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm.client import load_config
from llm.packet import build_packet, get_engine

EVAL_DIR = Path(__file__).resolve().parent

PATTERN_TO_TAXONOMY = {
    "P-ATO": "account_takeover",
    "P-STOLEN": "stolen_card",
    "P-SYNTH": "synthetic_ring",
    "P-NEVERPAY": "never_pay",
    "P-INR-ABUSE": "inr_abuse",
    "P-PROMO": "promo_abuse",
    "P-MERCH": "merchant_bustout",
}

# Ground-truth action mapping, defined once (FP-1 §3/§5 ladder):
TRUTH_ACTION = {
    "account_takeover": "decline_block",
    "stolen_card": "decline_block",
    "synthetic_ring": "decline_block",
    "never_pay": "decline_block",
    "inr_abuse": "hold_contact",
    "promo_abuse": "hold_contact",
    "merchant_bustout": "escalate",
    "benign": "clear",
}


def main() -> None:
    cfg = load_config()
    n_total = cfg["llm_eval"]["n_cases"]
    n_neg_min = cfg["llm_eval"]["hard_negatives_min"]
    rng = np.random.default_rng(cfg["seed"] + 7)

    alerts = pd.read_csv("data/alerts.csv", usecols=["alert_id", "order_id", "user_id"])
    labels = pd.read_csv("data/labels.csv")
    lab_by_order = labels.set_index("order_id")["pattern_id"].to_dict()
    alerts["pattern"] = alerts["order_id"].map(lab_by_order)
    alerts["taxonomy"] = alerts["pattern"].map(PATTERN_TO_TAXONOMY).fillna("benign")

    frauds = alerts[alerts.taxonomy != "benign"]
    negatives = alerts[alerts.taxonomy == "benign"]

    n_neg = max(n_neg_min, n_total - min(len(frauds), n_total - n_neg_min))
    n_fraud_budget = n_total - n_neg
    per_pattern = max(1, n_fraud_budget // frauds["taxonomy"].nunique())

    chosen: list[pd.DataFrame] = []
    for _pat, grp in frauds.groupby("taxonomy"):
        k = min(per_pattern, len(grp))
        chosen.append(grp.iloc[rng.choice(len(grp), k, replace=False)])
    fraud_sel = pd.concat(chosen)
    # top up to the fraud budget from the biggest classes if quotas undershot
    if len(fraud_sel) < n_fraud_budget:
        rest = frauds.drop(fraud_sel.index)
        k = min(n_fraud_budget - len(fraud_sel), len(rest))
        fraud_sel = pd.concat([fraud_sel, rest.iloc[rng.choice(len(rest), k, replace=False)]])
    neg_sel = negatives.iloc[rng.choice(len(negatives), min(n_neg, len(negatives)), replace=False)]
    sel = pd.concat([fraud_sel, neg_sel]).sample(frac=1, random_state=cfg["seed"]).reset_index()

    cases = [
        {
            "alert_id": int(r.alert_id),
            "truth_pattern": r.taxonomy,
            "truth_action": TRUTH_ACTION[r.taxonomy],
        }
        for r in sel.itertuples()
    ]
    (EVAL_DIR / "cases.json").write_text(json.dumps(cases, indent=1))

    pdir = EVAL_DIR / "packets"
    pdir.mkdir(exist_ok=True)
    engine = get_engine()
    for i, c in enumerate(cases):
        out = pdir / f"{c['alert_id']}.json"
        if not out.exists():
            pkt = build_packet(c["alert_id"], engine)
            out.write_text(json.dumps(pkt, indent=1, sort_keys=True))
        if (i + 1) % 25 == 0:
            print(f"packets {i + 1}/{len(cases)}")

    mix = sel["taxonomy"].value_counts().to_dict()
    print(f"eval set: {len(cases)} cases — {mix}")


if __name__ == "__main__":
    main()
