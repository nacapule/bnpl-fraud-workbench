"""Build the stratified eval set: N alerts with ground truth + frozen packets.

This is the ONE consumer of ``labels`` on the LLM side, and it runs offline
before evaluation; the packets it freezes contain no truth fields (the packet
builder enforces that). Selection is deterministic (config seed).

Output (committed to the repo so CI and reviewers reproduce with zero keys):
  llm/eval/cases.json          [{alert_id, truth_pattern, truth_action}, ...]
  llm/eval/packets/<id>.json   frozen packets
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm.client import load_config
from llm.packet import build_packet, get_engine

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parents[1]
PACKETS_DIR = EVAL_DIR / "packets"
PROVENANCE_PATH = EVAL_DIR / "provenance.json"

PATTERN_TO_TAXONOMY = {
    "P-ATO": "account_takeover",
    "P-STOLEN": "stolen_card",
    "P-SYNTH": "synthetic_ring",
    "P-NEVERPAY": "never_pay",
    "P-INR-ABUSE": "inr_abuse",
    "P-PROMO": "promo_abuse",
    "P-MERCH": "merchant_bustout",
}

# Ground-truth actions, derived from the FP-1 §3/§5 ladder. Where the policy
# itself names more than one defensible action, the truth is a SET and accuracy
# means "chose an in-policy action" (primary action listed first — used for
# reporting). Corrected 2026-07-31 after the v1 run exposed that the original
# single-action mapping contradicted FP-1 §5 (rings escalate); the correction
# is logged in ITERATION.md and applied to every arm/version equally.
TRUTH_ACTIONS: dict[str, list[str]] = {
    "account_takeover": ["decline_block", "escalate"],  # §5-ATO; §3 escalate >$2k aggregate
    "stolen_card": ["decline_block"],
    "synthetic_ring": ["escalate", "decline_block"],    # §5: rings escalate with linkage
    "never_pay": ["decline_block"],
    "inr_abuse": ["hold_contact"],
    "promo_abuse": ["hold_contact", "escalate"],        # §3: clusters are suspected rings
    "merchant_bustout": ["escalate"],
    "benign": ["clear"],
}
TRUTH_ACTION = {k: v[0] for k, v in TRUTH_ACTIONS.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_directory_sha256(packet_dir: Path = PACKETS_DIR) -> str:
    """Hash packet names and bytes in a stable order."""
    digest = hashlib.sha256()
    for path in sorted(packet_dir.glob("*.json")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def current_provenance(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    return {
        "rules_bands": {
            "review": cfg["rules"]["bands"]["review"],
            "decline": cfg["rules"]["bands"]["decline"],
        },
        "seed": cfg["seed"],
        "alerts_csv_sha256": _sha256_file(REPO / "data" / "alerts.csv"),
        "packet_dir_sha256": packet_directory_sha256(),
    }


def assert_provenance(cfg: dict[str, Any] | None = None) -> None:
    if not PROVENANCE_PATH.exists():
        raise RuntimeError(f"missing frozen-eval provenance: {PROVENANCE_PATH}")
    expected = json.loads(PROVENANCE_PATH.read_text())
    actual = current_provenance(cfg)
    if actual != expected:
        raise RuntimeError(
            "frozen-eval provenance mismatch; data, rules bands, seed, or packets changed\n"
            f"expected: {json.dumps(expected, sort_keys=True)}\n"
            f"actual:   {json.dumps(actual, sort_keys=True)}"
        )


def select_case_metadata(
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Return deterministic case metadata and the selected alert rows."""
    cfg = cfg or load_config()
    n_total = cfg["llm_eval"]["n_cases"]
    n_neg_min = cfg["llm_eval"]["hard_negatives_min"]
    rng = np.random.default_rng(cfg["seed"] + 7)

    alerts = pd.read_csv(
        REPO / "data" / "alerts.csv", usecols=["alert_id", "order_id", "user_id"]
    )
    labels = pd.read_csv(REPO / "data" / "labels.csv")
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
            "truth_actions": TRUTH_ACTIONS[r.taxonomy],
        }
        for r in sel.itertuples()
    ]
    return cases, sel


def main() -> None:
    cfg = load_config()
    assert_provenance(cfg)
    cases, selected = select_case_metadata(cfg)
    (EVAL_DIR / "cases.json").write_text(json.dumps(cases, indent=1))

    engine = None
    for i, (case, row) in enumerate(zip(cases, selected.itertuples(), strict=True)):
        out = PACKETS_DIR / f"{case['alert_id']}.json"
        expected_order_id = int(row.order_id)
        if not out.exists():
            engine = engine or get_engine()
            packet = build_packet(case["alert_id"], engine)
            out.write_text(json.dumps(packet, indent=1, sort_keys=True))
        else:
            packet = json.loads(out.read_text())
        packet_order_id = int(packet.get("alert", {}).get("order_id", -1))
        if packet_order_id != expected_order_id:
            raise RuntimeError(
                f"stale packet {out}: order_id {packet_order_id}, expected {expected_order_id}"
            )
        if (i + 1) % 25 == 0:
            print(f"packets {i + 1}/{len(cases)}")

    mix = selected["taxonomy"].value_counts().to_dict()
    print(f"eval set: {len(cases)} cases — {mix}")


if __name__ == "__main__":
    main()
