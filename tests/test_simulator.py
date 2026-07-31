"""Simulator invariants: determinism, base rate, pattern presence, story integrity.

Runs at SIM_SCALE=0.03 in a temp dir (fast); the committed demo uses full scale.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gen_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sim")
    _run(out)
    return out


def _run(out: Path) -> None:
    env = os.environ | {"SIM_SCALE": "0.03", "SIM_OUT": str(out)}
    subprocess.run(
        [sys.executable, "-m", "simulator.generate"], cwd=REPO, env=env, check=True,
        capture_output=True,
    )


def _hashes(d: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.glob("*.csv")) + [d / "stories.jsonl"]
    }


def test_determinism(gen_dir: Path, tmp_path: Path) -> None:
    _run(tmp_path)
    assert _hashes(gen_dir) == _hashes(tmp_path)


def test_base_rate_bounds(gen_dir: Path) -> None:
    orders = pd.read_csv(gen_dir / "orders.csv")
    labels = pd.read_csv(gen_dir / "labels.csv")
    approved = orders[orders.status == "approved"]
    rate = labels[labels.order_id.isin(set(approved.order_id))].order_id.nunique() / len(approved)
    assert 0.004 < rate < 0.05, rate  # wider at small scale; full-scale target ~1%


def test_all_patterns_present(gen_dir: Path) -> None:
    labels = pd.read_csv(gen_dir / "labels.csv")
    expected = {"P-ATO", "P-STOLEN", "P-SYNTH", "P-NEVERPAY", "P-INR-ABUSE", "P-PROMO", "P-MERCH"}
    assert expected <= set(labels.pattern_id.unique())


def test_repayment_shares(gen_dir: Path) -> None:
    inst = pd.read_csv(gen_dir / "installments.csv")
    labels = pd.read_csv(gen_dir / "labels.csv")
    plans = pd.read_csv(gen_dir / "plans.csv")
    benign_plans = plans[~plans.order_id.isin(set(labels.order_id))]
    b = inst[inst.plan_id.isin(set(benign_plans.plan_id))]
    paid_share = (b.outcome.isin(["paid", "late"])).mean()
    assert paid_share > 0.93, paid_share  # benign book mostly repays
    wo_share = (b.outcome == "written_off").mean()
    assert 0.005 < wo_share < 0.09, wo_share  # hardship exists but is bounded


def test_stories_reference_real_rows(gen_dir: Path) -> None:
    orders = set(pd.read_csv(gen_dir / "orders.csv").order_id)
    users = set(pd.read_csv(gen_dir / "users.csv").user_id)
    with open(gen_dir / "stories.jsonl") as f:
        for line in f:
            s = json.loads(line)
            assert set(s["order_ids"]) <= orders, s["story_id"]
            assert set(s["user_ids"]) <= users, s["story_id"]


def test_labels_only_on_existing_orders(gen_dir: Path) -> None:
    orders = pd.read_csv(gen_dir / "orders.csv")
    labels = pd.read_csv(gen_dir / "labels.csv")
    assert set(labels.order_id) <= set(orders.order_id)


def test_benign_edge_cases_exist(gen_dir: Path) -> None:
    """Travelers/foreign-IP benign orders must exist or the eval set has no
    hard negatives."""
    orders = pd.read_csv(gen_dir / "orders.csv")
    labels = pd.read_csv(gen_dir / "labels.csv")
    benign = orders[~orders.order_id.isin(set(labels.order_id)) & (orders.status == "approved")]
    assert (benign.ip_country != "US").mean() > 0.005
