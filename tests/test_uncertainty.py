"""Interval arithmetic, the paired prompt test, and the published counts behind them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.uncertainty import (
    bootstrap_pr_auc,
    mcnemar_exact,
    paired_prompt_tests,
    published_rates,
    rules_counts,
    threshold_sensitivity,
    wilson_interval,
)

REPO = Path(__file__).resolve().parents[1]


def _rate(name: str) -> dict:
    return next(row for row in published_rates() if row["rate"] == name)


def test_wilson_matches_a_hand_computable_case() -> None:
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.27753, abs=1e-5)


def test_wilson_stays_inside_the_unit_interval_at_the_boundaries() -> None:
    for successes, trials in [(0, 3), (3, 3), (1, 1), (0, 1)]:
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_rules_counts_match_the_committed_operating_point() -> None:
    operating = json.loads((REPO / "reports" / "operating_point.json").read_text())
    counts = rules_counts()
    assert counts == {
        "review_band": 30,
        "alerts": 2119,
        "alerts_on_fraud": 189,
        "holdout_fraud_orders": 316,
    }
    assert counts["alerts"] == operating["n_alerts"]
    assert counts["alerts_on_fraud"] / counts["alerts"] == pytest.approx(
        operating["precision"], abs=5e-4
    )
    assert counts["alerts_on_fraud"] / counts["holdout_fraud_orders"] == pytest.approx(
        operating["recall_overall"], abs=5e-4
    )


@pytest.mark.parametrize(
    ("name", "successes", "trials", "low", "high"),
    [
        ("rules precision at review ≥ 30", 189, 2119, 0.0778, 0.1021),
        ("rules recall over holdout fraud orders", 189, 316, 0.5432, 0.6507),
        ("claude-sonnet-5 memo_v2 action accuracy", 147, 200, 0.6698, 0.7913),
        (
            "gpt-5.6-luna memo_v2 action accuracy (valid outputs)",
            119,
            199,
            0.5286,
            0.6636,
        ),
        (
            "gpt-5.6-luna memo_v2 action accuracy (schema failure counted wrong)",
            119,
            200,
            0.5258,
            0.6606,
        ),
    ],
)
def test_published_intervals(
    name: str, successes: int, trials: int, low: float, high: float
) -> None:
    row = _rate(name)
    assert (row["successes"], row["trials"]) == (successes, trials)
    assert row["low"] == pytest.approx(low, abs=1e-4)
    assert row["high"] == pytest.approx(high, abs=1e-4)


def test_luna_denominators_differ_by_its_schema_failure() -> None:
    valid = _rate("gpt-5.6-luna memo_v2 action accuracy (valid outputs)")
    inclusive = _rate(
        "gpt-5.6-luna memo_v2 action accuracy (schema failure counted wrong)"
    )
    assert inclusive["trials"] == valid["trials"] + 1
    assert inclusive["successes"] == valid["successes"]
    assert inclusive["point"] < valid["point"]


def test_mcnemar_exact_matches_the_binomial_tail() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 2**10)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def test_prompt_change_is_significant_on_the_discordant_pairs() -> None:
    action, unsupported = paired_prompt_tests()
    assert action["comparison"] == "action correct"
    assert action["paired_cases"] == 200
    assert (action["improved_by_v2"], action["worsened_by_v2"]) == (33, 8)
    assert action["p_value"] < 0.001
    assert unsupported["improved_by_v2"] > unsupported["worsened_by_v2"]
    assert unsupported["p_value"] < 0.001


def test_bootstrap_is_seeded_and_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], [400, 40])
    scores = rng.normal(labels * 1.5, 1.0)
    first = bootstrap_pr_auc(labels, scores, resamples=50, seed=416)
    second = bootstrap_pr_auc(labels, scores, resamples=50, seed=416)
    assert first == second
    assert first["low"] < first["point"] < first["high"]


def test_threshold_sensitivity_reads_the_committed_frontier() -> None:
    operating = json.loads((REPO / "reports" / "operating_point.json").read_text())
    sensitivity = threshold_sensitivity()
    assert sensitivity["chosen"]["review_band"] == operating["review_band"]
    assert sensitivity["chosen"]["net_usd"] == operating["net_usd"]
    nets = [point["net_usd"] for point in sensitivity["points"]]
    assert nets == sorted(nets, reverse=True)
