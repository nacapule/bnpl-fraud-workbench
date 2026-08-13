"""Roster construction, stream replication, and the frontier's base cell."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from queue_sim.simulate import load_inputs, run_policy
from queue_sim.sweep import (
    HEADCOUNTS,
    POLICY,
    coverage_ceiling,
    replicate,
    roster,
    roster_config,
    run_cell,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    with (REPO / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def holdout_alerts() -> tuple[pd.DataFrame, dict]:
    alerts, cfg, _ = load_inputs()
    return alerts, cfg


def _committed_queue_rows() -> dict[str, dict]:
    records = {}
    inside = False
    for line in (REPO / "reports" / "queue.md").read_text().splitlines():
        if line.strip() == "```json":
            inside = True
            continue
        if inside and line.strip() == "```":
            break
        if inside and line.strip():
            record = json.loads(line)
            records[record["policy"]] = record
    return records


def test_roster_reuses_the_configured_analysts(config: dict) -> None:
    configured = config["queue"]["analysts"]
    assert roster(config, 1) == [configured[0]]
    assert roster(config, len(configured)) == configured


def test_roster_extends_the_shift_pattern(config: dict) -> None:
    extended = roster(config, 4)
    assert len(extended) == 4
    assert extended[:2] == config["queue"]["analysts"]
    for analyst in extended:
        assert len(analyst["shift_days"]) == 5
        assert len(set(analyst["shift_days"])) == 5
    covered = {day for analyst in extended for day in analyst["shift_days"]}
    assert covered == set(range(7))
    assert [analyst["shift_start"] for analyst in extended[2:]] == ["08:00", "11:00"]


def test_roster_config_leaves_the_rest_of_the_config_alone(config: dict) -> None:
    swapped = roster_config(config, 3)
    assert len(swapped["queue"]["analysts"]) == 3
    assert swapped["seed"] == config["seed"]
    assert (
        swapped["queue"]["service_time_arithmetic_mean_min"]
        == config["queue"]["service_time_arithmetic_mean_min"]
    )
    assert config["queue"]["analysts"] == roster(config, len(config["queue"]["analysts"]))


def test_replicate_scales_volume_and_keeps_arrivals_sorted() -> None:
    alerts = pd.DataFrame(
        {
            "alert_id": [1, 2, 3],
            "ts": pd.to_datetime(["2026-04-01", "2026-04-03", "2026-04-02"]),
            "score": [30, 40, 50],
        }
    )
    scaled = replicate(alerts, 4)
    assert len(scaled) == 12
    assert scaled["ts"].is_monotonic_increasing
    assert scaled["alert_id"].value_counts().to_dict() == {1: 4, 2: 4, 3: 4}
    assert replicate(alerts, 1).equals(alerts)


def test_base_cell_reproduces_the_committed_queue_report(
    holdout_alerts: tuple[pd.DataFrame, dict],
) -> None:
    alerts, cfg = holdout_alerts
    cell = run_cell(alerts, cfg, 1, len(cfg["queue"]["analysts"]))
    committed = _committed_queue_rows()[POLICY]
    for key in ["n_alerts", "sla_attainment", "ttd_p50_h", "ttd_p90_h",
                "max_backlog", "fraud_blocked_usd", "fraud_blocked_n"]:
        assert cell[key] == committed[key]


def test_more_analysts_never_shrink_coverage(
    holdout_alerts: tuple[pd.DataFrame, dict],
) -> None:
    alerts, cfg = holdout_alerts
    unreachable = [
        coverage_ceiling(alerts, cfg, count)["unreachable_alerts"] for count in HEADCOUNTS
    ]
    assert unreachable == sorted(unreachable, reverse=True)
    assert unreachable[0] > unreachable[-1]


def test_doubling_arrivals_never_improves_the_sla(
    holdout_alerts: tuple[pd.DataFrame, dict],
) -> None:
    alerts, cfg = holdout_alerts
    single = run_cell(alerts, cfg, 1, 1)
    doubled = run_cell(alerts, cfg, 2, 1)
    assert doubled["sla_attainment"] < single["sla_attainment"]
    assert doubled["n_alerts"] == 2 * single["n_alerts"]


def test_cells_are_deterministic(holdout_alerts: tuple[pd.DataFrame, dict]) -> None:
    alerts, cfg = holdout_alerts
    assert run_cell(alerts, cfg, 1, 2) == run_cell(alerts, cfg, 1, 2)


def test_run_policy_is_unaffected_by_the_roster_helper(
    holdout_alerts: tuple[pd.DataFrame, dict],
) -> None:
    alerts, cfg = holdout_alerts
    direct = run_policy(alerts.copy(), cfg, POLICY, np.random.default_rng(cfg["seed"]))
    swapped = run_policy(
        alerts.copy(),
        roster_config(cfg, len(cfg["queue"]["analysts"])),
        POLICY,
        np.random.default_rng(cfg["seed"]),
    )
    direct.pop("backlog_curve")
    swapped.pop("backlog_curve")
    assert direct == swapped
