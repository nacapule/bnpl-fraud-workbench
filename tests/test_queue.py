"""Queue simulation invariants on hand-computable toys."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from queue_sim.simulate import (
    Analyst,
    business_hours_between,
    finish_service,
    run_policy,
    sample_service_minutes,
)

CFG = {
    "queue": {
        "analysts": [
            {"name": "A1", "shift_days": [0, 1, 2, 3, 4], "shift_start": "09:00",
             "productive_hours": 8.0},
        ],
        "service_time_arithmetic_mean_min": 10.0,
        "service_time_sigma": 0.0001,  # ~deterministic
        "fulfillment_lag_hours": 12,
        "sla_target_hours": 4,
    },
}


def _alerts(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["llm_priority"] = np.nan
    return df


def test_alert_conservation_and_exact_ttd() -> None:
    # Monday 2026-06-01. Three alerts at 09:00, 09:01, 09:02; one analyst,
    # 10-minute service: resolutions at 09:10, 09:20, 09:30.
    rows = [
        {"alert_id": i, "order_id": i, "ts": datetime(2026, 6, 1, 9, i), "score": 50,
         "is_fraud": False, "loss_at_stake": 0.0}
        for i in range(3)
    ]
    r = run_policy(_alerts(rows), CFG, "fifo", np.random.default_rng(1))
    assert r["n_alerts"] == 3
    # 09:00 -> 09:10 = 10min; 09:01 -> 09:20 ~ 19min; 09:02 -> 09:30 ~ 28min
    assert abs(r["ttd_p50_h"] - (19 / 60)) < 0.02
    assert r["sla_attainment"] == 1.0


def test_priority_blocks_more_fraud_than_fifo() -> None:
    # Ten benign alerts arrive first; a high-value fraud alert arrives last but
    # ships in 12h. Under FIFO it waits behind everything into the next day;
    # under score priority it jumps the queue and is resolved pre-ship.
    rows = [
        {"alert_id": i, "order_id": i, "ts": datetime(2026, 6, 1, 16, 0 + i), "score": 30,
         "is_fraud": False, "loss_at_stake": 0.0}
        for i in range(10)
    ]
    rows.append({"alert_id": 99, "order_id": 99, "ts": datetime(2026, 6, 1, 16, 30),
                 "score": 120, "is_fraud": True, "loss_at_stake": 900.0})
    fifo = run_policy(_alerts(rows), CFG, "fifo", np.random.default_rng(2))
    prio = run_policy(_alerts(rows), CFG, "score", np.random.default_rng(2))
    assert prio["fraud_blocked_usd"] >= fifo["fraud_blocked_usd"]
    assert prio["fraud_blocked_usd"] == 900.0


def test_determinism() -> None:
    rows = [
        {"alert_id": i, "order_id": i, "ts": datetime(2026, 6, 1, 10, i), "score": 40 + i,
         "is_fraud": i % 2 == 0, "loss_at_stake": 50.0}
        for i in range(8)
    ]
    a = run_policy(_alerts(rows), CFG, "score", np.random.default_rng(7))
    b = run_policy(_alerts(rows), CFG, "score", np.random.default_rng(7))
    a.pop("backlog_curve")
    b.pop("backlog_curve")
    assert a == b


def test_business_hours_clock_skips_weekend() -> None:
    analysts = [Analyst("A", [0, 1, 2, 3, 4], 9.0, 8.0)]
    # Friday 16:00 -> Monday 10:00: 1h Friday (16-17) + 1h Monday (9-10)
    h = business_hours_between(
        datetime(2026, 6, 5, 16, 0), datetime(2026, 6, 8, 10, 0), analysts
    )
    assert abs(h - 2.0) < 1e-6


def test_business_hours_clock_uses_union_of_overlapping_shifts() -> None:
    analysts = [
        Analyst("A", [0], 9.0, 8.0),
        Analyst("B", [0], 13.0, 5.0),
    ]
    hours = business_hours_between(
        datetime(2026, 6, 1, 9), datetime(2026, 6, 1, 18), analysts
    )
    assert hours == 9.0


def test_service_crossing_shift_end_carries_only_remaining_work() -> None:
    analyst = Analyst("A", [0, 1], 9.0, 8.0)
    end = finish_service(
        analyst,
        datetime(2026, 6, 1, 16, 55),
        timedelta(minutes=10),
    )
    assert end == datetime(2026, 6, 2, 9, 5)


def test_sampled_lognormal_arithmetic_mean_matches_config() -> None:
    draws = sample_service_minutes(np.random.default_rng(416), 7.0, 0.6, size=200_000)
    assert abs(float(np.mean(draws)) - 7.0) < 0.05
