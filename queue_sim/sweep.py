"""Stress the alert queue over arrival volume and headcount.

Run: python -m queue_sim.sweep

``queue_sim.simulate`` answers one question — does the configured roster meet
its SLA on the observed holdout stream — and the answer was never in doubt:
that point runs at roughly a third of union shift coverage. This module sweeps
the same discrete-event machinery over arrival multipliers 1x/2x/4x/8x crossed
with one to four analysts, which is where the SLA target and the fraud-dollar
curve actually have break points.

Volume is scaled by replicating the holdout alert stream: at multiplier k every
alert appears k times at its own timestamp, so the weekly and diurnal shape of
arrivals is preserved and only intensity changes. Rosters extend the configured
shift pattern (see ``roster``). Output: reports/queue_frontier.md.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta

import numpy as np
import pandas as pd

from queue_sim.simulate import (
    REPO,
    build_analysts,
    business_hours_between,
    load_inputs,
    run_policy,
)

MULTIPLIERS = (1, 2, 4, 8)
HEADCOUNTS = (1, 2, 3, 4)
POLICY = "score"
ORDINALS = {2: "2nd", 3: "3rd", 4: "4th"}


def roster(cfg: dict, n_analysts: int) -> list[dict]:
    """Extend the configured shift pattern to ``n_analysts`` analysts.

    The configured analysts are reused verbatim, so a two-analyst roster is
    exactly ``config.yaml``. Each analyst beyond them continues the same
    construction: a five-day block starting two weekdays after the previous
    one, with the start hour alternating between the configured start times.
    """
    configured = cfg["queue"]["analysts"]
    if n_analysts <= len(configured):
        return [dict(item) for item in configured[:n_analysts]]
    first_day = configured[0]["shift_days"][0]
    span = len(configured[0]["shift_days"])
    day_step = (
        (configured[1]["shift_days"][0] - first_day) % 7 if len(configured) > 1 else 2
    )
    starts = [item["shift_start"] for item in configured]
    hours = [item["productive_hours"] for item in configured]
    extended = [dict(item) for item in configured]
    for index in range(len(configured), n_analysts):
        extended.append(
            {
                "name": f"A{index + 1}",
                "shift_days": [
                    (first_day + day_step * index + offset) % 7 for offset in range(span)
                ],
                "shift_start": starts[index % len(starts)],
                "productive_hours": hours[index % len(hours)],
            }
        )
    return extended


def roster_config(cfg: dict, n_analysts: int) -> dict:
    """``cfg`` with its analyst roster replaced, leaving everything else alone."""
    return {**cfg, "queue": {**cfg["queue"], "analysts": roster(cfg, n_analysts)}}


def replicate(alerts: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """Repeat every alert ``multiplier`` times at its own arrival timestamp."""
    if multiplier == 1:
        return alerts.copy()
    repeated = pd.concat([alerts] * multiplier, ignore_index=True)
    return repeated.sort_values("ts", kind="stable").reset_index(drop=True)


def run_cell(
    alerts: pd.DataFrame,
    cfg: dict,
    multiplier: int,
    n_analysts: int,
) -> dict:
    """One (volume, headcount) cell of the frontier."""
    stream = replicate(alerts, multiplier)
    cell_cfg = roster_config(cfg, n_analysts)
    result = run_policy(stream, cell_cfg, POLICY, np.random.default_rng(cfg["seed"]))
    result.pop("backlog_curve")

    coverage = business_hours_between(
        stream.ts.min().to_pydatetime(),
        stream.ts.max().to_pydatetime(),
        build_analysts(cell_cfg),
    )
    offered = len(stream) * cfg["queue"]["service_time_arithmetic_mean_min"] / 60
    at_risk = float(stream.loc[stream.is_fraud, "loss_at_stake"].sum())
    return {
        "arrival_multiplier": multiplier,
        "analysts": n_analysts,
        "n_alerts": result["n_alerts"],
        "utilization": round(offered / coverage, 3),
        "sla_attainment": result["sla_attainment"],
        "sla_target_met": result["sla_attainment"]
        >= cfg["queue"]["sla_target_attainment"],
        "ttd_p50_h": result["ttd_p50_h"],
        "ttd_p90_h": result["ttd_p90_h"],
        "max_backlog": result["max_backlog"],
        "fraud_blocked_usd": result["fraud_blocked_usd"],
        "fraud_at_risk_usd": round(at_risk, 0),
        "fraud_blocked_share": round(result["fraud_blocked_usd"] / at_risk, 3),
        "fraud_blocked_n": result["fraud_blocked_n"],
        "fraud_alerts": int(stream.is_fraud.sum()),
    }


def sweep(alerts: pd.DataFrame, cfg: dict) -> list[dict]:
    return [
        run_cell(alerts, cfg, multiplier, n_analysts)
        for multiplier in MULTIPLIERS
        for n_analysts in HEADCOUNTS
    ]


def _grid(cells: list[dict]) -> dict[tuple[int, int], dict]:
    return {(cell["arrival_multiplier"], cell["analysts"]): cell for cell in cells}


def _headcount_label(n_analysts: int) -> str:
    return f"{n_analysts} analyst" + ("s" if n_analysts > 1 else "")


def coverage_ceiling(alerts: pd.DataFrame, cfg: dict, n_analysts: int) -> dict:
    """Fraud alerts no roster of this shape can reach inside the ship window.

    An alert is unreachable when no analyst comes on duty within
    ``fulfillment_lag_hours`` of its arrival: the order ships before anyone is
    at a desk, whatever the headcount.
    """
    analysts = build_analysts(roster_config(cfg, n_analysts))
    lag = timedelta(hours=cfg["queue"]["fulfillment_lag_hours"])
    fraud = alerts[alerts.is_fraud]
    reachable = fraud.ts.map(
        lambda t: min(a.next_available(t.to_pydatetime()) for a in analysts)
        <= t.to_pydatetime() + lag
    )
    return {
        "analysts": n_analysts,
        "unreachable_alerts": int((~reachable).sum()),
        "unreachable_usd": round(float(fraud.loc[~reachable, "loss_at_stake"].sum()), 0),
        "fraud_at_risk_usd": round(float(fraud.loss_at_stake.sum()), 0),
    }


def _matrix(
    cells: list[dict],
    header: str,
    render: Callable[[dict], str],
) -> list[str]:
    grid = _grid(cells)
    lines = [
        f"| {header} | " + " | ".join(_headcount_label(n) for n in HEADCOUNTS) + " |",
        "|---" * (len(HEADCOUNTS) + 1) + "|",
    ]
    for multiplier in MULTIPLIERS:
        rate = grid[(multiplier, HEADCOUNTS[0])]["n_alerts"]
        row = [f"×{multiplier} ({rate:,} alerts)"]
        row += [render(grid[(multiplier, n)]) for n in HEADCOUNTS]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _minimum_headcount(cells: list[dict], multiplier: int) -> int | None:
    grid = _grid(cells)
    for n_analysts in HEADCOUNTS:
        if grid[(multiplier, n_analysts)]["sla_target_met"]:
            return n_analysts
    return None


def render_report(
    cells: list[dict],
    cfg: dict,
    alerts_per_day: float,
    ceiling: dict,
) -> str:
    grid = _grid(cells)
    target = cfg["queue"]["sla_target_attainment"]
    sla_hours = cfg["queue"]["sla_target_hours"]
    base = grid[(1, len(cfg["queue"]["analysts"]))]

    holds = {
        multiplier: _minimum_headcount(cells, multiplier) for multiplier in MULTIPLIERS
    }
    holds_text = ", ".join(
        f"×{multiplier} needs "
        + (
            _headcount_label(count)
            if count
            else f"more than {_headcount_label(HEADCOUNTS[-1])}"
        )
        for multiplier, count in holds.items()
    )

    lines = [
        "# Queue staffing frontier",
        "",
        (
            f"The single-point run in `reports/queue.md` puts the configured roster against the "
            f"observed holdout stream ({alerts_per_day:.1f} alerts/day) at "
            f"{base['utilization']:.1%} utilization of union shift coverage. Its "
            f"{base['sla_attainment']:.1%} SLA attainment could not have come out otherwise, so "
            "it is not a finding. The frontier below is."
        ),
        "",
        (
            "Volume is scaled by replicating the holdout alert stream: at ×k every alert appears "
            "k times at its own timestamp, so the weekly and diurnal arrival shape is preserved "
            "and only intensity changes. Rosters reuse the two configured analysts verbatim and "
            "extend the same pattern — each further analyst covers five consecutive days "
            "starting two weekdays later, alternating between the two configured start hours. "
            "Dispatch is score-priority in every cell, service times are drawn from the same "
            f"seeded stream (seed {cfg['seed']}), and the ×1 / "
            f"{len(cfg['queue']['analysts'])}-analyst cell reproduces the score-priority row of "
            "`reports/queue.md` exactly."
        ),
        "",
        f"## SLA attainment (decision within {sla_hours} business hours)",
        "",
    ]
    lines += _matrix(
        cells,
        "arrivals",
        lambda cell: f"{cell['sla_attainment']:.1%}"
        + (" ✓" if cell["sla_target_met"] else ""),
    )
    lines += [
        "",
        f"✓ marks cells at or above the configured {target:.0%} target: {holds_text}.",
        "",
        (
            "Attainment runs on the union-of-shifts clock, so a larger roster widens the clock "
            "as well as the capacity. Where the queue is already empty that can cost a tenth of "
            f"a point: the ×1 row slips from {grid[(1, 2)]['sla_attainment']:.1%} at two "
            f"analysts to {grid[(1, 3)]['sla_attainment']:.1%} at three while median "
            f"time-to-decision falls from {grid[(1, 2)]['ttd_p50_h']} to "
            f"{grid[(1, 3)]['ttd_p50_h']} business hours."
        ),
        "",
        "## Fraud dollars blocked before fulfillment",
        "",
    ]
    lines += _matrix(
        cells,
        "arrivals",
        lambda cell: (
            f"${cell['fraud_blocked_usd']:,.0f} ({cell['fraud_blocked_share']:.0%})"
        ),
    )
    lines += [
        "",
        (
            "Percentages are the share of fraud dollars at stake in that stream. Replication "
            "multiplies the exposure as well as the workload, so dollar figures compare down a "
            "column, not across a row; the share compares in both directions."
        ),
        "",
        "## Marginal fraud dollars per added analyst",
        "",
        "| arrivals | "
        + " | ".join(f"{ORDINALS[n]} analyst" for n in HEADCOUNTS[1:])
        + " |",
        "|---" * len(HEADCOUNTS) + "|",
    ]
    for multiplier in MULTIPLIERS:
        row = [f"×{multiplier}"]
        for index in range(1, len(HEADCOUNTS)):
            previous = grid[(multiplier, HEADCOUNTS[index - 1])]["fraud_blocked_usd"]
            current = grid[(multiplier, HEADCOUNTS[index])]["fraud_blocked_usd"]
            row.append(f"+${current - previous:,.0f}")
        lines.append("| " + " | ".join(row) + " |")

    def marginal(multiplier: int, index: int) -> float:
        return (
            grid[(multiplier, HEADCOUNTS[index])]["fraud_blocked_usd"]
            - grid[(multiplier, HEADCOUNTS[index - 1])]["fraud_blocked_usd"]
        )

    flat = [m for m in MULTIPLIERS if _minimum_headcount(cells, m) is not None]
    bound = [m for m in MULTIPLIERS if _minimum_headcount(cells, m) is None]
    flat_text = "; ".join(
        f"at ×{m} the third and fourth add ${marginal(m, 2):,.0f} and ${marginal(m, 3):,.0f} "
        f"against the second analyst's ${marginal(m, 1):,.0f}"
        for m in flat
    )
    bound_text = ", ".join(
        f"×{m} is still at {grid[(m, HEADCOUNTS[-1])]['utilization']:.2f} utilization with a "
        f"fourth analyst worth ${marginal(m, 3):,.0f}"
        for m in bound
    )
    top = grid[(1, HEADCOUNTS[-1])]
    unblocked = top["fraud_at_risk_usd"] - top["fraud_blocked_usd"]

    lines += [
        "",
        (
            "Fraud dollars flatten only where the queue is not capacity-bound: "
            + flat_text
            + ". Where the SLA target cannot be held they have not flattened at all — "
            + bound_text
            + " — so for those volumes the sweep's answer is that this roster is too small to "
            "price, not that another analyst is poor value."
        ),
        "",
        (
            "Headcount also runs into a ceiling that has nothing to do with capacity. At ×1 "
            f"with {_headcount_label(HEADCOUNTS[-1])} the queue is {top['utilization']:.0%} "
            f"utilized and {top['sla_attainment']:.1%} of alerts are decided inside the SLA, "
            f"yet ${unblocked:,.0f} of fraud exposure "
            f"({unblocked / top['fraud_at_risk_usd']:.0%}) still ships. Of those, "
            f"{ceiling['unreachable_alerts']} alerts carrying "
            f"${ceiling['unreachable_usd']:,.0f} arrive after the last shift start of the day: "
            "no analyst comes on duty inside their 12-hour fulfillment window, so no roster of "
            "this shape resolves them in time whatever its size. That residue is a shift-design "
            "question, not a staffing one."
        ),
        "",
        (
            "The two surfaces therefore answer different questions. SLA attainment is an "
            "average over every alert and degrades smoothly with load, so it prices the "
            "analyst experience; blocked fraud dollars depend on where a small number of "
            "high-exposure alerts fall relative to coverage boundaries, so they price the "
            "loss. Staffing to one is not staffing to the other."
        ),
        "",
        "```json",
        *[json.dumps(cell, sort_keys=True) for cell in cells],
        json.dumps(ceiling, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    alerts, cfg, _ = load_inputs()
    orders = pd.read_csv(REPO / "data" / "orders.csv", parse_dates=["ts"], usecols=["ts"])
    holdout_days = max((orders.ts.max() - pd.Timestamp(cfg["holdout_start"])).days, 1)
    cells = sweep(alerts, cfg)
    ceiling = coverage_ceiling(alerts, cfg, HEADCOUNTS[-1])
    destination = REPO / "reports" / "queue_frontier.md"
    destination.write_text(
        render_report(cells, cfg, len(alerts) / holdout_days, ceiling)
    )
    print(f"wrote {destination.relative_to(REPO)}")


if __name__ == "__main__":
    main()
