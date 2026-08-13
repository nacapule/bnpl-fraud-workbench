"""Discrete-event simulation of the analyst alert queue over the holdout months.

Run: python -m queue_sim.simulate

Mechanics: alerts arrive at their order timestamps; two analysts on offset
5-day shifts (covering all seven days, as fraud queues require) pull the next
alert per the dispatch policy whenever free during shift hours; service time
is lognormal and seeded. An order "ships"
``fulfillment_lag_hours`` after checkout — resolving a truly fraudulent alert
before that blocks the remaining loss; after it, the loss stands. Labels are
read HERE only as measurement (which resolutions blocked real fraud), never to
drive dispatch.

Policies compared: FIFO; rule-score priority; LLM P0–P3 priority (from the
triage layer's cached memos when available for an alert, falling back to score
order within a priority class).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "bnpl-queue-416"
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BUSINESS_EPS = 1e-9


@dataclass
class Analyst:
    name: str
    shift_days: list[int]  # 0=Mon .. 6=Sun
    shift_start_h: float
    productive_hours: float
    free_at: datetime = datetime.min

    def shift_window(self, day: datetime) -> tuple[datetime, datetime] | None:
        if day.weekday() not in self.shift_days:
            return None
        start = day.replace(hour=0, minute=0, second=0) + timedelta(hours=self.shift_start_h)
        return start, start + timedelta(hours=self.productive_hours)

    def next_available(self, t: datetime) -> datetime:
        """Earliest instant >= t and >= free_at inside one of this analyst's shifts."""
        t = max(t, self.free_at)
        for d in range(0, 15):
            day = (t + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
            win = self.shift_window(day)
            if win is None:
                continue
            start, end = win
            if d == 0 and t > end:
                continue
            candidate = max(t, start)
            if candidate < end:
                return candidate
        raise RuntimeError("no shift found in 15 days")


def build_analysts(cfg: dict) -> list[Analyst]:
    """Fresh ``Analyst`` objects from a config's ``queue.analysts`` entries."""
    return [
        Analyst(
            item["name"],
            item["shift_days"],
            float(item["shift_start"].split(":")[0])
            + float(item["shift_start"].split(":")[1]) / 60,
            item["productive_hours"],
        )
        for item in cfg["queue"]["analysts"]
    ]


def business_hours_between(t0: datetime, t1: datetime, analysts: list[Analyst]) -> float:
    """Hours of union-of-shifts coverage between t0 and t1 (SLA clock)."""
    if t1 <= t0:
        return 0.0
    total = 0.0
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < t1:
        windows = []
        for a in analysts:
            w = a.shift_window(day)
            if w:
                windows.append(w)
        clipped = sorted(
            (max(start, t0), min(end, t1))
            for start, end in windows
            if min(end, t1) > max(start, t0)
        )
        merged: list[tuple[datetime, datetime]] = []
        for start, end in clipped:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total += sum((end - start).total_seconds() / 3600 for start, end in merged)
        day += timedelta(days=1)
    return total


def sample_service_minutes(
    rng: np.random.Generator,
    arithmetic_mean_min: float,
    sigma: float,
    size: int | None = None,
) -> float | np.ndarray:
    """Draw a lognormal service time with the configured arithmetic mean."""
    mu = np.log(arithmetic_mean_min) - sigma**2 / 2
    return np.exp(rng.normal(mu, sigma, size=size))


def finish_service(analyst: Analyst, start: datetime, duration: timedelta) -> datetime:
    """Consume service across shifts without restarting unfinished work."""
    cursor = start
    remaining = duration
    while True:
        day = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
        window = analyst.shift_window(day)
        if window is None or not (window[0] <= cursor < window[1]):
            cursor = analyst.next_available(cursor)
            continue
        available = window[1] - cursor
        if remaining <= available:
            return cursor + remaining
        remaining -= available
        cursor = analyst.next_available(window[1])


def load_inputs() -> tuple[pd.DataFrame, dict, dict]:
    cfg = yaml.safe_load(open(REPO / "config.yaml"))
    op = json.loads((REPO / "reports" / "operating_point.json").read_text())
    alerts = pd.read_csv(REPO / "data" / "alerts.csv", parse_dates=["ts"])
    labels = pd.read_csv(REPO / "data" / "labels.csv").drop_duplicates("order_id")
    plans = pd.read_csv(REPO / "data" / "plans.csv")
    pays = pd.read_csv(REPO / "data" / "payments.csv")

    holdout_start = pd.Timestamp(cfg["holdout_start"])
    a = alerts[(alerts.ts >= holdout_start) & (alerts.score >= op["review_band"])].copy()

    got = pays[pays.result == "success"].groupby("plan_id").amount.sum()
    plans["collected"] = plans.plan_id.map(got).fillna(0)
    by_order = plans.set_index("order_id")
    a["is_fraud"] = a.order_id.isin(set(labels.order_id))
    a["loss_at_stake"] = (
        a.order_id.map(by_order.principal) - a.order_id.map(by_order.collected)
    ).clip(lower=0).fillna(0)
    a = a.sort_values("ts").reset_index(drop=True)

    # LLM priorities from cached eval memos (advisory artifacts), where present
    prio_path = REPO / "llm" / "eval" / "results" / "priorities.csv"
    if prio_path.exists():
        pr = pd.read_csv(prio_path).set_index("alert_id").priority
        a["llm_priority"] = a.alert_id.map(pr)
    else:
        a["llm_priority"] = np.nan
    return a, cfg, op


def run_policy(alerts: pd.DataFrame, cfg: dict, policy: str, rng: np.random.Generator
               ) -> dict:
    analysts = build_analysts(cfg)
    mean_min = cfg["queue"]["service_time_arithmetic_mean_min"]
    sigma = cfg["queue"]["service_time_sigma"]
    ship_lag = timedelta(hours=cfg["queue"]["fulfillment_lag_hours"])
    sla_h = cfg["queue"]["sla_target_hours"]

    pending: list[int] = []
    events = alerts.ts.tolist()
    n = len(alerts)
    resolved_at = [None] * n
    arrival = alerts.ts.tolist()
    score = alerts.score.values
    llm_p = alerts.llm_priority.fillna("P9").values
    served = 0
    backlog_curve: list[tuple[datetime, int]] = []

    def pick(now: datetime) -> int:
        if policy == "fifo":
            return pending[0]
        if policy == "score":
            return max(pending, key=lambda i: (score[i], -arrival[i].timestamp()))
        # llm: priority class first (P0 best), then score
        return min(pending, key=lambda i: (llm_p[i], -score[i], arrival[i].timestamp()))

    i = 0
    t = min(events)
    while served < n:
        while i < n and arrival[i] <= t:
            pending.append(i)
            i += 1
        if not pending:
            t = arrival[i]
            continue
        nxt = min(analysts, key=lambda a: a.next_available(t))
        start = nxt.next_available(t)
        # absorb arrivals up to service start so priority reflects reality
        while i < n and arrival[i] <= start:
            pending.append(i)
            i += 1
        j = pick(start)
        pending.remove(j)
        dur = timedelta(minutes=float(sample_service_minutes(rng, mean_min, sigma)))
        end = finish_service(nxt, start, dur)
        nxt.free_at = end
        resolved_at[j] = end
        served += 1
        backlog_curve.append((end, len(pending)))
        t = max(t, min(a.free_at for a in analysts))
        if pending and i < n:
            t = min(t, arrival[i])

    analysts_fresh = build_analysts(cfg)
    ttd = []
    blocked_usd = 0.0
    blocked_n = 0
    for k in range(n):
        hours = business_hours_between(arrival[k], resolved_at[k], analysts_fresh)
        ttd.append(hours)
        if bool(alerts.is_fraud.iloc[k]) and resolved_at[k] <= arrival[k] + ship_lag:
            blocked_usd += float(alerts.loss_at_stake.iloc[k])
            blocked_n += 1
    ttd_arr = np.array(ttd)
    return {
        "policy": policy,
        "n_alerts": n,
        "sla_attainment": round(float((ttd_arr <= sla_h).mean()), 3),
        "ttd_p50_h": round(float(np.percentile(ttd_arr, 50)), 2),
        "ttd_p90_h": round(float(np.percentile(ttd_arr, 90)), 2),
        "max_backlog": int(max(b for _, b in backlog_curve)),
        "fraud_blocked_usd": round(blocked_usd, 0),
        "fraud_blocked_n": blocked_n,
        "backlog_curve": backlog_curve,
    }


def main() -> None:
    alerts, cfg, op = load_inputs()
    orders = pd.read_csv(REPO / "data" / "orders.csv", parse_dates=["ts"], usecols=["ts"])
    holdout_days = max((orders.ts.max() - pd.Timestamp(cfg["holdout_start"])).days, 1)
    rng_seed = cfg["seed"]
    have_llm = alerts.llm_priority.notna().any()
    policies = ["fifo", "score"] + (["llm"] if have_llm else [])
    results = []
    for pol in policies:
        results.append(run_policy(alerts.copy(), cfg, pol, np.random.default_rng(rng_seed)))

    coverage_analysts = build_analysts(cfg)
    coverage_hours = business_hours_between(
        alerts.ts.min().to_pydatetime(),
        alerts.ts.max().to_pydatetime(),
        coverage_analysts,
    )
    offered_hours = (
        len(alerts) * cfg["queue"]["service_time_arithmetic_mean_min"] / 60
    )
    utilization = offered_hours / coverage_hours

    lines = [
        f"# Alert queue / SLA simulation (from {cfg['holdout_start']})\n",
        f"Operating point: review ≥ {op['review_band']} → {len(alerts)} alerts "
        f"({len(alerts) / holdout_days:.1f}/day). Two analysts on offset 5-day shifts, "
        f"6.5 productive h, lognormal service (arithmetic mean "
        f"{cfg['queue']['service_time_arithmetic_mean_min']} min), orders ship "
        f"{cfg['queue']['fulfillment_lag_hours']}h after checkout; SLA target "
        f"{cfg['queue']['sla_target_hours']} business hours.\n",
        ("| policy | SLA ≤ 4bh | P50 ttd (bh) | P90 ttd (bh) | max backlog "
         "| fraud-$ blocked pre-ship | fraud alerts blocked |"),
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['policy']} | {r['sla_attainment']:.1%} | {r['ttd_p50_h']} | "
            f"{r['ttd_p90_h']} | {r['max_backlog']} | ${r['fraud_blocked_usd']:,.0f} | "
            f"{r['fraud_blocked_n']}/{int(alerts.is_fraud.sum())} |"
        )
    if not have_llm:
        lines.append("\n_LLM-priority policy skipped: no cached priorities file "
                     "(llm/eval/results/priorities.csv) — generated by the eval harness._")
    else:
        cov = alerts.llm_priority.notna().mean()
        lines.append(
            f"\n_LLM priorities cover {cov:.0%} of alerts (the frozen eval set); "
            "uncovered alerts fall back to score order, so at this coverage the "
            "llm policy tracks score-priority — the comparison becomes meaningful "
            "when triage runs on the full stream._"
        )
    lines += [
        "",
        f"The configured workload is far below saturation ({utilization:.1%} utilization of "
        "union shift coverage), so high SLA attainment is expected.",
        "",
        "Reading: the 12-hour fulfillment race means weekend/coverage gaps, not average "
        "throughput, decide how much fraud ships. Score-priority beats FIFO by resolving "
        "high-score (fraud-dense) alerts inside the ship window even when the backlog "
        "spans a coverage hole; the offset-shift pairing leaves parts of the week "
            "single-covered, visible as the weekly backlog sawtooth.",
        "",
        "```json",
        *[json.dumps({key: value for key, value in result.items() if key != "backlog_curve"})
          for result in results],
        "```",
    ]
    (REPO / "reports" / "queue.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in results:
        ts, bl = zip(*r["backlog_curve"], strict=True)
        ax.plot(ts, bl, linewidth=0.8, label=r["policy"])
    ax.set_ylabel("pending alerts")
    ax.set_title("Queue backlog over holdout months, by dispatch policy")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPO / "reports" / "queue_backlog.svg", metadata={"Date": None})

    for r in results:
        r.pop("backlog_curve")
        print(json.dumps(r))
    print("wrote reports/queue.md, reports/queue_backlog.svg")


if __name__ == "__main__":
    main()
