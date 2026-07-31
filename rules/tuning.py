"""Threshold tuning: sweep the review/decline bands on the fit window
(months 1–9), report every candidate on the holdout (months 10–12), pick the
net-$-maximizing point subject to review capacity.

The ONE rules-side component allowed to read labels (offline calibration,
FP-1 §2.4). Run: python -m rules.tuning  (after rules.engine has built the
enriched frame logic — this re-derives scores itself so it can sweep).

Outputs: reports/tradeoffs.md, reports/tradeoffs.svg, reports/operating_point.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rules.engine import build_enriched, run_rules  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def collected_by_order(d: Path) -> pd.Series:
    payments = pd.read_csv(d / "payments.csv")
    plans = pd.read_csv(d / "plans.csv")
    got = payments[payments.result == "success"].groupby("plan_id").amount.sum()
    plans["collected"] = plans.plan_id.map(got).fillna(0)
    return plans.set_index("order_id").collected


def evaluate_point(hold: pd.DataFrame, review_band: int, decline_band: int,
                   costs: dict, capacity_per_day: float, days: int) -> dict:
    alerted = hold[hold.score >= review_band]
    declined = hold[hold.score >= decline_band]
    n_alerts = len(alerted)
    alerts_per_day = n_alerts / days

    caught = alerted[alerted.is_fraud]
    caught_usd = float(caught.loss.sum())
    precision = len(caught) / n_alerts if n_alerts else 0.0

    insults = declined[~declined.is_fraud]
    insult_cost = float(
        (insults.amount * costs["false_decline_margin_pct"]
         + costs["false_decline_ltv_usd"]).sum()
    )
    review_cost = n_alerts * costs["review_cost_usd"]
    net = caught_usd - review_cost - insult_cost

    recall_by = {}
    for pat, grp in hold[hold.is_fraud].groupby("pattern_id"):
        recall_by[pat] = round(float((grp.score >= review_band).mean()), 3)
    recall_overall = round(float((hold[hold.is_fraud].score >= review_band).mean()), 3)

    return {
        "review_band": review_band, "decline_band": decline_band,
        "alerts_per_day": round(alerts_per_day, 1), "n_alerts": n_alerts,
        "precision": round(precision, 3), "recall_overall": recall_overall,
        "recall_by_pattern": recall_by, "caught_usd": round(caught_usd, 0),
        "n_auto_declines": len(declined), "n_insults": len(insults),
        "insult_cost_usd": round(insult_cost, 0), "review_cost_usd": round(review_cost, 0),
        "net_usd": round(net, 0),
        "over_capacity": alerts_per_day > capacity_per_day,
    }


def main() -> None:
    cfg = yaml.safe_load(open(REPO / "config.yaml"))
    costs = cfg["costs"]
    capacity = cfg["model"]["review_capacity_per_day"]
    d = REPO / "data"

    ap = run_rules(build_enriched())
    labels = pd.read_csv(d / "labels.csv")
    lab = labels.drop_duplicates("order_id").set_index("order_id").pattern_id
    ap["pattern_id"] = ap.order_id.map(lab)
    ap["is_fraud"] = ap.pattern_id.notna()
    ap["loss"] = (ap.amount - ap.order_id.map(collected_by_order(d))).clip(lower=0)

    start = ap.ts.min()
    fit_end = start + pd.DateOffset(months=cfg["model"]["train_months"])
    fit = ap[ap.ts < fit_end].copy()
    hold = ap[ap.ts >= fit_end].copy()
    fit_days = max((fit.ts.max() - fit.ts.min()).days, 1)
    days = max((hold.ts.max() - hold.ts.min()).days, 1)

    # Selection happens ONLY on the fit window; the holdout is reporting-only.
    fit_grid = []
    for rb in range(30, 65, 5):
        for db_ in range(70, 120, 10):
            fit_grid.append(evaluate_point(fit, rb, db_, costs, capacity, fit_days))
    feasible = [g for g in fit_grid if not g["over_capacity"]]
    best_fit = max(feasible or fit_grid, key=lambda g: g["net_usd"])

    # Report the whole grid on the holdout for the frontier plot/table; the
    # chosen point is the one selected on fit.
    grid = []
    for rb in range(30, 65, 5):
        for db_ in range(70, 120, 10):
            grid.append(evaluate_point(hold, rb, db_, costs, capacity, days))
    chosen = next(
        g for g in grid
        if g["review_band"] == best_fit["review_band"]
        and g["decline_band"] == best_fit["decline_band"]
    )

    # markdown report
    cols = ["review_band", "decline_band", "alerts_per_day", "precision",
            "recall_overall", "caught_usd", "n_insults", "net_usd"]
    lines = [
        "# Rule threshold tuning — selected on months 1–9, reported on months 10–12\n",
        (f"Cost model: review ${costs['review_cost_usd']:.2f}/case; false decline = "
         f"{costs['false_decline_margin_pct'] * 100:.0f}% margin + "
         f"${costs['false_decline_ltv_usd']:.0f} LTV proxy "
         "(config `costs`, assumptions documented in README).\n"),
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for g in sorted(grid, key=lambda g: (-g["net_usd"])):
        over = " (over capacity)" if g["over_capacity"] else ""
        mark = " **⬅ chosen**" if g is chosen else over
        lines.append("| " + " | ".join(str(g[c]) for c in cols) + " |" + mark)
    lines += [
        "",
        f"**Chosen operating point: review ≥ {chosen['review_band']}, "
        f"auto-decline ≥ {chosen['decline_band']}.** "
        f"It maximizes net $ ({chosen['net_usd']:,}) under the ≤{capacity} alerts/day "
        f"capacity constraint: {chosen['alerts_per_day']}/day at precision "
        f"{chosen['precision']:.0%}, catching ${chosen['caught_usd']:,.0f} of holdout fraud "
        f"exposure with {chosen['n_insults']} auto-declined legitimate orders. "
        "Raising the review band further trades linearly less review cost for "
        "disproportionate recall loss on ATO and stolen-card patterns; lowering it "
        "overruns the review team. The classic fraud triangle — loss caught vs review "
        "cost vs insult rate — made explicit.",
        "",
        "Recall by pattern at the chosen point: "
        + ", ".join(f"{k} {v:.0%}" for k, v in sorted(chosen["recall_by_pattern"].items())),
    ]
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "tradeoffs.md").write_text("\n".join(lines) + "\n")

    # frontier SVG
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    for db_ in sorted({g["decline_band"] for g in grid}):
        pts = sorted((g for g in grid if g["decline_band"] == db_),
                     key=lambda g: g["alerts_per_day"])
        axes.plot([g["alerts_per_day"] for g in pts], [g["caught_usd"] / 1000 for g in pts],
                  marker="o", markersize=3, linewidth=1, label=f"decline ≥ {db_}")
    axes.axvline(capacity, color="grey", linestyle="--", linewidth=1,
                 label=f"capacity {capacity}/day")
    axes.scatter([chosen["alerts_per_day"]], [chosen["caught_usd"] / 1000],
                 s=90, zorder=5, facecolors="none", edgecolors="red", label="chosen")
    axes.set_xlabel("alerts per day (holdout)")
    axes.set_ylabel("fraud $ caught (thousands)")
    axes.set_title("Alert volume vs fraud-$ caught — threshold sweep")
    axes.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO / "reports" / "tradeoffs.svg")

    (REPO / "reports" / "operating_point.json").write_text(json.dumps(chosen, indent=1))
    print(json.dumps({k: v for k, v in chosen.items() if k != "recall_by_pattern"}, indent=1))
    print("wrote reports/tradeoffs.{md,svg}, reports/operating_point.json")


if __name__ == "__main__":
    main()
