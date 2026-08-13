"""Generate the README results table from committed result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def _json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    inside = False
    for line in path.read_text().splitlines():
        if line.strip() == "```json":
            inside = True
            continue
        if inside and line.strip() == "```":
            inside = False
            continue
        if inside and line.strip():
            records.append(json.loads(line))
    return records


def _result(prompt: str, model: str) -> dict[str, Any]:
    path = REPO / "llm" / "eval" / "results" / f"{prompt}__{model}.json"
    return json.loads(path.read_text())


def render_results() -> str:
    operating = json.loads((REPO / "reports" / "operating_point.json").read_text())
    model = _json_lines(REPO / "reports" / "model.md")[-1]
    queue = {
        record["policy"]: record for record in _json_lines(REPO / "reports" / "queue.md")
    }
    v1 = _result("memo_v1", "claude-sonnet-5")
    v2 = _result("memo_v2", "claude-sonnet-5")
    luna = _result("memo_v2", "gpt-5.6-luna")
    terra = _result("memo_v2", "gpt-5.6-terra")
    luna_inclusive_accuracy = luna["decision_metrics_including_schema_failures"][
        "action_accuracy"
    ]

    orders = pd.read_csv(REPO / "data" / "orders.csv", usecols=["status"])
    users = sum(1 for _ in (REPO / "data" / "users.csv").open()) - 1
    fraud_orders = pd.read_csv(REPO / "data" / "labels.csv", usecols=["order_id"])[
        "order_id"
    ].nunique()
    approved = int(orders["status"].eq("approved").sum())
    best = model["models"][model["best_model"]]
    logistic = model["models"]["Logistic Regression"]
    never_pay_recall = model["recall_by_pattern"]["P-NEVERPAY"][model["best_model"]]

    rows = [
        "| layer | result (configured holdout starts 2026-04-01 unless noted) |",
        "|---|---|",
        (
            f"| simulator | {len(orders):,} orders / {users:,} users / 12 × 30-day months "
            f"(360 days); fraud base rate {fraud_orders / approved:.2%} of approved orders; "
            "generation ~2 min on a laptop, deterministic (seed 416) |"
        ),
        (
            f"| rules @ tuned bands ({operating['review_band']}/{operating['decline_band']}) | "
            f"{operating['alerts_per_day']:.1f} alerts/day · precision "
            f"{operating['precision']:.1%} · recall {operating['recall_overall']:.1%} · "
            f"${operating['caught_usd']:,.0f} fraud caught · {operating['n_insults']} false "
            f"auto-declines · net +${operating['net_usd']:,.0f} under the cost model |"
        ),
        (
            f"| ML ({model['best_model'].replace(' ', '')}) | PR-AUC {best['pr_auc']:.2f} vs "
            f"logistic {logistic['pr_auc']:.2f} · precision@capacity "
            f"{best['precision_at_capacity']:.1%} · never-pay {never_pay_recall:.0%}: caught by "
            "the model, missed by transaction-time rules — see CASE-04 |"
        ),
        (
            f"| queue sim | 2 analysts, offset 7-day-coverage shifts · SLA(≤4 business h) "
            f"{queue['score']['sla_attainment']:.1%} · score-priority blocks "
            f"${queue['score']['fraud_blocked_usd']:,.0f} pre-fulfillment vs "
            f"${queue['fifo']['fraud_blocked_usd']:,.0f} FIFO |"
        ),
        (
            "| triage (claude-sonnet-5, prompt v2) | action accuracy "
            f"{v2['action_accuracy']:.1%} · decline precision {v2['decline_precision']:.1%} / "
            f"recall {v2['decline_recall']:.1%} · non-verbatim concrete-token fields in "
            f"{v2['non_verbatim_claim_rate_memo']:.1%} of memos; unsupported after derived-list "
            f"classification {v2['unsupported_claim_rate_memo']:.1%} · consistency "
            f"{v2['consistency_action_agreement']:.0%} (N={v2['n_cases']}) |"
        ),
        (
            "| prompt iteration v1→v2 (same model/cases) | non-verbatim concrete-token fields "
            f"{v1['non_verbatim_claim_rate_memo']:.0%}→"
            f"{v2['non_verbatim_claim_rate_memo']:.0%}; unsupported after derived-list "
            f"classification {v1['unsupported_claim_rate_memo']:.1%}→"
            f"{v2['unsupported_claim_rate_memo']:.1%} · action accuracy "
            f"{v2['action_accuracy'] - v1['action_accuracy']:+.1%} |"
        ),
        (
            f"| cross-model arms | gpt-5.6-terra {terra['action_accuracy']:.1%} accuracy / "
            f"{terra['decline_recall']:.0%} decline recall (N={terra['n_cases']}, quota-cut) · "
            f"gpt-5.6-luna {luna['action_accuracy']:.1%} valid-output accuracy "
            f"(N={luna['n_cases']}; {luna_inclusive_accuracy:.1%} "
            "with its schema failure counted wrong) |"
        ),
    ]
    return "\n".join(rows)


def rewrite_readme(path: Path = README) -> None:
    text = path.read_text()
    before, remainder = text.split(START, 1)
    _, after = remainder.split(END, 1)
    replacement = f"{START}\n{render_results()}\n{END}"
    path.write_text(before + replacement + after)


def main() -> None:
    rewrite_readme()
    print("wrote README.md results manifest")


if __name__ == "__main__":
    main()
