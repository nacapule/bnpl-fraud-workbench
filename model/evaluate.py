"""Offline evaluation for the chronological fraud-model holdout.

Run after training with ``python -m model.evaluate``.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bnpl-matplotlib"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from model.features import FEATURE_COLUMNS, build_features, load_feature_frames  # noqa: E402
from model.train import ARTIFACT_DIR, MODEL_FILES, chronological_split, load_config  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "bnpl-model-416"
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO / "reports"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render the small result frames without an optional tabulate dependency."""
    display = frame.copy()
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]

    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _ranked_selection(order_ids: pd.Series, scores: np.ndarray, capacity: int) -> set[int]:
    ranking = pd.DataFrame({"order_id": order_ids.to_numpy(), "score": scores})
    ranking = ranking.sort_values(["score", "order_id"], ascending=[False, True], kind="stable")
    return set(ranking.head(min(capacity, len(ranking)))["order_id"].astype(int))


def _capacity_threshold(scores: np.ndarray, capacity: int) -> float:
    if not len(scores):
        return math.nan
    index = min(max(capacity, 1), len(scores)) - 1
    return float(np.sort(scores)[::-1][index])


def _calibration_table(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    order = np.argsort(scores, kind="stable")
    bins = np.empty(len(scores), dtype=np.int8)
    bins[order] = np.minimum(np.arange(len(scores)) * 10 // len(scores), 9)
    raw = pd.DataFrame({"bin": bins + 1, "predicted": scores, "observed": labels})
    table = raw.groupby("bin", as_index=False).agg(
        orders=("observed", "size"),
        min_score=("predicted", "min"),
        max_score=("predicted", "max"),
        mean_predicted=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )
    for column in ["min_score", "max_score", "mean_predicted", "observed_rate"]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    return table


def _exposure_table(data_dir: Path) -> pd.DataFrame:
    plans = pd.read_csv(data_dir / "plans.csv", usecols=["plan_id", "order_id", "principal"])
    payments = pd.read_csv(
        data_dir / "payments.csv",
        usecols=["plan_id", "amount", "result"],
    )
    collected = (
        payments[payments["result"].eq("success")]
        .groupby("plan_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "collected"})
    )
    exposure = plans.merge(collected, on="plan_id", how="left")
    exposure["collected"] = exposure["collected"].fillna(0.0)
    exposure["loss_exposure"] = (exposure["principal"] - exposure["collected"]).clip(lower=0)
    return exposure[["order_id", "principal", "collected", "loss_exposure"]]


def _cost_optimal(
    order_ids: pd.Series,
    scores: np.ndarray,
    labels: np.ndarray,
    exposure: pd.DataFrame,
    costs: dict,
) -> dict[str, float | int]:
    rows = pd.DataFrame(
        {"order_id": order_ids.to_numpy(), "score": scores, "label": labels.astype(int)}
    ).merge(exposure, on="order_id", how="left", validate="one_to_one")
    rows[["principal", "collected", "loss_exposure"]] = rows[
        ["principal", "collected", "loss_exposure"]
    ].fillna(0.0)
    review_cost = float(costs["review_cost_usd"])
    insult = (
        rows["principal"] * float(costs["false_decline_margin_pct"])
        + float(costs["false_decline_ltv_usd"])
    )
    rows["delta"] = np.where(
        rows["label"].eq(1),
        review_cost - rows["loss_exposure"],
        review_cost + insult,
    )
    rows["tp"] = rows["label"]
    rows["fp"] = 1 - rows["label"]
    rows["caught_dollars"] = rows["loss_exposure"] * rows["label"]
    rows = rows.sort_values("score", ascending=False, kind="stable")
    by_threshold = rows.groupby("score", sort=False, as_index=False).agg(
        delta=("delta", "sum"),
        alerts=("label", "size"),
        true_positives=("tp", "sum"),
        false_positives=("fp", "sum"),
        caught_dollars=("caught_dollars", "sum"),
    )
    for column in ["delta", "alerts", "true_positives", "false_positives", "caught_dollars"]:
        by_threshold[column] = by_threshold[column].cumsum()
    baseline = float(rows.loc[rows["label"].eq(1), "loss_exposure"].sum())
    losses = baseline + by_threshold["delta"]
    best_position = int(losses.to_numpy().argmin()) if len(losses) else -1
    if best_position < 0 or float(losses.iloc[best_position]) >= baseline:
        return {
            "threshold": float(np.nextafter(scores.max(), np.inf)),
            "alerts": 0,
            "true_positives": 0,
            "false_positives": 0,
            "fraud_dollars_caught": 0.0,
            "total_cost": baseline,
        }
    best = by_threshold.iloc[best_position]
    return {
        "threshold": float(best["score"]),
        "alerts": int(best["alerts"]),
        "true_positives": int(best["true_positives"]),
        "false_positives": int(best["false_positives"]),
        "fraud_dollars_caught": float(best["caught_dollars"]),
        "total_cost": float(losses.iloc[best_position]),
    }


def _caught_dollars(selected: set[int], labeled: set[int], exposure: pd.DataFrame) -> float:
    caught = selected & labeled
    return float(exposure.loc[exposure["order_id"].isin(caught), "loss_exposure"].sum())


def _hybrid_table(
    holdout: pd.DataFrame,
    best_scores: np.ndarray,
    capacity: int,
    exposure: pd.DataFrame,
    alerts_path: Path,
) -> tuple[pd.DataFrame, str]:
    labeled = set(holdout.loc[holdout["label"].eq(1), "order_id"].astype(int))
    model_ranking = pd.DataFrame(
        {"order_id": holdout["order_id"].to_numpy(), "score": best_scores}
    ).sort_values(["score", "order_id"], ascending=[False, True], kind="stable")
    model_selected = set(model_ranking.head(capacity)["order_id"].astype(int))
    rows = [
        {
            "Strategy": "Model top-k",
            "Reviewed": len(model_selected),
            "Fraud $ caught": f"${_caught_dollars(model_selected, labeled, exposure):,.2f}",
        }
    ]
    if not alerts_path.exists():
        note = "`data/alerts.csv` was absent, so rules-only and hybrid were skipped."
        return pd.DataFrame(rows), note

    alerts = pd.read_csv(alerts_path)
    alerts = alerts[alerts["order_id"].isin(set(holdout["order_id"]))]
    alerts = alerts.sort_values(["score", "order_id"], ascending=[False, True], kind="stable")
    alerts = alerts.drop_duplicates("order_id")
    rule_rank = alerts["order_id"].astype(int).tolist()
    rule_selected = set(rule_rank[:capacity])
    rows.insert(
        0,
        {
            "Strategy": "Rules alerts",
            "Reviewed": len(rule_selected),
            "Fraud $ caught": f"${_caught_dollars(rule_selected, labeled, exposure):,.2f}",
        },
    )

    half = capacity // 2
    hybrid = set(rule_rank[:half])
    model_quota = capacity - len(hybrid)
    for order_id in model_ranking["order_id"].astype(int):
        if order_id not in hybrid:
            hybrid.add(order_id)
            model_quota -= 1
            if model_quota == 0:
                break
    rows.append(
        {
            "Strategy": "Hybrid (half rules / half model)",
            "Reviewed": len(hybrid),
            "Fraud $ caught": f"${_caught_dollars(hybrid, labeled, exposure):,.2f}",
        }
    )
    note = (
        "Hybrid takes up to half of capacity from ranked rules alerts and fills the remainder "
        "from the model ranking without duplicate reviews."
    )
    return pd.DataFrame(rows), note


def _save_pr_curve(labels: np.ndarray, scores: dict[str, np.ndarray], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        axis.plot(recall, precision, linewidth=2, label=name)
    axis.axhline(labels.mean(), color="0.45", linestyle="--", label="Holdout base rate")
    axis.set(xlabel="Recall", ylabel="Precision", title="Holdout precision-recall curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)


def _save_pattern_recall(table: pd.DataFrame, destination: Path) -> None:
    models = [column for column in table.columns if column not in {"Pattern", "Fraud orders"}]
    positions = np.arange(len(table))
    width = 0.8 / max(len(models), 1)
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for index, model in enumerate(models):
        recall = table[model].str.rstrip("%").astype(float) / 100
        axis.bar(positions + (index - (len(models) - 1) / 2) * width, recall, width, label=model)
    axis.set_xticks(positions, table["Pattern"], rotation=25, ha="right")
    axis.set(xlabel="Fraud pattern", ylabel="Recall at review capacity", ylim=(0, 1.05))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    config = load_config()
    data_dir = REPO / "data"
    frames = load_feature_frames(data_dir)
    features = build_features(**frames)
    train, holdout = chronological_split(features, config)
    del train

    missing = [
        filename
        for filename in MODEL_FILES.values()
        if not (ARTIFACT_DIR / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing model artifacts {missing}; run `.venv/bin/python -m model.train` first"
        )
    models = {name: joblib.load(ARTIFACT_DIR / filename) for name, filename in MODEL_FILES.items()}
    x_holdout = holdout[FEATURE_COLUMNS]
    labels = holdout["label"].to_numpy(dtype=int)
    scores = {name: model.predict_proba(x_holdout)[:, 1] for name, model in models.items()}

    holdout_days = 30 * (
        int(config["simulator"]["months"]) - int(config["model"]["train_months"])
    )
    capacity = min(
        len(holdout),
        int(config["model"]["review_capacity_per_day"]) * holdout_days,
    )
    selections = {
        name: _ranked_selection(holdout["order_id"], values, capacity)
        for name, values in scores.items()
    }

    metrics_rows = []
    for name, values in scores.items():
        selected = selections[name]
        selected_labels = holdout["order_id"].isin(selected)
        precision = float(holdout.loc[selected_labels, "label"].mean())
        metrics_rows.append(
            {
                "Model": name,
                "PR-AUC": f"{average_precision_score(labels, values):.4f}",
                f"Precision@{capacity:,}": f"{precision:.2%}",
                "Capacity threshold": f"{_capacity_threshold(values, capacity):.6f}",
            }
        )
    metrics = pd.DataFrame(metrics_rows)

    label_patterns = frames["labels"][["order_id", "pattern_id"]].drop_duplicates()
    label_patterns = label_patterns[label_patterns["order_id"].isin(set(holdout["order_id"]))]
    pattern_rows = []
    for pattern, group in label_patterns.groupby("pattern_id", sort=True):
        ids = set(group["order_id"].astype(int))
        row: dict[str, str | int] = {"Pattern": pattern, "Fraud orders": len(ids)}
        for name, selected in selections.items():
            row[name] = f"{len(ids & selected) / max(len(ids), 1):.2%}"
        pattern_rows.append(row)
    pattern_table = pd.DataFrame(pattern_rows)

    exposure = _exposure_table(data_dir)
    cost_rows = []
    cost_results: dict[str, dict[str, float | int]] = {}
    for name, values in scores.items():
        result = _cost_optimal(
            holdout["order_id"], values, labels, exposure, config["costs"]
        )
        cost_results[name] = result
        cost_rows.append(
            {
                "Model": name,
                "Threshold": f"{result['threshold']:.6f}",
                "Alerts": f"{result['alerts']:,}",
                "TP": f"{result['true_positives']:,}",
                "FP": f"{result['false_positives']:,}",
                "Fraud $ caught": f"${result['fraud_dollars_caught']:,.2f}",
                "Total cost": f"${result['total_cost']:,.2f}",
            }
        )
    cost_table = pd.DataFrame(cost_rows)

    pr_auc = {name: average_precision_score(labels, values) for name, values in scores.items()}
    best_name = max(pr_auc, key=pr_auc.get)
    hybrid, hybrid_note = _hybrid_table(
        holdout,
        scores[best_name],
        capacity,
        exposure,
        data_dir / "alerts.csv",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _save_pr_curve(labels, scores, REPORT_DIR / "model_pr_curve.svg")
    _save_pattern_recall(pattern_table, REPORT_DIR / "model_recall_by_pattern.svg")

    calibration_sections = []
    for name, values in scores.items():
        table = _markdown_table(_calibration_table(values, labels))
        calibration_sections.append(f"### {name}\n\n{table}")

    elapsed = time.perf_counter() - started
    reviews_per_day = int(config["model"]["review_capacity_per_day"])
    review_cost = float(config["costs"]["review_cost_usd"])
    report = f"""# Fraud model evaluation

Chronological holdout: {holdout['ts'].min()} through {holdout['ts'].max()}.

Orders: {len(holdout):,}; fraud orders: {labels.sum():,}; base rate: {labels.mean():.3%}.

Review capacity: {reviews_per_day:,}/day × {holdout_days} days = {capacity:,}.

## Detection performance

{_markdown_table(metrics)}

PR-AUC is reported instead of ROC-AUC because ROC-AUC can look strong while obscuring false
positives at this {labels.mean():.2%} fraud base rate. PR-AUC measures performance on the rare
positive class, while precision@capacity reflects the actual review constraint.

![Precision-recall curves](model_pr_curve.svg)

## Calibration deciles

Each bin contains one score-decile of holdout orders; predicted probability is compared with
the observed fraud rate.

{chr(10).join(calibration_sections)}

## Recall by fraud pattern at capacity

{_markdown_table(pattern_table)}

![Recall by fraud pattern](model_recall_by_pattern.svg)

## Cost-optimal thresholds

{_markdown_table(cost_table)}

Total cost counts realized principal not collected on missed fraud, ${review_cost:.2f}
per alert, and the configured margin-plus-LTV insult cost on false positives. A caught fraud
avoids its realized loss exposure; successful down and installment payments reduce that exposure.

## Equal-capacity fraud-dollar comparison

The higher-PR-AUC model ({best_name}) supplies the model ranking.

{_markdown_table(hybrid)}

{hybrid_note}

## Runtime

Evaluation, including CSV load, point-in-time feature construction, scoring, plots, and report:
{elapsed:.2f} seconds.
"""
    (REPORT_DIR / "model.md").write_text(report)

    recall_summary = "; ".join(
        f"{row['Pattern']} "
        + "/".join(f"{name} {row[name]}" for name in scores)
        for row in pattern_rows
    )
    notes_lines = [
        "# Model results",
        f"- Holdout: {len(holdout):,} approved orders; base rate {labels.mean():.3%}.",
        "- PR-AUC: " + "; ".join(f"{name} {pr_auc[name]:.4f}" for name in scores) + ".",
        "- Precision@capacity: "
        + "; ".join(
            f"{row['Model']} {row[f'Precision@{capacity:,}']}" for row in metrics_rows
        )
        + f" ({capacity:,} reviews).",
        f"- Recall by pattern: {recall_summary}.",
        f"- Full evaluation runtime: {elapsed:.2f} seconds.",
    ]
    (REPO / "model" / "README-notes.md").write_text("\n".join(notes_lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
