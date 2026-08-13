"""Put an interval around every published rate.

Run: python -m analysis.uncertainty   (writes reports/uncertainty.md)

Read-only. It re-reads committed artifacts — ``data/``, ``model/artifacts/``,
``llm/eval/results/``, ``reports/`` — and regenerates none of them. Four pieces:

1. Wilson score intervals at 95% on every published rate, with the raw
   numerator and denominator beside each one.
2. McNemar's exact test on the v1 -> v2 prompt change, which ran the same
   cases through the same model and is therefore a paired comparison.
3. A stratified bootstrap band on holdout PR-AUC, using the committed model
   artifacts and the same feature build the evaluation uses.
4. The threshold sensitivity of the chosen rules operating point, read off the
   committed tuning frontier.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.features import FEATURE_COLUMNS, build_features, load_feature_frames  # noqa: E402
from model.train import ARTIFACT_DIR, MODEL_FILES, chronological_split  # noqa: E402

Z_95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 1000
ACTIONS = ("clear", "hold_contact", "escalate", "decline_block")
SMALL_STRATUM = 10


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because the published rates sit
    near 0 and 1 on small denominators, where the normal interval leaves the
    unit interval.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    observed = successes / trials
    denominator = 1 + z**2 / trials
    center = (observed + z**2 / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(observed * (1 - observed) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def mcnemar_exact(discordant_b: int, discordant_c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs.

    Under the null the discordant pairs split like a fair coin, so the exact
    binomial tail is the test. The normal approximation is not used: the
    discordant counts here are small enough for it to matter.
    """
    total = discordant_b + discordant_c
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(discordant_b, discordant_c) + 1))
    return min(1.0, 2 * tail / 2**total)


def bootstrap_pr_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap on PR-AUC, resampling each class separately.

    Stratifying keeps the holdout base rate fixed across resamples, so the
    band describes sampling noise in the ranking rather than noise in how many
    fraud orders happened to land in the window.
    """
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    draws = np.empty(resamples)
    for index in range(resamples):
        picked = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        draws[index] = average_precision_score(labels[picked], scores[picked])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "point": float(average_precision_score(labels, scores)),
        "low": float(low),
        "high": float(high),
        "resamples": resamples,
    }


def _config() -> dict[str, Any]:
    with (REPO / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def _result(prompt_version: str, model: str) -> dict[str, Any]:
    path = REPO / "llm" / "eval" / "results" / f"{prompt_version}__{model}.json"
    return json.loads(path.read_text())


def rules_counts() -> dict[str, int]:
    """Alert and fraud counts behind the published rules precision and recall.

    Precision counts fraud among holdout alerts at the committed review band;
    recall counts alerted fraud among every approved holdout order carrying a
    fraud label.
    """
    config = _config()
    cutoff = pd.Timestamp(config["holdout_start"])
    review_band = json.loads(
        (REPO / "reports" / "operating_point.json").read_text()
    )["review_band"]
    alerts = pd.read_csv(REPO / "data" / "alerts.csv", parse_dates=["ts"])
    orders = pd.read_csv(
        REPO / "data" / "orders.csv", parse_dates=["ts"], usecols=["order_id", "ts", "status"]
    )
    fraud_orders = set(pd.read_csv(REPO / "data" / "labels.csv")["order_id"])

    holdout_alerts = alerts[alerts["ts"].ge(cutoff) & alerts["score"].ge(review_band)]
    approved = orders[orders["ts"].ge(cutoff) & orders["status"].eq("approved")]
    return {
        "review_band": int(review_band),
        "alerts": len(holdout_alerts),
        "alerts_on_fraud": int(holdout_alerts["order_id"].isin(fraud_orders).sum()),
        "holdout_fraud_orders": int(approved["order_id"].isin(fraud_orders).sum()),
    }


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    def is_decline(value: str) -> bool:
        return value == "decline_block"

    return {
        "correct": sum(row["action_ok"] for row in rows),
        "decline_tp": sum(
            is_decline(row["action"]) and is_decline(row["truth_action"]) for row in rows
        ),
        "decline_fp": sum(
            is_decline(row["action"]) and not is_decline(row["truth_action"]) for row in rows
        ),
        "decline_fn": sum(
            not is_decline(row["action"]) and is_decline(row["truth_action"]) for row in rows
        ),
        "pattern_hits": sum(row["top_pattern"] == row["truth_pattern"] for row in rows),
        "non_verbatim": sum(row["has_non_verbatim_claim"] for row in rows),
        "unsupported": sum(row["has_unsupported_claim"] for row in rows),
    }


def published_rates() -> list[dict[str, Any]]:
    """Every published rate with the integer counts it was computed from."""
    rules = rules_counts()
    v1 = _result("memo_v1", "claude-sonnet-5")
    v2 = _result("memo_v2", "claude-sonnet-5")
    luna = _result("memo_v2", "gpt-5.6-luna")
    terra = _result("memo_v2", "gpt-5.6-terra")
    counts = {name: _decision_counts(result["rows"]) for name, result in
              {"v1": v1, "v2": v2, "luna": luna, "terra": terra}.items()}

    rows: list[dict[str, Any]] = [
        {
            "rate": f"rules precision at review ≥ {rules['review_band']}",
            "successes": rules["alerts_on_fraud"],
            "trials": rules["alerts"],
        },
        {
            "rate": "rules recall over holdout fraud orders",
            "successes": rules["alerts_on_fraud"],
            "trials": rules["holdout_fraud_orders"],
        },
    ]
    for label, key, result in [
        ("claude-sonnet-5 memo_v1", "v1", v1),
        ("claude-sonnet-5 memo_v2", "v2", v2),
        ("gpt-5.6-terra memo_v2 (quota-cut)", "terra", terra),
    ]:
        rows += [
            {
                "rate": f"{label} action accuracy",
                "successes": counts[key]["correct"],
                "trials": len(result["rows"]),
            },
        ]
    rows += [
        {
            "rate": "gpt-5.6-luna memo_v2 action accuracy (valid outputs)",
            "successes": counts["luna"]["correct"],
            "trials": len(luna["rows"]),
        },
        {
            "rate": "gpt-5.6-luna memo_v2 action accuracy (schema failure counted wrong)",
            "successes": counts["luna"]["correct"],
            "trials": len(luna["rows"]) + luna["schema_failures"],
        },
        {
            "rate": "claude-sonnet-5 memo_v2 decline precision",
            "successes": counts["v2"]["decline_tp"],
            "trials": counts["v2"]["decline_tp"] + counts["v2"]["decline_fp"],
        },
        {
            "rate": "claude-sonnet-5 memo_v2 decline recall",
            "successes": counts["v2"]["decline_tp"],
            "trials": counts["v2"]["decline_tp"] + counts["v2"]["decline_fn"],
        },
        {
            "rate": "claude-sonnet-5 memo_v2 pattern identification",
            "successes": counts["v2"]["pattern_hits"],
            "trials": len(v2["rows"]),
        },
        {
            "rate": "claude-sonnet-5 memo_v2 memos with a non-verbatim concrete-token field",
            "successes": counts["v2"]["non_verbatim"],
            "trials": len(v2["rows"]),
        },
        {
            "rate": "claude-sonnet-5 memo_v2 memos with an unsupported field",
            "successes": counts["v2"]["unsupported"],
            "trials": len(v2["rows"]),
        },
        {
            "rate": "claude-sonnet-5 memo_v2 perturbation consistency",
            "successes": round(
                v2["consistency_action_agreement"] * v2["consistency_n"]
            ),
            "trials": v2["consistency_n"],
        },
    ]
    for row in rows:
        low, high = wilson_interval(row["successes"], row["trials"])
        row["point"] = row["successes"] / row["trials"]
        row["low"], row["high"] = low, high
    return rows


def per_pattern_rates(prompt_version: str = "memo_v2", model: str = "claude-sonnet-5"
                      ) -> list[dict[str, Any]]:
    """Per-pattern action accuracy with its interval; small strata are marked."""
    result = _result(prompt_version, model)
    rows = []
    for pattern, stratum in sorted(result["per_pattern"].items()):
        trials = stratum["n"]
        successes = round(stratum["action_accuracy"] * trials)
        low, high = wilson_interval(successes, trials)
        rows.append(
            {
                "pattern": pattern,
                "successes": successes,
                "trials": trials,
                "point": successes / trials,
                "low": low,
                "high": high,
                "indicative_only": trials < SMALL_STRATUM,
            }
        )
    return rows


def paired_prompt_tests() -> list[dict[str, Any]]:
    """McNemar on v1 -> v2, which ran the same cases through the same model."""
    v1 = {row["alert_id"]: row for row in _result("memo_v1", "claude-sonnet-5")["rows"]}
    v2 = {row["alert_id"]: row for row in _result("memo_v2", "claude-sonnet-5")["rows"]}
    shared = sorted(set(v1) & set(v2))

    tests = []
    for label, field, better_is_true in [
        ("action correct", "action_ok", True),
        ("memo carries an unsupported field", "has_unsupported_claim", False),
    ]:
        b = sum(1 for k in shared if v1[k][field] and not v2[k][field])
        c = sum(1 for k in shared if not v1[k][field] and v2[k][field])
        gained, lost = (c, b) if better_is_true else (b, c)
        tests.append(
            {
                "comparison": label,
                "paired_cases": len(shared),
                "v1_rate": sum(bool(v1[k][field]) for k in shared) / len(shared),
                "v2_rate": sum(bool(v2[k][field]) for k in shared) / len(shared),
                "improved_by_v2": gained,
                "worsened_by_v2": lost,
                "discordant": b + c,
                "p_value": mcnemar_exact(b, c),
            }
        )
    return tests


def holdout_scores() -> tuple[np.ndarray, dict[str, np.ndarray], pd.Series, int]:
    """Holdout labels and committed-model scores, built exactly as model.evaluate does."""
    config = _config()
    features = build_features(**load_feature_frames(REPO / "data"))
    _, holdout = chronological_split(features, config)
    models = {
        name: joblib.load(ARTIFACT_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    scores = {
        name: model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
        for name, model in models.items()
    }
    holdout_days = max(
        (features["ts"].max() - pd.Timestamp(config["holdout_start"])).days, 1
    )
    capacity = min(
        len(holdout), int(config["model"]["review_capacity_per_day"]) * holdout_days
    )
    return (
        holdout["label"].to_numpy(dtype=int),
        scores,
        holdout["order_id"],
        capacity,
    )


def precision_at_capacity_counts(
    labels: np.ndarray,
    scores: np.ndarray,
    order_ids: pd.Series,
    capacity: int,
) -> tuple[int, int]:
    ranking = pd.DataFrame(
        {"order_id": order_ids.to_numpy(), "score": scores, "label": labels}
    ).sort_values(["score", "order_id"], ascending=[False, True], kind="stable")
    selected = ranking.head(capacity)
    return int(selected["label"].sum()), len(selected)


def threshold_sensitivity() -> dict[str, Any]:
    """Read the committed tuning frontier at the chosen decline band."""
    operating = json.loads((REPO / "reports" / "operating_point.json").read_text())
    columns: list[str] = []
    grid: list[dict[str, float]] = []
    for line in (REPO / "reports" / "tradeoffs.md").read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().lstrip("|").split("|")]
        if not columns:
            columns = cells[: cells.index("net_usd") + 1]
            continue
        if set(cells[0]) <= {"-"}:
            continue
        # the chosen row and over-capacity rows carry a trailing annotation
        values = (float(cell) for cell in cells[: len(columns)])
        grid.append(dict(zip(columns, values, strict=True)))
    at_decline = sorted(
        (point for point in grid if point["decline_band"] == operating["decline_band"]),
        key=lambda point: point["review_band"],
    )
    chosen = next(
        point
        for point in at_decline
        if point["review_band"] == operating["review_band"]
    )
    nets = [point["net_usd"] for point in at_decline]
    position = at_decline.index(chosen)
    return {
        "decline_band": int(operating["decline_band"]),
        "chosen": chosen,
        "points": at_decline,
        "next_band": at_decline[min(position + 1, len(at_decline) - 1)],
        "worst": min(at_decline, key=lambda point: point["net_usd"]),
        "net_spread_share": (max(nets) - min(nets)) / max(nets),
        "chosen_is_grid_edge": chosen is at_decline[0],
        "capacity_per_day": float(_config()["model"]["review_capacity_per_day"]),
    }


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_report() -> str:
    rates = published_rates()
    patterns = per_pattern_rates()
    paired = paired_prompt_tests()
    sensitivity = threshold_sensitivity()
    labels, scores, order_ids, capacity = holdout_scores()
    seed = int(_config()["seed"])

    bootstrap = {
        name: bootstrap_pr_auc(labels, values, seed=seed)
        for name, values in scores.items()
    }
    for name, values in scores.items():
        successes, trials = precision_at_capacity_counts(
            labels, values, order_ids, capacity
        )
        low, high = wilson_interval(successes, trials)
        rates.append(
            {
                "rate": f"{name} precision at review capacity",
                "successes": successes,
                "trials": trials,
                "point": successes / trials,
                "low": low,
                "high": high,
            }
        )

    lines = [
        "# Uncertainty",
        "",
        (
            "Every rate this repository publishes is a proportion on a finite sample, and "
            "several of them sit on denominators small enough that the point estimate is the "
            "least interesting part. This report attaches an interval to each one, tests the "
            "prompt comparison as the paired experiment it is, puts a bootstrap band on the "
            "holdout PR-AUC, and reads the sensitivity of the rules operating point off the "
            "committed tuning frontier. Nothing here regenerates an artifact; it re-reads them."
        ),
        "",
        "## Wilson 95% intervals on published rates",
        "",
        "| rate | k / n | point | 95% interval |",
        "|---|---|---|---|",
    ]
    for row in rates:
        lines.append(
            f"| {row['rate']} | {row['successes']:,} / {row['trials']:,} | "
            f"{_pct(row['point'])} | [{_pct(row['low'])}, {_pct(row['high'])}] |"
        )
    widest = max(rates, key=lambda row: row["high"] - row["low"])
    by_rate = {row["rate"]: row for row in rates}
    sonnet = by_rate["claude-sonnet-5 memo_v2 action accuracy"]
    luna = by_rate["gpt-5.6-luna memo_v2 action accuracy (valid outputs)"]
    lines += [
        "",
        (
            "The interval is the Wilson score interval, which stays inside the unit interval "
            "on the small and near-boundary denominators here. The widest entry above is "
            f"{widest['rate']} at ±{(widest['high'] - widest['low']) / 2 * 100:.1f} points on "
            f"n={widest['trials']:,}. Only the two rules rates carry denominators in the "
            "thousands; the cross-model comparison lives entirely on 200 cases, and the "
            f"sonnet-v2 interval ({_pct(sonnet['low'])}–{_pct(sonnet['high'])}) "
            + (
                "clears the luna interval"
                if sonnet["low"] > luna["high"]
                else "overlaps the luna interval"
            )
            + f" ({_pct(luna['low'])}–{_pct(luna['high'])}) by "
            f"{abs(sonnet['low'] - luna['high']) * 100:.1f} points. The headline gap between "
            "those two arms rests on that margin and nothing more."
        ),
        "",
        "## Per-pattern strata (claude-sonnet-5, memo_v2)",
        "",
        "| truth pattern | k / n | action accuracy | 95% interval |",
        "|---|---|---|---|",
    ]
    for row in patterns:
        mark = " (indicative only)" if row["indicative_only"] else ""
        lines.append(
            f"| {row['pattern']}{mark} | {row['successes']} / {row['trials']} | "
            f"{_pct(row['point'])} | [{_pct(row['low'])}, {_pct(row['high'])}] |"
        )
    small = [row for row in patterns if row["indicative_only"]]
    thinnest = min(patterns, key=lambda row: row["trials"])
    lines += [
        "",
        (
            f"Strata with fewer than {SMALL_STRATUM} cases are marked indicative only: "
            + ", ".join(row["pattern"] for row in small)
            + f". The {thinnest['pattern']} figure of {_pct(thinnest['point'])} on "
            f"{thinnest['trials']} cases is consistent with anything up to "
            f"{_pct(thinnest['high'])}, so the per-pattern table describes where the eval set "
            "is thin as much as it describes the model."
        ),
        "",
        "## Prompt v1 → v2, tested as a paired comparison",
        "",
        (
            "Both arms ran the same cases through the same model, so the informative quantity "
            "is the discordant pairs — cases the two prompts decided differently — not the "
            "difference of two independent-looking rates. McNemar's exact test is used because "
            "the discordant counts are small."
        ),
        "",
        "| comparison | v1 | v2 | v2 better | v2 worse | discordant | exact p |",
        "|---|---|---|---|---|---|---|",
    ]
    for test in paired:
        lines.append(
            f"| {test['comparison']} | {_pct(test['v1_rate'])} | {_pct(test['v2_rate'])} | "
            f"{test['improved_by_v2']} | {test['worsened_by_v2']} | {test['discordant']} | "
            f"{test['p_value']:.2e} |"
        )
    lines += [
        "",
        (
            f"Both changes hold up on {paired[0]['paired_cases']} paired cases. That is a "
            "statement about this prompt on this eval set and nothing wider: the cases were "
            "frozen once, and the prompt was written with knowledge of the v1 failure modes."
        ),
        "",
        "## Holdout PR-AUC, bootstrapped",
        "",
        "| model | PR-AUC | 95% interval | resamples |",
        "|---|---|---|---|",
    ]
    for name, band in bootstrap.items():
        lines.append(
            f"| {name} | {band['point']:.4f} | [{band['low']:.4f}, {band['high']:.4f}] | "
            f"{band['resamples']:,} |"
        )
    ordered = sorted(bootstrap.items(), key=lambda item: -item[1]["point"])
    gap = ordered[0][1]["low"] > ordered[1][1]["high"]
    lines += [
        "",
        (
            f"Resampling is stratified by class, so the holdout base rate is held fixed and "
            f"the band describes noise in the ranking. The {ordered[0][0]} interval "
            + ("does not overlap " if gap else "overlaps ")
            + f"the {ordered[1][0]} interval, so the gap between them "
            + ("survives resampling" if gap else "is not resolved at this sample size")
            + ". The fourth decimal place in the point estimate does not: the interval is two "
            "orders of magnitude wider than that, and quoting PR-AUC to four figures overstates "
            "what one holdout supports."
        ),
        "",
        (
            "One caveat the resample cannot fix: orders are drawn independently, but fraud in "
            "this world arrives in rings and episodes that share users, devices, and addresses. "
            f"The effective sample is therefore smaller than {len(labels):,} orders and the "
            "band above is optimistic by an amount this method cannot measure."
        ),
        "",
        "## Threshold sensitivity at the rules operating point",
        "",
    ]
    chosen = sensitivity["chosen"]
    step = sensitivity["next_band"]
    points = sensitivity["points"]
    last = points[-1]
    middle = points[len(points) // 2]
    ladder = [chosen, middle, last]

    paragraphs = [
        f"Holding the decline band at {sensitivity['decline_band']}, net dollars only fall "
        "as the review band rises: "
        + ", ".join(
            f"${point['net_usd']:,.0f} at {point['review_band']:.0f}" for point in points
        )
        + f" — a spread of {sensitivity['net_spread_share']:.0%} across the swept range. "
        "The operating point is sensitive to the threshold, and sensitive in one direction."
    ]
    if step is not chosen:
        paragraphs.append(
            f"One grid step up, to {step['review_band']:.0f}, costs "
            f"${chosen['net_usd'] - step['net_usd']:,.0f} "
            f"({(chosen['net_usd'] - step['net_usd']) / chosen['net_usd']:.1%}) and "
            f"{(chosen['recall_overall'] - step['recall_overall']) * 100:.1f} points of "
            "recall, so the choice between the first two bands is immaterial. Beyond that "
            "the sweep buys precision with recall — precision "
            + " → ".join(f"{point['precision']:.1%}" for point in ladder)
            + " against recall "
            + " → ".join(f"{point['recall_overall']:.1%}" for point in ladder)
            + " — and the cost model prices that as a losing trade at every step, because "
            "the review cost it saves is small next to the fraud exposure it stops catching."
        )
    if sensitivity["chosen_is_grid_edge"]:
        paragraphs.append(
            "The selected band is the lowest value on the grid, so the sweep bounds the "
            f"operating point from above and not from below. At {chosen['alerts_per_day']} "
            f"alerts/day it also sits well inside the "
            f"{sensitivity['capacity_per_day']:.0f}/day review capacity, so what stops the "
            "search going lower is the grid, not the queue. Whether a band below the grid "
            "edge would price better is an open question this report does not answer."
        )
    else:
        paragraphs.append(
            "The selected band sits inside the swept range, bracketed at "
            f"{points[0]['review_band']:.0f} below and {last['review_band']:.0f} above."
        )
    for paragraph in paragraphs:
        lines += [paragraph, ""]

    lines += [
        "```json",
        json.dumps(
            {
                "wilson": [
                    {
                        "rate": row["rate"],
                        "successes": row["successes"],
                        "trials": row["trials"],
                        "point": round(row["point"], 4),
                        "low": round(row["low"], 4),
                        "high": round(row["high"], 4),
                    }
                    for row in rates
                ],
                "per_pattern": [
                    {
                        "pattern": row["pattern"],
                        "successes": row["successes"],
                        "trials": row["trials"],
                        "low": round(row["low"], 4),
                        "high": round(row["high"], 4),
                    }
                    for row in patterns
                ],
                "mcnemar": [
                    {
                        "comparison": test["comparison"],
                        "improved_by_v2": test["improved_by_v2"],
                        "worsened_by_v2": test["worsened_by_v2"],
                        "p_value": test["p_value"],
                    }
                    for test in paired
                ],
                "pr_auc_bootstrap": {
                    name: {key: round(value, 6) for key, value in band.items()}
                    for name, band in bootstrap.items()
                },
                "seed": seed,
            },
            sort_keys=True,
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    destination = REPO / "reports" / "uncertainty.md"
    destination.write_text(render_report())
    print(f"wrote {destination.relative_to(REPO)}")


if __name__ == "__main__":
    main()
