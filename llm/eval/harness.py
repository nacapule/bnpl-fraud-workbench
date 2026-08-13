"""Measured, cache-only-capable evaluation of the triage memo layer.

The decision metrics use policy-derived acceptable actions from ``cases.json``.
The grounding metric is deliberately narrower: it checks concrete numeric,
entity-id, timestamp, and money tokens in ``signals_observed`` and is not a
semantic-truth score.  Cost is not published until CLI usage is trustworthy.

Usage:
  python -m llm.eval.harness --offline
  python -m llm.eval.harness --arms claude-sonnet-5
  python -m llm.eval.harness --prompt-version memo_v1 --n 50
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from llm.client import load_config
from llm.eval.select_cases import assert_provenance
from llm.eval.verifier import verify_memo
from llm.triage import draft_memo

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parents[1]
RESULTS_DIR = EVAL_DIR / "results"
REPORT_PATH = REPO / "reports" / "llm_eval.md"


def top_hypothesis(memo: dict[str, Any]) -> str | None:
    order = {"high": 2, "med": 1, "low": 0}
    hypotheses = memo.get("hypotheses", [])
    if not hypotheses:
        return None
    _, best = max(
        enumerate(hypotheses),
        key=lambda item: (order.get(item[1].get("likelihood"), -1), -item[0]),
    )
    return best.get("pattern")


def _packet(case: dict[str, Any]) -> dict[str, Any]:
    return json.loads((EVAL_DIR / "packets" / f"{case['alert_id']}.json").read_text())


def _decision_metrics(
    rows: list[dict[str, Any]],
    schema_failure_cases: list[dict[str, Any]],
    *,
    include_schema_failures: bool,
) -> dict[str, float | int | None]:
    def is_decline(value: str) -> bool:
        return value == "decline_block"

    denominator = len(rows) + (len(schema_failure_cases) if include_schema_failures else 0)
    true_positives = sum(
        is_decline(row["action"]) and is_decline(row["truth_action"]) for row in rows
    )
    false_positives = sum(
        is_decline(row["action"]) and not is_decline(row["truth_action"]) for row in rows
    )
    false_negatives = sum(
        not is_decline(row["action"]) and is_decline(row["truth_action"]) for row in rows
    )
    if include_schema_failures:
        false_negatives += sum(
            is_decline(case["truth_action"]) for case in schema_failure_cases
        )
    return {
        "n": denominator,
        "action_accuracy": (
            round(sum(row["action_ok"] for row in rows) / denominator, 3)
            if denominator
            else None
        ),
        "decline_precision": (
            round(true_positives / (true_positives + false_positives), 3)
            if true_positives + false_positives
            else None
        ),
        "decline_recall": (
            round(true_positives / (true_positives + false_negatives), 3)
            if true_positives + false_negatives
            else None
        ),
    }


def evaluate_arm(
    cases: list[dict[str, Any]],
    *,
    model: str,
    prompt_version: str,
    offline: bool,
    consistency_cases: int,
    consistency_runs: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    schema_failure_cases: list[dict[str, Any]] = []
    primary_cache_misses = 0
    for case in cases:
        packet = _packet(case)
        acceptable = case.get("truth_actions", [case["truth_action"]])
        try:
            memo, response = draft_memo(
                packet, model=model, prompt_version=prompt_version, offline=offline
            )
        except FileNotFoundError:
            primary_cache_misses += 1
            continue
        except ValueError:
            # Invalid/truncated output is a measured arm failure, not a harness error.
            schema_failure_cases.append(case)
            continue
        verification = verify_memo(memo, packet)
        rows.append(
            {
                "alert_id": case["alert_id"],
                "truth_action": case["truth_action"],
                "acceptable_actions": acceptable,
                "truth_pattern": case["truth_pattern"],
                "action": memo["recommended_action"],
                "action_ok": memo["recommended_action"] in acceptable,
                "priority": memo["priority"],
                "top_pattern": top_hypothesis(memo),
                "has_unsupported_claim": verification["has_unsupported_claim"],
                "has_non_verbatim_claim": (
                    verification["n_derived_claims"] + verification["n_unsupported_claims"]
                    > 0
                ),
                "n_checked_fields": verification["n_checked_fields"],
                "n_verbatim": verification["n_verbatim_claims"],
                "n_derived": verification["n_derived_claims"],
                "n_unsupported": verification["n_unsupported_claims"],
                "n_invalid_citations": verification["n_invalid_citations"],
                "citation_fired_rate": verification["citation_fired_rate"],
                "duration_ms": response.duration_ms,
                "cached": response.cached,
            }
        )

    valid_metrics = _decision_metrics(
        rows, schema_failure_cases, include_schema_failures=False
    )
    inclusive_metrics = _decision_metrics(
        rows, schema_failure_cases, include_schema_failures=True
    )
    n_valid = len(rows)
    n_attempted = n_valid + len(schema_failure_cases)
    checked_fields = sum(row["n_checked_fields"] for row in rows)
    unsupported = sum(row["n_unsupported"] for row in rows)
    derived = sum(row["n_derived"] for row in rows)
    fired_rates = [
        row["citation_fired_rate"]
        for row in rows
        if row["citation_fired_rate"] is not None
    ]

    # Neutral numbered lines perturb the prompt without adding information.
    agreements: list[bool] = []
    consistency_cache_misses = 0
    consistency_cache_hits = 0
    consistency_attempts = 0
    if consistency_runs > 1 and consistency_cases > 0:
        for case in cases[:consistency_cases]:
            packet = _packet(case)
            actions = []
            for index in range(consistency_runs):
                consistency_attempts += 1
                perturbed = dict(packet)
                perturbed["_consistency_probe"] = (
                    f"probe {index + 1} (no informational content)"
                )
                try:
                    memo, _ = draft_memo(
                        perturbed,
                        model=model,
                        prompt_version=prompt_version,
                        offline=offline,
                    )
                    actions.append(memo["recommended_action"])
                    consistency_cache_hits += 1
                except FileNotFoundError:
                    consistency_cache_misses += 1
                    actions = []
                    break
                except ValueError:
                    actions = []
                    break
            if len(actions) == consistency_runs:
                agreements.append(len(set(actions)) == 1)

    per_pattern: dict[str, dict[str, Any]] = {}
    for pattern in sorted({row["truth_pattern"] for row in rows}):
        subset = [row for row in rows if row["truth_pattern"] == pattern]
        per_pattern[pattern] = {
            "n": len(subset),
            "action_accuracy": round(
                sum(row["action_ok"] for row in subset) / len(subset), 3
            ),
            "pattern_id_rate": round(
                sum(row["top_pattern"] == row["truth_pattern"] for row in subset)
                / len(subset),
                3,
            ),
        }

    total_cache_opportunities = (
        n_attempted + primary_cache_misses + consistency_attempts
    )
    cache_hits = n_attempted + consistency_cache_hits
    consistency_agreement = (
        round(sum(agreements) / len(agreements), 3) if agreements else None
    )
    return {
        "model": model,
        "prompt_version": prompt_version,
        "n_cases": n_valid,
        "n_cases_requested": len(cases),
        "cache_misses": primary_cache_misses + consistency_cache_misses,
        "schema_failures": len(schema_failure_cases),
        "schema_failure_rate": (
            round(len(schema_failure_cases) / n_attempted, 4) if n_attempted else None
        ),
        # Legacy top-level decision keys remain valid-output metrics so the
        # published 0.735 / 0.598 series remains directly comparable.
        "action_accuracy": valid_metrics["action_accuracy"],
        "decline_precision": valid_metrics["decline_precision"],
        "decline_recall": valid_metrics["decline_recall"],
        "decision_metrics_valid_only": valid_metrics,
        "decision_metrics_including_schema_failures": inclusive_metrics,
        "pattern_id_rate": (
            round(
                sum(row["top_pattern"] == row["truth_pattern"] for row in rows)
                / n_valid,
                3,
            )
            if n_valid
            else None
        ),
        "unsupported_claim_rate_memo": (
            round(sum(row["has_unsupported_claim"] for row in rows) / n_valid, 3)
            if n_valid
            else None
        ),
        "non_verbatim_claim_rate_memo": (
            round(sum(row["has_non_verbatim_claim"] for row in rows) / n_valid, 3)
            if n_valid
            else None
        ),
        "non_verbatim_claim_rate_checked_fields": (
            round((unsupported + derived) / checked_fields, 4) if checked_fields else None
        ),
        "unsupported_claim_rate_checked_fields": (
            round(unsupported / checked_fields, 4) if checked_fields else None
        ),
        "derived_claim_rate_checked_fields": (
            round(derived / checked_fields, 4) if checked_fields else None
        ),
        "grounding_metric_scope": (
            "signals_observed concrete numeric/id/timestamp/money tokens; not semantic truth"
        ),
        "invalid_citation_memos": sum(
            row["n_invalid_citations"] > 0 for row in rows
        ),
        "citation_fired_rate_mean": (
            round(statistics.mean(fired_rates), 3) if fired_rates else None
        ),
        "consistency_action_agreement": consistency_agreement,
        "consistency_n": len(agreements),
        "latency_p50_ms": (
            int(statistics.median(row["duration_ms"] for row in rows)) if rows else None
        ),
        "cache_hit_rate": (
            round(cache_hits / total_cache_opportunities, 3)
            if total_cache_opportunities
            else None
        ),
        "per_pattern": per_pattern,
        "action_confusion": {
            f"{truth}->{action}": count
            for (truth, action), count in Counter(
                (row["truth_action"], row["action"]) for row in rows
            ).items()
        },
        "rows": rows,
    }


def _write_report(summaries: list[dict[str, Any]], n_requested: int) -> None:
    lines = [
        "# LLM triage eval\n",
        f"prompt version: **{summaries[0]['prompt_version']}** · requested cases: {n_requested}\n",
        (
            "Grounding scope: concrete numeric, entity-id, timestamp, and money tokens in "
            "`signals_observed`; token presence is not semantic-truth verification.\n"
        ),
        "| metric | " + " | ".join(summary["model"] for summary in summaries) + " |",
        "|---|" + "---|" * len(summaries),
    ]
    for key in [
        "n_cases",
        "cache_misses",
        "schema_failures",
        "action_accuracy",
        "decline_precision",
        "decline_recall",
        "pattern_id_rate",
        "non_verbatim_claim_rate_memo",
        "non_verbatim_claim_rate_checked_fields",
        "unsupported_claim_rate_memo",
        "unsupported_claim_rate_checked_fields",
        "derived_claim_rate_checked_fields",
        "consistency_action_agreement",
        "consistency_n",
        "latency_p50_ms",
    ]:
        lines.append(
            f"| {key} | " + " | ".join(str(summary[key]) for summary in summaries) + " |"
        )
    lines.extend(
        [
            "",
            "Decision accuracy denominators:",
        ]
    )
    for summary in summaries:
        valid = summary["decision_metrics_valid_only"]
        inclusive = summary["decision_metrics_including_schema_failures"]
        lines.append(
            f"- {summary['model']}: {valid['action_accuracy']} over {valid['n']} valid outputs; "
            f"{inclusive['action_accuracy']} over {inclusive['n']} outputs with schema failures "
            "counted as wrong."
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_PATH.relative_to(REPO)}")


def main() -> None:
    cfg = load_config()
    assert_provenance(cfg)
    task_config = cfg["llm"]["tasks"]["triage_memo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--arms", nargs="*", default=task_config["eval_arms"])
    parser.add_argument("--prompt-version", default=task_config["prompt_version"])
    parser.add_argument("--n", type=int, default=None, help="limit cases (debug)")
    parser.add_argument("--no-consistency", action="store_true")
    args = parser.parse_args()

    cases = json.loads((EVAL_DIR / "cases.json").read_text())
    if args.n:
        cases = cases[: args.n]
    RESULTS_DIR.mkdir(exist_ok=True)

    summaries = []
    for model in args.arms:
        result = evaluate_arm(
            cases,
            model=model,
            prompt_version=args.prompt_version,
            offline=args.offline,
            consistency_cases=(
                0 if args.no_consistency else cfg["llm_eval"]["consistency_cases"]
            ),
            consistency_runs=(
                0 if args.no_consistency else cfg["llm_eval"]["consistency_runs"]
            ),
        )
        output = RESULTS_DIR / f"{args.prompt_version}__{model}.json"
        output.write_text(json.dumps(result, indent=1, sort_keys=True))
        summaries.append(result)
        omitted = {"rows", "per_pattern", "action_confusion"}
        print(json.dumps({k: v for k, v in result.items() if k not in omitted}, indent=1))

    _write_report(summaries, len(cases))

    # Export priorities only when the configured arm ran.  This is an advisory
    # queue input, not ground truth.
    default_model = task_config["model"]
    for summary in summaries:
        if summary["model"] == default_model:
            priorities = pd.DataFrame(
                [
                    {"alert_id": row["alert_id"], "priority": row["priority"]}
                    for row in summary["rows"]
                ]
            )
            output = RESULTS_DIR / "priorities.csv"
            priorities.to_csv(output, index=False)
            print(f"wrote {output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
