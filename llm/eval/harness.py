"""Measured evaluation of the Claude triage layer on the frozen eval set.

Metrics (definitions repeated in ITERATION.md; ground truth from cases.json):

1. action accuracy        recommended_action ∈ the policy-derived acceptable
                          set for the true pattern (cases.json truth_actions;
                          single-action for most patterns, see select_cases.py)
2. decline precision/recall   positive class = decline_block ONLY (escalate is
                          not counted as a decline — strict on the dangerous
                          direction)
3. pattern id rate        top hypothesis (highest likelihood, ties -> first
                          listed) matches truth_pattern
4. hallucination rate     memo-level: any signals_observed claim with a
                          concrete token absent from the packet (verifier.py);
                          claim-level rate also reported
5. citation validity      cited rule ids exist; fired-rate reported alongside
6. consistency            K extra runs on M cases under a neutral prompt
                          perturbation (a numbered no-op line); reported as
                          all-runs-agree rate on recommended_action. With
                          temperature 0 and a byte-identical prompt the cache
                          would hide any variance, so perturbation-consistency
                          is the honest measurement.
7. cost & latency         API-equivalent cost from token counts when the
                          backend reports them, else n/a; wall latency always.

Usage:
  python -m llm.eval.harness --offline            # cache only (CI mode)
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
from llm.eval.verifier import verify_memo
from llm.triage import draft_memo

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def top_hypothesis(memo: dict[str, Any]) -> str | None:
    order = {"high": 2, "med": 1, "low": 0}
    hyps = memo.get("hypotheses", [])
    if not hyps:
        return None
    best = max(enumerate(hyps), key=lambda kv: (order.get(kv[1].get("likelihood"), -1), -kv[0]))
    return best[1].get("pattern")


def evaluate_arm(
    cases: list[dict[str, Any]],
    *,
    model: str,
    prompt_version: str,
    offline: bool,
    consistency_cases: int,
    consistency_runs: int,
) -> dict[str, Any]:
    rows = []
    schema_failures = 0
    for c in cases:
        packet = json.loads((EVAL_DIR / "packets" / f"{c['alert_id']}.json").read_text())
        acceptable = c.get("truth_actions", [c["truth_action"]])
        try:
            memo, resp = draft_memo(
                packet, model=model, prompt_version=prompt_version, offline=offline
            )
        except FileNotFoundError:
            raise SystemExit(
                f"offline mode but no cached response for alert {c['alert_id']} "
                f"({model}/{prompt_version}) — run live first"
            ) from None
        except ValueError:
            # unusable output (truncated/invalid JSON, schema violation) is a
            # scored failure mode of the arm, not a harness error
            schema_failures += 1
            continue
        v = verify_memo(memo, packet)
        rows.append(
            {
                "alert_id": c["alert_id"],
                "truth_action": c["truth_action"],
                "acceptable_actions": acceptable,
                "truth_pattern": c["truth_pattern"],
                "action": memo["recommended_action"],
                "action_ok": memo["recommended_action"] in acceptable,
                "priority": memo["priority"],
                "top_pattern": top_hypothesis(memo),
                "hallucinated": v["hallucinated"],
                "n_claims": v["n_claims"],
                "n_unsupported": v["n_unsupported_claims"],
                "n_invalid_citations": v["n_invalid_citations"],
                "citation_fired_rate": v["citation_fired_rate"],
                "cost_usd": resp.cost_usd,
                "duration_ms": resp.duration_ms,
                "cached": resp.cached,
            }
        )

    n = len(rows)
    def _is_decl(x: str) -> bool:
        return x == "decline_block"

    tp = sum(1 for r in rows if _is_decl(r["action"]) and _is_decl(r["truth_action"]))
    fp = sum(1 for r in rows if _is_decl(r["action"]) and not _is_decl(r["truth_action"]))
    fn = sum(1 for r in rows if not _is_decl(r["action"]) and _is_decl(r["truth_action"]))
    claims = sum(r["n_claims"] for r in rows)
    unsupported = sum(r["n_unsupported"] for r in rows)
    fired_rates = [r["citation_fired_rate"] for r in rows if r["citation_fired_rate"] is not None]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]

    # consistency probes: neutral numbered line appended to the packet prompt
    agree = None
    if consistency_runs > 1 and consistency_cases > 0:
        agreements = []
        for c in cases[:consistency_cases]:
            packet = json.loads((EVAL_DIR / "packets" / f"{c['alert_id']}.json").read_text())
            actions = []
            for k in range(consistency_runs):
                p2 = dict(packet)
                p2["_consistency_probe"] = f"probe {k + 1} (no informational content)"
                try:
                    m2, _ = draft_memo(
                        p2, model=model, prompt_version=prompt_version, offline=offline
                    )
                    actions.append(m2["recommended_action"])
                except FileNotFoundError:
                    actions = []
                    break
            if actions:
                agreements.append(len(set(actions)) == 1)
        agree = sum(agreements) / len(agreements) if agreements else None

    per_pattern: dict[str, dict[str, Any]] = {}
    for pat in sorted({r["truth_pattern"] for r in rows}):
        sub = [r for r in rows if r["truth_pattern"] == pat]
        per_pattern[pat] = {
            "n": len(sub),
            "action_accuracy": round(sum(r["action_ok"] for r in sub) / len(sub), 3),
            "pattern_id_rate": round(
                sum(r["top_pattern"] == r["truth_pattern"] for r in sub) / len(sub), 3
            ),
        }

    return {
        "model": model,
        "prompt_version": prompt_version,
        "n_cases": n,
        "schema_failures": schema_failures,
        "schema_failure_rate": round(schema_failures / (n + schema_failures), 4)
        if n + schema_failures else None,
        "action_accuracy": round(sum(r["action_ok"] for r in rows) / n, 3),
        "decline_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "decline_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "pattern_id_rate": round(
            sum(r["top_pattern"] == r["truth_pattern"] for r in rows) / n, 3
        ),
        "hallucination_rate_memo": round(sum(r["hallucinated"] for r in rows) / n, 3),
        "hallucination_rate_claim": round(unsupported / claims, 4) if claims else None,
        "invalid_citation_memos": sum(1 for r in rows if r["n_invalid_citations"] > 0),
        "citation_fired_rate_mean": round(statistics.mean(fired_rates), 3) if fired_rates else None,
        "consistency_action_agreement": round(agree, 3) if agree is not None else None,
        "cost_per_case_usd": round(statistics.mean(costs), 5) if costs else None,
        "latency_p50_ms": int(statistics.median(r["duration_ms"] for r in rows)),
        "cache_hit_rate": round(sum(r["cached"] for r in rows) / n, 3),
        "per_pattern": per_pattern,
        "action_confusion": {
            f"{t}->{a}": c
            for (t, a), c in Counter((r["truth_action"], r["action"]) for r in rows).items()
        },
        "rows": rows,
    }


def main() -> None:
    cfg = load_config()
    tcfg = cfg["llm"]["tasks"]["triage_memo"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--arms", nargs="*", default=tcfg["eval_arms"])
    ap.add_argument("--prompt-version", default=tcfg["prompt_version"])
    ap.add_argument("--n", type=int, default=None, help="limit cases (debug)")
    ap.add_argument("--no-consistency", action="store_true")
    args = ap.parse_args()

    cases = json.loads((EVAL_DIR / "cases.json").read_text())
    if args.n:
        cases = cases[: args.n]
    RESULTS_DIR.mkdir(exist_ok=True)

    summaries = []
    for model in args.arms:
        res = evaluate_arm(
            cases,
            model=model,
            prompt_version=args.prompt_version,
            offline=args.offline,
            consistency_cases=0 if args.no_consistency else cfg["llm_eval"]["consistency_cases"],
            consistency_runs=0 if args.no_consistency else cfg["llm_eval"]["consistency_runs"],
        )
        out = RESULTS_DIR / f"{args.prompt_version}__{model}.json"
        out.write_text(json.dumps(res, indent=1, sort_keys=True))
        summaries.append(res)
        drop = {"rows", "per_pattern", "action_confusion"}
        keep = {k: v for k, v in res.items() if k not in drop}
        print(json.dumps(keep, indent=1))

    lines = [
        "# LLM triage eval\n",
        f"prompt version: **{args.prompt_version}** · cases: {len(cases)}\n",
        "| metric | " + " | ".join(s["model"] for s in summaries) + " |",
        "|---|" + "---|" * len(summaries),
    ]
    for key in [
        "action_accuracy",
        "decline_precision",
        "decline_recall",
        "pattern_id_rate",
        "hallucination_rate_memo",
        "hallucination_rate_claim",
        "consistency_action_agreement",
        "cost_per_case_usd",
        "latency_p50_ms",
    ]:
        lines.append(f"| {key} | " + " | ".join(str(s[key]) for s in summaries) + " |")
    Path("reports/llm_eval.md").write_text("\n".join(lines) + "\n")
    print("wrote reports/llm_eval.md")

    # Export per-alert priorities from the configured default arm so the queue
    # simulation can dispatch on LLM priority (advisory artifact, not truth).
    default_model = tcfg["model"]
    for s in summaries:
        if s["model"] == default_model:
            pr = pd.DataFrame(
                [{"alert_id": r["alert_id"], "priority": r["priority"]} for r in s["rows"]]
            )
            out_pr = RESULTS_DIR / "priorities.csv"
            pr.to_csv(out_pr, index=False)
            print(f"wrote {out_pr}")


if __name__ == "__main__":
    main()
