"""Populate the eval cache by running the triage model live over the eval set,
in parallel (the CLI backend is ~60s/case; serial would take a day).

Every response lands in llm/eval/cache/ keyed by (model, prompt) — after this,
`python -m llm.eval.harness --offline` reproduces metrics with zero calls.

Usage:
  python -m llm.eval.run_live --arms claude-sonnet-5 claude-haiku-4-5-20251001 \
      --prompt-version memo_v1 --workers 6 [--consistency-arm claude-sonnet-5]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm.client import load_config
from llm.triage import draft_memo

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    cfg = load_config()
    tcfg = cfg["llm"]["tasks"]["triage_memo"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=tcfg["eval_arms"])
    ap.add_argument("--prompt-version", default=tcfg["prompt_version"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--consistency-arm", default=None,
                    help="run consistency probes for this arm only")
    args = ap.parse_args()

    cases = json.loads((EVAL_DIR / "cases.json").read_text())
    if args.n:
        cases = cases[: args.n]

    jobs: list[tuple[str, dict, str]] = []  # (model, packet, tag)
    for model in args.arms:
        for c in cases:
            packet = json.loads((EVAL_DIR / "packets" / f"{c['alert_id']}.json").read_text())
            jobs.append((model, packet, f"{model[:20]}/{c['alert_id']}"))
    if args.consistency_arm:
        k_cases = cases[: cfg["llm_eval"]["consistency_cases"]]
        for c in k_cases:
            base = json.loads((EVAL_DIR / "packets" / f"{c['alert_id']}.json").read_text())
            for k in range(cfg["llm_eval"]["consistency_runs"]):
                p2 = dict(base)
                p2["_consistency_probe"] = f"probe {k + 1} (no informational content)"
                jobs.append((args.consistency_arm, p2, f"cons{k + 1}/{c['alert_id']}"))

    print(f"{len(jobs)} calls, {args.workers} workers, prompt={args.prompt_version}")
    done = failed = 0
    lock = threading.Lock()
    t0 = time.time()

    def run(job: tuple[str, dict, str]) -> str | None:
        model, packet, tag = job
        try:
            draft_memo(packet, model=model, prompt_version=args.prompt_version)
            return None
        except Exception as e:  # noqa: BLE001 - collect, report at end
            return f"{tag}: {type(e).__name__} {str(e)[:160]}"

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for fut in as_completed(futs):
            err = fut.result()
            with lock:
                if err:
                    failed += 1
                    errors.append(err)
                else:
                    done += 1
                if (done + failed) % 20 == 0:
                    rate = (done + failed) / max(time.time() - t0, 1)
                    eta = (len(jobs) - done - failed) / max(rate, 1e-9) / 60
                    print(f"  {done + failed}/{len(jobs)} (fail {failed}) eta {eta:.0f}m",
                          flush=True)

    print(f"done={done} failed={failed} in {(time.time() - t0) / 60:.1f}m")
    for e in errors[:20]:
        print("ERR", e)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
