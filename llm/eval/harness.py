"""The memo benchmark: live runs under a call cap, cached replay, coverage gate and scores.

A benchmark lives in ``llm/eval/benchmarks/<id>/`` and is fixed before any call:

``benchmark.json``
    id, phase (``development`` or ``final``), policy id and SHA-256, prompt version and
    SHA-256 of the system prompt, SHA-256 of the referee and packet code, the arms
    (backend, model, effort), the cases (case id built from seed, family, order and
    decision time, so an alert keeps its identity whatever the policy bands; stratum;
    cluster; sampling weight; the latent diagnostic) and the invariance probes.
``packets/<case>.json``, ``referee.json``
    the packets and the referee's view of each, written by :mod:`llm.eval.select_cases`.
``cache/<key>.json``
    one record per case, probe and arm: the call's identity (backend, CLI version, model,
    effort, isolation settings, prompt, policy, referee and packet hashes), the response,
    its event-log summary, duration and transport attempts. A response whose log shows a
    tool, file or error event, or that names a private term, is kept out: its record
    holds only the identity, the summary and the reason, and the case counts as a
    protocol failure. Full event logs go to ``--log-dir``, outside the repository.

Scoring replays the cache only. Every case and probe of every arm must have a record
(the coverage gate); malformed output is a counted failure; nothing is retried for
format. Usage::

  python -m llm.eval.harness --benchmark 2026-10-dev --arm sol --live \\
      --log-dir <dir> --private-terms <file> --max-calls 40
  python -m llm.eval.harness --benchmark 2026-10-dev           # score from the cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm import client, memo, referee
from llm.eval import verifier
from llm.packet import ENTITIES, placeholders

EVAL = Path(__file__).resolve().parent
BENCHMARKS = EVAL / "benchmarks"
PROBES = ("primary", "shuffled", "renamed")
PROBE_SEED = 7919  # shuffles the context facts / draws fresh placeholder names per case


class CoverageError(RuntimeError):
    """An arm lacks a cached record for a case or probe the benchmark requires."""


@dataclass(frozen=True)
class Arm:
    name: str
    backend: str
    model: str
    effort: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha256() -> dict[str, str]:
    """Hashes of what scores a memo, recorded in the benchmark definition."""
    llm_dir = EVAL.parent
    return {name: sha256_file(llm_dir / name) for name in (
        "referee.py", "packet.py", "memo.py", "eval/verifier.py", "eval/tokens.py")}


def load_benchmark(directory: Path) -> dict[str, Any]:
    definition = json.loads((directory / "benchmark.json").read_text())
    definition["arms"] = {name: Arm(name, **arm) for name, arm in definition["arms"].items()}
    return definition


def probe_packet(packet: Mapping[str, Any], probe: str, case_id: str) -> str:
    """The user message for a probe: the packet as is, with its context facts in another
    order, or with fresh placeholder names."""
    seed = int(hashlib.sha256(f"{PROBE_SEED}:{case_id}".encode()).hexdigest()[:8], 16)
    if probe == "primary":
        return memo.user_prompt(packet)
    if probe == "shuffled":
        return memo.user_prompt(packet, shuffle_seed=seed)
    if probe == "renamed":
        names = placeholders(seed)
        order = {**packet["order"], **{entity: names[entity] for entity in ENTITIES}}
        return memo.user_prompt({**packet, "order": order})
    raise ValueError(f"unknown probe {probe!r}")


def call_identity(definition: Mapping[str, Any], arm: Arm, case_id: str, probe: str,
                  user_prompt: str, system_prompt: str) -> dict[str, Any]:
    return {
        "benchmark": definition["id"], "case": case_id, "probe": probe,
        "backend": arm.backend, "model": arm.model, "effort": arm.effort,
        "policy": definition["policy"]["id"],
        "policy_sha256": definition["policy"]["sha256"],
        "prompt_version": definition["prompt"]["version"],
        "prompt_sha256": client.sha256_text(system_prompt),
        "code_sha256": definition["code_sha256"],
        "packet_sha256": client.sha256_text(user_prompt),
    }


def cache_key(identity: Mapping[str, Any]) -> str:
    return client.sha256_text(client.canonical_json(identity))[:32]


def required_calls(definition: Mapping[str, Any]) -> list[tuple[str, str]]:
    calls = [(case["case_id"], "primary") for case in definition["cases"]]
    for probe, case_ids in sorted(definition.get("probes", {}).items()):
        calls += [(case_id, probe) for case_id in case_ids]
    return calls


# ------------------------------------------------------------------------- live runs


def run_live(directory: Path, arm_name: str, *, log_dir: Path, private_terms: Sequence[str],
             max_calls: int, max_tokens: int | None = None,
             backend_factory: Callable[[str], Any] = client.backend) -> dict[str, int]:
    """Call the arm's model for every case and probe without a cached record.

    At most one transport retry per case; no retry for format. Stops before exceeding
    ``max_calls`` calls (attempts included) or ``max_tokens`` tokens.
    """
    if not private_terms:
        raise ValueError("live calls need the private terms to scan responses for")
    definition = load_benchmark(directory)
    arm = definition["arms"][arm_name]
    system_prompt = memo.system_prompt(definition["prompt"]["version"])
    if client.sha256_text(system_prompt) != definition["prompt"]["sha256"]:
        raise RuntimeError("the system prompt differs from the benchmark definition")
    backend = backend_factory(arm.backend)
    cache = directory / "cache"
    cache.mkdir(exist_ok=True)
    logs = log_dir / definition["id"] / arm.name
    quarantine = log_dir / "quarantine"
    logs.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    used = {"calls": 0, "tokens": 0, "recorded": 0, "quarantined": 0, "transport_failures": 0}
    for case_id, probe in required_calls(definition):
        packet = json.loads((directory / "packets" / f"{case_id}.json").read_text())
        user_prompt = probe_packet(packet, probe, case_id)
        identity = call_identity(definition, arm, case_id, probe, user_prompt, system_prompt)
        path = cache / f"{cache_key(identity)}.json"
        if path.exists():
            continue
        errors: list[str] = []
        response = None
        for _attempt in range(2):
            if used["calls"] >= max_calls or (max_tokens and used["tokens"] >= max_tokens):
                print(f"cap reached: {used}", file=sys.stderr)
                return used
            used["calls"] += 1
            try:
                response = backend.complete(client.Request(
                    arm.backend, arm.model, arm.effort, system_prompt, user_prompt))
                break
            except client.BackendError as error:
                errors.append(str(error)[:500])
        stem = f"{case_id}-{probe}"
        record: dict[str, Any] = {"identity": identity, "transport_errors": errors}
        if response is None:
            used["transport_failures"] += 1
            record["outcome"] = "transport_failure"
        else:
            used["tokens"] += (response.summary.input_tokens or 0) + (
                response.summary.output_tokens or 0)
            (logs / f"{stem}.events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in response.events))
            reasons = []
            if not response.summary.protocol_ok:
                reasons.append("tool, file or error events in the log")
            if response.summary.model != arm.model:
                reasons.append(f"answered by {response.summary.model}")
            if client.private_matches(response.text, private_terms):
                reasons.append("names a private term")
            record.update(summary=response.summary.as_dict(),
                          duration_ms=response.duration_ms)
            if reasons:
                used["quarantined"] += 1
                record["outcome"] = "protocol_failure"
                record["quarantined"] = reasons
                (quarantine / f"{definition['id']}-{arm.name}-{stem}.txt").write_text(
                    response.text)
            else:
                record["outcome"] = "response"
                record["text"] = response.text
        path.write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        used["recorded"] += 1
    return used


# --------------------------------------------------------------------------- scoring


def cached_records(directory: Path, definition: Mapping[str, Any], arm: Arm
                   ) -> dict[tuple[str, str], dict[str, Any]]:
    """The arm's record for every required case and probe; raises CoverageError if any
    is missing."""
    system_prompt = memo.system_prompt(definition["prompt"]["version"])
    records: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for case_id, probe in required_calls(definition):
        packet = json.loads((directory / "packets" / f"{case_id}.json").read_text())
        identity = call_identity(definition, arm, case_id, probe,
                                 probe_packet(packet, probe, case_id), system_prompt)
        path = directory / "cache" / f"{cache_key(identity)}.json"
        if not path.exists():
            missing.append(f"{case_id}/{probe}")
            continue
        record = json.loads(path.read_text())
        if record["identity"] != identity:
            raise RuntimeError(f"cache record {path.name} does not match its identity")
        records[(case_id, probe)] = record
    if missing:
        raise CoverageError(f"arm {arm.name} lacks {len(missing)} of "
                            f"{len(required_calls(definition))} calls: {missing[:5]}")
    return records


def score_record(record: Mapping[str, Any], packet: Mapping[str, Any],
                 view: referee.RefereeView) -> dict[str, Any]:
    """One case's outcome: a transport, protocol or format failure, or the referee's and
    the verifier's scores."""
    if record["outcome"] != "response":
        return {"outcome": record["outcome"], "acceptable": False, "disposition": None}
    parsed, problems = memo.parse(record["text"])
    if parsed is None:
        return {"outcome": "format_failure", "acceptable": False, "disposition": None,
                "problems": problems[:10]}
    scored = referee.score(parsed, view)
    return {"outcome": "scored", **scored.as_dict(),
            "verification": verifier.verify_memo(parsed, packet)}


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "value": round(numerator / denominator, 4) if denominator else None}


def summarize_arm(definition: Mapping[str, Any], rows: Mapping[tuple[str, str], dict[str, Any]]
                  ) -> dict[str, Any]:
    """Rates over the primary cases (failures count as not acceptable), per stratum, the
    rate reweighted to the natural alert mix, cluster counts and invariance agreement."""
    cases = definition["cases"]
    primary = [rows[(case["case_id"], "primary")] for case in cases]
    n = len(cases)
    outcomes = Counter(row["outcome"] for row in primary)
    scored = [row for row in primary if row["outcome"] == "scored"]
    acceptable = [bool(row["acceptable"]) for row in primary]
    weights = [float(case["weight"]) for case in cases]
    strata: dict[str, list[bool]] = defaultdict(list)
    for case, ok in zip(cases, acceptable, strict=True):
        strata[case["stratum"]].append(ok)
    invariance = {}
    for probe, case_ids in sorted(definition.get("probes", {}).items()):
        same = sum(rows[(case_id, probe)].get("disposition") is not None
                   and rows[(case_id, probe)]["disposition"]
                   == rows[(case_id, "primary")]["disposition"] for case_id in case_ids)
        invariance[probe] = _rate(same, len(case_ids))
    claims = sum(row["verification"]["n_claims"] for row in scored)
    claim_errors = sum(row["verification"]["n_claim_errors"] for row in scored)
    return {
        "n_cases": n,
        "outcomes": dict(sorted(outcomes.items())),
        "acceptable": _rate(sum(acceptable), n),
        "acceptable_reweighted": (round(sum(w * ok for w, ok in zip(weights, acceptable,
                                                                    strict=True))
                                        / sum(weights), 4) if sum(weights) else None),
        "standard": _rate(sum(bool(row.get("standard")) for row in primary), n),
        "prohibited": _rate(sum(bool(row.get("prohibited")) for row in primary), n),
        "needs_check": _rate(sum(bool(row.get("needs_check")) for row in primary), n),
        "next_check_ok": _rate(sum(bool(row.get("next_check_ok")) for row in scored),
                               len(scored)),
        "citation_ok": _rate(sum(bool(row.get("citation_ok")) for row in scored), len(scored)),
        "nonpayment_violations": sum(bool(row.get("nonpayment_violation")) for row in scored),
        "claim_errors": _rate(claim_errors, claims),
        "memos_with_claim_error": _rate(
            sum(row["verification"]["has_claim_error"] for row in scored), len(scored)),
        "memos_with_unmatched_token": _rate(
            sum(row["verification"]["has_unmatched_token"] for row in scored), len(scored)),
        "per_stratum": {stratum: _rate(sum(oks), len(oks))
                        for stratum, oks in sorted(strata.items())},
        "clusters": len({case["cluster"] for case in cases}),
        "invariance": invariance,
    }


def disagreement_table(definition: Mapping[str, Any], views: Mapping[str, Mapping[str, Any]],
                       rows: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, int]:
    """Counts of (the referee's standard disposition, the memo's disposition)."""
    table: Counter[str] = Counter()
    for case in definition["cases"]:
        standard = "|".join(views[case["case_id"]]["standard"])
        disposition = rows[(case["case_id"], "primary")].get("disposition") or "failure"
        table[f"{standard} -> {disposition}"] += 1
    return dict(sorted(table.items()))


def latent_diagnostic(definition: Mapping[str, Any],
                      rows: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Against simulation truth, reported apart from the policy score: how often the
    memo suggested an adverse disposition for legitimate orders and cleared fraud."""
    adverse = ("hold", "decline", "escalate", "needs_check")
    by_class: dict[str, list[str | None]] = defaultdict(list)
    for case in definition["cases"]:
        by_class[case["latent"]["class"]].append(
            rows[(case["case_id"], "primary")].get("disposition"))
    legitimate, fraud = by_class.get("legitimate", []), by_class.get("fraud", [])
    return {
        "legitimate_adverse": _rate(sum(d in adverse for d in legitimate), len(legitimate)),
        "fraud_cleared": _rate(sum(d == "clear" for d in fraud), len(fraud)),
    }


def statistics(definition: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]
               ) -> dict[str, Any]:
    """Wilson and cluster-bootstrap intervals for each arm's acceptable rate, and paired
    comparisons of every two arms on the same cases."""
    import numpy as np

    from core.stats import cluster_bootstrap, paired_outcomes, wilson_interval

    ids = [case["case_id"] for case in definition["cases"]]
    clusters = [case["cluster"] for case in definition["cases"]]
    ok = {name: {case_id: bool(arm["cases"][f"{case_id}/primary"]["acceptable"])
                 for case_id in ids} for name, arm in arms.items()}
    out: dict[str, Any] = {"arms": {}, "paired": {}}
    for name, outcomes in ok.items():
        values = np.array([outcomes[case_id] for case_id in ids], dtype=float)
        wilson = wilson_interval(int(values.sum()), len(values))
        cluster = cluster_bootstrap(clusters, lambda idx, v=values: float(v[idx].mean()),
                                    resamples=2000, seed=0)
        out["arms"][name] = {"wilson": [round(wilson.low, 4), round(wilson.high, 4)],
                             "cluster_bootstrap": [round(cluster.low, 4),
                                                   round(cluster.high, 4)]}
    names = sorted(ok)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            paired = paired_outcomes(ok[first], ok[second])
            out["paired"][f"{first} vs {second}"] = {
                "both": paired.both, "only_first": paired.only_first,
                "only_second": paired.only_second, "neither": paired.neither,
                "p_value": round(paired.p_value, 6)}
    return out


def evaluate(directory: Path) -> dict[str, Any]:
    """Score every arm from the cache (coverage gate first)."""
    definition = load_benchmark(directory)
    views = json.loads((directory / "referee.json").read_text())
    results: dict[str, Any] = {"benchmark": definition["id"], "arms": {}}
    for arm in definition["arms"].values():
        records = cached_records(directory, definition, arm)
        rows = {}
        for (case_id, probe), record in records.items():
            packet = json.loads((directory / "packets" / f"{case_id}.json").read_text())
            view = referee.view(packet)
            if view.as_dict() != views[case_id]:
                raise RuntimeError(f"the referee's view of {case_id} changed since it was fixed")
            rows[(case_id, probe)] = score_record(record, packet, view)
        versions = {(record.get("summary") or {}).get("cli_version") for record in
                    records.values()} - {None}
        summaries = [record.get("summary") or {} for record in records.values()]
        spend = {
            "calls": sum(len(record["transport_errors"])
                         + (record["outcome"] != "transport_failure")
                         for record in records.values()),
            "input_tokens": sum(s.get("input_tokens") or 0 for s in summaries),
            "output_tokens": sum(s.get("output_tokens") or 0 for s in summaries),
        }
        results["arms"][arm.name] = {
            "arm": arm.__dict__, "cli_versions": sorted(versions), "spend": spend,
            "summary": summarize_arm(definition, rows),
            "disagreement": disagreement_table(definition, views, rows),
            "latent_diagnostic": latent_diagnostic(definition, rows),
            "cases": {f"{case_id}/{probe}": row for (case_id, probe), row in sorted(rows.items())},
        }
    results["statistics"] = statistics(definition, results["arms"])
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--arm")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--private-terms", type=Path,
                        help="file with one private term per line (kept outside the repository)")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args(argv)
    directory = BENCHMARKS / args.benchmark
    if args.live:
        if not (args.arm and args.log_dir and args.private_terms and args.max_calls):
            parser.error("--live needs --arm, --log-dir, --private-terms and --max-calls")
        terms = [line.strip() for line in args.private_terms.read_text().splitlines()
                 if line.strip()]
        used = run_live(directory, args.arm, log_dir=args.log_dir, private_terms=terms,
                        max_calls=args.max_calls, max_tokens=args.max_tokens)
        print(json.dumps(used))
        return 0
    try:
        results = evaluate(directory)
    except CoverageError as error:
        print(f"coverage gate: {error}", file=sys.stderr)
        return 1
    out = directory / "results.json"
    out.write_text(json.dumps(results, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    for name, arm in results["arms"].items():
        print(name, json.dumps({k: v for k, v in arm["summary"].items()
                                if k in ("n_cases", "outcomes", "acceptable")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
