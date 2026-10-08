"""A benchmark's committed results: the harness's scores, with the endpoints named.

``python -m llm.eval.results --benchmark <id>`` scores every arm that has records
(``harness.evaluate``, from the cache only) and writes ``results.json`` next to the
benchmark. The file is a pure function of the committed benchmark and records: keys are
sorted, and it holds no time, path or host. Beside the harness's output it adds:

- ``endpoints``: the endpoints of ``PROTOCOL.md`` under their names there, each value
  copied from the harness's output with its JSON pointer (``source``). Nothing is
  computed here, so no statistic appears that the protocol does not name.
- ``case_memos`` (phase ``cases`` only): per case, its case file and slot, the arm's
  validated memo and its score, so the case files can be rendered from this file alone.

The scoring code is not part of this module: a change here never changes a score, and
the benchmarks' frozen-code check does not cover it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm import memo
from llm.eval import harness

PROTOCOL = "llm/eval/PROTOCOL.md"
NOT_PAIRED = "made only when the benchmark has two arms and both are scored"


def dumps(results: Mapping[str, Any]) -> str:
    """The exact bytes of ``results.json`` (as text)."""
    return json.dumps(results, indent=1, sort_keys=True) + "\n"


def write(path: Path, results: Mapping[str, Any]) -> None:
    """Write ``results.json`` whole or not at all."""
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(dumps(results))
    os.replace(partial, path)


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer(*tokens: str) -> str:
    return "".join(f"/{_escape(str(token))}" for token in tokens)


def resolve(results: Mapping[str, Any], pointer: str) -> Any:
    """The value a JSON pointer names in ``results`` (KeyError when absent)."""
    value: Any = results
    for token in pointer.split("/")[1:]:
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def _ref(results: Mapping[str, Any], *tokens: str) -> dict[str, Any] | None:
    pointer = _pointer(*tokens)
    try:
        return {"source": pointer, "value": resolve(results, pointer)}
    except KeyError:
        return None


def _refs(results: Mapping[str, Any], spec: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    out = {}
    for name, tokens in spec.items():
        ref = _ref(results, *tokens)
        if ref is not None:
            out[name] = ref
    return out


def _arm_endpoints(results: Mapping[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    stats = ("statistics", "arms", arm)
    summary = ("arms", arm, "summary")
    primary = _refs(results, {
        "natural_mix": (*stats, "complete_pass", "natural"),
        "natural_mix_cluster_bootstrap": (*stats, "complete_pass", "natural_cluster_bootstrap"),
        "unweighted": (*summary, "complete_pass"),
        "unweighted_cluster_bootstrap": (*stats, "complete_pass", "cluster_bootstrap"),
        "unweighted_wilson": (*stats, "complete_pass", "wilson"),
        "degenerate": (*stats, "complete_pass", "degenerate"),
    })
    primary["per_axis"] = {
        axis: _refs(results, {"unweighted": (*summary, "per_axis", axis, "complete_pass"),
                              "weighted": (*summary, "per_axis", axis,
                                           "complete_pass_weighted")})
        for axis in sorted(resolve(results, _pointer(*summary, "per_axis")))}
    secondary = {
        "components": _ref(results, *summary, "components"),
        "acceptable_rate": _refs(results, {
            "natural_mix": (*stats, "acceptable", "natural"),
            "natural_mix_cluster_bootstrap": (*stats, "acceptable", "natural_cluster_bootstrap"),
            "unweighted": (*summary, "acceptable"),
            "unweighted_cluster_bootstrap": (*stats, "acceptable", "cluster_bootstrap"),
            "unweighted_wilson": (*stats, "acceptable", "wilson"),
            "degenerate": (*stats, "acceptable", "degenerate"),
        }),
        "standard_action_agreement": _refs(results, {
            "rate": (*summary, "standard_action"),
            "table": ("arms", arm, "disagreement"),
        }),
        "per_policy_row": _ref(results, *summary, "per_referee_row"),
        "functional_invariance": _ref(results, *summary, "invariance"),
        "spend": _ref(results, "arms", arm, "spend"),
    }
    bound = (*stats, "complete_pass", "exact_clusters")
    return {"primary": primary,
            "secondary": {k: v for k, v in secondary.items() if v is not None},
            "diagnostics": {
                # the protocol's one-sided bound on the share of clusters with a failing
                # memo, with the counts it rests on (not the two-sided interval beside it)
                "failing_cluster_bound": _refs(results, {
                    "clusters": (*bound, "clusters"),
                    "clusters_with_failure": (*bound, "clusters_with_failure"),
                    "failure_share_upper_one_sided": (*bound, "failure_share_upper_one_sided"),
                    "level": (*bound, "level"),
                }),
                "against_simulation_truth": _ref(results, "arms", arm, "latent_diagnostic"),
            }}


def endpoints(results: Mapping[str, Any]) -> dict[str, Any]:
    """The protocol's endpoints, named as there, each pointing into ``results``."""
    arms = {arm: _arm_endpoints(results, arm) for arm in sorted(results["arms"])}
    paired = _ref(results, "statistics", "paired")
    if paired is not None and not paired["value"]:  # the harness leaves it empty
        paired = None
    return {
        "protocol": PROTOCOL,
        "primary": {"name": "complete-memo pass rate, per arm",
                    "reported_as": "natural_mix",
                    "arms": {arm: e["primary"] for arm, e in arms.items()}},
        "secondary": {
            "arms": {arm: e["secondary"] for arm, e in arms.items()},
            "paired_comparison": paired if paired is not None else {"not_made": NOT_PAIRED},
        },
        "diagnostics": {arm: e["diagnostics"] for arm, e in arms.items()},
    }


def case_memos(directory: Path, definition: Mapping[str, Any],
               results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per case of a ``cases`` benchmark, in its order: the case file and slot (its
    stratum is ``<file>:<slot>``), and for each scored arm the validated memo (``None``
    when the response failed) with its score."""
    records = {name: harness.cached_records(directory, definition, definition["arms"][name])
               for name in sorted(results["arms"])}
    out = []
    for case in definition["cases"]:
        case_id = case["case_id"]
        file, slot = case["stratum"].split(":", 1)
        arms = {}
        for name in records:
            parsed, problems = memo.parse(records[name][(case_id, "primary")]["text"])
            arms[name] = {"memo": parsed, "problems": problems,
                          "score": results["arms"][name]["cases"][f"{case_id}/primary"]}
        out.append({"case_id": case_id, "file": file, "slot": slot, "arms": arms})
    return out


def scored_arms(directory: Path, definition: Mapping[str, Any]) -> list[str]:
    """The benchmark's arms that have run (a pinned CLI), in its order."""
    pins = harness.read_pins(directory)
    return [name for name in definition["arms"] if name in pins]


def build(directory: Path, *, amend: str | None = None) -> dict[str, Any]:
    """The results of every arm that has run, with the endpoints named."""
    definition = harness.load_benchmark(directory)
    arms = scored_arms(directory, definition)
    if not arms:
        raise harness.CoverageError(f"{directory.name}: no arm has run")
    results = harness.evaluate(directory, amend=amend, arms=arms)
    results["endpoints"] = endpoints(results)
    if definition["phase"] == "cases":
        results["case_memos"] = case_memos(directory, definition, results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a benchmark's results.json.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--amend-scoring", metavar="REASON",
                        help="score although the scoring code changed, recording why")
    args = parser.parse_args(argv)
    directory = harness.BENCHMARKS / args.benchmark
    try:
        results = build(directory, amend=args.amend_scoring)
    except (harness.CoverageError, harness.FrozenError, harness.ShapeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    write(directory / "results.json", results)
    for name, value in results["endpoints"]["primary"]["arms"].items():
        print(name, json.dumps({k: v["value"] for k, v in value.items() if "value" in v}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
