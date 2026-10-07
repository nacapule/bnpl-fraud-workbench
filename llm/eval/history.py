"""Offline replay of the archived August 2026 memo study, with corrected statistics.

The study under ``benchmarks/2026-08-dev/original/`` drafted memos for 200 alerts
with four arms (Claude Sonnet 5 with prompts v1 and v2, GPT-5.6 Luna and Terra
with prompt v2) and stored every response. This module replays those stored
responses with the study's own protocol, needs no network, no model and no
database, and writes corrected statistics to ``corrected/``:

* ``attempt_chains.json``: for every arm, case and consistency probe, each
  attempt (its cache file, whether it was stored, the validator's problems) and
  the output finally accepted;
* ``statistics.json``: every number with its numerator and denominator;
* ``summary.md``: the same numbers as tables.

The historical protocol, reproduced exactly: the prompt is the template with
``{packet_json}`` replaced by the packet as JSON (indent 1, sorted keys); a
response is stored under ``sha256(model NUL prompt)[:32].json``; a reply that
fails parsing or validation is retried once with the problems appended, and the
retry is stored under its own prompt. The archived results replayed only first
attempts, so a stored, valid retry was ignored; here first-attempt and final
results are reported separately. Before anything else the replay rebuilds the
four archived result files from the same chains with the archived token check
and records whether each one is reproduced exactly.

Every accepted output is also checked strictly (every field present with its
type before any enumerated value; a benign hypothesis; non-empty prose). An
output that fails the strict check counts as a failure, never as a crash.

Facts that need the superseded world (all-time linkage, payments after the
alert, the category ratio's definition, story groups, identifier ranges) come
from ``corrected/world_checks.json``, written by ``--groups-from`` from that
world's CSV files and checked against the archived alerts hash; the replay only
reads it.

Usage::

    python -m llm.eval.history                    # regenerate corrected/
    python -m llm.eval.history --check            # fail unless corrected/ is current
    python -m llm.eval.history --groups-from data/   # recompute world_checks.json
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.results import Interval, Metric, StageResult, file_sha256
from core.stats import cluster_bootstrap, paired_outcomes, wilson_interval
from llm.eval import tokens

BENCH_DIR = Path(__file__).resolve().parent / "benchmarks" / "2026-08-dev"
GENERATED_FILES = ("attempt_chains.json", "statistics.json", "summary.md")
WORLD_CHECKS = "world_checks.json"

# config.yaml at the source commit called dates from here on the "holdout".
HOLDOUT_START = "2026-04-01"
PROBE_CASES, PROBE_RUNS = 50, 3
# The arms whose historical runs asked for consistency probes. Luna's run asked
# but none was stored (its archived result counts 50 cache misses); Terra's ran
# with --no-consistency.
PROBES_REQUESTED = frozenset(
    {"memo_v1__claude-sonnet-5", "memo_v2__claude-sonnet-5", "memo_v2__gpt-5.6-luna"}
)
SLUGS = {"claude-sonnet-5": "sonnet", "gpt-5.6-luna": "luna", "gpt-5.6-terra": "terra"}
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

# The archived memo contract (prompts memo_v1 and memo_v2).
ACTIONS = ("clear", "hold_contact", "decline_block", "escalate")
PRIORITIES = ("P0", "P1", "P2", "P3")
PATTERNS = (
    "account_takeover",
    "stolen_card",
    "synthetic_ring",
    "never_pay",
    "inr_abuse",
    "promo_abuse",
    "merchant_bustout",
    "benign",
)
LIKELIHOODS = ("low", "med", "high")
REQUIRED_FIELDS = (
    "signals_observed",
    "hypotheses",
    "policy_citations",
    "recommended_action",
    "priority",
    "evidence_gaps",
    "memo_markdown",
)
ENDPOINTS = ("first_attempt", "final")


class CoverageError(RuntimeError):
    """An arm lacks a stored response for a case it declares."""


# ----------------------------------------------------------------- the historical protocol
def render_prompt(template: str, packet: Mapping[str, Any]) -> str:
    return template.replace("{packet_json}", json.dumps(packet, indent=1, sort_keys=True))


def cache_file_name(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()[:32] + ".json"


def probe_packet(packet: Mapping[str, Any], index: int) -> dict[str, Any]:
    """The packet with the neutral line the study added for consistency probe ``index``."""
    probe = dict(packet)
    probe["_consistency_probe"] = f"probe {index + 1} (no informational content)"
    return probe


def retry_prompt(prompt: str, problems: Sequence[str]) -> str:
    return (
        prompt
        + "\n\nYour previous reply was rejected: "
        + "; ".join(problems)
        + "\nReturn ONLY the corrected JSON object."
    )


def parse_response(text: str) -> Any:
    """The study's tolerant parse: a fenced block, trailing commas and junk after the object."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body
        body = body.rsplit("```", 1)[0]
    start = body.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {body[:200]!r}")
    body = re.sub(r",\s*([}\]])", r"\1", body[start:])
    value, _ = json.JSONDecoder(strict=False).raw_decode(body)
    return value


def _member(value: Any, options: Sequence[str]) -> bool:
    return isinstance(value, str) and value in options


def historical_problems(memo: Any) -> list[str]:
    """The study's validator, message for message.

    The original raised ``TypeError`` on an unhashable action, priority or
    pattern; here such a value is reported as a problem like any other bad value.
    """
    if not isinstance(memo, dict):
        return ["memo must be an object"]
    problems = [f"missing field {name}" for name in REQUIRED_FIELDS if name not in memo]
    if problems:
        return problems
    if not _member(memo["recommended_action"], ACTIONS):
        problems.append(f"bad action {memo['recommended_action']!r}")
    if not _member(memo["priority"], PRIORITIES):
        problems.append(f"bad priority {memo['priority']!r}")
    if not isinstance(memo["signals_observed"], list) or not memo["signals_observed"]:
        problems.append("signals_observed empty")
    if not isinstance(memo["hypotheses"], list):
        problems.append("hypotheses must be a list")
    else:
        for hypothesis in memo["hypotheses"]:
            if not isinstance(hypothesis, dict):
                problems.append("each hypothesis must be an object")
                continue
            if not _member(hypothesis.get("pattern"), PATTERNS):
                problems.append(f"bad hypothesis pattern {hypothesis.get('pattern')!r}")
            if not _member(hypothesis.get("likelihood"), LIKELIHOODS):
                problems.append(f"bad likelihood {hypothesis.get('likelihood')!r}")
    citations = memo["policy_citations"]
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        problems.append("policy_citations must be a list of strings")
    if not isinstance(memo["memo_markdown"], str):
        problems.append("memo_markdown must be a string")
    return problems


def _strings(value: Any, *, non_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not non_empty)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def strict_problems(memo: Any) -> list[str]:
    """Full validation, types before enumerated values; never raises.

    Every field is present with its type; signals are a non-empty list of
    non-empty strings; hypotheses a non-empty list of objects, each with a known
    pattern, a likelihood and non-empty reasoning, one of them benign; citations
    and evidence gaps lists of strings; the memo text a non-empty string.
    """
    if not isinstance(memo, dict):
        return ["memo must be an object"]
    problems = [f"missing field {name}" for name in REQUIRED_FIELDS if name not in memo]
    if problems:
        return problems
    for name, options in (("recommended_action", ACTIONS), ("priority", PRIORITIES)):
        value = memo[name]
        if not isinstance(value, str):
            problems.append(f"{name} must be a string")
        elif value not in options:
            problems.append(f"{name} {value!r} is not one of {', '.join(options)}")
    if not _strings(memo["signals_observed"], non_empty=True):
        problems.append("signals_observed must be a non-empty list of non-empty strings")
    hypotheses = memo["hypotheses"]
    if not isinstance(hypotheses, list) or not hypotheses:
        problems.append("hypotheses must be a non-empty list")
    else:
        patterns = []
        for index, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, dict):
                problems.append(f"hypothesis {index} must be an object")
                continue
            for name, options in (("pattern", PATTERNS), ("likelihood", LIKELIHOODS)):
                value = hypothesis.get(name)
                if not isinstance(value, str):
                    problems.append(f"hypothesis {index} {name} must be a string")
                elif value not in options:
                    problems.append(f"hypothesis {index} {name} {value!r} is not allowed")
            reasoning = hypothesis.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                problems.append(f"hypothesis {index} reasoning must be a non-empty string")
            patterns.append(hypothesis.get("pattern"))
        if "benign" not in patterns:
            problems.append("no benign hypothesis")
    for name in ("policy_citations", "evidence_gaps"):
        if not _strings(memo[name], non_empty=False):
            problems.append(f"{name} must be a list of strings")
    text = memo["memo_markdown"]
    if not isinstance(text, str) or not text.strip():
        problems.append("memo_markdown must be a non-empty string")
    return problems


def top_hypothesis(memo: Mapping[str, Any]) -> str | None:
    """The most likely hypothesis's pattern; ties go to the one listed first."""
    order = {"high": 2, "med": 1, "low": 0}
    hypotheses = memo.get("hypotheses") or []
    if not hypotheses:
        return None
    _, best = max(
        enumerate(hypotheses),
        key=lambda item: (order.get(item[1].get("likelihood"), -1), -item[0]),
    )
    return best.get("pattern")


def fired_rules(packet: Mapping[str, Any]) -> set[str]:
    return {
        str(rule.get("id"))
        for rule in (packet.get("alert", {}).get("fired_rules") or [])
        if isinstance(rule, dict)
    }


def memo_texts(memo: Mapping[str, Any]) -> list[str]:
    """Every free text of a memo: signals, hypothesis reasoning, evidence gaps and prose."""
    texts = list(memo["signals_observed"])
    texts += [hypothesis["reasoning"] for hypothesis in memo["hypotheses"]]
    texts += list(memo["evidence_gaps"])
    texts.append(memo["memo_markdown"])
    return texts


# ----------------------------------------------------------------------------- archive
@dataclass(frozen=True)
class Arm:
    id: str
    model: str
    prompt_version: str
    backend: str
    case_limit: int
    result_file: str
    probes_requested: bool

    @property
    def slug(self) -> str:
        return f"{SLUGS[self.model]}_{self.prompt_version.removeprefix('memo_')}"


class Archive:
    """The archived study: cases, packets, prompts and the response cache."""

    def __init__(self, bench_dir: Path = BENCH_DIR):
        self.dir = Path(bench_dir)
        self.original = self.dir / "original"
        self.manifest = _read_json(self.dir / "MANIFEST.json")
        self.cases: list[dict[str, Any]] = _read_json(
            self.original / self.manifest["cases"]["file"]
        )
        self.packets = {
            case["alert_id"]: _read_json(self.original / "packets" / f"{case['alert_id']}.json")
            for case in self.cases
        }
        self.arms = tuple(
            Arm(
                id=Path(arm["result"]).stem,
                model=arm["model"],
                prompt_version=arm["prompt_version"],
                backend=arm["backend"],
                case_limit=arm["case_limit"],
                result_file=arm["result"],
                probes_requested=Path(arm["result"]).stem in PROBES_REQUESTED,
            )
            for arm in self.manifest["arms"]
        )
        self.templates = {
            arm.prompt_version: (self.original / "prompts" / f"{arm.prompt_version}.md").read_text()
            for arm in self.arms
        }
        self.cache_dir = self.original / "cache"
        self.cache_files = sorted(path.name for path in self.cache_dir.glob("*.json"))
        self._records: dict[str, Any] = {}

    def record(self, name: str) -> dict[str, Any] | None:
        if name not in self._records:
            path = self.cache_dir / name
            self._records[name] = _read_json(path) if path.exists() else None
        return self._records[name]

    def user(self, alert_id: int) -> int:
        return int(self.packets[alert_id]["alert"]["user_id"])


@dataclass(frozen=True)
class Attempt:
    cache_file: str
    present: bool
    problems: tuple[str, ...]
    duration_ms: int | None

    @property
    def valid(self) -> bool:
        return self.present and not self.problems


@dataclass(frozen=True)
class Chain:
    """One request's attempts and the output the historical protocol accepted."""

    alert_id: int
    variant: str  # "primary", "probe 1", "probe 2" or "probe 3"
    attempts: tuple[Attempt, ...]
    memo: dict[str, Any] | None  # accepted after any retry (the live protocol)
    first_memo: dict[str, Any] | None  # accepted at the first attempt (the archived replay)

    @property
    def cached(self) -> bool:
        return self.attempts[0].present

    @property
    def duration_ms(self) -> int:
        return sum(attempt.duration_ms or 0 for attempt in self.attempts if attempt.present)


@dataclass(frozen=True)
class ArmReplay:
    arm: Arm
    cases: tuple[dict[str, Any], ...]
    primary: tuple[Chain, ...]
    probes: tuple[tuple[Chain, ...], ...]  # the first PROBE_CASES cases, when requested


def _attempt(archive: Archive, model: str, prompt: str) -> tuple[Attempt, dict[str, Any] | None]:
    name = cache_file_name(model, prompt)
    record = archive.record(name)
    if record is None:
        return Attempt(name, False, (), None), None
    response = record.get("response") if isinstance(record, dict) else None
    response = response if isinstance(response, dict) else {}
    text = response.get("text")
    try:
        if not isinstance(text, str):
            raise ValueError("stored response has no text")
        memo = parse_response(text)
        problems = historical_problems(memo)
    except (ValueError, RecursionError) as error:  # ValueError includes JSONDecodeError
        memo, problems = None, [str(error) or type(error).__name__]
    duration = response.get("duration_ms")
    attempt = Attempt(name, True, tuple(problems), duration if isinstance(duration, int) else None)
    return attempt, (memo if attempt.valid else None)


def replay_chain(
    archive: Archive, arm: Arm, packet: Mapping[str, Any], alert_id: int, variant: str
) -> Chain:
    prompt = render_prompt(archive.templates[arm.prompt_version], packet)
    first, memo = _attempt(archive, arm.model, prompt)
    if not first.present or first.valid:
        return Chain(alert_id, variant, (first,), memo, memo)
    retry, memo = _attempt(archive, arm.model, retry_prompt(prompt, first.problems))
    return Chain(alert_id, variant, (first, retry), memo, None)


def replay_arm(archive: Archive, arm: Arm) -> ArmReplay:
    """Replay an arm's declared cases; a declared case without a stored response is an error."""
    cases = tuple(archive.cases[: arm.case_limit])
    primary = tuple(
        replay_chain(archive, arm, archive.packets[case["alert_id"]], case["alert_id"], "primary")
        for case in cases
    )
    missing = [chain.alert_id for chain in primary if not chain.cached]
    if missing:
        raise CoverageError(
            f"{arm.id}: {len(missing)} of {len(cases)} declared cases have no stored "
            f"response (first: alert {missing[0]})"
        )
    probes: tuple[tuple[Chain, ...], ...] = ()
    if arm.probes_requested:
        probes = tuple(
            tuple(
                replay_chain(
                    archive,
                    arm,
                    probe_packet(archive.packets[case["alert_id"]], index),
                    case["alert_id"],
                    f"probe {index + 1}",
                )
                for index in range(PROBE_RUNS)
            )
            for case in cases[:PROBE_CASES]
        )
    return ArmReplay(arm, cases, primary, probes)


# ------------------------------------------------------------- the archived result format
def _archived_scorable(memo: Mapping[str, Any]) -> bool:
    # The archived token check raised TypeError on a non-string signal.
    return all(isinstance(signal, str) for signal in memo["signals_observed"])


def archived_result(archive: Archive, replay: ArmReplay) -> dict[str, Any]:
    """The arm's result file as the archived harness wrote it (first attempts only)."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case, chain in zip(replay.cases, replay.primary, strict=True):
        memo = chain.first_memo
        if memo is None or not _archived_scorable(memo):
            failures.append(case)
            continue
        packet = archive.packets[case["alert_id"]]
        acceptable = case.get("truth_actions", [case["truth_action"]])
        checks = tokens.archived_check_texts(memo["signals_observed"], packet)
        derived = sum(check.classification == "derived" for check in checks)
        unsupported = sum(check.classification == "unmatched" for check in checks)
        citations = tokens.check_citations(
            memo["policy_citations"], tokens.FP1_RULE_IDS, fired_rules(packet)
        )
        with_id = [citation for citation in citations if citation["rule_id"]]
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
                "has_unsupported_claim": unsupported > 0,
                "has_non_verbatim_claim": derived + unsupported > 0,
                "n_checked_fields": len(checks),
                "n_verbatim": len(checks) - derived - unsupported,
                "n_derived": derived,
                "n_unsupported": unsupported,
                "n_invalid_citations": sum(citation["valid"] is False for citation in citations),
                "citation_fired_rate": (
                    sum(bool(citation["fired"]) for citation in with_id) / len(with_id)
                    if with_id
                    else None
                ),
                "duration_ms": chain.attempts[0].duration_ms,
                "cached": True,
            }
        )

    agreements: list[bool] = []
    probe_misses = probe_hits = probe_attempts = 0
    for chains in replay.probes:
        actions: list[str] = []
        for chain in chains:
            probe_attempts += 1
            if not chain.cached:
                probe_misses += 1
                actions = []
                break
            if chain.first_memo is None:
                actions = []
                break
            actions.append(chain.first_memo["recommended_action"])
            probe_hits += 1
        if len(actions) == PROBE_RUNS:
            agreements.append(len(set(actions)) == 1)

    valid_metrics = _archived_decisions(rows, failures, include_failures=False)
    inclusive_metrics = _archived_decisions(rows, failures, include_failures=True)
    n_valid = len(rows)
    n_attempted = n_valid + len(failures)
    checked = sum(row["n_checked_fields"] for row in rows)
    unsupported = sum(row["n_unsupported"] for row in rows)
    derived = sum(row["n_derived"] for row in rows)
    fired_rates = [
        row["citation_fired_rate"] for row in rows if row["citation_fired_rate"] is not None
    ]
    per_pattern = {}
    for pattern in sorted({row["truth_pattern"] for row in rows}):
        subset = [row for row in rows if row["truth_pattern"] == pattern]
        per_pattern[pattern] = {
            "n": len(subset),
            "action_accuracy": round(sum(row["action_ok"] for row in subset) / len(subset), 3),
            "pattern_id_rate": round(
                sum(row["top_pattern"] == row["truth_pattern"] for row in subset) / len(subset),
                3,
            ),
        }
    opportunities = n_attempted + probe_attempts
    return {
        "model": replay.arm.model,
        "prompt_version": replay.arm.prompt_version,
        "n_cases": n_valid,
        "n_cases_requested": len(replay.cases),
        "cache_misses": probe_misses,
        "schema_failures": len(failures),
        "schema_failure_rate": round(len(failures) / n_attempted, 4) if n_attempted else None,
        "action_accuracy": valid_metrics["action_accuracy"],
        "decline_precision": valid_metrics["decline_precision"],
        "decline_recall": valid_metrics["decline_recall"],
        "decision_metrics_valid_only": valid_metrics,
        "decision_metrics_including_schema_failures": inclusive_metrics,
        "pattern_id_rate": (
            round(sum(row["top_pattern"] == row["truth_pattern"] for row in rows) / n_valid, 3)
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
            round((unsupported + derived) / checked, 4) if checked else None
        ),
        "unsupported_claim_rate_checked_fields": (
            round(unsupported / checked, 4) if checked else None
        ),
        "derived_claim_rate_checked_fields": round(derived / checked, 4) if checked else None,
        "grounding_metric_scope": (
            "signals_observed concrete numeric/id/timestamp/money tokens; not semantic truth"
        ),
        "invalid_citation_memos": sum(row["n_invalid_citations"] > 0 for row in rows),
        "citation_fired_rate_mean": (
            round(statistics.mean(fired_rates), 3) if fired_rates else None
        ),
        "consistency_action_agreement": (
            round(sum(agreements) / len(agreements), 3) if agreements else None
        ),
        "consistency_n": len(agreements),
        "latency_p50_ms": (
            int(statistics.median(row["duration_ms"] for row in rows)) if rows else None
        ),
        "cache_hit_rate": (
            round((n_attempted + probe_hits) / opportunities, 3) if opportunities else None
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


def _archived_decisions(
    rows: list[dict[str, Any]], failures: list[dict[str, Any]], *, include_failures: bool
) -> dict[str, Any]:
    decline = "decline_block"
    denominator = len(rows) + (len(failures) if include_failures else 0)
    tp = sum(row["action"] == decline and row["truth_action"] == decline for row in rows)
    fp = sum(row["action"] == decline and row["truth_action"] != decline for row in rows)
    fn = sum(row["action"] != decline and row["truth_action"] == decline for row in rows)
    if include_failures:
        fn += sum(case["truth_action"] == decline for case in failures)
    return {
        "n": denominator,
        "action_accuracy": (
            round(sum(row["action_ok"] for row in rows) / denominator, 3) if denominator else None
        ),
        "decline_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "decline_recall": round(tp / (tp + fn), 3) if tp + fn else None,
    }


# ------------------------------------------------------------------ corrected statistics
def _num(value: float | int) -> float | int:
    """Ints stay ints; floats keep six significant digits, so the bytes are stable."""
    if isinstance(value, bool) or isinstance(value, int | np.integer):
        return int(value)
    return float(f"{float(value):.6g}")


def _ratio(numerator: int, denominator: int, *, wilson: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": _num(numerator / denominator) if denominator else None,
    }
    if wilson and denominator:
        interval = wilson_interval(int(numerator), int(denominator))
        out["wilson_95"] = [_num(interval.low), _num(interval.high)]
    return out


def _bootstrap(clusters: Sequence[Any], statistic: Callable[[np.ndarray], float]) -> list[float]:
    interval = cluster_bootstrap(
        list(clusters), statistic, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )
    return [_num(interval.low), _num(interval.high)]


_NO_TEXTS = {"n_texts": 0, "n_with_tokens": 0, "n_derived": 0, "n_unmatched": 0}


@dataclass(frozen=True)
class Scored:
    """One case at one endpoint; a case without an accepted output keeps the defaults."""

    alert_id: int
    truth_decline: bool
    duration_ms: int
    accepted: bool = False  # an output passed the historical and the strict validator
    strict_rejected: bool = False  # the historical protocol accepted it, the strict check not
    correct: bool = False
    decline: bool = False
    pattern_named: bool = False
    tokens_archived: Mapping[str, int] = field(default_factory=lambda: _NO_TEXTS)  # signals
    tokens_corrected: Mapping[str, int] = field(default_factory=lambda: _NO_TEXTS)  # signals
    tokens_all_text: Mapping[str, int] = field(default_factory=lambda: _NO_TEXTS)  # every text
    invalid_citations: int = 0
    citations_without_rule_id: int = 0
    signal_changes: int = 0  # signals classified differently by the corrected check


def _score(archive: Archive, case: Mapping[str, Any], chain: Chain, endpoint: str) -> Scored:
    memo = chain.first_memo if endpoint == "first_attempt" else chain.memo
    duration = (
        (chain.attempts[0].duration_ms or 0) if endpoint == "first_attempt" else (chain.duration_ms)
    )
    truth_decline = case["truth_action"] == "decline_block"
    if memo is None or strict_problems(memo):
        return Scored(case["alert_id"], truth_decline, duration, strict_rejected=memo is not None)
    packet = archive.packets[case["alert_id"]]
    signals = memo["signals_observed"]
    archived = tokens.archived_check_texts(signals, packet)
    corrected = tokens.check_texts(signals, packet)
    citations = tokens.check_citations(
        memo["policy_citations"], tokens.FP1_RULE_IDS, fired_rules(packet)
    )
    action = memo["recommended_action"]
    return Scored(
        case["alert_id"],
        truth_decline,
        duration,
        accepted=True,
        correct=action in case.get("truth_actions", [case["truth_action"]]),
        decline=action == "decline_block",
        pattern_named=top_hypothesis(memo) == case["truth_pattern"],
        tokens_archived=tokens.summarize(archived),
        tokens_corrected=tokens.summarize(corrected),
        tokens_all_text=tokens.summarize(tokens.check_texts(memo_texts(memo), packet)),
        invalid_citations=sum(citation["valid"] is False for citation in citations),
        citations_without_rule_id=sum(citation["rule_id"] is None for citation in citations),
        signal_changes=sum(
            (old.classification, old.missing) != (new.classification, new.missing)
            for old, new in zip(archived, corrected, strict=True)
        ),
    )


def score_arm(archive: Archive, replay: ArmReplay) -> dict[str, list[Scored]]:
    """Each declared case of an arm scored at the first attempt and at the final output."""
    first, final = [], []
    for case, chain in zip(replay.cases, replay.primary, strict=True):
        final.append(_score(archive, case, chain, "final"))
        # A chain accepted at its first attempt scores the same at both endpoints.
        single = len(chain.attempts) == 1
        first.append(final[-1] if single else _score(archive, case, chain, "first_attempt"))
    return {"first_attempt": first, "final": final}


def _endpoint_statistics(scored: list[Scored]) -> dict[str, Any]:
    cases = len(scored)
    accepted = [row for row in scored if row.accepted]
    valid = len(accepted)
    correct = sum(row.correct for row in accepted)
    tp = sum(row.decline and row.truth_decline for row in accepted)
    declines = sum(row.decline for row in accepted)
    truth_declines = sum(row.truth_decline for row in scored)
    truth_declines_valid = sum(row.truth_decline for row in accepted)
    fields = sum(row.tokens_corrected["n_texts"] for row in accepted)
    durations = [row.duration_ms for row in accepted]
    return {
        "cases": cases,
        "valid_outputs": _ratio(valid, cases),
        "failures": [row.alert_id for row in scored if not row.accepted],
        "strict_rejections": [row.alert_id for row in scored if row.strict_rejected],
        "action_correct": _ratio(correct, cases, wilson=True),
        "action_correct_valid_only": _ratio(correct, valid, wilson=True),
        "decline_precision": _ratio(tp, declines),
        "decline_recall": _ratio(tp, truth_declines),
        "decline_recall_valid_only": _ratio(tp, truth_declines_valid),
        "pattern_named": _ratio(sum(row.pattern_named for row in accepted), valid),
        "unmatched_token_memos": _ratio(
            sum(row.tokens_corrected["n_unmatched"] > 0 for row in accepted), valid
        ),
        "unmatched_token_memos_archived_check": _ratio(
            sum(row.tokens_archived["n_unmatched"] > 0 for row in accepted), valid
        ),
        "non_verbatim_memos_archived_check": _ratio(
            sum(
                row.tokens_archived["n_unmatched"] + row.tokens_archived["n_derived"] > 0
                for row in accepted
            ),
            valid,
        ),
        "unmatched_token_signals": _ratio(
            sum(row.tokens_corrected["n_unmatched"] for row in accepted), fields
        ),
        "derived_token_signals": _ratio(
            sum(row.tokens_corrected["n_derived"] for row in accepted), fields
        ),
        "signals_without_tokens": _ratio(
            sum(
                row.tokens_corrected["n_texts"] - row.tokens_corrected["n_with_tokens"]
                for row in accepted
            ),
            fields,
        ),
        "unmatched_token_memos_any_text": _ratio(
            sum(row.tokens_all_text["n_unmatched"] > 0 for row in accepted), valid
        ),
        "signals_changed_by_token_fixes": sum(row.signal_changes for row in accepted),
        "invalid_rule_citation_memos": sum(row.invalid_citations > 0 for row in accepted),
        "citations_without_rule_id": sum(row.citations_without_rule_id for row in accepted),
        "latency_p50_ms": int(statistics.median(durations)) if durations else None,
    }


def _arm_statistics(
    archive: Archive, replay: ArmReplay, scored: dict[str, list[Scored]]
) -> dict[str, Any]:
    retried = [chain for chain in replay.primary if len(chain.attempts) > 1]
    final = {row.alert_id: row for row in scored["final"]}
    per_pattern: dict[str, dict[str, Any]] = {}
    for pattern in sorted({case["truth_pattern"] for case in replay.cases}):
        rows = [
            final[case["alert_id"]] for case in replay.cases if case["truth_pattern"] == pattern
        ]
        per_pattern[pattern] = {
            "action_correct": _ratio(sum(row.correct for row in rows), len(rows)),
            "pattern_named": _ratio(sum(row.pattern_named for row in rows), len(rows)),
        }
    beyond = len(_cached_beyond_limit(archive, replay.arm))
    return {
        "model": replay.arm.model,
        "prompt_version": replay.arm.prompt_version,
        "backend": replay.arm.backend,
        "cases": len(replay.cases),
        "case_set": f"the first {len(replay.cases)} cases of cases.json",
        "first_attempt": _endpoint_statistics(scored["first_attempt"]),
        "final": _endpoint_statistics(scored["final"]),
        "retries": {
            "cases": [chain.alert_id for chain in retried],
            "stored": sum(chain.attempts[1].present for chain in retried),
            "valid": sum(chain.attempts[1].valid for chain in retried),
        },
        "final_per_pattern": per_pattern,
        "primary_responses_beyond_case_limit": beyond,
    }


def _cached_beyond_limit(archive: Archive, arm: Arm) -> list[str]:
    """Stored first responses for cases after the arm's declared limit (not scored)."""
    template = archive.templates[arm.prompt_version]
    names = (
        cache_file_name(arm.model, render_prompt(template, archive.packets[case["alert_id"]]))
        for case in archive.cases[arm.case_limit :]
    )
    return [name for name in names if archive.record(name) is not None]


def _probe_sets(replay: ArmReplay) -> dict[int, dict[str, Any]]:
    """Per probe case, with final outputs: the original action and the three probe actions."""
    sets = {}
    for chain, probes in zip(replay.primary, replay.probes, strict=False):
        accepted = [probe.memo is not None and not strict_problems(probe.memo) for probe in probes]
        original_ok = chain.memo is not None and not strict_problems(chain.memo)
        sets[chain.alert_id] = {
            "complete": all(accepted) and original_ok,
            "original": chain.memo["recommended_action"] if original_ok else None,
            "probes": [
                probe.memo["recommended_action"] if ok else None
                for probe, ok in zip(probes, accepted, strict=True)
            ],
        }
    return sets


def _consistency(replays: dict[str, ArmReplay]) -> dict[str, Any]:
    out: dict[str, Any] = {"arms": {}}
    complete_by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for arm_id, replay in sorted(replays.items()):
        if not replay.arm.probes_requested:
            continue
        sets = _probe_sets(replay)
        complete = {alert: item for alert, item in sets.items() if item["complete"]}
        complete_by_arm[arm_id] = complete
        agree = [alert for alert, item in complete.items() if len(set(item["probes"])) == 1]
        with_original = [
            alert
            for alert, item in complete.items()
            if all(action == item["original"] for action in item["probes"])
        ]
        out["arms"][arm_id] = {
            "probe_cases": len(sets),
            "complete_probe_sets": _ratio(len(complete), len(sets)),
            "incomplete_cases": sorted(alert for alert in sets if alert not in complete),
            "probe_agreement": _ratio(len(agree), len(complete), wilson=True),
            "agreement_with_original": _ratio(len(with_original), len(complete), wilson=True),
            "unanimous_probes_differing_from_original": sorted(set(agree) - set(with_original)),
        }
    prompts = ("memo_v1__claude-sonnet-5", "memo_v2__claude-sonnet-5")
    if all(arm_id in complete_by_arm for arm_id in prompts):
        (first_id, first), (second_id, second) = ((a, complete_by_arm[a]) for a in prompts)
        common = sorted(set(first) & set(second))
        out["common_complete_cases"] = {
            "cases": len(common),
            "arms_compared": list(prompts),
            "arms": {
                arm_id: {
                    "probe_agreement": _ratio(
                        sum(len(set(sets[alert]["probes"])) == 1 for alert in common), len(common)
                    ),
                    "agreement_with_original": _ratio(
                        sum(
                            all(a == sets[alert]["original"] for a in sets[alert]["probes"])
                            for alert in common
                        ),
                        len(common),
                    ),
                }
                for arm_id, sets in ((first_id, first), (second_id, second))
            },
        }
    return out


def _paired(
    archive: Archive,
    groups: Mapping[str, str],
    first: Mapping[int, bool],
    second: Mapping[int, bool],
    *,
    first_arm: str,
    second_arm: str,
    outcome: str,
    case_set: str,
) -> dict[str, Any]:
    ids = sorted(first)
    pair = paired_outcomes({i: bool(first[i]) for i in ids}, {i: bool(second[i]) for i in ids})
    a = np.array([first[i] for i in ids], dtype=float)
    b = np.array([second[i] for i in ids], dtype=float)
    clusterings = {
        "user": [f"user {archive.user(i)}" for i in ids],
        "user_or_story": [groups[str(i)] for i in ids],
    }
    first_rate = _ratio(pair.both + pair.only_first, pair.n, wilson=True)
    second_rate = _ratio(pair.both + pair.only_second, pair.n, wilson=True)
    difference: dict[str, Any] = {
        "numerator": pair.only_second - pair.only_first,
        "denominator": pair.n,
        "value": _num((pair.only_second - pair.only_first) / pair.n),
    }
    for name, clusters in clusterings.items():
        first_rate[f"{name}_bootstrap_95"] = _bootstrap(clusters, lambda pos: a[pos].mean())
        second_rate[f"{name}_bootstrap_95"] = _bootstrap(clusters, lambda pos: b[pos].mean())
        difference[f"{name}_bootstrap_95"] = _bootstrap(
            clusters, lambda pos: b[pos].mean() - a[pos].mean()
        )
        difference[f"{name}_clusters"] = len(set(clusters))
    return {
        "first": first_arm,
        "second": second_arm,
        "outcome": outcome,
        "case_set": case_set,
        "cases": pair.n,
        "both": pair.both,
        "only_first": pair.only_first,
        "only_second": pair.only_second,
        "neither": pair.neither,
        "exact_mcnemar_p": _num(pair.p_value),
        "first_rate": first_rate,
        "second_rate": second_rate,
        "difference_second_minus_first": difference,
    }


def _comparisons(
    archive: Archive, scored: dict[str, dict[str, list[Scored]]], groups: Mapping[str, str]
) -> dict[str, Any]:
    def final(arm_id: str, value: Callable[[Scored], bool], ids: Iterable[int] | None = None):
        rows = {row.alert_id: row for row in scored[arm_id]["final"]}
        keep = rows if ids is None else ids
        return {alert: value(rows[alert]) for alert in keep}

    def clean(row: Scored) -> bool:
        return row.accepted and row.tokens_corrected["n_unmatched"] == 0

    def clean_archived(row: Scored) -> bool:
        return row.accepted and row.tokens_archived["n_unmatched"] == 0

    def correct(row: Scored) -> bool:
        return row.correct

    v1, v2 = "memo_v1__claude-sonnet-5", "memo_v2__claude-sonnet-5"
    luna, terra = "memo_v2__gpt-5.6-luna", "memo_v2__gpt-5.6-terra"
    terra_ids = [row.alert_id for row in scored[terra]["final"]]
    return {
        "prompt_v1_vs_v2_action": _paired(
            archive,
            groups,
            final(v1, correct),
            final(v2, correct),
            first_arm=v1,
            second_arm=v2,
            outcome="action in the acceptable set",
            case_set="all 200 cases; final outputs",
        ),
        "prompt_v1_vs_v2_unmatched_tokens": _paired(
            archive,
            groups,
            final(v1, clean),
            final(v2, clean),
            first_arm=v1,
            second_arm=v2,
            outcome="no unmatched token in signals_observed (a memo with one fails)",
            case_set="all 200 cases; final outputs",
        ),
        "prompt_v1_vs_v2_unmatched_tokens_archived_check": _paired(
            archive,
            groups,
            final(v1, clean_archived),
            final(v2, clean_archived),
            first_arm=v1,
            second_arm=v2,
            outcome="no unmatched token in signals_observed under the archived token check",
            case_set="all 200 cases; final outputs",
        ),
        "sonnet_vs_luna_action": _paired(
            archive,
            groups,
            final(v2, correct),
            final(luna, correct),
            first_arm=v2,
            second_arm=luna,
            outcome="action in the acceptable set",
            case_set="all 200 cases; final outputs (Luna's stored retry for alert 6188 included)",
        ),
        "sonnet_vs_terra_action": _paired(
            archive,
            groups,
            final(v2, correct, terra_ids),
            final(terra, correct),
            first_arm=v2,
            second_arm=terra,
            outcome="action in the acceptable set",
            case_set=f"Terra's {len(terra_ids)} cases (the first {len(terra_ids)} of cases.json)",
        ),
    }


def _case_set(archive: Archive, world: Mapping[str, Any]) -> dict[str, Any]:
    cases = archive.cases
    packets = [archive.packets[case["alert_id"]] for case in cases]
    dates = sorted(packet["alert"]["ts"][:10] for packet in packets)
    users = Counter(int(packet["alert"]["user_id"]) for packet in packets)
    strata = Counter(case["truth_pattern"] for case in cases)
    benign_tenure = [
        packet["account"]["tenure_days"]
        for case, packet in zip(cases, packets, strict=True)
        if case["truth_pattern"] == "benign"
    ]
    terra = next(arm for arm in archive.arms if arm.model == "gpt-5.6-terra")
    prefix = cases[: terra.case_limit]
    return {
        "description": "development set drawn from all dates",
        "cases": len(cases),
        "alert_dates": {"first": dates[0], "last": dates[-1]},
        "cutoff": HOLDOUT_START,
        "before_cutoff": _ratio(sum(date < HOLDOUT_START for date in dates), len(cases)),
        "users": len(users),
        "users_with_two_or_more_cases": sum(count > 1 for count in users.values()),
        "labelled_fraud": _ratio(len(cases) - strata["benign"], len(cases)),
        "strata": dict(sorted(strata.items())),
        "benign_min_tenure_days": min(benign_tenure),
        "user_or_story_groups": world["summary"]["user_or_story_groups"],
        "stories_by_pattern": world["summary"]["stories_by_pattern"],
        "labelled_fraud_among_all_alerts": world["summary"]["labelled_fraud_among_all_alerts"],
        "terra_prefix": {
            "cases": len(prefix),
            "strata": dict(sorted(Counter(case["truth_pattern"] for case in prefix).items())),
            "before_cutoff": _ratio(
                sum(
                    archive.packets[c["alert_id"]]["alert"]["ts"][:10] < HOLDOUT_START
                    for c in prefix
                ),
                len(prefix),
            ),
        },
    }


def _id_ranges(cases: Sequence[Mapping[str, Any]], order_ids: Mapping[int, int]) -> dict[str, Any]:
    """Order-id ranges spanned by each labelled pattern's selected cases.

    A pattern's range is pure when no selected case with another label falls
    inside it; the cases of pure ranges are then labelled by their id alone.
    """
    by_pattern: dict[str, list[int]] = {}
    for case in cases:
        if case["truth_pattern"] != "benign":
            by_pattern.setdefault(case["truth_pattern"], []).append(order_ids[case["alert_id"]])
    ranges = {}
    for pattern, ids in sorted(by_pattern.items()):
        low, high = min(ids), max(ids)
        others = sum(
            low <= order_ids[case["alert_id"]] <= high
            for case in cases
            if case["truth_pattern"] != pattern
        )
        ranges[pattern] = {
            "low": low,
            "high": high,
            "cases": len(ids),
            "other_cases_inside": others,
        }
    pure = [pattern for pattern, item in ranges.items() if item["other_cases_inside"] == 0]
    covered = sum(ranges[pattern]["cases"] for pattern in pure)
    return {
        "ranges": ranges,
        "pure_patterns": pure,
        "cases_labelled_by_id_range": _ratio(
            covered, len(cases) - sum(case["truth_pattern"] == "benign" for case in cases)
        ),
    }


def _limitations(archive: Archive, world: Mapping[str, Any]) -> dict[str, Any]:
    negative = sorted(
        alert for alert, packet in archive.packets.items() if packet["account"]["tenure_days"] < 0
    )
    r06 = []
    for alert, packet in sorted(archive.packets.items()):
        for rule in packet["alert"]["fired_rules"]:
            match = re.search(r"shared_root_accounts=(\d+)", rule.get("rationale", ""))
            if rule.get("id") == "R06" and match:
                r06.append(
                    (
                        alert,
                        int(match.group(1)),
                        packet["linkage"]["other_accounts_same_email_root"],
                    )
                )
    order_ids = {
        alert: int(packet["alert"]["order_id"]) for alert, packet in archive.packets.items()
    }
    world_facts = (
        "future_linkage_packets",
        "payments_after_alert",
        "category_ratio",
        "r06_rationale_future_accounts",
        "order_id_ranges_in_world",
        "packet_reconstruction",
    )
    return {
        **{key: world["summary"][key] for key in world_facts},
        "negative_tenure_packets": {"count": len(negative), "alert_ids": negative},
        "r06_rationale_vs_packet_linkage": {
            "packets_with_r06": len(r06),
            "rationale_differs_from_other_accounts_same_email_root": sum(
                count != other for _, count, other in r06
            ),
            "rationale_equals_other_accounts_plus_one": sum(
                count == other + 1 for _, count, other in r06
            ),
        },
        "order_id_ranges_in_sample": _id_ranges(archive.cases, order_ids),
    }


def _replay_integrity(
    archive: Archive, replays: dict[str, ArmReplay], reproduced: dict[str, bool]
) -> dict[str, Any]:
    used: set[str] = set()
    beyond: set[str] = set()
    for replay in replays.values():
        for chain in (*replay.primary, *(c for probes in replay.probes for c in probes)):
            used.update(attempt.cache_file for attempt in chain.attempts if attempt.present)
        beyond.update(_cached_beyond_limit(archive, replay.arm))
    stored = set(archive.cache_files)
    accepted = [
        chain.memo
        for replay in replays.values()
        for chain in (*replay.primary, *(c for probes in replay.probes for c in probes))
        if chain.memo is not None
    ]
    return {
        "strict_validation": {
            "accepted_outputs": len(accepted),
            "rejected": sum(bool(strict_problems(memo)) for memo in accepted),
        },
        "cache_files": len(stored),
        "cache_files_in_chains": len(used & stored),
        "cache_files_beyond_case_limits": len(beyond - used),
        "cache_files_matching_no_protocol_prompt": len(stored - used - beyond),
        "archived_results_reproduced": reproduced,
    }


def build(bench_dir: Path = BENCH_DIR, world_checks: Path | None = None) -> dict[str, str]:
    """Replay the archive and return the generated files' contents by name."""
    archive = Archive(bench_dir)
    world_path = Path(world_checks) if world_checks else archive.dir / "corrected" / WORLD_CHECKS
    if not world_path.exists():
        raise FileNotFoundError(
            f"{world_path} is missing; run python -m llm.eval.history --groups-from <data dir>"
        )
    world = _read_json(world_path)
    if world["alerts_csv_sha256"] != archive.manifest["world"]["alerts_csv_sha256"]:
        raise ValueError(f"{world_path} was computed on a different world than the archive's")

    replays = {arm.id: replay_arm(archive, arm) for arm in archive.arms}
    reproduced = {}
    for arm in archive.arms:
        expected = _read_json(archive.original / arm.result_file)
        reproduced[arm.id] = archived_result(archive, replays[arm.id]) == expected
    scored = {arm_id: score_arm(archive, replay) for arm_id, replay in replays.items()}
    stats = {
        "benchmark": archive.manifest["benchmark"],
        "source_commit": archive.manifest["source_commit"],
        "replay": _replay_integrity(archive, replays, reproduced),
        "arms": {
            arm_id: _arm_statistics(archive, replays[arm_id], scored[arm_id])
            for arm_id in sorted(replays)
        },
        "consistency": _consistency(replays),
        "paired": _comparisons(archive, scored, world["groups"]),
        "case_set": _case_set(archive, world),
        "limitations": _limitations(archive, world),
        "definitions": DEFINITIONS,
    }
    chains = _chains_document(replays)
    return {
        "attempt_chains.json": _dump(chains),
        "statistics.json": _dump(stats),
        "summary.md": render_summary(stats),
    }


DEFINITIONS = {
    "action_correct": (
        "the recommended action is in the case's acceptable-action set from the archived "
        "cases.json (a referee that maps the simulator's hidden pattern to actions)"
    ),
    "decline_precision_recall": (
        "against the referee's primary action (the first of the acceptable set), as archived"
    ),
    "first_attempt": "the first stored response only, as the archived results were replayed",
    "final": "the output accepted after the study's single retry, when a retry was stored",
    "valid_outputs": "outputs that pass the study's validator and the strict validation",
    "unmatched_token_memos": (
        "memos with at least one signals_observed entry whose concrete tokens the packet "
        "does not contain and that is not a count or sum over a named packet list; a token "
        "check, not semantic truth"
    ),
    "unmatched_token_memos_any_text": (
        "the same check over every memo text: signals, hypothesis reasoning, evidence gaps "
        "and the memo prose; thresholds quoted from the prompt (such as $2,000 and $500) "
        "count as unmatched, since they are not packet facts"
    ),
    "probe_agreement": (
        "share of complete probe sets (three probes, each with a valid output) whose three "
        "actions agree"
    ),
    "agreement_with_original": (
        "share of complete probe sets whose three actions all equal the original memo's action"
    ),
    "user_bootstrap_95": "cluster bootstrap over users (2,000 resamples, seed 0)",
    "user_or_story_bootstrap_95": (
        "cluster bootstrap over injected stories, or users outside any story (2,000 "
        "resamples, seed 0)"
    ),
    "wilson_95": "Wilson score interval, treating cases as independent",
}


def _chains_document(replays: dict[str, ArmReplay]) -> dict[str, Any]:
    def chain_entry(chain: Chain) -> dict[str, Any]:
        final = None
        if chain.memo is not None:
            final = {
                "cache_file": chain.attempts[-1].cache_file,
                "recommended_action": chain.memo["recommended_action"],
                "strict_problems": strict_problems(chain.memo),
            }
        return {
            "alert_id": chain.alert_id,
            "variant": chain.variant,
            "attempts": [
                {
                    "cache_file": attempt.cache_file,
                    "stored": attempt.present,
                    "valid": attempt.valid,
                    "problems": list(attempt.problems),
                    "duration_ms": attempt.duration_ms,
                }
                for attempt in chain.attempts
            ],
            "final": final,
        }

    arms = {}
    for arm_id, replay in sorted(replays.items()):
        chains = list(replay.primary) + [chain for probes in replay.probes for chain in probes]
        arms[arm_id] = {
            "model": replay.arm.model,
            "prompt_version": replay.arm.prompt_version,
            "case_limit": replay.arm.case_limit,
            "probes": (
                {"cases": len(replay.probes), "runs": PROBE_RUNS} if replay.probes else None
            ),
            "chains": [chain_entry(chain) for chain in chains],
        }
    return {
        "protocol": (
            "attempt 1 is the stored response to the rendered prompt; when it fails parsing "
            "or validation, attempt 2 is the stored response to the retry prompt (the "
            "problems appended); final is the output the study's protocol accepted, null "
            "when none was"
        ),
        "arms": arms,
    }


# ------------------------------------------------------------------------- summary.md
def _pct(item: Mapping[str, Any]) -> str:
    if item["value"] is None:
        return f"{item['numerator']}/{item['denominator']} (not evaluated)"
    return f"{item['numerator']}/{item['denominator']} ({100 * item['value']:.1f}%)"


def _interval(bounds: Sequence[float] | None) -> str:
    if not bounds:
        return "–"
    return f"{100 * bounds[0]:.1f}–{100 * bounds[1]:.1f}%"


def _seconds(milliseconds: int | None) -> str:
    return "–" if milliseconds is None else f"{milliseconds / 1000:.1f} s"


def _pct_ci(item: Mapping[str, Any]) -> str:
    return f"{_pct(item)}, {_interval(item.get('wilson_95'))}"


def _pp(bounds: Sequence[float]) -> str:
    return f"{100 * bounds[0]:+.1f} to {100 * bounds[1]:+.1f} pp"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def render_summary(stats: Mapping[str, Any]) -> str:
    arms = stats["arms"]
    lines = [
        "# August 2026 memo study: corrected statistics",
        "",
        "Generated by `python -m llm.eval.history` from the responses stored in `original/`; "
        "every number is in `statistics.json` with its numerator and denominator. "
        "`HISTORY.md` explains what the study can and cannot show.",
        "",
        "## Arms: first attempt and final output",
        "",
        "First attempt counts only the first stored response, as the archived results did; "
        "final includes the study's single retry where one was stored. Failures count as wrong.",
        "",
    ]
    lines += _table(
        (
            "arm",
            "cases",
            "valid, first",
            "valid, final",
            "action correct, first",
            "action correct, final",
            "95% Wilson, final",
        ),
        (
            (
                arm_id,
                arm["cases"],
                _pct(arm["first_attempt"]["valid_outputs"]),
                _pct(arm["final"]["valid_outputs"]),
                _pct(arm["first_attempt"]["action_correct"]),
                _pct(arm["final"]["action_correct"]),
                _interval(arm["final"]["action_correct"].get("wilson_95")),
            )
            for arm_id, arm in arms.items()
        ),
    )
    lines += [
        "Final outputs. Decline precision and recall use the referee's primary action; "
        "recall counts every case whose primary action is decline. Pattern named and "
        "unmatched-token memos (the token check over `signals_observed`) count valid outputs.",
        "",
    ]
    lines += _table(
        (
            "arm",
            "decline precision",
            "decline recall",
            "pattern named",
            "unmatched-token memos",
            "stored latency p50",
        ),
        (
            (
                arm_id,
                _pct(arm["final"]["decline_precision"]),
                _pct(arm["final"]["decline_recall"]),
                _pct(arm["final"]["pattern_named"]),
                _pct(arm["final"]["unmatched_token_memos"]),
                _seconds(arm["final"]["latency_p50_ms"]),
            )
            for arm_id, arm in arms.items()
        ),
    )
    lines += ["## Consistency probes", ""]
    consistency = stats["consistency"]
    lines += _table(
        ("arm", "complete probe sets", "probes agree", "probes agree with the original memo"),
        (
            (
                arm_id,
                _pct(item["complete_probe_sets"]),
                _pct_ci(item["probe_agreement"]),
                _pct_ci(item["agreement_with_original"]),
            )
            for arm_id, item in consistency["arms"].items()
        ),
    )
    common = consistency.get("common_complete_cases")
    if common:
        lines += [f"On the {common['cases']} cases complete for both prompts:", ""]
        lines += _table(
            ("arm", "probes agree", "probes agree with the original memo"),
            (
                (arm_id, _pct(item["probe_agreement"]), _pct(item["agreement_with_original"]))
                for arm_id, item in common["arms"].items()
            ),
        )
    lines += [
        "## Paired comparisons on matched cases",
        "",
        "Intervals: Wilson treats cases as independent; the bootstrap resamples users, or "
        "injected stories and users outside any story.",
        "",
    ]
    lines += _table(
        (
            "comparison",
            "cases",
            "first rate",
            "second rate",
            "only first",
            "only second",
            "exact McNemar p",
            "difference, user bootstrap",
            "difference, story bootstrap",
        ),
        (
            (
                f"{item['first']} vs {item['second']}: {item['outcome']}",
                item["cases"],
                _pct(item["first_rate"]),
                _pct(item["second_rate"]),
                item["only_first"],
                item["only_second"],
                f"{item['exact_mcnemar_p']:.3g}",
                _pp(item["difference_second_minus_first"]["user_bootstrap_95"]),
                _pp(item["difference_second_minus_first"]["user_or_story_bootstrap_95"]),
            )
            for item in stats["paired"].values()
        ),
    )
    case_set = stats["case_set"]
    lines += [
        "## Case set",
        "",
        f"A {case_set['description']}: {case_set['cases']} alerts dated "
        f"{case_set['alert_dates']['first']} to {case_set['alert_dates']['last']}; "
        f"{_pct(case_set['before_cutoff'])} predate {case_set['cutoff']}, the date the "
        f"configuration of the time called the holdout start. {case_set['users']} users "
        f"({case_set['user_or_story_groups']} groups when injected stories are merged); "
        f"labelled fraud {_pct(case_set['labelled_fraud'])} against "
        f"{_pct(case_set['labelled_fraud_among_all_alerts'])} of all alerts; the benign "
        f"cases' minimum tenure is {case_set['benign_min_tenure_days']} days.",
        "",
    ]
    lines += _table(
        ("pattern", "cases", "stories"),
        (
            (pattern, count, case_set["stories_by_pattern"].get(pattern, "–"))
            for pattern, count in case_set["strata"].items()
        ),
    )
    limits = stats["limitations"]
    world_ranges = limits["order_id_ranges_in_world"]
    lines += [
        "## Limitations",
        "",
        f"- Packets with negative tenure: {limits['negative_tenure_packets']['count']} "
        f"(alerts {', '.join(map(str, limits['negative_tenure_packets']['alert_ids']))}).",
        f"- Packets whose linkage counts include accounts or orders after the alert: "
        f"{limits['future_linkage_packets']['count']}.",
        f"- Packets counting installment payments made after the alert: "
        f"{limits['payments_after_alert']['count']} "
        f"(alerts {', '.join(map(str, limits['payments_after_alert']['alert_ids']))}).",
        f"- `amount_over_category_median` equals the amount over the full-period category "
        f"mean in {_pct(limits['category_ratio']['equals_full_period_mean'])} of packets.",
        f"- R06 rationales counting email-root accounts that signed up after the alert: "
        f"{_pct(limits['r06_rationale_future_accounts'])}; in all "
        f"{limits['r06_rationale_vs_packet_linkage']['packets_with_r06']} R06 packets the "
        f"rationale's count is the packet's `other_accounts_same_email_root` plus one.",
        f"- Order-id ranges: the selected cases of "
        f"{len(world_ranges['pure_patterns'])} patterns lie in id blocks holding no other "
        f"alert, so their id alone labels "
        f"{_pct(world_ranges['selected_cases_labelled_by_id_range'])} of the labelled cases.",
        "",
    ]
    replay = stats["replay"]
    lines += [
        "## Replay",
        "",
        f"- Archived result files reproduced exactly: "
        f"{sum(replay['archived_results_reproduced'].values())} of "
        f"{len(replay['archived_results_reproduced'])}.",
        f"- Stored responses: {replay['cache_files']}; in the replayed chains: "
        f"{replay['cache_files_in_chains']}; for cases beyond an arm's limit: "
        f"{replay['cache_files_beyond_case_limits']}; matching no prompt of the protocol: "
        f"{replay['cache_files_matching_no_protocol_prompt']}.",
        f"- Strict validation rejected {replay['strict_validation']['rejected']} of the "
        f"{replay['strict_validation']['accepted_outputs']} outputs the study accepted.",
    ]
    for arm_id, arm in arms.items():
        if arm["primary_responses_beyond_case_limit"]:
            lines.append(
                f"- {arm_id}: {arm['primary_responses_beyond_case_limit']} stored responses "
                f"for cases beyond its {arm['cases']}-case limit, not scored."
            )
        if arm["retries"]["cases"]:
            lines.append(
                f"- {arm_id}: retried cases {', '.join(map(str, arm['retries']['cases']))}; "
                f"stored retries {arm['retries']['stored']}, valid {arm['retries']['valid']}."
            )
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- world checks
def world_checks(data_dir: Path, bench_dir: Path = BENCH_DIR) -> dict[str, Any]:
    """Facts about the archived packets that need the superseded world's CSV files."""
    import pandas as pd

    archive = Archive(bench_dir)
    data_dir = Path(data_dir)
    alerts_sha = file_sha256(data_dir / "alerts.csv")
    if alerts_sha != archive.manifest["world"]["alerts_csv_sha256"]:
        raise ValueError(
            f"{data_dir / 'alerts.csv'} is not the world the archive was built on "
            f"(sha256 {alerts_sha})"
        )
    orders = pd.read_csv(
        data_dir / "orders.csv",
        usecols=[
            "order_id",
            "user_id",
            "merchant_id",
            "ts",
            "amount",
            "device_id",
            "ship_address_id",
            "status",
        ],
        parse_dates=["ts"],
    )
    users = pd.read_csv(
        data_dir / "users.csv",
        usecols=["user_id", "signup_ts", "email", "email_domain"],
        parse_dates=["signup_ts"],
    )
    merchants = pd.read_csv(data_dir / "merchants.csv", usecols=["merchant_id", "category"])
    plans = pd.read_csv(data_dir / "plans.csv", usecols=["plan_id", "order_id"])
    installments = pd.read_csv(
        data_dir / "installments.csv",
        usecols=["plan_id", "due_ts", "paid_ts", "outcome"],
        parse_dates=["due_ts", "paid_ts"],
    )
    labels = pd.read_csv(data_dir / "labels.csv", usecols=["order_id", "pattern_id"])
    alerts = pd.read_csv(data_dir / "alerts.csv", usecols=["order_id"])
    stories = [
        json.loads(line)
        for line in (data_dir / "stories.jsonl").read_text().splitlines()
        if line.strip()
    ]

    users["root"] = users["email"].str.replace(".", "", regex=False).str.split("+").str[0]
    repayment = installments.merge(plans, on="plan_id").merge(
        orders[["order_id", "user_id"]], on="order_id"
    )
    approved = orders[orders["status"] == "approved"].merge(merchants, on="merchant_id")
    full_mean = approved.groupby("category")["amount"].mean()
    story_of = {}
    for story in sorted(stories, key=lambda item: item["story_id"]):
        for user in story["user_ids"]:
            story_of.setdefault(int(user), story["story_id"])

    groups: dict[str, str] = {}
    future_linkage, late_payments, r06_future, r06_total = [], [], [], 0
    reconstruction = Counter()
    ratio_mean = ratio_median_differs = 0
    for case in archive.cases:
        alert_id = case["alert_id"]
        packet = archive.packets[alert_id]
        alert, account = packet["alert"], packet["account"]
        user, at = int(alert["user_id"]), pd.Timestamp(alert["ts"])
        groups[str(alert_id)] = f"story {story_of[user]}" if user in story_of else f"user {user}"
        others = orders[orders["user_id"] != user]
        device = others[others["device_id"] == alert["device_id"]]
        address = others[others["ship_address_id"] == alert["ship_address_id"]]
        domain = account["email_domain"]
        root = account["email"].replace(".", "").split("+")[0]
        same_root = users[(users["email_domain"] == domain) & (users["root"] == root)]
        all_time = {
            "other_accounts_on_device": device["user_id"].nunique(),
            "other_accounts_on_ship_address": address["user_id"].nunique(),
            "other_accounts_same_email_root": len(same_root) - 1,
        }
        at_alert = {
            "other_accounts_on_device": device.loc[device["ts"] <= at, "user_id"].nunique(),
            "other_accounts_on_ship_address": address.loc[address["ts"] <= at, "user_id"].nunique(),
            "other_accounts_same_email_root": int(
                ((same_root["user_id"] != user) & (same_root["signup_ts"] <= at)).sum()
            ),
        }
        reconstruction["linkage"] += all_time == packet["linkage"]
        if any(packet["linkage"][key] > at_alert[key] for key in at_alert):
            future_linkage.append(alert_id)

        due = repayment[(repayment["user_id"] == user) & (repayment["due_ts"] < at)]
        frozen = packet["repayment_history"]
        rebuilt = {
            "installments_due": len(due),
            "paid": int((due["outcome"] == "paid").sum()) if len(due) else None,
            "late": int((due["outcome"] == "late").sum()) if len(due) else None,
            "failed_or_written_off": (
                int(due["outcome"].isin(["failed", "written_off"]).sum()) if len(due) else None
            ),
        }
        reconstruction["repayment"] += rebuilt == frozen
        if (due["outcome"].isin(["paid", "late"]) & (due["paid_ts"] > at)).any():
            late_payments.append(alert_id)

        category = alert["merchant_category"]
        frozen_ratio = account["amount_over_category_median"]
        ratio_mean += round(alert["amount"] / full_mean[category], 2) == frozen_ratio
        before = approved[(approved["category"] == category) & (approved["ts"] < at)]
        median_ratio = (
            round(alert["amount"] / before["amount"].median(), 2) if len(before) else None
        )
        ratio_median_differs += median_ratio != frozen_ratio

        for rule in alert["fired_rules"]:
            match = re.search(r"shared_root_accounts=(\d+)", rule.get("rationale", ""))
            if rule.get("id") == "R06" and match:
                r06_total += 1
                if int(match.group(1)) > int((same_root["signup_ts"] <= at).sum()):
                    r06_future.append(alert_id)

    n = len(archive.cases)
    fraud_stories: dict[str, set[str]] = {}
    for case in archive.cases:
        if case["truth_pattern"] != "benign":
            fraud_stories.setdefault(case["truth_pattern"], set()).add(
                groups[str(case["alert_id"])]
            )
    ranges = labels.groupby("pattern_id")["order_id"].agg(["min", "max", "count"])
    alert_labels = alerts.merge(labels, on="order_id", how="left")
    pure = {}
    for pattern, row in ranges.iterrows():
        inside = alert_labels[alert_labels["order_id"].between(row["min"], row["max"])]
        pure[pattern] = {
            "low": int(row["min"]),
            "high": int(row["max"]),
            "labelled_orders": int(row["count"]),
            "alerts_inside": len(inside),
            "alerts_inside_with_another_label": int((inside["pattern_id"] != pattern).sum()),
        }
    pure_patterns = sorted(
        p for p, item in pure.items() if not item["alerts_inside_with_another_label"]
    )
    case_orders = {
        c["alert_id"]: int(archive.packets[c["alert_id"]]["alert"]["order_id"])
        for c in archive.cases
    }
    order_label = dict(zip(labels["order_id"], labels["pattern_id"], strict=True))
    labelled_cases = [c for c in archive.cases if c["truth_pattern"] != "benign"]
    in_pure = sum(
        any(
            pure[p]["low"] <= case_orders[c["alert_id"]] <= pure[p]["high"]
            and order_label.get(case_orders[c["alert_id"]]) == p
            for p in pure_patterns
        )
        for c in labelled_cases
    )
    return {
        "alerts_csv_sha256": alerts_sha,
        "definitions": {
            "groups": "the injected story of the alert's user, or the user when in no story",
            "linkage_at_alert": (
                "other accounts with an order on the device or to the ship-to address at or "
                "before the alert; accounts with the same email domain and root (dots removed, "
                "text before '+') signed up at or before it"
            ),
            "payments_after_alert": (
                "an installment due before the alert counted as paid or late, paid after it"
            ),
            "category_ratio": (
                "the packet's amount_over_category_median against the amount over the mean of "
                "approved orders in the category over the whole period, and over the median "
                "of those placed before the alert"
            ),
            "r06_rationale_future_accounts": (
                "R06's shared_root_accounts exceeds the accounts with that email root signed "
                "up at or before the alert"
            ),
            "order_id_ranges_in_world": (
                "a pattern's labelled orders span an id block; the block is pure when every "
                "alert inside it carries that label"
            ),
        },
        "groups": groups,
        "summary": {
            "user_or_story_groups": len(set(groups.values())),
            "stories_by_pattern": {
                pattern: sum(group.startswith("story ") for group in found)
                for pattern, found in sorted(fraud_stories.items())
            },
            "labelled_fraud_among_all_alerts": _ratio(
                int(alerts["order_id"].isin(labels["order_id"]).sum()), len(alerts)
            ),
            "packet_reconstruction": {
                "linkage_equals_all_time_counts": _ratio(reconstruction["linkage"], n),
                "repayment_equals_final_outcomes": _ratio(reconstruction["repayment"], n),
            },
            "future_linkage_packets": {"count": len(future_linkage), "alert_ids": future_linkage},
            "payments_after_alert": {"count": len(late_payments), "alert_ids": late_payments},
            "category_ratio": {
                "equals_full_period_mean": _ratio(ratio_mean, n),
                "differs_from_median_before_alert": _ratio(ratio_median_differs, n),
            },
            "r06_rationale_future_accounts": {
                **_ratio(len(r06_future), r06_total),
                "alert_ids": r06_future,
            },
            "order_id_ranges_in_world": {
                "patterns": pure,
                "pure_patterns": pure_patterns,
                "selected_cases_labelled_by_id_range": _ratio(in_pure, len(labelled_cases)),
            },
        },
    }


# ------------------------------------------------------------------------ stage and CLI
def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float | np.integer | np.floating):
        return _num(value)
    raise TypeError(f"cannot write {type(value).__name__} to JSON")


def _dump(data: Any) -> str:
    return json.dumps(_clean(data), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write(
    out_dir: Path, bench_dir: Path = BENCH_DIR, world_checks_path: Path | None = None
) -> dict[str, Path]:
    """Regenerate the corrected files into ``out_dir``; returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, text in build(bench_dir, world_checks_path).items():
        paths[name] = out_dir / name
        paths[name].write_bytes(text.encode("utf-8"))
    return paths


def check(bench_dir: Path = BENCH_DIR) -> list[str]:
    """Differences between a fresh replay and the committed ``corrected/`` files."""
    corrected = Path(bench_dir) / "corrected"
    differences = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = write(Path(tmp), bench_dir, corrected / WORLD_CHECKS)
        for name, path in fresh.items():
            committed = corrected / name
            new = path.read_text(encoding="utf-8")
            if not committed.exists():
                differences.append(f"{name}: not committed")
                continue
            old = committed.read_text(encoding="utf-8")
            if old != new:
                diff = list(
                    difflib.unified_diff(
                        old.splitlines(),
                        new.splitlines(),
                        f"committed/{name}",
                        f"replay/{name}",
                        lineterm="",
                        n=1,
                    )
                )
                shown = "\n".join(diff[:40]) + ("\n..." if len(diff) > 40 else "")
                differences.append(f"{name}: differs\n{shown}")
    return differences


def _metric_rate(item: Mapping[str, Any], population: str, *, wilson: bool = False) -> Metric:
    interval = None
    if wilson and item.get("wilson_95"):
        low, high = item["wilson_95"]
        interval = Interval(low, high, "wilson")
    if not item["denominator"]:
        return Metric.not_evaluated(
            unit="rate",
            population=population,
            window="all",
            reason="nothing to count",
            numerator=0,
            denominator=0,
        )
    return Metric.from_ratio(
        item["numerator"],
        item["denominator"],
        population=population,
        window="all",
        interval=interval,
    )


def _headline_metrics(stats: Mapping[str, Any]) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {}
    for arm_id, arm in stats["arms"].items():
        slug = f"{SLUGS[arm['model']]}_{arm['prompt_version'].removeprefix('memo_')}"
        base = f"llm.history.{slug}"
        for endpoint in ENDPOINTS:
            item = arm[endpoint]
            population = f"{arm['cases']} archived development cases, {arm_id}, {endpoint}"
            metrics[f"{base}.{endpoint}.action_correct"] = _metric_rate(
                item["action_correct"], population, wilson=True
            )
            metrics[f"{base}.{endpoint}.valid_outputs"] = _metric_rate(
                item["valid_outputs"], population
            )
        final = arm["final"]
        population = f"{arm['cases']} archived development cases, {arm_id}, final"
        metrics[f"{base}.final.decline_recall"] = _metric_rate(final["decline_recall"], population)
        metrics[f"{base}.final.unmatched_token_memos"] = _metric_rate(
            final["unmatched_token_memos"], f"valid final outputs, {arm_id}"
        )
    for arm_id, item in stats["consistency"]["arms"].items():
        arm = stats["arms"][arm_id]
        slug = f"{SLUGS[arm['model']]}_{arm['prompt_version'].removeprefix('memo_')}"
        population = f"complete probe sets of the first {item['probe_cases']} cases, {arm_id}"
        metrics[f"llm.history.{slug}.probe_agreement"] = _metric_rate(
            item["probe_agreement"], population, wilson=True
        )
        metrics[f"llm.history.{slug}.probe_agreement_with_original"] = _metric_rate(
            item["agreement_with_original"], population, wilson=True
        )
    for name, item in stats["paired"].items():
        population = f"{item['case_set']}; {item['outcome']}"
        for side in ("only_first", "only_second"):
            metrics[f"llm.history.paired.{name}.{side}"] = Metric(
                value=item[side], unit="count", population=population, window="all"
            )
        difference = item["difference_second_minus_first"]
        low, high = difference["user_bootstrap_95"]
        metrics[f"llm.history.paired.{name}.difference"] = Metric.from_ratio(
            difference["numerator"],
            difference["denominator"],
            population=population,
            window="all",
            unit="share",
            interval=Interval(low, high, "cluster_bootstrap"),
            note="second arm minus first; interval resamples users",
        )
    return metrics


def run(out_dir: Path) -> StageResult:
    """Pipeline stage ``llm``: write the corrected files into ``out_dir``; headline metrics."""
    paths = write(out_dir)
    stats = _read_json(paths["statistics.json"])
    return StageResult(
        stage="llm",
        versions={"llm_history": stats["benchmark"]},
        inputs={
            "llm/eval/benchmarks/2026-08-dev/MANIFEST.json": file_sha256(
                BENCH_DIR / "MANIFEST.json"
            ),
            f"llm/eval/benchmarks/2026-08-dev/corrected/{WORLD_CHECKS}": file_sha256(
                BENCH_DIR / "corrected" / WORLD_CHECKS
            ),
        },
        metrics=_headline_metrics(stats),
        notes=[
            "historical development study replayed from stored responses; not a held-out "
            "evaluation (see llm/eval/benchmarks/2026-08-dev/HISTORY.md)"
        ],
    )


def replay(out_dir: Path) -> dict[str, Metric]:
    """The stage's metrics alone, for callers that build the result file themselves."""
    return dict(run(out_dir).metrics)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--check", action="store_true", help="regenerate into a temporary directory and compare"
    )
    parser.add_argument(
        "--groups-from",
        type=Path,
        metavar="DATA_DIR",
        help="recompute corrected/world_checks.json from the superseded world",
    )
    args = parser.parse_args(argv)
    corrected = BENCH_DIR / "corrected"
    if args.groups_from:
        corrected.mkdir(exist_ok=True)
        (corrected / WORLD_CHECKS).write_text(_dump(world_checks(args.groups_from)))
        print(f"wrote {corrected / WORLD_CHECKS}")
        return 0
    try:
        if args.check:
            differences = check()
            for difference in differences:
                print(difference)
            if differences:
                print(
                    f"corrected/ is out of date: {len(differences)} file(s) differ; "
                    "run python -m llm.eval.history"
                )
                return 1
            print("corrected/ matches the replay")
            return 0
        for path in write(corrected).values():
            print(f"wrote {path}")
        return 0
    except (CoverageError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
