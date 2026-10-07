"""The memo benchmark: live runs under call and token caps, cached replay, coverage gate
and scores.

A benchmark lives in ``llm/eval/benchmarks/<id>/`` and is fixed before any call:

``benchmark.json``
    id, phase (``development`` or ``final``), policy id and SHA-256, prompt version and
    SHA-256 of the system prompt, SHA-256 of every file that scores a memo, the arms
    (backend, model, effort), the cases (case id built from seed, family, order and
    decision time, so an alert keeps its identity whatever the policy bands; packet
    SHA-256; stratum; cluster; sampling weight; the latent diagnostic) and the
    invariance probes. :func:`check_shape` holds it to the configured sizes, seeds and
    cluster caps before any call or score.
``packets/<case>.json``, ``referee.json``
    the packets and the referee's view of each, written by :mod:`llm.eval.select_cases`.
``pins.json``
    each arm's CLI version and isolation hash, pinned at its first live call
    (:meth:`client.CodexBackend.fingerprint`); a later run whose CLI or isolation differs
    is refused, so one setting's records never replay for another.
``cache/<key>.json``
    one record per case, probe and arm. The key hashes the call's identity: benchmark,
    case, probe, arm name, backend, CLI version, model, effort, isolation, policy,
    prompt and scoring-code hashes, and the exact user message. The record holds the
    identity, the response, its event-log summary, duration and attempt counts. A
    response whose log shows a tool, file or hook event or an event the summary does not
    recognise, that another model answered, or whose record would name a private term,
    is kept out: its record holds only the identity and the reason, the case counts as a
    protocol failure, and the full record goes to ``<log dir>/quarantine/``. Transport
    error messages and full event logs go to ``--log-dir`` too, outside the repository.

``ledger.json`` beside the benchmarks keeps every arm's spend across benchmarks and
invocations. Each call is bounded before it is made: its input by the bytes of its
prompts plus ``input_overhead_tokens`` (a token covers at least one byte), its output by
``max_output_tokens``, which the backend enforces (a token cap is refused for a backend
that cannot). That bound is reserved in the ledger, on disk, before the call and must fit
under the arm's configured caps; afterwards the reservation becomes the reported usage,
or stays charged in full when the usage is unknown or the run was interrupted.

Scoring replays the cache only, after checking that the policy, prompt, packets, the
referee's views and the scoring code still have their recorded hashes (a scoring change
must be named with ``--amend-scoring``, and is recorded in the results). Every case and
probe of every arm must have a record (the coverage gate); malformed output is a
counted failure; nothing is retried for format. Usage::

  python -m llm.eval.harness --benchmark 2026-10-dev --arm sol --live \\
      --log-dir <dir> --private-terms <file> --max-calls 40
  python -m llm.eval.harness --benchmark 2026-10-dev           # score from the cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import load
from llm import client, memo, referee
from llm.eval import verifier
from llm.packet import ENTITIES, placeholders

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent.parent
BENCHMARKS = EVAL / "benchmarks"
LEDGER = BENCHMARKS / "ledger.json"
PROBES = ("primary", "shuffled", "renamed")
PROBE_SEED = 7919  # shuffles the context facts / draws fresh placeholder names per case
SIZES = load("llm")["benchmark"]
CAPS = load("llm")["caps"]
# Everything that turns a cached response into a score.
SCORING_CODE = ("core/actions.py", "core/asof.py", "core/evidence.py", "core/stats.py",
                "llm/eval/harness.py", "llm/eval/tokens.py", "llm/eval/verifier.py",
                "llm/memo.py", "llm/packet.py", "llm/referee.py")
CONSECUTIVE_FAILURES = 3  # a run stops after this many cases in a row fail in transport


class CoverageError(RuntimeError):
    """An arm lacks a cached record for a case or probe the benchmark requires."""


class FrozenError(RuntimeError):
    """A file the benchmark was fixed with has changed, or a record does not fit it."""


class ShapeError(ValueError):
    """The benchmark does not have the configured size, seeds, probes or cluster caps."""


class BudgetError(RuntimeError):
    """A call would exceed an arm's configured cap."""


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
    return {name: sha256_file(REPO / name) for name in SCORING_CODE}


def load_benchmark(directory: Path) -> dict[str, Any]:
    definition = json.loads((directory / "benchmark.json").read_text())
    definition["arms"] = {name: Arm(name, **arm) for name, arm in definition["arms"].items()}
    return definition


def _write_json(path: Path, value: Any) -> None:
    """Write ``value`` whole or not at all."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(partial, path)


# ------------------------------------------------------------------ fixed inputs


def check_shape(definition: Mapping[str, Any], sizes: Mapping[str, Any] | None = None,
                protocol: Any = None) -> None:
    """The benchmark has its phase's configured size, seeds, family and probes, and no
    account, episode or cluster holds more than the configured number of cases."""
    from core.protocol import load_protocol

    sizes = SIZES if sizes is None else sizes
    protocol = protocol or load_protocol()
    phase, cases = definition.get("phase"), definition.get("cases") or []
    if phase not in ("development", "final"):
        raise ShapeError(f"unknown phase {phase!r}")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ShapeError("a case id occurs twice")
    wanted = int(sizes["development_cases" if phase == "development" else "final_cases"])
    if len(ids) != wanted:
        raise ShapeError(f"a {phase} benchmark has {wanted} cases, this one {len(ids)}")
    seeds = {int(case["seed"]) for case in cases}
    if phase == "development":
        if not seeds <= set(protocol.development_seeds):
            raise ShapeError(f"development cases from non-development seeds {sorted(seeds)}")
        if {case["family"] for case in cases} != {"baseline"}:
            raise ShapeError("development cases come from baseline worlds only")
    else:
        if not seeds <= set(protocol.final_seeds) or len(seeds) < int(sizes["min_final_seeds"]):
            raise ShapeError(f"final cases need at least {sizes['min_final_seeds']} final "
                             f"seeds, got {sorted(seeds)}")
        if not {case["family"] for case in cases} <= set(protocol.family_starts):
            raise ShapeError("a final case comes from an unknown family")
    cap = int(sizes["max_cases_per_cluster"])
    for key in ("account", "episode", "cluster"):
        counts = Counter(case[key] for case in cases if case.get(key) is not None)
        if counts and max(counts.values()) > cap:
            raise ShapeError(f"more than {cap} cases share a {key}")
    for case in cases:
        weight = case.get("weight")
        if not isinstance(weight, int | float) or not math.isfinite(weight) or weight <= 0:
            raise ShapeError(f"case {case['case_id']} has no positive sampling weight")
        if not case.get("packet_sha256"):
            raise ShapeError(f"case {case['case_id']} has no packet hash")
    probes = definition.get("probes") or {}
    kinds = set(sizes["probe_kinds"])
    if not set(probes) <= kinds:
        raise ShapeError(f"unknown probes {sorted(set(probes) - kinds)}")
    if phase == "final" and set(probes) != kinds:
        raise ShapeError(f"a final benchmark runs every probe: {sorted(kinds)}")
    for kind, probe_ids in probes.items():
        if len(set(probe_ids)) != len(probe_ids) or not set(probe_ids) <= set(ids):
            raise ShapeError(f"probe {kind} names a case twice or a case not in the benchmark")
        if phase == "final" and len(probe_ids) != int(sizes["probe_cases"]):
            raise ShapeError(f"probe {kind} has {len(probe_ids)} cases, not "
                             f"{sizes['probe_cases']}")
    if not definition.get("arms"):
        raise ShapeError("a benchmark needs at least one arm")


def check_frozen(directory: Path, definition: Mapping[str, Any], *,
                 amend: str | None = None) -> dict[str, Any]:
    """The policy, prompt, packets and referee views still have their recorded hashes
    (always required), and so does the scoring code unless ``amend`` names why it
    changed. Returns the amendment record (empty when nothing changed)."""
    problems = []
    if sha256_file(memo.POLICY_PATH) != definition["policy"]["sha256"]:
        problems.append("the policy text")
    if (client.sha256_text(memo.system_prompt(definition["prompt"]["version"]))
            != definition["prompt"]["sha256"]):
        problems.append("the system prompt")
    if sha256_file(directory / "referee.json") != definition.get("referee_sha256"):
        problems.append("referee.json")
    for case in definition["cases"]:
        path = directory / "packets" / f"{case['case_id']}.json"
        if not path.exists() or sha256_file(path) != case["packet_sha256"]:
            problems.append(f"packet {case['case_id']}")
    if problems:
        raise FrozenError(f"changed since the benchmark was fixed: {', '.join(problems[:5])}")
    recorded, current = definition["code_sha256"], code_sha256()
    changed = sorted(name for name in set(recorded) | set(current)
                     if recorded.get(name) != current.get(name))
    if changed and not amend:
        raise FrozenError(f"scoring code changed since the benchmark was fixed: {changed}; "
                          "score with an amendment (--amend-scoring) or restore it")
    return {"changed_code": changed, "reason": amend} if changed else {}


def probe_input(packet: Mapping[str, Any], probe: str, case_id: str
                ) -> tuple[dict[str, Any], str]:
    """The packet a probe shows the model and its user message: the packet as is, with
    its context facts in another order, or with fresh placeholder names."""
    seed = int(hashlib.sha256(f"{PROBE_SEED}:{case_id}".encode()).hexdigest()[:8], 16)
    if probe == "primary":
        return dict(packet), memo.user_prompt(packet)
    if probe == "shuffled":
        return dict(packet), memo.user_prompt(packet, shuffle_seed=seed)
    if probe == "renamed":
        names = placeholders(seed)
        renamed = {**packet, "order": {**packet["order"],
                                       **{entity: names[entity] for entity in ENTITIES}}}
        return renamed, memo.user_prompt(renamed)
    raise ValueError(f"unknown probe {probe!r}")


def call_identity(definition: Mapping[str, Any], arm: Arm, pin: Mapping[str, str],
                  case_id: str, probe: str, user_prompt: str, system_prompt: str
                  ) -> dict[str, Any]:
    return {
        "benchmark": definition["id"], "case": case_id, "probe": probe,
        "arm": arm.name, "backend": arm.backend, "model": arm.model, "effort": arm.effort,
        "cli_version": pin["cli_version"], "isolation": pin["isolation"],
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


def read_pins(directory: Path) -> dict[str, dict[str, str]]:
    path = directory / "pins.json"
    return json.loads(path.read_text()) if path.exists() else {}


def pin_arm(directory: Path, arm: Arm, fingerprint: Mapping[str, str]) -> dict[str, str]:
    """Record the arm's CLI version and isolation at its first live run; refuse a run
    whose CLI or isolation differs from the pinned ones."""
    pins = read_pins(directory)
    pin = {"backend": arm.backend, "model": arm.model, "effort": arm.effort,
           "cli_version": str(fingerprint["cli_version"]),
           "isolation": str(fingerprint["isolation"])}
    if arm.name in pins and pins[arm.name] != pin:
        raise FrozenError(f"arm {arm.name} is pinned to {pins[arm.name]}, but this run "
                          f"would use {pin}; a changed CLI needs a new benchmark")
    pins[arm.name] = pin
    _write_json(directory / "pins.json", pins)
    return pin


# ----------------------------------------------------------------------- budgets


class Ledger:
    """An arm's spend across every benchmark and invocation. A call's bound is reserved
    on disk before it is made and settled after it, so an interrupted call stays
    charged in full."""

    def __init__(self, path: Path, arm: str, *, calls: int | None, tokens: int | None):
        self.path, self.arm = path, arm
        self.caps = {"calls": calls, "tokens": tokens}
        self.data = json.loads(path.read_text()) if path.exists() else {"arms": {}}
        self.entry = self.data["arms"].setdefault(arm, {
            "calls": 0, "charged_tokens": 0, "input_tokens": 0, "output_tokens": 0,
            "unknown_usage_calls": 0, "interrupted_calls": 0, "by_benchmark": {},
            "pending": {}})
        pending = self.entry.setdefault("pending", {})
        for key in sorted(pending):  # reserved by a run that never settled them
            benchmark, amount = pending[key]["benchmark"], pending[key]["tokens"]
            self.entry["interrupted_calls"] = self.entry.get("interrupted_calls", 0) + 1
            self._charge(benchmark, amount, None, None)
            del pending[key]
        _write_json(self.path, self.data)

    def refusal(self, bound: int) -> str | None:
        """Why a call with this token bound would break a cap, or None."""
        if self.caps["calls"] is not None and self.entry["calls"] + 1 > self.caps["calls"]:
            return f"arm {self.arm} has used {self.entry['calls']} of {self.caps['calls']} calls"
        if (self.caps["tokens"] is not None
                and self.entry["charged_tokens"] + bound > self.caps["tokens"]):
            return (f"arm {self.arm} has charged {self.entry['charged_tokens']} of "
                    f"{self.caps['tokens']} tokens; the next call may use up to {bound}")
        return None

    def reserve(self, benchmark: str, bound: int) -> str:
        """Record, before a call, that it may use up to ``bound`` tokens."""
        key = f"{benchmark}:{os.getpid()}:{len(self.entry['pending'])}:{os.urandom(4).hex()}"
        self.entry["pending"][key] = {"benchmark": benchmark, "tokens": int(bound)}
        _write_json(self.path, self.data)
        return key

    def release(self, key: str) -> None:
        """A reservation for a call that never reached the model."""
        del self.entry["pending"][key]
        _write_json(self.path, self.data)

    def settle(self, key: str, input_tokens: int | None, output_tokens: int | None) -> int:
        """Replace a reservation by the call's reported usage, or charge it in full when
        the usage is unknown. Returns the charge."""
        reservation = self.entry["pending"].pop(key)
        charged = self._charge(reservation["benchmark"], reservation["tokens"], input_tokens,
                               output_tokens)
        _write_json(self.path, self.data)
        return charged

    def _charge(self, benchmark: str, bound: int, input_tokens: int | None,
                output_tokens: int | None) -> int:
        unknown = input_tokens is None or output_tokens is None
        known = (input_tokens or 0) + (output_tokens or 0)
        charged = max(known, bound) if unknown else known
        self.entry["calls"] += 1
        self.entry["charged_tokens"] += charged
        self.entry["input_tokens"] += input_tokens or 0
        self.entry["output_tokens"] += output_tokens or 0
        self.entry["unknown_usage_calls"] += int(unknown)
        spent = self.entry["by_benchmark"].setdefault(benchmark, {"calls": 0,
                                                                  "charged_tokens": 0})
        spent["calls"] += 1
        spent["charged_tokens"] += charged
        return charged


def call_bound(system_prompt: str, user_prompt: str) -> int:
    """The most tokens one call can use: its prompts' bytes plus the CLI's own input
    and the output limit."""
    return (len(system_prompt.encode()) + len(user_prompt.encode())
            + int(SIZES["input_overhead_tokens"]) + int(SIZES["max_output_tokens"]))


# --------------------------------------------------------------------- live runs


def _contaminated(summary: client.EventSummary) -> bool:
    return bool(summary.n_tool_events or summary.n_file_events or summary.n_hook_events
                or summary.n_unrecognized_events or summary.tools_offered)


_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [text for key, item in value.items() for text in (*_strings(key), *_strings(item))]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _strings(item)]
    return [value] if isinstance(value, str) else [str(value)] if value is not None else []


def private_terms_in(value: Any, terms: Sequence[str]) -> list[str]:
    """The private terms in any string of ``value``, also after decoding the JSON
    escapes a model may write (``@`` for ``@``) and the JSON a response holds."""
    found: set[str] = set()
    for text in _strings(value):
        variants = [text, _ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)]
        body = text.strip()
        fenced = memo.FENCE.match(body)
        try:
            decoded = json.loads(fenced.group(1) if fenced else body)
        except (ValueError, RecursionError):
            decoded = None
        if isinstance(decoded, dict | list):
            variants += _strings(decoded)
        for variant in variants:
            found.update(client.private_matches(variant, terms))
    return sorted(found)


def run_live(directory: Path, arm_name: str, *, log_dir: Path, private_terms: Sequence[str],
             max_calls: int, max_tokens: int | None = None, ledger_path: Path | None = None,
             backend_factory: Callable[[str], Any] = client.backend) -> dict[str, Any]:
    """Call the arm's model for every case and probe without a cached record.

    The benchmark's shape and hashes are checked and the arm's CLI version and
    isolation pinned first. At most one retry per case, for a transport failure or an
    error event in the log; none for format. Each call's token bound
    (:func:`call_bound`) is reserved in the arm's :class:`Ledger` before the call, and a
    call is not made if its bound would exceed ``max_calls`` or ``max_tokens`` in this
    run or the arm's configured caps across runs. Stops after
    :data:`CONSECUTIVE_FAILURES` cases in a row fail in transport.
    """
    terms = [term for term in private_terms if term.strip()]
    if not terms:
        raise ValueError("live calls need the private terms to scan responses for")
    definition = load_benchmark(directory)
    check_shape(definition)
    check_frozen(directory, definition)
    arm = definition["arms"][arm_name]
    if private_terms_in([definition["id"], arm.__dict__], terms):
        raise ValueError("the benchmark id or the arm names a private term")
    system_prompt = memo.system_prompt(definition["prompt"]["version"])
    backend = backend_factory(arm.backend)
    caps = CAPS.get(arm.name, {})
    bounded = bool(getattr(backend, "bounds_output", False))
    if (caps.get("tokens") is not None or max_tokens is not None) and not bounded:
        raise ValueError(f"a token cap needs a backend that bounds its output; "
                         f"{arm.backend} does not")
    max_output = int(SIZES["max_output_tokens"]) if bounded else None
    fingerprint = backend.fingerprint(arm.model, arm.effort, max_output)
    if private_terms_in(fingerprint, terms):
        raise ValueError("the CLI's version or isolation names a private term")
    pin = pin_arm(directory, arm, fingerprint)
    ledger = Ledger(ledger_path or LEDGER, arm.name, calls=caps.get("calls"),
                    tokens=caps.get("tokens"))
    cache = directory / "cache"
    cache.mkdir(exist_ok=True)
    logs = log_dir / definition["id"] / arm.name
    quarantine = log_dir / "quarantine"
    logs.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    used: dict[str, Any] = {"calls": 0, "tokens": 0, "recorded": 0, "quarantined": 0,
                            "transport_failures": 0, "stopped": None}
    failures_in_a_row = 0

    def refusal(bound: int) -> str | None:
        if used["calls"] + 1 > max_calls:
            return f"this run's cap of {max_calls} calls"
        if max_tokens is not None and used["tokens"] + bound > max_tokens:
            return f"this run's cap of {max_tokens} tokens"
        return ledger.refusal(bound)

    for case_id, probe in required_calls(definition):
        packet = json.loads((directory / "packets" / f"{case_id}.json").read_text())
        _, user_prompt = probe_input(packet, probe, case_id)
        identity = call_identity(definition, arm, pin, case_id, probe, user_prompt,
                                 system_prompt)
        path = cache / f"{cache_key(identity)}.json"
        if path.exists():
            continue
        bound = call_bound(system_prompt, user_prompt)
        stem = f"{case_id}-{probe}"
        errors: list[str] = []
        response = None
        attempts = 0
        for _attempt in range(2):
            if reason := refusal(bound):
                used["stopped"] = reason
                print(f"stopped before a call: {reason}", file=sys.stderr)
                return used
            attempts += 1
            reservation = ledger.reserve(definition["id"], bound)
            try:
                response = backend.complete(client.Request(
                    arm.backend, arm.model, arm.effort, system_prompt, user_prompt,
                    max_output_tokens=max_output))
            except client.BackendError as error:
                errors.append(str(error))
                if error.called:
                    used["calls"] += 1
                    used["tokens"] += ledger.settle(reservation, None, None)
                else:
                    ledger.release(reservation)
                response = None
                continue
            used["calls"] += 1
            charged = ledger.settle(reservation, response.summary.input_tokens,
                                    response.summary.output_tokens)
            used["tokens"] += charged
            (logs / f"{stem}.attempt{attempts}.events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in response.events))
            if charged > bound:
                raise BudgetError(f"a call used {charged} tokens, more than its bound {bound}")
            if (response.summary.cli_version != pin["cli_version"]
                    or response.summary.isolation != pin["isolation"]):
                raise FrozenError(f"the CLI or its isolation changed during the run "
                                  f"({response.summary.cli_version})")
            if response.summary.n_error_events and not _contaminated(response.summary):
                errors.append("error events in the log")
                (logs / f"{stem}.attempt{attempts}.text.txt").write_text(response.text)
                response = None
                continue
            break
        if errors:
            (logs / f"{stem}.errors.txt").write_text("\n".join(errors) + "\n")
        record: dict[str, Any] = {"identity": identity, "attempts": attempts,
                                  "failed_attempts": len(errors)}
        if response is None:
            used["transport_failures"] += 1
            failures_in_a_row += 1
            record["outcome"] = "transport_failure"
        else:
            failures_in_a_row = 0
            reasons = []
            if _contaminated(response.summary):
                reasons.append("tool, file, hook or unrecognised events in the log")
            if response.summary.model != arm.model:
                reasons.append("answered by another model")
            record.update(summary=response.summary.as_dict(),
                          duration_ms=response.duration_ms, outcome="response",
                          text=response.text)
            if not reasons and private_terms_in(record, terms):
                reasons.append("names a private term")
            if reasons:
                used["quarantined"] += 1
                _write_json(quarantine / f"{definition['id']}-{arm.name}-{stem}.json",
                            {**record, "quarantined": reasons})
                record = {"identity": identity, "attempts": attempts,
                          "failed_attempts": len(errors), "outcome": "protocol_failure",
                          "quarantined": reasons}
        if private_terms_in(record, terms):  # only the identity can remain; it was checked
            raise ValueError(f"the record for {stem} names a private term")
        _write_json(path, record)
        used["recorded"] += 1
        if failures_in_a_row >= CONSECUTIVE_FAILURES:
            used["stopped"] = f"{failures_in_a_row} cases in a row failed in transport"
            print(f"stopped: {used['stopped']}", file=sys.stderr)
            return used
    return used


# ----------------------------------------------------------------------- scoring


def cached_records(directory: Path, definition: Mapping[str, Any], arm: Arm
                   ) -> dict[tuple[str, str], dict[str, Any]]:
    """The arm's record for every required case and probe; raises CoverageError if any
    is missing and FrozenError if a record's summary does not fit the pinned arm."""
    pin = read_pins(directory).get(arm.name)
    calls = required_calls(definition)
    if pin is None:
        raise CoverageError(f"arm {arm.name} has no pinned CLI: none of {len(calls)} calls ran")
    system_prompt = memo.system_prompt(definition["prompt"]["version"])
    records: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for case_id, probe in calls:
        packet = json.loads((directory / "packets" / f"{case_id}.json").read_text())
        _, user_prompt = probe_input(packet, probe, case_id)
        identity = call_identity(definition, arm, pin, case_id, probe, user_prompt,
                                 system_prompt)
        path = directory / "cache" / f"{cache_key(identity)}.json"
        if not path.exists():
            missing.append(f"{case_id}/{probe}")
            continue
        record = json.loads(path.read_text())
        if record.get("identity") != identity:
            raise FrozenError(f"cache record {path.name} does not match its identity")
        summary = record.get("summary")
        if record.get("outcome") == "response":
            expected = {"backend": arm.backend, "model": arm.model, "effort": arm.effort,
                        "cli_version": pin["cli_version"], "isolation": pin["isolation"],
                        "protocol_ok": True}
            if not isinstance(summary, Mapping) or any(
                    summary.get(key) != value for key, value in expected.items()):
                raise FrozenError(f"cache record {path.name} has a summary that does not "
                                  f"fit arm {arm.name}")
        elif record.get("outcome") not in ("transport_failure", "protocol_failure"):
            raise FrozenError(f"cache record {path.name} has no known outcome")
        records[(case_id, probe)] = record
    if missing:
        raise CoverageError(f"arm {arm.name} lacks {len(missing)} of {len(calls)} calls: "
                            f"{missing[:5]}")
    return records


def score_record(record: Mapping[str, Any], packet: Mapping[str, Any],
                 view: referee.RefereeView) -> dict[str, Any]:
    """One case's outcome: a transport, protocol or format failure, or the referee's and
    the verifier's scores. ``packet`` is the packet the model was shown."""
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
    rate reweighted to the eligible review decisions (the two-phase Hájek estimate with
    the selection's weights), cluster counts and invariance agreement."""
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
    """Per arm: Wilson and cluster-bootstrap intervals for the acceptable rate, and a
    cluster-bootstrap interval for the reweighted rate. Per pair of arms, on the same
    cases: the case counts, a cluster-bootstrap interval for the difference in rates
    (the same clusters resampled for both arms) and a sign test over clusters (each
    cluster's net count of cases only one arm got right), so linked cases are not
    counted as independent evidence."""
    import numpy as np

    from core.stats import cluster_bootstrap, paired_outcomes, sign_test, wilson_interval

    ids = [case["case_id"] for case in definition["cases"]]
    clusters = [case["cluster"] for case in definition["cases"]]
    weights = np.array([float(case["weight"]) for case in definition["cases"]])
    ok = {name: {case_id: bool(arm["cases"][f"{case_id}/primary"]["acceptable"])
                 for case_id in ids} for name, arm in arms.items()}
    out: dict[str, Any] = {"arms": {}, "paired": {}, "clusters": len(set(clusters))}

    def interval(statistic: Callable[[np.ndarray], float]) -> list[float]:
        bounds = cluster_bootstrap(clusters, statistic, resamples=2000, seed=0)
        return [round(bounds.low, 4), round(bounds.high, 4)]

    for name, outcomes in ok.items():
        values = np.array([outcomes[case_id] for case_id in ids], dtype=float)
        wilson = wilson_interval(int(values.sum()), len(values))
        out["arms"][name] = {
            "wilson": [round(wilson.low, 4), round(wilson.high, 4)],
            "cluster_bootstrap": interval(lambda idx, v=values: float(v[idx].mean())),
            "reweighted_cluster_bootstrap": interval(
                lambda idx, v=values: float((weights[idx] * v[idx]).sum()
                                            / weights[idx].sum())),
        }
    names = sorted(ok)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            paired = paired_outcomes(ok[first], ok[second])
            difference = np.array([ok[first][i] - ok[second][i] for i in ids], dtype=float)
            net: dict[str, float] = defaultdict(float)
            for cluster, value in zip(clusters, difference, strict=True):
                net[cluster] += value
            test = sign_test(net.values())
            out["paired"][f"{first} vs {second}"] = {
                "both": paired.both, "only_first": paired.only_first,
                "only_second": paired.only_second, "neither": paired.neither,
                "difference": round(float(difference.mean()), 4),
                "difference_cluster_bootstrap": interval(
                    lambda idx, d=difference: float(d[idx].mean())),
                "clusters_favouring_first": test.positive,
                "clusters_favouring_second": test.negative,
                "clusters_tied": test.zero,
                "cluster_sign_test_p": round(test.p_value, 6),
            }
    return out


def evaluate(directory: Path, *, amend: str | None = None) -> dict[str, Any]:
    """Score every arm from the cache: shape and hashes first, then the coverage gate."""
    definition = load_benchmark(directory)
    check_shape(definition)
    amendment = check_frozen(directory, definition, amend=amend)
    views = json.loads((directory / "referee.json").read_text())
    results: dict[str, Any] = {"benchmark": definition["id"], "arms": {},
                               "pins": read_pins(directory)}
    if amendment:
        results["scoring_amendment"] = amendment
    for arm in definition["arms"].values():
        records = cached_records(directory, definition, arm)
        rows = {}
        for (case_id, probe), record in records.items():
            packet = json.loads((directory / "packets" / f"{case_id}.json").read_text())
            shown, _ = probe_input(packet, probe, case_id)
            view = referee.view(shown)
            if view.as_dict() != views[case_id]:
                raise FrozenError(f"the referee's view of {case_id}/{probe} differs from "
                                  "the one fixed with the benchmark")
            rows[(case_id, probe)] = score_record(record, shown, view)
        summaries = [record.get("summary") or {} for record in records.values()]
        spend = {
            "calls": sum(record["attempts"] for record in records.values()),
            "input_tokens": sum(s.get("input_tokens") or 0 for s in summaries),
            "output_tokens": sum(s.get("output_tokens") or 0 for s in summaries),
        }
        results["arms"][arm.name] = {
            "arm": arm.__dict__, "spend": spend,
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
    parser.add_argument("--amend-scoring", metavar="REASON",
                        help="score although the scoring code changed, recording why")
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
        results = evaluate(directory, amend=args.amend_scoring)
    except CoverageError as error:
        print(f"coverage gate: {error}", file=sys.stderr)
        return 1
    except (FrozenError, ShapeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    _write_json(directory / "results.json", results)
    for name, arm in results["arms"].items():
        print(name, json.dumps({k: v for k, v in arm["summary"].items()
                                if k in ("n_cases", "outcomes", "acceptable")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
