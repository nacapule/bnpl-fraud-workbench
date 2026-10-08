"""The memo benchmark: live runs under call and token caps, cached replay, coverage gate
and scores.

A benchmark lives in ``llm/eval/benchmarks/<id>/`` and is fixed before any call:

``benchmark.json``
    id, phase (``development``, ``final``, or ``cases`` for the memos of the case files'
    alerts), policy id and SHA-256, prompt version and
    SHA-256 of the system prompt, SHA-256 of every file that scores a memo, the arms
    (backend, model, effort), the cases (case id built from seed, family, order and
    decision time, so an alert keeps its identity whatever the policy bands; packet
    SHA-256; decision-point axis, ``review`` or ``check_completed``; stratum; cluster;
    sampling weight; the latent diagnostic), each axis's cases, eligible decisions and
    share of them (``axes``, the natural-mix weights) and the invariance probes.
    :func:`check_shape` holds it to the configured sizes per axis, seeds and cluster
    caps before any call or score.
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

``canaries.json``
    each live run's isolation check (:func:`run_live`): arm, CLI version, isolation,
    verdict, outcome and token counts; the answer itself goes to ``--log-dir``.

``ledger.json`` beside the benchmarks keeps every arm's spend across benchmarks and
invocations; one live run per arm at a time holds it (a lock), and every change is made
under a file lock on the current state. The call caps are hard: a call is not made unless
it fits, counting retries and isolation checks. A token cap is an admission threshold:
a call is made only while the tokens charged, those reserved and the call's bound
(:func:`call_bound`) stay ``token_stop_margin`` below it. The bound covers every request
the installed CLI sends for one call that the model answers (see :data:`REQUESTS_PER_CALL`),
each with at most its prompts' bytes, ``input_overhead_tokens``, the CLI's short message
before it and the earlier requests' output as input (a token covers at least one byte)
and ``max_output_tokens`` as output, which the backend enforces (a token cap is refused
for a backend that cannot). It does not cover requests the CLI repeats after an API error
or a broken stream; those are billed little or nothing, and the margin is for them. The
bound is reserved, with the case it is for, before the call; afterwards the reservation
becomes a receipt of the reported usage, or of the whole bound when the usage is unknown
or the run was interrupted. A case's record takes its attempts and usage from its
receipts, so the records add up to the ledger wherever a run stopped; an isolation
check's receipt is its own, and its outcome is in ``canaries.json``. A call whose
reported usage exceeds its bound, or a failed isolation check, parks the arm in the same
ledger change that charges it: the ledger records why, and no live run of the arm starts
until someone has looked and removed the ``parked`` entry.

The request count assumes Claude Code runs each call itself, with one model turn
(``--max-turns 1``). A launcher in ``CLAUDE_CLI_BIN`` must keep a call to one turn as
well before the next request is sent; that is the operator's precondition, like signing
in with a personal subscription only, and a call over its bound still parks the arm.

Scoring replays the cache only, after checking that the policy, prompt, packets, the
referee's views and the scoring code still have their recorded hashes (a scoring change
must be named with ``--amend-scoring``, and is recorded in the results). Every case and
probe of every scored arm must have a record (the coverage gate); ``--arm`` scores one
arm alone, and paired comparisons are made only when every arm is scored. Malformed
output is a counted failure; nothing is retried for format. The headline is the
complete-memo pass rate (:func:`score_record`) beside its components and the
acceptable rate, each also as the natural-mix rate over the decision-point axes
(:func:`natural_rate`); intervals are cluster-aware, with exact bounds that stay
informative when nothing fails (:func:`statistics`). Usage::

  python -m llm.eval.harness --benchmark 2026-10-dev --arm sol --live \\
      --log-dir <dir> --private-terms <file> --max-calls 40
  python -m llm.eval.harness --benchmark 2026-10-dev           # score every arm
  python -m llm.eval.harness --benchmark 2026-10-dev --arm sol # one arm alone
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
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
# development and final benchmarks, and the memos for the case files' alerts
PHASES = ("development", "final", "cases")
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


class IsolationError(RuntimeError):
    """The model's context held instructions the isolation should have kept out."""


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
    """Write ``value`` whole or not at all, as ASCII JSON (any string survives)."""
    partial = path.with_name(f"{path.name}.{os.getpid()}.{os.urandom(4).hex()}.partial")
    partial.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """An exclusive lock held while the block runs (``path`` is a lock file)."""
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _write_log(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", errors="backslashreplace")


# ------------------------------------------------------------------ fixed inputs


def final_axes(sizes: Mapping[str, Any] | None = None) -> dict[str, int]:
    """The final cohort's cases on each decision-point axis (all review decisions when
    the configuration does not split them); they add up to ``final_cases``."""
    sizes = SIZES if sizes is None else sizes
    axes = {str(axis): int(n) for axis, n in (sizes.get("final_axes") or {
        "review": sizes["final_cases"]}).items()}
    if sum(axes.values()) != int(sizes["final_cases"]):
        raise ShapeError(f"final_axes {axes} do not add up to {sizes['final_cases']} cases")
    return axes


def development_shapes(sizes: Mapping[str, Any] | None = None) -> list[dict[str, int]]:
    """The cases a development benchmark may take on each axis: review decisions, or
    decisions at check completions, or the configured subset of both."""
    sizes = SIZES if sizes is None else sizes
    shapes = [{"review": int(sizes["development_cases"])}]
    if sizes.get("development_check_cases"):
        shapes.append({"check_completed": int(sizes["development_check_cases"])})
    if sizes.get("development_combined"):
        shapes.append({str(axis): int(n) for axis, n in sizes["development_combined"].items()})
    return shapes


def case_axis(case: Mapping[str, Any]) -> str:
    """A case's decision-point axis (review decisions before the axes existed)."""
    return str(case.get("axis", "review"))


def axis_shares(definition: Mapping[str, Any]) -> dict[str, float]:
    """Each axis's weight in the natural-mix rate: its share of the eligible decisions
    (each axis's share of the cases when the definition does not record them)."""
    cases = definition.get("cases") or []
    recorded = definition.get("axes")
    if recorded:
        return {axis: float(item["share"]) for axis, item in recorded.items()}
    counts = Counter(case_axis(case) for case in cases)
    return {axis: count / len(cases) for axis, count in counts.items()}


def check_shape(definition: Mapping[str, Any], sizes: Mapping[str, Any] | None = None,
                protocol: Any = None) -> None:
    """The benchmark has its phase's configured size on each decision-point axis,
    seeds, family and probes, and no account, episode or cluster holds more than the
    configured number of cases."""
    from core.protocol import load_protocol

    sizes = SIZES if sizes is None else sizes
    protocol = protocol or load_protocol()
    phase, cases = definition.get("phase"), definition.get("cases") or []
    if phase not in PHASES:
        raise ShapeError(f"unknown phase {phase!r}")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ShapeError("a case id occurs twice")
    if phase == "cases":
        _check_case_memos(definition, sizes, protocol)
        return
    axes = dict(Counter(case_axis(case) for case in cases))
    allowed = development_shapes(sizes) if phase == "development" else [final_axes(sizes)]
    if axes not in allowed:
        raise ShapeError(f"a {phase} benchmark has {' or '.join(map(str, allowed))} cases "
                         f"by decision point, this one {axes}")
    recorded = definition.get("axes")
    if recorded is not None:
        if set(recorded) != set(axes) or any(int(recorded[axis]["cases"]) != axes[axis]
                                             for axis in axes):
            raise ShapeError(f"the recorded axes {recorded} do not fit the cases {axes}")
        shares = [float(item["share"]) for item in recorded.values()]
        if any(not 0 < share <= 1 for share in shares) or abs(sum(shares) - 1) > 1e-9:
            raise ShapeError(f"the axes' shares {shares} must be positive and add up to 1")
    seeds = {int(case["seed"]) for case in cases}
    if {case["family"] for case in cases} != {"baseline"}:
        raise ShapeError(f"{phase} cases come from baseline worlds only")
    if phase == "development":
        if not seeds <= set(protocol.development_seeds):
            raise ShapeError(f"development cases from non-development seeds {sorted(seeds)}")
    elif not seeds <= set(protocol.final_seeds) or len(seeds) < int(sizes["min_final_seeds"]):
        raise ShapeError(f"final cases need at least {sizes['min_final_seeds']} final "
                         f"seeds, got {sorted(seeds)}")
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


def _check_case_memos(definition: Mapping[str, Any], sizes: Mapping[str, Any],
                      protocol: Any) -> None:
    """The case files' memos: one to ``case_memos`` review decisions of the canonical
    world, one arm or more, no probes."""
    cases = definition["cases"]
    world = protocol.raw["cases"]["world"]
    if not 1 <= len(cases) <= int(sizes.get("case_memos", 0)):
        raise ShapeError(f"case memos number 1 to {sizes.get('case_memos', 0)}, "
                         f"not {len(cases)}")
    if {(int(case["seed"]), case["family"]) for case in cases} != {
            (int(world["seed"]), str(world["family"]))}:
        raise ShapeError(f"case memos come from the case world {world}")
    if any(case_axis(case) != "review" for case in cases) or definition.get("probes"):
        raise ShapeError("case memos are review decisions, without probes")
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
    whose CLI or isolation differs from the pinned ones. Runs of other arms may pin at
    the same time; the file is reloaded and replaced under a lock."""
    pin = {"backend": arm.backend, "model": arm.model, "effort": arm.effort,
           "cli_version": str(fingerprint["cli_version"]),
           "isolation": str(fingerprint["isolation"])}
    with _file_lock(directory / ".pins.lock"):
        pins = read_pins(directory)
        if arm.name in pins and pins[arm.name] != pin:
            raise FrozenError(f"arm {arm.name} is pinned to {pins[arm.name]}, but this run "
                              f"would use {pin}; a changed CLI needs a new benchmark")
        if arm.name not in pins:
            pins[arm.name] = pin
            _write_json(directory / "pins.json", pins)
    return pin


# ----------------------------------------------------------------------- budgets


class Ledger:
    """An arm's spend across every benchmark and invocation.

    Opening it takes the arm's run lock, so one live run per arm holds it at a time;
    reservations still pending then were left by a run that ended without settling them
    and are charged in full. Every change reloads the file under a file lock, so runs
    of different arms never overwrite each other. A call's bound is reserved, with the
    case it is for, before it is made, and settled after it into a receipt for that
    case; a case's record takes its usage from its receipts (:meth:`case_usage`)."""

    def __init__(self, path: Path, arm: str, *, calls: int | None, tokens: int | None,
                 margin: int = 0):
        self.path, self.arm, self.margin = path, arm, int(margin)
        self.caps = {"calls": calls, "tokens": tokens}
        path.parent.mkdir(parents=True, exist_ok=True)
        self._run_lock = (path.parent / f".{path.name}.{arm}.lock").open("a")
        try:
            fcntl.flock(self._run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._run_lock.close()
            raise BudgetError(f"another live run of arm {arm} holds the ledger") from None
        with self._update() as entry:
            for key in sorted(entry["pending"]):
                reservation = entry["pending"].pop(key)
                entry["interrupted_calls"] += 1
                self._charge(entry, key, reservation,
                             attempt_usage(reservation["tokens"], None, None))

    def close(self) -> None:
        if not self._run_lock.closed:
            fcntl.flock(self._run_lock, fcntl.LOCK_UN)
            self._run_lock.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _update(self) -> Iterator[dict[str, Any]]:
        """This arm's entry in the current file, written back after the change."""
        with _file_lock(self.path.parent / f".{self.path.name}.lock"):
            data = json.loads(self.path.read_text()) if self.path.exists() else {"arms": {}}
            entry = data["arms"].setdefault(self.arm, {})
            for key, empty in (("calls", 0), ("charged_tokens", 0), ("input_tokens", 0),
                               ("output_tokens", 0), ("unknown_usage_calls", 0),
                               ("interrupted_calls", 0), ("by_benchmark", {}), ("pending", {}),
                               ("receipts", {})):
                entry.setdefault(key, empty)
            yield entry
            _write_json(self.path, data)

    def refusal(self, bound: int) -> str | None:
        """Why a call with this token bound may not be made, or None: the arm is parked,
        or the call would break the call cap or come within ``margin`` of the token cap."""
        with self._update() as entry:
            if entry.get("parked"):
                return (f"arm {self.arm} is parked ({entry['parked']}); remove the entry "
                        f"from {self.path.name} once that is resolved")
            pending = sum(item["tokens"] for item in entry["pending"].values())
            calls_cap, tokens_cap = self.caps["calls"], self.caps["tokens"]
            if calls_cap is not None and entry["calls"] + len(entry["pending"]) + 1 > calls_cap:
                return f"arm {self.arm} has used {entry['calls']} of {calls_cap} calls"
            if tokens_cap is not None and (entry["charged_tokens"] + pending + bound
                                           + self.margin > tokens_cap):
                return (f"arm {self.arm} has charged {entry['charged_tokens']} of "
                        f"{tokens_cap} tokens; the next call may use up to {bound}, and "
                        f"calls stop {self.margin} short of the cap")
        return None

    def park(self, reason: str) -> None:
        """Stop every live run of the arm until someone removes the entry."""
        with self._update() as entry:
            entry["parked"] = reason

    def reserve(self, benchmark: str, bound: int, *, case: str) -> str:
        """Record, before a call for ``case``, that it may use up to ``bound`` tokens."""
        key = f"{benchmark}:{os.getpid()}:{os.urandom(6).hex()}"
        with self._update() as entry:
            entry["pending"][key] = {"benchmark": benchmark, "case": case,
                                     "tokens": int(bound)}
        return key

    def release(self, key: str) -> None:
        """A reservation for a call that never reached the model: no charge, no receipt."""
        with self._update() as entry:
            del entry["pending"][key]

    def settle(self, key: str, input_tokens: int | None, output_tokens: int | None, *,
               park: str | None = None) -> int:
        """Replace a reservation by a receipt of the call's reported usage, or of its
        whole bound when the usage is unknown. A charge over the bound parks the arm in
        the same change, and so does ``park``. Returns the charge."""
        with self._update() as entry:
            reservation = entry["pending"].pop(key)
            charged = self._charge(entry, key, reservation, attempt_usage(
                reservation["tokens"], input_tokens, output_tokens))
            if charged > reservation["tokens"]:
                entry["parked"] = (f"a call used {charged} tokens, more than its bound "
                                   f"{reservation['tokens']}")
            elif park:
                entry["parked"] = park
        return charged

    def case_usage(self, case: str) -> dict[str, int]:
        """The usage of every call made for ``case``, across runs."""
        with self._update() as entry:
            receipts = [receipt for receipt in entry["receipts"].values()
                        if receipt["case"] == case]
        return {name: sum(receipt[name] for receipt in receipts) for name in USAGE_FIELDS}

    @staticmethod
    def _charge(entry: dict[str, Any], key: str, reservation: Mapping[str, Any],
                usage: Mapping[str, int]) -> int:
        for name in USAGE_FIELDS:
            entry[name] += usage[name]
        spent = entry["by_benchmark"].setdefault(reservation["benchmark"],
                                                 {"calls": 0, "charged_tokens": 0})
        spent["calls"] += 1
        spent["charged_tokens"] += usage["charged_tokens"]
        entry["receipts"][key] = {"case": reservation["case"], **usage}
        return usage["charged_tokens"]


USAGE_FIELDS = ("calls", "input_tokens", "output_tokens", "charged_tokens",
                "unknown_usage_calls")


def attempt_usage(bound: int, input_tokens: int | None, output_tokens: int | None
                  ) -> dict[str, int]:
    """One call's usage as the ledger charges it: the reported tokens, or the call's
    whole bound (at least what was reported) when the usage is unknown."""
    unknown = input_tokens is None or output_tokens is None
    known = (input_tokens or 0) + (output_tokens or 0)
    return {"calls": 1, "input_tokens": input_tokens or 0, "output_tokens": output_tokens or 0,
            "charged_tokens": max(known, bound) if unknown else known,
            "unknown_usage_calls": int(unknown)}


# Requests the model answers in one call of the Claude Code CLI (its query loop, 2.1.292,
# with no tools and one turn): the first; up to three continuations after the output
# limit; one retry after a reply that ends in a tool call it cannot read, which starts
# the continuation count again, so three more; one nudge after a reply without text; and
# one retry after a refusal (switched off, and counted anyway). A reply that calls a
# tool ends the call (``--max-turns 1``) instead of starting a turn that would reset
# those counts. Compaction is turned off (:data:`client.CLAUDE_SETTINGS_ENV`), and the
# CLI would compact otherwise only for a request too long for the context window, which
# :func:`largest_input` keeps every request below.
OUTPUT_CONTINUATIONS = 3
REQUESTS_PER_CALL = 1 + OUTPUT_CONTINUATIONS + 1 + OUTPUT_CONTINUATIONS + 1 + 1
NUDGE_TOKENS = 1000  # the CLI's message before a later request (under 300 bytes)


def _requests(system_prompt: str, user_prompt: str) -> list[tuple[int, int]]:
    """Each request's most input and output tokens: request k (from 0) has the prompts'
    bytes, the CLI's own input and k earlier outputs and messages as input."""
    prompt = (len(system_prompt.encode()) + len(user_prompt.encode())
              + int(SIZES["input_overhead_tokens"]))
    output = int(SIZES["max_output_tokens"])
    return [(prompt + k * (output + NUDGE_TOKENS), output) for k in range(REQUESTS_PER_CALL)]


def call_bound(system_prompt: str, user_prompt: str) -> int:
    """The most tokens one call can use, over every request it may send."""
    return sum(tokens for request in _requests(system_prompt, user_prompt)
               for tokens in request)


def largest_input(system_prompt: str, user_prompt: str) -> int:
    """The longest input a call's requests may have, in tokens."""
    return max(tokens for tokens, _ in _requests(system_prompt, user_prompt))


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


CANARY_SYSTEM = "You answer questions about your own context in one word."
CANARY_PROMPT = (
    "Apart from this question and the one-line instruction before it, does anything in "
    "your context give instructions about, or describe, a particular person: their name, "
    "preferences, background, career or projects? A note giving only an account email "
    "address, the date or the working environment does not count. Answer YES or NO.")


_VERDICT = re.compile(r"\s*(no|yes)\s*[.!]?\s*", re.IGNORECASE | re.ASCII)


def canary_verdict(text: str) -> str:
    """``no``, ``yes`` or ``unclear``: the isolation check's answer, which must be the
    one word alone (with a full stop at most)."""
    match = _VERDICT.fullmatch(text)
    return match.group(1).lower() if match else "unclear"


def _record_canary(directory: Path, entry: Mapping[str, Any]) -> None:
    path = directory / "canaries.json"
    with _file_lock(directory / ".canaries.lock"):
        entries = json.loads(path.read_text()) if path.exists() else []
        _write_json(path, [*entries, dict(entry)])


def check_isolation(directory: Path, definition: Mapping[str, Any], arm: Arm,
                    pin: Mapping[str, str], ledger: Ledger, backend: Any, *, logs: Path,
                    used: dict[str, Any], refusal: Callable[[int], str | None],
                    max_output: int | None) -> str | None:
    """A run's isolation check (:func:`run_live`): a reason to stop the run, or None when
    it passed. A failed check is parked in the ledger change that charges it, then
    raises :class:`IsolationError`; a check over its bound is parked the same way and
    raises :class:`BudgetError`. Its spend is a ledger receipt (case
    ``isolation-check:...``), and each check that reached the model, with its outcome, is
    in ``canaries.json``."""
    bound = call_bound(CANARY_SYSTEM, CANARY_PROMPT)
    if reason := refusal(bound):
        return reason
    stamp = f"canary-{os.getpid()}-{os.urandom(3).hex()}"
    reservation = ledger.reserve(definition["id"], bound,
                                 case=f"isolation-check:{definition['id']}:{stamp}")
    entry = {"arm": arm.name, "cli_version": None, "isolation": None, "verdict": None,
             "outcome": None, "input_tokens": None, "output_tokens": None}
    try:
        response = backend.complete(client.Request(
            arm.backend, arm.model, arm.effort, CANARY_SYSTEM, CANARY_PROMPT,
            max_output_tokens=max_output))
    except client.BackendError as error:
        if not error.called:
            ledger.release(reservation)
            return f"the isolation check could not start: {error}"
        used["calls"] += 1
        used["tokens"] += ledger.settle(reservation, None, None)
        _record_canary(directory, {**entry, "outcome": "transport failure"})
        return f"the isolation check failed in transport: {error}"
    summary = response.summary
    verdict = canary_verdict(response.text)
    problems = []
    if _contaminated(summary):
        problems.append("tool, file, hook or unrecognised events in the log")
    if summary.model != arm.model:
        problems.append("answered by another model")
    if summary.cli_version != pin["cli_version"] or summary.isolation != pin["isolation"]:
        outcome = "changed CLI or isolation"
    elif not problems and summary.n_error_events:
        outcome = "error events"
    else:
        problems += [f"the model answered {verdict}"] if verdict != "no" else []
        outcome = "failed" if problems else "passed"
    failure = "the isolation check failed: " + "; ".join(problems)
    charged = ledger.settle(reservation, summary.input_tokens, summary.output_tokens,
                            park=failure if outcome == "failed" else None)
    used["calls"] += 1
    used["tokens"] += charged
    if charged > bound:  # parked by the settlement
        outcome = "over its bound"
    _record_canary(directory, {
        **entry, "cli_version": summary.cli_version, "isolation": summary.isolation,
        "verdict": verdict, "outcome": outcome, "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens})
    if outcome == "over its bound":
        raise BudgetError(f"the isolation check used {charged} tokens, more than its "
                          f"bound {bound}")
    _write_log(logs / f"{stamp}.events.jsonl",
               "".join(json.dumps(event) + "\n" for event in response.events))
    _write_log(logs / f"{stamp}.text.txt", response.text)
    if outcome == "changed CLI or isolation":
        raise FrozenError(f"the CLI or its isolation changed during the run "
                          f"({summary.cli_version})")
    if outcome == "failed":
        raise IsolationError(f"{failure} (log {stamp} in {logs})")
    if outcome == "error events":
        return "error events in the isolation check's log"
    return None


def run_live(directory: Path, arm_name: str, *, log_dir: Path, private_terms: Sequence[str],
             max_calls: int, max_tokens: int | None = None, ledger_path: Path | None = None,
             backend_factory: Callable[[str], Any] = client.backend) -> dict[str, Any]:
    """Call the arm's model for every case and probe without a cached record.

    The benchmark's shape and hashes are checked and the arm's CLI version and
    isolation pinned first. Before the run's first memo call, one call through the same
    backend asks the model whether its context gives instructions about, or describes, a
    particular person (:data:`CANARY_PROMPT`). A log with error events stops the run; a
    log with tool, file, hook or unrecognised events, another model, or any answer but a
    plain no parks the arm and raises :class:`IsolationError`. The check is a tripwire for
    instructions the isolation missed, not a proof that there are none. At most one
    retry per case, for a transport failure or an error event in the log; none for
    format. Each call's token bound (:func:`call_bound`) is reserved in the
    arm's :class:`Ledger` before the call, and a call is not made if it would exceed
    ``max_calls`` or come within the stop margin of ``max_tokens`` in this run, or break
    the arm's configured caps across runs (the module docstring). Stops after
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
    with Ledger(ledger_path or LEDGER, arm.name, calls=caps.get("calls"),
                tokens=caps.get("tokens"), margin=int(SIZES["token_stop_margin"])) as ledger:
        pin = pin_arm(directory, arm, fingerprint)
        return _calls(directory, definition, arm, pin, ledger, backend, terms, system_prompt,
                      log_dir=log_dir, max_calls=max_calls, max_tokens=max_tokens,
                      max_output=max_output)


def _calls(directory: Path, definition: Mapping[str, Any], arm: Arm, pin: Mapping[str, str],
           ledger: Ledger, backend: Any, terms: Sequence[str], system_prompt: str, *,
           log_dir: Path, max_calls: int, max_tokens: int | None, max_output: int | None
           ) -> dict[str, Any]:
    cache = directory / "cache"
    cache.mkdir(exist_ok=True)
    logs = log_dir / definition["id"] / arm.name
    quarantine = log_dir / "quarantine"
    logs.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    used: dict[str, Any] = {"calls": 0, "tokens": 0, "recorded": 0, "quarantined": 0,
                            "transport_failures": 0, "stopped": None}
    failures_in_a_row = 0
    checked: list[bool] = []

    def refusal(bound: int) -> str | None:
        if used["calls"] + 1 > max_calls:
            return f"this run's cap of {max_calls} calls"
        if max_tokens is not None and (used["tokens"] + bound + ledger.margin > max_tokens):
            return (f"this run's cap of {max_tokens} tokens, which calls stop "
                    f"{ledger.margin} short of")
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
        if max_output is not None and (largest_input(system_prompt, user_prompt)
                                       > int(SIZES["context_tokens"])):
            raise BudgetError(f"a request for {stem} could outgrow the context window, "
                              f"where the CLI would add compaction requests")
        case = cache_key(identity)
        attempts = ledger.case_usage(case)["calls"]  # earlier runs' calls; none answered
        errors: list[str] = ["an earlier run's attempt"] * attempts
        response = None
        while attempts < 2:
            reason = refusal(bound)
            if reason is None and not checked:
                reason = check_isolation(directory, definition, arm, pin, ledger, backend,
                                         logs=logs, used=used, refusal=refusal,
                                         max_output=max_output) or refusal(bound)
                checked.append(True)
            if reason:
                used["stopped"] = reason
                print(f"stopped before a call: {reason}", file=sys.stderr)
                return used
            attempts += 1
            reservation = ledger.reserve(definition["id"], bound, case=case)
            try:
                response = backend.complete(client.Request(
                    arm.backend, arm.model, arm.effort, system_prompt, user_prompt,
                    max_output_tokens=max_output))
            except client.BackendError as error:
                if not error.called:  # never reached the model: not an attempt
                    ledger.release(reservation)
                    used["stopped"] = f"the CLI failed before asking the model: {error}"
                    print(f"stopped: {used['stopped']}", file=sys.stderr)
                    return used
                errors.append(str(error))
                used["calls"] += 1
                used["tokens"] += ledger.settle(reservation, None, None)
                response = None
                continue
            summary = response.summary
            charged = ledger.settle(reservation, summary.input_tokens, summary.output_tokens)
            used["calls"] += 1
            used["tokens"] += charged
            if charged > bound:  # parked by the settlement
                raise BudgetError(f"a call used {charged} tokens, more than its bound {bound}")
            _write_log(logs / f"{stem}.attempt{attempts}.events.jsonl",
                       "".join(json.dumps(event) + "\n" for event in response.events))
            if summary.cli_version != pin["cli_version"] or summary.isolation != pin["isolation"]:
                raise FrozenError(f"the CLI or its isolation changed during the run "
                                  f"({summary.cli_version})")
            if summary.n_error_events and not _contaminated(summary):
                errors.append("error events in the log")
                _write_log(logs / f"{stem}.attempt{attempts}.text.txt", response.text)
                response = None
                continue
            break
        if errors:
            _write_log(logs / f"{stem}.errors.txt", "\n".join(errors) + "\n")
        usage = ledger.case_usage(case)
        record: dict[str, Any] = {"identity": identity, "attempts": attempts,
                                  "failed_attempts": len(errors), "usage": usage}
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
                          "failed_attempts": len(errors), "usage": usage,
                          "outcome": "protocol_failure", "quarantined": reasons}
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
        if not isinstance(record.get("usage"), Mapping):
            raise FrozenError(f"cache record {path.name} has no usage")
        records[(case_id, probe)] = record
    if missing:
        raise CoverageError(f"arm {arm.name} lacks {len(missing)} of {len(calls)} calls: "
                            f"{missing[:5]}")
    return records


# The fraud explanation of the memo vocabulary (llm.memo) that names each latent pattern
# (core.world.PATTERNS); a legitimate order is named by any benign explanation.
PATTERN_EXPLANATION = {
    "P-ATO": "account_takeover", "P-STOLEN": "stolen_card", "P-SYNTH": "synthetic_identity",
    "P-NEVERPAY": "never_pay", "P-INR-ABUSE": "inr_abuse", "P-PROMO": "promo_abuse",
    "P-MERCH": "merchant_bustout",
}
LIKELIHOOD_RANK = {"low": 0, "medium": 1, "high": 2}
# Rows of the policy a case is decided under, in the order results list them; a row
# without cases is reported as not evaluated.
REFEREE_ROWS = ("§6.6(c)", "§6.6(b), one family", "§6.6(b), two or more families",
                "§6.6(a)", "§5.3(a)", "§5.3(b), without Linkage", "§5.3(b), with Linkage",
                "§5.3(c)")
NOT_EVALUATED = {"§5.3(c)": "a hold without an answer expires; no analyst decision follows"}
COMPONENTS = ("format_valid", "acceptable", "next_check_ok", "citations_valid",
              "no_claim_error")


def canonical_action(disposition: str | None, next_check: str | None,
                     view: referee.RefereeView, packet: Mapping[str, Any]) -> str | None:
    """The action a memo's disposition takes: ``needs_check`` is the memo's form of the
    hold (fraud policy §4.3) when it names a required check that has not run yet;
    otherwise every disposition is its own action."""
    if disposition != "needs_check":
        return disposition
    done = {str(item["check"]) for item in packet["decision"]["checks"]}
    if next_check in view.required_checks and next_check not in done:
        return "hold"
    return disposition


def top_hypothesis(hypotheses: Sequence[Mapping[str, Any]]) -> tuple[str | None, bool]:
    """The single most likely hypothesis's explanation, or None with True when several
    share the highest likelihood."""
    best = max(LIKELIHOOD_RANK[h["likelihood"]] for h in hypotheses)
    top = [h["explanation"] for h in hypotheses if LIKELIHOOD_RANK[h["likelihood"]] == best]
    return (top[0], False) if len(top) == 1 else (None, True)


def referee_row(view: Mapping[str, Any]) -> str:
    """The policy row a case is decided under (:data:`REFEREE_ROWS`), from the referee's
    view: the clause of its standard disposition, with row (b) split by the number of
    adverse families and §5.3(b) by Linkage (escalate is standard only with it)."""
    clauses = {view["clauses"][disposition] for disposition in view["standard"]}
    if len(clauses) != 1:
        raise FrozenError(f"a view with standard dispositions under {sorted(clauses)}")
    clause = clauses.pop()
    if clause == "§6.6(b)":
        return "§6.6(b), one family" if len(view["families"]) == 1 \
            else "§6.6(b), two or more families"
    if clause == "§5.3(b)":
        return "§5.3(b), with Linkage" if "escalate" in view["standard"] \
            else "§5.3(b), without Linkage"
    return clause


def score_record(record: Mapping[str, Any], packet: Mapping[str, Any],
                 view: referee.RefereeView) -> dict[str, Any]:
    """One case's outcome: a transport, protocol or format failure, or the referee's and
    the verifier's scores. ``packet`` is the packet the model was shown. Also the
    memo's canonical action (:func:`canonical_action`) and whether it takes the
    standard action, its single most likely hypothesis, and the complete-memo pass: a
    valid format, an acceptable disposition, the correct next check, valid citations
    (no unknown or non-holding id, the disposition's clause cited, no non-payment
    grounds without an installment due) and no verified claim error."""
    failed = {"acceptable": False, "disposition": None, "action": None, "next_check": None,
              "standard_action": False, "format_valid": False, "complete_pass": False,
              "components": dict.fromkeys(COMPONENTS, False),
              "top_hypothesis": None, "top_hypothesis_tied": False}
    if record["outcome"] != "response":
        return {"outcome": record["outcome"], **failed}
    parsed, problems = memo.parse(record["text"])
    if parsed is None:
        return {"outcome": "format_failure", **failed, "problems": problems[:10]}
    scored = referee.score(parsed, view)
    verification = verifier.verify_memo(parsed, packet)
    action = canonical_action(scored.disposition, parsed["next_check"], view, packet)
    components = {
        "format_valid": True, "acceptable": scored.acceptable,
        "next_check_ok": scored.next_check_ok,
        "citations_valid": scored.citation_ok and not scored.nonpayment_violation,
        "no_claim_error": not verification["has_claim_error"],
    }
    top, tied = top_hypothesis(parsed["hypotheses"])
    return {"outcome": "scored", **scored.as_dict(), "verification": verification,
            "next_check": parsed["next_check"], "action": action,
            "standard_action": action in view.standard, "format_valid": True,
            "components": components, "complete_pass": all(components.values()),
            "top_hypothesis": top, "top_hypothesis_tied": tied}


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "value": round(numerator / denominator, 4) if denominator else None}


def natural_rate(definition: Mapping[str, Any], values: Sequence[float],
                 index: Sequence[int] | None = None) -> float | None:
    """The natural-mix rate of ``values`` (one per case, in the definition's order):
    within each decision-point axis the two-phase Hájek rate with the selection's
    weights, and the axes weighted by their shares of the eligible decisions
    (:func:`axis_shares`). ``index`` lists the cases to use (a bootstrap resample, with
    repeats); an axis absent from it leaves the others' shares renormalised."""
    cases = definition["cases"]
    shares = axis_shares(definition)
    positions = range(len(cases)) if index is None else index
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for position in positions:
        case = cases[position]
        weight = float(case["weight"])
        sums[case_axis(case)][0] += weight * float(values[position])
        sums[case_axis(case)][1] += weight
    present = {axis: total for axis, total in sums.items() if total[1] > 0}
    if not present:
        return None
    mass = sum(shares.get(axis, 0.0) for axis in present)
    if mass <= 0:
        return None
    return sum(shares.get(axis, 0.0) * total[0] / total[1]
               for axis, total in present.items()) / mass


def summarize_arm(definition: Mapping[str, Any], rows: Mapping[tuple[str, str], dict[str, Any]],
                  views: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Rates over the primary cases (failures count as failing every measure): the
    complete-memo pass rate (the headline) beside its components and the acceptable
    rate; each also as the natural-mix rate (:func:`natural_rate`); the standard action
    (``needs_check`` counted as the hold only as :func:`canonical_action` allows) beside
    the raw standard disposition; per stratum, per decision-point axis (raw, and the
    axis's weighted Hájek rate) and, with ``views``, per policy row
    (:data:`REFEREE_ROWS`); cluster counts; and invariance under each probe (raw
    disposition, canonical action, next check, complete-memo pass, and all three
    functional measures together)."""
    cases = definition["cases"]
    primary = [rows[(case["case_id"], "primary")] for case in cases]
    n = len(cases)
    outcomes = Counter(row["outcome"] for row in primary)
    scored = [row for row in primary if row["outcome"] == "scored"]
    acceptable = [bool(row["acceptable"]) for row in primary]
    passed = [bool(row.get("complete_pass")) for row in primary]
    weights = [float(case["weight"]) for case in cases]
    strata: dict[str, list[int]] = defaultdict(list)
    axes: dict[str, list[int]] = defaultdict(list)
    policy_rows: dict[str, list[int]] = defaultdict(list)
    for position, case in enumerate(cases):
        strata[case["stratum"]].append(position)
        axes[case_axis(case)].append(position)
        if views is not None:
            policy_rows[referee_row(views[case["case_id"]])].append(position)

    def group(positions: Sequence[int]) -> dict[str, Any]:
        return {"acceptable": _rate(sum(acceptable[i] for i in positions), len(positions)),
                "complete_pass": _rate(sum(passed[i] for i in positions), len(positions))}

    def axis_group(positions: Sequence[int]) -> dict[str, Any]:
        # the axis's two-phase Hájek rates (the natural mix restricted to one axis)
        return {**group(positions),
                "acceptable_weighted": _round(natural_rate(definition, acceptable, positions)),
                "complete_pass_weighted": _round(natural_rate(definition, passed, positions))}

    invariance = {}
    for probe, case_ids in sorted(definition.get("probes", {}).items()):
        pairs = [(rows[(case_id, "primary")], rows[(case_id, probe)]) for case_id in case_ids]

        def agree(key: str, pairs: list[tuple[dict[str, Any], dict[str, Any]]] = pairs) -> int:
            return sum(first.get(key) is not None and first.get(key) == second.get(key)
                       for first, second in pairs)

        functional = sum(
            first.get("action") is not None and first.get("next_check") is not None
            and all(first.get(key) == second.get(key)
                    for key in ("action", "next_check", "complete_pass"))
            for first, second in pairs)
        invariance[probe] = {
            "disposition": _rate(agree("disposition"), len(pairs)),
            "action": _rate(agree("action"), len(pairs)),
            "next_check": _rate(agree("next_check"), len(pairs)),
            "complete_pass": _rate(sum(first.get("complete_pass") == second.get("complete_pass")
                                       for first, second in pairs), len(pairs)),
            "functional": _rate(functional, len(pairs)),
        }
    claims = sum(row["verification"]["n_claims"] for row in scored)
    claim_errors = sum(row["verification"]["n_claim_errors"] for row in scored)
    summary = {
        "n_cases": n,
        "outcomes": dict(sorted(outcomes.items())),
        "complete_pass": _rate(sum(passed), n),
        "complete_pass_natural": _round(natural_rate(definition, passed)),
        "components": {name: _rate(sum(bool(row["components"][name]) for row in primary
                                       if "components" in row), n)
                       for name in COMPONENTS},
        "acceptable": _rate(sum(acceptable), n),
        "acceptable_natural": _round(natural_rate(definition, acceptable)),
        "acceptable_reweighted": (round(sum(w * ok for w, ok in zip(weights, acceptable,
                                                                    strict=True))
                                        / sum(weights), 4) if sum(weights) else None),
        "standard_action": _rate(sum(bool(row.get("standard_action")) for row in primary), n),
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
        "per_stratum": {stratum: _rate(sum(acceptable[i] for i in positions), len(positions))
                        for stratum, positions in sorted(strata.items())},
        "per_axis": {axis: axis_group(positions) for axis, positions in sorted(axes.items())},
        "clusters": len({case["cluster"] for case in cases}),
        "invariance": invariance,
    }
    if views is not None:
        summary["per_referee_row"] = {
            row: (group(policy_rows[row]) if policy_rows.get(row) else
                  {"n": 0, "not_evaluated": NOT_EVALUATED.get(row, "no case in the cohort")})
            for row in REFEREE_ROWS}
    return summary


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


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
    memo suggested an adverse disposition for legitimate orders and cleared fraud, and
    how often its single most likely hypothesis names the latent pattern (any benign
    explanation for a legitimate order; a tie at the top names nothing)."""
    adverse = ("hold", "decline", "escalate", "needs_check")
    by_class: dict[str, list[str | None]] = defaultdict(list)
    named: dict[str, list[bool]] = defaultdict(list)
    ties = 0
    for case in definition["cases"]:
        row = rows[(case["case_id"], "primary")]
        latent = case["latent"]
        by_class[latent["class"]].append(row.get("disposition"))
        top = row.get("top_hypothesis")
        ties += bool(row.get("top_hypothesis_tied"))
        if latent["class"] == "legitimate":
            named["legitimate"].append(top in memo.BENIGN_EXPLANATIONS)
        else:
            named["fraud"].append(top is not None
                                  and top == PATTERN_EXPLANATION.get(latent["pattern"]))
    legitimate, fraud = by_class.get("legitimate", []), by_class.get("fraud", [])
    every = named["legitimate"] + named["fraud"]
    return {
        "legitimate_adverse": _rate(sum(d in adverse for d in legitimate), len(legitimate)),
        "fraud_cleared": _rate(sum(d == "clear" for d in fraud), len(fraud)),
        "top_hypothesis_names_latent": _rate(sum(every), len(every)),
        "top_hypothesis_names_latent_by_class": {
            name: _rate(sum(values), len(values)) for name, values in sorted(named.items())},
        "top_hypothesis_tied": ties,
    }


TIE_NOTE = ("the arms' unweighted rates are equal on these cases (the natural-mix difference "
            "is reported beside); a tie is not evidence that they are equivalent")


def cluster_bound(values: Sequence[float], clusters: Sequence[str],
                  level: float = 0.95) -> dict[str, Any]:
    """Exact Clopper–Pearson bounds with clusters as the units, which stay informative
    when nothing fails: of ``k`` clusters, ``f`` hold a failing case. Gives the
    one-sided upper bound on the share of clusters with a failure (``1 - (1 - level) **
    (1 / k)`` when ``f`` is 0, about ``3 / k``) and the two-sided interval for the share
    without one."""
    from scipy.stats import beta

    k = len(set(clusters))
    f = len({cluster for cluster, value in zip(clusters, values, strict=True) if not value})
    alpha = 1 - level
    upper = 1.0 if f == k else float(beta.ppf(level, f + 1, k - f))
    s = k - f
    low = 0.0 if s == 0 else float(beta.ppf(alpha / 2, s, k - s + 1))
    high = 1.0 if s == k else float(beta.ppf(1 - alpha / 2, s + 1, k - s))
    return {"clusters": k, "clusters_with_failure": f,
            "failure_share_upper_one_sided": round(upper, 4),
            "pass_share_interval": [round(low, 4), round(high, 4)], "level": level}


def statistics(definition: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]], *,
               paired: bool = True) -> dict[str, Any]:
    """Per arm, for the complete-memo pass (the primary endpoint) and the acceptable
    disposition: a cluster-bootstrap interval for the natural-mix rate and for the raw
    rate, Wilson for the raw rate, and exact bounds with clusters as units
    (:func:`cluster_bound`). When every case passes (or none does) a bootstrap interval
    collapses to a point, so it is reported as null and the exact bound stands alone.
    With ``paired``, per pair of arms on the same cases: the case counts, a
    cluster-bootstrap interval for the difference in rates (the same clusters
    resampled for both arms; null when every case agrees) and a sign test over clusters
    (each cluster's net count of cases only one arm passed), so linked cases are not
    counted as independent evidence; the same for the difference in natural-mix rates. Equal
    rates (as many cases only one arm passed as only the other) are reported as a tie, not
    equivalence."""
    import numpy as np

    from core.stats import cluster_bootstrap, paired_outcomes, sign_test, wilson_interval

    ids = [case["case_id"] for case in definition["cases"]]
    clusters = [case["cluster"] for case in definition["cases"]]
    measures = ("complete_pass", "acceptable")
    marks = {measure: {name: {case_id: bool(arm["cases"][f"{case_id}/primary"][measure])
                              for case_id in ids} for name, arm in arms.items()}
             for measure in measures}
    out: dict[str, Any] = {"arms": {}, "paired": {}, "clusters": len(set(clusters))}

    def interval(values: np.ndarray, statistic: Callable[[np.ndarray], float]
                 ) -> list[float] | None:
        if values.min() == values.max():
            return None  # every resample gives the same value
        bounds = cluster_bootstrap(clusters, statistic, resamples=2000, seed=0)
        return [round(bounds.low, 4), round(bounds.high, 4)]

    for name in arms:
        out["arms"][name] = {}
        for measure in measures:
            values = np.array([marks[measure][name][case_id] for case_id in ids], dtype=float)
            wilson = wilson_interval(int(values.sum()), len(values))
            out["arms"][name][measure] = {
                "rate": round(float(values.mean()), 4),
                "natural": _round(natural_rate(definition, values)),
                "natural_cluster_bootstrap": interval(
                    values, lambda idx, v=values: float(natural_rate(definition, v, idx))),
                "cluster_bootstrap": interval(values,
                                              lambda idx, v=values: float(v[idx].mean())),
                "wilson": [round(wilson.low, 4), round(wilson.high, 4)],
                "exact_clusters": cluster_bound(values.tolist(), clusters),
                **({"degenerate": "every case passed" if values.min() == 1 else
                    "no case passed"} if values.min() == values.max() else {}),
            }
    if not paired:
        out["paired_note"] = "paired comparisons are made when every arm is scored"
        return out
    names = sorted(arms)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            comparison = {}
            for measure in measures:
                ok = marks[measure]
                counts = paired_outcomes(ok[first], ok[second])
                one = np.array([ok[first][i] for i in ids], dtype=float)
                two = np.array([ok[second][i] for i in ids], dtype=float)
                difference = one - two
                net: dict[str, float] = defaultdict(float)
                for cluster, value in zip(clusters, difference, strict=True):
                    net[cluster] += value
                test = sign_test(net.values())

                def natural_gap(idx: np.ndarray | None = None, one: np.ndarray = one,
                                two: np.ndarray = two) -> float:
                    return float(natural_rate(definition, one, idx)
                                 - natural_rate(definition, two, idx))

                comparison[measure] = {
                    "both": counts.both, "only_first": counts.only_first,
                    "only_second": counts.only_second, "neither": counts.neither,
                    "difference": round(float(difference.mean()), 4),
                    "difference_cluster_bootstrap": interval(
                        difference, lambda idx, d=difference: float(d[idx].mean())),
                    "natural_difference": round(natural_gap(), 4),
                    "natural_difference_cluster_bootstrap": interval(difference, natural_gap),
                    "clusters_favouring_first": test.positive,
                    "clusters_favouring_second": test.negative,
                    "clusters_tied": test.zero,
                    "cluster_sign_test_p": round(test.p_value, 6),
                    **({"tie": TIE_NOTE} if counts.only_first == counts.only_second else {}),
                }
            out["paired"][f"{first} vs {second}"] = comparison
    return out


def evaluate(directory: Path, *, amend: str | None = None,
             arms: Sequence[str] | None = None) -> dict[str, Any]:
    """Score the arms (every arm, or those named) from the cache: shape and hashes
    first, then each arm's coverage gate. Paired comparisons are made only when every
    arm of the benchmark is scored."""
    definition = load_benchmark(directory)
    check_shape(definition)
    amendment = check_frozen(directory, definition, amend=amend)
    views = json.loads((directory / "referee.json").read_text())
    names = list(definition["arms"]) if arms is None else list(arms)
    if unknown := set(names) - set(definition["arms"]):
        raise ShapeError(f"the benchmark has no arm {sorted(unknown)}")
    results: dict[str, Any] = {"benchmark": definition["id"], "arms": {},
                               "pins": read_pins(directory),
                               "axes": {axis: round(share, 4) for axis, share
                                        in sorted(axis_shares(definition).items())}}
    if amendment:
        results["scoring_amendment"] = amendment
    for name in names:
        arm = definition["arms"][name]
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
        spend = {key: sum(int(record["usage"][key]) for record in records.values())
                 for key in ("calls", "input_tokens", "output_tokens", "charged_tokens",
                             "unknown_usage_calls")}
        results["arms"][arm.name] = {
            "arm": arm.__dict__, "spend": spend,
            "summary": summarize_arm(definition, rows, views),
            "disagreement": disagreement_table(definition, views, rows),
            "latent_diagnostic": latent_diagnostic(definition, rows),
            "cases": {f"{case_id}/{probe}": row for (case_id, probe), row in sorted(rows.items())},
        }
    results["statistics"] = statistics(definition, results["arms"],
                                       paired=set(names) == set(definition["arms"]))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--arm", help="the arm to run live, or the one arm to score")
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
        results = evaluate(directory, amend=args.amend_scoring,
                           arms=None if args.arm is None else [args.arm])
    except CoverageError as error:
        print(f"coverage gate: {error}", file=sys.stderr)
        return 1
    except (FrozenError, ShapeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    _write_json(directory / ("results.json" if args.arm is None
                             else f"results-{args.arm}.json"), results)
    for name, arm in results["arms"].items():
        print(name, json.dumps({k: v for k, v in arm["summary"].items()
                                if k in ("n_cases", "outcomes", "complete_pass_natural",
                                         "complete_pass", "acceptable_natural",
                                         "acceptable")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
