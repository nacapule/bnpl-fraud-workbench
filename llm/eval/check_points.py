"""Decisions at check completions, as the incumbent's replay made them.

A review that holds an order runs the checks the policy requires; when one answers, the
analyst decides again on the evidence known then (fraud policy §5.3). The replay
(``queue_sim.replay``) makes that decision from the context row it holds for the order
at that moment: the row assembled at the start of the completion's day (at the checkout,
for an order placed that day) and rebuilt under the incumbent's own decisions before that
day. When the check answers on the day of the first decision, that is the first
decision's row. The decision time is the completion, and the facts are those of that
row, as the replay's reviewer saw them.

The pipeline keeps only the first decisions (``review_decisions.pkl``,
``queue_sim.stage.review_decisions``). :func:`completion_decisions` therefore replays the
incumbent's main run on the world again with the frozen code (its policy at the tuned
thresholds, base staffing, the test window, history rebuilt under its decisions) with a
reviewer that records what it is asked. Before it does, the world's tables must match
their manifest, and the world, the policy configuration and the tuning results must be
the ones the run's replay stage recorded in its lineage, so the replay reads what the
run's replay read. After it, the replay is refused unless it reproduces the kept first
decisions exactly, every value of every row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from core import asof, config
from core.evidence import CheckResult
from core.protocol import load_protocol
from core.world import SchemaError, coerce, read_world, verify_manifest
from model.train import RuleScorer
from queue_sim import policies, stage
from queue_sim.replay import PolicyHistory, Settings
from queue_sim.replay import replay as run_replay
from queue_sim.reviewer import Decision, Reviewer
from queue_sim.roster import ServiceCalendar
from rules import engine

ROW_COLUMNS = (*asof.KEY_COLUMNS, *asof.COLUMN_NAMES)
POLICY_CONFIG = Path(__file__).resolve().parents[2] / "config" / "policy.yaml"


class ReplayMismatch(RuntimeError):
    """Replaying the incumbent did not reproduce the decisions the run kept."""


class _Recorder(Reviewer):
    """The evidence-based reviewer, keeping every decision it makes after a check."""

    def __init__(self) -> None:
        self.after_checks: list[tuple[int, dict[str, Any], tuple[CheckResult, ...],
                                      Decision]] = []

    def decide(self, order_id: int, row: Mapping[str, Any],
               checks: Sequence[CheckResult]) -> Decision:
        decision = super().decide(order_id, row, checks)
        if checks:
            self.after_checks.append((int(order_id), dict(row), tuple(checks), decision))
        return decision


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def replay_inputs(run_dir: Path) -> dict[str, str]:
    """The inputs the run's replay stage recorded in ``<run>/lineage.json`` (name ->
    SHA-256); empty without a lineage."""
    lineage = Path(run_dir) / "lineage.json"
    if not lineage.exists():
        return {}
    return dict(json.loads(lineage.read_text()).get("stages", {}).get("replay", {})
                .get("inputs", {}))


def run_paths(world_dir: Path) -> tuple[Path, Path]:
    """The tuning results and the fitted-models folder of the run a world belongs to
    (``<run>/worlds/<seed>-<family>``). The tuning results are ``<run>/results/tune.json``
    or, for a run that published its results, the one in the folder its lineage names:
    the one whose hash the replay stage recorded, else the first that exists (which
    :func:`check_inputs` then refuses), else the lineage's."""
    run_dir = Path(world_dir).resolve().parent.parent
    places = [run_dir / "results" / "tune.json"]
    lineage = run_dir / "lineage.json"
    if lineage.exists():
        named = json.loads(lineage.read_text()).get("results_dir")
        if named:
            places.append(Path(named) / "tune.json")
    existing = [path for path in places if path.exists()]
    recorded = replay_inputs(run_dir).get("results/tune.json")
    matching = [path for path in existing if _sha256(path) == recorded]
    return (matching or existing or places[::-1])[0], run_dir / "fit"


def check_inputs(world_dir: Path, tables: Mapping[str, pd.DataFrame], tune_path: Path
                 ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """The replay's inputs are the run's. Returns the tables, each coerced to its schema
    once (``core.world.coerce``: what is verified is what is replayed), and the policy
    configuration parsed from the verified bytes. The tables must match the world's
    manifest; the world's identity, ``config/policy.yaml`` and the tuning results must
    be those the run's replay stage recorded in ``<run>/lineage.json``; and the policy
    configuration the frozen modules read (``core.config.load``) must be the file's.
    Raises :class:`ReplayMismatch`."""
    from pipeline import manifest_identity

    world_dir = Path(world_dir)
    manifest = json.loads((world_dir / "manifest.json").read_text())
    try:
        coerced = {name: coerce(name, frame) for name, frame in tables.items()}
        verify_manifest(coerced, manifest)
    except (SchemaError, ValueError, KeyError) as error:
        raise ReplayMismatch(f"{world_dir.name}: {error}") from error
    recorded = replay_inputs(world_dir.resolve().parent.parent)
    if not recorded:
        raise ReplayMismatch(f"no lineage of the replay beside {world_dir.name}: the run's "
                             "replay inputs are unknown")
    policy_bytes = POLICY_CONFIG.read_bytes()
    current = {f"world:{world_dir.name}": manifest_identity(manifest),
               "config/policy.yaml": _sha256_bytes(policy_bytes),
               "results/tune.json": _sha256(tune_path)}
    differ = sorted(name for name, digest in current.items() if recorded.get(name) != digest)
    if differ:
        raise ReplayMismatch(f"not the inputs the run's replay recorded: {differ}")
    policy_cfg = yaml.safe_load(policy_bytes)
    if policy_cfg != config.load("policy"):
        raise ReplayMismatch("the policy configuration in use is not the file's")
    return coerced, policy_cfg


def incumbent_policy(tune_path: Path, fit_dir: Path, seed: int) -> policies.Policy:
    """The incumbent rules at the thresholds tuning chose for ``seed``. The rule score
    routes by itself (the calibration is not used), so the scorer is the rule engine
    under the fitted metadata's name and version; the policy's version must be the one
    tuning recorded."""
    tune = json.loads(Path(tune_path).read_text())
    chosen = [row for row in tune["tables"]["tune.chosen"]
              if int(row["seed"]) == seed and row["policy"] == stage.INCUMBENT]
    if len(chosen) != 1:
        raise ReplayMismatch(f"tuning records {len(chosen)} incumbent choices for seed {seed}")
    meta = json.loads((Path(fit_dir) / str(seed) / "model_rules.json").read_text())
    scorer = RuleScorer(str(meta["name"]), str(meta["version"]), tuple(meta["columns"]),
                        engine.score, None)
    policy = policies.single_scorer(stage.INCUMBENT, scorer).with_thresholds(
        chosen[0]["review_threshold"], chosen[0]["decline_threshold"])
    if policy.version != chosen[0]["chosen_version"]:
        raise ReplayMismatch(f"the rebuilt incumbent has version {policy.version}, tuning "
                             f"chose {chosen[0]['chosen_version']}")
    return policy


def completion_decisions(world_dir: Path, kept: pd.DataFrame, *, seed: int,
                         tables: Mapping[str, pd.DataFrame] | None = None,
                         tune_path: Path | None = None, fit_dir: Path | None = None
                         ) -> pd.DataFrame:
    """Each review's decision at its last check completion on one world.

    ``kept`` is the world's ``review_decisions.pkl`` as the run kept it. Returns one row
    per review whose checks answered: the context row the reviewer read
    (:data:`ROW_COLUMNS`, with ``decision_at`` the last completion and ``assembled_at``
    the time the row was assembled), ``checks`` (every completed check),
    ``disposition`` (the reviewer's decision then) and ``first_decision_at`` (the
    ``decision_at`` of the review's first decision). Raises :class:`ReplayMismatch`
    unless the inputs are the run's (:func:`check_inputs`) and the replay reproduces
    ``kept`` exactly.
    """
    world_dir = Path(world_dir)
    default_tune, default_fit = run_paths(world_dir)
    tune_path = tune_path or default_tune
    tables, policy_cfg = check_inputs(world_dir, read_world(world_dir) if tables is None
                                      else tables, tune_path)
    protocol = load_protocol()
    policy = incumbent_policy(tune_path, fit_dir or default_fit, seed)
    versions = set(kept["policy_version"])
    if versions and versions != {policy.version}:
        raise ReplayMismatch(f"the kept decisions are of policy versions {sorted(versions)}, "
                             f"not {policy.version}")
    bench = stage.Bench.of(tables, asof.build_context(tables), seed=seed,
                           observed_until=protocol.observed_until, policy_cfg=policy_cfg)
    recorder = _Recorder()
    window = protocol.windows["test"]
    result = run_replay(
        bench.world, policy, window=(window.start, window.end),
        roster=stage.base_staffing(policy_cfg).roster(policy_cfg),
        calendar=ServiceCalendar.from_config(policy_cfg), reviewer=recorder,
        verification=bench.verification(),
        history=PolicyHistory(bench.world, neighbours=bench.neighbours, frozen=bench.frozen),
        settings=Settings.from_config(policy_cfg))
    again = stage.decision_rows(policy, result)
    try:
        pd.testing.assert_frame_equal(again.reset_index(drop=True),
                                      kept.reset_index(drop=True), check_exact=True)
    except AssertionError as error:
        raise ReplayMismatch(f"the replay does not reproduce the kept review decisions of "
                             f"{world_dir.name}: {str(error).splitlines()[0]}") from error
    first = kept.set_index("order_id")["decision_at"]
    last: dict[int, tuple[dict[str, Any], tuple[CheckResult, ...], Decision]] = {}
    for order_id, row, checks, decision in recorder.after_checks:
        last[order_id] = (row, checks, decision)  # in time order: the last one stays
    records = []
    for order_id, (row, checks, decision) in sorted(last.items()):
        completed = max(pd.Timestamp(result.completed_at) for result in checks)
        records.append({**{column: row[column] for column in ROW_COLUMNS},
                        "decision_at": completed,
                        "assembled_at": pd.Timestamp(row["decision_at"]),
                        "checks": checks, "disposition": decision.disposition.value,
                        "first_decision_at": pd.Timestamp(first.loc[order_id])})
    frame = pd.DataFrame(records, columns=[*ROW_COLUMNS, "assembled_at", "checks",
                                           "disposition", "first_decision_at"])
    expected = {int(order): [(str(check), str(outcome), pd.Timestamp(at))
                             for check, outcome, at in later]
                for order, later in zip(kept["order_id"], kept["checks_later"], strict=True)
                if len(later)}
    recorded = {int(order): [(str(c.check), str(c.outcome), pd.Timestamp(c.completed_at))
                             for c in checks]
                for order, checks in zip(frame["order_id"], frame["checks"], strict=True)}
    if recorded != expected:
        raise ReplayMismatch("the recorded completions differ from the kept checks_later")
    return frame
