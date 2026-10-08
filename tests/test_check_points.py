"""Decisions at check completions: the incumbent's replay run again on a small generated
world reproduces the review decisions the run kept, and the rows it returns are the
ones the replay's reviewer read when the checks answered."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core import asof, config
from core.protocol import load_protocol
from core.world import read_world
from llm import referee
from llm.eval import check_points
from llm.packet import build_packets
from model.train import RuleScorer
from queue_sim import policies, stage
from rules import definitions, engine

SEED = 416
REVIEW_AT = 10.0  # a low threshold, so the small world has holds whose checks answer


def incumbent(version: str = "test-rules") -> policies.Policy:
    scorer = RuleScorer("rules", version, tuple(definitions.COLUMNS), engine.score, None)
    return policies.single_scorer(stage.INCUMBENT, scorer).with_thresholds(REVIEW_AT, 1000.0)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[Path, pd.DataFrame]:
    """A run folder: a small world, the fitted rules' metadata, the tuning choice and the
    review decisions its replay kept, as the pipeline writes them."""
    from simulator.generate import generate_world

    root = tmp_path_factory.mktemp("run")
    world_dir = root / "worlds" / f"{SEED}-baseline"
    generate_world(SEED, "baseline", world_dir, scale=0.02)
    (root / "fit" / str(SEED)).mkdir(parents=True)
    (root / "fit" / str(SEED) / "model_rules.json").write_text(json.dumps(
        {"name": "rules", "version": "test-rules", "columns": list(definitions.COLUMNS)}))
    policy = incumbent()
    (root / "results").mkdir()
    (root / "results" / "tune.json").write_text(json.dumps({"tables": {"tune.chosen": [
        {"seed": SEED, "policy": stage.INCUMBENT, "review_threshold": REVIEW_AT,
         "decline_threshold": 1000.0, "chosen_version": policy.version},
        {"seed": SEED, "policy": "boosting", "review_threshold": 0.5,
         "decline_threshold": 0.9, "chosen_version": "other"}]}}))
    tables = read_world(world_dir)
    protocol = load_protocol()
    bench = stage.Bench.of(tables, asof.build_context(tables), seed=SEED,
                           observed_until=protocol.observed_until)
    window = protocol.windows["test"]
    result = bench.run(policy, (window.start, window.end),
                       stage.base_staffing(config.load("policy")))
    kept = stage.decision_rows(policy, result)
    kept.to_pickle(world_dir / "review_decisions.pkl")
    from pipeline import manifest_identity

    manifest = json.loads((world_dir / "manifest.json").read_text())
    (root / "lineage.json").write_text(json.dumps({"stages": {"replay": {"inputs": {
        f"world:{world_dir.name}": manifest_identity(manifest),
        "config/policy.yaml": check_points._sha256(check_points.POLICY_CONFIG),
        "results/tune.json": check_points._sha256(root / "results" / "tune.json")}}}}))
    return world_dir, kept


def test_completions_are_the_rows_the_reviewer_read_when_the_checks_answered(run) -> None:
    world_dir, kept = run
    later = kept[kept["checks_later"].map(len) > 0]
    assert len(later)  # the comparison bites
    found = check_points.completion_decisions(world_dir, kept, seed=SEED)
    assert sorted(found["order_id"]) == sorted(later["order_id"])
    first = kept.set_index("order_id")
    tables = read_world(world_dir)
    for row in found.itertuples():
        done = first.loc[row.order_id, "checks_later"]
        assert row.decision_at == max(pd.Timestamp(at) for _, _, at in done)
        assert [(str(c.check), str(c.outcome)) for c in row.checks] == [
            (check, outcome) for check, outcome, _ in done]
        assert row.first_decision_at == first.loc[row.order_id, "decision_at"]
        # the row the replay held then: the first decision's on its day, else the one
        # assembled at the start of the completion's day
        assert row.assembled_at <= row.decision_at
        decided = pd.Timestamp(first.loc[row.order_id, "decided_at"])
        if decided.normalize() == row.decision_at.normalize():
            assert row.assembled_at == row.first_decision_at
        else:
            assert row.assembled_at == row.decision_at.normalize()
    keys = [(int(o), pd.Timestamp(t)) for o, t in zip(found["order_id"], found["decision_at"],
                                                      strict=True)]
    packets = build_packets(tables, found, checks=dict(zip(keys, found["checks"], strict=True)))
    for key, disposition in zip(keys, found["disposition"], strict=True):
        assert packets[key]["decision"]["point"] == "check_completed"
        assert disposition in referee.view(packets[key]).standard


def test_a_replay_that_does_not_reproduce_the_kept_decisions_is_refused(run) -> None:
    world_dir, kept = run
    altered = kept.copy()
    altered.loc[altered.index[0], "final"] = "escalate" if altered["final"].iloc[0] != \
        "escalate" else "clear"
    with pytest.raises(check_points.ReplayMismatch, match="does not reproduce"):
        check_points.completion_decisions(world_dir, altered, seed=SEED)


def test_the_replay_reads_only_the_runs_own_inputs(run, tmp_path) -> None:
    world_dir, kept = run
    tables = read_world(world_dir)
    changed = {**tables, "account_events": tables["account_events"].iloc[1:]}
    with pytest.raises(check_points.ReplayMismatch, match="does not match its manifest"):
        check_points.completion_decisions(world_dir, kept, seed=SEED, tables=changed)
    lineage = world_dir.parent.parent / "lineage.json"
    original = lineage.read_text()
    recorded = json.loads(original)
    recorded["stages"]["replay"]["inputs"]["config/policy.yaml"] = "0" * 64
    lineage.write_text(json.dumps(recorded))
    try:
        with pytest.raises(check_points.ReplayMismatch, match="config/policy.yaml"):
            check_points.completion_decisions(world_dir, kept, seed=SEED, tables=tables)
    finally:
        lineage.write_text(original)


def test_the_incumbent_must_have_the_version_tuning_chose(run, tmp_path) -> None:
    world_dir, _ = run
    tune, fit = check_points.run_paths(world_dir)
    assert check_points.incumbent_policy(tune, fit, SEED).version == incumbent().version
    chosen = json.loads(tune.read_text())
    chosen["tables"]["tune.chosen"][0]["chosen_version"] = "0" * 12
    (tmp_path / "tune.json").write_text(json.dumps(chosen))
    with pytest.raises(check_points.ReplayMismatch, match="tuning chose"):
        check_points.incumbent_policy(tmp_path / "tune.json", fit, SEED)
    with pytest.raises(check_points.ReplayMismatch, match="0 incumbent choices"):
        check_points.incumbent_policy(tune, fit, SEED + 1)


def test_the_run_paths_follow_the_lineage(tmp_path) -> None:
    tmp_path = tmp_path.resolve()
    world_dir = tmp_path / "run" / "worlds" / "416-baseline"
    world_dir.mkdir(parents=True)
    assert check_points.run_paths(world_dir) == (tmp_path / "run" / "results" / "tune.json",
                                                tmp_path / "run" / "fit")
    (tmp_path / "run" / "lineage.json").write_text(json.dumps(
        {"results_dir": str(tmp_path / "elsewhere")}))
    assert check_points.run_paths(world_dir)[0] == tmp_path / "elsewhere" / "tune.json"
    (tmp_path / "run" / "results").mkdir()
    (tmp_path / "run" / "results" / "tune.json").write_text("{}")  # the run's own first
    assert check_points.run_paths(world_dir)[0] == tmp_path / "run" / "results" / "tune.json"
