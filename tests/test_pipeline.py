"""The pipeline: stage order, run directories, lineage, the freeze gate and evaluation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

import pipeline
from core import protocol as proto
from core import world as world_module
from core.results import (
    Metric,
    StageResult,
    VersionMismatch,
    read_result,
    read_summary,
    write_result,
)

REPO = Path(__file__).resolve().parents[1]
MINI_WORLD = REPO / "tests" / "fixtures" / "mini_world"


def test_stages_run_in_one_fixed_order() -> None:
    assert pipeline.STAGE_NAMES == (
        "world", "validate", "load", "context", "fit", "tune", "replay", "alerts",
        "evaluate", "llm",
    )
    assert [s.name for s in pipeline.select("fit", "replay")] == ["fit", "tune", "replay"]
    assert [s.name for s in pipeline.select(until="validate")] == ["world", "validate"]
    with pytest.raises(pipeline.PipelineError, match="unknown stage"):
        pipeline.select("demo")
    with pytest.raises(pipeline.PipelineError, match="comes after"):
        pipeline.select("replay", "fit")


def test_profiles_choose_seeds_and_where_results_go(tmp_path: Path) -> None:
    protocol = proto.load_protocol()
    dev = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path)
    assert dev.seeds == protocol.development_seeds
    lag = set(protocol.base_only_families)
    assert lag == {"lag_half", "lag_double"}
    assert dev.base_only_families == protocol.base_only_families
    assert set(dev.families) == set(protocol.family_starts) - lag  # lag families: final only
    assert dev.database_world == pipeline.WorldRef(protocol.canonical_seed, "baseline")
    assert dev.results_dir == tmp_path / "dev" / "results"
    assert dev.docs_dir == tmp_path / "dev" / "docs"

    final = pipeline.make_run(pipeline.PROFILES["final"], runs=tmp_path)
    assert final.seeds == protocol.final_seeds
    assert set(final.families) == set(protocol.family_starts)
    named = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path,
                              families=["baseline", "lag_half"])
    assert named.families == ("baseline", "lag_half")
    assert final.results_dir == pipeline.RESULTS and final.docs_dir == REPO
    assert pipeline.WorldRef(protocol.canonical_seed, "baseline") in final.all_worlds
    assert protocol.canonical_seed in final.fit_seeds
    narrowed = pipeline.make_run(pipeline.PROFILES["final"], runs=tmp_path, seeds=[1, 2])
    assert narrowed.results_dir == tmp_path / "final" / "results"  # a partial run never publishes

    ci = pipeline.make_run(pipeline.PROFILES["ci"], runs=tmp_path)
    assert (ci.seeds, ci.families, ci.scale) == ((416,), ("baseline",), pipeline.CI_SCALE)

    given = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    assert given.worlds == [pipeline.WorldRef(0, "baseline")]
    with pytest.raises(pipeline.PipelineError, match="baseline"):
        pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, families=["fraud_mix_shift"])
    with pytest.raises(pipeline.PipelineError, match="unknown families"):
        pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, families=["baseline", "x"])


def test_the_final_run_is_refused_before_the_freeze(tmp_path: Path) -> None:
    run = pipeline.make_run(pipeline.PROFILES["final"], runs=tmp_path)
    with pytest.raises(proto.FreezeError):
        pipeline.preflight(pipeline.PROFILES["final"], run)
    assert pipeline.main(["run", "--profile", "final", "--name", "refused-final-test"]) == 1
    assert not (pipeline.RUNS / "refused-final-test").exists()


def test_no_profile_generates_a_final_seed_before_the_freeze(tmp_path: Path) -> None:
    final_seed = proto.load_protocol().final_seeds[0]
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[final_seed, 416])
    with pytest.raises(proto.FreezeError):
        pipeline.preflight(pipeline.PROFILES["dev"], run)
    with pytest.raises(proto.FreezeError):  # the world stage checks again before generating
        pipeline.stage_world(run)
    assert not (run.directory / "worlds").exists()


def test_a_missing_stage_implementation_fails_clearly(monkeypatch) -> None:
    with pytest.raises(pipeline.StageUnavailable, match="no_such_module.fit is not available"):
        pipeline.entry("fit", "no_such_module", "fit")
    module = types.ModuleType("half_built")
    module.ready = lambda: "done"

    def pending():
        raise NotImplementedError

    module.pending = pending
    monkeypatch.setitem(sys.modules, "half_built", module)
    assert pipeline.entry("x", "half_built", "ready")() == "done"
    with pytest.raises(pipeline.StageUnavailable, match="half_built.missing is not available"):
        pipeline.entry("x", "half_built", "missing")
    with pytest.raises(pipeline.StageUnavailable, match="half_built.pending is not implemented"):
        pipeline.entry("x", "half_built", "pending")()


def test_world_and_validate_stages_on_a_given_world(tmp_path: Path) -> None:
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="validate"), log=lambda _: None)
    world = read_result(run.results_dir / "world.json")
    validate = read_result(run.results_dir / "validate.json")
    assert world.tables["world.worlds"] == [{
        "seed": 0, "family": "baseline", "accounts": 19, "order_attempts": 26,
        "processor_approved": 24, "positive_labels": 13,
    }]
    assert validate.metrics["validate.worlds_valid"].value == 1
    assert world.versions["world"] == validate.versions["world"]
    manifest = json.loads((run.world_dir(run.worlds[0]) / "manifest.json").read_text())
    identity = validate.inputs["world:0-baseline"]
    assert identity == pipeline.manifest_identity(manifest)
    assert identity == pipeline.manifest_identity(manifest | {"code_commit": "0123abc"})
    lineage = json.loads((run.directory / "lineage.json").read_text())
    assert set(lineage["stages"]) == {"world", "validate"}
    outputs = lineage["stages"]["world"]["outputs"]
    assert "worlds/0-baseline/order_attempts.csv" in outputs
    assert "worlds/0-baseline/manifest.json" in outputs  # the raw file, commit and all
    assert lineage["stages"]["validate"]["inputs"]["world:0-baseline"] == identity
    assert "commit" in lineage["stages"]["world"]["code"]
    assert not (run.results_dir / "summary.json").exists()  # a partial run has no summary


def test_a_given_world_cannot_name_a_path_as_its_family(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = json.loads((MINI_WORLD / "manifest.json").read_text())
    for family in ("baseline/../../../../results", "Baseline", "unknown_family"):
        (source / "manifest.json").write_text(json.dumps(manifest | {"family": family}))
        with pytest.raises(pipeline.PipelineError):
            pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path / "runs", world=source)
    with pytest.raises(pipeline.PipelineError):
        pipeline.WorldRef(416, "../results")


def test_a_world_changed_after_it_was_written_is_refused(tmp_path: Path) -> None:
    """Resuming after a world file changed must not mix it with earlier results."""
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="validate"), log=lambda _: None)
    ref = run.worlds[0]
    path = run.world_dir(ref) / "order_attempts.csv"
    path.write_text(path.read_text().replace(",", ", ", 1))  # one byte of difference
    run.memory.clear()
    assert pipeline.input_hash(run, "world:0-baseline") is None
    assert pipeline.stale_inputs(run, [read_result(run.results_dir / "validate.json")])
    with pytest.raises(pipeline.PipelineError, match="differ from its manifest"):
        run.tables(ref)
    path.unlink()
    assert pipeline.input_hash(run, "world:0-baseline") is None


def test_a_changed_world_is_refused_before_the_database_is_touched(
    tmp_path: Path, monkeypatch
) -> None:
    """The load replaces the database's tables, so it must check the world first."""
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="validate"), log=lambda _: None)
    path = run.world_dir(run.worlds[0]) / "accounts.csv"
    lines = path.read_text().split("\n")
    assert lines[1].endswith(",1971")
    lines[1] = lines[1][:-len("1971")] + "1972"  # a contract-valid edit, manifest untouched
    path.write_text("\n".join(lines))
    loaded = []
    monkeypatch.setattr(pipeline, "_require_mysql", lambda: None)
    monkeypatch.setattr(pipeline, "_load_world_module", lambda: types.SimpleNamespace(
        load_tables=lambda tables: loaded.append(tables) or {"accounts": 1},
        load_world=lambda directory: loaded.append(directory) or {"accounts": 1}))
    run.memory.clear()
    with pytest.raises(pipeline.PipelineError, match="load: input world:0-baseline"):
        pipeline.execute(run, pipeline.select("load", "load"), log=lambda _: None)
    assert loaded == []  # refused before the loader ran


def test_a_stage_whose_inputs_change_while_it_runs_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    def evaluate(run: pipeline.Run) -> pipeline.StageOutput:
        replay = run.results_dir / "replay.json"
        replay.write_text(replay.read_text().replace("7", "8"))
        return pipeline.StageOutput()

    _fake_stages(monkeypatch, overrides={"evaluate": evaluate},
                 inputs={"evaluate": ["results/replay.json"]})
    run = _dev_run(tmp_path)
    with pytest.raises(pipeline.PipelineError, match="evaluate: inputs changed while it ran"):
        pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)


def test_a_given_world_written_another_way_is_stored_in_canonical_form(tmp_path: Path) -> None:
    """Rows in another order are the same world; changed rows are not."""
    source = tmp_path / "source"
    shutil.copytree(MINI_WORLD, source)
    path = source / "order_attempts.csv"
    header, *rows = path.read_text().rstrip("\n").split("\n")
    path.write_text("\n".join([header, *reversed(rows)]) + "\n")
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path / "runs", world=source)
    pipeline.execute(run, pipeline.select(until="validate"), log=lambda _: None)
    stored = run.world_dir(run.worlds[0]) / "order_attempts.csv"
    assert stored.read_bytes() == (MINI_WORLD / "order_attempts.csv").read_bytes()
    path.write_text("\n".join([header, *rows[1:]]) + "\n")  # a row fewer
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path / "other", world=source)
    with pytest.raises(pipeline.PipelineError, match="does not match its manifest"):
        pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)


def test_a_runs_own_world_cannot_be_given_as_its_source(tmp_path: Path) -> None:
    """Copying a world onto itself would delete it first."""
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    own = run.world_dir(run.worlds[0])
    again = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=own)
    with pytest.raises(pipeline.PipelineError, match="this run's own world directory"):
        pipeline.execute(again, pipeline.select(until="world"), log=lambda _: None)
    assert (own / "order_attempts.csv").read_bytes() == \
        (MINI_WORLD / "order_attempts.csv").read_bytes()


def test_an_unreadable_world_file_is_a_pipeline_error(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(MINI_WORLD, source)
    (source / "accounts.csv").chmod(0)
    try:  # unreadable when copied
        run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path / "runs", world=source)
        with pytest.raises(pipeline.PipelineError, match="world 0-baseline: cannot copy"):
            pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    finally:
        (source / "accounts.csv").chmod(0o644)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path / "other", world=source)
    pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    stored = run.world_dir(run.worlds[0]) / "accounts.csv"
    stored.chmod(0)
    try:  # unreadable when hashed
        assert pipeline.input_hash(run, "world:0-baseline") is None
        with pytest.raises(pipeline.PipelineError, match="missing or differs"):
            pipeline.execute(run, pipeline.select("validate", "validate"), log=lambda _: None)
    finally:
        stored.chmod(0o644)


def test_world_files_that_cannot_be_read_or_rewritten_are_pipeline_errors(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    shutil.copytree(MINI_WORLD, source)
    (source / "manifest.json").chmod(0)
    try:
        with pytest.raises(pipeline.PipelineError, match="unreadable manifest"):
            pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path / "runs", world=source)
    finally:
        (source / "manifest.json").chmod(0o644)
    path = source / "order_attempts.csv"  # valid rows in another order: rewritten canonically
    header, *rows = path.read_text().rstrip("\n").split("\n")
    path.write_text("\n".join([header, *reversed(rows)]) + "\n")

    def refuse(tables, directory):
        raise PermissionError("read-only")

    monkeypatch.setattr(world_module, "write_world", refuse)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path / "runs", world=source)
    with pytest.raises(pipeline.PipelineError, match="world 0-baseline: cannot rewrite it"):
        pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)


def _given_world(tmp_path: Path) -> tuple[pipeline.Run, Path]:
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    return run, run.world_dir(run.worlds[0])


def _order_before_signup(tables: dict) -> None:
    orders, accounts = tables["order_attempts"], tables["accounts"]
    signup = accounts.set_index("user_id").loc[orders.loc[0, "user_id"], "created_at"]
    orders.loc[0, ["occurred_at", "known_at"]] = signup - pd.Timedelta(days=1)


def test_validation_rejects_a_world_that_no_longer_matches_its_manifest(tmp_path: Path) -> None:
    run, directory = _given_world(tmp_path)
    tables = world_module.read_world(directory)
    _order_before_signup(tables)
    world_module.write_world({"order_attempts": tables["order_attempts"]}, directory)
    with pytest.raises(pipeline.PipelineError, match="differs from its manifest"):
        pipeline.execute(run, pipeline.select("validate", "validate"), log=lambda _: None)


def test_validation_rejects_a_world_that_breaks_chronology(tmp_path: Path) -> None:
    run, directory = _given_world(tmp_path)
    tables = world_module.read_world(directory)
    _order_before_signup(tables)
    world_module.write_world(tables, directory)
    old = json.loads((directory / "manifest.json").read_text())
    manifest = world_module.build_manifest(
        tables, generator_version=old["generator_version"], config={}, seed=old["seed"],
        family=old["family"], **old["horizon"],
    )
    world_module.write_manifest(manifest, directory / "manifest.json")
    with pytest.raises(world_module.WorldError, match="order_before_account"):
        pipeline.execute(run, pipeline.select("validate", "validate"), log=lambda _: None)


# ---------------------------------------------------------------- whole runs with stand-in stages
def _fake_stages(monkeypatch, *, value: int = 7, overrides: dict | None = None,
                 inputs: dict | None = None) -> None:
    overrides, inputs = overrides or {}, inputs or {}

    def make(name: str):
        def function(run: pipeline.Run) -> pipeline.StageOutput:
            if name in overrides:
                return overrides[name](run)
            if name == "world":
                for ref in run.all_worlds:
                    target = run.world_dir(ref)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "manifest.json").write_text(json.dumps(
                        {"seed": ref.seed, "family": ref.family, "code_commit": "abc",
                         "tables": {"x": value}}))
            metric = Metric(value=value, unit="count", population=f"{name} things",
                            window="test")
            return pipeline.StageOutput(metrics={f"{name}.things": metric})
        return function

    def reads(name: str):
        return lambda run: list(inputs.get(name, ["config/policy.yaml"]))

    stages = tuple(pipeline.Stage(stage.name, make(stage.name), stage.versions, stage.description,
                                  reads(stage.name))
                   for stage in pipeline.STAGES)
    monkeypatch.setattr(pipeline, "STAGES", stages)


def test_a_complete_run_assembles_its_summary_and_reruns_byte_for_byte(
    tmp_path: Path, monkeypatch
) -> None:
    _fake_stages(monkeypatch)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[416],
                            families=["baseline"])
    pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)
    summary = read_summary(run.results_dir / "summary.json")
    assert set(summary["stages"]) == set(pipeline.STAGE_NAMES)
    assert summary["metrics"]["evaluate.things"]["value"] == 7
    assert set(summary["versions"]) == {"world", "features", "policy", "protocol", "benchmark",
                                        "models", "tuning"}
    first = {path.name: path.read_bytes() for path in run.results_dir.iterdir()}
    first_lineage = (run.directory / "lineage.json").read_text()
    pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)
    assert {path.name: path.read_bytes() for path in run.results_dir.iterdir()} == first
    assert json.loads(first_lineage)["stages"].keys() == set(pipeline.STAGE_NAMES)
    for path in run.results_dir.iterdir():  # no clock or commit in a result file
        text = path.read_text()
        assert "started_at" not in text and "commit" not in text


def test_a_rerun_stage_on_a_different_world_cannot_join_the_summary(
    tmp_path: Path, monkeypatch
) -> None:
    _fake_stages(monkeypatch)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[416],
                            families=["baseline"])
    pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)
    _fake_stages(monkeypatch, value=8)  # the world changes ...
    pipeline.execute(run, pipeline.select("world", "world"), log=lambda _: None)
    with pytest.raises(VersionMismatch, match="world"):  # ... but later stages are stale
        pipeline.execute(run, pipeline.select("llm", "llm"), log=lambda _: None)


def _dev_run(tmp_path: Path) -> pipeline.Run:
    return pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[416],
                             families=["baseline"])


def test_a_stale_evaluation_cannot_join_a_fresh_replay(tmp_path: Path, monkeypatch) -> None:
    def replay_with(value: int):
        def replay(run: pipeline.Run) -> pipeline.StageOutput:
            item = Metric(value=value, unit="count", population="replayed", window="test")
            return pipeline.StageOutput(metrics={"replay.things": item})
        return replay

    def evaluate(run: pipeline.Run) -> pipeline.StageOutput:
        replayed = read_result(run.results_dir / "replay.json").metrics["replay.things"].value
        item = Metric(value=replayed, unit="count", population="evaluated", window="test")
        return pipeline.StageOutput(metrics={"evaluate.things": item})

    reads = {"evaluate": ["results/replay.json"]}
    _fake_stages(monkeypatch, overrides={"replay": replay_with(10), "evaluate": evaluate},
                 inputs=reads)
    run = _dev_run(tmp_path)
    pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)
    _fake_stages(monkeypatch, overrides={"replay": replay_with(20), "evaluate": evaluate},
                 inputs=reads)
    pipeline.execute(run, pipeline.select("replay", "replay"), log=lambda _: None)
    with pytest.raises(pipeline.PipelineError, match="evaluate: results/replay.json changed"):
        pipeline.execute(run, pipeline.select("llm", "llm"), log=lambda _: None)
    pipeline.execute(run, pipeline.select("evaluate"), log=lambda _: None)  # rerun from there
    assert read_summary(run.results_dir / "summary.json")["metrics"]["evaluate.things"][
        "value"] == 20


def test_replays_from_other_models_cannot_join_the_summary(tmp_path: Path, monkeypatch) -> None:
    def fit_with(text: str):
        def fit(run: pipeline.Run) -> pipeline.StageOutput:
            (run.stage_dir("fit") / "model.json").write_text(text)
            run.memory["scorers"] = {416: {}}
            return pipeline.StageOutput()
        return fit

    _fake_stages(monkeypatch, overrides={"fit": fit_with('{"c": 1}')})
    run = _dev_run(tmp_path)
    pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)
    _fake_stages(monkeypatch, overrides={"fit": fit_with('{"c": 2}')})
    pipeline.execute(run, pipeline.select("fit", "fit"), log=lambda _: None)
    with pytest.raises(VersionMismatch, match="models"):
        pipeline.execute(run, pipeline.select("llm", "llm"), log=lambda _: None)


def test_models_live_in_memory_so_later_stages_need_the_fit(tmp_path: Path) -> None:
    with pytest.raises(pipeline.PipelineError, match="run from the fit stage"):
        _dev_run(tmp_path).scorers(416)


def test_run_names_stay_inside_the_runs_directory(tmp_path: Path) -> None:
    for name in ("..", "../results", "a/b", ".hidden", ""):
        with pytest.raises(pipeline.PipelineError, match="plain directory name"):
            pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, name=name or "..x/")
    assert pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, name="dev-2.b").directory \
        == tmp_path / "dev-2.b"


def test_any_override_turns_the_final_profile_into_a_trial_run(tmp_path: Path) -> None:
    final = pipeline.PROFILES["final"]
    for overrides in ({"scale": 0.1}, {"seeds": [1]}, {"families": ["baseline"]},
                      {"world": MINI_WORLD}):
        run = pipeline.make_run(final, runs=tmp_path, **overrides)
        assert run.results_dir.is_relative_to(tmp_path) and run.docs_dir.is_relative_to(tmp_path)


def test_untracked_files_make_the_tree_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=T", "-c",
                        "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
                       check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("runs/\n")
    git("add", "-A")
    git("commit", "-q", "-m", "start")
    assert pipeline.code_identity(tmp_path)["dirty"] is False
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "lineage.json").write_text("{}")
    assert pipeline.code_identity(tmp_path)["dirty"] is False  # ignored output
    (tmp_path / "replay.py").write_text("print('uncommitted')\n")
    assert pipeline.code_identity(tmp_path)["dirty"] is True
    assert pipeline.code_identity(tmp_path / "missing")["dirty"] is None


def test_the_summary_needs_every_stage(tmp_path: Path, monkeypatch) -> None:
    _fake_stages(monkeypatch)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[416],
                            families=["baseline"])
    pipeline.execute(run, pipeline.select(until="replay"), log=lambda _: None)
    with pytest.raises(pipeline.PipelineError, match="not complete"):
        pipeline.execute(run, pipeline.select("llm", "llm"), log=lambda _: None)


def _committed_results() -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in pipeline.RESULTS.rglob("*") if path.is_file()}


def test_a_development_run_never_writes_the_committed_results(
    tmp_path: Path, monkeypatch
) -> None:
    _fake_stages(monkeypatch)
    committed = _committed_results()
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[416],
                            families=["baseline"])
    pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)
    assert run.results_dir.is_relative_to(tmp_path)
    assert _committed_results() == committed


def test_the_benchmark_version_follows_the_benchmark_manifests(tmp_path: Path,
                                                               monkeypatch) -> None:
    run = _dev_run(tmp_path)
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    assert pipeline.versions(run, ("benchmark",)) == {"benchmark": "none"}
    benchmark = tmp_path / "llm" / "eval" / "benchmarks" / "2026-08-dev"
    benchmark.mkdir(parents=True)
    (benchmark / "MANIFEST.json").write_text('{"cases": 1}')
    first = pipeline.versions(run, ("benchmark",))["benchmark"]
    assert first != "none"
    (benchmark / "MANIFEST.json").write_text('{"cases": 2}')
    assert pipeline.versions(run, ("benchmark",))["benchmark"] != first


def test_the_protocol_states_the_world_size_the_generator_uses() -> None:
    from core import config

    assert proto.load_protocol().raw["world_size"]["target_orders"] == \
        config.load("world")["volume"]["target_orders"]


def test_world_version_ignores_the_commit_but_not_the_content(tmp_path: Path) -> None:
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[416],
                            families=["baseline"])
    target = run.world_dir(run.worlds[0])
    target.mkdir(parents=True)
    manifest = {"seed": 416, "family": "baseline", "code_commit": "a", "tables": {"x": 1}}
    (target / "manifest.json").write_text(json.dumps(manifest))
    first = pipeline.world_version(run)
    (target / "manifest.json").write_text(json.dumps(manifest | {"code_commit": "b"}))
    assert pipeline.world_version(run) == first
    (target / "manifest.json").write_text(json.dumps(manifest | {"tables": {"x": 2}}))
    assert pipeline.world_version(run) != first


@dataclass
class OtherStageOutput:  # another module's stage output, with the same fields
    metrics: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)


def _stage_module(monkeypatch, calls: list, *, incumbent: object = "fitted") -> types.ModuleType:
    """A stand-in for queue_sim.stage with its signatures, recording the calls;
    ``incumbent`` None: tuning found the incumbent no feasible point."""
    module = types.ModuleType("queue_sim.stage")
    module.TUNING_HISTORIES = ("policy", "frozen", "shortlist")

    def tune(run, **options):
        calls.append(("tune", options))
        run.memory["tuned"] = {0: {"approve_all": "fitted", "incumbent_rules": incumbent}}
        return OtherStageOutput(tables={"tune.chosen": []})

    def replay(run):
        calls.append(("replay",))
        run.memory["incumbent"] = {}
        unfit = {"seed": 0, "family": "baseline", "policy": "incumbent_rules",
                 "capacity_level": "base", "evaluated": False}
        return OtherStageOutput(tables={"replay.outcomes": [outcome(0, "approve_all", 0),
                                                            unfit]})

    def routing_frame(run, ref):
        calls.append(("routing_frame", ref))
        return pd.DataFrame(columns=list(pipeline.ALERT_COLUMNS))

    def review_decisions(run, ref):
        calls.append(("review_decisions", ref, "incumbent" in run.memory))
        return pd.DataFrame({
            "order_id": [1, 2],
            "decision_at": pd.to_datetime(["2025-01-01 10:00", "2025-01-02 09:30"]),
            "disposition": ["hold", "clear"],
        })

    module.tune, module.replay, module.review_decisions = tune, replay, review_decisions
    module.routing_frame = routing_frame
    monkeypatch.setitem(sys.modules, "queue_sim.stage", module)
    return module


def test_a_policy_left_unevaluated_keeps_its_row_through_the_result_file(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []
    _stage_module(monkeypatch, calls)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    run.memory["scorers"] = {0: {}}
    pipeline.execute(run, pipeline.select("tune", "replay"), log=lambda _: None)
    rows = read_result(run.results_dir / "replay.json").tables["replay.outcomes"]
    assert set(rows[0]) == set(rows[1]) and rows[1]["evaluated"] is False
    assert rows[1]["net_cents"] is None and rows[1]["layout"] is None
    # what the replay wrote is what evaluate reads
    full = [outcome(seed, policy, 0) for seed in (1, 2) for policy in POLICIES
            if (seed, policy) != (2, "hybrid")]
    full.append({"seed": 2, "family": "baseline", "policy": "hybrid", "capacity_level": "base",
                 "evaluated": False})
    stored = pipeline.full_rows(full)
    metrics, _ = evaluate(stored, seeds=(1, 2))
    assert metrics["evaluate.net_contribution.baseline.base.hybrid"].value is None


def test_an_incumbent_with_no_feasible_point_leaves_no_decisions_or_alerts(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []
    _stage_module(monkeypatch, calls, incumbent=None)
    written: list = []
    monkeypatch.setattr(pipeline, "_require_mysql", lambda: None)
    monkeypatch.setattr(pipeline, "write_alerts", lambda frame: written.append(frame) or 0)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    stale = run.world_dir(run.worlds[0]) / pipeline.REVIEW_DECISIONS
    stale.write_bytes(b"from an earlier run")
    run.memory["scorers"] = {0: {}}
    pipeline.execute(run, pipeline.select("tune", "alerts"), log=lambda _: None)
    assert [call[0] for call in calls] == ["tune", "replay"]
    assert not stale.exists()
    assert any("no review decisions for 0-baseline" in note
               for note in read_result(run.results_dir / "replay.json").notes)
    with pytest.raises(pipeline.PipelineError, match="no feasible operating point"):
        pipeline.review_decisions(run, run.worlds[0])
    assert len(written) == 1 and written[0].empty  # the alerts table is emptied
    assert list(written[0].columns) == list(pipeline.ALERT_COLUMNS)
    assert read_result(run.results_dir / "alerts.json").notes[0].startswith("no alerts for")


def test_replay_and_alerts_depend_on_the_tuning_they_used(tmp_path: Path, monkeypatch) -> None:
    reads = {stage.name: stage.inputs for stage in pipeline.STAGES}
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    assert "results/tune.json" in reads["replay"](run)
    assert "results/tune.json" in reads["alerts"](run)
    _fake_stages(monkeypatch)
    run = _dev_run(tmp_path)
    pipeline.execute(run, list(pipeline.STAGES), log=lambda _: None)
    retuned = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, seeds=[416],
                                families=["baseline"], tuning_history="frozen")
    pipeline.execute(retuned, pipeline.select("tune", "tune"), log=lambda _: None)
    with pytest.raises(VersionMismatch, match="tuning"):  # the old replay cannot join it
        pipeline.execute(retuned, pipeline.select("llm", "llm"), log=lambda _: None)


def test_tune_and_replay_call_the_stage_module(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _stage_module(monkeypatch, calls)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD,
                            workers=3)
    assert run.workers == 3  # read by queue_sim.stage.workers(run)
    pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    with pytest.raises(pipeline.PipelineError, match="fitted models are kept in memory"):
        pipeline.execute(run, pipeline.select("tune", "tune"), log=lambda _: None)
    with pytest.raises(pipeline.PipelineError, match="results/tune.json is missing"):
        pipeline.execute(run, pipeline.select("replay", "replay"), log=lambda _: None)
    run.memory["scorers"] = {0: {}}  # stand-in for the fit stage's models
    pipeline.execute(run, pipeline.select("tune", "tune"), log=lambda _: None)
    tuned = run.memory.pop("tuned")  # a later process has the result file, not the policies
    with pytest.raises(pipeline.PipelineError, match="tuned policies are kept in memory"):
        pipeline.execute(run, pipeline.select("replay", "replay"), log=lambda _: None)
    run.memory["tuned"] = tuned
    calls.clear()
    pipeline.execute(run, pipeline.select("tune", "replay"), log=lambda _: None)
    ref = run.worlds[0]
    # review decisions are asked for after the replay, which keeps the incumbent's runs
    assert calls == [("tune", {}), ("replay",), ("review_decisions", ref, True)]
    kept = pipeline.review_decisions(run, ref)
    assert list(kept["disposition"]) == ["hold", "clear"]
    assert str(kept["decision_at"].dtype).startswith("datetime64")
    lineage = json.loads((run.directory / "lineage.json").read_text())
    assert "worlds/0-baseline/review_decisions.pkl" in lineage["stages"]["replay"]["outputs"]
    assert read_result(run.results_dir / "tune.json").notes[-1] == \
        "tuning history: the tuning default"


def test_the_tuning_history_comes_from_the_protocol_or_an_override(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []
    _stage_module(monkeypatch, calls)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD,
                            tuning_history="shortlist")  # any mode tune lists passes through
    run.memory["scorers"] = {0: {}}
    pipeline.stage_tune(run)
    assert calls[-1] == ("tune", {"history": "shortlist"})
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    run.protocol.raw.setdefault("tuning", {})["history"] = "frozen"
    run.memory["scorers"] = {0: {}}
    assert "tuning history: frozen" in pipeline.stage_tune(run).notes
    assert calls[-1] == ("tune", {"history": "frozen"})
    run.protocol.raw["tuning"]["history"] = "TO_COMPLETE_AT_FREEZE"
    with pytest.raises(pipeline.PipelineError, match="protocol tuning.history must be one of"):
        pipeline.preflight(pipeline.PROFILES["dev"], run)  # before anything is written
    run.protocol.raw["tuning"]["history"] = "cached"  # not a mode tune lists
    with pytest.raises(pipeline.PipelineError, match="'shortlist'"):
        pipeline.stage_tune(run)
    with pytest.raises(pipeline.PipelineError, match="lower-case name"):
        pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, tuning_history="Frozen ")
    with pytest.raises(pipeline.PipelineError, match="workers"):
        pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, workers=0)
    final = pipeline.make_run(pipeline.PROFILES["final"], runs=tmp_path, tuning_history="policy")
    assert final.results_dir.is_relative_to(tmp_path)  # an override is a trial run
    assert pipeline.make_run(pipeline.PROFILES["final"], runs=tmp_path,
                             workers=4).results_dir == pipeline.RESULTS


def test_output_that_does_not_fit_a_result_file_names_the_stage(tmp_path: Path,
                                                                 monkeypatch) -> None:
    def replay(run: pipeline.Run) -> pipeline.StageOutput:
        return pipeline.StageOutput(tables={"replay.outcomes": [{"reviews_P0": 1}]})

    _fake_stages(monkeypatch, overrides={"replay": replay})
    run = _dev_run(tmp_path)
    with pytest.raises(pipeline.PipelineError, match="replay: its output does not fit"):
        pipeline.execute(run, pipeline.select(until="replay"), log=lambda _: None)


def test_another_modules_stage_output_is_converted_or_refused() -> None:
    item = Metric(value=1, unit="count", population="things", window="test")
    converted = pipeline.stage_output(OtherStageOutput(metrics={"x.n": item}, notes=["a"]), "x")
    assert isinstance(converted, pipeline.StageOutput) and converted.metrics["x.n"] == item
    with pytest.raises(pipeline.PipelineError, match="not core.results.Metric"):
        pipeline.stage_output(OtherStageOutput(metrics={"x.n": 1}), "x")
    with pytest.raises(pipeline.PipelineError, match="declare them"):
        pipeline.stage_output(OtherStageOutput(inputs={"policy": Path("config/policy.yaml")}),
                              "x")
    with pytest.raises(pipeline.PipelineError, match="not a stage output"):
        pipeline.stage_output({"tables": {}}, "x")


# ---------------------------------------------------------------- evaluate
POLICIES = ("approve_all", "incumbent_rules", "hybrid")
BASE = ("base", "current", "policy", "evidence", "verification")
FROZEN = ("base", "current", "frozen", "evidence", "verification")


def outcome(seed: int, policy: str, net: int, *, family: str = "baseline", cell=BASE,
            held: int = 10, legit: int = 10_000, used: float = 300,
            available: float = 400) -> dict:
    """A replay outcome row with queue_sim.outcomes' key and outcome columns."""
    return {
        "seed": seed, "family": family, "policy": policy,
        **dict(zip(pipeline.CELL_COLUMNS, cell, strict=True)),
        "evaluated": True, "policy_version": f"{policy}-v1",
        "orders": 20_000, "net_cents": net, "loss_cents": 5_000, "gmv_cents": 1_000_000,
        "legitimate_held": held, "legitimate_declined": 0, "legitimate_cancelled": 0,
        "legitimate_orders": legit, "friction_cost_cents": 0,
        "review_minutes_used": used, "available_minutes": available,
        "review_band": int(policy != "approve_all"),
        "decided_after_shipping": 2, "wait_p90_minutes": 30,
        **{f"{column}_{p}": 0 for column in ("reviews", "sla_met")
           for p in ("p0", "p1", "p2", "p3")},
    }


def outcomes(cell=BASE) -> list[dict]:
    rows = []
    for seed, (approve, incumbent, hybrid) in {
        1: (0, 1_000, 1_500), 2: (0, 2_000, 1_900), 3: (0, 1_000, 1_400),
    }.items():
        rows += [outcome(seed, "approve_all", approve, held=0, cell=cell),
                 outcome(seed, "incumbent_rules", incumbent, held=20, cell=cell),
                 outcome(seed, "hybrid", hybrid, held=12 + seed, cell=cell)]
    return rows


def evaluate(rows: list[dict], seeds=(1, 2, 3), families=("baseline",), policies=POLICIES,
             cells=(BASE,)):
    return pipeline.evaluate_outcomes(rows, seeds, families, policies, cells, "current")


def test_evaluate_pairs_policies_by_seed() -> None:
    metrics, table = evaluate(outcomes())
    net = metrics["evaluate.net_contribution.vs_incumbent_rules.baseline.base.hybrid"]
    assert net.seeds.per_seed == {1: 500, 2: -100, 3: 400}
    assert net.value == pytest.approx(800 / 3)
    assert net.unit == "cents" and net.window == "test"
    assert net.seeds.sign_count == (2, 1, 0)
    vs_approve = metrics["evaluate.net_contribution.vs_approve_all.baseline.base.hybrid"]
    assert vs_approve.seeds.per_seed == {1: 1_500, 2: 1_900, 3: 1_400}
    # ratios: pooled over seeds for the value, per seed for the spread
    held = metrics["evaluate.legitimate_held_per_10k.baseline.base.hybrid"]
    assert (held.numerator, held.denominator) == (13 + 14 + 15, 30_000)
    assert held.value == pytest.approx(10_000 * 42 / 30_000)
    assert held.seeds.per_seed == {1: 13.0, 2: 14.0, 3: 15.0}
    held_vs = metrics["evaluate.legitimate_held_per_10k.vs_incumbent_rules.baseline.base.hybrid"]
    assert held_vs.seeds.per_seed == {1: -7.0, 2: -6.0, 3: -5.0}
    # a share's paired difference is in basis points
    used = metrics["evaluate.review_minutes_used_share.vs_approve_all.baseline.base.hybrid"]
    assert used.unit == "bps" and used.seeds.per_seed == {1: 0.0, 2: 0.0, 3: 0.0}
    assert "evaluate.net_contribution.vs_approve_all.baseline.base.approve_all" not in metrics
    row = next(r for r in table if r["policy"] == "hybrid")
    assert (row["capacity"], row["capacity_level"], row["history"]) == ("base", "base", "policy")
    assert row["net_contribution_cents"] == pytest.approx(1_600)
    assert row["net_contribution_vs_incumbent_rules_cents"] == pytest.approx(800 / 3)
    assert row["review_minutes_used_share_vs_approve_all_bps"] == 0.0
    assert row["net_contribution_vs_incumbent_rules_positive_seeds"] == 2


def test_policies_pair_within_their_replay_variant() -> None:
    frozen = [r | {"net_cents": r["net_cents"] * 2} for r in outcomes(FROZEN)]
    metrics, table = evaluate(outcomes() + frozen, cells=(BASE, FROZEN))
    main = metrics["evaluate.net_contribution.vs_incumbent_rules.baseline.base.hybrid"]
    other = metrics["evaluate.net_contribution.vs_incumbent_rules.baseline.base_frozen_history."
                    "hybrid"]
    assert other.seeds.per_seed == {seed: 2 * value for seed, value in main.seeds.per_seed.items()}
    assert {row["capacity"] for row in table} == {"base", "base_frozen_history"}


def test_cells_are_named_by_level_layout_and_variant() -> None:
    name = pipeline.capacity_name
    assert name(BASE, "current") == "base"
    assert name(("high", "current", "policy", "evidence", "verification"), "current") == "high"
    assert name(("base", "evening", "policy", "evidence", "verification"), "current") == \
        "redesigned_layout"
    assert name(("low", "evening", "policy", "evidence", "verification"), "current") == \
        "low_redesigned_layout"
    assert name(FROZEN, "current") == "base_frozen_history"
    assert name(("base", "current", "policy", "perfect", "verification"), "current") == \
        "base_perfect_reviewer"
    assert name(("base", "current", "policy", "evidence", "verification_weak"), "current") == \
        "base_weak_verification"


def test_the_expected_cells_are_the_replays_staffing_and_variants() -> None:
    from core import config

    stage = pytest.importorskip("queue_sim.stage")
    policy_cfg = config.load("policy")
    cells = pipeline.expected_cells(policy_cfg)
    staffing = stage.staffing(policy_cfg)
    base = stage.base_staffing(policy_cfg)
    expected = {(base.level, base.layout, *variant) for variant in stage.VARIANTS} | {
        (staff.level, staff.layout, *stage.VARIANTS[0]) for staff in staffing}
    assert set(cells) == expected  # every variant at the base, the main one elsewhere
    assert len(cells) == len(stage.VARIANTS) + len(staffing) - 1
    assert pipeline.main_cell(cells, policy_cfg["roster"]["layout"]) == \
        (base.level, base.layout, *stage.VARIANTS[0])


WEAK = ("base", "current", "policy", "evidence", "verification_weak")
PERFECT = ("base", "current", "policy", "perfect", "verification")
LOW = ("low", "current", "policy", "evidence", "verification")
EVENING = ("base", "evening", "policy", "evidence", "verification")


def test_the_rule_is_applied_in_the_primary_cell_and_each_sensitivity_alone() -> None:
    cells = (BASE, FROZEN, PERFECT, WEAK, LOW, EVENING,
             ("low", "evening", "policy", "evidence", "verification"),  # two at once
             ("low", "current", "policy", "evidence", "verification_weak"))
    where = pipeline.operating_cells(("baseline", "lag_half", "acquisition_surge"), cells,
                                     "current", (500, 4_500))
    assert [(c.name, c.varies, c.family, c.capacity, c.ltv_cents) for c in where] == [
        ("primary", "primary", "baseline", "base", None),
        ("acquisition_surge", "family", "acquisition_surge", "base", None),
        ("lag_half", "family", "lag_half", "base", None),
        ("base_weak_verification", "verification", "baseline", "base_weak_verification", None),
        ("low", "allotment", "baseline", "low", None),
        ("redesigned_layout", "layout", "baseline", "redesigned_layout", None),
        ("ltv_5_usd", "ltv", "baseline", "base", 500),
        ("ltv_45_usd", "ltv", "baseline", "base", 4_500),
    ]
    assert where[0].where == dict(zip(pipeline.CELL_COLUMNS, BASE, strict=True))
    with pytest.raises(pipeline.PipelineError, match="baseline"):
        pipeline.operating_cells(("acquisition_surge",), cells, "current")
    with pytest.raises(pipeline.PipelineError, match="no base cell"):
        pipeline.operating_cells(("baseline",), (LOW, FROZEN), "current")


def test_a_lag_family_is_evaluated_at_the_main_cell_alone() -> None:
    rows = outcomes() + outcomes(FROZEN)
    lag = [r | {"family": "lag_half"} for r in outcomes()]
    metrics, table = pipeline.evaluate_outcomes(
        rows + lag, (1, 2, 3), ("baseline", "lag_half"), POLICIES, (BASE, FROZEN), "current",
        base_only=("lag_half",))
    assert "evaluate.net_contribution.lag_half.base.hybrid" in metrics
    assert "evaluate.net_contribution.lag_half.base_frozen_history.hybrid" not in metrics
    assert "evaluate.net_contribution.baseline.base_frozen_history.hybrid" in metrics
    with pytest.raises(pipeline.PipelineError, match="unexpected"):  # replayed in every cell
        pipeline.evaluate_outcomes(
            rows + lag + [r | {"family": "lag_half"} for r in outcomes(FROZEN)], (1, 2, 3),
            ("baseline", "lag_half"), POLICIES, (BASE, FROZEN), "current",
            base_only=("lag_half",))
    with pytest.raises(pipeline.PipelineError, match="missing"):
        pipeline.evaluate_outcomes(rows + lag, (1, 2, 3), ("baseline", "lag_half"), POLICIES,
                                   (BASE, FROZEN), "current")


def test_the_evaluate_stage_publishes_the_recommendation_and_its_flips(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "expected_cells", lambda policy_cfg: (BASE, FROZEN, WEAK))
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    policies = tuple(run.protocol.raw["policies"])
    net = {"approve_all": 0, "incumbent_rules": 1_000_000, "hybrid": 1_500_000}
    rows = [outcome(0, policy, net.get(policy, 900_000), cell=cell)
            for cell in (BASE, FROZEN, WEAK) for policy in policies]
    write_result(StageResult(stage="replay", versions={}, inputs={}, metrics={},
                             tables={"replay.outcomes": rows}), run.results_dir)
    output = pipeline.stage_evaluate(run)
    StageResult(stage="evaluate", versions={}, inputs={}, metrics=output.metrics,
                tables=output.tables, notes=output.notes)  # fits a result file
    flips = output.tables["evaluate.flips"]
    assert [r["cell"] for r in flips] == ["primary", "base_weak_verification", "ltv_5_usd",
                                          "ltv_45_usd"]
    assert {r["recommended"] for r in flips} == {"hybrid"}
    assert output.metrics["evaluate.recommendation.holds"].value == 1
    assert any("1 seed, fewer than the 8" in note for note in output.notes)
    assert any("no p-value" in note for note in output.notes)
    assert "evaluate.net_contribution.vs_incumbent_rules.baseline.base.hybrid" in output.metrics
    run.protocol = proto.parse_protocol(
        {**run.protocol.raw, "reporting": {"recommendation_rule": proto.PLACEHOLDER}})
    with pytest.raises(pipeline.PipelineError, match="evaluate: the protocol's recommendation"):
        pipeline.stage_evaluate(run)
    with pytest.raises(pipeline.PipelineError, match="recommendation rule is incomplete"):
        pipeline.preflight(pipeline.PROFILES["dev"], run)


def test_a_policy_with_no_feasible_point_is_not_evaluated_on_that_seed() -> None:
    rows = [r for r in outcomes() if not (r["policy"] == "hybrid" and r["seed"] == 2)]
    rows.append({"seed": 2, "family": "baseline", "policy": "hybrid", "capacity_level": "base",
                 "evaluated": False})
    metrics, table = evaluate(rows)
    net = metrics["evaluate.net_contribution.baseline.base.hybrid"]
    assert net.value is None and "[2]" in net.note
    held = metrics["evaluate.legitimate_held_per_10k.baseline.base.hybrid"]
    assert held.value is None and (held.numerator, held.denominator) == (13 + 15, 20_000)
    paired = metrics["evaluate.net_contribution.vs_incumbent_rules.baseline.base.hybrid"]
    assert paired.value is None and "[2]" in paired.note
    assert metrics["evaluate.net_contribution.vs_approve_all.baseline.base.incumbent_rules"] \
        .value == pytest.approx(4_000 / 3)
    row = next(r for r in table if r["policy"] == "hybrid")
    assert row["net_contribution_cents"] is None
    assert row["net_contribution_vs_incumbent_rules_positive_seeds"] is None
    # the incumbent unfit on a seed withholds every comparison against it there
    rows = [r for r in outcomes() if not (r["policy"] == "incumbent_rules" and r["seed"] == 1)]
    rows.append({"seed": 1, "family": "baseline", "policy": "incumbent_rules",
                 "capacity_level": "base", "evaluated": False})
    metrics, _ = evaluate(rows)
    assert metrics["evaluate.net_contribution.vs_incumbent_rules.baseline.base.hybrid"] \
        .value is None
    assert metrics["evaluate.net_contribution.vs_approve_all.baseline.base.hybrid"].value \
        == pytest.approx(4_800 / 3)
    with pytest.raises(pipeline.PipelineError, match="unexpected"):  # unfit, yet replayed
        evaluate(outcomes() + [rows[-1]])
    unfit_reference = [r for r in outcomes() if not (r["policy"] == "approve_all"
                                                     and r["seed"] == 1)]
    unfit_reference.append({"seed": 1, "family": "baseline", "policy": "approve_all",
                            "capacity_level": "base", "evaluated": False})
    with pytest.raises(pipeline.PipelineError, match="always be evaluated"):
        evaluate(unfit_reference)


def test_fractional_quantities_are_pooled_without_truncation() -> None:
    rows = [r | {"review_minutes_used": 0.75, "available_minutes": 1.5} for r in outcomes()]
    metrics, _ = evaluate(rows)
    used = metrics["evaluate.review_minutes_used_share.baseline.base.hybrid"]
    assert (used.numerator, used.denominator, used.value) == (2.25, 4.5, 0.5)


def test_evaluate_refuses_incomplete_or_repeated_outcomes() -> None:
    rows = outcomes()
    with pytest.raises(pipeline.PipelineError, match="missing"):
        evaluate(rows[:-1])
    with pytest.raises(pipeline.PipelineError, match="repeated"):
        evaluate(rows + rows[:1])
    with pytest.raises(pipeline.PipelineError, match="lack columns"):
        evaluate([{k: v for k, v in r.items() if k != "gmv_cents"} for r in rows])
    with pytest.raises(pipeline.PipelineError, match="lack columns"):
        evaluate([{k: v for k, v in r.items() if k != "evaluated"} for r in rows])
    with pytest.raises(pipeline.PipelineError, match="true or false"):
        evaluate([r | {"evaluated": 1} for r in rows])
    with pytest.raises(pipeline.PipelineError, match="zero legitimate_orders"):
        evaluate([r | {"legitimate_orders": 0} for r in rows])
    without_incumbent = [r for r in rows if r["policy"] != "incumbent_rules"]
    with pytest.raises(pipeline.PipelineError, match="missing"):
        evaluate(without_incumbent)
    with pytest.raises(pipeline.PipelineError, match="references"):
        evaluate(without_incumbent, policies=("approve_all", "hybrid"))
    with pytest.raises(pipeline.PipelineError, match="missing"):  # a family left out
        evaluate(rows, families=("baseline", "fraud_mix_shift"))
    frozen = [r for r in outcomes(FROZEN) if r["policy"] != "hybrid"]
    with pytest.raises(pipeline.PipelineError, match="missing"):  # hybrid absent in a variant
        evaluate(rows + frozen, cells=(BASE, FROZEN))
    with pytest.raises(pipeline.PipelineError, match="missing"):  # a whole variant absent
        evaluate(rows, cells=(BASE, FROZEN))
    with pytest.raises(pipeline.PipelineError, match="unexpected"):  # a variant not expected
        evaluate(rows + outcomes(FROZEN))
    with pytest.raises(pipeline.PipelineError, match="unexpected"):
        evaluate(rows + [outcome(4, "hybrid", 0)])
    with pytest.raises(pipeline.PipelineError, match="share a name"):
        evaluate(rows, cells=(BASE, BASE))


def test_pooling_per_seed_metrics() -> None:
    def rate(n: int, d: int) -> Metric:
        return Metric.from_ratio(n, d, population="fit-window orders", window="fit")

    pooled = pipeline.pool_over_seeds({1: rate(1, 4), 2: rate(3, 4)}, "x")
    assert (pooled.value, pooled.numerator, pooled.denominator) == (0.5, 4, 8)
    assert pooled.seeds.per_seed == {1: 0.25, 2: 0.75}
    score = Metric(value=0.8, unit="score", population="AP, calibration month",
                   window="calibration")
    other = Metric(**{**score.__dict__, "value": 0.6})
    mean = pipeline.pool_over_seeds({1: score, 2: other}, "ap")
    assert mean.value == pytest.approx(0.7)
    with pytest.raises(pipeline.PipelineError, match="disagree"):
        pipeline.pool_over_seeds({1: rate(1, 4), 2: score}, "x")


def test_pooling_keeps_support_when_a_seed_is_withheld() -> None:
    def rate(n: int, d: int) -> Metric:
        return Metric.from_ratio(n, d, population="never-pay orders", window="test")

    withheld = Metric.not_evaluated(unit="rate", population="never-pay orders", window="test",
                                    reason="below minimum support", numerator=1, denominator=5)
    pooled = pipeline.pool_over_seeds({1: rate(42, 50), 2: withheld}, "x")
    assert not pooled.evaluated
    assert (pooled.numerator, pooled.denominator) == (43, 55)
    assert "seeds [2]" in pooled.note


def test_a_metric_missing_on_a_seed_is_an_error_not_a_smaller_sample() -> None:
    score = Metric(value=0.8, unit="score", population="AP", window="calibration")
    with pytest.raises(pipeline.PipelineError, match="expected"):
        pipeline.pool_over_seeds({1: score}, "ap", seeds=(1, 2))


# ---------------------------------------------------------------- MySQL
def _alerts(order_ids: list[int], users: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "alert_id": [f"{order}-v1" for order in order_ids],
        "order_id": order_ids,
        "user_id": users,
        "ts": ["2025-01-02 10:00:00"] * len(order_ids),
        "score": [55.0] * len(order_ids),
        "band": ["review"] * len(order_ids),
        "fired_rules": [["R01", "R06(b)"]] * len(order_ids),
        "policy": ["incumbent_rules"] * len(order_ids),
        "policy_version": ["v1"] * len(order_ids),
    })


@pytest.mark.mysql
@pytest.mark.reloads_mysql
def test_load_and_alerts_on_the_mini_world(tmp_path: Path) -> None:
    import pymysql

    from core.config import db_settings

    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="load"), log=lambda _: None)
    load = read_result(run.results_dir / "load.json")
    loaded = {row["table"]: row["rows"] for row in load.tables["load.tables"]}
    assert loaded["order_attempts"] == 26 and loaded["accounts"] == 19

    orders = pd.read_csv(MINI_WORLD / "order_attempts.csv")
    first = orders.iloc[:2]
    assert pipeline.write_alerts(_alerts(list(first["order_id"]), list(first["user_id"]))) == 2
    connection = pymysql.connect(**db_settings().pymysql_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT alert_id, band, JSON_EXTRACT(fired_rules, '$[1]') FROM alerts "
                           "ORDER BY alert_id")
            rows = cursor.fetchall()
    finally:
        connection.close()
    assert rows[0][1] == "review" and json.loads(rows[0][2]) == "R06(b)"
    with pytest.raises(pymysql.err.IntegrityError):  # an order the world does not have
        pipeline.write_alerts(_alerts([999_999], [int(first["user_id"].iloc[0])]))
    other_user = int(orders.loc[orders["user_id"] != first["user_id"].iloc[0], "user_id"].iloc[0])
    with pytest.raises(pipeline.PipelineError, match="another account"):
        pipeline.write_alerts(_alerts([int(first["order_id"].iloc[0])], [other_user]))
    connection = pymysql.connect(**db_settings().pymysql_kwargs())
    try:
        with connection.cursor() as cursor:  # both failures left the earlier alerts in place
            cursor.execute("SELECT alert_id FROM alerts ORDER BY alert_id")
            kept = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()
    assert kept == sorted(f"{order}-v1" for order in first["order_id"])
