"""The pipeline: stage order, run directories, lineage, the freeze gate and evaluation."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

import pipeline
from core import protocol as proto
from core import world as world_module
from core.results import Metric, VersionMismatch, read_result, read_summary

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
    assert set(dev.families) == set(protocol.family_starts)
    assert dev.database_world == pipeline.WorldRef(protocol.canonical_seed, "baseline")
    assert dev.results_dir == tmp_path / "dev" / "results"
    assert dev.docs_dir == tmp_path / "dev" / "docs"

    final = pipeline.make_run(pipeline.PROFILES["final"], runs=tmp_path)
    assert final.seeds == protocol.final_seeds
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
    manifest_sha = validate.inputs["worlds/0-baseline/manifest.json"]
    assert len(manifest_sha) == 64
    lineage = json.loads((run.directory / "lineage.json").read_text())
    assert set(lineage["stages"]) == {"world", "validate"}
    outputs = lineage["stages"]["world"]["outputs"]
    assert "worlds/0-baseline/order_attempts.csv" in outputs
    validate_inputs = lineage["stages"]["validate"]["inputs"]
    assert validate_inputs["worlds/0-baseline/manifest.json"] == manifest_sha
    assert "commit" in lineage["stages"]["world"]["code"]
    assert not (run.results_dir / "summary.json").exists()  # a partial run has no summary


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
    with pytest.raises(ValueError, match="does not match its manifest"):
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
def _fake_stages(monkeypatch, *, value: int = 7) -> None:
    def make(name: str):
        def function(run: pipeline.Run) -> pipeline.StageOutput:
            if name == "world":
                for ref in run.all_worlds:
                    target = run.world_dir(ref)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "manifest.json").write_text(json.dumps(
                        {"seed": ref.seed, "family": ref.family, "code_commit": "abc",
                         "tables": {"x": value}}))
            metric = Metric(value=value, unit="count", population=f"{name} things",
                            window="test")
            return pipeline.StageOutput(metrics={f"{name}.things": metric},
                                        inputs={"config/policy.yaml": REPO / "config/policy.yaml"})
        return function

    stages = tuple(pipeline.Stage(stage.name, make(stage.name), stage.versions, stage.description)
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
    assert set(summary["versions"]) == {"world", "features", "policy", "protocol", "benchmark"}
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


def test_the_replay_keeps_each_worlds_review_decisions(tmp_path: Path, monkeypatch) -> None:
    module = types.ModuleType("queue_sim.replay")
    rows = [outcome(0, "approve_all", 0)]
    module.replay = lambda run: pipeline.StageOutput(tables={"replay.outcomes": rows})
    module.review_decisions = lambda run, ref: pd.DataFrame({
        "order_id": [1, 2], "decision_at": pd.to_datetime(["2025-01-01 10:00", "2025-01-02 09:30"]),
        "disposition": ["hold", "clear"],
    })
    monkeypatch.setitem(sys.modules, "queue_sim.replay", module)
    run = pipeline.make_run(pipeline.PROFILES["dev"], runs=tmp_path, world=MINI_WORLD)
    pipeline.execute(run, pipeline.select(until="world"), log=lambda _: None)
    pipeline.execute(run, pipeline.select("replay", "replay"), log=lambda _: None)
    kept = pipeline.review_decisions(run, run.worlds[0])
    assert list(kept["disposition"]) == ["hold", "clear"]
    assert str(kept["decision_at"].dtype).startswith("datetime64")
    lineage = json.loads((run.directory / "lineage.json").read_text())
    assert "worlds/0-baseline/review_decisions.pkl" in lineage["stages"]["replay"]["outputs"]


# ---------------------------------------------------------------- evaluate
def outcome(seed: int, policy: str, net: int, *, family: str = "baseline",
            capacity: str = "base", held: int = 10, legit: int = 10_000) -> dict:
    return {
        "seed": seed, "family": family, "capacity": capacity, "policy": policy,
        "net_cents": net, "loss_cents": 5_000, "gmv_cents": 1_000_000,
        "legit_held": held, "legit_declined": 0, "legit_orders": legit,
        "review_minutes_used": 300, "review_minutes_available": 400,
        "decided_after_shipping": 2,
    }


def outcomes() -> list[dict]:
    rows = []
    for seed, (approve, incumbent, hybrid) in {
        1: (0, 1_000, 1_500), 2: (0, 2_000, 1_900), 3: (0, 1_000, 1_400),
    }.items():
        rows += [outcome(seed, "approve_all", approve, held=0),
                 outcome(seed, "incumbent_rules", incumbent, held=20),
                 outcome(seed, "hybrid", hybrid, held=12 + seed)]
    return rows


def test_evaluate_pairs_policies_by_seed() -> None:
    metrics, table = pipeline.evaluate_outcomes(outcomes(), (1, 2, 3))
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
    assert row["net_contribution_vs_incumbent_rules_positive_seeds"] == 2


def test_evaluate_refuses_incomplete_or_repeated_outcomes() -> None:
    rows = outcomes()
    with pytest.raises(pipeline.PipelineError, match="covers seeds"):
        pipeline.evaluate_outcomes(rows[:-1], (1, 2, 3))
    with pytest.raises(pipeline.PipelineError, match="repeated"):
        pipeline.evaluate_outcomes(rows + rows[:1], (1, 2, 3))
    with pytest.raises(pipeline.PipelineError, match="lack columns"):
        pipeline.evaluate_outcomes([{k: v for k, v in r.items() if k != "gmv_cents"}
                                    for r in rows], (1, 2, 3))
    with pytest.raises(pipeline.PipelineError, match="zero legit_orders"):
        pipeline.evaluate_outcomes([r | {"legit_orders": 0} for r in rows], (1, 2, 3))


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
    gap = Metric.not_evaluated(unit="rate", population="fit-window orders", window="fit",
                               reason="none", numerator=0, denominator=0)
    assert not pipeline.pool_over_seeds({1: rate(1, 4), 2: gap}, "x").evaluated
    with pytest.raises(pipeline.PipelineError, match="disagree"):
        pipeline.pool_over_seeds({1: rate(1, 4), 2: score}, "x")


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
