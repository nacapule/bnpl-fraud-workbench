"""Metric objects, stage result files and the summary."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.results import (
    Interval,
    Metric,
    SeedSpread,
    SignCount,
    StageResult,
    VersionMismatch,
    assemble_summary,
    canonical_json,
    file_sha256,
    metric,
    read_result,
    read_summary,
    table,
    write_result,
    write_summary,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def approval_rate(**overrides) -> Metric:
    fields = {
        "value": 0.75,
        "unit": "rate",
        "population": "orders scored, test window",
        "window": "test",
        "numerator": 3,
        "denominator": 4,
    }
    fields.update(overrides)
    return Metric(**fields)


def full_metric() -> Metric:
    return Metric(
        value=12_345,
        unit="cents",
        population="legitimate approved orders, test window",
        window="test",
        interval=Interval(-200, 30_000, "cluster_bootstrap", 0.9),
        seeds=SeedSpread({11: 2.5, 3: -1.0, 7: 0.0}),
        note="paired difference against approve-all",
    )


def stage(name: str = "replay", **overrides) -> StageResult:
    fields = {
        "stage": name,
        "versions": {"world": "w1", "policy": "fp2"},
        "inputs": {"data/orders.parquet": SHA_A},
        "metrics": {f"{name}.approval_rate": approval_rate()},
    }
    fields.update(overrides)
    return StageResult(**fields)


# Round trips


def test_metric_round_trip_keeps_every_field_and_type() -> None:
    original = full_metric()
    restored = Metric.from_dict(json.loads(canonical_json(original.to_dict())))
    assert restored == original
    assert type(restored.value) is int
    assert type(restored.interval.low) is int
    assert type(restored.interval.level) is float
    assert list(restored.seeds.per_seed) == [3, 7, 11]


def test_interval_and_seed_spread_round_trip() -> None:
    interval = Interval(0.1, 0.2, "wilson")
    assert Interval.from_dict(json.loads(json.dumps(interval.to_dict()))) == interval
    spread = SeedSpread({10: 1, 2: -0.5})
    data = json.loads(json.dumps(spread.to_dict()))
    assert data == {"per_seed": {"10": 1, "2": -0.5}}
    assert SeedSpread.from_dict(data) == spread


def test_stage_result_round_trips_through_its_file(tmp_path) -> None:
    original = stage(
        metrics={
            "replay.approval_rate": approval_rate(),
            "replay.net_contribution_vs_approve_all.baseline": full_metric(),
            "replay.unmatched_disputes": Metric.not_evaluated(
                unit="rate",
                population="disputed orders, test window",
                window="test",
                numerator=0,
                denominator=0,
                reason="no disputes in the window",
            ),
        },
        tables={
            "replay.by_policy": [
                {"policy": "baseline", "orders": 4, "rate": 0.75, "capped": False, "comment": None}
            ]
        },
        notes=["Amounts are in cents."],
    )
    path = write_result(original, tmp_path / "results")
    assert path == tmp_path / "results" / "replay.json"
    restored = read_result(path)
    assert restored == original
    assert restored.to_dict() == original.to_dict()


# Rates, shares and quotients


def test_rate_value_must_equal_numerator_over_denominator() -> None:
    with pytest.raises(ValueError, match="does not equal"):
        approval_rate(value=0.7)
    assert approval_rate(value=1 / 3, numerator=1, denominator=3).value == pytest.approx(1 / 3)
    approval_rate(value=0.75 + 5e-13)
    with pytest.raises(ValueError, match="does not equal"):
        approval_rate(value=0.75 + 5e-12)


def test_rate_and_share_need_numerator_and_denominator() -> None:
    with pytest.raises(ValueError, match="needs its numerator and denominator"):
        approval_rate(numerator=None, denominator=None)
    with pytest.raises(ValueError, match="needs its numerator and denominator"):
        Metric(value=0.5, unit="share", population="loss cents", window="test")
    with pytest.raises(ValueError, match="together"):
        approval_rate(denominator=None)


def test_any_metric_with_a_quotient_must_match_it() -> None:
    Metric(
        value=12.5,
        unit="minutes",
        population="reviewed cases",
        window="test",
        numerator=50,
        denominator=4,
    )
    with pytest.raises(ValueError, match="does not equal"):
        Metric(
            value=12,
            unit="minutes",
            population="reviewed cases",
            window="test",
            numerator=50,
            denominator=4,
        )
    assert Metric.from_ratio(3, 400, unit="bps", population="orders", window="test").value == 75
    with pytest.raises(ValueError, match="does not equal"):
        Metric(
            value=0.0075,
            unit="bps",
            population="orders",
            window="test",
            numerator=3,
            denominator=400,
        )


def test_from_ratio_computes_the_value() -> None:
    built = Metric.from_ratio(3, 4, population="orders scored, test window", window="test")
    assert built == approval_rate()
    assert Metric.from_ratio(1, 2, unit="share", population="loss", window="all").value == 0.5


def test_zero_denominator_is_not_a_metric() -> None:
    with pytest.raises(ValueError, match="not_evaluated"):
        approval_rate(value=0.0, numerator=0, denominator=0)
    with pytest.raises(ValueError, match="not_evaluated"):
        Metric.from_ratio(0, 0, population="orders", window="test")
    with pytest.raises(ValueError, match="negative"):
        approval_rate(value=0.75, numerator=-3, denominator=-4)


def test_not_evaluated_shows_n_and_the_reason() -> None:
    empty = Metric.not_evaluated(
        unit="rate",
        population="never-pay orders, test window",
        window="test",
        numerator=0,
        denominator=0,
        reason="no never-pay orders in the test window",
    )
    assert empty.value is None
    assert not empty.evaluated
    assert (empty.numerator, empty.denominator) == (0, 0)
    assert empty.note == "no never-pay orders in the test window"
    assert approval_rate().evaluated
    withheld = Metric.not_evaluated(
        unit="rate",
        population="orders",
        window="test",
        numerator=2,
        denominator=7,
        reason="N=7 is below 30",
    )
    assert withheld.denominator == 7


def test_not_evaluated_needs_a_reason_and_no_spread() -> None:
    with pytest.raises(ValueError, match="needs a note"):
        Metric(value=None, unit="count", population="orders", window="test")
    with pytest.raises(ValueError, match="needs its numerator"):
        Metric.not_evaluated(unit="rate", population="orders", window="test", reason="none")
    with pytest.raises(ValueError, match="numerator must be 0"):
        Metric.not_evaluated(
            unit="rate",
            population="orders",
            window="test",
            numerator=1,
            denominator=0,
            reason="none",
        )
    with pytest.raises(ValueError, match="interval or seeds"):
        Metric(
            value=None,
            unit="count",
            population="orders",
            window="test",
            note="none",
            interval=Interval(0, 1, "exact"),
        )


def test_metric_field_checks() -> None:
    with pytest.raises(ValueError, match="unit"):
        approval_rate(unit="percent")
    with pytest.raises(ValueError, match="population"):
        approval_rate(population=" ")
    with pytest.raises(ValueError, match="window"):
        approval_rate(window="Test window")
    with pytest.raises(ValueError, match="must be a number"):
        approval_rate(value=True)
    with pytest.raises(ValueError, match="must be a number"):
        approval_rate(value="0.75")


def test_numpy_scalars_become_python_numbers() -> None:
    built = approval_rate(value=np.float64(0.75), numerator=np.int64(3), denominator=np.int64(4))
    assert (type(built.value), type(built.numerator), type(built.denominator)) == (float, int, int)
    canonical_json(built.to_dict())


# Intervals and seed spreads


def test_interval_ordering_and_fields() -> None:
    assert Interval(0.2, 0.2, "exact").level == 0.95
    with pytest.raises(ValueError, match="above high"):
        Interval(0.3, 0.2, "wilson")
    with pytest.raises(ValueError, match="level"):
        Interval(0.1, 0.2, "wilson", 1.0)
    with pytest.raises(ValueError, match="method"):
        Interval(0.1, 0.2, "Wilson score")


def test_seed_spread_arithmetic() -> None:
    spread = SeedSpread({3: 0.0, 1: 2.0, 4: 5.0, 2: -1.0})
    assert spread.n == 4
    assert spread.mean == 1.5
    assert spread.min == -1.0
    assert spread.max == 5.0
    assert spread.sign_count == SignCount(positive=2, negative=1, zero=1)
    assert list(spread.per_seed) == [1, 2, 3, 4]


def test_seed_spread_counts_positive_seeds() -> None:
    per_seed = {seed: 1.0 for seed in range(1, 10)}
    per_seed[10] = -1.0
    spread = SeedSpread(per_seed)
    assert spread.sign_count == (9, 1, 0)
    assert spread.mean == 0.8


def test_seed_spread_checks() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SeedSpread({})
    with pytest.raises(ValueError, match="non-negative int"):
        SeedSpread({True: 1.0})
    with pytest.raises(ValueError, match="non-negative int"):
        SeedSpread({"416": 1.0})
    with pytest.raises(ValueError, match="decimal integer"):
        SeedSpread.from_dict({"per_seed": {"0416": 1.0}})


# No NaN or infinity anywhere


@pytest.mark.parametrize(
    "build",
    [
        lambda: approval_rate(value=math.nan),
        lambda: Metric(value=math.inf, unit="score", population="orders", window="test"),
        lambda: Interval(math.nan, 1.0, "wilson"),
        lambda: SeedSpread({1: float("-inf")}),
        lambda: stage(tables={"replay.t": [{"x": math.nan}]}),
    ],
)
def test_nan_and_infinity_are_rejected(build) -> None:
    with pytest.raises(ValueError, match="finite"):
        build()


def test_nan_in_a_file_is_rejected(tmp_path) -> None:
    path = write_result(stage(), tmp_path)
    path.write_text(path.read_text().replace('"value": 0.75', '"value": NaN'))
    with pytest.raises(ValueError, match="NaN"):
        read_result(path)
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


# Determinism


def test_file_format_is_fixed(tmp_path) -> None:
    result = StageResult(
        stage="world",
        versions={"world": "w1"},
        inputs={},
        metrics={"world.orders": Metric(value=40, unit="count", population="orders", window="all")},
    )
    expected = """{
  "inputs": {},
  "metrics": {
    "world.orders": {
      "denominator": null,
      "interval": null,
      "note": null,
      "numerator": null,
      "population": "orders",
      "seeds": null,
      "unit": "count",
      "value": 40,
      "window": "all"
    }
  },
  "notes": [],
  "schema_version": 1,
  "stage": "world",
  "tables": {},
  "versions": {
    "world": "w1"
  }
}
"""
    assert write_result(result, tmp_path).read_bytes() == expected.encode()


def test_output_does_not_depend_on_insertion_order_or_time(tmp_path) -> None:
    first = stage(
        versions={"world": "w1", "policy": "fp2"},
        inputs={"b": SHA_B, "a": SHA_A},
        metrics={"replay.z": approval_rate(), "replay.a": full_metric()},
    )
    second = stage(
        versions={"policy": "fp2", "world": "w1"},
        inputs={"a": SHA_A, "b": SHA_B},
        metrics={"replay.a": full_metric(), "replay.z": approval_rate()},
    )
    one = write_result(first, tmp_path / "one").read_bytes()
    again = write_result(first, tmp_path / "one").read_bytes()
    two = write_result(second, tmp_path / "two").read_bytes()
    assert one == again == two
    assert one.endswith(b"}\n")
    summaries = [
        write_summary(order, tmp_path / name).read_bytes()
        for name, order in (
            ("s1", [stage("replay"), stage("detection")]),
            ("s2", [stage("detection"), stage("replay")]),
        )
    ]
    assert summaries[0] == summaries[1]


# Result files


def test_read_result_checks_the_file(tmp_path) -> None:
    path = write_result(stage(), tmp_path)
    moved = tmp_path / "detection.json"
    moved.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="expected replay.json"):
        read_result(moved)
    data = json.loads(path.read_text())
    data["generated_at"] = "2026-10-06T00:00:00"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown fields"):
        read_result(path)
    del data["generated_at"]
    data["schema_version"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema version"):
        read_result(path)


def test_stage_result_field_checks() -> None:
    with pytest.raises(ValueError, match="sha256"):
        stage(inputs={"orders": "not-a-digest"})
    with pytest.raises(ValueError, match="reserved"):
        stage("summary")
    with pytest.raises(ValueError, match="different columns"):
        stage(tables={"replay.t": [{"a": 1}, {"b": 2}]})
    with pytest.raises(ValueError, match="column"):
        stage(tables={"replay.t": [{"Net $": 1}]})
    with pytest.raises(ValueError, match="must be a Metric"):
        stage(metrics={"replay.x": 0.5})


def test_file_sha256_matches_a_known_digest(tmp_path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize(
    "key",
    ["replay.net_contribution_vs_approve_all.baseline", "world", "llm.arm_2.agreement_50"],
)
def test_key_syntax_accepts_dotted_lower_case(key) -> None:
    assert key in stage(metrics={key: approval_rate()}).metrics


@pytest.mark.parametrize(
    "key", ["Replay.rate", "replay..rate", ".replay", "replay.", "replay-rate", "replay rate", ""]
)
def test_key_syntax_rejects_other_names(key) -> None:
    with pytest.raises(ValueError, match="metric key"):
        stage(metrics={key: approval_rate()})
    with pytest.raises(ValueError, match="table name"):
        stage(tables={key: []})


# Summary


def test_summary_carries_versions_inputs_metrics_and_tables(tmp_path) -> None:
    world = StageResult(
        stage="world",
        versions={"world": "w1"},
        inputs={"config/world.yaml": SHA_A},
        metrics={"world.orders": Metric(value=40, unit="count", population="orders", window="all")},
    )
    replay = stage(
        inputs={"config/world.yaml": SHA_A, "config/policy.yaml": SHA_B},
        tables={"replay.by_policy": [{"policy": "baseline", "orders": 4}]},
        notes=["Amounts are in cents."],
    )
    summary = assemble_summary([world, replay])
    assert summary["versions"] == {"policy": "fp2", "world": "w1"}
    assert summary["stages"]["world"]["inputs"] == {"config/world.yaml": SHA_A}
    assert summary["stages"]["replay"]["metrics"] == ["replay.approval_rate"]
    assert summary["stages"]["replay"]["tables"] == ["replay.by_policy"]
    assert metric(summary, "replay.approval_rate") == approval_rate()
    assert metric(summary, "world.orders").value == 40
    assert table(summary, "replay.by_policy") == [{"policy": "baseline", "orders": 4}]

    path = write_summary([world, replay], tmp_path)
    assert path == tmp_path / "summary.json"
    assert read_summary(path) == summary


def test_summary_refuses_a_version_disagreement() -> None:
    replay = stage("replay", versions={"world": "w1", "policy": "fp2"})
    detection = stage("detection", versions={"world": "w2", "features": "f1"})
    context = stage("context", versions={"world": "w1"})
    with pytest.raises(VersionMismatch, match="'world'") as raised:
        assemble_summary([replay, detection, context])
    assert isinstance(raised.value, ValueError)
    assert raised.value.kind == "version"
    assert raised.value.key == "world"
    assert raised.value.stages_by_value == {"w1": ("context", "replay"), "w2": ("detection",)}
    assert "detection" in str(raised.value) and "context, replay" in str(raised.value)


def test_summary_refuses_different_hashes_of_one_input() -> None:
    replay = stage("replay", inputs={"data/orders.parquet": SHA_A})
    detection = stage("detection", inputs={"data/orders.parquet": SHA_B})
    with pytest.raises(VersionMismatch, match="input 'data/orders.parquet'") as raised:
        assemble_summary([replay, detection])
    assert raised.value.kind == "input"


def test_summary_refuses_duplicates() -> None:
    with pytest.raises(ValueError, match="'replay' appears more than once"):
        assemble_summary([stage("replay"), stage("replay")])
    shared = {"shared.rate": approval_rate()}
    with pytest.raises(ValueError, match="'shared.rate' is reported by both 'replay' and 'llm'"):
        assemble_summary([stage("replay", metrics=shared), stage("llm", metrics=shared)])
    rows = {"shared.table": []}
    with pytest.raises(ValueError, match="table name 'shared.table'"):
        assemble_summary([stage("replay", tables=rows), stage("llm", tables=rows)])
    with pytest.raises(ValueError, match="no stage results"):
        assemble_summary([])


def test_lookup_raises_key_error_naming_the_key() -> None:
    summary = assemble_summary([stage()])
    with pytest.raises(KeyError, match=r"replay\.net_contribution\.missing"):
        metric(summary, "replay.net_contribution.missing")
    with pytest.raises(KeyError, match=r"replay\.no_table"):
        table(summary, "replay.no_table")
