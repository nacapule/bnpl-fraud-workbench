"""Detection models: labels used only once known, the calibration month kept out of
fitting, row-by-row scoring, metadata instead of binaries, supporting metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import asof
from core.protocol import load_protocol
from model.evaluate import detection_metrics, shortcut_sensitivity
from model.features import FEATURES, TREE_FEATURES, labels_known_before
from model.train import RuleScorer, fit

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol()


def synthetic(seed: int = 7, n: int = 6_000) -> dict[str, pd.DataFrame]:
    """Context rows, attempts and dated labels over the protocol's windows; fraud depends on
    a few context columns so the models have something to learn."""
    rng = np.random.default_rng(seed)
    start, end = PROTOCOL.order_start, PROTOCOL.order_end
    seconds = np.sort(rng.integers(start.value // 10**9, end.value // 10**9, n))
    at = pd.to_datetime(seconds, unit="s").astype("datetime64[s]")
    context = pd.DataFrame({"order_id": np.arange(1, n + 1), "user_id": np.arange(1, n + 1),
                            "merchant_id": 1, "decision_at": at})
    for column in asof.COLUMNS:
        if column.dtype == "int8":
            values = rng.random(n) < 0.1
        elif column.dtype == "int64":
            values = rng.poisson(1.0, n)
        else:
            values = rng.gamma(2.0, 20.0, n)
        context[column.name] = pd.Series(values).astype(column.dtype)
    risk = (-5 + 3 * (context["account_age_days"] < 10) + 2 * context["bin_ip_country_mismatch"]
            + 1.5 * (context["accounts_on_device_30d"] >= 3))
    fraud = rng.random(n) < 1 / (1 + np.exp(-risk.to_numpy()))
    known = at + pd.to_timedelta(rng.integers(20, 50, n), unit="D")
    labels = pd.DataFrame({"order_id": context["order_id"], "label": fraud.astype(np.int64),
                           "basis": np.where(fraud, "third_party_fraud", "no_finding"),
                           "label_known_at": known})
    attempts = pd.DataFrame({"order_id": context["order_id"], "known_at": at,
                             "processor_result": np.where(rng.random(n) < 0.97, "approved",
                                                          "declined")})
    return {"context": context, "labels": labels, "order_attempts": attempts}


@pytest.fixture(scope="module")
def world() -> dict[str, pd.DataFrame]:
    return synthetic()


@pytest.fixture(scope="module")
def base(world, tmp_path_factory):
    directory = tmp_path_factory.mktemp("base")
    return _fit(world, directory), directory


def _fit(world: dict[str, pd.DataFrame], tmp_path: Path, labels: pd.DataFrame | None = None):
    tables = {"order_attempts": world["order_attempts"],
              "labels": world["labels"] if labels is None else labels}
    return fit(tables, world["context"], PROTOCOL, tmp_path)


def _probe(world: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return world["context"].sample(300, random_state=1)


def test_labels_known_at_the_freeze_are_not_yet_known() -> None:
    labels = pd.DataFrame({"order_id": [1, 2, 2, 3], "label": [1, 0, 1, 0],
                           "label_known_at": pd.to_datetime(
                               ["2024-09-30 23:59:59", "2024-09-01 00:00:00", "2024-10-01 00:00:00",
                                "2024-10-02 00:00:00"])})
    known = labels_known_before(labels, pd.Timestamp("2024-10-01"))
    assert known.to_dict() == {1: 1, 2: 0}  # order 2's positive arrives at the freeze itself


def test_fit_ignores_labels_known_after_the_classifier_freeze(world, base, tmp_path) -> None:
    """Training rows carry only what was known at the freeze: flipping fit-window labels
    that became known later changes nothing, while flipping earlier-known ones does."""
    base = base[0]
    labels = world["labels"].copy()
    fit_window = PROTOCOL.windows["fit"]
    in_fit = (labels["order_id"].isin(world["context"].loc[
        (world["context"]["decision_at"] >= fit_window.start)
        & (world["context"]["decision_at"] < fit_window.end), "order_id"]))
    late = labels[in_fit].assign(label=1 - labels.loc[in_fit, "label"],
                                 label_known_at=PROTOCOL.freezes["classifier"])
    later = _fit(world, tmp_path / "later", pd.concat([labels, late], ignore_index=True))
    probe = _probe(world)
    for name in ("tree", "logistic", "boosting"):
        assert np.array_equal(base.scorers[name].score(probe), later.scorers[name].score(probe))
        assert base.scorers[name].version == later.scorers[name].version
    early = late.assign(label_known_at=PROTOCOL.freezes["classifier"] - pd.Timedelta(seconds=1))
    sooner = _fit(world, tmp_path / "sooner", pd.concat([labels, early], ignore_index=True))
    assert not np.array_equal(base.scorers["boosting"].score(probe),
                              sooner.scorers["boosting"].score(probe))


def test_calibration_month_is_not_used_to_fit_the_classifiers(world, base, tmp_path) -> None:
    base, directory = base
    labels = world["labels"].copy()
    span = PROTOCOL.windows["calibration"]
    month = world["context"].loc[(world["context"]["decision_at"] >= span.start)
                                 & (world["context"]["decision_at"] < span.end), "order_id"]
    flipped = labels["order_id"].isin(month)
    labels.loc[flipped, "label"] = 1 - labels.loc[flipped, "label"]
    changed = _fit(world, tmp_path / "changed", labels)
    probe = _probe(world)
    for name in ("tree", "logistic", "boosting"):
        assert np.array_equal(base.scorers[name].score(probe), changed.scorers[name].score(probe))
        assert not np.array_equal(base.scorers[name].probability(probe),
                                  changed.scorers[name].probability(probe))
    meta = json.loads((directory / "model_boosting.json").read_text())
    assert meta["fit"]["window"] == ["2024-04-01", "2024-08-01"]
    assert meta["calibration"]["window"] == ["2024-10-01", "2024-11-01"]


def test_scores_are_row_by_row_and_reproducible(world, base, tmp_path) -> None:
    first, directory = base
    second = _fit(world, tmp_path)
    probe = _probe(world)
    for name, scorer in first.scorers.items():
        batch = scorer.score(probe)
        single = np.concatenate([scorer.score(probe.iloc[[i]]) for i in range(60)])
        assert np.array_equal(batch[:60], single), name
        assert np.array_equal(batch, second.scorers[name].score(probe)), name
        assert scorer.version == second.scorers[name].version
        assert ((scorer.probability(probe) >= 0) & (scorer.probability(probe) <= 1)).all()
    for name in ("rules", "tree", "logistic", "boosting"):
        assert (directory / f"model_{name}.json").read_bytes() == (
            tmp_path / f"model_{name}.json").read_bytes()


def test_scorers_read_only_their_columns(world, base) -> None:
    result = base[0]
    assert set(result.scorers) == {"rules", "tree", "logistic", "boosting"}
    assert result.scorers["tree"].columns == TREE_FEATURES and len(TREE_FEATURES) <= 4
    assert set(result.scorers["boosting"].columns) == set(FEATURES)
    assert not set(FEATURES) & {"account_blocked", "shipped_at_decision"}
    probe = _probe(world)
    with pytest.raises(KeyError, match="account_age_days"):
        result.scorers["tree"].score(probe.drop(columns="account_age_days"))
    unrelated = probe.assign(merchant_age_days=-1.0)  # not a column the tree reads
    assert np.array_equal(result.scorers["tree"].score(probe),
                          result.scorers["tree"].score(unrelated))
    rules = result.scorers["rules"]
    assert isinstance(rules, RuleScorer)
    assert list(rules.fired(probe).columns[:11]) == [f"R{i:02d}" for i in range(1, 12)]


def test_metrics_are_metric_objects_with_counts(base) -> None:
    result = base[0]
    metrics = result.metrics
    for name in ("rules", "tree", "logistic", "boosting"):
        assert 0 <= metrics[f"detection.{name}.average_precision.validation"].value <= 1
        assert metrics[f"detection.{name}.brier.validation"].value >= 0
    assert metrics["detection.validation.orders"].value > metrics[
        "detection.validation.positives"].value > 0


def test_a_validation_window_without_known_labels_is_reported_not_evaluated(world,
                                                                           tmp_path) -> None:
    span = PROTOCOL.windows["validation"]
    context = world["context"]
    validation = context.loc[(context["decision_at"] >= span.start)
                             & (context["decision_at"] < span.end), "order_id"]
    labels = world["labels"][~world["labels"]["order_id"].isin(validation)]
    result = _fit(world, tmp_path, labels)
    assert result.metrics["detection.validation.orders"].value == 0
    for name in ("rules", "tree", "logistic", "boosting"):
        for kind in ("average_precision", "brier"):
            assert result.metrics[f"detection.{name}.{kind}.validation"].value is None
    tables = {"order_attempts": world["order_attempts"], "labels": labels}
    sensitivity = shortcut_sensitivity(tables, context, PROTOCOL, result.scorers["boosting"])
    assert all(metric.value is None for metric in sensitivity.values())


def test_average_precision_and_brier_on_a_hand_case() -> None:
    class Fixed:
        columns = ()

        def score(self, rows):
            return rows["s"].to_numpy()

        def probability(self, rows):
            return rows["p"].to_numpy()

    rows = pd.DataFrame({"s": [0.9, 0.8, 0.3, 0.1], "p": [1.0, 0.5, 0.5, 0.0],
                         "label": [1, 0, 1, 0]})
    metrics = detection_metrics({"m": Fixed()}, rows, "validation")
    # precision 1 at the first positive, 2/3 at the second: AP = (1 + 2/3) / 2
    assert metrics["detection.m.average_precision.validation"].value == pytest.approx(5 / 6)
    assert metrics["detection.m.brier.validation"].value == pytest.approx((0 + .25 + .25 + 0) / 4)
    empty = detection_metrics({"m": Fixed()}, rows.assign(label=0), "validation")
    assert empty["detection.m.average_precision.validation"].value is None
    # what is counted is the same in every world, so seeds pool; the counts are metrics
    other = detection_metrics({"m": Fixed()}, rows.iloc[:3], "validation")
    for key, item in metrics.items():
        assert (item.unit, item.population, item.window) == \
            (other[key].unit, other[key].population, other[key].window)
    assert (metrics["detection.validation.orders"].value,
            other["detection.validation.orders"].value) == (4, 3)


def test_shortcut_sensitivity_reports_each_check(world, base) -> None:
    result = base[0]
    tables = {"order_attempts": world["order_attempts"], "labels": world["labels"]}
    metrics = shortcut_sensitivity(tables, world["context"], PROTOCOL, result.scorers["boosting"])
    for name in ("full", "age_tree", "construction_tree", "three_feature_boosting"):
        assert metrics[f"detection.sensitivity.{name}.average_precision.validation"].evaluated
    before = metrics["detection.sensitivity.first_orders.as_observed.median_score"].value
    after = metrics["detection.sensitivity.first_orders.new_account_profile.median_score"].value
    assert after > before  # in this world, young accounts are riskier by construction
    assert not any(char.isdigit() for item in metrics.values()
                   for char in item.population.split("; profile")[0])  # no per-world counts


def test_no_model_binaries_are_committed() -> None:
    binaries = [p for p in (REPO / "model").rglob("*") if p.suffix in {".joblib", ".pkl"}]
    assert binaries == []
