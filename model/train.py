"""Fit the detection scorers on one world's pre-test history.

:func:`fit` is the pipeline's fit stage. On processor-approved orders at checkout it
fits three classifiers on the protocol's fit window, using only labels known before
the classifier freeze (an order whose outcome is not yet determined is left out),
then fits one isotonic calibrator per scorer on the calibration month, which no
classifier is fitted on, with labels known before the calibrator freeze. The rule
score (``rules.engine``) is calibrated the same way. Every scorer reads only as-of
context columns.

Class weighting: every classifier uses ``class_weight="balanced"``, so its raw score
ranks orders but is not a probability; :meth:`Scorer.probability` is the calibrated
probability. No model binaries are written: each scorer's metadata (inputs,
windows, row counts, hyperparameters, fitted parameters, calibration map and a
version hash) goes to ``out_dir`` as JSON, and refitting on the same world
reproduces it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from core.protocol import Protocol
from core.results import Metric
from model.features import FEATURES, TREE_FEATURES, checkout_rows, labelled
from rules import definitions, engine

RANDOM_STATE = 0
CLASS_WEIGHT = "balanced"
HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "tree": {"max_depth": 3},
    "logistic": {"impute": "median with missing-value indicators",
                 "transform": "sign(x) * log1p(|x|), then standardized", "C": 1.0,
                 "max_iter": 2000},
    "boosting": {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31,
                 "early_stopping": False},
}


def _signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def make_models() -> dict[str, tuple[Any, tuple[str, ...]]]:
    """The three classifiers and the context columns each reads."""
    logistic = HYPERPARAMETERS["logistic"]
    boosting = HYPERPARAMETERS["boosting"]
    return {
        "tree": (DecisionTreeClassifier(max_depth=HYPERPARAMETERS["tree"]["max_depth"],
                                        class_weight=CLASS_WEIGHT,
                                        random_state=RANDOM_STATE), TREE_FEATURES),
        "logistic": (Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(_signed_log)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=logistic["C"], class_weight=CLASS_WEIGHT,
                                         max_iter=logistic["max_iter"])),
        ]), FEATURES),
        "boosting": (HistGradientBoostingClassifier(
            max_iter=boosting["max_iter"], learning_rate=boosting["learning_rate"],
            max_leaf_nodes=boosting["max_leaf_nodes"], early_stopping=False,
            class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE), FEATURES),
    }


@dataclass(frozen=True, eq=False)
class Scorer:
    """One detection model as the replay calls it.

    ``score`` is the raw score (higher is riskier) and ``probability`` the calibrated
    probability that the order's adjudicated label is positive. Both are computed
    row by row from ``columns`` alone, deterministically and without I/O, so a
    batch gives the same values as its rows one at a time.
    """

    name: str
    version: str
    columns: tuple[str, ...]
    raw: Callable[[pd.DataFrame], np.ndarray] = field(repr=False)
    calibrator: IsotonicRegression = field(repr=False)

    def _inputs(self, rows: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.columns if column not in rows.columns]
        if missing:
            raise KeyError(f"{self.name} needs context columns {missing}")
        return rows[list(self.columns)]

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        with threadpool_limits(limits=1):  # small batches; one thread is fast and steady
            return np.asarray(self.raw(self._inputs(rows)), dtype=np.float64)

    def probability(self, rows: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.score(rows))


@dataclass(frozen=True, eq=False)
class RuleScorer(Scorer):
    """The rule score, with the rules that fire (for queue priority, FP-2 §7.1)."""

    def fired(self, rows: pd.DataFrame) -> pd.DataFrame:
        return engine.fired(self._inputs(rows))


@dataclass(frozen=True)
class FitResult:
    scorers: dict[str, Scorer]
    metrics: dict[str, Metric]


def _raw_score(name: str, estimator: Any) -> Callable[[pd.DataFrame], np.ndarray]:
    """The fitted classifier's positive-class score for each row. The logistic model's
    linear term is summed row by row, so a row scores the same alone or in any batch
    (a matrix product would change the last bits with the batch)."""
    if name != "logistic":
        return lambda frame: estimator.predict_proba(frame)[:, 1]
    head, model = estimator[:-1], estimator[-1]
    weights, intercept = model.coef_[0], model.intercept_[0]

    def logistic(frame: pd.DataFrame) -> np.ndarray:
        inputs = np.ascontiguousarray(head.transform(frame), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-((inputs * weights).sum(axis=1) + intercept)))

    return logistic


def _calibrate(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, labels)


def _rows_sha256(rows: pd.DataFrame, columns: tuple[str, ...]) -> str:
    table = rows[["order_id", "label", *columns]].sort_values("order_id")
    return hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _floats(values: Any) -> list[Any]:
    return np.asarray(values, dtype=float).round(12).tolist()


def _parameters(name: str, estimator: Any) -> dict[str, Any]:
    if name == "tree":
        tree = estimator.tree_
        return {"feature": tree.feature.tolist(), "threshold": _floats(tree.threshold),
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "positive_share": _floats(tree.value[:, 0, 1] / tree.value[:, 0].sum(axis=1)),
                "samples": tree.n_node_samples.tolist()}
    if name == "logistic":
        impute = estimator.named_steps["impute"]
        scale = estimator.named_steps["scale"]
        model = estimator.named_steps["model"]
        return {"impute_medians": _floats(impute.statistics_),
                "missing_indicators": [] if impute.indicator_ is None
                else impute.indicator_.features_.tolist(),
                "scale_mean": _floats(scale.mean_), "scale_sd": _floats(scale.scale_),
                "coefficients": _floats(model.coef_[0]), "intercept": float(model.intercept_[0])}
    return {"iterations": int(estimator.n_iter_),
            "note": "fitted trees are not stored; refitting on the same world reproduces them"}


def _metadata(name: str, columns: tuple[str, ...], train: pd.DataFrame | None,
              calibration: pd.DataFrame, calibrator: IsotonicRegression,
              parameters: dict[str, Any], protocol: Protocol) -> dict[str, Any]:
    windows = protocol.windows
    meta: dict[str, Any] = {"name": name, "columns": list(columns)}
    if train is not None:
        meta["fit"] = {
            "window": [str(windows["fit"].start.date()), str(windows["fit"].end.date())],
            "labels_known_before": str(protocol.freezes["classifier"].date()),
            "orders": len(train), "positives": int(train["label"].sum()),
            "rows_sha256": _rows_sha256(train, columns), "class_weight": CLASS_WEIGHT,
            "random_state": RANDOM_STATE, "hyperparameters": HYPERPARAMETERS[name],
        }
    meta["parameters"] = parameters
    meta["calibration"] = {
        "method": "isotonic",
        "window": [str(windows["calibration"].start.date()),
                   str(windows["calibration"].end.date())],
        "labels_known_before": str(protocol.freezes["calibrator"].date()),
        "orders": len(calibration), "positives": int(calibration["label"].sum()),
        "thresholds": _floats(calibrator.X_thresholds_),
        "probabilities": _floats(calibrator.y_thresholds_),
    }
    text = json.dumps(meta, sort_keys=True, allow_nan=False)
    meta["version"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    return meta


def fit(tables: Mapping[str, pd.DataFrame], context: pd.DataFrame, protocol: Protocol,
        out_dir: Path) -> FitResult:
    """Fit the rules' calibration and the three classifiers on one world's pre-test
    history (``tables`` with its ``labels``; ``context`` from ``core.asof.build_context``)
    and write each scorer's metadata to ``out_dir/model_<name>.json``. Returns the
    scorers (keys rules, tree, logistic, boosting) and their validation metrics."""
    from model.evaluate import detection_metrics

    labels = tables["labels"]
    rows = checkout_rows(context, tables)
    windows, freezes = protocol.windows, protocol.freezes
    train = labelled(rows, labels, windows["fit"], freezes["classifier"])
    calibration = labelled(rows, labels, windows["calibration"], freezes["calibrator"])
    validation = labelled(rows, labels, windows["validation"], freezes["policy"])
    for name, part in (("fit", train), ("calibration", calibration)):
        if part["label"].nunique() < 2:
            raise ValueError(f"the {name} window needs known positive and negative labels")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fitted: dict[str, tuple[Callable[[pd.DataFrame], np.ndarray], tuple[str, ...],
                            dict[str, Any], pd.DataFrame | None, type[Scorer]]] = {
        "rules": (engine.score, definitions.COLUMNS, {
            "weights": definitions.WEIGHTS,
            "conditions": {c.id: list(c.columns) for c in definitions.CONDITIONS},
            "definitions_sha256": hashlib.sha256(
                Path(definitions.__file__).read_bytes()).hexdigest()}, None, RuleScorer),
    }
    for name, (estimator, columns) in make_models().items():
        with threadpool_limits(limits=1):
            estimator.fit(train[list(columns)], train["label"])
        fitted[name] = (_raw_score(name, estimator), columns, _parameters(name, estimator),
                        train, Scorer)

    scorers: dict[str, Scorer] = {}
    for name, (raw, columns, parameters, fit_rows, kind) in fitted.items():
        calibrator = _calibrate(np.asarray(raw(calibration[list(columns)]), float),
                                calibration["label"].to_numpy())
        meta = _metadata(name, columns, fit_rows, calibration, calibrator, parameters, protocol)
        (out_dir / f"model_{name}.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n")
        scorers[name] = kind(name, meta["version"], tuple(columns), raw, calibrator)
    metrics = detection_metrics(scorers, validation, "validation")
    for name, part in (("fit", train), ("calibration", calibration)):
        population = f"processor-approved orders at checkout, labels known in time ({name})"
        metrics[f"detection.{name}.orders"] = Metric(value=len(part), unit="count",
                                                     population=population, window=name)
        metrics[f"detection.{name}.positives"] = Metric(
            value=int(part["label"].sum()), unit="count", population=population, window=name)
    return FitResult(scorers, metrics)
