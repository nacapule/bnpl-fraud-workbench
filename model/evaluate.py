"""Supporting detection metrics and the construction-shortcut sensitivity.

Average precision (of each raw score) and the Brier score (of each calibrated
probability) are supporting rows; the policy comparison itself is the replay's.
The shortcut sensitivity asks whether a few construction features explain most of
the full model's ranking: shallow models on account age, first-attempt, address
and amount features against the full boosting model, and how the full model's
scores move when legitimate first orders are given a new-account profile. It is a
reported check on the world, not a target for it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from core.protocol import Protocol
from core.results import Metric
from model.features import checkout_rows, labelled
from model.train import CLASS_WEIGHT, RANDOM_STATE, Scorer

SHORTCUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "age_tree": ("tree", ("account_age_days",)),
    "construction_tree": ("tree", ("account_age_days", "is_first_attempt_user",
                                   "ship_address_link_age_hours",
                                   "amount_over_category_median")),
    "three_feature_boosting": ("boosting", ("account_age_days", "hours_since_credential_change",
                                            "accounts_on_address_ever")),
}
# The new-account profile given to legitimate first orders in the perturbation.
PROFILE = {"account_age_days": 5.0, "amount_over_category_median": 1.3}
TOP_SHARE = 0.01  # the cutoff is the validation window's top 1% of full-model scores


def _population(window: str, rows: pd.DataFrame) -> str:
    return (f"processor-approved orders at checkout in the {window} window with a known "
            f"label ({len(rows)} orders, {int(rows['label'].sum())} positive)")


def _ap(labels: np.ndarray, scores: np.ndarray, *, population: str, window: str) -> Metric:
    if 0 < labels.sum() < len(labels):
        return Metric(value=float(average_precision_score(labels, scores)), unit="score",
                      population=population, window=window)
    return Metric.not_evaluated(unit="score", population=population, window=window,
                                reason="needs both positive and negative labels")


def detection_metrics(scorers: Mapping[str, Scorer], rows: pd.DataFrame,
                      window: str) -> dict[str, Metric]:
    """AP of each scorer's raw score and Brier of its calibrated probability on labelled
    ``rows`` (from ``model.features.labelled``), plus the row and positive counts."""
    labels = rows["label"].to_numpy()
    population = _population(window, rows)
    metrics = {
        f"detection.{window}.orders": Metric(value=len(rows), unit="count",
                                             population=population, window=window),
        f"detection.{window}.positives": Metric(value=int(labels.sum()), unit="count",
                                                population=population, window=window),
    }
    for name, scorer in scorers.items():
        metrics[f"detection.{name}.average_precision.{window}"] = _ap(
            labels, scorer.score(rows), population=population, window=window)
        if len(rows):
            brier = float(np.mean((scorer.probability(rows) - labels) ** 2))
            metrics[f"detection.{name}.brier.{window}"] = Metric(
                value=brier, unit="score", population=population, window=window)
        else:
            metrics[f"detection.{name}.brier.{window}"] = Metric.not_evaluated(
                unit="score", population=population, window=window, reason="no labelled orders")
    return metrics


def metrics_on_test_window(scorers: Mapping[str, Scorer], tables: Mapping[str, pd.DataFrame],
                        context: pd.DataFrame, protocol: Protocol) -> dict[str, Metric]:
    """Supporting metrics on one world's test window, with labels as known at the end of
    the follow-up."""
    rows = labelled(checkout_rows(context, tables), tables["labels"], protocol.windows["test"],
                    protocol.observed_end)
    return detection_metrics(scorers, rows, "test")


def shortcut_sensitivity(tables: Mapping[str, pd.DataFrame], context: pd.DataFrame,
                         protocol: Protocol, full: Scorer) -> dict[str, Metric]:
    """Validation AP of the shortcut models (fitted like the classifiers) against the
    ``full`` scorer's, and the perturbation of legitimate first orders."""
    rows = checkout_rows(context, tables)
    train = labelled(rows, tables["labels"], protocol.windows["fit"],
                     protocol.freezes["classifier"])
    valid = labelled(rows, tables["labels"], protocol.windows["validation"],
                     protocol.freezes["policy"])
    labels = valid["label"].to_numpy()
    population = _population("validation", valid)
    prefix = "detection.sensitivity"
    metrics = {f"{prefix}.full.average_precision.validation": _ap(
        labels, full.score(valid), population=population, window="validation")}
    for name, (kind, columns) in SHORTCUTS.items():
        if kind == "tree":
            model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                           class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE)
        else:
            model = HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                                   class_weight=CLASS_WEIGHT,
                                                   random_state=RANDOM_STATE)
        with threadpool_limits(limits=1):
            model.fit(train[list(columns)], train["label"])
            scores = model.predict_proba(valid[list(columns)])[:, 1]
        metrics[f"{prefix}.{name}.average_precision.validation"] = _ap(
            labels, scores, population=population, window="validation")

    scores = full.score(valid)
    cutoff = np.quantile(scores, 1 - TOP_SHARE) if len(scores) else np.nan
    first = valid[(valid["label"] == 0) & (valid["is_first_attempt_user"] == 1)]
    moved = first.assign(**PROFILE)
    first_population = (f"legitimate first orders in the validation window ({len(first)}); "
                        f"profile {PROFILE}")
    for name, frame in (("as_observed", first), ("new_account_profile", moved)):
        above = int((full.score(frame) > cutoff).sum()) if len(frame) else 0
        key = f"{prefix}.first_orders.{name}"
        if len(frame):
            metrics[f"{key}.median_score"] = Metric(
                value=float(np.median(full.score(frame))), unit="score",
                population=first_population, window="validation")
            metrics[f"{key}.above_top_share"] = Metric.from_ratio(
                above, len(frame), unit="share", population=first_population,
                window="validation", note=f"above the top {TOP_SHARE:.0%} validation cutoff")
        else:
            metrics[f"{key}.median_score"] = Metric.not_evaluated(
                unit="score", population=first_population, window="validation",
                reason="no legitimate first orders")
            metrics[f"{key}.above_top_share"] = Metric.not_evaluated(
                unit="share", population=first_population, window="validation",
                reason="no legitimate first orders", numerator=0, denominator=0)
    return metrics
