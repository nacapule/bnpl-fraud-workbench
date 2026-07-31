"""Train the two chronological fraud baselines.

Run with ``python -m model.train`` from the repository root.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.features import FEATURE_COLUMNS, build_features, load_feature_frames

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO / "model" / "artifacts"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "HistGradient Boosting": "hist_gradient_boosting.joblib",
}


def load_config(path: str | Path = REPO / "config.yaml") -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def chronological_split(
    features: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Months 1--N train, remaining simulator months holdout.

    There is intentionally no shuffle: deployment predicts later orders from
    earlier behavior, and a shuffled split would leak future population and
    fraud-pattern conditions into training.
    """
    start = pd.Timestamp(config["simulator"]["start_date"])
    cutoff = start + pd.Timedelta(days=30 * int(config["model"]["train_months"]))
    train = features[features["ts"] < cutoff].copy()
    holdout = features[features["ts"] >= cutoff].copy()
    if train.empty or holdout.empty:
        raise ValueError(f"chronological cutoff {cutoff} produced an empty split")
    if train["ts"].max() >= holdout["ts"].min():
        raise AssertionError("chronological split overlaps")
    return train, holdout


def make_models(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "HistGradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=False,
            max_iter=100,
            random_state=seed,
        ),
    }


def main() -> None:
    config = load_config()
    frames = load_feature_frames(REPO / "data")
    features = build_features(**frames)
    train, holdout = chronological_split(features, config)

    print(
        f"train:   {len(train):,} orders | base rate {train['label'].mean():.3%} | "
        f"{train['ts'].min()} to {train['ts'].max()}"
    )
    print(
        f"holdout: {len(holdout):,} orders | base rate {holdout['label'].mean():.3%} | "
        f"{holdout['ts'].min()} to {holdout['ts'].max()}"
    )

    x_train = train[FEATURE_COLUMNS]
    y_train = train["label"]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in make_models(int(config["seed"])).items():
        model.fit(x_train, y_train)
        destination = ARTIFACT_DIR / MODEL_FILES[name]
        joblib.dump(model, destination)
        print(f"saved {name}: {destination.relative_to(REPO)}")

    with (ARTIFACT_DIR / "feature_list.json").open("w") as handle:
        json.dump(FEATURE_COLUMNS, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
