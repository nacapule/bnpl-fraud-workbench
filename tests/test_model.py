from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.features import FEATURE_COLUMNS, build_features, load_feature_frames  # noqa: E402
from model.train import chronological_split, load_config, make_models  # noqa: E402


@pytest.fixture(scope="session")
def tiny_world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    destination = tmp_path_factory.mktemp("model-world")
    environment = os.environ.copy()
    environment.update({"SIM_SCALE": "0.02", "SIM_OUT": str(destination)})
    subprocess.run(
        [sys.executable, "-m", "simulator.generate"],
        cwd=REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return load_feature_frames(destination)


def _truncate_world(
    frames: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    truncated = {name: frame.copy() for name, frame in frames.items()}
    for name, column in {
        "orders": "ts",
        "users": "signup_ts",
        "addresses": "added_ts",
        "account_events": "ts",
        "promo_redemptions": "ts",
    }.items():
        truncated[name] = truncated[name][truncated[name][column] <= cutoff].copy()

    installments = truncated["installments"]
    installments = installments[installments["due_ts"] <= cutoff].copy()
    installments.loc[installments["paid_ts"] > cutoff, "paid_ts"] = pd.NaT
    truncated["installments"] = installments

    kept_orders = set(truncated["orders"]["order_id"])
    truncated["plans"] = truncated["plans"][
        truncated["plans"]["order_id"].isin(kept_orders)
    ].copy()
    truncated["labels"] = truncated["labels"][
        truncated["labels"]["order_id"].isin(kept_orders)
    ].copy()
    return truncated


def test_point_in_time_leakage_guard(tiny_world: dict[str, pd.DataFrame]) -> None:
    full = build_features(**tiny_world)
    sample = full.sample(n=min(25, len(full)), random_state=load_config()["seed"])

    for expected in sample.itertuples(index=False):
        cutoff = pd.Timestamp(expected.ts)
        rebuilt = build_features(**_truncate_world(tiny_world, cutoff))
        actual_row = rebuilt[rebuilt["order_id"].eq(expected.order_id)][FEATURE_COLUMNS]
        expected_row = full[full["order_id"].eq(expected.order_id)][FEATURE_COLUMNS]
        assert len(actual_row) == 1
        assert_frame_equal(
            actual_row.reset_index(drop=True),
            expected_row.reset_index(drop=True),
            check_exact=True,
        )


def test_feature_build_is_deterministic(tiny_world: dict[str, pd.DataFrame]) -> None:
    first = build_features(**tiny_world)
    second = build_features(**tiny_world)
    assert_frame_equal(first, second, check_exact=True)


def test_chronological_split_has_no_overlap(tiny_world: dict[str, pd.DataFrame]) -> None:
    features = build_features(**tiny_world)
    config = load_config()
    train, holdout = chronological_split(features, config)
    assert train["ts"].max() < holdout["ts"].min()
    assert train["ts"].max() < pd.Timestamp(config["holdout_start"])
    assert holdout["ts"].min() >= pd.Timestamp(config["holdout_start"])


def test_tiny_world_logistic_smoke(tiny_world: dict[str, pd.DataFrame]) -> None:
    features = build_features(**tiny_world)
    train, holdout = chronological_split(features, load_config())
    model = make_models(load_config()["seed"])["Logistic Regression"]
    model.fit(train[FEATURE_COLUMNS], train["label"])
    scores = model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    assert average_precision_score(holdout["label"], scores) > 0.05
