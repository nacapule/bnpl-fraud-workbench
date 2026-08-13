"""Reproduce the prevention counterfactuals quoted in CASE-01 through CASE-05."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.features import build_features, load_feature_frames  # noqa: E402
from rules.definitions import RULES  # noqa: E402
from rules.engine import build_enriched, run_rules  # noqa: E402


def _config() -> dict[str, Any]:
    with (REPO / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def _operating_point() -> dict[str, Any]:
    return json.loads((REPO / "reports" / "operating_point.json").read_text())


def _holdout_days() -> int:
    orders = pd.read_csv(REPO / "data" / "orders.csv", parse_dates=["ts"], usecols=["ts"])
    return max((orders["ts"].max() - pd.Timestamp(_config()["holdout_start"])).days, 1)


@lru_cache(maxsize=1)
def _enriched() -> pd.DataFrame:
    frame = run_rules(build_enriched(REPO / "data"))
    labels = pd.read_csv(REPO / "data" / "labels.csv").drop_duplicates("order_id")
    pattern = labels.set_index("order_id")["pattern_id"]
    frame["pattern_id"] = frame["order_id"].map(pattern)
    frame["is_fraud"] = frame["pattern_id"].notna()
    return frame


def _holdout() -> pd.DataFrame:
    cutoff = pd.Timestamp(_config()["holdout_start"])
    return _enriched()[_enriched()["ts"] >= cutoff].copy()


def case01_ato_address_bonus() -> dict[str, int]:
    holdout = _holdout()
    addresses = pd.read_csv(
        REPO / "data" / "addresses.csv", parse_dates=["added_ts"]
    ).set_index("address_id")["added_ts"]
    address_age_hours = (
        holdout["ts"] - holdout["ship_address_id"].map(addresses)
    ).dt.total_seconds() / 3600
    conjunction = holdout["fired_R01"] & address_age_hours.between(
        0, 48, inclusive="left"
    )
    decline_band = int(_operating_point()["decline_band"])
    ato = holdout["pattern_id"].eq("P-ATO")
    proposed_score = holdout["score"] + 15 * conjunction.astype(int)
    return {
        "holdout_ato_orders": int(ato.sum()),
        "baseline_auto_declines": int((ato & holdout["score"].ge(decline_band)).sum()),
        "proposed_auto_declines": int((ato & proposed_score.ge(decline_band)).sum()),
        "added_benign_auto_declines": int(
            (
                ~holdout["is_fraud"]
                & holdout["score"].lt(decline_band)
                & proposed_score.ge(decline_band)
            ).sum()
        ),
    }


def case02_device_cooldown() -> dict[str, int]:
    orders = pd.read_csv(
        REPO / "data" / "orders.csv",
        parse_dates=["ts"],
        usecols=["user_id", "device_id", "ts", "status"],
    )
    declines = orders[orders["status"].eq("declined")].sort_values(["device_id", "ts"])
    cooled_devices = []
    for device_id, group in declines.groupby("device_id"):
        timestamps = group["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
        left = np.searchsorted(timestamps, timestamps - 24 * 3600 * 10**9, side="left")
        if np.any(np.arange(len(timestamps)) - left + 1 >= 5):
            cooled_devices.append(device_id)

    fraud_users = set(pd.read_csv(REPO / "data" / "labels.csv")["user_id"])
    linked_users = (
        orders[orders["device_id"].isin(cooled_devices)]
        .groupby("device_id")["user_id"]
        .agg(set)
    )
    fraud_linked = sum(any(user in fraud_users for user in users) for users in linked_users)
    benign_linked = sum(any(user not in fraud_users for user in users) for users in linked_users)
    return {
        "devices_cooled": len(cooled_devices),
        "fraud_linked_devices": int(fraud_linked),
        "benign_linked_devices": int(benign_linked),
    }


def case03_r08_threshold() -> dict[str, int | float]:
    holdout = _holdout()
    review_band = int(_operating_point()["review_band"])
    r08 = next(rule for rule in RULES if rule.id == "R08")
    newly_fired = holdout["users_on_ship_addr"].ge(2) & ~holdout["fired_R08"]
    added = holdout["score"].lt(review_band) & (
        holdout["score"] + r08.weight * newly_fired.astype(int)
    ).ge(review_band)
    return {
        "added_alerts": int(added.sum()),
        "added_benign_alerts": int((added & ~holdout["is_fraud"]).sum()),
        "added_fraud_alerts": int((added & holdout["is_fraud"]).sum()),
        "added_alerts_per_day": round(float(added.sum()) / _holdout_days(), 1),
    }


def case03_rejected_address_attach() -> dict[str, int]:
    approved = pd.read_csv(
        REPO / "data" / "orders.csv",
        usecols=["order_id", "user_id", "ship_address_id", "status"],
    )
    approved = approved[approved["status"].eq("approved")]
    attached_accounts = approved.groupby("ship_address_id")["user_id"].nunique()
    flagged = set(attached_accounts[attached_accounts.ge(3)].index)
    fraud_orders = set(pd.read_csv(REPO / "data" / "labels.csv")["order_id"])
    orders_by_address = (
        approved[approved["ship_address_id"].isin(flagged)]
        .groupby("ship_address_id")["order_id"]
        .agg(list)
    )
    benign = sum(
        any(order_id not in fraud_orders for order_id in order_ids)
        for order_ids in orders_by_address
    )
    return {
        "flagged_addresses": len(flagged),
        "addresses_with_benign_orders": int(benign),
    }


@lru_cache(maxsize=1)
def _model_features() -> pd.DataFrame:
    return build_features(**load_feature_frames(REPO / "data"))


def case04_never_pay_candidate() -> dict[str, int | float | str]:
    holdout = _holdout()
    features = _model_features()
    cutoff = pd.Timestamp(_config()["holdout_start"])
    features = features[features["ts"] >= cutoff].merge(
        holdout[["order_id", "score", "pattern_id", "is_fraud"]],
        on="order_id",
        validate="one_to_one",
    )
    never_pay = features["pattern_id"].eq("P-NEVERPAY")
    candidate = (
        features["is_first_order"].eq(1)
        & features["amount_over_category_median"].gt(1.5)
        & features["account_age_days"].lt(2)
    )
    review_band = int(_operating_point()["review_band"])
    added = features["score"].lt(review_band) & (
        features["score"] + 15 * candidate.astype(int)
    ).ge(review_band)
    added_patterns = sorted(features.loc[added, "pattern_id"].dropna().unique())
    benign_first = features[~features["is_fraud"] & features["is_first_order"].eq(1)]
    cap_rate = float(benign_first["amount_over_category_median"].gt(1.5).mean())
    return {
        "holdout_never_pay_orders": int(never_pay.sum()),
        "currently_alerted_never_pay_orders": int(
            (never_pay & features["score"].ge(review_band)).sum()
        ),
        "candidate_added_alerts": int(added.sum()),
        "candidate_added_pattern": added_patterns[0],
        "benign_first_order_cap_rate": round(cap_rate, 3),
    }


def case05_r03_suppression() -> dict[str, int | float]:
    holdout = _holdout()
    cutoff = pd.Timestamp(_config()["holdout_start"])
    alerts = pd.read_csv(REPO / "data" / "alerts.csv", parse_dates=["ts"])
    alerts = alerts[alerts["ts"] >= cutoff].copy()
    alerts["rule_ids"] = alerts["fired_rules"].map(
        lambda value: [rule["id"] for rule in json.loads(value)]
    )
    r03_only_orders = set(
        alerts.loc[alerts["rule_ids"].map(lambda ids: ids == ["R03"]), "order_id"]
    )
    r03_only = holdout[holdout["order_id"].isin(r03_only_orders)]
    suppressed = (
        r03_only["account_age_days"].gt(180)
        & ~r03_only["avs_bad"]
        & r03_only["cred_change_hours"].gt(72)
    )
    return {
        "r03_only_alerts": len(r03_only),
        "suppressed_alerts": int(suppressed.sum()),
        "suppressed_fraud_alerts": int((suppressed & r03_only["is_fraud"]).sum()),
        "suppressed_alerts_per_day": round(float(suppressed.sum()) / _holdout_days(), 1),
    }


def render_report() -> str:
    sections = [
        ("CASE-01 — ATO address conjunction", case01_ato_address_bonus()),
        ("CASE-02 — device cooldown", case02_device_cooldown()),
        ("CASE-03 — R08 threshold", case03_r08_threshold()),
        ("CASE-03 — rejected address-attach idea", case03_rejected_address_attach()),
        ("CASE-04 — never-pay checkout candidate", case04_never_pay_candidate()),
        ("CASE-05 — R03 suppression", case05_r03_suppression()),
    ]
    lines = [
        "# Case prevention follow-ups",
        "",
        "Read-only counterfactuals over `data/`, using `config.yaml` holdout_start and the "
        "committed operating point in `reports/operating_point.json`.",
    ]
    for title, result in sections:
        lines.extend(["", f"## {title}", "", f"```json\n{json.dumps(result, sort_keys=True)}\n```"])
    return "\n".join(lines) + "\n"


def main() -> None:
    destination = REPO / "reports" / "followups.md"
    destination.write_text(render_report())
    print(f"wrote {destination.relative_to(REPO)}")


if __name__ == "__main__":
    main()
