"""Every rule: one minimal frame where it MUST fire, one near-miss where it
must NOT. Rules consume the enriched-frame columns, so these are pure unit
tests — no DB, no CSVs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rules.definitions import RULES

RULE = {r.id: r for r in RULES}

BASE = {
    "account_age_days": 400.0, "cred_change_hours": np.inf, "device_new": False,
    "users_on_device": 1, "users_on_ship_addr": 1, "is_first_order": False,
    "amount_vs_p95": 0.4, "account_age_hours": 9600.0, "n_user_24h": 1, "n_dev_24h": 1,
    "bin_ip_mismatch": False, "avs_bad": False, "cvv_bad": False,
    "email_disposable": False, "email_root_dup": False, "email_root_accounts": 1,
    "card_test_declines": 0, "prior_inr_cbs": 0, "promo_cluster_size": 0,
    "geo_kmh": 0.0, "prev_ip_country": "", "vendor_email_score": np.nan,
    "vendor_ip_score": np.nan, "device_id": 7, "ship_address_id": 9, "amount": 100.0,
    "avs_result": "Y", "cvv_result": "M", "bin_country": "US", "ip_country": "US",
}


def frame(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **overrides}])


CASES = {
    # rule_id: (must_fire_overrides, near_miss_overrides)
    "R01": (
        dict(cred_change_hours=20.0, device_new=True),
        dict(cred_change_hours=20.0, device_new=True, account_age_days=30.0),  # young acct
    ),
    "R02": (dict(users_on_device=3), dict(users_on_device=2)),
    "R03": (
        dict(bin_ip_mismatch=True, avs_bad=True),
        dict(bin_ip_mismatch=True),  # mismatch alone is noise
    ),
    "R04": (
        dict(is_first_order=True, amount_vs_p95=1.3, account_age_days=1.0),
        dict(is_first_order=True, amount_vs_p95=1.3, account_age_days=30.0),
    ),
    "R05": (dict(n_user_24h=4), dict(n_user_24h=3, n_dev_24h=5)),
    "R06": (dict(email_disposable=True), dict()),
    "R07": (dict(card_test_declines=3), dict(card_test_declines=2)),
    "R08": (dict(users_on_ship_addr=3), dict(users_on_ship_addr=2)),
    "R09": (dict(prior_inr_cbs=2), dict(prior_inr_cbs=1)),
    "R10": (dict(promo_cluster_size=3), dict(promo_cluster_size=2)),
    "R11": (dict(geo_kmh=2000.0, prev_ip_country="DE"), dict(geo_kmh=800.0)),
    "R12": (dict(vendor_ip_score=90.0), dict(vendor_ip_score=80.0, vendor_email_score=60.0)),
}


@pytest.mark.parametrize("rule_id", list(CASES))
def test_rule_fires_and_near_miss(rule_id: str) -> None:
    fire_over, miss_over = CASES[rule_id]
    rule = RULE[rule_id]
    assert bool(rule.fire(frame(**fire_over)).iloc[0]), f"{rule_id} must fire"
    assert not bool(rule.fire(frame(**miss_over)).fillna(False).iloc[0]), (
        f"{rule_id} must not fire on near-miss"
    )


@pytest.mark.parametrize("rule_id", list(CASES))
def test_rationale_contains_concrete_value(rule_id: str) -> None:
    fire_over, _ = CASES[rule_id]
    rule = RULE[rule_id]
    df = frame(**fire_over)
    text = rule.explain(df[rule.fire(df)]).iloc[0]
    has_number = any(ch.isdigit() for ch in text)
    has_code = any(tok in text for tok in ("US", "DE", "AVS=", "CVV="))  # R03 cites codes
    assert has_number or has_code, f"{rule_id} rationale has no concrete value: {text}"


def test_band_math() -> None:
    """Score = sum of fired weights; a frame firing R01+R05 crosses review at 75."""
    df = frame(cred_change_hours=2.0, device_new=True, n_user_24h=6)
    score = sum(r.weight for r in RULES if bool(r.fire(df).fillna(False).iloc[0]))
    assert score == RULE["R01"].weight + RULE["R05"].weight == 75


def test_monotonicity_of_bands() -> None:
    """Raising the decline band can never increase auto-declines."""
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.integers(0, 150, 500))
    declines = [int((scores >= band).sum()) for band in (70, 80, 90, 100, 110)]
    assert declines == sorted(declines, reverse=True)
