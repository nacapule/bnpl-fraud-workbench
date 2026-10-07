"""Every rule condition: one minimal context row where it must fire, one near miss where
it must not. Conditions read as-of context columns, so these are pure unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import asof
from rules.definitions import COLUMNS, CONDITIONS, RULE_IDS, WEIGHTS
from rules.engine import fired, rationales, score

CONDITION = {c.id: c for c in CONDITIONS}

BASE = {
    "hours_since_credential_change": asof.NO_EVENT_HOURS, "account_age_days": 400.0,
    "device_link_age_hours": 5000.0, "accounts_on_device_30d": 1, "bin_ip_country_mismatch": 0,
    "avs_mismatch": 0, "cvv_mismatch": 0, "is_first_attempt_user": 0,
    "amount_over_category_p95": 0.4, "attempts_user_24h": 1, "attempts_device_24h": 1,
    "email_domain_class": 0, "email_root_other_accounts": 0, "processor_declines_card_24h": 0,
    "processor_declines_device_24h": 0, "accounts_on_address_30d": 1,
    "inr_disputes_opened_user": 0, "promo_uses_linked_accounts": 0,
    "geo_kmh_from_previous_attempt": 0.0,
}


def frame(**overrides: object) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **overrides}])


CASES = {
    # condition: (must fire, near miss)
    "R01": (dict(hours_since_credential_change=20.0, device_link_age_hours=10.0),
            dict(hours_since_credential_change=20.0, device_link_age_hours=10.0,
                 account_age_days=89.0)),
    "R02": (dict(accounts_on_device_30d=3), dict(accounts_on_device_30d=2)),
    "R03": (dict(bin_ip_country_mismatch=1, avs_mismatch=1), dict(bin_ip_country_mismatch=1)),
    "R04": (dict(is_first_attempt_user=1, amount_over_category_p95=1.3, account_age_days=1.0),
            dict(is_first_attempt_user=1, amount_over_category_p95=1.3, account_age_days=30.0)),
    "R05": (dict(attempts_user_24h=4), dict(attempts_user_24h=3, attempts_device_24h=5)),
    "R06(a)": (dict(email_domain_class=2), dict(email_domain_class=1)),
    "R06(b)": (dict(email_root_other_accounts=1), dict()),
    "R07": (dict(processor_declines_device_24h=3), dict(processor_declines_card_24h=2,
                                                        processor_declines_device_24h=2)),
    "R08": (dict(accounts_on_address_30d=3), dict(accounts_on_address_30d=2)),
    "R09": (dict(inr_disputes_opened_user=2), dict(inr_disputes_opened_user=1)),
    "R10": (dict(promo_uses_linked_accounts=3), dict(promo_uses_linked_accounts=2)),
    "R11": (dict(geo_kmh_from_previous_attempt=2000.0),
            dict(geo_kmh_from_previous_attempt=900.0)),
}


def test_active_rule_ids_are_r01_to_r11_with_two_r06_parts() -> None:
    """R12 is retired and its id is never reused; R06 has two parts under one id."""
    assert RULE_IDS == tuple(f"R{i:02d}" for i in range(1, 12))
    assert [c.id for c in CONDITIONS if c.rule == "R06"] == ["R06(a)", "R06(b)"]
    assert set(CASES) == set(CONDITION)


def test_conditions_read_only_context_columns() -> None:
    assert set(COLUMNS) <= set(asof.COLUMN_NAMES)
    assert set(COLUMNS) == set(BASE)


@pytest.mark.parametrize("condition_id", list(CASES))
def test_condition_fires_and_near_miss(condition_id: str) -> None:
    must, miss = CASES[condition_id]
    condition = CONDITION[condition_id]
    assert bool(condition.fire(frame(**must)).iloc[0]), f"{condition_id} must fire"
    assert not bool(condition.fire(frame(**miss)).iloc[0]), f"{condition_id} near miss fired"


@pytest.mark.parametrize("condition_id", list(CASES))
def test_rationale_cites_a_value(condition_id: str) -> None:
    rows = frame(**CASES[condition_id][0])
    (reasons,) = rationales(rows)
    text = next(r["rationale"] for r in reasons if r["id"] == condition_id)
    assert any(ch.isdigit() for ch in text) or condition_id == "R06(a)", text


def test_repeat_claims_rationale_states_only_the_count() -> None:
    """R09 rests on a dispute count alone; its rationale must not claim repayment evidence
    that was never computed."""
    (reasons,) = rationales(frame(inr_disputes_opened_user=3))
    text = reasons[0]["rationale"]
    assert "3 item-not-received disputes" in text
    assert "repay" not in text.lower()


def test_r04_needs_a_category_reference() -> None:
    """Below the minimum sample the category percentile is missing and R04 cannot fire."""
    rows = frame(is_first_attempt_user=1, amount_over_category_p95=np.nan, account_age_days=1.0)
    assert not fired(rows)["R04"].iloc[0]


def test_r06_counts_once_when_both_parts_fire() -> None:
    both = frame(email_domain_class=2, email_root_other_accounts=2)
    flags = fired(both)
    assert flags[["R06", "R06(a)", "R06(b)"]].iloc[0].tolist() == [True, True, True]
    assert score(both)[0] == WEIGHTS["R06"]


def test_score_is_the_sum_of_fired_weights() -> None:
    rows = frame(hours_since_credential_change=2.0, device_link_age_hours=1.0,
                 attempts_user_24h=6)
    assert score(rows)[0] == WEIGHTS["R01"] + WEIGHTS["R05"] == 75


def test_raising_the_decline_band_never_adds_declines() -> None:
    rows = pd.concat([
        frame(), frame(email_domain_class=2), frame(inr_disputes_opened_user=2),
        frame(hours_since_credential_change=2.0, device_link_age_hours=1.0, attempts_user_24h=6),
        frame(hours_since_credential_change=2.0, device_link_age_hours=1.0, attempts_user_24h=6,
              email_domain_class=2),
    ], ignore_index=True)
    scores = score(rows)
    declines = [int((scores >= band).sum()) for band in (70, 80, 90, 100, 110)]
    assert declines == sorted(declines, reverse=True)


def test_missing_input_column_raises() -> None:
    with pytest.raises(KeyError, match="accounts_on_device_30d"):
        fired(frame().drop(columns="accounts_on_device_30d"))
