"""Rule conditions R01–R11 of the fraud policy (FP-2 §6.2) and their score weights.

Each condition reads only as-of context columns (``core.asof``), evaluated at the
decision time, and returns a boolean fire mask plus a rationale with the values it
rests on (FP-2 §6.1). R06 has two parts under one rule id: R06(a), a disposable
email domain (Context), and R06(b), a normalized email shared with another account
(Linkage); the rule fires when either part does and its weight counts once.
R12 (a vendor score) is retired and its id is never reused.

Thresholds must match the policy text (FP-2 §12.2); weights add up to the rule
score that checkout routing compares with the bands in ``config/policy.yaml``
(FP-2 §4.1). Families, settling outcomes and household exceptions belong to the
evidence classification (``core.evidence``), not to the score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

Mask = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Condition:
    id: str  # "R01" ... "R11", or "R06(a)" / "R06(b)"
    rule: str  # the rule id it belongs to
    name: str
    columns: tuple[str, ...]  # the core.asof columns it reads
    fire: Mask
    explain: Callable[[pd.DataFrame], pd.Series]  # only called on fired rows


def _number(values: pd.Series, digits: int = 0) -> pd.Series:
    rounded = values.astype(float).round(digits)
    return (rounded.astype(int) if digits == 0 else rounded).astype(str)


def _r01(df: pd.DataFrame) -> pd.Series:
    return ((df.hours_since_credential_change <= 48) & (df.account_age_days >= 90)
            & (df.device_link_age_hours <= 72))


def _r01_text(df: pd.DataFrame) -> pd.Series:
    return ("password or email change " + _number(df.hours_since_credential_change, 1)
            + "h before the order on a " + _number(df.account_age_days) + "-day-old account; "
            + "device first used on it " + _number(df.device_link_age_hours, 1) + "h before")


def _r02(df: pd.DataFrame) -> pd.Series:
    return df.accounts_on_device_30d >= 3


def _r02_text(df: pd.DataFrame) -> pd.Series:
    return (_number(df.accounts_on_device_30d) + " accounts, this one included, with an "
            "attempt or account event on this device in the past 30 days")


def _r03(df: pd.DataFrame) -> pd.Series:
    return (df.bin_ip_country_mismatch == 1) & ((df.avs_mismatch == 1) | (df.cvv_mismatch == 1))


def _r03_text(df: pd.DataFrame) -> pd.Series:
    return ("card issuing country differs from the IP country; AVS mismatch="
            + _number(df.avs_mismatch) + ", CVV mismatch=" + _number(df.cvv_mismatch))


def _r04(df: pd.DataFrame) -> pd.Series:
    return ((df.is_first_attempt_user == 1) & (df.amount_over_category_p95 > 1.0)
            & (df.account_age_days < 7))


def _r04_text(df: pd.DataFrame) -> pd.Series:
    return ("first order attempt at " + _number(df.amount_over_category_p95, 2)
            + "x the category's 95th percentile of earlier approved amounts, account "
            + _number(df.account_age_days * 24) + "h old")


def _r05(df: pd.DataFrame) -> pd.Series:
    return (df.attempts_user_24h > 3) | (df.attempts_device_24h > 5)


def _r05_text(df: pd.DataFrame) -> pd.Series:
    return (_number(df.attempts_user_24h) + " order attempts on the account and "
            + _number(df.attempts_device_24h) + " on the device in the 24h up to this one")


def _r06a(df: pd.DataFrame) -> pd.Series:
    return df.email_domain_class == 2


def _r06a_text(df: pd.DataFrame) -> pd.Series:
    return pd.Series("disposable email domain", index=df.index)


def _r06b(df: pd.DataFrame) -> pd.Series:
    return df.email_root_other_accounts >= 1


def _r06b_text(df: pd.DataFrame) -> pd.Series:
    return (_number(df.email_root_other_accounts)
            + " other accounts hold an email that normalizes to this account's")


def _r07(df: pd.DataFrame) -> pd.Series:
    return (df.processor_declines_card_24h >= 3) | (df.processor_declines_device_24h >= 3)


def _r07_text(df: pd.DataFrame) -> pd.Series:
    return (_number(df.processor_declines_card_24h) + " processor declines on this card and "
            + _number(df.processor_declines_device_24h) + " on this device in the 24h before")


def _r08(df: pd.DataFrame) -> pd.Series:
    return df.accounts_on_address_30d >= 3


def _r08_text(df: pd.DataFrame) -> pd.Series:
    return (_number(df.accounts_on_address_30d) + " accounts, this one included, with an "
            "attempt to this shipping address in the past 30 days")


def _r09(df: pd.DataFrame) -> pd.Series:
    return df.inr_disputes_opened_user >= 2


def _r09_text(df: pd.DataFrame) -> pd.Series:
    return (_number(df.inr_disputes_opened_user)
            + " item-not-received disputes opened on the account's earlier orders")


def _r10(df: pd.DataFrame) -> pd.Series:
    return df.promo_uses_linked_accounts >= 3


def _r10_text(df: pd.DataFrame) -> pd.Series:
    return ("first-purchase promotion used by " + _number(df.promo_uses_linked_accounts)
            + " accounts linked by a shared device or email, this one included")


def _r11(df: pd.DataFrame) -> pd.Series:
    return df.geo_kmh_from_previous_attempt > 900


def _r11_text(df: pd.DataFrame) -> pd.Series:
    return ("previous attempt from another country implies "
            + _number(df.geo_kmh_from_previous_attempt) + " km/h")


CONDITIONS: tuple[Condition, ...] = (
    Condition("R01", "R01", "credential change before the order", (
        "hours_since_credential_change", "account_age_days", "device_link_age_hours"),
        _r01, _r01_text),
    Condition("R02", "R02", "multi-account device", ("accounts_on_device_30d",), _r02, _r02_text),
    Condition("R03", "R03", "geography and verification mismatch", (
        "bin_ip_country_mismatch", "avs_mismatch", "cvv_mismatch"), _r03, _r03_text),
    Condition("R04", "R04", "oversized first order", (
        "is_first_attempt_user", "amount_over_category_p95", "account_age_days"),
        _r04, _r04_text),
    Condition("R05", "R05", "order burst", ("attempts_user_24h", "attempts_device_24h"),
              _r05, _r05_text),
    Condition("R06(a)", "R06", "disposable email domain", ("email_domain_class",),
              _r06a, _r06a_text),
    Condition("R06(b)", "R06", "shared normalized email", ("email_root_other_accounts",),
              _r06b, _r06b_text),
    Condition("R07", "R07", "card testing", (
        "processor_declines_card_24h", "processor_declines_device_24h"), _r07, _r07_text),
    Condition("R08", "R08", "shared shipping address", ("accounts_on_address_30d",),
              _r08, _r08_text),
    Condition("R09", "R09", "repeat item-not-received disputes", ("inr_disputes_opened_user",),
              _r09, _r09_text),
    Condition("R10", "R10", "promotion used across linked accounts",
              ("promo_uses_linked_accounts",), _r10, _r10_text),
    Condition("R11", "R11", "impossible geo-velocity", ("geo_kmh_from_previous_attempt",),
              _r11, _r11_text),
)

# Score weight per rule id (R06 counts once, whichever part fires). R12 is retired.
WEIGHTS: dict[str, int] = {
    "R01": 45, "R02": 40, "R03": 35, "R04": 30, "R05": 30, "R06": 25,
    "R07": 40, "R08": 35, "R09": 45, "R10": 30, "R11": 25,
}
RULE_IDS = tuple(WEIGHTS)
COLUMNS = tuple(dict.fromkeys(column for c in CONDITIONS for column in c.columns))
