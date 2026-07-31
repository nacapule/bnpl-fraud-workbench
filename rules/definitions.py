"""Rule definitions R01–R12. Intents and weights mirror policy/fraud-policy.md §4;
if you change a threshold here, update the policy doc (FP-1 §8 change control).

Each rule consumes the point-in-time enriched order frame built by
``rules.engine`` (every column is computed strictly from data at or before the
order timestamp) and returns a boolean fire mask plus a vectorized rationale
string with concrete values — alerts must be explainable (FP-1 §2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    targets: tuple[str, ...]
    rationale: str  # intent, one sentence (policy §4 wording)
    weight: int
    fire: Callable[[pd.DataFrame], pd.Series]
    explain: Callable[[pd.DataFrame], pd.Series]  # only called on fired rows


def _r01_fire(df: pd.DataFrame) -> pd.Series:
    return (df.account_age_days >= 90) & (df.cred_change_hours <= 48) & df.device_new


def _r01_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "credential/contact change " + df.cred_change_hours.round(1).astype(str)
        + "h before order on " + df.account_age_days.astype(int).astype(str)
        + "d-old account, order from new device d_" + df.device_id.astype(str)
    )


def _r02_fire(df: pd.DataFrame) -> pd.Series:
    return df.users_on_device >= 3


def _r02_explain(df: pd.DataFrame) -> pd.Series:
    return (
        df.users_on_device.astype(int).astype(str)
        + " accounts have transacted on device d_" + df.device_id.astype(str)
        + " as of this order"
    )


def _r03_fire(df: pd.DataFrame) -> pd.Series:
    return df.bin_ip_mismatch & (df.avs_bad | df.cvv_bad)


def _r03_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "card BIN country " + df.bin_country + " vs IP country " + df.ip_country
        + " with AVS=" + df.avs_result + "/CVV=" + df.cvv_result
    )


def _r04_fire(df: pd.DataFrame) -> pd.Series:
    return df.is_first_order & (df.amount_vs_p95 > 1.0) & (df.account_age_days < 7)


def _r04_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "first-ever order $" + df.amount.round(2).astype(str) + " is "
        + df.amount_vs_p95.round(2).astype(str) + "x the category P95, account "
        + (df.account_age_days * 24).round(0).astype(int).astype(str) + "h old"
    )


def _r05_fire(df: pd.DataFrame) -> pd.Series:
    return (df.n_user_24h > 3) | (df.n_dev_24h > 5)


def _r05_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "burst: " + df.n_user_24h.astype(int).astype(str) + " orders/24h on account, "
        + df.n_dev_24h.astype(int).astype(str) + " on device d_" + df.device_id.astype(str)
    )


def _r06_fire(df: pd.DataFrame) -> pd.Series:
    return df.email_disposable | df.email_root_dup


def _r06_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "email identity cost-reduction: disposable_domain=" + df.email_disposable.astype(str)
        + ", shared_root_accounts=" + df.email_root_accounts.astype(int).astype(str)
    )


def _r07_fire(df: pd.DataFrame) -> pd.Series:
    return df.card_test_declines >= 3


def _r07_explain(df: pd.DataFrame) -> pd.Series:
    return (
        df.card_test_declines.astype(int).astype(str)
        + " declined attempts on this card/device in the prior 24h, then this approval"
    )


def _r08_fire(df: pd.DataFrame) -> pd.Series:
    return df.users_on_ship_addr >= 3


def _r08_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "ship-to address a_" + df.ship_address_id.astype(str) + " already used by "
        + df.users_on_ship_addr.astype(int).astype(str) + " distinct accounts"
    )


def _r09_fire(df: pd.DataFrame) -> pd.Series:
    return df.prior_inr_cbs >= 2


def _r09_explain(df: pd.DataFrame) -> pd.Series:
    return (
        df.prior_inr_cbs.astype(int).astype(str)
        + " prior item-not-received chargebacks on this account with healthy repayment"
    )


def _r10_fire(df: pd.DataFrame) -> pd.Series:
    return df.promo_cluster_size >= 3


def _r10_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "promo redeemed inside a device/address cluster with "
        + df.promo_cluster_size.astype(int).astype(str) + " linked redemptions"
    )


def _r11_fire(df: pd.DataFrame) -> pd.Series:
    return df.geo_kmh > 900


def _r11_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "impossible travel: " + df.prev_ip_country + "→" + df.ip_country + " implies "
        + df.geo_kmh.round(0).astype(int).astype(str) + " km/h"
    )


def _r12_fire(df: pd.DataFrame) -> pd.Series:
    return (df.vendor_email_score >= 85) | (df.vendor_ip_score >= 85)


def _r12_explain(df: pd.DataFrame) -> pd.Series:
    return (
        "vendor risk score email=" + df.vendor_email_score.fillna(-1).astype(int).astype(str)
        + " ip=" + df.vendor_ip_score.fillna(-1).astype(int).astype(str) + " (threshold 85)"
    )


RULES: list[Rule] = [
    Rule("R01", "credential change before order", ("P-ATO",),
         "Account takeover leaves a manipulation trail before the money moves",
         45, _r01_fire, _r01_explain),
    Rule("R02", "multi-account device", ("P-SYNTH", "P-PROMO"),
         "Device reuse is the cheapest ring signal", 40, _r02_fire, _r02_explain),
    Rule("R03", "geo + verification mismatch", ("P-STOLEN",),
         "BIN/IP mismatch compounds with AVS/CVV failure; either alone is noise",
         35, _r03_fire, _r03_explain),
    Rule("R04", "oversized first order", ("P-STOLEN", "P-NEVERPAY"),
         "New accounts starting at the top of the amount distribution front-load risk",
         30, _r04_fire, _r04_explain),
    Rule("R05", "order burst", ("P-STOLEN", "P-ATO"),
         "Velocity beyond organic shopping rhythm", 30, _r05_fire, _r05_explain),
    Rule("R06", "identity cost reduction", ("P-SYNTH", "P-PROMO"),
         "Disposable/duplicated mailboxes precede scaled abuse", 25, _r06_fire, _r06_explain),
    Rule("R07", "card testing", ("P-STOLEN",),
         "Decline burst then success is credential validation before spend",
         40, _r07_fire, _r07_explain),
    Rule("R08", "shared drop address", ("P-SYNTH",),
         "Goods land somewhere; drops collapse rings", 35, _r08_fire, _r08_explain),
    Rule("R09", "repeat INR disputes", ("P-INR-ABUSE",),
         "Repeat INR with healthy repayment is the friendly-fraud shape",
         45, _r09_fire, _r09_explain),
    Rule("R10", "promo farming cluster", ("P-PROMO",),
         "Promo economics attract multi-accounting", 30, _r10_fire, _r10_explain),
    Rule("R11", "impossible geo-velocity", ("P-ATO", "P-STOLEN"),
         "Two locations faster than travel allows means two actors or proxying",
         25, _r11_fire, _r11_explain),
    Rule("R12", "vendor risk signal", ("P-STOLEN", "P-SYNTH"),
         "Independent external signal on identity infrastructure", 25, _r12_fire, _r12_explain),
]
