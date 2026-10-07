"""Small worlds and stand-in context for replay tests.

The replay reads context rows (core.asof) and scores (fitted detectors). These tests
check the replay's own mechanics, so the context here is a stand-in: every row is a
quiet order (no FP-2 condition holds) unless a test sets columns for an order, and a
test scorer reads a ``test_score`` column the test sets. Nothing here reads labels or
latent truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import asof, ledger, world
from queue_sim import policies
from queue_sim.replay import FrozenHistory, Settings, World
from queue_sim.reviewer import Verification
from queue_sim.roster import Roster, ServiceCalendar, Shift

MINI = Path(__file__).resolve().parent / "fixtures" / "mini_world"

QUIET: dict[str, Any] = {
    "hours_since_password_change": 10_000.0, "hours_since_password_reset": 10_000.0,
    "hours_since_email_change": 10_000.0, "hours_since_phone_change": 10_000.0,
    "hours_since_credential_change": 10_000.0, "account_age_days": 700.0,
    "device_link_age_hours": 9_000.0, "geo_kmh_from_previous_attempt": 0.0,
    "bin_ip_country_mismatch": 0, "avs_mismatch": 0, "cvv_mismatch": 0,
    "processor_declines_card_24h": 0, "processor_declines_device_24h": 0,
    "attempts_user_24h": 1, "attempts_device_24h": 1, "accounts_on_device_30d": 1,
    "accounts_on_address_30d": 1, "accounts_on_address_ever": 1,
    "email_root_other_accounts": 0, "promo_uses_linked_accounts": 0,
    "email_domain_class": 0, "is_first_attempt_user": 0, "amount_over_category_p95": 0.4,
    "inr_disputes_opened_user": 0, "ship_to_home": 1, "home_address_age_days": 600.0,
    "ship_address_link_age_hours": 14_000.0, "ip_country_not_home": 0,
    "unauthorized_disputes_lost_user": 0, "victim_reports_user": 0,
    "never_pay_determined_user": 0, "inr_claims_rejected_user": 0,
    "installments_due_user": 0,
}
CARD = {"bin_ip_country_mismatch": 1, "avs_mismatch": 1}  # R03: one Card condition
TAKEOVER = {"hours_since_password_reset": 3.0, "device_link_age_hours": 3.0}  # R01


def mini_tables() -> dict[str, pd.DataFrame]:
    return world.read_world(MINI)


@dataclass
class StubContext:
    """Context rows at any decision time: quiet, plus per-order overrides."""

    tables: Mapping[str, pd.DataFrame]
    overrides: dict[int, dict[str, Any]] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)
    calls: list[pd.DataFrame] = field(default_factory=list)

    def build(self, tables: Mapping[str, pd.DataFrame],
              decisions: pd.DataFrame | None = None) -> pd.DataFrame:
        attempts = self.tables["order_attempts"]
        if decisions is None:
            decisions = attempts[["order_id", "known_at"]].rename(
                columns={"known_at": "decision_at"})
        self.calls.append(decisions)
        merchants = self.tables["merchants"].set_index("merchant_id")
        info = attempts.set_index("order_id")
        terms = ledger.ProductTerms.from_config()
        rows = []
        for order, at in zip(decisions["order_id"], decisions["decision_at"], strict=True):
            a = info.loc[order]
            amount = int(a["amount_cents"])
            down = ledger.split_principal(amount - int(a["promo_discount_cents"]), terms)[0]
            row = {name: 0 for name in asof.COLUMN_NAMES}
            row.update(QUIET)
            row.update({
                "order_id": int(order), "user_id": int(a["user_id"]),
                "merchant_id": int(a["merchant_id"]), "decision_at": pd.Timestamp(at),
                "amount_cents": amount,
                "order_exposure_cents": ledger.settlement_cents(
                    amount, int(a["promo_discount_cents"]), terms)
                + int(a["promo_discount_cents"]) - down,
                "merchant_fulfilment_median_hours": float(
                    merchants.loc[a["merchant_id"], "fulfilment_median_hours"]),
                "test_score": self.scores.get(int(order), 0.0),
            })
            row.update(self.overrides.get(int(order), {}))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame["decision_at"] = frame["decision_at"].astype("datetime64[s]")
        return frame


@dataclass
class StubScorer:
    """Reads the stand-in ``test_score`` column; probability = the score clipped to [0, 1]."""

    name: str = "test"
    version: str = "t1"
    columns: tuple[str, ...] = ("test_score",)

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["test_score"].to_numpy(dtype=float)

    def probability(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.score(rows), 0.0, 1.0)


def scored_policy(review: float | None = 1.0, decline: float | None = None) -> policies.Policy:
    return policies.single_scorer("scored", StubScorer()).with_thresholds(review, decline)


def stub_world(tables: Mapping[str, pd.DataFrame], stub: StubContext, *,
               observed_until: str = "2025-06-30 23:59:59", seed: int = 416) -> World:
    return World(tables=tables, context=stub.build(tables), seed=seed,
                 observed_until=pd.Timestamp(observed_until),
                 terms=ledger.ProductTerms.from_config())


def always_on_roster(analysts: int = 1) -> Roster:
    """Analysts on shift all day, every day (one shift of 24 hours)."""
    return Roster((Shift("all", tuple(range(7)), 0, 24 * 60),), {"all": analysts})


def verification(rates: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None, *,
                 classes: Mapping[int, str] | None = None, seed: int = 416,
                 median_hours: float = 2.0, sigma: float = 0.0) -> Verification:
    """Check outcomes at chosen rates: by default every check passes after about 2 hours."""
    certain = {"contact": {"passed": 1.0, "failed": 0.0},
               "id_check": {"passed": 1.0, "failed": 0.0}}
    rates = rates or {}
    full = {actor: dict(rates.get(actor, certain))
            for actor in ("legitimate", "first_party", "takeover", "third_party")}
    return Verification(seed=seed, actor_class=dict(classes or {}), rates=full,
                        response_median_hours=median_hours, response_sigma=sigma)


SETTINGS = Settings(hold_max_hours=48.0, service_mean_minutes=10.0, service_sigma=0.0,
                    senior_minutes=20.0)
CALENDAR = ServiceCalendar(tuple(range(7)), 8 * 60, 20 * 60)


def frozen(world_: World, stub: StubContext) -> FrozenHistory:
    return FrozenHistory(world_, build=stub.build)
