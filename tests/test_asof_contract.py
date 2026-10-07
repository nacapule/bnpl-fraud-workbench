"""The as-of context specification and the one email-identity rule."""

from __future__ import annotations

import pandas as pd
import pytest

from core import asof


def test_column_names_are_unique_and_kinds_known() -> None:
    assert len(set(asof.COLUMN_NAMES)) == len(asof.COLUMN_NAMES)
    assert {c.kind for c in asof.COLUMNS} == {asof.ATTEMPT, asof.OUTCOME}
    assert not set(asof.COLUMN_NAMES) & set(asof.KEY_COLUMNS)


def test_different_business_definitions_are_separate_columns() -> None:
    kinds = {c.name: c.kind for c in asof.COLUMNS}
    assert kinds["attempts_user_24h"] == asof.ATTEMPT
    assert kinds["approved_orders_user_24h"] == asof.OUTCOME
    assert {"accounts_on_address_30d", "accounts_on_address_ever"} <= set(kinds)


def test_repayment_dispute_and_block_history_is_outcome_derived() -> None:
    """A policy that prevented an order must not later see that order's outcomes."""
    for name in ("installments_due_user", "installments_paid_user", "open_balance_user_cents",
                 "inr_claims_rejected_user", "promo_redemptions_user", "account_blocked",
                 "unauthorized_disputes_lost_user", "victim_reports_user"):
        assert name in asof.OUTCOME_COLUMNS


def test_context_never_reads_labels_or_latent_truth() -> None:
    from core.world import LATENT_TABLES

    assert "labels" not in asof.INPUT_TABLES
    assert not set(LATENT_TABLES) & set(asof.INPUT_TABLES)


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ("alice+1@gmail.com", "alice@gmail.com", True),  # plus tags removed for every provider
        ("bob+shop@outlook.com", "bob@outlook.com", True),
        ("A.Lice@GMail.com", "alice@gmail.com", True),  # dots removed only for Gmail
        ("a.lice@googlemail.com", "alice@gmail.com", True),
        ("a.lice@yahoo.com", "alice@yahoo.com", False),
        ("alice1@gmail.com", "alice@gmail.com", False),  # digits are never stripped
        ("alice12@gmail.com", "alice1@gmail.com", False),
    ],
)
def test_email_normalization(left: str, right: str, same: bool) -> None:
    assert (asof.normalize_email(left) == asof.normalize_email(right)) is same


def test_email_normalization_rejects_non_addresses() -> None:
    with pytest.raises(ValueError):
        asof.normalize_email("not-an-address")


def test_email_history_keeps_every_address_from_when_it_was_known() -> None:
    accounts = pd.DataFrame({"user_id": [1, 2], "email": ["Ann.B@gmail.com", "cy@outlook.com"],
                             "created_at": pd.to_datetime(["2025-01-01", "2025-01-02"])})
    events = pd.DataFrame({
        "user_id": [2, 2, 1], "kind": ["email_change", "login", "email_change"],
        "email": ["annb+x@googlemail.com", None, "ann@proton.me"],
        "known_at": pd.to_datetime(["2025-02-01", "2025-02-02", "2025-03-01"])})
    history = asof.email_history(accounts, events).sort_values(["user_id", "since"])
    assert history.values.tolist() == [
        [1, "annb@gmail.com", pd.Timestamp("2025-01-01")],
        [1, "ann@proton.me", pd.Timestamp("2025-03-01")],
        [2, "cy@outlook.com", pd.Timestamp("2025-01-02")],
        [2, "annb@gmail.com", pd.Timestamp("2025-02-01")],
    ]


def test_policy_state_records_pending_holds() -> None:
    state = asof.PolicyState.approve_all({"order_attempts": pd.DataFrame({
        "order_id": [1, 2], "processor_result": ["approved", "declined"],
        "known_at": pd.to_datetime(["2025-01-01", "2025-01-02"])})})
    assert state.approved["order_id"].tolist() == [1]
    assert list(state.held.columns) == ["order_id", "held_at", "released_at", "outcome"]
    assert state.held.empty and state.voided.empty and state.blocked.empty
