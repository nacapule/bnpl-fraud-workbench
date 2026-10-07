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


def test_columns_are_anchored_as_the_policy_words_them() -> None:
    """Order-relative rule windows stay at the order; linkage counted before decision
    time, the current email and outcomes move with the decision (policy §2.4, §2.5)."""
    anchor = {column.name: column.anchor for column in asof.COLUMNS}
    for name in ("hours_since_password_change", "device_link_age_hours", "attempts_user_24h",
                 "attempts_device_24h", "processor_declines_card_24h",
                 "geo_kmh_from_previous_attempt", "is_first_attempt_user", "account_age_days",
                 "ship_to_home", "home_address_age_days", "amount_over_category_p95"):
        assert anchor[name] == asof.ORDER, name
    for name in ("accounts_on_device_30d", "accounts_on_address_30d",
                 "email_root_other_accounts", "email_domain_class", *asof.OUTCOME_COLUMNS):
        assert anchor[name] == asof.DECISION, name
    for column in asof.COLUMNS:
        if column.anchor == asof.ORDER:
            assert "the decision" not in column.window + column.definition, column.name


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


def test_email_history_holds_one_address_at_a_time() -> None:
    accounts = pd.DataFrame({"user_id": [1, 2], "email": ["Ann.B@gmail.com", "cy@outlook.com"],
                             "created_at": pd.to_datetime(["2025-01-01", "2025-01-02"])})
    events = pd.DataFrame({
        "user_id": [2, 2, 1, 1], "kind": ["email_change", "login", "email_change", "email_change"],
        "email": ["annb+x@googlemail.com", None, "ann@proton.me", "a.n.n.b@gmail.com"],
        "known_at": pd.to_datetime(["2025-02-01", "2025-02-02", "2025-03-01", "2025-04-01"]),
        "event_id": [1, 2, 3, 4]})
    T = pd.Timestamp
    assert asof.email_history(accounts, events).values.tolist() == [
        [1, "annb@gmail.com", T("2025-01-01"), T("2025-03-01")],
        [1, "ann@proton.me", T("2025-03-01"), T("2025-04-01")],
        [1, "annb@gmail.com", T("2025-04-01"), pd.NaT],  # changed back: a new holding
        [2, "cy@outlook.com", T("2025-01-02"), T("2025-02-01")],
        [2, "annb@gmail.com", T("2025-02-01"), pd.NaT],
    ]


def test_email_changes_at_one_time_follow_event_ids_not_row_order() -> None:
    accounts = pd.DataFrame({"user_id": [1], "email": ["a@outlook.com"],
                             "created_at": pd.to_datetime(["2025-01-01"])})
    at = pd.Timestamp("2025-02-01")
    events = pd.DataFrame({"user_id": [1, 1], "kind": ["email_change"] * 2,
                           "email": ["private@outlook.com", "shared@gmail.com"],
                           "known_at": [at, at], "event_id": [12, 11]})
    expected = [[1, "a@outlook.com", pd.Timestamp("2025-01-01"), at],
                [1, "shared@gmail.com", at, at],  # held for no time at all
                [1, "private@outlook.com", at, pd.NaT]]
    for order in ([0, 1], [1, 0]):
        shuffled = events.iloc[order].reset_index(drop=True)
        assert asof.email_history(accounts, shuffled).values.tolist() == expected


def test_policy_state_records_pending_holds() -> None:
    state = asof.PolicyState.approve_all({"order_attempts": pd.DataFrame({
        "order_id": [1, 2], "processor_result": ["approved", "declined"],
        "known_at": pd.to_datetime(["2025-01-01", "2025-01-02"])})})
    assert state.approved["order_id"].tolist() == [1]
    assert list(state.held.columns) == ["order_id", "held_at", "released_at", "outcome",
                                        "before_shipment"]
    assert state.held.empty and state.voided.empty and state.blocked.empty
