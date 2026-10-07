"""As-of order context: the one definition of history, linkage, repayment state and exposure.

Rules, ML features, the reviewer, packets and the referee read this context; no
module outside ``core/`` computes history, linkage, repayment state or exposure
(``tests/test_history_in_core.py``). It is built from observable tables only and
never reads labels or latent truth.

Decision point. A context row describes one order attempt at one decision time
``decision_at``: checkout (the attempt's own ``known_at``) for routing, rules
and ML; the review and each check's completion for the reviewer, packets and
the referee, which re-evaluate the policy on the evidence known then (fraud
policy §2.4, §6.6). Each column has an ``anchor``, following the policy's
wording. ``order``-anchored columns describe the attempt as it was placed (the
attempt's own attributes, velocity, tenure, link ages, credential changes and
anything the policy measures "before the order"): they see the events ordered
before the attempt in the world's total order ``(known_at, kind rank,
event_id)``, whatever the decision time. ``decision``-anchored columns see the
events known before ``decision_at``: linkage counted "before decision time"
(§2.5, R02, R08), the current email (R06), and every outcome-derived column
(earlier outcomes that settle an order, repayment, disputes, shipment and
cancellation status, blocks). At checkout the two coincide. Entities are
visible from their creation time. "Current" below means the attempt being
decided; a column that includes it says so.

Two kinds of column (the replay depends on the difference):

* ``attempt``-derived columns describe what customers and fraudsters tried
  (velocity, linkage, tenure, credential changes, attempts). Real systems log
  declined attempts, so these are computed once per world and shared by every
  policy.
* ``outcome``-derived columns describe what happened because orders were let
  through (approvals, installments due and paid, payments, disputes, promotion
  redemptions, blocks). A policy that declined an order must not later see that
  order's repayments or disputes, so these are rebuilt from each policy's own
  state by :func:`outcome_columns`; the world-level frame holds their
  approve-all values. Up to one replay day of staleness in them is allowed and
  stated.

Intentionally different business definitions are separate columns (for example
``attempts_user_24h`` vs ``approved_orders_user_24h``, ``accounts_on_address_30d``
vs ``accounts_on_address_ever``). :data:`COLUMNS` is the specification; the
implementation must match it column for column, and its tests include prefix
invariance (truncating the world at any ``known_at``, including equal
timestamps, leaves earlier rows unchanged) and a label-mutation test.

Email identity has one rule, :func:`normalize_email`: lower-case; remove a
``+tag`` from the local part for every provider; remove dots from the local
part only for Gmail (``gmail.com``, ``googlemail.com``, folded to ``gmail.com``);
never strip digits. An account holds one address at a time
(:func:`email_history`: the signup email from ``created_at``, then each
``email_change``'s new address from its ``known_at`` until the next change).
Two accounts share an email while they hold addresses that normalize alike at
the same time, as two accounts share a device while their links overlap.

Implemented here: the specification, :class:`PolicyState`,
:func:`normalize_email` and :func:`email_history`. The builders raise
``NotImplementedError`` until the context is implemented from the
point-in-time kernels in ``model/features.py``
(``_asof_event_count``, ``_hours_since_event``, ``_rolling_distinct_accounts``,
``_installment_history``), which then move here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import pandas as pd

ATTEMPT = "attempt"
OUTCOME = "outcome"

# Observable inputs the context may read; labels and latent tables are never inputs.
INPUT_TABLES = (
    "accounts", "merchants", "devices", "device_links", "addresses", "address_links", "cards",
    "promotions", "plans", "installment_schedule", "account_events", "order_attempts",
    "payment_attempts", "fulfilments", "deliveries", "payment_reversals", "dispute_openings",
    "dispute_resolutions", "victim_reports", "plan_writeoffs", "cash_events",
)
GMAIL_DOMAINS = {"gmail.com", "googlemail.com"}
DISPOSABLE_DOMAINS = {"guerrillamail.com", "mailinator.com", "tempmailo.com"}
COMMON_DOMAINS = {"gmail.com", "hotmail.com", "icloud.com", "outlook.com", "proton.me",
                  "yahoo.com"}


@dataclass(frozen=True)
class AsofColumn:
    name: str
    kind: str  # ATTEMPT or OUTCOME
    dtype: str
    definition: str
    eligibility: str  # which events count
    window: str  # e.g. "24h before the order", "ever", "at the decision"
    includes_current: bool | None  # None where the question does not arise
    used_by: tuple[str, ...]  # rules, ml, reviewer, packets, referee, replay
    anchor: str = "decision"  # ORDER or DECISION (module docstring)


def _col(name: str, kind: str, dtype: str, definition: str, eligibility: str, window: str,
         current: bool | None, *used_by: str) -> AsofColumn:
    return AsofColumn(name, kind, dtype, definition, eligibility, window, current, used_by)


A, OC = ATTEMPT, OUTCOME
ALL = ("rules", "ml", "reviewer", "packets", "referee")

COLUMNS: tuple[AsofColumn, ...] = (
    # ---- the attempt itself and static context
    _col("amount_cents", A, "int64", "Merchant's price of the current attempt.",
         "current attempt", "at the decision", True, *ALL),
    _col("order_exposure_cents", A, "int64",
         "Cash at risk if approved: merchant settlement plus promotion funding minus the "
         "down payment (core.ledger terms).", "current attempt", "at the decision", True,
         "rules", "ml", "replay"),
    _col("account_age_days", A, "float64", "Days from account creation to the decision.",
         "the account", "at the decision", None, *ALL),
    _col("merchant_age_days", A, "float64", "Days from merchant onboarding to the decision.",
         "the merchant", "at the decision", None, "reviewer", "packets", "referee"),
    _col("merchant_fulfilment_median_hours", A, "float64",
         "The merchant's stated median hours from approval to shipment.", "the merchant",
         "static", None, "reviewer", "packets", "referee", "replay"),
    _col("avs_mismatch", A, "int8", "Address verification failed (avs_result = N).",
         "current attempt", "at the decision", True, *ALL),
    _col("cvv_mismatch", A, "int8", "Card verification failed (cvv_result = N).",
         "current attempt", "at the decision", True, *ALL),
    _col("bin_ip_country_mismatch", A, "int8", "Card BIN country differs from the IP country.",
         "current attempt", "at the decision", True, *ALL),
    _col("ip_country_not_home", A, "int8",
         "IP country differs from the account's home country.", "current attempt",
         "at the decision", True, *ALL),
    _col("night_order", A, "int8", "Checkout between 00:00 and 05:59 platform time.",
         "current attempt", "at the decision", True, "rules", "ml"),
    _col("email_domain_class", A, "int8",
         "Of the account's current email: 0 common provider, 1 other, 2 disposable (lists "
         "in this module).", "the account", "at the decision", None, *ALL),
    _col("email_root_other_accounts", A, "int64",
         "Other accounts holding, at the decision, an address that normalizes like this "
         "account's current one (email_history).", "accounts and their email holdings",
         "at the decision", False, *ALL),
    _col("amount_over_category_median", A, "float64",
         "Amount over the median amount of earlier processor-approved attempts in the "
         "merchant's category.", "processor-approved attempts in the category",
         "ever, before the decision", False, *ALL),
    _col("amount_over_category_p95", A, "float64",
         "Amount over the 95th percentile of the same set.",
         "processor-approved attempts in the category", "ever, before the decision", False,
         *ALL),
    # ---- velocity
    _col("attempts_user_1h", A, "int64", "Order attempts by the account.",
         "attempts of any processor result", "1h before the decision", True, "rules", "ml"),
    _col("attempts_user_24h", A, "int64", "Order attempts by the account.",
         "attempts of any processor result", "24h before the decision", True, *ALL),
    _col("attempts_user_7d", A, "int64", "Order attempts by the account.",
         "attempts of any processor result", "7 days before the decision", True, "rules",
         "ml"),
    _col("amount_attempted_user_24h", A, "int64", "Sum of attempted amounts by the account.",
         "attempts of any processor result", "24h before the decision", True, "rules", "ml"),
    _col("amount_attempted_user_7d", A, "int64", "Sum of attempted amounts by the account.",
         "attempts of any processor result", "7 days before the decision", True, "ml"),
    _col("attempts_device_24h", A, "int64", "Order attempts on the device, any account.",
         "attempts of any processor result", "24h before the decision", True, *ALL),
    _col("processor_declines_card_24h", A, "int64",
         "Processor-declined attempts with the same card.", "processor declines",
         "24h before the decision", False, *ALL),
    _col("processor_declines_device_24h", A, "int64",
         "Processor-declined attempts on the same device, any account.", "processor declines",
         "24h before the decision", False, *ALL),
    _col("is_first_attempt_user", A, "int8", "No earlier order attempt by the account.",
         "attempts of any processor result", "ever", False, *ALL),
    _col("geo_kmh_from_previous_attempt", A, "float64",
         "Implied km/h between the IP countries of the account's previous attempt and this "
         "one (0 when the same country or more than 12h apart).",
         "the account's previous attempt", "12h before the decision", False, *ALL),
    # ---- devices, addresses and cards on the account
    _col("device_link_age_hours", A, "float64",
         "Hours since the device was first used on this account.", "device_links",
         "at the decision", None, *ALL),
    _col("distinct_devices_user_30d", A, "int64",
         "Distinct devices in the account's attempts and account events.",
         "attempts and account events", "30 days before the decision", True, "ml", "packets"),
    _col("card_link_age_hours", A, "float64", "Hours since the card was added to the account.",
         "cards", "at the decision", None, *ALL),
    _col("card_first_use_age_hours", A, "float64",
         "Hours since the account's first attempt with this card (0 for the first use).",
         "the account's attempts", "ever", True, "reviewer", "packets", "referee"),
    _col("ship_address_link_age_hours", A, "float64",
         "Hours since the account linked the shipping address.", "address_links",
         "at the decision", None, *ALL),
    _col("ship_address_first_use_age_hours", A, "float64",
         "Hours since the account's first attempt shipping to this address (0 for the first).",
         "the account's attempts", "ever", True, "reviewer", "packets", "referee"),
    _col("ship_to_home", A, "int8",
         "The shipping address is the account's active home address.", "address_links",
         "at the decision", None, *ALL),
    _col("home_address_age_days", A, "float64",
         "Days since the account's current home address was registered on it.",
         "address_links (role home, active)", "at the decision", None, "reviewer", "packets",
         "referee"),
    _col("distinct_ship_addresses_user_ever", A, "int64",
         "Distinct shipping addresses in the account's attempts.", "the account's attempts",
         "ever", True, "ml"),
    # ---- linkage
    _col("accounts_on_device_30d", A, "int64",
         "Distinct accounts with an attempt or account event on the device.",
         "attempts and account events", "30 days before the decision", True, *ALL),
    _col("accounts_on_address_30d", A, "int64",
         "Distinct accounts with an attempt shipping to the address.", "attempts",
         "30 days before the decision", True, *ALL),
    _col("accounts_on_address_ever", A, "int64",
         "Distinct accounts with an attempt shipping to, or a link to, the address.",
         "attempts and address_links", "ever", True, "ml", "packets"),
    # ---- credential changes
    _col("hours_since_password_change", A, "float64",
         "Hours since the last password change (10,000 when none).",
         "account_events kind password_change", "ever", None, *ALL),
    _col("hours_since_password_reset", A, "float64",
         "Hours since the last password reset (10,000 when none).",
         "account_events kind password_reset", "ever", None, *ALL),
    _col("hours_since_email_change", A, "float64",
         "Hours since the last email change (10,000 when none).",
         "account_events kind email_change", "ever", None, *ALL),
    _col("hours_since_phone_change", A, "float64",
         "Hours since the last phone change (10,000 when none).",
         "account_events kind phone_change", "ever", None, *ALL),
    _col("hours_since_credential_change", A, "float64",
         "Minimum of the password change, password reset and email change columns.",
         "account_events", "ever", None, "rules", "ml"),
    # ---- outcomes under the policy (rebuilt per policy)
    _col("approved_orders_user_24h", OC, "int64",
         "Orders by the account the policy let through and has not voided.",
         "policy-approved orders", "24h before the decision", False, "rules", "ml"),
    _col("approved_orders_user_ever", OC, "int64", "As above, without a window.",
         "policy-approved orders", "ever", False, *ALL),
    _col("installments_due_user", OC, "int64",
         "Installments (seq >= 1) due before the decision on the account's live plans.",
         "plans of policy-approved orders", "ever, due before the decision", False, *ALL),
    _col("installments_paid_user", OC, "int64",
         "Of those, paid in full: successful attempts on the installment known before the "
         "decision, less reversals of them known by then, reach its scheduled amount.",
         "plans of policy-approved orders", "ever", False, *ALL),
    _col("installments_failed_user", OC, "int64",
         "Of those, with a failed attempt known before the decision and not paid in full.",
         "plans of policy-approved orders", "ever", False, *ALL),
    _col("installments_paid_share_user", OC, "float64",
         "installments_paid_user / installments_due_user (0 when none due).",
         "plans of policy-approved orders", "ever", False, *ALL),
    _col("open_balance_user_cents", OC, "int64",
         "Exposure: principal of the account's live plans minus standing payments, excluding "
         "written-off plans.", "plans of policy-approved orders", "at the decision", False,
         *ALL),
    _col("unauthorized_disputes_on_card", OC, "int64",
         "Unauthorized-use disputes known before the decision on orders paid with this card "
         "(a learned feature; the fraud policy does not use it).",
         "disputes on policy-approved orders", "ever", False, "ml"),
    _col("unauthorized_disputes_lost_user", OC, "int64",
         "Unauthorized-use disputes resolved lost on the account's earlier orders, the "
         "resolution known before the decision.", "disputes on policy-approved orders",
         "ever", False, *ALL),
    _col("victim_reports_user", OC, "int64",
         "Reports by the account holder of orders they did not place, known before the "
         "decision.", "victim reports on policy-approved orders", "ever", False, *ALL),
    _col("inr_disputes_opened_user", OC, "int64",
         "Item-not-received disputes opened by the account, known before the decision.",
         "disputes on policy-approved orders", "ever", False, *ALL),
    _col("inr_claims_rejected_user", OC, "int64",
         "Item-not-received disputes resolved against the account (won) on orders with a "
         "carrier-confirmed delivery, the resolution and the delivery both known before the "
         "decision (the fraud policy's item-not-received abuse counts these).",
         "disputes on policy-approved orders", "ever", False, *ALL),
    _col("never_pay_determined_user", OC, "int8",
         "A never-pay determination on one of the account's earlier plans was known before "
         "the decision; computed from payments with core.world.adjudicate's rule, never by "
         "reading labels.", "plans of policy-approved orders", "ever", False, *ALL),
    _col("promo_redemptions_user", OC, "int64", "Promotions used on the account's orders.",
         "policy-approved orders", "ever, before the decision", False, "rules", "ml"),
    _col("promo_uses_linked_accounts", OC, "int64",
         "Accounts that shared a device (overlapping links) or a normalized email "
         "(overlapping holdings) with this one, not an address, and used the same "
         "first-purchase promotion, including this account's use.", "policy-approved orders",
         "ever, before the decision", True, *ALL),
    _col("shipped_at_decision", OC, "int8",
         "The order's fulfilment is known at the decision (always 0 at checkout).",
         "the current order under the policy", "at the decision", True, "reviewer", "packets",
         "referee", "replay"),
    _col("cancelled_at_decision", OC, "int8",
         "The policy voided or cancelled the order before the decision.",
         "the current order under the policy", "at the decision", True, "reviewer", "packets",
         "referee", "replay"),
    _col("account_blocked", OC, "int8", "The policy blocked the account before the decision.",
         "policy blocks", "at the decision", None, "replay", "reviewer"),
    _col("linked_account_blocked_30d", OC, "int8",
         "An account linked by device or address in the last 30 days was blocked by the "
         "policy before the decision.", "policy blocks", "30 days before the decision", None,
         "replay", "reviewer"),
)
ORDER, DECISION = "order", "decision"
_DECISION_ANCHORED_ATTEMPT_COLUMNS = frozenset({
    "email_domain_class", "email_root_other_accounts",  # the current email (R06)
    "accounts_on_device_30d", "accounts_on_address_30d", "accounts_on_address_ever",  # §2.5
})


def _anchored(column: AsofColumn) -> AsofColumn:
    if column.kind == OUTCOME or column.name in _DECISION_ANCHORED_ATTEMPT_COLUMNS:
        return column
    return replace(column, anchor=ORDER,
                   definition=column.definition.replace("the decision", "the order"),
                   window=column.window.replace("the decision", "the order"))


COLUMNS = tuple(_anchored(column) for column in COLUMNS)
COLUMN_NAMES = tuple(column.name for column in COLUMNS)
ATTEMPT_COLUMNS = tuple(c.name for c in COLUMNS if c.kind == ATTEMPT)
OUTCOME_COLUMNS = tuple(c.name for c in COLUMNS if c.kind == OUTCOME)
KEY_COLUMNS = ("order_id", "user_id", "merchant_id", "decision_at")


def normalize_email(email: str) -> str:
    """The one email-identity rule (see the module docstring)."""
    local, _, domain = email.strip().lower().rpartition("@")
    if not local:
        raise ValueError(f"not an email address: {email!r}")
    local = local.split("+", 1)[0]
    if domain in GMAIL_DOMAINS:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"


def email_history(accounts: pd.DataFrame, account_events: pd.DataFrame) -> pd.DataFrame:
    """Each account's addresses as holding intervals: user_id, email_root, since, until.

    The signup email is held from the account's ``created_at``, each
    ``email_change``'s new address from its ``known_at``; a holding ends when the
    next change is known (``until`` is NaT for the current address); changes known
    at the same time follow their event ids. Changing back to an earlier address
    starts a new interval.
    """
    changes = account_events.loc[account_events["kind"] == "email_change",
                                 ["user_id", "email", "known_at", "event_id"]]
    signups = accounts[["user_id", "email", "created_at"]].assign(event_id=-1)
    # the world's event order: by known_at, then event id; the signup comes first
    history = pd.concat([
        signups.rename(columns={"created_at": "since"}),
        changes.rename(columns={"known_at": "since"}),
    ], ignore_index=True).sort_values(["user_id", "since", "event_id"], kind="stable")
    history["until"] = history.groupby("user_id")["since"].shift(-1)
    history["email_root"] = [normalize_email(email) for email in history["email"]]
    return history[["user_id", "email_root", "since", "until"]].reset_index(drop=True)


def _frame(**columns: str) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


@dataclass(frozen=True)
class PolicyState:
    """What one policy has done that its event tables do not show.

    Outcome-derived columns are computed from the policy's realized tables (the
    replay's output, in the world schema: an auto-declined order keeps its
    attempt and nothing after it; a held order's shipment and schedule move to
    after its release; a voided or cancelled order loses the events after the
    void and gains its refunds) together with this state:

    * ``approved``: order_id, approved_at, when the order went through (checkout,
      or the release of a hold placed before shipment);
    * ``held``: order_id, held_at, released_at (NaT while the hold is pending),
      outcome (``cleared``, ``cancelled`` or ``declined``; null while pending),
      before_shipment. A hold placed before shipment pauses the order: until its
      release the order is neither approved nor voided, and the realized tables
      move its shipment and later schedule after the release (or cancel it and
      refund the checkout payment). A hold placed after shipment pauses nothing:
      the order stays approved from checkout and its realized events stand; only a
      decline after a failed check applies;
    * ``voided``: order_id, at (decline, void, cancellation or abandonment before
      shipping);
    * ``blocked``: user_id, at.

    Under approve-all the realized tables are the world's own, every
    processor-approved order is approved at its checkout and nothing is held,
    voided or blocked.
    """

    approved: pd.DataFrame
    voided: pd.DataFrame
    blocked: pd.DataFrame
    held: pd.DataFrame = field(default_factory=lambda: _frame(
        order_id="int64", held_at="datetime64[s]", released_at="datetime64[s]",
        outcome="object", before_shipment="bool"))

    @classmethod
    def approve_all(cls, tables: Mapping[str, pd.DataFrame]) -> PolicyState:
        orders = tables["order_attempts"]
        approved = orders.loc[orders["processor_result"] == "approved",
                              ["order_id", "known_at"]].rename(columns={"known_at": "approved_at"})
        return cls(
            approved=approved.reset_index(drop=True),
            voided=_frame(order_id="int64", at="datetime64[s]"),
            blocked=_frame(user_id="int64", at="datetime64[s]"),
        )


def build_context(
    tables: Mapping[str, pd.DataFrame],
    decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The world-level context: one row per decision with KEY_COLUMNS + COLUMN_NAMES.

    ``decisions`` has ``order_id`` and ``decision_at``; by default every order
    attempt at checkout. Outcome-derived columns take their approve-all values.
    Reads only :data:`INPUT_TABLES`.
    """
    raise NotImplementedError("the as-of context is built in core/asof.py from the kernels")


def outcome_columns(
    tables: Mapping[str, pd.DataFrame],
    state: PolicyState,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Outcome-derived columns (KEY_COLUMNS + OUTCOME_COLUMNS) under one policy.

    ``tables`` are that policy's realized observations and ``state`` what it did
    (:class:`PolicyState`); under approve-all, the world's tables and
    ``PolicyState.approve_all``.
    """
    raise NotImplementedError("the as-of context is built in core/asof.py from the kernels")
