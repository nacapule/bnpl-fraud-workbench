"""As-of order context: the one definition of history, linkage, repayment state and exposure.

Rules, ML features, the reviewer, packets and the referee read this context; no
module outside ``core/`` computes history, linkage, repayment state or exposure
(``tests/test_history_in_core.py``). It is built from observable tables only and
never reads labels or latent truth.

Decision point. A context row describes one order attempt at one decision time
``decision_at``: checkout (the attempt's own ``known_at``) for routing, rules
and ML; the review and each check's completion for the reviewer, packets and
the referee, which re-evaluate the policy on the evidence known then (fraud
policy §2.4, §6.6). What a row can see:

* A checkout decision sees the events ordered before the attempt in the world's
  total order ``(known_at, kind rank, event_id)``, so events in the same second
  follow the tie order. A later decision at ``T`` sees every event with
  ``known_at <= T`` (to the second). A ``decision_at`` equal to the order's own
  checkout time is the checkout decision; an earlier one raises.
* Entities (accounts, links, cards, ...) are visible from their creation time,
  also within the same second.
* Times that are not world events (due dates, default dates, determinations,
  and the policy's own approvals at a hold's release, voids, holds and blocks)
  count as known at the end of their second: visible to a later decision at
  ``T`` when they are at or before ``T``, and to a checkout only when strictly
  earlier.

Each column has an ``anchor``, following the policy's wording. ``order``-anchored
columns describe the attempt as it was placed (the attempt's own attributes,
velocity, tenure, link ages, credential changes and anything the policy
measures "before the order"): they are computed at the checkout point whatever
the decision time. ``decision``-anchored columns see what is known at the
decision: linkage counted "before decision time" (§2.5, R02, R08), the current
email (R06) and every outcome-derived column (earlier outcomes that settle an
order, repayment, disputes, shipment and cancellation status, blocks). At
checkout the two coincide. A window "24h before" includes events known at
exactly 24 hours before. "Current" below means the attempt being decided; a
column that includes it says so.

Two kinds of column (the replay depends on the difference):

* ``attempt``-derived columns describe what customers and fraudsters tried
  (velocity, linkage, tenure, credential changes, attempts). Real systems log
  declined attempts, so these are computed once per world and shared by every
  policy (:func:`attempt_columns`).
* ``outcome``-derived columns describe what happened because orders were let
  through (approvals, installments due and paid, payments, disputes, promotion
  redemptions, blocks). A policy that declined an order must not later see that
  order's repayments or disputes, so these are rebuilt from each policy's own
  realized tables and :class:`PolicyState` by :func:`outcome_columns`; the
  world-level frame (:func:`build_context`) holds their approve-all values. Up to
  one replay day of staleness in them is allowed and stated.
  :func:`policy_rows` joins the two for the replay.

Intentionally different business definitions are separate columns (for example
``attempts_user_24h`` vs ``approved_orders_user_24h``, ``accounts_on_address_30d``
vs ``accounts_on_address_ever``). :data:`COLUMNS` is the specification; the
implementation matches it column for column, and its tests include prefix
invariance (truncating the world at any ``known_at``, including equal
timestamps and knowledge that arrives after the event, leaves earlier rows
unchanged), an independent row-by-row oracle and a label-mutation test.

Email identity has one rule, :func:`normalize_email`: lower-case; remove a
``+tag`` from the local part for every provider; remove dots from the local
part only for Gmail (``gmail.com``, ``googlemail.com``, folded to ``gmail.com``);
never strip digits. An account holds one address at a time
(:func:`email_history`: the signup email from ``created_at``, then each
``email_change``'s new address from its ``known_at`` until the next change).
Two accounts share an email while they hold addresses that normalize alike at
the same time, as two accounts share a device while their links overlap.
:func:`linked_accounts` lists the §2.5 linked accounts behind R02 and R08.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import cache

import numpy as np
import pandas as pd

from core import config, ledger, world

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
         "merchant's category (missing while the category has fewer than "
         "CATEGORY_MIN_SAMPLE of them).", "processor-approved attempts in the category",
         "ever, before the decision", False, *ALL),
    _col("amount_over_category_p95", A, "float64",
         "Amount over the 95th percentile (linear interpolation) of the same set, with the "
         "same minimum.", "processor-approved attempts in the category",
         "ever, before the decision", False, *ALL),
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
         "Implied km/h between the IP countries' centroids of the account's previous attempt "
         "and this one, over at least 0.02h (0 when the same country, no earlier attempt or "
         "12h or more apart).",
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
         "Hours since the last password change, capped at 10,000 (also when none).",
         "account_events kind password_change", "ever", None, *ALL),
    _col("hours_since_password_reset", A, "float64",
         "Hours since the last password reset, capped at 10,000 (also when none).",
         "account_events kind password_reset", "ever", None, *ALL),
    _col("hours_since_email_change", A, "float64",
         "Hours since the last email change, capped at 10,000 (also when none).",
         "account_events kind email_change", "ever", None, *ALL),
    _col("hours_since_phone_change", A, "float64",
         "Hours since the last phone change, capped at 10,000 (also when none).",
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
         "reading labels. A voided or cancelled plan defaults only before it stops; another "
         "determination on the same order does not hide this one.",
         "plans of policy-approved orders", "ever", False, *ALL),
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


@cache
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


# ======================================================================== implementation
# Every row in the computation carries an int64 order key: whole seconds since
# the epoch shifted left by SUB_BITS, plus a position inside the second. World
# events take their position in the total order (1, 2, ... within the second),
# entities position 0 (visible to everything in their second) and derived times
# the last position (visible only from the next second, or to a later decision
# at that second). A decision sees the rows whose key is below its cut.
SUB_BITS = 24
_SUB_END = (1 << SUB_BITS) - 1
_NEVER = np.iinfo(np.int64).max
_HOUR = 3_600
_DAY = 86_400
NO_EVENT_HOURS = 10_000.0  # hours-since columns when there is no such event
CATEGORY_MIN_SAMPLE = 100  # earlier approved amounts a category needs for its reference
GEO_WINDOW_HOURS = 12
GEO_MIN_GAP_HOURS = 0.02
# Country centroids (latitude, longitude) for the geo-velocity column.
CENTROIDS = {
    "US": (39.8, -98.6), "CA": (56.1, -106.3), "MX": (23.6, -102.6), "GB": (54.0, -2.9),
    "IE": (53.4, -8.2), "DE": (51.2, 10.4), "FR": (46.6, 2.4), "ES": (40.3, -3.7),
    "PT": (39.6, -8.0), "IT": (42.8, 12.6), "NL": (52.2, 5.6), "PL": (52.1, 19.4),
    "RO": (45.9, 24.9), "UA": (49.0, 31.4), "RU": (61.5, 105.3), "TR": (39.1, 35.2),
    "BR": (-10.8, -52.9), "AR": (-35.4, -65.2), "CO": (4.1, -72.9), "IN": (22.9, 79.6),
    "PK": (29.9, 69.4), "NG": (9.6, 8.1), "ZA": (-29.0, 25.1), "EG": (26.6, 29.8),
    "VN": (16.6, 106.3), "CN": (36.5, 103.8), "JP": (36.6, 138.0), "KR": (36.5, 127.8),
    "PH": (12.9, 122.9), "ID": (-2.2, 117.3), "AU": (-25.7, 134.5),
}
# Event tables whose rows take part in the total order of this module.
_KEYED = ("account_events", "order_attempts", "payment_attempts", "fulfilments", "deliveries",
          "payment_reversals", "dispute_openings", "dispute_resolutions", "victim_reports",
          "plan_writeoffs")
_DTYPE = {column.name: column.dtype for column in COLUMNS}


def _seconds(values: pd.Series | np.ndarray) -> np.ndarray:
    """Whole seconds since the epoch (int64); NaT becomes the int64 minimum."""
    array = np.asarray(values)
    if array.dtype.kind != "M":
        array = pd.to_datetime(pd.Series(values)).to_numpy()
    return array.astype("datetime64[s]").astype(np.int64)


def _entity_key(seconds: np.ndarray) -> np.ndarray:
    return seconds << SUB_BITS


def _derived_key(seconds: np.ndarray) -> np.ndarray:
    return np.where(seconds == np.iinfo(np.int64).min, _NEVER, (seconds << SUB_BITS) | _SUB_END)


def _event_keys(tables: Mapping[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Each event row's key from the total order ``(known_at, kind rank, event_id)``."""
    names = [name for name in _KEYED if name in tables]
    seconds = [_seconds(tables[name]["known_at"]) for name in names]
    if not names or not sum(len(s) for s in seconds):
        return {name: np.empty(0, np.int64) for name in names}
    sec = np.concatenate(seconds)
    rank = np.concatenate([np.full(len(s), world.EVENT_RANK[n], np.int64)
                           for n, s in zip(names, seconds, strict=True)])
    event_id = np.concatenate([tables[n]["event_id"].to_numpy(np.int64) for n in names])
    order = np.lexsort((event_id, rank, sec))
    ordered = sec[order]
    position = np.arange(len(ordered)) - np.searchsorted(ordered, ordered, side="left") + 1
    if position.max() >= _SUB_END:
        raise ValueError("too many events in one second for the order key")
    keys = np.empty(len(sec), np.int64)
    keys[order] = (ordered << SUB_BITS) | position
    bounds = np.cumsum([0] + [len(s) for s in seconds])
    return {name: keys[bounds[i]:bounds[i + 1]] for i, name in enumerate(names)}


class _Index:
    """Rows grouped by an integer id and ordered by key within each group.

    Queries name a group and key bounds and are answered for all queries at
    once with sorted searches.
    """

    def __init__(self, group: np.ndarray, key: np.ndarray) -> None:
        group = np.asarray(group, np.int64)
        key = np.asarray(key, np.int64)
        self.groups = np.unique(group)
        self.keys = np.unique(key)
        self.width = len(self.keys) + 1
        composite = np.searchsorted(self.groups, group) * self.width + np.searchsorted(
            self.keys, key)
        self.order = np.argsort(composite, kind="stable")
        self.composite = composite[self.order]
        self.sorted_group = group[self.order]
        self.sorted_key = key[self.order]

    def _code(self, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        group = np.asarray(group, np.int64)
        if not len(self.groups):
            return np.zeros(len(group), bool), np.zeros(len(group), np.int64)
        code = np.searchsorted(self.groups, group)
        present = (code < len(self.groups)) & (
            self.groups[np.minimum(code, len(self.groups) - 1)] == group)
        return present, code

    def position(self, group: np.ndarray, bound: np.ndarray | int | None) -> np.ndarray:
        """Sorted position of the first row of the group with key >= bound (None: the
        group's first row); for a group without rows, a position with nothing before it
        in that group (only differences of positions are meaningful)."""
        present, code = self._code(group)
        if bound is None:
            rank = np.zeros(len(code), np.int64)
        else:
            rank = np.searchsorted(self.keys, np.broadcast_to(np.asarray(bound, np.int64),
                                                              code.shape), side="left")
        position = np.searchsorted(self.composite, code * self.width + rank, side="left")
        return np.where(present, position, 0)

    def count(self, group: np.ndarray, low: np.ndarray | int | None,
              high: np.ndarray | int) -> np.ndarray:
        """Rows of the group with low <= key < high."""
        return np.maximum(self.position(group, high) - self.position(group, low), 0)

    def total(self, values: np.ndarray, group: np.ndarray, low: np.ndarray | int | None,
              high: np.ndarray | int) -> np.ndarray:
        """Sum of ``values`` (aligned with the rows given to the constructor) over the
        same rows as :meth:`count`."""
        running = np.concatenate([[0], np.cumsum(np.asarray(values)[self.order])])
        low_position = self.position(group, low)
        high_position = np.maximum(self.position(group, high), low_position)
        return running[high_position] - running[low_position]

    def last_below(self, group: np.ndarray, high: np.ndarray | int) -> np.ndarray:
        """Constructor row of the group's last row with key < high, or -1."""
        present, _ = self._code(group)
        if not len(self.order):
            return np.full(len(present), -1, np.int64)
        position = self.position(group, high) - 1
        found = present & (position >= 0)
        found[found] &= self.sorted_group[position[found]] == np.asarray(group)[found]
        return _pick(self.order, np.where(found, position, -1), -1)


def _members(group: np.ndarray, member: np.ndarray, key: np.ndarray, query_group: np.ndarray,
             high: np.ndarray, low: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Distinct members with a row in each query's group and low <= key < high.

    Returns (query position, member) pairs, one per member and query.
    """
    group = np.asarray(group, np.int64)
    member = np.asarray(member, np.int64)
    if not len(group):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    pairs, pair_code = np.unique(np.stack([group, member], axis=1), axis=0,
                                 return_inverse=True)
    index = _Index(pair_code.ravel(), key)
    query_group = np.asarray(query_group, np.int64)
    starts = np.searchsorted(pairs[:, 0], query_group, side="left")
    sizes = np.searchsorted(pairs[:, 0], query_group, side="right") - starts
    query = np.repeat(np.arange(len(query_group)), sizes)
    offset = np.arange(len(query)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    pair = starts[query] + offset
    last = index.last_below(pair, np.asarray(high, np.int64)[query])
    seen = last >= 0
    if low is not None:
        seen[seen] &= np.asarray(key, np.int64)[last[seen]] >= np.asarray(low, np.int64)[
            query[seen]]
    return query[seen], pairs[pair[seen], 1]


def _first_by_pair(left: np.ndarray, right: np.ndarray, value: np.ndarray) -> pd.Series:
    """Minimum ``value`` per (left, right) pair."""
    frame = pd.DataFrame({"left": left, "right": right, "value": value})
    return frame.groupby(["left", "right"])["value"].min()


def _pick(values: np.ndarray, index: np.ndarray, default: object = 0) -> np.ndarray:
    """``values[index]`` where ``index >= 0`` and ``default`` elsewhere (also when
    ``values`` is empty)."""
    index = np.asarray(index)
    if not len(values):
        return np.full(len(index), default)
    return np.where(index >= 0, values[np.maximum(index, 0)], default)


def _take(series: pd.Series, labels: pd.Index | np.ndarray, default: int) -> np.ndarray:
    """``series`` (int64 values) at ``labels``, ``default`` where absent, never through
    floats (keys exceed float precision)."""
    values = series.to_numpy(np.int64)
    position = series.index.get_indexer(labels)
    if not len(values):
        return np.full(len(position), default, np.int64)
    return np.where(position >= 0, values[np.maximum(position, 0)], default)


def _lookup(series: pd.Series, left: np.ndarray, right: np.ndarray, default: int) -> np.ndarray:
    return _take(series, pd.MultiIndex.from_arrays([left, right]), default)


def _country_km(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    unknown = sorted({c for c in [*first, *second] if c not in CENTROIDS})
    if unknown:
        raise ValueError(f"no centroid for IP countries {unknown}; add them to CENTROIDS")
    lat1, lon1 = (np.radians([CENTROIDS[c][i] for c in first]) for i in (0, 1))
    lat2, lon2 = (np.radians([CENTROIDS[c][i] for c in second]) for i in (0, 1))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        (lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _domain_class(email_root: pd.Series) -> np.ndarray:
    domain = email_root.str.rpartition("@")[2]
    return np.select([domain.isin(DISPOSABLE_DOMAINS), domain.isin(COMMON_DOMAINS)], [2, 0],
                     default=1).astype(np.int8)


@dataclass
class _Decisions:
    """Decisions joined to their attempts, with their order keys and cuts."""

    frame: pd.DataFrame  # order_id, decision_at in the caller's order
    row: np.ndarray  # row of the attempt in tables["order_attempts"]
    order_key: np.ndarray  # the attempt's key
    order_sec: np.ndarray  # the attempt's known_at in seconds
    decision_sec: np.ndarray
    cut: np.ndarray  # rows with key < cut are visible at the decision
    cut_current: np.ndarray  # the same, also counting the attempt itself

    def attempt(self, tables: Mapping[str, pd.DataFrame], column: str) -> np.ndarray:
        return tables["order_attempts"][column].to_numpy()[self.row]


def _decisions(tables: Mapping[str, pd.DataFrame], decisions: pd.DataFrame | None,
               keys: Mapping[str, np.ndarray]) -> _Decisions:
    attempts = tables["order_attempts"]
    if not attempts["order_id"].is_unique:
        raise ValueError("order_attempts.order_id must be unique")
    if decisions is None:
        order = np.argsort(keys["order_attempts"], kind="stable")
        decisions = pd.DataFrame({"order_id": attempts["order_id"].to_numpy()[order],
                                  "decision_at": attempts["known_at"].to_numpy()[order]})
    missing_columns = {"order_id", "decision_at"} - set(decisions.columns)
    if missing_columns:
        raise ValueError(f"decisions need columns order_id and decision_at, missing "
                         f"{sorted(missing_columns)}")
    frame = pd.DataFrame({
        "order_id": decisions["order_id"].to_numpy(np.int64),
        "decision_at": pd.to_datetime(decisions["decision_at"]).to_numpy().astype(
            "datetime64[s]"),
    })
    row = pd.Index(attempts["order_id"].to_numpy(np.int64)).get_indexer(frame["order_id"])
    if (row < 0).any():
        missing = frame.loc[row < 0, "order_id"].head(5).tolist()
        raise ValueError(f"decided orders are not in order_attempts: {missing}")
    order_key = keys["order_attempts"][row]
    order_sec = _seconds(attempts["known_at"])[row]
    decision_sec = _seconds(frame["decision_at"])
    if (decision_sec < order_sec).any():
        early = frame.loc[decision_sec < order_sec, "order_id"].head(5).tolist()
        raise ValueError(f"decision_at is before the order's checkout for orders {early}")
    checkout = decision_sec == order_sec
    cut = np.where(checkout, order_key, (decision_sec + 1) << SUB_BITS)
    cut_current = np.where(checkout, order_key + 1, cut)
    return _Decisions(frame, row, order_key, order_sec, decision_sec, cut, cut_current)


def _result(d: _Decisions, tables: Mapping[str, pd.DataFrame],
            columns: Mapping[str, np.ndarray], names: Iterable[str]) -> pd.DataFrame:
    out = pd.DataFrame({
        "order_id": d.frame["order_id"].to_numpy(np.int64),
        "user_id": d.attempt(tables, "user_id").astype(np.int64),
        "merchant_id": d.attempt(tables, "merchant_id").astype(np.int64),
        "decision_at": d.frame["decision_at"].to_numpy(),
    })
    values = {name: np.asarray(columns[name]).astype(_DTYPE[name]) for name in names}
    return pd.concat([out, pd.DataFrame(values)], axis=1)


def _empty_result(names: Iterable[str]) -> pd.DataFrame:
    frame = _frame(order_id="int64", user_id="int64", merchant_id="int64",
                   decision_at="datetime64[s]")
    for name in names:
        frame[name] = pd.Series(dtype=_DTYPE[name])
    return frame


# ------------------------------------------------------------- attempt-derived columns
def attempt_columns(
    tables: Mapping[str, pd.DataFrame],
    decisions: pd.DataFrame | None = None,
    *,
    terms: ledger.ProductTerms | None = None,
    world_context: pd.DataFrame | None = None,
    neighbours: Neighbours | None = None,
) -> pd.DataFrame:
    """KEY_COLUMNS + ATTEMPT_COLUMNS for each decision (default: every attempt at checkout).

    Policy-independent: the replay may compute them once and reuse them for
    every policy. Rows follow ``decisions``; the default is every order attempt
    in the world's total order.

    For later decisions (reviews), pass ``world_context``, rows of an earlier build
    that include each decided order's checkout row: the order-anchored columns are
    taken from it, since they do not change with the decision time, and only the
    decision-anchored ones are computed; with ``neighbours`` as well the work is
    restricted to the decided accounts and their neighbours. The result equals a
    full build.
    """
    if world_context is not None and decisions is not None:
        tables = _relevant(tables, decisions, neighbours)
    keys = _event_keys(tables)
    d = _decisions(tables, decisions, keys)
    if not len(d.row):
        return _empty_result(ATTEMPT_COLUMNS)
    c = _decision_anchored(tables, keys, d)
    if world_context is None:
        c.update(_order_anchored(tables, keys, d, terms or ledger.ProductTerms.from_config()))
    else:
        c.update(_from_checkout(world_context, tables, d))
    return _result(d, tables, c, ATTEMPT_COLUMNS)


def _from_checkout(world_context: pd.DataFrame, tables: Mapping[str, pd.DataFrame],
                   d: _Decisions) -> dict[str, np.ndarray]:
    """The order-anchored columns of each decided order, from its checkout row."""
    names = [column.name for column in COLUMNS
             if column.kind == ATTEMPT and column.anchor == ORDER]
    checkout = pd.DataFrame({"order_id": d.frame["order_id"].to_numpy(np.int64),
                             "decision_at": d.attempt(tables, "known_at").astype(
                                 "datetime64[s]")})
    candidates = world_context[world_context["order_id"].isin(checkout["order_id"])]
    rows = checkout.merge(candidates[["order_id", "decision_at", *names]],
                          on=["order_id", "decision_at"], how="left", validate="many_to_one",
                          indicator=True)
    if (rows["_merge"] != "both").any():
        missing = rows.loc[rows["_merge"] != "both", "order_id"].head(5).tolist()
        raise ValueError(f"world_context has no checkout row for orders {missing}")
    return {name: rows[name].to_numpy() for name in names}


def _decision_anchored(tables: Mapping[str, pd.DataFrame], keys: Mapping[str, np.ndarray],
                       d: _Decisions) -> dict[str, np.ndarray]:
    """The current email and the §2.5 linkage, as known at each decision."""
    attempts = tables["order_attempts"]
    a_key = keys["order_attempts"]
    a_user = attempts["user_id"].to_numpy(np.int64)
    a_device = attempts["device_id"].to_numpy(np.int64)
    a_ship = attempts["ship_address_id"].to_numpy(np.int64)
    user = a_user[d.row]
    c: dict[str, np.ndarray] = {}

    # the current email at the decision (§2.5 counts holdings at the same time)
    history = email_history(tables["accounts"], tables["account_events"])
    root_code, roots = pd.factorize(history["email_root"])
    since = _seconds(history["since"])
    until = _seconds(history["until"])
    by_user = _Index(history["user_id"].to_numpy(np.int64), since)
    held = by_user.last_below(user, d.decision_sec + 1)
    if (held < 0).any():
        raise ValueError("an account holds no email at its decision (created after it?)")
    held_root = root_code[held]
    c["email_domain_class"] = _domain_class(pd.Series(roots[held_root]))
    started = _Index(root_code, since).count(held_root, None, d.decision_sec + 1)
    ended = until != np.iinfo(np.int64).min
    stopped = _Index(root_code[ended], until[ended]).count(held_root, None, d.decision_sec + 1)
    c["email_root_other_accounts"] = started - stopped - 1


    # linkage known at the decision (§2.5), this account included
    events = tables["account_events"]
    e_key = keys["account_events"]
    e_user = events["user_id"].to_numpy(np.int64)
    e_device = events["device_id"].to_numpy(np.int64)
    address_links = tables["address_links"]
    l_user = address_links["user_id"].to_numpy(np.int64)
    l_address = address_links["address_id"].to_numpy(np.int64)
    l_created = _seconds(address_links["created_at"])
    low = _entity_key(d.decision_sec - 30 * _DAY)
    query, _ = _members(np.concatenate([a_device, e_device]), np.concatenate([a_user, e_user]),
                        np.concatenate([a_key, e_key]), a_device[d.row], d.cut_current, low)
    c["accounts_on_device_30d"] = np.bincount(query, minlength=len(user))
    query, _ = _members(a_ship, a_user, a_key, a_ship[d.row], d.cut_current, low)
    c["accounts_on_address_30d"] = np.bincount(query, minlength=len(user))
    query, _ = _members(np.concatenate([a_ship, l_address]), np.concatenate([a_user, l_user]),
                        np.concatenate([a_key, _entity_key(l_created)]), a_ship[d.row],
                        d.cut_current)
    c["accounts_on_address_ever"] = np.bincount(query, minlength=len(user))

    return c


def _order_anchored(tables: Mapping[str, pd.DataFrame], keys: Mapping[str, np.ndarray],
                    d: _Decisions, terms: ledger.ProductTerms) -> dict[str, np.ndarray]:
    """Everything measured at or before the order, whatever the decision time."""
    attempts = tables["order_attempts"]
    a_key = keys["order_attempts"]
    a_sec = _seconds(attempts["known_at"])
    a_user = attempts["user_id"].to_numpy(np.int64)
    a_device = attempts["device_id"].to_numpy(np.int64)
    a_card = attempts["card_id"].to_numpy(np.int64)
    a_ship = attempts["ship_address_id"].to_numpy(np.int64)
    a_amount = attempts["amount_cents"].to_numpy(np.int64)
    events = tables["account_events"]
    e_key = keys["account_events"]
    e_user = events["user_id"].to_numpy(np.int64)
    e_device = events["device_id"].to_numpy(np.int64)
    user = a_user[d.row]
    t = d.order_sec
    here, upto = d.order_key, d.order_key + 1  # before / including the current attempt
    c: dict[str, np.ndarray] = {}

    # the attempt itself and static context
    amount = a_amount[d.row]
    discount = attempts["promo_discount_cents"].to_numpy(np.int64)[d.row]
    fee = (amount * terms.merchant_discount_bps + 5_000) // 10_000  # ledger.round_bps
    down = (amount - discount) * terms.down_payment_bps // 10_000  # ledger.split_principal
    c["amount_cents"] = amount
    c["order_exposure_cents"] = amount - fee - down
    accounts = tables["accounts"].set_index("user_id")
    merchants = tables["merchants"].set_index("merchant_id")
    merchant = d.attempt(tables, "merchant_id")
    c["account_age_days"] = (t - _seconds(accounts["created_at"].reindex(user))) / _DAY
    c["merchant_age_days"] = (t - _seconds(merchants["created_at"].reindex(merchant))) / _DAY
    c["merchant_fulfilment_median_hours"] = merchants["fulfilment_median_hours"].reindex(
        merchant).to_numpy(float)
    c["avs_mismatch"] = d.attempt(tables, "avs_result") == "N"
    c["cvv_mismatch"] = d.attempt(tables, "cvv_result") == "N"
    ip_country = d.attempt(tables, "ip_country").astype(str)
    bin_country = tables["cards"].set_index("card_id")["bin_country"].reindex(a_card[d.row])
    c["bin_ip_country_mismatch"] = bin_country.to_numpy(str) != ip_country
    c["ip_country_not_home"] = accounts["home_country"].reindex(user).to_numpy(str) != ip_country
    hour = d.attempt(tables, "occurred_at").astype("datetime64[h]").astype(np.int64) % 24
    c["night_order"] = hour < 6

    # amounts against earlier processor-approved amounts in the category
    category, _ = pd.factorize(merchants["category"].reindex(
        attempts["merchant_id"]).to_numpy(str))
    approved = (attempts["processor_result"] == "approved").to_numpy()
    for name, quantile in (("amount_over_category_median", 0.5),
                           ("amount_over_category_p95", 0.95)):
        reference = _category_reference(category[approved], a_key[approved],
                                        a_amount[approved], category[d.row], here, quantile)
        c[name] = amount / reference

    # velocity (attempts up to and including the current one)
    by_account = _Index(a_user, a_key)
    for name, width in (("attempts_user_1h", _HOUR), ("attempts_user_24h", _DAY),
                        ("attempts_user_7d", 7 * _DAY)):
        c[name] = by_account.count(user, _entity_key(t - width), upto)
    for name, width in (("amount_attempted_user_24h", _DAY),
                        ("amount_attempted_user_7d", 7 * _DAY)):
        c[name] = by_account.total(a_amount, user, _entity_key(t - width), upto)
    by_device = _Index(a_device, a_key)
    c["attempts_device_24h"] = by_device.count(a_device[d.row], _entity_key(t - _DAY), upto)
    declined = ~approved
    for name, column in (("processor_declines_card_24h", a_card),
                         ("processor_declines_device_24h", a_device)):
        c[name] = _Index(column[declined], a_key[declined]).count(
            column[d.row], _entity_key(t - _DAY), here)
    previous = by_account.last_below(user, here)
    c["is_first_attempt_user"] = previous < 0
    a_country = attempts["ip_country"].to_numpy(str)
    gap_hours = (t - _pick(a_sec, previous)) / _HOUR
    moved = (previous >= 0) & (gap_hours < GEO_WINDOW_HOURS)
    moved[moved] &= a_country[previous[moved]] != ip_country[moved]
    geo = np.zeros(len(user))
    geo[moved] = _country_km(a_country[previous[moved]], ip_country[moved]) / np.maximum(
        gap_hours[moved], GEO_MIN_GAP_HOURS)
    c["geo_kmh_from_previous_attempt"] = geo

    # devices, cards and addresses on the account (as at the order)
    links = tables["device_links"]
    first_link = _first_by_pair(links["user_id"].to_numpy(np.int64),
                                links["device_id"].to_numpy(np.int64),
                                _seconds(links["created_at"]))
    linked = _lookup(first_link, user, a_device[d.row], _NEVER)
    c["device_link_age_hours"] = np.where(linked <= t, (t - linked) / _HOUR, np.nan)
    query, _ = _members(np.concatenate([a_user, e_user]), np.concatenate([a_device, e_device]),
                        np.concatenate([a_key, e_key]), user, upto, _entity_key(t - 30 * _DAY))
    c["distinct_devices_user_30d"] = np.bincount(query, minlength=len(user))
    cards = tables["cards"].set_index("card_id")
    c["card_link_age_hours"] = (t - _seconds(cards["created_at"].reindex(a_card[d.row]))) / _HOUR
    for name, column in (("card_first_use_age_hours", a_card),
                         ("ship_address_first_use_age_hours", a_ship)):
        first = _lookup(_first_by_pair(a_user, column, a_key), user, column[d.row], _NEVER)
        c[name] = (t - (first >> SUB_BITS)) / _HOUR
    address_links = tables["address_links"]
    l_user = address_links["user_id"].to_numpy(np.int64)
    l_address = address_links["address_id"].to_numpy(np.int64)
    l_created = _seconds(address_links["created_at"])
    l_removed = _seconds(address_links["removed_at"])
    l_removed = np.where(l_removed == np.iinfo(np.int64).min, _NEVER, l_removed)
    first_address = _lookup(_first_by_pair(l_user, l_address, l_created), user, a_ship[d.row],
                            _NEVER)
    c["ship_address_link_age_hours"] = np.where(first_address <= t,
                                                (t - first_address) / _HOUR, np.nan)
    homes = pd.DataFrame({"user_id": l_user, "address_id": l_address, "created": l_created,
                          "removed": l_removed})[(address_links["role"] == "home").to_numpy()]
    active = pd.DataFrame({"position": np.arange(len(user)), "user_id": user, "t": t,
                           "ship": a_ship[d.row]}).merge(homes, on="user_id")
    active = active[(active["created"] <= active["t"]) & (active["removed"] > active["t"])]
    c["ship_to_home"] = np.zeros(len(user), bool)
    c["ship_to_home"][active.loc[active["address_id"] == active["ship"], "position"]] = True
    newest = active.groupby("position")["created"].max()
    c["home_address_age_days"] = np.full(len(user), np.nan)
    c["home_address_age_days"][newest.index] = (t[newest.index] - newest.to_numpy()) / _DAY
    first_ship = _first_by_pair(a_user, a_ship, a_key)
    c["distinct_ship_addresses_user_ever"] = _Index(
        first_ship.index.get_level_values(0).to_numpy(np.int64), first_ship.to_numpy()
    ).count(user, None, upto)

    # credential changes before the order
    kind = events["kind"].to_numpy(str)
    e_sec = _seconds(events["known_at"])
    for name, event_kind in (("hours_since_password_change", "password_change"),
                             ("hours_since_password_reset", "password_reset"),
                             ("hours_since_email_change", "email_change"),
                             ("hours_since_phone_change", "phone_change")):
        mask = kind == event_kind
        last = _Index(e_user[mask], e_key[mask]).last_below(user, here)
        elapsed = (t - _pick(e_sec[mask], last)) / _HOUR
        c[name] = np.where(last >= 0, np.minimum(elapsed, NO_EVENT_HOURS), NO_EVENT_HOURS)
    c["hours_since_credential_change"] = np.minimum.reduce([
        c["hours_since_password_change"], c["hours_since_password_reset"],
        c["hours_since_email_change"]])
    return c


def _category_reference(category: np.ndarray, key: np.ndarray, amount: np.ndarray,
                        query_category: np.ndarray, query_key: np.ndarray,
                        quantile: float) -> np.ndarray:
    """The quantile of each category's amounts with key < the query's (NaN below the
    minimum sample)."""
    index = _Index(category, key)
    ordered = amount[index.order].astype(float)
    running = np.empty(len(ordered))
    for value in np.unique(index.sorted_group):
        rows = np.flatnonzero(index.sorted_group == value)
        running[rows] = pd.Series(ordered[rows]).expanding().quantile(quantile).to_numpy()
    position = index.position(query_category, query_key)
    before = position - index.position(query_category, None)
    present, _ = index._code(query_category)
    usable = present & (before >= CATEGORY_MIN_SAMPLE)
    return _pick(running, np.where(usable, position - 1, -1), np.nan)


# ------------------------------------------------------------- outcome-derived columns
def outcome_columns(
    tables: Mapping[str, pd.DataFrame],
    state: PolicyState,
    decisions: pd.DataFrame,
    *,
    neighbours: Neighbours | None = None,
) -> pd.DataFrame:
    """Outcome-derived columns (KEY_COLUMNS + OUTCOME_COLUMNS) under one policy.

    ``tables`` are that policy's realized observations and ``state`` what it did
    (:class:`PolicyState`); under approve-all, the world's tables and
    ``PolicyState.approve_all``. Which orders count follows from ``state`` and
    from what ``tables`` hold (a row that is absent does not exist yet); events
    after a decision's cut are ignored, so passing later rows is harmless.
    ``order_attempts`` must hold each decided order's attempt. The work is
    restricted to the decisions' accounts and every account that ever shared a
    device, an address or an email with one of them; pass the world's
    :class:`Neighbours` (built once) so a day's decisions cost little. Rows follow
    ``decisions``.
    """
    tables = _relevant(tables, decisions, neighbours)
    keys = _event_keys(tables)
    d = _decisions(tables, decisions, keys)
    if not len(d.row):
        return _empty_result(OUTCOME_COLUMNS)
    attempts = tables["order_attempts"]
    a_key = keys["order_attempts"]
    a_user = attempts["user_id"].to_numpy(np.int64)
    a_order = attempts["order_id"].to_numpy(np.int64)
    user = a_user[d.row]
    current = d.frame["order_id"].to_numpy(np.int64)
    n = len(user)
    s = _state_keys(state, attempts, a_key)
    c: dict[str, np.ndarray] = {}

    # the account's other orders the policy let through, as known at each decision
    went_through = s["approved"] < _NEVER
    mine = pd.DataFrame({"position": np.arange(n), "user_id": user, "current": current,
                         "cut": d.cut, "decision_sec": d.decision_sec}).merge(
        pd.DataFrame({"user_id": a_user[went_through], "order_id": a_order[went_through],
                      "approved": s["approved"][went_through],
                      "voided": s["voided"][went_through],
                      "promo": attempts["promo_id"].notna().to_numpy()[went_through]}),
        on="user_id")
    mine = mine[mine["order_id"] != mine["current"]]
    live = mine[(mine["approved"] < mine["cut"]) & (mine["voided"] >= mine["cut"])]
    c["approved_orders_user_ever"] = _per(live, n)
    recent = (live["approved"].to_numpy(np.int64) >> SUB_BITS) >= live["decision_sec"] - _DAY
    c["approved_orders_user_24h"] = _per(live[recent], n)
    c["promo_redemptions_user"] = _per(live[live["promo"]], n)
    c.update(_repayment(tables, keys, live, n))

    # disputes and reports on the account's other orders that went through
    other = mine[mine["approved"] < _NEVER][["position", "order_id", "cut"]]
    disputes = _disputes(tables, keys)
    on_mine = other.merge(disputes, on="order_id")
    opened = on_mine[on_mine["opened"] < on_mine["cut"]]
    resolved = on_mine[on_mine["resolved"] < on_mine["cut"]]
    unauthorized = resolved["reason"] == "unauthorized"
    c["unauthorized_disputes_lost_user"] = _per(resolved[unauthorized
                                                         & (resolved["outcome"] == "lost")], n)
    inr = "item_not_received"
    c["inr_disputes_opened_user"] = _per(opened[opened["reason"] == inr], n)
    rejected = resolved[(resolved["reason"] == inr) & (resolved["outcome"] == "won")
                        & (resolved["delivered"] < resolved["cut"])]
    c["inr_claims_rejected_user"] = _per(rejected, n)
    reports = tables["victim_reports"]
    reported = pd.DataFrame({"user_id": reports["user_id"].to_numpy(np.int64),
                             "order_id": reports["order_id"].to_numpy(np.int64),
                             "key": keys["victim_reports"]})
    reported = mine[mine["approved"] < _NEVER][["position", "user_id", "order_id", "cut"]].merge(
        reported, on=["user_id", "order_id"])
    c["victim_reports_user"] = _per(reported[reported["key"] < reported["cut"]], n)
    a_card = attempts["card_id"].to_numpy(np.int64)
    card_disputes = disputes[disputes["order_id"].isin(a_order[went_through])
                             & (disputes["reason"] == "unauthorized")]
    on_card = pd.DataFrame({"position": np.arange(n), "card": a_card[d.row], "current": current,
                            "cut": d.cut}).merge(
        card_disputes.assign(card=_map(card_disputes["order_id"], attempts, "card_id")),
        on="card")
    on_card = on_card[(on_card["order_id"] != on_card["current"])
                      & (on_card["opened"] < on_card["cut"])]
    c["unauthorized_disputes_on_card"] = _per(on_card, n)

    # never-pay determinations on the account's earlier plans (core.world.adjudicate's rule)
    stopped = np.minimum(s["voided"], s["cancelled"])
    determined = _never_pay(tables, a_order[went_through], s["approved"][went_through],
                            stopped[went_through], other["order_id"].unique(),
                            d.decision_sec.max())
    known = other.merge(determined, on="order_id")
    c["never_pay_determined_user"] = _per(known[known["determined"] < known["cut"]], n) > 0

    c["promo_uses_linked_accounts"] = _linked_promo_uses(tables, attempts, d, s, user)

    # the current order and the account under the policy
    shipped = _first_key(tables["fulfilments"]["order_id"], keys["fulfilments"])
    c["shipped_at_decision"] = _take(shipped, current, _NEVER) < d.cut
    stopped = np.minimum(s["voided"], s["cancelled"])[d.row]
    c["cancelled_at_decision"] = stopped < d.cut
    blocked = s["blocked"]
    c["account_blocked"] = _take(blocked, user, _NEVER) < d.cut
    query, other_user = _linked(tables, keys, d)
    blocked_then = _take(blocked, other_user, _NEVER) < d.cut[query]
    c["linked_account_blocked_30d"] = np.bincount(query[blocked_then], minlength=n) > 0
    return _result(d, tables, c, OUTCOME_COLUMNS)


def _per(frame: pd.DataFrame, n: int) -> np.ndarray:
    """Rows per decision position."""
    return np.bincount(frame["position"].to_numpy(np.int64), minlength=n)


def _map(order_id: pd.Series, attempts: pd.DataFrame, column: str) -> np.ndarray:
    return attempts.set_index("order_id")[column].reindex(order_id).to_numpy()


def _first_key(ids: pd.Series, keys: np.ndarray) -> pd.Series:
    return pd.Series(keys, index=ids.to_numpy(np.int64)).groupby(level=0).min()


def _state_keys(state: PolicyState, attempts: pd.DataFrame,
                a_key: np.ndarray) -> dict[str, np.ndarray | pd.Series]:
    """Per attempt row: when the order went through, was voided or cancelled (keys,
    _NEVER when not); per account: when it was blocked."""
    position = pd.Index(attempts["order_id"].to_numpy(np.int64))
    order_sec = _seconds(attempts["known_at"])

    def per_order(ids: pd.Series, keys: np.ndarray) -> np.ndarray:
        out = np.full(len(attempts), _NEVER, np.int64)
        row = position.get_indexer(ids.to_numpy(np.int64))
        present = row >= 0  # rows of orders that are not in the tables do not exist yet
        np.minimum.at(out, row[present], keys[present])
        return out

    approved = state.approved
    row = position.get_indexer(approved["order_id"].to_numpy(np.int64))
    approved_sec = _seconds(approved["approved_at"])
    present = row >= 0
    if (approved_sec[present] < order_sec[row[present]]).any():
        raise ValueError("an order is approved before its checkout")
    at_checkout = present & (approved_sec == _pick(order_sec, row))
    approved_key = np.where(at_checkout, _pick(a_key, row), _derived_key(approved_sec))
    held = state.held
    # a hold placed after shipment pauses nothing, so its end never cancels the order
    cancelled = held[held["outcome"].isin(["cancelled", "declined"])
                     & held["released_at"].notna() & held["before_shipment"].astype(bool)]
    blocked = state.blocked
    blocked_key = pd.Series(_derived_key(_seconds(blocked["at"])),
                            index=blocked["user_id"].to_numpy(np.int64))
    return {
        "approved": per_order(approved["order_id"], approved_key),
        "voided": per_order(state.voided["order_id"], _derived_key(_seconds(state.voided["at"]))),
        "cancelled": per_order(cancelled["order_id"],
                               _derived_key(_seconds(cancelled["released_at"]))),
        "blocked": blocked_key.groupby(level=0).min(),
    }


def _repayment(tables: Mapping[str, pd.DataFrame], keys: Mapping[str, np.ndarray],
               live: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    """Installment and balance columns over each decision's live plans."""
    plans = tables["plans"][["plan_id", "order_id", "principal_cents"]]
    held = live[["position", "order_id", "cut"]].merge(plans, on="order_id")
    payments = tables["payment_attempts"]
    p_key = keys["payment_attempts"]
    success = (payments["result"] == "success").to_numpy()
    p_plan = payments["plan_id"].to_numpy(np.int64)
    p_seq = payments["seq"].to_numpy(np.int64)
    p_amount = payments["amount_cents"].to_numpy(np.int64)
    reversals = tables["payment_reversals"]
    reversed_payment = pd.Index(payments["event_id"].to_numpy(np.int64)).get_indexer(
        reversals["payment_event_id"].to_numpy(np.int64))
    if (reversed_payment < 0).any():
        raise ValueError("a payment reversal names a payment that is not in payment_attempts")
    r_key = keys["payment_reversals"]
    r_amount = reversals["amount_cents"].to_numpy(np.int64)
    r_plan, r_seq = p_plan[reversed_payment], p_seq[reversed_payment]
    schedule = tables["installment_schedule"]
    width = int(max(p_seq.max(initial=0), schedule["seq"].max() if len(schedule) else 0)) + 1
    schedule = schedule[schedule["seq"] >= 1]
    due = held[["position", "plan_id", "cut"]].merge(
        schedule[["plan_id", "seq", "due_at", "amount_cents"]], on="plan_id")
    due = due[_derived_key(_seconds(due["due_at"])) < due["cut"].to_numpy()]
    slot = due["plan_id"].to_numpy(np.int64) * width + due["seq"].to_numpy(np.int64)
    cut = due["cut"].to_numpy(np.int64)
    paid_in = _Index(p_plan[success] * width + p_seq[success], p_key[success]).total(
        p_amount[success], slot, None, cut)
    paid_back = _Index(r_plan * width + r_seq, r_key).total(r_amount, slot, None, cut)
    paid = paid_in - paid_back >= due["amount_cents"].to_numpy(np.int64)
    failed = _Index(p_plan[~success] * width + p_seq[~success], p_key[~success]).count(
        slot, None, cut) > 0
    position = due["position"].to_numpy(np.int64)
    out = {
        "installments_due_user": np.bincount(position, minlength=n),
        "installments_paid_user": np.bincount(position[paid], minlength=n),
        "installments_failed_user": np.bincount(position[failed & ~paid], minlength=n),
    }
    out["installments_paid_share_user"] = np.divide(
        out["installments_paid_user"], out["installments_due_user"],
        out=np.zeros(n), where=out["installments_due_user"] > 0)
    writeoffs = _first_key(tables["plan_writeoffs"]["plan_id"], keys["plan_writeoffs"])
    held = held[_take(writeoffs, held["plan_id"].to_numpy(np.int64), _NEVER)
                >= held["cut"].to_numpy()]
    plan = held["plan_id"].to_numpy(np.int64)
    cut = held["cut"].to_numpy(np.int64)
    standing = (_Index(p_plan[success], p_key[success]).total(p_amount[success], plan, None, cut)
                - _Index(r_plan, r_key).total(r_amount, plan, None, cut))
    balance = held["principal_cents"].to_numpy(np.int64) - standing
    out["open_balance_user_cents"] = np.bincount(held["position"].to_numpy(np.int64),
                                                 weights=balance, minlength=n).astype(np.int64)
    return out


def _disputes(tables: Mapping[str, pd.DataFrame],
              keys: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One row per dispute: order, reason, opening key, resolution key and outcome, and
    the key of the order's first delivery (_NEVER where none)."""
    openings = tables["dispute_openings"]
    resolutions = tables["dispute_resolutions"]
    disputes = pd.DataFrame({"dispute_id": openings["dispute_id"].to_numpy(np.int64),
                             "order_id": openings["order_id"].to_numpy(np.int64),
                             "reason": openings["reason"].to_numpy(str),
                             "opened": keys["dispute_openings"]})
    resolution = pd.Index(resolutions["dispute_id"].to_numpy(np.int64)).get_indexer(
        disputes["dispute_id"])
    if not resolutions["dispute_id"].is_unique:
        raise ValueError("a dispute has more than one resolution")
    outcome = resolutions["outcome"].to_numpy(str)
    disputes["outcome"] = _pick(outcome, resolution, "")
    disputes["resolved"] = _pick(keys["dispute_resolutions"], resolution, _NEVER)
    delivered = _first_key(tables["deliveries"]["order_id"], keys["deliveries"])
    disputes["delivered"] = _take(delivered, disputes["order_id"].to_numpy(np.int64), _NEVER)
    return disputes


def _never_pay(tables: Mapping[str, pd.DataFrame], went_through: np.ndarray,
               approved: np.ndarray, stopped: np.ndarray, asked: np.ndarray,
               latest_sec: int) -> pd.DataFrame:
    """order_id, determined (key): never-pay determinations (fraud policy §8.3) among the
    ``asked`` orders, judging the orders that went through as approved; ``approved`` is
    when each went through and ``stopped`` when it was voided or cancelled (keys, the
    maximum when never). Labels are never read.

    The rule is core.world.adjudicate's, run on these tables with three changes. An
    order counts from when the policy let it through, so a shipping address an order
    released from a hold goes to is shared from the release, not the attempt. A
    voided or cancelled plan owes nothing after it stops, so it defaults (and supplies
    a marker) only when its default came before that. And the labelling function
    keeps one determination per order, the first known, so the evidence of every
    other determination (disputes, victim reports, first-purchase promotions) is
    withheld: the never-pay rule reads none of it, and a plan that meets §8.3 is
    reported even when, say, a victim report labelled its order first. Without a
    zero-effort default among the asked plans nothing is computed."""
    none = pd.DataFrame({"order_id": np.empty(0, np.int64), "determined": np.empty(0, np.int64)})
    rules = config.load("world")["labels"]
    plans = tables["plans"]
    plans = plans[plans["order_id"].isin(went_through)]
    stop_sec = pd.Series(stopped >> SUB_BITS, index=went_through)
    stop_sec = stop_sec[stopped < _NEVER]
    schedule = tables["installment_schedule"]
    if len(stop_sec):
        stop = _take(stop_sec, _take(plans.set_index("plan_id")["order_id"],
                                     schedule["plan_id"].to_numpy(np.int64), -1), _NEVER)
        default = _seconds(schedule["due_at"]) + rules["default_grace_days"] * _DAY
        schedule = schedule[(schedule["seq"] < 1).to_numpy() | (default < stop)]
    mine = plans[plans["order_id"].isin(asked)][["plan_id", "order_id", "created_at"]].merge(
        tables["order_attempts"][["order_id", "user_id"]], on="order_id")
    observed = {**tables, "installment_schedule": schedule}
    if mine.empty or world._zero_effort_defaults(
            observed, mine, rules["default_grace_days"]).empty:
        return none
    orders = tables["order_attempts"].copy()
    through = _take(pd.Series(approved >> SUB_BITS, index=went_through),
                    orders["order_id"].to_numpy(np.int64), -1)
    orders["processor_result"] = np.where(through >= 0, "approved", "declined")
    orders["occurred_at"] = np.where(through >= 0, through.astype("datetime64[s]"),
                                     orders["occurred_at"].to_numpy("datetime64[s]"))
    withheld = {name: world.empty(name) for name in (
        "dispute_openings", "dispute_resolutions", "victim_reports")}
    labels = world.adjudicate(
        {**observed, **withheld, "order_attempts": orders,
         "promotions": tables["promotions"].assign(first_purchase_only=False)},
        horizon_days=0, observed_until=pd.Timestamp(latest_sec, unit="s"), **rules)
    labels = labels[(labels["basis"] == "never_pay") & labels["order_id"].isin(asked)]
    return pd.DataFrame({"order_id": labels["order_id"].to_numpy(np.int64),
                         "determined": _derived_key(_seconds(labels["label_known_at"]))})


def _linked_promo_uses(tables: Mapping[str, pd.DataFrame], attempts: pd.DataFrame,
                       d: _Decisions, s: Mapping[str, np.ndarray], user: np.ndarray
                       ) -> np.ndarray:
    """R10's input: accounts sharing a device or email with this one (this one included)
    that used the current order's first-purchase promotion on an order that went
    through and was not voided by the decision; 0 without such a promotion."""
    promotions = tables["promotions"]
    first_purchase = set(promotions.loc[promotions["first_purchase_only"].astype(bool),
                                        "promo_id"].astype(np.int64))
    promo = attempts["promo_id"].to_numpy()[d.row]
    uses = pd.notna(promo) & np.isin(pd.Series(promo).fillna(-1).astype(np.int64),
                                     list(first_purchase))
    out = uses.astype(np.int64)
    if not uses.any():
        return out
    pairs = world._shared_accounts(tables, attempts, shipping=False)
    asked = pd.DataFrame({"position": np.flatnonzero(uses), "user_id": user[uses],
                          "promo": pd.Series(promo[uses]).astype(np.int64).to_numpy(),
                          "decision_sec": d.decision_sec[uses], "cut": d.cut[uses]})
    asked = asked.merge(pairs, on="user_id")
    asked = asked[_seconds(asked["shared_at"]) <= asked["decision_sec"].to_numpy()]
    used = attempts["promo_id"].notna().to_numpy() & (s["approved"] < _NEVER)
    others = pd.DataFrame({"other_id": attempts["user_id"].to_numpy(np.int64)[used],
                           "promo": attempts["promo_id"].to_numpy()[used].astype(np.int64),
                           "approved": s["approved"][used], "voided": s["voided"][used]})
    asked = asked.merge(others, on=["other_id", "promo"])
    asked = asked[(asked["approved"] < asked["cut"]) & (asked["voided"] >= asked["cut"])]
    counted = asked.drop_duplicates(["position", "other_id"])
    out += np.bincount(counted["position"].to_numpy(np.int64), minlength=len(out))
    return out


def _linked(tables: Mapping[str, pd.DataFrame], keys: Mapping[str, np.ndarray],
            d: _Decisions) -> tuple[np.ndarray, np.ndarray]:
    """(decision position, user) for the §2.5 linked accounts: other accounts with an
    attempt or account event on the order's device, or an attempt to its shipping
    address, in the 30 days before the decision."""
    attempts = tables["order_attempts"]
    events = tables["account_events"]
    a_user = attempts["user_id"].to_numpy(np.int64)
    a_device = attempts["device_id"].to_numpy(np.int64)
    a_ship = attempts["ship_address_id"].to_numpy(np.int64)
    low = _entity_key(d.decision_sec - 30 * _DAY)
    on_device = _members(np.concatenate([a_device, events["device_id"].to_numpy(np.int64)]),
                         np.concatenate([a_user, events["user_id"].to_numpy(np.int64)]),
                         np.concatenate([keys["order_attempts"], keys["account_events"]]),
                         a_device[d.row], d.cut, low)
    on_address = _members(a_ship, a_user, keys["order_attempts"], a_ship[d.row], d.cut, low)
    frame = pd.DataFrame({"query": np.concatenate([on_device[0], on_address[0]]),
                          "user_id": np.concatenate([on_device[1], on_address[1]])})
    frame = frame[frame["user_id"].to_numpy() != a_user[d.row][frame["query"].to_numpy()]]
    frame = frame.drop_duplicates().sort_values(["query", "user_id"])
    return frame["query"].to_numpy(np.int64), frame["user_id"].to_numpy(np.int64)


def _account_items(tables: Mapping[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Unique (account, item) pairs for everything accounts can share: devices (attempts,
    account events, links), addresses (attempts, links) and normalized emails (held
    ever). Items are integer codes, distinct across the three kinds."""
    attempts = tables["order_attempts"]
    events = tables["account_events"]
    device_links = tables["device_links"]
    address_links = tables["address_links"]
    accounts = tables["accounts"]
    changes = events[events["kind"] == "email_change"]
    emails, _ = pd.factorize(pd.concat([accounts["email"], changes["email"]]).map(
        normalize_email).to_numpy(str))
    pairs = (
        (attempts["user_id"], attempts["device_id"].to_numpy(np.int64) * 3),
        (events["user_id"], events["device_id"].to_numpy(np.int64) * 3),
        (device_links["user_id"], device_links["device_id"].to_numpy(np.int64) * 3),
        (attempts["user_id"], attempts["ship_address_id"].to_numpy(np.int64) * 3 + 1),
        (address_links["user_id"], address_links["address_id"].to_numpy(np.int64) * 3 + 1),
        (pd.concat([accounts["user_id"], changes["user_id"]]), emails.astype(np.int64) * 3 + 2),
    )
    stacked = np.unique(np.stack([np.concatenate([u.to_numpy(np.int64) for u, _ in pairs]),
                                  np.concatenate([i for _, i in pairs])], axis=1), axis=0)
    return stacked[:, 0], stacked[:, 1]


@dataclass(frozen=True)
class Neighbours:
    """Accounts that ever shared a device, an address or an email with each account.

    Built once per world from its policy-independent tables (:meth:`of`) and passed
    to :func:`outcome_columns` so that each call restricts its work to the decided
    accounts and their neighbours without scanning the world for them.
    """

    users: np.ndarray  # account ids, sorted
    start: np.ndarray  # neighbours of users[k] are other[start[k]:start[k + 1]]
    other: np.ndarray

    @classmethod
    def of(cls, tables: Mapping[str, pd.DataFrame]) -> Neighbours:
        user, item = _account_items(tables)
        pairs = pd.DataFrame({"user_id": user, "item": item})
        pairs = pairs.merge(pairs, on="item", suffixes=("", "_other"))[
            ["user_id", "user_id_other"]].drop_duplicates().sort_values(
            ["user_id", "user_id_other"])
        users, counts = np.unique(pairs["user_id"].to_numpy(np.int64), return_counts=True)
        return cls(users, np.concatenate([[0], np.cumsum(counts)]),
                   pairs["user_id_other"].to_numpy(np.int64))

    def around(self, users: np.ndarray) -> np.ndarray:
        """``users`` and every account that shared something with one of them."""
        users = np.asarray(users, np.int64)
        position = np.searchsorted(self.users, users)
        known = (position < len(self.users)) & (
            self.users[np.minimum(position, max(len(self.users) - 1, 0))] == users
        ) if len(self.users) else np.zeros(len(users), bool)
        parts = [users] + [self.other[self.start[k]:self.start[k + 1]]
                           for k in position[known]]
        return np.unique(np.concatenate(parts))


def _relevant(tables: Mapping[str, pd.DataFrame], decisions: pd.DataFrame,
              neighbours: Neighbours | None = None) -> Mapping[str, pd.DataFrame]:
    """The tables restricted to the decisions' accounts and every account that ever shared
    a device, an address or an email with one of them (enough for every outcome column).
    Unrestricted when the decisions cover most accounts."""
    attempts = tables["order_attempts"]
    a_user = attempts["user_id"].to_numpy(np.int64)
    decided = np.unique(a_user[attempts["order_id"].isin(decisions["order_id"]).to_numpy()])
    if len(decided) > 0.3 * len(tables["accounts"]):
        return tables
    if neighbours is not None:
        keep = neighbours.around(decided)
    else:
        user, item = _account_items(tables)
        keep = np.union1d(decided, user[np.isin(item, item[np.isin(user, decided)])])
    orders = attempts.loc[np.isin(a_user, keep), "order_id"].to_numpy(np.int64)
    plans = tables["plans"].loc[tables["plans"]["order_id"].isin(orders), "plan_id"].to_numpy(
        np.int64)
    openings = tables["dispute_openings"]
    disputes = openings.loc[openings["order_id"].isin(orders), "dispute_id"].to_numpy(np.int64)
    by = {"user_id": keep, "order_id": orders, "plan_id": plans, "dispute_id": disputes}
    restrict = {
        "accounts": "user_id", "account_events": "user_id", "device_links": "user_id",
        "address_links": "user_id", "cards": "user_id", "order_attempts": "user_id",
        "plans": "order_id", "installment_schedule": "plan_id", "payment_attempts": "plan_id",
        "payment_reversals": "plan_id", "plan_writeoffs": "plan_id", "fulfilments": "order_id",
        "deliveries": "order_id", "dispute_openings": "order_id",
        "dispute_resolutions": "dispute_id", "victim_reports": "order_id",
        "cash_events": "order_id",
    }
    out = dict(tables)
    for name, column in restrict.items():
        if name in out:
            frame = out[name]
            out[name] = frame[np.isin(frame[column].to_numpy(np.int64), by[column])].reset_index(
                drop=True)
    return out


# ------------------------------------------------------------------------ public builders
def build_context(
    tables: Mapping[str, pd.DataFrame],
    decisions: pd.DataFrame | None = None,
    *,
    terms: ledger.ProductTerms | None = None,
    world_context: pd.DataFrame | None = None,
    neighbours: Neighbours | None = None,
) -> pd.DataFrame:
    """The world-level context: one row per decision with KEY_COLUMNS + COLUMN_NAMES.

    ``decisions`` has ``order_id`` and ``decision_at``; by default every order
    attempt at checkout, in the world's total order. Outcome-derived columns take
    their approve-all values. Reads only :data:`INPUT_TABLES`. For decisions after
    checkout, ``world_context`` (an earlier build holding each order's checkout row)
    and ``neighbours`` make a call cost little; see :func:`attempt_columns`. The
    result equals a full build either way.
    """
    tables = {name: tables[name] for name in INPUT_TABLES if name in tables}
    if world_context is not None and decisions is not None:
        tables = _relevant(tables, decisions, neighbours)
    attempt = attempt_columns(tables, decisions, terms=terms, world_context=world_context,
                              neighbours=neighbours)
    asked = attempt[["order_id", "decision_at"]]
    outcome = outcome_columns(tables, PolicyState.approve_all(tables), asked,
                              neighbours=neighbours)
    return pd.concat([attempt, outcome[list(OUTCOME_COLUMNS)]], axis=1)[
        [*KEY_COLUMNS, *COLUMN_NAMES]]


def policy_rows(
    world_context: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    state: PolicyState,
    decisions: pd.DataFrame,
    *,
    neighbours: Neighbours | None = None,
) -> pd.DataFrame:
    """Context rows for the replay: attempt-derived columns from ``world_context`` (rows
    of :func:`build_context` or :func:`attempt_columns` at the same decision times)
    and outcome-derived columns under one policy (:func:`outcome_columns` on its
    realized ``tables`` and ``state``). Rows follow ``decisions``."""
    asked = pd.DataFrame({
        "order_id": decisions["order_id"].to_numpy(np.int64),
        "decision_at": pd.to_datetime(decisions["decision_at"]).to_numpy().astype(
            "datetime64[s]"),
    })
    attempt = asked.merge(world_context[[*KEY_COLUMNS, *ATTEMPT_COLUMNS]],
                          on=["order_id", "decision_at"], how="left", validate="many_to_one")
    if attempt["user_id"].isna().any():
        missing = attempt.loc[attempt["user_id"].isna(), "order_id"].head(5).tolist()
        raise ValueError(f"world_context has no row for these decisions: orders {missing}")
    outcome = outcome_columns(tables, state, asked, neighbours=neighbours)
    return pd.concat([attempt.reset_index(drop=True), outcome[list(OUTCOME_COLUMNS)]], axis=1)[
        [*KEY_COLUMNS, *COLUMN_NAMES]].astype({"user_id": "int64", "merchant_id": "int64"})


def linked_accounts(tables: Mapping[str, pd.DataFrame], decisions: pd.DataFrame) -> pd.DataFrame:
    """The fraud policy's §2.5 linked accounts at each decision: order_id, decision_at,
    user_id for every other account with an order attempt or account event on the
    order's device, or an order attempt to its shipping address, in the 30 days
    before the decision (what ``escalate`` blocks; the accounts behind
    ``accounts_on_device_30d`` and ``accounts_on_address_30d``)."""
    keys = _event_keys(tables)
    d = _decisions(tables, decisions, keys)
    query, users = _linked(tables, keys, d)
    return pd.DataFrame({"order_id": d.frame["order_id"].to_numpy(np.int64)[query],
                         "decision_at": d.frame["decision_at"].to_numpy()[query],
                         "user_id": users}).reset_index(drop=True)
