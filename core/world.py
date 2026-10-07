"""World contract: tables, truth layers, event order, manifest, labels and validation.

The synthetic world is a set of tables in three layers, stored separately:

* **observable**: what the platform records. Entities with creation times
  (accounts, devices, addresses and their links to accounts, cards, merchants,
  promotions, plans), operational events, the installment schedule and the
  cash ledger (:mod:`core.ledger`).
* **latent**: simulator-only truth about actors, episodes and intent. Rules,
  features, the reviewer, packets, the referee and prompts never read it; it is
  reported only as a separate diagnostic.
* **adjudicated**: labels from a deterministic function of observable outcomes
  (:func:`adjudicate`), each with the time it became known. Unknown is not
  negative.

Time. Every operational event has ``occurred_at`` (when it happened) and
``known_at`` (when the platform could know it), with ``known_at >= occurred_at``.
As-of code filters on ``known_at`` only. Entity rows are known from their
creation column (``created_at``; ``valid_from`` for promotions). Timestamps are
naive platform-local time at one-second resolution; money is integer cents.

Order. Simultaneous events are ordered by ``(known_at, kind rank, event_id)``,
where the kind is the event's table (:data:`EVENT_RANK`). Event ids are unique
across all event tables and assigned after that global sort, so they follow it
and carry no trace of how an event was generated. Entity ids likewise increase
with creation time.

Processes, not final states. The schedule says what is due; payment attempts,
reversals and write-offs, and dispute openings and resolutions, are dated
events. The world holds every attempted order with the outcomes it would have
if approved (only processor declines happen inside the world); policies act
only in the replay.

:func:`validate_world` raises :class:`WorldError` on every violation it finds;
nothing is clipped or repaired.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import ledger

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DAY = pd.Timedelta(days=1)

# Logical column types and their pandas dtypes.
DTYPES = {
    "id": "int64",
    "int": "int64",
    "cents": "int64",
    "float": "float64",
    "ts": "datetime64[s]",
    "str": "str",
    "bool": "bool",
}
NULLABLE_DTYPES = {"id": "Int64", "int": "Int64", "cents": "Int64"}
INTEGER_TYPES = frozenset(NULLABLE_DTYPES)

LAYERS = ("observable", "latent", "adjudicated")
ROLES = ("entity", "schedule", "event", "truth", "label")

PATTERNS = (
    "P-ATO",  # account takeover
    "P-STOLEN",  # stolen card, including card testing and aged sleeper accounts
    "P-SYNTH",  # synthetic-identity ring
    "P-NEVERPAY",  # first-party never-pay
    "P-INR-ABUSE",  # false item-not-received claims
    "P-PROMO",  # promotion farming across linked accounts
    "P-MERCH",  # merchant bust-out
)
DISPUTE_REASONS = ("unauthorized", "item_not_received", "not_as_described")
LABEL_POSITIVE_BASES = (
    "third_party_fraud",
    "account_takeover",
    "never_pay",
    "inr_abuse",
    "promo_abuse",
    "merchant_bustout",
)
LABEL_NEGATIVE_BASES = ("credit_loss", "no_finding")
CASH_KINDS = tuple(ledger.CASH_KINDS)


@dataclass(frozen=True)
class Column:
    name: str
    type: str  # a key of DTYPES
    meaning: str
    nullable: bool = False
    values: tuple[str, ...] | None = None  # closed vocabulary
    references: str | None = None  # "table.column"

    @property
    def dtype(self) -> str:
        if self.nullable and self.type in NULLABLE_DTYPES:
            return NULLABLE_DTYPES[self.type]
        return DTYPES[self.type]


@dataclass(frozen=True)
class TableSpec:
    name: str
    layer: str
    role: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    description: str
    created: str | None = None  # entity creation column
    id_column: str | None = None  # entity id that must increase with `created`

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"{self.name} has no column {name!r}")


def _c(name: str, type_: str, meaning: str, **kw: Any) -> Column:
    return Column(name, type_, meaning, **kw)


def _event_columns() -> tuple[Column, ...]:
    return (
        _c("event_id", "id", "World-unique event id, assigned in the global event order."),
        _c("occurred_at", "ts", "When the event happened."),
        _c("known_at", "ts", "When the platform could know it; as-of code filters on this."),
    )


TABLES: dict[str, TableSpec] = {}


def _add(spec: TableSpec) -> None:
    assert spec.layer in LAYERS and spec.role in ROLES, spec.name
    TABLES[spec.name] = spec


# --------------------------------------------------------------- entities
_add(TableSpec(
    "accounts", "observable", "entity", ("user_id",), (
        _c("user_id", "id", "Customer account id."),
        _c("created_at", "ts", "Signup time."),
        _c("email", "str", "Signup email as entered (normalised only in core.asof); later "
           "addresses are email_change events."),
        _c("email_domain", "str", "Domain of the signup email, lower case."),
        _c("home_country", "str", "Country of residence from KYC; drives the home IP country."),
        _c("dob_year", "int", "Year of birth."),
    ),
    "One row per customer account.", created="created_at", id_column="user_id",
))
_add(TableSpec(
    "merchants", "observable", "entity", ("merchant_id",), (
        _c("merchant_id", "id", "Merchant id."),
        _c("created_at", "ts", "Onboarding time."),
        _c("name", "str", "Trading name."),
        _c("category", "str", "Merchandise category."),
        _c("risk_tier", "int", "Onboarding risk tier, 1 (low) to 3 (high)."),
        _c("fulfilment_median_hours", "float", "Median hours from approval to shipment."),
        _c("closed_at", "ts", "When the merchant stopped trading and became unreachable.",
           nullable=True),
    ),
    "One row per merchant. A closed merchant takes no orders and cannot be charged back.",
    created="created_at", id_column="merchant_id",
))
_add(TableSpec(
    "devices", "observable", "entity", ("device_id",), (
        _c("device_id", "id", "Device id."),
        _c("created_at", "ts", "First time the device was seen on the platform."),
        _c("fingerprint", "str", "Device fingerprint."),
        _c("ua_family", "str", "Browser or OS family."),
    ),
    "One row per physical device.", created="created_at", id_column="device_id",
))
_add(TableSpec(
    "device_links", "observable", "entity", ("user_id", "device_id", "created_at"), (
        _c("user_id", "id", "Account.", references="accounts.user_id"),
        _c("device_id", "id", "Device.", references="devices.device_id"),
        _c("created_at", "ts", "First use of the device on this account."),
        _c("removed_at", "ts", "When the device stopped being usable on the account.",
           nullable=True),
    ),
    "Device association: an account may use a device in [created_at, removed_at).",
    created="created_at",
))
_add(TableSpec(
    "addresses", "observable", "entity", ("address_id",), (
        _c("address_id", "id", "Physical address id."),
        _c("created_at", "ts", "First time any account registered the address."),
        _c("line_hash", "str", "Hash of the normalised street line."),
        _c("city", "str", "City."),
        _c("region", "str", "State or province."),
        _c("country", "str", "Country."),
    ),
    "One row per physical address; households and drops share rows.",
    created="created_at", id_column="address_id",
))
_add(TableSpec(
    "address_links", "observable", "entity", ("user_id", "address_id", "created_at"), (
        _c("user_id", "id", "Account.", references="accounts.user_id"),
        _c("address_id", "id", "Address.", references="addresses.address_id"),
        _c("created_at", "ts", "When the account added the address."),
        _c("removed_at", "ts", "When the account removed it (a move ends the old home).",
           nullable=True),
        _c("role", "str", "home: the account's residence while active; shipping: an "
           "extra delivery address (gift recipient, office, drop).",
           values=("home", "shipping")),
    ),
    "Address association: an order may ship to an address linked and active at its time.",
    created="created_at",
))
_add(TableSpec(
    "cards", "observable", "entity", ("card_id",), (
        _c("card_id", "id", "Card id (one per account registration)."),
        _c("user_id", "id", "Account that registered the card.", references="accounts.user_id"),
        _c("created_at", "ts", "When the card was added to the account."),
        _c("removed_at", "ts", "When it was removed.", nullable=True),
        _c("bin_country", "str", "Issuing country from the BIN."),
        _c("network", "str", "Card network."),
        _c("last4", "str", "Last four digits."),
    ),
    "Payment cards on accounts.", created="created_at", id_column="card_id",
))
_add(TableSpec(
    "promotions", "observable", "entity", ("promo_id",), (
        _c("promo_id", "id", "Promotion id."),
        _c("code", "str", "Promotion code."),
        _c("discount_bps", "int", "Discount on the merchant's price, in basis points."),
        _c("first_purchase_only", "bool", "Only an account's first approved order may use it."),
        _c("valid_from", "ts", "Start of validity (creation)."),
        _c("valid_to", "ts", "End of validity (exclusive)."),
    ),
    "Platform-funded promotions.", created="valid_from", id_column="promo_id",
))
_add(TableSpec(
    "plans", "observable", "entity", ("plan_id",), (
        _c("plan_id", "id", "Plan id."),
        _c("order_id", "id", "The processor-approved order it finances.",
           references="order_attempts.order_id"),
        _c("created_at", "ts", "Approval time (equals the order's occurred_at)."),
        _c("principal_cents", "cents", "Customer obligation: amount minus promotion discount."),
        _c("down_payment_cents", "cents", "Collected at approval (schedule seq 0)."),
        _c("n_installments", "int", "Installments after the down payment."),
    ),
    "One pay-in-4 plan per processor-approved order (the approve-all potential outcome).",
    created="created_at", id_column="plan_id",
))
_add(TableSpec(
    "installment_schedule", "observable", "schedule", ("plan_id", "seq"), (
        _c("plan_id", "id", "Plan.", references="plans.plan_id"),
        _c("seq", "int", "0 = down payment due at approval; 1..n installments."),
        _c("due_at", "ts", "Due time."),
        _c("amount_cents", "cents", "Amount due (core.ledger.split_principal)."),
    ),
    "What each plan owes and when; known from the plan's creation.",
))

# ----------------------------------------------------------------- events
_add(TableSpec(
    "account_events", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("user_id", "id", "Account.", references="accounts.user_id"),
        _c("kind", "str", "Account activity.", values=(
            "login", "password_reset", "password_change", "email_change", "phone_change")),
        _c("device_id", "id", "Device used; must be linked to the account at the time.",
           references="devices.device_id"),
        _c("ip", "str", "IP address."),
        _c("ip_country", "str", "IP geolocation country."),
        _c("email", "str", "The new address of an email_change (null for other kinds).",
           nullable=True),
    ),
    "Logins and credential changes. Device and address additions are the link tables.",
))
_add(TableSpec(
    "order_attempts", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("order_id", "id", "Order id, increasing with occurred_at."),
        _c("user_id", "id", "Ordering account.", references="accounts.user_id"),
        _c("merchant_id", "id", "Merchant.", references="merchants.merchant_id"),
        _c("device_id", "id", "Device used.", references="devices.device_id"),
        _c("card_id", "id", "Card charged at checkout.", references="cards.card_id"),
        _c("ship_address_id", "id", "Delivery address.", references="addresses.address_id"),
        _c("amount_cents", "cents", "Merchant's price."),
        _c("promo_id", "id", "Promotion used.", nullable=True, references="promotions.promo_id"),
        _c("promo_discount_cents", "cents", "Platform-funded discount (0 without promotion)."),
        _c("ip", "str", "Checkout IP address."),
        _c("ip_country", "str", "IP geolocation country."),
        _c("avs_result", "str", "Address verification: Y match, N mismatch.", values=("Y", "N")),
        _c("cvv_result", "str", "Card verification: M match, N mismatch.", values=("M", "N")),
        _c("processor_result", "str", "Processor decision; only processor declines happen "
           "inside the world.", values=("approved", "declined")),
    ),
    "Every attempted checkout, with its potential outcomes under approve-all; known at "
    "the attempt (known_at = occurred_at).",
))
_add(TableSpec(
    "payment_attempts", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("plan_id", "id", "Plan.", references="plans.plan_id"),
        _c("seq", "int", "Schedule entry being paid."),
        _c("attempt_no", "int", "1 for the first try, then retries."),
        _c("amount_cents", "cents", "Amount attempted."),
        _c("result", "str", "Collection result.", values=("success", "failed")),
    ),
    "Collection attempts; the down payment is seq 0 at approval.",
))
_add(TableSpec(
    "fulfilments", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("order_id", "id", "Order shipped.", references="order_attempts.order_id"),
    ),
    "Merchant reports shipment; the merchant is settled then.",
))
_add(TableSpec(
    "deliveries", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("order_id", "id", "Order delivered.", references="order_attempts.order_id"),
    ),
    "Carrier-confirmed delivery. A shipped order without one was not delivered.",
))
_add(TableSpec(
    "payment_reversals", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("payment_event_id", "id", "The successful payment attempt reversed."),
        _c("plan_id", "id", "Plan.", references="plans.plan_id"),
        _c("amount_cents", "cents", "Amount reversed (positive)."),
        _c("reason", "str", "Why the collection came back.",
           values=("bank_return", "card_reversal")),
    ),
    "Collections that bounced after succeeding.",
))
_add(TableSpec(
    "dispute_openings", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("dispute_id", "id", "Dispute id, increasing with known_at."),
        _c("order_id", "id", "Disputed order.", references="order_attempts.order_id"),
        _c("reason", "str", "Reason given by the customer.", values=DISPUTE_REASONS),
        _c("amount_cents", "cents", "Collected payments disputed (positive)."),
    ),
    "occurred_at: the customer files; known_at: the platform is notified.",
))
_add(TableSpec(
    "dispute_resolutions", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("dispute_id", "id", "Dispute.", references="dispute_openings.dispute_id"),
        _c("outcome", "str", "won: resolved for the platform (against the customer); "
           "lost: for the customer.", values=("won", "lost")),
    ),
    "A dispute without a resolution is pending.",
))
_add(TableSpec(
    "victim_reports", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("user_id", "id", "Account owner reporting.", references="accounts.user_id"),
        _c("order_id", "id", "Order the owner says they did not place.",
           references="order_attempts.order_id"),
    ),
    "Account owner disowns orders after a takeover; one row per disowned order.",
))
_add(TableSpec(
    "plan_writeoffs", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("plan_id", "id", "Plan.", references="plans.plan_id"),
        _c("outstanding_cents", "cents", "Unpaid balance written off (a status, not cash)."),
    ),
    "Plans written off after the last installment stays unpaid.",
))
_add(TableSpec(
    "cash_events", "observable", "event", ("event_id",), (
        *_event_columns(),
        _c("order_id", "id", "Order.", references="order_attempts.order_id"),
        _c("plan_id", "id", "Plan.", references="plans.plan_id"),
        _c("merchant_id", "id", "Merchant.", references="merchants.merchant_id"),
        _c("kind", "str", "Cash movement (core.ledger.CASH_KINDS).", values=CASH_KINDS),
        _c("amount_cents", "cents", "Signed cents from the platform's point of view."),
        _c("ref_event_id", "id", "World event that caused it (for action-caused cash, the "
           "payment it compensates)."),
        _c("cause", "str", "'natural', or the replay action that produced it."),
    ),
    "The platform's cash ledger (core.ledger). Loss over a horizon = -sum(amount_cents).",
))

# ------------------------------------------------------------ latent truth
_add(TableSpec(
    "latent_episodes", "latent", "truth", ("episode_id",), (
        _c("episode_id", "id", "Episode id."),
        _c("pattern_id", "str", "Fraud pattern.", values=PATTERNS),
        _c("started_at", "ts", "First event of the episode."),
        _c("ended_at", "ts", "Last event of the episode.", nullable=True),
    ),
    "Simulator-only: fraud episodes (one actor or ring acting together).",
))
_add(TableSpec(
    "latent_accounts", "latent", "truth", ("user_id",), (
        _c("user_id", "id", "Account.", references="accounts.user_id"),
        _c("actor", "str", "Who controls the account.",
           values=("legitimate", "fraudster", "synthetic_identity")),
        _c("episode_id", "id", "Episode the account belongs to.", nullable=True,
           references="latent_episodes.episode_id"),
        _c("profile", "str", "Behaviour profile or benign mimic (traveller, mover, household, "
           "hardship, new_customer, sleeper, ...).", nullable=True),
    ),
    "Simulator-only: account truth. A taken-over account stays legitimate here.",
))
_add(TableSpec(
    "latent_orders", "latent", "truth", ("order_id",), (
        _c("order_id", "id", "Order.", references="order_attempts.order_id"),
        _c("pattern_id", "str", "Fraud pattern; null for legitimate orders.", nullable=True,
           values=PATTERNS),
        _c("episode_id", "id", "Episode.", nullable=True,
           references="latent_episodes.episode_id"),
        _c("intent", "str", "Intent behind the order.", values=("legitimate", "fraud", "abuse")),
        _c("mimic", "str", "Benign behaviour that resembles fraud, if any.", nullable=True),
    ),
    "Simulator-only: order truth. Repaid ring warm-ups stay episode members.",
))

# --------------------------------------------------------------- labels
_add(TableSpec(
    "labels", "adjudicated", "label", ("order_id", "label_known_at"), (
        _c("order_id", "id", "Processor-approved order.", references="order_attempts.order_id"),
        _c("label", "int", "1 = abusive order as determined, 0 = not."),
        _c("basis", "str", "Determination behind the label.",
           values=LABEL_POSITIVE_BASES + LABEL_NEGATIVE_BASES),
        _c("label_known_at", "ts", "When the determination was known."),
    ),
    "Adjudicated label history from core.world.adjudicate: at most one negative row "
    "(known when the horizon has passed) and one positive row (the first determination). "
    "An order has no label before its first row; use labels_as_of().",
))

EVENT_RANK: dict[str, int] = {
    "account_events": 10,
    "order_attempts": 20,
    "payment_attempts": 30,
    "fulfilments": 40,
    "deliveries": 50,
    "payment_reversals": 60,
    "dispute_openings": 70,
    "dispute_resolutions": 80,
    "victim_reports": 90,
    "plan_writeoffs": 100,
    "cash_events": 110,
}
EVENT_TABLES = tuple(EVENT_RANK)
ENTITY_TABLES = tuple(name for name, spec in TABLES.items() if spec.role == "entity")
LATENT_TABLES = tuple(name for name, spec in TABLES.items() if spec.layer == "latent")
OBSERVABLE_TABLES = tuple(name for name, spec in TABLES.items() if spec.layer == "observable")

# Entity ids that must increase with creation, beyond each spec's own id_column.
ORDERED_IDS = (
    ("order_attempts", "order_id", "occurred_at"),
    ("dispute_openings", "dispute_id", "known_at"),
)


# ============================================================ coercion / IO
class SchemaError(ValueError):
    """A table does not match its specification."""


def coerce(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with exactly the spec's columns, in order, with spec dtypes."""
    spec = TABLES[name]
    missing = [c for c in spec.column_names if c not in frame.columns]
    extra = [c for c in frame.columns if c not in spec.column_names]
    if missing or extra:
        raise SchemaError(f"{name}: missing columns {missing}, unexpected columns {extra}")
    out = pd.DataFrame(index=frame.index)
    for column in spec.columns:
        values = frame[column.name]
        try:
            if column.type == "ts":
                if values.dtype.kind != "M":
                    values = pd.to_datetime(values, format=TS_FORMAT)
                values = values.astype("datetime64[s]")
            elif column.type == "bool":
                if values.dtype != bool:
                    text = values.astype("str").str.lower()
                    if not text.isin(["true", "false", "1", "0"]).all():
                        raise ValueError("not a boolean")
                    values = text.isin(["true", "1"])
                values = values.astype(bool)
            elif column.type == "str":
                values = values.astype("str")
            else:
                if column.type in INTEGER_TYPES and _has_fraction(values):
                    raise ValueError("fractional values in an integer column")
                values = values.astype(column.dtype)
                if values.dtype.kind == "f" and np.isinf(values.to_numpy(dtype=float)).any():
                    raise ValueError("infinite values")
        except (ValueError, TypeError, OverflowError) as error:
            raise SchemaError(f"{name}.{column.name}: {error}") from error
        out[column.name] = values
    return out.reset_index(drop=True)


def _has_fraction(values: pd.Series) -> bool:
    """Whether any value has a fractional part, judged at its own precision."""
    if values.dtype.kind in "iub":
        return False
    if values.dtype.kind == "f":
        return bool((values.notna() & (values % 1 != 0)).any())
    for value in values.dropna():
        try:
            number = value if isinstance(value, Decimal) else Decimal(str(value))
        except InvalidOperation:
            continue  # not a number: the cast reports it
        if number.is_finite() and number != number.to_integral_value():
            return True
    return False


def empty(name: str) -> pd.DataFrame:
    """An empty table with the spec's columns and dtypes."""
    spec = TABLES[name]
    return pd.DataFrame({c.name: pd.Series(dtype=c.dtype) for c in spec.columns})


def canonical_csv(name: str, frame: pd.DataFrame) -> bytes:
    """Deterministic CSV bytes: spec column order, rows sorted by key, fixed formats."""
    spec = TABLES[name]
    table = coerce(name, frame).sort_values(list(spec.key), kind="stable")
    out = pd.DataFrame(index=table.index)
    for column in spec.columns:
        values = table[column.name]
        if column.type == "ts":
            out[column.name] = values.dt.strftime(TS_FORMAT).where(values.notna(), "")
        elif column.type == "bool":
            out[column.name] = values.astype(int)
        else:
            out[column.name] = values
    buffer = io.StringIO()
    out.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode()


def read_table(path: Path, name: str) -> pd.DataFrame:
    spec = TABLES[name]
    dtypes = {
        c.name: ("str" if c.type in ("ts", "bool") else c.dtype)
        for c in spec.columns
    }
    frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""])
    for column in spec.columns:
        if column.type == "ts":
            frame[column.name] = pd.to_datetime(frame[column.name], format=TS_FORMAT)
    return coerce(name, frame)


def read_world(
    directory: str | Path, names: Iterable[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Read ``<name>.csv`` for each table (all specified tables by default)."""
    root = Path(directory)
    return {name: read_table(root / f"{name}.csv", name) for name in (names or TABLES)}


def write_world(tables: Mapping[str, pd.DataFrame], directory: str | Path) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        (root / f"{name}.csv").write_bytes(canonical_csv(name, frame))


def _map(keys: pd.Series, mapping: Mapping[Any, Any] | pd.Series) -> pd.Series:
    """``keys.map(mapping)``, also when the mapping is empty."""
    if not isinstance(mapping, pd.Series):
        mapping = pd.Series(mapping)
    if mapping.empty:
        kind = mapping.dtype.kind
        return pd.Series(index=keys.index, dtype=mapping.dtype if kind in "Mf" else "object")
    return keys.map(mapping)


# ================================================================ manifest
def table_sha256(name: str, frame: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_csv(name, frame)).hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def build_manifest(
    tables: Mapping[str, pd.DataFrame],
    *,
    generator_version: str,
    config: Mapping[str, Any],
    seed: int,
    family: str,
    order_start: str,
    order_end: str,
    observed_until: str,
    bustout_merchant_ids: Iterable[int] = (),
    code_commit: str | None = None,
) -> dict[str, Any]:
    """World identity: what generated it and a hash and row count per table.

    ``order_start``/``order_end`` bound order attempts (end exclusive);
    ``observed_until`` is the end of the outcome follow-up. Bust-out merchant
    ids are for evaluation only and must never reach a decision.
    """
    return {
        "generator_version": generator_version,
        "config_sha256": config_sha256(config),
        "seed": int(seed),
        "family": family,
        "horizon": {
            "order_start": order_start,
            "order_end": order_end,
            "observed_until": observed_until,
        },
        "code_commit": code_commit,
        "evaluation_only": {"bustout_merchant_ids": sorted(int(m) for m in bustout_merchant_ids)},
        "tables": {
            name: {"rows": len(frame), "sha256": table_sha256(name, frame)}
            for name, frame in sorted(tables.items())
        },
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_manifest(tables: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> None:
    """Raise if any table's rows or hash differ from the manifest."""
    problems = []
    for name, entry in manifest["tables"].items():
        if name not in tables:
            problems.append(f"{name}: missing")
            continue
        rows, digest = len(tables[name]), table_sha256(name, tables[name])
        if rows != entry["rows"] or digest != entry["sha256"]:
            problems.append(f"{name}: rows {rows} vs {entry['rows']}, hash differs")
    extra = sorted(set(tables) - set(manifest["tables"]))
    problems += [f"{name}: not in manifest" for name in extra]
    if problems:
        raise ValueError("world does not match its manifest: " + "; ".join(problems))


# ============================================================ event order
def event_frame(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """All events stacked as (event_id, kind, rank, occurred_at, known_at), in total order."""
    parts = []
    for name, rank in EVENT_RANK.items():
        if name not in tables:
            continue
        frame = tables[name]
        parts.append(pd.DataFrame({
            "event_id": frame["event_id"].astype("Int64"),
            "kind": name,
            "rank": rank,
            "occurred_at": frame["occurred_at"],
            "known_at": frame["known_at"],
        }))
    if not parts:
        return pd.DataFrame(columns=["event_id", "kind", "rank", "occurred_at", "known_at"])
    stacked = pd.concat(parts, ignore_index=True)
    return sort_events(stacked)


def sort_events(events: pd.DataFrame) -> pd.DataFrame:
    """Sort by the total order ``(known_at, kind rank, event_id)``.

    ``events`` needs ``known_at``, ``event_id`` and either ``rank`` or ``kind``
    (a table name in :data:`EVENT_RANK`).
    """
    frame = events
    if "rank" not in frame.columns:
        frame = frame.assign(rank=_map(frame["kind"], EVENT_RANK))
    return frame.sort_values(["known_at", "rank", "event_id"], kind="stable").reset_index(
        drop=True
    )


def renumber_events(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Assign final event ids 1..N in the total order, remapping references.

    Input event ids are provisional: unique across event tables, or null for
    cash events (as :func:`core.ledger.derive_cash_events` returns them), which
    get provisional ids after every other event, in their row order. Ties in
    ``(known_at, kind rank)`` keep the provisional order, so a generator should
    make provisional ids carry no pattern information either (for example by
    drawing them at random). ``payment_reversals.payment_event_id`` and
    ``cash_events.ref_event_id`` are remapped.
    """
    out = {name: frame.copy() for name, frame in tables.items()}
    if "cash_events" in out and out["cash_events"]["event_id"].isna().any():
        cash = out["cash_events"]
        others = [out[n]["event_id"].max() for n in EVENT_TABLES
                  if n in out and n != "cash_events" and len(out[n])]
        known = [v for v in [*others, cash["event_id"].max()] if pd.notna(v)]
        start = int(max(known, default=0)) + 1
        missing = cash["event_id"].isna().to_numpy()
        ids = cash["event_id"].astype("Int64").copy()
        ids[missing] = np.arange(start, start + missing.sum())
        cash["event_id"] = ids.astype("int64")
    events = event_frame(out)
    if events["event_id"].duplicated().any():
        raise ValueError("provisional event ids must be unique across event tables")
    new_id = pd.Series(np.arange(1, len(events) + 1, dtype="int64"),
                       index=events["event_id"].astype("int64").to_numpy())
    for name in EVENT_TABLES:
        if name in out:
            out[name]["event_id"] = out[name]["event_id"].astype("int64").map(new_id)
    if "payment_reversals" in out:
        reversals = out["payment_reversals"]
        reversals["payment_event_id"] = reversals["payment_event_id"].astype("int64").map(new_id)
    if "cash_events" in out:
        cash = out["cash_events"]
        cash["ref_event_id"] = cash["ref_event_id"].astype("int64").map(new_id)
    return out


# ================================================================== labels
def adjudicate(
    tables: Mapping[str, pd.DataFrame],
    *,
    horizon_days: int,
    observed_until: pd.Timestamp | str,
    default_grace_days: int = 30,
    linked_plan_days: int = 7,
    shared_default_days: int = 30,
    promo_linked_accounts: int = 3,
    promo_quiet_days: int = 90,
) -> pd.DataFrame:
    """Adjudicated labels from observable outcomes only (the fraud policy's determinations).

    Positive determinations, each labelling the named orders when it becomes known:

    * ``third_party_fraud``: an unauthorized-use dispute lost by the platform, known
      at its resolution;
    * ``account_takeover``: a victim report on the order, known at the report;
    * ``never_pay``: a zero-effort default with an intent marker. A zero-effort
      default is a plan whose first installment after the checkout payment (seq 1)
      is still unpaid ``default_grace_days`` after its due date, with no payment on
      the plan after the checkout payment (seq 0 never counts). The markers: another
      plan of the account opened within ``linked_plan_days`` is also a zero-effort
      default; or the account shares a device, a shipping address or a normalized
      email with two or more other accounts whose plans became zero-effort defaults
      within ``shared_default_days`` of it. Known when the last qualifying default
      (and the sharing) is known; final, so later payments are recoveries. An
      unmarked zero-effort default is a credit loss;
    * ``inr_abuse``: the account's second item-not-received dispute resolved against
      the customer, counting only orders with a carrier-confirmed delivery; each claim
      counts from the later of its resolution and its delivery, both orders are known
      when the second claim counts, later ones when their own does;
    * ``promo_abuse``: a use of a first-purchase promotion when
      ``promo_linked_accounts - 1`` or more other accounts sharing a device or a
      normalized email with its account (not an address: households share addresses)
      used the same promotion within ``promo_quiet_days`` of it, and none of these
      accounts ordered without a promotion within ``promo_quiet_days`` after its own
      use; known that many days after the latest of these uses (or when the sharing is
      known, if later), taking the other accounts that make it known earliest;
    * ``merchant_bustout``: an item-not-received dispute resolved for the customer on
      an order the merchant reported shipped, with no carrier-confirmed delivery,
      where the merchant closed at or before the resolution; known at the latest of
      the resolution, the closure and the shipment report (no delivery known by then).

    Every determination is dated by the last fact it rests on.

    Negative: known at ``order time + horizon_days`` when no positive is known by
    then, postponed to the resolution of any dispute opened by that time;
    with a dispute still pending at ``observed_until`` the order stays unknown. The
    basis is ``credit_loss`` when a default or write-off is known by then, else
    ``no_finding``. A positive determined later adds a second row.

    Only rows known by ``observed_until`` are returned. Latent tables are never read.
    """
    observed_until = pd.Timestamp(observed_until)
    horizon = pd.Timedelta(days=horizon_days)
    orders = tables["order_attempts"]
    orders = orders[orders["processor_result"] == "approved"][
        ["order_id", "user_id", "merchant_id", "occurred_at", "promo_id"]
    ]
    plans = tables["plans"][["plan_id", "order_id", "created_at"]].merge(
        orders[["order_id", "user_id"]], on="order_id", how="inner"
    )
    positives: list[pd.DataFrame] = []

    def found(frame: pd.DataFrame, basis: str) -> None:
        positives.append(pd.DataFrame({
            "order_id": frame["order_id"].astype("int64"),
            "basis": basis,
            "label_known_at": frame["label_known_at"].astype("datetime64[s]"),
        }))

    openings = tables["dispute_openings"]
    resolutions = tables["dispute_resolutions"]
    disputes = openings[["dispute_id", "order_id", "reason", "known_at"]].rename(
        columns={"known_at": "opened_known_at"}
    ).merge(
        resolutions[["dispute_id", "outcome", "known_at"]].rename(
            columns={"known_at": "resolved_known_at"}
        ),
        on="dispute_id",
        how="left",
    )

    # (a) unauthorized-use dispute lost by the platform
    lost = disputes[(disputes["reason"] == "unauthorized") & (disputes["outcome"] == "lost")]
    found(lost.rename(columns={"resolved_known_at": "label_known_at"}), "third_party_fraud")

    # (b) victim report
    reports = tables["victim_reports"]
    found(reports.rename(columns={"known_at": "label_known_at"}), "account_takeover")

    # (c) never-pay: a zero-effort default with an intent marker
    found(never_pay_determinations(
        tables, orders, default_grace_days=default_grace_days,
        linked_plan_days=linked_plan_days, shared_default_days=shared_default_days,
    ), "never_pay")

    # (d) INR claims resolved against the customer, second and later per account
    delivered = tables["deliveries"].groupby("order_id")["known_at"].min()
    rejected = disputes[
        (disputes["reason"] == "item_not_received") & (disputes["outcome"] == "won")
    ].merge(orders[["order_id", "user_id"]], on="order_id")
    rejected["delivered_at"] = _map(rejected["order_id"], delivered)
    rejected = rejected[rejected["delivered_at"].notna()]
    # each claim counts once both its resolution and its delivery are known
    rejected["ready"] = rejected[["resolved_known_at", "delivered_at"]].max(axis=1)
    rejected = rejected.sort_values(["user_id", "ready", "dispute_id"])
    rejected["nth"] = rejected.groupby("user_id").cumcount() + 1
    second = rejected[rejected["nth"] == 2].set_index("user_id")["ready"]
    rejected["second_at"] = _map(rejected["user_id"], second)
    inr = rejected[rejected["second_at"].notna()].copy()
    inr["label_known_at"] = inr[["ready", "second_at"]].max(axis=1)
    found(inr, "inr_abuse")

    # (e) promotion farming across accounts sharing a device or an email
    found(_promo_abuse(tables, orders, promo_linked_accounts, promo_quiet_days), "promo_abuse")

    # (f) merchant bust-out: shipped, never delivered, claim upheld after the merchant closed
    shipped = tables["fulfilments"].groupby("order_id")["known_at"].min()
    closed_at = tables["merchants"].set_index("merchant_id")["closed_at"]
    upheld = disputes[(disputes["reason"] == "item_not_received")
                      & (disputes["outcome"] == "lost")].merge(
        orders[["order_id", "merchant_id"]], on="order_id"
    ).merge(resolutions[["dispute_id", "occurred_at"]], on="dispute_id")
    upheld["closed_at"] = _map(upheld["merchant_id"], closed_at)
    upheld["shipped_at"] = _map(upheld["order_id"], shipped)
    upheld["label_known_at"] = upheld[["resolved_known_at", "closed_at", "shipped_at"]].max(
        axis=1)
    bustout = upheld[
        upheld["shipped_at"].notna()
        & ~(_map(upheld["order_id"], delivered) <= upheld["label_known_at"])
        & (upheld["closed_at"] <= upheld["occurred_at"])
    ]
    found(bustout, "merchant_bustout")

    basis_order = {basis: i for i, basis in enumerate(LABEL_POSITIVE_BASES)}
    positive = pd.concat(positives, ignore_index=True)
    positive = positive[positive["order_id"].isin(orders["order_id"])]
    positive = positive.assign(_rank=_map(positive["basis"], basis_order))
    positive = positive.sort_values(["order_id", "label_known_at", "_rank"]).drop_duplicates(
        "order_id"
    ).drop(columns="_rank")
    positive["label"] = 1

    # Negatives
    negative = orders[["order_id", "occurred_at"]].copy()
    negative["label_known_at"] = negative["occurred_at"] + horizon
    pending = disputes.merge(negative[["order_id", "label_known_at"]], on="order_id")
    pending = pending[pending["opened_known_at"] <= pending["label_known_at"]]
    unresolved = set(pending.loc[pending["resolved_known_at"].isna(), "order_id"])
    resolved = _map(negative["order_id"], pending.groupby("order_id")["resolved_known_at"].max())
    later = resolved > negative["label_known_at"]
    negative["label_known_at"] = negative["label_known_at"].where(~later, resolved)
    negative = negative[~negative["order_id"].isin(unresolved)]
    first_positive = positive.set_index("order_id")["label_known_at"]
    pos_at = _map(negative["order_id"], first_positive)
    negative = negative[~(pos_at <= negative["label_known_at"])]

    zero_effort = _zero_effort_defaults(tables, plans, default_grace_days)
    defaults = pd.concat([
        zero_effort[["order_id", "default_at"]].rename(columns={"default_at": "at"}),
        tables["plan_writeoffs"][["plan_id", "known_at"]].merge(
            plans[["plan_id", "order_id"]], on="plan_id"
        )[["order_id", "known_at"]].rename(columns={"known_at": "at"}),
    ])
    first_default = defaults.groupby("order_id")["at"].min()
    default_at = _map(negative["order_id"], first_default)
    negative["basis"] = np.where(
        default_at <= negative["label_known_at"], "credit_loss", "no_finding"
    )
    negative["label"] = 0

    columns = ["order_id", "label", "basis", "label_known_at"]
    labels = pd.concat([positive[columns], negative[columns]], ignore_index=True)
    labels = labels[labels["label_known_at"] <= observed_until]
    labels = labels.sort_values(["order_id", "label_known_at"], kind="stable")
    return coerce("labels", labels)


def never_pay_determinations(
    tables: Mapping[str, pd.DataFrame],
    orders: pd.DataFrame,
    *,
    default_grace_days: int,
    linked_plan_days: int,
    shared_default_days: int,
) -> pd.DataFrame:
    """Every never-pay determination (fraud policy 8.3): columns order_id, label_known_at.

    ``orders`` are the processor-approved orders to judge (columns order_id,
    user_id and occurred_at at least). A plan is never-pay when it is a
    zero-effort default with an intent marker: another plan of the account opened
    within ``linked_plan_days`` is also a zero-effort default, or two or more
    other accounts sharing a device, a shipping address or a normalized email
    with the account (:func:`shared_accounts`) have zero-effort defaults within
    ``shared_default_days`` of it. Known when the last qualifying fact is known.
    Unlike :func:`adjudicate`, which keeps an order's first determination, this
    returns every order with a never-pay determination.
    """
    plans = tables["plans"][["plan_id", "order_id", "created_at"]].merge(
        orders[["order_id", "user_id"]], on="order_id", how="inner"
    )
    zero_effort = _zero_effort_defaults(tables, plans, default_grace_days)
    marked = []
    same_account = zero_effort.merge(
        zero_effort[["plan_id", "user_id", "created_at", "default_at"]],
        on="user_id", suffixes=("", "_other"),
    )
    same_account = same_account[
        (same_account["plan_id"] != same_account["plan_id_other"])
        & ((same_account["created_at"] - same_account["created_at_other"]).abs()
           <= pd.Timedelta(days=linked_plan_days))
    ]
    marked.append(same_account.assign(
        at=same_account[["default_at", "default_at_other"]].max(axis=1))[["order_id", "at"]])
    others = zero_effort[["plan_id", "order_id", "user_id", "default_at"]].merge(
        shared_accounts(tables, orders), on="user_id"
    ).merge(
        zero_effort[["user_id", "default_at"]].rename(
            columns={"user_id": "other_id", "default_at": "other_default_at"}),
        on="other_id",
    )
    others = others[(others["other_default_at"] - others["default_at"]).abs()
                    <= pd.Timedelta(days=shared_default_days)]
    others["at"] = others[["other_default_at", "shared_at"]].max(axis=1)
    per_other = others.groupby(["plan_id", "order_id", "default_at", "other_id"],
                               as_index=False)["at"].min()
    per_other = per_other.sort_values(["plan_id", "at", "other_id"], kind="stable")
    per_other["nth"] = per_other.groupby("plan_id").cumcount() + 1
    second = per_other[per_other["nth"] == 2]
    marked.append(second.assign(at=second[["at", "default_at"]].max(axis=1))[["order_id", "at"]])
    never_pay = pd.concat(marked, ignore_index=True).groupby("order_id", as_index=False)["at"].min()
    return pd.DataFrame({
        "order_id": never_pay["order_id"].astype("int64").to_numpy(),
        "label_known_at": never_pay["at"].astype("datetime64[s]").to_numpy(),
    })


def _zero_effort_defaults(
    tables: Mapping[str, pd.DataFrame], plans: pd.DataFrame, grace_days: int
) -> pd.DataFrame:
    """Plans whose seq 1 is unpaid ``grace_days`` after due, with nothing paid after seq 0.

    "Nothing paid" is judged at the default: successful installment payments (seq
    >= 1) known by then, less the reversals of them known by then, total zero.
    """
    schedule = tables["installment_schedule"]
    first_due = schedule[schedule["seq"] == 1][["plan_id", "due_at"]]
    state = plans.merge(first_due, on="plan_id", how="inner")
    state["default_at"] = state["due_at"] + pd.Timedelta(days=grace_days)
    cutoff = state.set_index("plan_id")["default_at"]
    attempts = tables["payment_attempts"]
    paid = attempts[(attempts["result"] == "success") & (attempts["seq"] >= 1)]
    paid = paid[paid["known_at"] <= _map(paid["plan_id"], cutoff)]
    reversals = tables["payment_reversals"]
    reversals = reversals[reversals["payment_event_id"].isin(paid["event_id"])
                          & (reversals["known_at"] <= _map(reversals["plan_id"], cutoff))]
    standing = paid.groupby("plan_id")["amount_cents"].sum().sub(
        reversals.groupby("plan_id")["amount_cents"].sum(), fill_value=0)
    some_paid = set(standing.index[standing > 0])
    return state[~state["plan_id"].isin(some_paid)].reset_index(drop=True)


def shared_accounts(
    tables: Mapping[str, pd.DataFrame], orders: pd.DataFrame, *, shipping: bool = True
) -> pd.DataFrame:
    """Pairs of accounts that shared a device, a normalized email or (``shipping``) an
    address an approved order shipped to.

    Sharing means holding the same item at the same time: overlapping device links,
    overlapping email holdings (core.asof.email_history); a shipping address counts
    from the account's first approved order to it. Columns user_id, other_id,
    shared_at (the start of the earliest overlap).
    """
    from core.asof import email_history

    emails = email_history(tables["accounts"], tables["account_events"])
    emails = emails.rename(columns={"since": "start", "until": "end"}).assign(
        item="e" + emails["email_root"].astype("str"))
    links = tables["device_links"]
    devices = links.rename(columns={"created_at": "start", "removed_at": "end"}).assign(
        item="d" + links["device_id"].astype(str))
    frames = [emails, devices]
    if shipping:
        shipped = orders[["order_id", "user_id", "occurred_at"]].merge(
            tables["order_attempts"][["order_id", "ship_address_id"]], on="order_id")
        shipped = shipped.groupby(["user_id", "ship_address_id"],
                                  as_index=False)["occurred_at"].min()
        frames.append(shipped.rename(columns={"occurred_at": "start"}).assign(
            end=pd.NaT, item="a" + shipped["ship_address_id"].astype(str)))
    items = pd.concat([frame[["user_id", "item", "start", "end"]] for frame in frames],
                      ignore_index=True)
    pairs = items.merge(items, on="item", suffixes=("", "_other"))
    pairs = pairs[pairs["user_id"] != pairs["user_id_other"]]
    shared_at = pairs[["start", "start_other"]].max(axis=1)
    ends = pairs[["end", "end_other"]].min(axis=1)  # NaT (still held) is skipped
    pairs = pairs.assign(shared_at=shared_at)[ends.isna() | (shared_at < ends)]
    return pairs.rename(columns={"user_id_other": "other_id"}).groupby(
        ["user_id", "other_id"], as_index=False)["shared_at"].min()


def _promo_abuse(
    tables: Mapping[str, pd.DataFrame], orders: pd.DataFrame, threshold: int, quiet_days: int
) -> pd.DataFrame:
    """Promotion-abuse uses of first-purchase promotions and when each is known.

    See ``adjudicate``: each use needs ``threshold - 1`` other accounts sharing a
    device or email whose uses fall within ``quiet_days`` of it, all of them quiet
    (no order without a promotion) for ``quiet_days`` after their own use.
    """
    promotions = tables["promotions"]
    first_purchase = set(promotions.loc[promotions["first_purchase_only"], "promo_id"])
    uses = orders[orders["promo_id"].isin(first_purchase)][
        ["order_id", "user_id", "occurred_at", "promo_id"]
    ]
    quiet = pd.Timedelta(days=quiet_days)
    plain = orders.loc[orders["promo_id"].isna(), ["user_id", "occurred_at"]]
    later = uses.merge(plain.rename(columns={"occurred_at": "plain_at"}), on="user_id")
    loud = later[(later["plain_at"] > later["occurred_at"])
                 & (later["plain_at"] <= later["occurred_at"] + quiet)]
    uses = uses[~uses["order_id"].isin(loud["order_id"])]
    pairs = uses.merge(shared_accounts(tables, orders, shipping=False), on="user_id").merge(
        uses.rename(columns={"order_id": "other_order", "user_id": "other_id",
                             "occurred_at": "other_at"}),
        on=["other_id", "promo_id"],
    )
    pairs = pairs[(pairs["other_at"] - pairs["occurred_at"]).abs() <= quiet]
    # each other account's use counts from when it and the sharing are both known
    pairs = pairs.assign(ready=pd.concat(
        [pairs["other_at"] + quiet, pairs["shared_at"]], axis=1).max(axis=1))
    per_other = pairs.groupby(["order_id", "occurred_at", "other_id"],
                              as_index=False)["ready"].min()
    per_other = per_other.sort_values(["order_id", "ready", "other_id"], kind="stable")
    per_other["nth"] = per_other.groupby("order_id").cumcount() + 1
    enough = per_other[per_other["nth"] == threshold - 1]
    known = pd.concat([enough["occurred_at"] + quiet, enough["ready"]], axis=1).max(axis=1)
    return pd.DataFrame({"order_id": enough["order_id"].to_numpy(),
                         "label_known_at": known.to_numpy()})


def labels_as_of(labels: pd.DataFrame, at: pd.Timestamp | str) -> pd.DataFrame:
    """The label of each order as known at ``at`` (orders without a row are unknown)."""
    at = pd.Timestamp(at)
    known = labels[labels["label_known_at"] <= at]
    known = known.sort_values(["order_id", "label_known_at"], kind="stable")
    return known.drop_duplicates("order_id", keep="last").reset_index(drop=True)


# ============================================================== validation
@dataclass
class Violation:
    check: str
    table: str
    count: int
    examples: list[Any] = field(default_factory=list)
    message: str = ""

    def __str__(self) -> str:
        shown = ", ".join(str(e) for e in self.examples[:5])
        return f"{self.check} [{self.table}] x{self.count}: {self.message} (e.g. {shown})"


class WorldError(ValueError):
    """The world violates its contract; ``violations`` lists every failed check."""

    def __init__(self, violations: list[Violation]):
        self.violations = violations
        super().__init__(
            f"{len(violations)} world contract violation(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    @property
    def checks(self) -> set[str]:
        return {v.check for v in self.violations}


class _Checker:
    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def flag(self, check: str, table: str, bad: pd.DataFrame, ident: str, message: str) -> None:
        if len(bad):
            examples = bad[ident].head(5).tolist() if ident in bad.columns else []
            self.violations.append(Violation(check, table, len(bad), examples, message))


def check_world(
    tables: Mapping[str, pd.DataFrame],
    *,
    require_all: bool = True,
    terms: ledger.ProductTerms | None = None,
) -> list[Violation]:
    """Every contract violation in ``tables`` (an empty list when the world is valid).

    ``terms`` are the product terms the cash events must follow (default:
    ``config/world.yaml``).
    """
    checker = _Checker()
    world: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        if name not in tables:
            if require_all:
                checker.violations.append(Violation("missing_table", name, 1, [], "absent"))
            world[name] = empty(name)
            continue
        try:
            world[name] = coerce(name, tables[name])
        except SchemaError as error:
            checker.violations.append(Violation("schema", name, 1, [], str(error)))
            world[name] = empty(name)
    for name in tables:
        if name not in TABLES:
            checker.violations.append(Violation("unknown_table", name, 1, [], "not specified"))

    _check_columns(checker, world)
    _check_references(checker, world)
    _check_ids(checker, world)
    _check_entities(checker, world)
    _check_orders(checker, world)
    _check_processes(checker, world)
    _check_cash(checker, world, terms or ledger.ProductTerms.from_config())
    _check_labels(checker, world)
    return checker.violations


def validate_world(
    tables: Mapping[str, pd.DataFrame],
    *,
    require_all: bool = True,
    terms: ledger.ProductTerms | None = None,
) -> None:
    """Raise :class:`WorldError` listing every violation; return None for a valid world."""
    violations = check_world(tables, require_all=require_all, terms=terms)
    if violations:
        raise WorldError(violations)


def _ident(spec: TableSpec) -> str:
    return spec.key[0]


def _check_columns(c: _Checker, w: dict[str, pd.DataFrame]) -> None:
    for name, spec in TABLES.items():
        frame = w[name]
        ident = _ident(spec)
        duplicated = frame[frame.duplicated(list(spec.key), keep=False)]
        c.flag("duplicate_key", name, duplicated, ident, f"key {spec.key} not unique")
        for column in spec.columns:
            values = frame[column.name]
            if not column.nullable:
                c.flag("null_value", name, frame[values.isna()], ident,
                       f"{column.name} must not be null")
            if column.values is not None:
                bad = values.notna() & ~values.isin(column.values)
                c.flag("bad_value", name, frame[bad], ident,
                       f"{column.name} outside {column.values}")
        if spec.role == "event":
            c.flag("known_before_occurred", name, frame[frame["known_at"] < frame["occurred_at"]],
                   ident, "known_at earlier than occurred_at")
        for start, end in (("created_at", "removed_at"), ("created_at", "closed_at"),
                           ("started_at", "ended_at")):
            if start in frame.columns and end in frame.columns:
                c.flag("interval_reversed", name, frame[frame[end] < frame[start]], ident,
                       f"{end} earlier than {start}")
    promotions = w["promotions"]
    c.flag("interval_reversed", "promotions",
           promotions[promotions["valid_to"] <= promotions["valid_from"]], "promo_id",
           "valid_to not after valid_from")


def _check_references(c: _Checker, w: dict[str, pd.DataFrame]) -> None:
    for name, spec in TABLES.items():
        frame = w[name]
        for column in spec.columns:
            if column.references is None:
                continue
            target_table, target_column = column.references.split(".")
            values = frame[column.name]
            known = values.isna() | values.isin(w[target_table][target_column])
            c.flag("missing_reference", name, frame[~known], _ident(spec),
                   f"{column.name} not in {column.references}")
    reversals = w["payment_reversals"]
    c.flag("missing_reference", "payment_reversals",
           reversals[~reversals["payment_event_id"].isin(w["payment_attempts"]["event_id"])],
           "event_id", "payment_event_id not in payment_attempts.event_id")
    attempts = w["payment_attempts"]
    scheduled = attempts.merge(w["installment_schedule"][["plan_id", "seq"]],
                               on=["plan_id", "seq"], how="left", indicator=True)
    c.flag("missing_reference", "payment_attempts", scheduled[scheduled["_merge"] == "left_only"],
           "event_id", "(plan_id, seq) not in installment_schedule")
    all_events = set(event_frame({n: w[n] for n in EVENT_TABLES if n != "cash_events"})
                     ["event_id"].dropna().astype("int64"))
    cash = w["cash_events"]
    c.flag("missing_reference", "cash_events", cash[~cash["ref_event_id"].isin(all_events)],
           "event_id", "ref_event_id is not a world event")


def _check_ids(c: _Checker, w: dict[str, pd.DataFrame]) -> None:
    events = event_frame({name: w[name] for name in EVENT_TABLES})
    duplicated = events[events["event_id"].duplicated(keep=False)]
    c.flag("duplicate_event_id", "events", duplicated, "event_id",
           "event ids must be unique across event tables")
    by_id = events.dropna(subset=["event_id"]).sort_values("event_id", kind="stable")
    seconds = by_id["known_at"].astype("int64").to_numpy()
    order_key = seconds * 1000 + by_id["rank"].to_numpy()
    behind = np.zeros(len(order_key), dtype=bool)
    if len(order_key) > 1:
        behind[1:] = order_key[1:] < np.maximum.accumulate(order_key)[:-1]
    c.flag("ids_out_of_order", "events", by_id[behind], "event_id",
           "event ids do not follow (known_at, kind rank)")
    checks = [(n, s.id_column, s.created) for n, s in TABLES.items() if s.id_column]
    for name, id_column, time_column in [*checks, *ORDERED_IDS]:
        frame = w[name].sort_values(id_column, kind="stable")
        times = frame[time_column]
        bad = frame[times < times.cummax()]
        c.flag("ids_out_of_order", name, bad, id_column,
               f"{id_column} does not increase with {time_column}")


def _active(
    rows: pd.DataFrame, at: str, links: pd.DataFrame, on: list[str]
) -> pd.Series:
    """For each row, whether a link on ``on`` is active at ``rows[at]``."""
    if rows.empty:
        return pd.Series(dtype=bool)
    merged = rows[[*on, at]].reset_index().merge(links[[*on, "created_at", "removed_at"]],
                                                 on=on, how="left")
    ok = (merged["created_at"] <= merged[at]) & ~(merged["removed_at"] <= merged[at])
    return ok.groupby(merged["index"]).any().reindex(rows.index, fill_value=False)


def _lookup(frame: pd.DataFrame, key: str, column: str) -> pd.Series:
    return frame.drop_duplicates(key).set_index(key)[column]


def _check_entities(c: _Checker, w: dict[str, pd.DataFrame]) -> None:
    signup = _lookup(w["accounts"], "user_id", "created_at")
    for name in ("device_links", "address_links", "cards"):
        frame = w[name]
        early = frame["created_at"] < _map(frame["user_id"], signup)
        c.flag("link_before_account", name, frame[early], _ident(TABLES[name]),
               "linked before the account existed")
    links = w["device_links"]
    born = _map(links["device_id"], _lookup(w["devices"], "device_id", "created_at"))
    c.flag("link_before_entity", "device_links", links[links["created_at"] < born], "user_id",
           "linked before the device was first seen")
    links = w["address_links"]
    born = _map(links["address_id"], _lookup(w["addresses"], "address_id", "created_at"))
    c.flag("link_before_entity", "address_links", links[links["created_at"] < born], "user_id",
           "linked before the address existed")
    plans = w["plans"].merge(
        w["order_attempts"][["order_id", "occurred_at", "processor_result", "amount_cents",
                             "promo_discount_cents"]],
        on="order_id", how="inner",
    )
    c.flag("plan_for_declined_order", "plans", plans[plans["processor_result"] != "approved"],
           "plan_id", "plans exist only for processor-approved orders")
    c.flag("plan_time", "plans", plans[plans["created_at"] != plans["occurred_at"]], "plan_id",
           "plan created_at must equal the order time")
    c.flag("plan_amount", "plans",
           plans[plans["principal_cents"] != plans["amount_cents"] - plans["promo_discount_cents"]],
           "plan_id", "principal must equal amount minus promotion discount")
    schedule = w["installment_schedule"].merge(
        w["plans"][["plan_id", "created_at", "principal_cents", "down_payment_cents",
                    "n_installments"]], on="plan_id", how="inner")
    c.flag("due_before_plan", "installment_schedule",
           schedule[schedule["due_at"] < schedule["created_at"]], "plan_id",
           "due before the plan existed")
    totals = schedule.groupby("plan_id").agg(
        total=("amount_cents", "sum"), entries=("seq", "count"), last=("seq", "max"),
        principal=("principal_cents", "first"), n=("n_installments", "first"),
    ).reset_index()
    bad = totals[(totals["total"] != totals["principal"]) | (totals["entries"] != totals["n"] + 1)
                 | (totals["last"] != totals["n"])]
    c.flag("schedule_mismatch", "installment_schedule", bad, "plan_id",
           "schedule must be seq 0..n summing to the principal")
    down = schedule[schedule["seq"] == 0]
    c.flag("schedule_mismatch", "installment_schedule",
           down[(down["amount_cents"] != down["down_payment_cents"])
                | (down["due_at"] != down["created_at"])], "plan_id",
           "seq 0 must be the down payment due at approval")
    approved = w["order_attempts"]
    approved = approved[approved["processor_result"] == "approved"]
    c.flag("approved_without_plan", "order_attempts",
           approved[~approved["order_id"].isin(w["plans"]["order_id"])], "order_id",
           "every processor-approved order has a plan")


def _check_orders(c: _Checker, w: dict[str, pd.DataFrame]) -> None:
    orders = w["order_attempts"]
    t = orders["occurred_at"]
    signup = _map(orders["user_id"], _lookup(w["accounts"], "user_id", "created_at"))
    c.flag("order_before_account", "order_attempts", orders[t < signup], "order_id",
           "order placed before the account existed")
    device_ok = _active(orders, "occurred_at", w["device_links"], ["user_id", "device_id"])
    c.flag("order_before_device_link", "order_attempts", orders[~device_ok], "order_id",
           "device not linked to the account at the order time")
    orders_addr = orders.rename(columns={"ship_address_id": "address_id"})
    address_ok = _active(orders_addr, "occurred_at", w["address_links"], ["user_id", "address_id"])
    c.flag("order_before_address_link", "order_attempts", orders[~address_ok], "order_id",
           "shipping address not linked to the account at the order time")
    card_ok = _active(orders, "occurred_at", w["cards"], ["user_id", "card_id"])
    c.flag("order_before_card", "order_attempts", orders[~card_ok], "order_id",
           "card not on the account at the order time")
    merchants = w["merchants"].set_index("merchant_id")
    opened = _map(orders["merchant_id"], merchants["created_at"])
    closed = _map(orders["merchant_id"], merchants["closed_at"])
    c.flag("order_outside_merchant", "order_attempts", orders[(t < opened) | (t >= closed)],
           "order_id", "order outside the merchant's trading period")
    promotions = w["promotions"].set_index("promo_id")
    with_promo = orders[orders["promo_id"].notna()]
    starts = _map(with_promo["promo_id"], promotions["valid_from"])
    ends = _map(with_promo["promo_id"], promotions["valid_to"])
    when = with_promo["occurred_at"]
    c.flag("promotion_not_valid", "order_attempts", with_promo[(when < starts) | (when >= ends)],
           "order_id", "promotion used outside its validity")
    discount = orders["promo_discount_cents"]
    c.flag("promotion_not_valid", "order_attempts",
           orders[(discount < 0) | (discount >= orders["amount_cents"])
                  | ((discount > 0) & orders["promo_id"].isna())],
           "order_id", "discount needs a promotion and must be below the amount")
    c.flag("non_positive_amount", "order_attempts", orders[orders["amount_cents"] <= 0],
           "order_id", "amount must be positive")
    c.flag("order_known_late", "order_attempts", orders[orders["known_at"] != t], "order_id",
           "an order attempt is known when it happens")
    events = w["account_events"]
    signup = _map(events["user_id"], _lookup(w["accounts"], "user_id", "created_at"))
    c.flag("account_event_before_account", "account_events", events[events["occurred_at"] < signup],
           "event_id", "account event before the account existed")
    ok = _active(events, "occurred_at", w["device_links"], ["user_id", "device_id"])
    c.flag("account_event_before_device_link", "account_events", events[~ok], "event_id",
           "device not linked to the account at the event time")
    c.flag("bad_value", "account_events",
           events[events["kind"].eq("email_change") != events["email"].notna()], "event_id",
           "an email_change carries the new address and no other kind carries one")


def _after(
    c: _Checker,
    check: str,
    table: str,
    frame: pd.DataFrame,
    time_column: str,
    parent: pd.DataFrame,
    key: str,
    parent_time: str,
    message: str,
) -> None:
    """Flag rows of ``frame`` whose ``time_column`` is before the parent's ``parent_time``."""
    parent_times = _lookup(parent, key, parent_time)
    before = frame[time_column] < _map(frame[key], parent_times)
    c.flag(check, table, frame[before], "event_id", message)


def _check_processes(c: _Checker, w: dict[str, pd.DataFrame]) -> None:
    orders = w["order_attempts"]
    approved = orders[orders["processor_result"] == "approved"]
    for name in ("fulfilments", "deliveries", "dispute_openings", "victim_reports"):
        frame = w[name]
        c.flag("event_for_unapproved_order", name,
               frame[~frame["order_id"].isin(approved["order_id"])], "event_id",
               "only processor-approved orders have outcomes")
    plans = w["plans"]
    attempts = w["payment_attempts"]
    _after(c, "payment_before_plan", "payment_attempts", attempts, "occurred_at", plans,
           "plan_id", "created_at", "payment attempted before the plan existed")
    _after(c, "fulfilment_before_order", "fulfilments", w["fulfilments"], "occurred_at", orders,
           "order_id", "occurred_at", "shipped before the order")
    deliveries = w["deliveries"]
    shipped = _lookup(w["fulfilments"], "order_id", "occurred_at")
    shipped_at = _map(deliveries["order_id"], shipped)
    c.flag("delivery_before_fulfilment", "deliveries",
           deliveries[shipped_at.isna() | (deliveries["occurred_at"] < shipped_at)], "event_id",
           "delivered without or before shipment")
    c.flag("duplicate_outcome", "fulfilments",
           w["fulfilments"][w["fulfilments"]["order_id"].duplicated(keep=False)], "event_id",
           "an order ships once")
    c.flag("duplicate_outcome", "deliveries",
           deliveries[deliveries["order_id"].duplicated(keep=False)], "event_id",
           "an order is delivered once")
    reversals = w["payment_reversals"]
    paid = attempts.set_index("event_id")
    c.flag("reversal_of_failed_payment", "payment_reversals",
           reversals[_map(reversals["payment_event_id"], paid["result"]) != "success"],
           "event_id", "only successful payments can be reversed")
    c.flag("reversal_before_payment", "payment_reversals",
           reversals[reversals["occurred_at"]
                     < _map(reversals["payment_event_id"], paid["occurred_at"])],
           "event_id", "reversed before the payment")
    openings = w["dispute_openings"]
    _after(c, "dispute_before_order", "dispute_openings", openings, "occurred_at", orders,
           "order_id", "occurred_at", "dispute filed before the order")
    resolutions = w["dispute_resolutions"]
    opened = openings.set_index("dispute_id")
    c.flag("resolution_before_opening", "dispute_resolutions",
           resolutions[resolutions["known_at"]
                       < _map(resolutions["dispute_id"], opened["known_at"])],
           "event_id", "resolved before the platform knew of the dispute")
    c.flag("resolution_before_opening", "dispute_resolutions",
           resolutions[resolutions["occurred_at"]
                       < _map(resolutions["dispute_id"], opened["occurred_at"])],
           "event_id", "resolved before the dispute was filed")
    c.flag("duplicate_outcome", "dispute_resolutions",
           resolutions[resolutions["dispute_id"].duplicated(keep=False)], "event_id",
           "a dispute resolves once")
    reports = w["victim_reports"]
    _after(c, "report_before_order", "victim_reports", reports, "occurred_at", orders, "order_id",
           "occurred_at", "order disowned before it was placed")
    owner = _map(reports["order_id"], _lookup(orders, "order_id", "user_id"))
    c.flag("report_owner_mismatch", "victim_reports", reports[owner != reports["user_id"]],
           "event_id", "victims report orders on their own account")
    writeoffs = w["plan_writeoffs"]
    last_due = w["installment_schedule"].groupby("plan_id")["due_at"].max()
    c.flag("writeoff_before_due", "plan_writeoffs",
           writeoffs[writeoffs["occurred_at"] < _map(writeoffs["plan_id"], last_due)], "event_id",
           "written off before the last installment was due")


def _check_cash(c: _Checker, w: dict[str, pd.DataFrame], terms: ledger.ProductTerms) -> None:
    cash = w["cash_events"]
    expected = _map(cash["kind"], ledger.CASH_KINDS)
    c.flag("cash_sign", "cash_events", cash[np.sign(cash["amount_cents"]) != expected],
           "event_id", "amount sign does not match the kind")
    events = event_frame({n: w[n] for n in EVENT_TABLES if n != "cash_events"})
    cause_time = _lookup(events.dropna(subset=["event_id"]).astype({"event_id": "int64"}),
                         "event_id", "occurred_at")
    c.flag("cash_before_cause", "cash_events",
           cash[cash["occurred_at"] < _map(cash["ref_event_id"], cause_time)], "event_id",
           "cash moved before the event that caused it")
    plans = w["plans"].set_index("plan_id")
    c.flag("cash_plan_mismatch", "cash_events",
           cash[_map(cash["plan_id"], plans["order_id"]) != cash["order_id"]], "event_id",
           "plan does not belong to the order")
    try:
        derived = coerce("cash_events", ledger.derive_cash_events(w, terms).assign(event_id=0))
    except (ValueError, SchemaError) as error:
        c.violations.append(Violation("cash_mismatch", "cash_events", 1, [],
                                      f"cash events cannot be derived: {error}"))
        return
    columns = [name for name in TABLES["cash_events"].column_names if name != "event_id"]
    stored, expected = cash[columns].copy(), derived[columns].copy()
    for frame in (stored, expected):  # a multiset: number repeated rows
        frame["copy"] = frame.groupby(columns, dropna=False).cumcount()
    both = stored.assign(_s=1).merge(expected.assign(_e=1), on=[*columns, "copy"], how="outer")
    extra, missing = both[both["_e"].isna()], both[both["_s"].isna()]
    if len(extra) or len(missing):
        c.violations.append(Violation(
            "cash_mismatch", "cash_events", len(extra) + len(missing),
            [*extra["ref_event_id"].head(3).tolist(), *missing["ref_event_id"].head(3).tolist()],
            f"{len(extra)} stored cash events the ledger does not derive, {len(missing)} "
            "derived events missing (or amounts or times differ)"))


def _check_labels(c: _Checker, w: dict[str, pd.DataFrame]) -> None:
    labels = w["labels"]
    placed = _map(labels["order_id"], _lookup(w["order_attempts"], "order_id", "occurred_at"))
    c.flag("label_before_order", "labels", labels[labels["label_known_at"] < placed], "order_id",
           "label known before the order was placed")
    positive = labels["basis"].isin(LABEL_POSITIVE_BASES)
    c.flag("label_basis_mismatch", "labels", labels[positive != labels["label"].eq(1)],
           "order_id", "label value does not match its basis")
    latent = w["latent_orders"]
    c.flag("latent_inconsistent", "latent_orders",
           latent[latent["pattern_id"].isna() != latent["intent"].eq("legitimate")], "order_id",
           "legitimate orders have no pattern and patterned orders are not legitimate")
