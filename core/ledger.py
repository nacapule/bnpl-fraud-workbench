"""The platform's cash ledger, in integer cents.

Every money figure comes from one table of cash events seen from the platform's
side: cash in is positive, cash out is negative. The net contribution of a set
of events is the sum of their ``amount_cents``; loss is its negative.

| kind                | sign | when, and how much                                         |
|---------------------|------|------------------------------------------------------------|
| merchant_settlement | -    | at fulfilment: price - merchant fee - promotion discount   |
| promotion_funding   | -    | at fulfilment, when discounted: the discount               |
| customer_payment    | +    | each successful payment attempt                            |
| payment_reversal    | -    | each reversal of a collected payment (bank return)         |
| refund              | -    | only from actions: a still-standing payment given back     |
| dispute_debit       | -    | when the platform is notified of a dispute: the amount     |
| dispute_fee         | -    | at the same time: the network's dispute fee                |
| dispute_won_credit  | +    | when a won resolution is known: the disputed amount        |
| merchant_recourse   | +    | when a lost resolution is known, the merchant is liable    |
|                     |      | and still open: the disputed amount                        |
| recovery            | +    | write-off + recovery lag: a share of the written-off sum   |

The merchant is settled net of its fee, so the fee is never a separate credit;
together the settlement and the promotion funding pay the merchant its price
less the fee. Write-off is a plan status, not cash: only the later recovery is.

:func:`derive_cash_events` turns a world's event tables into these natural
events; an action in the replay keeps the events it does not prevent and adds
its own (:func:`cancel_before_fulfilment` adds refunds).

The product terms (``config/world.yaml``, section ``product``) are modelling
assumptions read into :class:`ProductTerms`. Down payments are rounded down to
the cent and leftover installment cents go to the earliest installments, one
cent each; fees and recoveries round to the nearest cent, halves up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from core import config

NATURAL = "natural"

CASH_KINDS: dict[str, int] = {
    "merchant_settlement": -1,
    "promotion_funding": -1,
    "customer_payment": 1,
    "payment_reversal": -1,
    "refund": -1,
    "dispute_debit": -1,
    "dispute_fee": -1,
    "dispute_won_credit": 1,
    "merchant_recourse": 1,
    "recovery": 1,
}

CASH_COLUMNS: tuple[str, ...] = (
    "event_id",
    "order_id",
    "plan_id",
    "merchant_id",
    "kind",
    "amount_cents",
    "occurred_at",
    "known_at",
    "ref_event_id",
    "cause",
)

DISPUTE_REASONS: tuple[str, ...] = ("unauthorized", "item_not_received", "not_as_described")
LIABLE_PARTIES: tuple[str, ...] = ("platform", "merchant")
REMAINDER_RULE = "earliest_installments_first"
PROMOTION_FUNDER = "platform"

# The world tables and columns derive_cash_events reads; other columns are ignored.
INPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "order_attempts": (
        "order_id", "merchant_id", "amount_cents", "promo_discount_cents", "processor_result",
    ),
    "plans": ("plan_id", "order_id"),
    "fulfilments": ("event_id", "order_id", "occurred_at", "known_at"),
    "payment_attempts": (
        "event_id", "plan_id", "occurred_at", "known_at", "amount_cents", "result",
    ),
    "payment_reversals": (
        "event_id", "payment_event_id", "plan_id", "occurred_at", "known_at", "amount_cents",
    ),
    "dispute_openings": (
        "event_id", "dispute_id", "order_id", "reason", "amount_cents", "known_at",
    ),
    "dispute_resolutions": ("event_id", "dispute_id", "outcome", "occurred_at", "known_at"),
    "plan_writeoffs": ("event_id", "plan_id", "outstanding_cents", "occurred_at"),
    "merchants": ("merchant_id", "closed_at"),
}

_BPS = 10_000
_KIND_RANK = {kind: rank for rank, kind in enumerate(CASH_KINDS)}
_ID_COLUMNS = frozenset(
    {"event_id", "order_id", "plan_id", "merchant_id", "payment_event_id", "dispute_id"}
)
_MONEY_COLUMNS = frozenset({"amount_cents", "promo_discount_cents", "outstanding_cents"})
_TIME_COLUMNS = frozenset({"occurred_at", "known_at", "closed_at"})
_NULLABLE = frozenset({("merchants", "closed_at")})
# Largest amount whose product with a rate of at most 10000 bps stays inside int64.
_MAX_RATED_CENTS = (np.iinfo(np.int64).max - _BPS) // _BPS
_CANCELLABLE_KINDS = ("customer_payment", "payment_reversal", "refund")


def _is_int(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _check_int(name: str, value: Any, minimum: int) -> int:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


@dataclass(frozen=True)
class ProductTerms:
    """The pay-in-4 product's terms; every value is a modelling assumption."""

    down_payment_bps: int
    n_installments: int
    installment_interval_days: int
    merchant_discount_bps: int
    dispute_fee_cents: int
    liability: Mapping[str, str]
    writeoff_after_days: int
    recovery_lag_days: int
    recovery_rate_bps: int

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name == "liability":
                continue
            minimum = 1 if field.name in ("n_installments", "installment_interval_days") else 0
            value = _check_int(field.name, getattr(self, field.name), minimum)
            if field.name.endswith("_bps") and value > _BPS:
                raise ValueError(f"{field.name} must be at most {_BPS}, got {value}")
            object.__setattr__(self, field.name, value)
        if not isinstance(self.liability, Mapping):
            raise ValueError("liability must map each dispute reason to a party")
        liability = dict(self.liability)
        missing = [reason for reason in DISPUTE_REASONS if reason not in liability]
        unknown = sorted(set(liability) - set(DISPUTE_REASONS))
        if missing or unknown:
            raise ValueError(
                f"liability must cover exactly {DISPUTE_REASONS}; "
                f"missing {missing}, unknown {unknown}"
            )
        parties = {reason: party for reason, party in liability.items()
                   if party not in LIABLE_PARTIES}
        if parties:
            raise ValueError(f"liability parties must be one of {LIABLE_PARTIES}, got {parties}")
        object.__setattr__(self, "liability", liability)

    def __hash__(self) -> int:
        values = [getattr(self, field.name) for field in fields(self) if field.name != "liability"]
        return hash((*values, tuple(sorted(self.liability.items()))))

    @classmethod
    def from_config(cls, world: Mapping[str, Any] | None = None) -> ProductTerms:
        """Terms from a world configuration (default: ``config/world.yaml``)."""
        world = config.load("world") if world is None else world
        product = world.get("product")
        if not isinstance(product, Mapping):
            raise ValueError("the world configuration has no product section")
        if product.get("remainder_rule", REMAINDER_RULE) != REMAINDER_RULE:
            raise ValueError(f"only the {REMAINDER_RULE!r} remainder rule is implemented")
        if product.get("promotions_funded_by", PROMOTION_FUNDER) != PROMOTION_FUNDER:
            raise ValueError("only platform-funded promotions are implemented")
        names = [field.name for field in fields(cls)]
        missing = [name for name in names if name not in product]
        if missing:
            raise ValueError(f"product terms missing from the configuration: {missing}")
        return cls(**{name: product[name] for name in names})


def round_bps(cents: int, bps: int) -> int:
    """``cents * bps / 10000`` to the nearest cent, halves up, in integer arithmetic."""
    cents = _check_int("cents", cents, 0)
    bps = _check_int("bps", bps, 0)
    return (cents * bps + _BPS // 2) // _BPS


def merchant_fee_cents(amount_cents: int, terms: ProductTerms) -> int:
    """The merchant's fee on its price (before any promotion discount)."""
    return round_bps(amount_cents, terms.merchant_discount_bps)


def settlement_cents(amount_cents: int, promo_discount_cents: int, terms: ProductTerms) -> int:
    """What the merchant settlement pays (a positive magnitude).

    The platform-funded discount is paid separately as promotion funding.
    """
    amount_cents = _check_int("amount_cents", amount_cents, 1)
    promo_discount_cents = _check_int("promo_discount_cents", promo_discount_cents, 0)
    settlement = amount_cents - merchant_fee_cents(amount_cents, terms) - promo_discount_cents
    if settlement <= 0:
        raise ValueError(
            f"a {promo_discount_cents}-cent discount on {amount_cents} cents leaves no settlement"
        )
    return settlement


def recovery_cents(outstanding_cents: int, terms: ProductTerms) -> int:
    """The single recovery on a written-off balance."""
    return round_bps(outstanding_cents, terms.recovery_rate_bps)


def split_principal(principal_cents: int, terms: ProductTerms) -> list[int]:
    """``[down, installment_1, ..., installment_n]``, summing to the principal.

    The down payment is rounded down to the cent; the rest is split evenly and
    leftover cents go to the earliest installments, one cent each.
    """
    principal_cents = _check_int("principal_cents", principal_cents, 1)
    down = principal_cents * terms.down_payment_bps // _BPS
    base, leftover = divmod(principal_cents - down, terms.n_installments)
    return [down] + [base + 1] * leftover + [base] * (terms.n_installments - leftover)


def payment_schedule(principal_cents: int, approved_at: Any, terms: ProductTerms) -> pd.DataFrame:
    """Due dates and amounts: seq 0 at approval, seq k k intervals later."""
    amounts = split_principal(principal_cents, terms)
    start = _timestamp(approved_at, "approved_at").to_datetime64().astype("datetime64[s]")
    seq = np.arange(len(amounts), dtype=np.int64)
    due = start + seq * np.timedelta64(terms.installment_interval_days, "D")
    return pd.DataFrame(
        {
            "seq": seq,
            "due_at": due.astype("datetime64[s]"),
            "amount_cents": np.array(amounts, dtype=np.int64),
        }
    )


def liable_party(reason: str, merchant_closed_at: Any, at: Any, terms: ProductTerms) -> str:
    """Who bears a lost dispute: the configured party, or the platform when the
    merchant is liable but had closed at or before ``at``."""
    try:
        party = terms.liability[reason]
    except KeyError:
        raise ValueError(f"unknown dispute reason {reason!r}") from None
    if party == "merchant" and not pd.isna(merchant_closed_at):
        if _timestamp(merchant_closed_at, "merchant_closed_at") <= _timestamp(at, "at"):
            return "platform"
    return party


def derive_cash_events(tables: Mapping[str, pd.DataFrame], terms: ProductTerms) -> pd.DataFrame:
    """The natural cash events of a world, sorted by known_at, kind and reference.

    Only processor-approved orders produce cash; rows of declined orders are
    ignored, and rows naming an order, plan or dispute that does not exist are
    an error. ``event_id`` is left empty for the caller to assign.
    """
    world = _read_inputs(tables)
    orders = world["order_attempts"]
    _require_unique(orders, "order_id", "order_attempts")
    _require_values(orders, "processor_result", ("approved", "declined"), "order_attempts")
    approved = orders.loc[orders["processor_result"].eq("approved")].drop(
        columns="processor_result"
    )
    _check_order_amounts(approved, terms)

    plans = world["plans"]
    _require_unique(plans, "plan_id", "plans")
    _require_unique(plans, "order_id", "plans")
    _require_known(plans, "order_id", orders["order_id"], "plans")
    order_keys = plans.merge(approved, on="order_id", validate="one_to_one")
    without_plan = ~approved["order_id"].isin(order_keys["order_id"])
    if without_plan.any():
        raise ValueError(
            f"{int(without_plan.sum())} approved orders have no plan "
            f"(first: order {int(approved.loc[without_plan, 'order_id'].iloc[0])})"
        )
    plan_keys = order_keys[["plan_id", "order_id", "merchant_id"]]

    events = pd.concat(
        [
            _fulfilment_cash(world["fulfilments"], orders, order_keys, terms),
            *_payment_cash(world["payment_attempts"], world["payment_reversals"], plans, plan_keys),
            *_dispute_cash(
                world["dispute_openings"],
                world["dispute_resolutions"],
                world["merchants"],
                orders,
                order_keys,
                terms,
            ),
            _recovery_cash(world["plan_writeoffs"], plans, plan_keys, terms),
        ],
        ignore_index=True,
    )
    events.insert(0, "event_id", pd.array([pd.NA] * len(events), dtype="Int64"))
    events["cause"] = NATURAL
    events = _sorted(_typed(events))
    validate_cash_events(events)
    return events


def cancel_before_fulfilment(
    order_events: pd.DataFrame,
    at: Any,
    cause: str,
    *,
    reversals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One order's cash when it is cancelled at ``at``, before it ships.

    Keeps the events realised by ``at`` (``occurred_at <= at``) and adds one
    refund at ``at`` for each realised customer payment still standing, that is
    not already reversed or refunded. ``reversals`` is the world's
    ``payment_reversals`` table; it says which payment a realised reversal
    returned and is needed only when the order has one.

    Raises ``ValueError`` when the order has been settled by ``at`` (it has
    shipped, which needs a different action) or has dispute or recovery cash by
    then (a cancellation cannot settle those).
    """
    validate_cash_events(order_events)
    if not isinstance(cause, str) or not cause or cause == NATURAL:
        raise ValueError(f"cause must name the action, got {cause!r}")
    if order_events["order_id"].nunique() > 1:
        raise ValueError("cancel_before_fulfilment takes the events of one order")
    at = _timestamp(at, "at")
    realised = order_events.loc[order_events["occurred_at"] <= at]

    shipped = realised["kind"].isin(["merchant_settlement", "promotion_funding"])
    if shipped.any():
        first = realised.loc[shipped, "occurred_at"].min()
        raise ValueError(
            f"order {int(realised['order_id'].iloc[0])} was settled at {first}, "
            f"before the cancellation at {at}: it has shipped"
        )
    blocking = ~realised["kind"].isin(_CANCELLABLE_KINDS)
    if blocking.any():
        kinds = sorted(set(realised.loc[blocking, "kind"]))
        raise ValueError(f"cannot cancel an order with {kinds} cash by {at}")

    # Each payment's standing amount: collected, less what was already refunded
    # (a refund refers to its payment) or reversed (found through the reversals).
    payments = realised.loc[realised["kind"].eq("customer_payment")]
    flows = [realised.loc[realised["kind"].isin(["customer_payment", "refund"])]]
    reversed_rows = realised.loc[realised["kind"].eq("payment_reversal")]
    if len(reversed_rows):
        if reversals is None:
            raise ValueError("pass the payment_reversals table to cancel an order with reversals")
        link = reversals[["event_id", "payment_event_id"]].drop_duplicates("event_id")
        linked = reversed_rows[["ref_event_id", "amount_cents"]].merge(
            link, left_on="ref_event_id", right_on="event_id", how="left", indicator=True
        )
        if not linked["_merge"].eq("both").all():
            raise ValueError("a realised reversal is missing from the payment_reversals table")
        flows.append(
            linked[["payment_event_id", "amount_cents"]].rename(
                columns={"payment_event_id": "ref_event_id"}
            )
        )
    flows = [flow[["ref_event_id", "amount_cents"]].astype(np.int64) for flow in flows]
    standing = pd.concat(flows).groupby("ref_event_id")["amount_cents"].sum()
    if not standing.index.isin(payments["ref_event_id"]).all():
        raise ValueError(f"refund or reversal of a payment not collected by {at}")
    if (standing < 0).any():
        raise ValueError("more was returned on a payment than was collected")
    standing = standing.loc[standing > 0]

    source = payments.drop_duplicates("ref_event_id").set_index("ref_event_id")
    source = source.loc[standing.index]
    stamp = np.full(len(standing), at.to_datetime64(), dtype="datetime64[s]")
    refunds = pd.DataFrame(
        {
            "event_id": pd.array([pd.NA] * len(standing), dtype="Int64"),
            "order_id": source["order_id"].to_numpy(),
            "plan_id": source["plan_id"].to_numpy(),
            "merchant_id": source["merchant_id"].to_numpy(),
            "kind": "refund",
            "amount_cents": -standing.to_numpy(),
            "occurred_at": stamp,
            "known_at": stamp,
            "ref_event_id": standing.index.to_numpy(),
            "cause": cause,
        }
    )
    result = _sorted(_typed(pd.concat([realised[list(CASH_COLUMNS)], refunds], ignore_index=True)))
    validate_cash_events(result)
    return result


def net_cents(events: pd.DataFrame, *, until: Any = None, clock: str = "occurred_at") -> int:
    """The net contribution of ``events``: the sum of ``amount_cents``, counting
    only events whose ``clock`` time is at or before ``until`` when given."""
    if clock not in ("occurred_at", "known_at"):
        raise ValueError(f"clock must be occurred_at or known_at, got {clock!r}")
    amount = events["amount_cents"]
    if not _is_integer_series(amount) or amount.isna().any():
        raise ValueError(f"amount_cents must be whole cents, got dtype {amount.dtype}")
    if until is not None:
        amount = amount.loc[events[clock] <= _timestamp(until, "until")]
    return int(amount.sum())


def validate_cash_events(events: pd.DataFrame) -> None:
    """Raise ``ValueError`` naming every way ``events`` breaks the ledger's rules."""
    missing = [column for column in CASH_COLUMNS if column not in events.columns]
    if missing:
        raise ValueError(f"cash events are missing columns {missing}")
    problems: list[str] = []

    kind = events["kind"]
    unknown = ~kind.isin(list(CASH_KINDS))
    if unknown.any():
        problems.append(f"unknown kinds {sorted(set(kind.loc[unknown].astype(str)))}")

    amount = events["amount_cents"]
    if not _is_integer_series(amount):
        problems.append(f"amount_cents must be integer cents, got dtype {amount.dtype}")
    elif amount.isna().any():
        problems.append(f"{int(amount.isna().sum())} amounts are missing")
    else:
        values = amount.to_numpy(dtype=np.int64)
        zero = values == 0
        if zero.any():
            problems.append(f"{int(zero.sum())} zero amounts ({_kinds(kind, zero)})")
        expected = kind.map(CASH_KINDS).to_numpy(dtype=float, na_value=np.nan)
        wrong = ~unknown.to_numpy() & ~zero & (np.sign(values) != expected)
        if wrong.any():
            problems.append(
                f"{int(wrong.sum())} amounts with the wrong sign ({_kinds(kind, wrong)})"
            )

    times_ok = True
    for column in ("occurred_at", "known_at"):
        if not pdt.is_datetime64_dtype(events[column].dtype):
            problems.append(f"{column} must be naive datetime64, got dtype {events[column].dtype}")
            times_ok = False
        elif events[column].isna().any():
            problems.append(f"{int(events[column].isna().sum())} missing {column} times")
            times_ok = False
    if times_ok:
        early = events["known_at"] < events["occurred_at"]
        if early.any():
            problems.append(f"{int(early.sum())} events known before they occurred")

    for column in ("order_id", "plan_id", "merchant_id", "ref_event_id"):
        if not _is_integer_series(events[column]) or events[column].isna().any():
            problems.append(f"{column} must be a complete integer column")
    if not _is_integer_series(events["event_id"]):
        problems.append(f"event_id must be integer or empty, got {events['event_id'].dtype}")
    elif events["event_id"].dropna().duplicated().any():
        problems.append("duplicate event_id values")
    cause = events["cause"].astype(object)
    if len(cause) and (pdt.infer_dtype(cause, skipna=False) != "string" or cause.eq("").any()):
        problems.append("cause must name 'natural' or the action on every event")

    if problems:
        raise ValueError("invalid cash events: " + "; ".join(problems))


def _kinds(kind: pd.Series, mask: np.ndarray) -> str:
    return ", ".join(sorted(set(kind.loc[mask].astype(str))))


def _is_integer_series(series: pd.Series) -> bool:
    return pdt.is_integer_dtype(series.dtype) and not pdt.is_bool_dtype(series.dtype)


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"{name} is missing")
    if stamp.tz is not None:
        raise ValueError(f"{name} must be a naive platform-local time, got {stamp}")
    return stamp.floor("s")


def _read_inputs(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The columns derive_cash_events reads, checked and cast to int64 and datetime64[s]."""
    absent = [name for name in INPUT_COLUMNS if name not in tables]
    if absent:
        raise ValueError(f"world tables missing: {absent}")
    world: dict[str, pd.DataFrame] = {}
    for name, columns in INPUT_COLUMNS.items():
        frame = tables[name]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}")
        out = {}
        for column in columns:
            series = frame[column]
            if column in _ID_COLUMNS or column in _MONEY_COLUMNS:
                if not _is_integer_series(series) or series.isna().any():
                    raise ValueError(
                        f"{name}.{column} must be a complete integer column, got {series.dtype}"
                    )
                out[column] = series.to_numpy(dtype=np.int64)
            elif column in _TIME_COLUMNS:
                if not pdt.is_datetime64_dtype(series.dtype):
                    raise ValueError(
                        f"{name}.{column} must be naive datetime64, got {series.dtype}"
                    )
                if (name, column) not in _NULLABLE and series.isna().any():
                    raise ValueError(f"{name}.{column} has missing times")
                out[column] = series.to_numpy().astype("datetime64[s]")
            else:
                out[column] = series.to_numpy()
        world[name] = pd.DataFrame(out)
    return world


def _require_unique(frame: pd.DataFrame, column: str, table: str) -> None:
    duplicated = frame[column].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{table}.{column} must be unique; "
            f"{int(frame.loc[duplicated, column].iloc[0])} repeats"
        )


def _require_known(frame: pd.DataFrame, column: str, known: pd.Series, table: str) -> None:
    unknown = ~frame[column].isin(known)
    if unknown.any():
        raise ValueError(
            f"{table}: {int(unknown.sum())} rows name an unknown {column} "
            f"(first: {int(frame.loc[unknown, column].iloc[0])})"
        )


def _require_values(frame: pd.DataFrame, column: str, allowed: tuple[str, ...], table: str) -> None:
    other = ~frame[column].isin(allowed)
    if other.any():
        found = sorted(set(frame.loc[other, column].astype(str)))
        raise ValueError(f"{table}.{column} must be one of {allowed}; found {found}")


def _require_positive(frame: pd.DataFrame, column: str, table: str) -> None:
    if (frame[column] <= 0).any():
        raise ValueError(f"{table}.{column} must be positive")


def _rated(cents: pd.Series | np.ndarray, bps: int) -> np.ndarray:
    """Vectorised :func:`round_bps` for non-negative int64 cents."""
    values = np.asarray(cents, dtype=np.int64)
    if (values < 0).any() or (values > _MAX_RATED_CENTS).any():
        raise ValueError("amounts must be non-negative and below the int64 rating limit")
    return (values * np.int64(bps) + _BPS // 2) // _BPS


def _check_order_amounts(approved: pd.DataFrame, terms: ProductTerms) -> None:
    _require_positive(approved, "amount_cents", "order_attempts")
    if (approved["promo_discount_cents"] < 0).any():
        raise ValueError("order_attempts.promo_discount_cents must be non-negative")
    fee = _rated(approved["amount_cents"], terms.merchant_discount_bps)
    short = approved["amount_cents"].to_numpy() - fee - approved["promo_discount_cents"] <= 0
    if short.any():
        raise ValueError(
            f"{int(short.sum())} approved orders have a discount that leaves no merchant "
            f"settlement (first: order {int(approved.loc[short, 'order_id'].iloc[0])})"
        )


def _cash(
    kind: str,
    rows: pd.DataFrame,
    magnitude: pd.Series | np.ndarray,
    occurred: pd.Series | np.ndarray,
    known: pd.Series | np.ndarray,
    ref: pd.Series | np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": rows["order_id"].to_numpy(dtype=np.int64),
            "plan_id": rows["plan_id"].to_numpy(dtype=np.int64),
            "merchant_id": rows["merchant_id"].to_numpy(dtype=np.int64),
            "kind": np.full(len(rows), kind, dtype=object),
            "amount_cents": CASH_KINDS[kind] * np.asarray(magnitude, dtype=np.int64),
            "occurred_at": np.asarray(occurred, dtype="datetime64[s]"),
            "known_at": np.asarray(known, dtype="datetime64[s]"),
            "ref_event_id": np.asarray(ref, dtype=np.int64),
        }
    )


def _attach(
    frame: pd.DataFrame, keys: pd.DataFrame, on: str, known: pd.Series, table: str
) -> pd.DataFrame:
    """Rows of approved orders, with their order, plan and merchant ids."""
    _require_known(frame, on, known, table)
    return frame.merge(keys, on=on, how="inner", validate="many_to_one")


def _fulfilment_cash(
    fulfilments: pd.DataFrame, orders: pd.DataFrame, order_keys: pd.DataFrame, terms: ProductTerms
) -> pd.DataFrame:
    _require_unique(fulfilments, "event_id", "fulfilments")
    _require_unique(fulfilments, "order_id", "fulfilments")
    shipped = _attach(fulfilments, order_keys, "order_id", orders["order_id"], "fulfilments")
    fee = _rated(shipped["amount_cents"], terms.merchant_discount_bps)
    settlement = shipped["amount_cents"].to_numpy() - fee - shipped["promo_discount_cents"]
    promoted = shipped.loc[shipped["promo_discount_cents"] > 0]
    return pd.concat(
        [
            _cash(
                "merchant_settlement", shipped, settlement,
                shipped["occurred_at"], shipped["known_at"], shipped["event_id"],
            ),
            _cash(
                "promotion_funding", promoted, promoted["promo_discount_cents"],
                promoted["occurred_at"], promoted["known_at"], promoted["event_id"],
            ),
        ],
        ignore_index=True,
    )


def _payment_cash(
    attempts: pd.DataFrame, reversals: pd.DataFrame, plans: pd.DataFrame, plan_keys: pd.DataFrame
) -> list[pd.DataFrame]:
    _require_unique(attempts, "event_id", "payment_attempts")
    _require_values(attempts, "result", ("success", "failed"), "payment_attempts")
    _require_known(attempts, "plan_id", plans["plan_id"], "payment_attempts")
    collected = attempts.loc[attempts["result"].eq("success")]
    _require_positive(collected, "amount_cents", "payment_attempts")

    _require_unique(reversals, "event_id", "payment_reversals")
    _require_positive(reversals, "amount_cents", "payment_reversals")
    target = collected[["event_id", "plan_id", "amount_cents"]].rename(
        columns={"event_id": "payment_event_id", "plan_id": "paid_plan", "amount_cents": "paid"}
    )
    linked = reversals.merge(target, on="payment_event_id", how="left", validate="many_to_one")
    stray = linked["paid_plan"].isna() | (linked["paid_plan"] != linked["plan_id"])
    if stray.any():
        raise ValueError(
            f"payment_reversals: {int(stray.sum())} rows do not reverse a successful payment "
            f"of the same plan (first: event {int(linked.loc[stray, 'event_id'].iloc[0])})"
        )
    totals = linked.groupby("payment_event_id").agg(reversed=("amount_cents", "sum"),
                                                    paid=("paid", "first"))
    if (totals["reversed"] > totals["paid"]).any():
        raise ValueError("payment_reversals: a payment is reversed by more than was collected")

    collected = collected.merge(plan_keys, on="plan_id", how="inner", validate="many_to_one")
    reversed_ = reversals.merge(plan_keys, on="plan_id", how="inner", validate="many_to_one")
    return [
        _cash(
            "customer_payment", collected, collected["amount_cents"],
            collected["occurred_at"], collected["known_at"], collected["event_id"],
        ),
        _cash(
            "payment_reversal", reversed_, reversed_["amount_cents"],
            reversed_["occurred_at"], reversed_["known_at"], reversed_["event_id"],
        ),
    ]


def _dispute_cash(
    openings: pd.DataFrame,
    resolutions: pd.DataFrame,
    merchants: pd.DataFrame,
    orders: pd.DataFrame,
    order_keys: pd.DataFrame,
    terms: ProductTerms,
) -> list[pd.DataFrame]:
    _require_unique(openings, "event_id", "dispute_openings")
    _require_unique(openings, "dispute_id", "dispute_openings")
    _require_values(openings, "reason", DISPUTE_REASONS, "dispute_openings")
    _require_positive(openings, "amount_cents", "dispute_openings")
    _require_unique(resolutions, "event_id", "dispute_resolutions")
    _require_unique(resolutions, "dispute_id", "dispute_resolutions")
    _require_values(resolutions, "outcome", ("won", "lost"), "dispute_resolutions")
    _require_known(resolutions, "dispute_id", openings["dispute_id"], "dispute_resolutions")

    keys = order_keys[["order_id", "plan_id", "merchant_id"]]
    opened = _attach(openings, keys, "order_id", orders["order_id"], "dispute_openings")
    fees = opened if terms.dispute_fee_cents > 0 else opened.iloc[0:0]

    disputed = opened[
        ["dispute_id", "order_id", "plan_id", "merchant_id", "reason", "amount_cents"]
    ]
    resolved = resolutions.merge(disputed, on="dispute_id", how="inner", validate="one_to_one")
    won = resolved.loc[resolved["outcome"].eq("won")]
    lost = resolved.loc[resolved["outcome"].eq("lost")]
    _require_unique(merchants, "merchant_id", "merchants")
    _require_known(lost, "merchant_id", merchants["merchant_id"], "dispute_resolutions")
    lost = lost.merge(merchants, on="merchant_id", how="left", validate="many_to_one")
    merchant_liable = lost["reason"].map(terms.liability).eq("merchant")
    closed = lost["closed_at"].notna() & (lost["closed_at"] <= lost["occurred_at"])
    recourse = lost.loc[merchant_liable & ~closed]

    return [
        _cash(
            "dispute_debit", opened, opened["amount_cents"],
            opened["known_at"], opened["known_at"], opened["event_id"],
        ),
        _cash(
            "dispute_fee", fees, np.full(len(fees), terms.dispute_fee_cents, dtype=np.int64),
            fees["known_at"], fees["known_at"], fees["event_id"],
        ),
        _cash(
            "dispute_won_credit", won, won["amount_cents"],
            won["known_at"], won["known_at"], won["event_id"],
        ),
        _cash(
            "merchant_recourse", recourse, recourse["amount_cents"],
            recourse["known_at"], recourse["known_at"], recourse["event_id"],
        ),
    ]


def _recovery_cash(
    writeoffs: pd.DataFrame, plans: pd.DataFrame, plan_keys: pd.DataFrame, terms: ProductTerms
) -> pd.DataFrame:
    _require_unique(writeoffs, "event_id", "plan_writeoffs")
    _require_unique(writeoffs, "plan_id", "plan_writeoffs")
    if (writeoffs["outstanding_cents"] < 0).any():
        raise ValueError("plan_writeoffs.outstanding_cents must be non-negative")
    written = _attach(writeoffs, plan_keys, "plan_id", plans["plan_id"], "plan_writeoffs")
    amount = _rated(written["outstanding_cents"], terms.recovery_rate_bps)
    keep = amount > 0
    written = written.loc[keep]
    at = written["occurred_at"].to_numpy() + np.timedelta64(terms.recovery_lag_days, "D")
    return _cash("recovery", written, amount[keep], at, at, written["event_id"])


def _typed(events: pd.DataFrame) -> pd.DataFrame:
    """The cash columns in order, with the ledger's dtypes."""
    return pd.DataFrame(
        {
            "event_id": pd.array(events["event_id"], dtype="Int64"),
            "order_id": events["order_id"].to_numpy(dtype=np.int64),
            "plan_id": events["plan_id"].to_numpy(dtype=np.int64),
            "merchant_id": events["merchant_id"].to_numpy(dtype=np.int64),
            "kind": events["kind"].astype(str).to_numpy(),
            "amount_cents": events["amount_cents"].to_numpy(dtype=np.int64),
            "occurred_at": events["occurred_at"].to_numpy().astype("datetime64[s]"),
            "known_at": events["known_at"].to_numpy().astype("datetime64[s]"),
            "ref_event_id": events["ref_event_id"].to_numpy(dtype=np.int64),
            "cause": events["cause"].astype(str).to_numpy(),
        }
    ).astype({"kind": str, "cause": str})


def _sorted(events: pd.DataFrame) -> pd.DataFrame:
    """Deterministic order: known_at, then kind in table order, then the reference."""
    rank = events["kind"].map(_KIND_RANK)
    order = np.lexsort(
        (
            events["ref_event_id"].to_numpy(),
            rank.to_numpy(),
            events["known_at"].to_numpy(),
        )
    )
    return events.iloc[order].reset_index(drop=True)
