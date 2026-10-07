"""The cash ledger on hand-checked cases, in integer cents.

Every expected amount is a literal worked out by hand, with the arithmetic in a
comment. Default terms are config/world.yaml's: 25% down, three fortnightly
installments, a 5% merchant fee, a $15 dispute fee, 20% recovered 30 days after
write-off.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from core.ledger import (
    CASH_COLUMNS,
    CASH_KINDS,
    NATURAL,
    ProductTerms,
    cancel_before_fulfilment,
    derive_cash_events,
    liable_party,
    merchant_fee_cents,
    net_cents,
    payment_schedule,
    recovery_cents,
    round_bps,
    settlement_cents,
    split_principal,
    validate_cash_events,
)

TERMS = ProductTerms.from_config()

T0 = pd.Timestamp("2026-03-02 10:00:00")  # order approved, down payment collected
SHIPPED = pd.Timestamp("2026-03-02 22:00:00")  # fulfilment 12 hours later
DUE_1 = pd.Timestamp("2026-03-16 10:00:00")  # T0 + 14 days
DUE_2 = pd.Timestamp("2026-03-30 10:00:00")  # T0 + 28 days
DUE_3 = pd.Timestamp("2026-04-13 10:00:00")  # T0 + 42 days
WRITTEN_OFF = pd.Timestamp("2026-04-27 10:00:00")  # last due date + 14 days
RECOVERED = pd.Timestamp("2026-05-27 10:00:00")  # write-off + 30 days

SCHEMA: dict[str, dict[str, str]] = {
    "order_attempts": {
        "event_id": "int64", "order_id": "int64", "user_id": "int64", "merchant_id": "int64",
        "occurred_at": "datetime64[s]", "known_at": "datetime64[s]", "amount_cents": "int64",
        "promo_id": "Int64", "promo_discount_cents": "int64", "processor_result": "str",
    },
    "plans": {
        "plan_id": "int64", "order_id": "int64", "created_at": "datetime64[s]",
        "principal_cents": "int64", "down_payment_cents": "int64", "n_installments": "int64",
    },
    "payment_attempts": {
        "event_id": "int64", "plan_id": "int64", "seq": "int64", "attempt_no": "int64",
        "occurred_at": "datetime64[s]", "known_at": "datetime64[s]", "amount_cents": "int64",
        "result": "str",
    },
    "payment_reversals": {
        "event_id": "int64", "payment_event_id": "int64", "plan_id": "int64",
        "occurred_at": "datetime64[s]", "known_at": "datetime64[s]", "amount_cents": "int64",
        "reason": "str",
    },
    "fulfilments": {
        "event_id": "int64", "order_id": "int64", "occurred_at": "datetime64[s]",
        "known_at": "datetime64[s]",
    },
    "dispute_openings": {
        "event_id": "int64", "dispute_id": "int64", "order_id": "int64", "reason": "str",
        "amount_cents": "int64", "occurred_at": "datetime64[s]", "known_at": "datetime64[s]",
    },
    "dispute_resolutions": {
        "event_id": "int64", "dispute_id": "int64", "outcome": "str",
        "occurred_at": "datetime64[s]", "known_at": "datetime64[s]",
    },
    "plan_writeoffs": {
        "event_id": "int64", "plan_id": "int64", "outstanding_cents": "int64",
        "occurred_at": "datetime64[s]", "known_at": "datetime64[s]",
    },
    "merchants": {"merchant_id": "int64", "closed_at": "datetime64[s]"},
}


@dataclass(frozen=True)
class Placed:
    order_id: int
    plan_id: int


class World:
    """A small hand-built world: one row per call, in the tables the ledger reads."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {name: [] for name in SCHEMA}
        self._event_ids = itertools.count(101)
        self._order_ids = itertools.count(1)
        self._dispute_ids = itertools.count(1)
        self.merchant(1)

    def merchant(self, merchant_id: int, closed_at: pd.Timestamp | None = None) -> None:
        self.rows["merchants"].append({"merchant_id": merchant_id, "closed_at": closed_at})

    def order(
        self, amount: int = 10000, promo: int = 0, merchant: int = 1, approved: bool = True
    ) -> Placed:
        order_id = next(self._order_ids)
        self.rows["order_attempts"].append({
            "event_id": next(self._event_ids), "order_id": order_id, "user_id": 7,
            "merchant_id": merchant, "occurred_at": T0, "known_at": T0, "amount_cents": amount,
            "promo_id": 1 if promo else None, "promo_discount_cents": promo,
            "processor_result": "approved" if approved else "declined",
        })
        plan_id = 1000 + order_id
        if approved:
            principal = amount - promo
            self.rows["plans"].append({
                "plan_id": plan_id, "order_id": order_id, "created_at": T0,
                "principal_cents": principal, "down_payment_cents": principal // 4,
                "n_installments": 3,
            })
        return Placed(order_id, plan_id)

    def pay(self, placed: Placed, seq: int, at: pd.Timestamp, amount: int, ok: bool = True) -> int:
        event_id = next(self._event_ids)
        self.rows["payment_attempts"].append({
            "event_id": event_id, "plan_id": placed.plan_id, "seq": seq, "attempt_no": 1,
            "occurred_at": at, "known_at": at, "amount_cents": amount,
            "result": "success" if ok else "failed",
        })
        return event_id

    def reverse(
        self, placed: Placed, payment: int, at: pd.Timestamp, known: pd.Timestamp, amount: int
    ) -> int:
        event_id = next(self._event_ids)
        self.rows["payment_reversals"].append({
            "event_id": event_id, "payment_event_id": payment, "plan_id": placed.plan_id,
            "occurred_at": at, "known_at": known, "amount_cents": amount,
            "reason": "bank_return",
        })
        return event_id

    def fulfil(self, placed: Placed, at: pd.Timestamp = SHIPPED) -> int:
        event_id = next(self._event_ids)
        self.rows["fulfilments"].append(
            {"event_id": event_id, "order_id": placed.order_id, "occurred_at": at, "known_at": at}
        )
        return event_id

    def dispute(
        self, placed: Placed, reason: str, amount: int, filed: pd.Timestamp, notified: pd.Timestamp
    ) -> tuple[int, int]:
        dispute_id, event_id = next(self._dispute_ids), next(self._event_ids)
        self.rows["dispute_openings"].append({
            "event_id": event_id, "dispute_id": dispute_id, "order_id": placed.order_id,
            "reason": reason, "amount_cents": amount, "occurred_at": filed, "known_at": notified,
        })
        return dispute_id, event_id

    def resolve(
        self, dispute_id: int, outcome: str, decided: pd.Timestamp, known: pd.Timestamp
    ) -> int:
        event_id = next(self._event_ids)
        self.rows["dispute_resolutions"].append({
            "event_id": event_id, "dispute_id": dispute_id, "outcome": outcome,
            "occurred_at": decided, "known_at": known,
        })
        return event_id

    def write_off(self, placed: Placed, outstanding: int, at: pd.Timestamp = WRITTEN_OFF) -> int:
        event_id = next(self._event_ids)
        self.rows["plan_writeoffs"].append({
            "event_id": event_id, "plan_id": placed.plan_id, "outstanding_cents": outstanding,
            "occurred_at": at, "known_at": at,
        })
        return event_id

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            name: pd.DataFrame(self.rows[name], columns=list(columns)).astype(columns)
            for name, columns in SCHEMA.items()
        }

    def cash(self, terms: ProductTerms = TERMS) -> pd.DataFrame:
        return derive_cash_events(self.tables(), terms)


def repaid(world: World, amount: int = 10000, promo: int = 0, merchant: int = 1,
           payments: tuple[int, int, int, int] = (2500, 2500, 2500, 2500)) -> Placed:
    """An order that ships 12 hours after approval and is paid on every due date."""
    placed = world.order(amount, promo, merchant)
    for seq, (at, paid) in enumerate(zip((T0, DUE_1, DUE_2, DUE_3), payments, strict=True)):
        world.pay(placed, seq, at, paid)
    world.fulfil(placed)
    return placed


def defaulted(world: World, amount: int = 10000, promo: int = 0, down: int = 2500,
              installment: int = 2500) -> Placed:
    """An order that ships, collects the down payment, then fails every installment."""
    placed = world.order(amount, promo)
    world.pay(placed, 0, T0, down)
    for seq, at in enumerate((DUE_1, DUE_2, DUE_3), start=1):
        world.pay(placed, seq, at, installment, ok=False)
    world.fulfil(placed)
    return placed


def cash_rows(*rows: tuple[str, int]) -> pd.DataFrame:
    """Explicit cash events for one order, a day apart, from (kind, signed cents)."""
    return pd.DataFrame({
        "event_id": pd.array(range(1, len(rows) + 1), dtype="Int64"),
        "order_id": np.ones(len(rows), dtype=np.int64),
        "plan_id": np.ones(len(rows), dtype=np.int64),
        "merchant_id": np.ones(len(rows), dtype=np.int64),
        "kind": [kind for kind, _ in rows],
        "amount_cents": np.array([amount for _, amount in rows], dtype=np.int64),
        "occurred_at": np.array(
            [T0 + pd.Timedelta(days=day) for day in range(len(rows))], dtype="datetime64[s]"
        ),
        "known_at": np.array(
            [T0 + pd.Timedelta(days=day) for day in range(len(rows))], dtype="datetime64[s]"
        ),
        "ref_event_id": np.arange(501, 501 + len(rows), dtype=np.int64),
        "cause": [NATURAL if kind != "refund" else "cancellation" for kind, _ in rows],
    })


# 1. Reconciliation cases, fee-free, as explicit event lists.

RECONCILIATION_CASES = {
    # -10000 + 2500 + 2500 + 2500 + 2500 = 0
    "settled and fully repaid": (
        [("merchant_settlement", -10000)] + [("customer_payment", 2500)] * 4,
        0,
    ),
    # -10000 + 2500 - 2500 = -10000: the reversed $25 is not collected
    "down payment collected then reversed": (
        [("merchant_settlement", -10000), ("customer_payment", 2500), ("payment_reversal", -2500)],
        -10000,
    ),
    # -10000 + 4 x 2500 - 4 x 2500 = -10000
    "fully repaid then refunded": (
        [("merchant_settlement", -10000)]
        + [("customer_payment", 2500)] * 4
        + [("refund", -2500)] * 4,
        -10000,
    ),
    # -9000 - 1000 + 4 x 2250 = -1000: the platform funds the $10 discount
    "funded discount, repaid": (
        [("merchant_settlement", -9000), ("promotion_funding", -1000)]
        + [("customer_payment", 2250)] * 4,
        -1000,
    ),
    # -10000 + 2000 = -8000
    "deficit then recovery": (
        [("merchant_settlement", -10000), ("recovery", 2000)],
        -8000,
    ),
}


@pytest.mark.parametrize("case", list(RECONCILIATION_CASES))
def test_reconciliation_cases(case: str) -> None:
    rows, expected = RECONCILIATION_CASES[case]
    events = cash_rows(*rows)
    validate_cash_events(events)
    assert net_cents(events) == expected


# 2. Merchant settled net of its fee.

def test_repaid_order_settled_at_95_nets_500() -> None:
    world = World()
    placed = repaid(world)
    fulfilment = world.rows["fulfilments"][0]["event_id"]
    cash = world.cash()
    settlement = cash.loc[cash["kind"] == "merchant_settlement"]
    # 10000 - 5% of 10000 = 10000 - 500 = 9500
    assert settlement["amount_cents"].tolist() == [-9500]
    assert settlement["occurred_at"].tolist() == [SHIPPED]
    assert settlement["ref_event_id"].tolist() == [fulfilment]
    assert cash["kind"].tolist().count("customer_payment") == 4
    assert set(cash["order_id"]) == {placed.order_id}
    # -9500 + 4 x 2500 = +500: the fee is not a second credit
    assert net_cents(cash) == 500


# 3. Unauthorized dispute after full repayment.

FILED = pd.Timestamp("2026-04-20 09:00:00")
NOTIFIED = pd.Timestamp("2026-04-21 09:00:00")
DECIDED = pd.Timestamp("2026-05-20 09:00:00")
DECISION_KNOWN = pd.Timestamp("2026-05-21 09:00:00")


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("lost", -11000),  # -9500 + 10000 - 10000 - 1500 = -11000
        ("won", -1000),  # -9500 + 10000 - 10000 - 1500 + 10000 = -1000
    ],
)
def test_unauthorized_dispute_after_full_repayment(outcome: str, expected: int) -> None:
    world = World()
    placed = repaid(world)
    dispute_id, opening = world.dispute(placed, "unauthorized", 10000, FILED, NOTIFIED)
    resolution = world.resolve(dispute_id, outcome, DECIDED, DECISION_KNOWN)
    cash = world.cash()
    debit = cash.loc[cash["kind"].isin(["dispute_debit", "dispute_fee"])]
    assert debit["amount_cents"].tolist() == [-10000, -1500]
    assert debit["occurred_at"].tolist() == [NOTIFIED, NOTIFIED]
    assert debit["known_at"].tolist() == [NOTIFIED, NOTIFIED]
    assert debit["ref_event_id"].tolist() == [opening, opening]
    assert "merchant_recourse" not in set(cash["kind"])
    if outcome == "won":
        credit = cash.loc[cash["kind"] == "dispute_won_credit"]
        assert credit["amount_cents"].tolist() == [10000]
        assert credit["occurred_at"].tolist() == [DECISION_KNOWN]
        assert credit["ref_event_id"].tolist() == [resolution]
    else:
        assert "dispute_won_credit" not in set(cash["kind"])
    assert net_cents(cash) == expected


# 4. Down payment disputed and lost; the rest defaults and is written off.

def test_disputed_down_payment_then_default_and_recovery() -> None:
    world = World()
    placed = defaulted(world)
    dispute_id, _ = world.dispute(
        placed, "unauthorized", 2500, pd.Timestamp("2026-03-20 09:00:00"),
        pd.Timestamp("2026-03-21 09:00:00"),
    )
    world.resolve(dispute_id, "lost", pd.Timestamp("2026-04-20 09:00:00"),
                  pd.Timestamp("2026-04-21 09:00:00"))
    writeoff = world.write_off(placed, 7500)
    cash = world.cash()
    assert cash["kind"].tolist() == [
        "customer_payment",  # +2500 at approval
        "merchant_settlement",  # -9500 at shipping
        "dispute_debit",  # -2500
        "dispute_fee",  # -1500
        "recovery",  # 20% of 7500 = +1500
    ]
    # -9500 + 2500 - 2500 - 1500 + 1500 = -9500
    assert net_cents(cash) == -9500
    # The write-off itself is not cash: only the recovery refers to it, 30 days later.
    from_writeoff = cash.loc[cash["ref_event_id"] == writeoff]
    assert from_writeoff["kind"].tolist() == ["recovery"]
    assert from_writeoff["amount_cents"].tolist() == [1500]
    assert from_writeoff["occurred_at"].tolist() == [RECOVERED]
    assert not (cash["occurred_at"] == WRITTEN_OFF).any()


# 5. Platform-funded promotion.

def test_promotion_split_and_settlement() -> None:
    # principal 10000 - 1000 = 9000; down 25% of 9000 = 2250; 6750 / 3 = 2250 each
    assert split_principal(9000, TERMS) == [2250, 2250, 2250, 2250]
    # 10000 - 500 fee - 1000 discount = 8500
    assert settlement_cents(10000, 1000, TERMS) == 8500


def test_promotion_repaid() -> None:
    world = World()
    repaid(world, amount=10000, promo=1000, payments=(2250, 2250, 2250, 2250))
    cash = world.cash()
    at_shipping = cash.loc[cash["occurred_at"] == SHIPPED]
    assert at_shipping["kind"].tolist() == ["merchant_settlement", "promotion_funding"]
    assert at_shipping["amount_cents"].tolist() == [-8500, -1000]
    # -8500 - 1000 + 4 x 2250 = -500
    assert net_cents(cash) == -500


def test_promotion_unpaid_after_down_payment() -> None:
    world = World()
    placed = defaulted(world, amount=10000, promo=1000, down=2250, installment=2250)
    world.write_off(placed, 6750)
    cash = world.cash()
    recovery = cash.loc[cash["kind"] == "recovery"]
    assert recovery["amount_cents"].tolist() == [1350]  # 20% of 6750 = 1350
    # -8500 - 1000 + 2250 + 1350 = -5900
    assert net_cents(cash) == -5900


# 6. Default then partial recovery.

def test_default_then_partial_recovery() -> None:
    world = World()
    placed = defaulted(world)
    writeoff = world.write_off(placed, 7500)
    cash = world.cash()
    recovery = cash.loc[cash["kind"] == "recovery"]
    assert recovery["amount_cents"].tolist() == [1500]  # 7500 x 2000 / 10000 = 1500
    assert recovery["occurred_at"].tolist() == [RECOVERED]  # 2026-04-27 + 30 days
    assert recovery["known_at"].tolist() == [RECOVERED]
    assert recovery["ref_event_id"].tolist() == [writeoff]
    # -9500 + 2500 + 1500 = -5500
    assert net_cents(cash) == -5500
    # Before the recovery lands, the loss is the whole shortfall: -9500 + 2500 = -7000.
    assert net_cents(cash, until=pd.Timestamp("2026-05-26 10:00:00")) == -7000


# 7. Principal split under the remainder rule.

def test_split_of_100_01() -> None:
    # down floor(10001 x 0.25) = floor(2500.25) = 2500; 7501 = 3 x 2500 + 1 -> [2501, 2500, 2500]
    parts = split_principal(10001, TERMS)
    assert parts == [2500, 2501, 2500, 2500]
    assert sum(parts) == 10001


@pytest.mark.parametrize(
    "terms",
    [
        TERMS,
        dataclasses.replace(TERMS, down_payment_bps=0, n_installments=5),
        dataclasses.replace(TERMS, down_payment_bps=3333, n_installments=7),
        dataclasses.replace(TERMS, down_payment_bps=10000, n_installments=1),
    ],
)
def test_split_properties(terms: ProductTerms) -> None:
    rng = np.random.default_rng(416)
    principals = list(range(1, 2001)) + [int(p) for p in rng.integers(1, 10**9, 2000)]
    for principal in principals:
        down, *installments = split_principal(principal, terms)
        assert down + sum(installments) == principal
        assert down == principal * terms.down_payment_bps // 10000
        assert len(installments) == terms.n_installments
        assert max(installments) - min(installments) <= 1
        assert installments == sorted(installments, reverse=True)  # extra cents come first


def test_split_rejects_non_positive_and_fractional_principal() -> None:
    for principal in (0, -100):
        with pytest.raises(ValueError):
            split_principal(principal, TERMS)
    with pytest.raises(TypeError):
        split_principal(100.0, TERMS)


def test_payment_schedule() -> None:
    schedule = payment_schedule(10001, T0, TERMS)
    assert schedule.columns.tolist() == ["seq", "due_at", "amount_cents"]
    assert schedule["seq"].tolist() == [0, 1, 2, 3]
    assert schedule["due_at"].tolist() == [T0, DUE_1, DUE_2, DUE_3]
    assert schedule["amount_cents"].tolist() == [2500, 2501, 2500, 2500]
    assert str(schedule["due_at"].dtype) == "datetime64[s]"
    assert str(schedule["amount_cents"].dtype) == "int64"


# 8. Cancellation before fulfilment.

def _down_payment_order() -> tuple[World, Placed, int]:
    world = World()
    placed = world.order()
    down = world.pay(placed, 0, T0, 2500)
    world.pay(placed, 1, DUE_1, 2500)
    world.fulfil(placed)
    return world, placed, down


def test_cancel_after_down_payment_refunds_it() -> None:
    world, placed, down = _down_payment_order()
    cancelled_at = pd.Timestamp("2026-03-02 16:00:00")  # six hours in, before shipping
    events = cancel_before_fulfilment(world.cash(), cancelled_at, "hold_abandoned")
    assert events.columns.tolist() == list(CASH_COLUMNS)
    assert events["kind"].tolist() == ["customer_payment", "refund"]
    assert events["amount_cents"].tolist() == [2500, -2500]
    refund = events.iloc[1]
    assert (refund["occurred_at"], refund["known_at"]) == (cancelled_at, cancelled_at)
    assert refund["ref_event_id"] == down
    assert refund["cause"] == "hold_abandoned"
    assert events.iloc[0]["cause"] == NATURAL
    assert pd.isna(refund["event_id"])
    assert set(events["order_id"]) == {placed.order_id}
    # +2500 - 2500 = 0
    assert net_cents(events) == 0


def test_cancel_after_shipping_is_refused() -> None:
    world, _, _ = _down_payment_order()
    with pytest.raises(ValueError, match="shipped"):
        cancel_before_fulfilment(world.cash(), pd.Timestamp("2026-03-02 23:00:00"), "void")


def test_cancel_does_not_refund_a_reversed_payment() -> None:
    world = World()
    placed = world.order()
    down = world.pay(placed, 0, T0, 2500)
    world.reverse(placed, down, pd.Timestamp("2026-03-02 12:00:00"),
                  pd.Timestamp("2026-03-02 13:00:00"), 2500)
    world.fulfil(placed, pd.Timestamp("2026-03-04 10:00:00"))
    cash = world.cash()
    at = pd.Timestamp("2026-03-03 10:00:00")
    with pytest.raises(ValueError, match="payment_reversals"):
        cancel_before_fulfilment(cash, at, "hold_abandoned")
    tables = world.tables()
    events = cancel_before_fulfilment(cash, at, "hold_abandoned",
                                      reversals=tables["payment_reversals"])
    assert events["kind"].tolist() == ["customer_payment", "payment_reversal"]
    # +2500 - 2500 = 0, with nothing left to refund
    assert net_cents(events) == 0


def test_cancel_with_an_open_dispute_is_refused() -> None:
    world = World()
    placed = world.order()
    world.pay(placed, 0, T0, 2500)
    world.dispute(placed, "unauthorized", 2500, pd.Timestamp("2026-03-02 11:00:00"),
                  pd.Timestamp("2026-03-02 12:00:00"))
    world.fulfil(placed, pd.Timestamp("2026-03-04 10:00:00"))
    with pytest.raises(ValueError, match="dispute"):
        cancel_before_fulfilment(world.cash(), pd.Timestamp("2026-03-03 10:00:00"), "void")


def test_cancel_needs_an_action_name() -> None:
    world, _, _ = _down_payment_order()
    with pytest.raises(ValueError, match="cause"):
        cancel_before_fulfilment(world.cash(), pd.Timestamp("2026-03-02 16:00:00"), NATURAL)


# 9. Item not received: merchant recourse unless the merchant has closed.

@pytest.mark.parametrize(
    ("closed_at", "expected"),
    [
        (None, -1000),  # -9500 + 10000 - 10000 - 1500 + 10000 = -1000
        (pd.Timestamp("2026-06-01 00:00:00"), -1000),  # closed after the decision: recourse
        # closed after the decision but before the platform learned of it: still recourse
        (pd.Timestamp("2026-05-20 12:00:00"), -1000),
        (pd.Timestamp("2026-05-01 00:00:00"), -11000),  # -9500 + 10000 - 10000 - 1500
        (DECIDED, -11000),  # closed at the decision itself: no one to charge back
    ],
)
def test_item_not_received_recourse(closed_at: pd.Timestamp | None, expected: int) -> None:
    world = World()
    world.merchant(2, closed_at)
    placed = repaid(world, merchant=2)
    dispute_id, _ = world.dispute(placed, "item_not_received", 10000, FILED, NOTIFIED)
    resolution = world.resolve(dispute_id, "lost", DECIDED, DECISION_KNOWN)
    cash = world.cash()
    recourse = cash.loc[cash["kind"] == "merchant_recourse"]
    if expected == -1000:
        assert recourse["amount_cents"].tolist() == [10000]
        assert recourse["occurred_at"].tolist() == [DECISION_KNOWN]
        assert recourse["ref_event_id"].tolist() == [resolution]
    else:
        assert recourse.empty
    assert net_cents(cash) == expected


def test_liable_party() -> None:
    at = pd.Timestamp("2026-05-20 09:00:00")
    assert liable_party("unauthorized", None, at, TERMS) == "platform"
    assert liable_party("item_not_received", None, at, TERMS) == "merchant"
    assert liable_party("not_as_described", pd.NaT, at, TERMS) == "merchant"
    assert liable_party("item_not_received", pd.Timestamp("2026-05-20 09:00:01"), at,
                        TERMS) == "merchant"
    assert liable_party("item_not_received", at, at, TERMS) == "platform"
    assert liable_party("not_as_described", pd.Timestamp("2026-01-01"), at, TERMS) == "platform"
    with pytest.raises(ValueError):
        liable_party("friendly_fraud", None, at, TERMS)


# 10. A payment reversed by a bank return.

def test_reversed_installment() -> None:
    world = World()
    placed = world.order()
    world.pay(placed, 0, T0, 2500)
    first = world.pay(placed, 1, DUE_1, 2500)
    returned = pd.Timestamp("2026-03-19 10:00:00")
    reported = pd.Timestamp("2026-03-20 08:00:00")
    reversal = world.reverse(placed, first, returned, reported, 2500)
    world.fulfil(placed)
    cash = world.cash()
    rows = cash.loc[cash["kind"] == "payment_reversal"]
    assert rows["amount_cents"].tolist() == [-2500]
    assert rows["occurred_at"].tolist() == [returned]
    assert rows["known_at"].tolist() == [reported]
    assert rows["ref_event_id"].tolist() == [reversal]
    # -9500 + 2500 + 2500 - 2500 = -7000
    assert net_cents(cash) == -7000
    # Counted by when it happened versus when the platform learned of it:
    # on 2026-03-19 12:00 the reversal has happened (-7000) but is not yet known (-4500).
    noon = pd.Timestamp("2026-03-19 12:00:00")
    assert net_cents(cash, until=noon) == -7000
    assert net_cents(cash, until=noon, clock="known_at") == -4500


# 11. Fee and recovery rounding.

def test_fee_rounds_half_up() -> None:
    assert merchant_fee_cents(1999, TERMS) == 100  # 1999 x 500 / 10000 = 99.95 -> 100
    assert settlement_cents(1999, 0, TERMS) == 1899  # 1999 - 100
    assert recovery_cents(7500, TERMS) == 1500  # 7500 x 2000 / 10000
    assert recovery_cents(7, TERMS) == 1  # 7 x 2000 / 10000 = 1.4 -> 1
    assert recovery_cents(2, TERMS) == 0  # 0.4 -> 0


def test_derived_settlement_rounds_the_fee_half_up() -> None:
    world = World()
    repaid(world, amount=1999, payments=(499, 500, 500, 500))
    cash = world.cash()
    settlement = cash.loc[cash["kind"] == "merchant_settlement", "amount_cents"]
    assert settlement.tolist() == [-1899]  # 1999 - 100
    # -1899 + 499 + 500 + 500 + 500 = +100, the fee
    assert net_cents(cash) == 100


def test_derived_amounts_agree_with_the_scalar_helpers() -> None:
    tables = synthetic_world(3_000, seed=7)
    cash = derive_cash_events(tables, TERMS)
    orders = tables["order_attempts"].set_index("order_id")
    settlements = cash.loc[cash["kind"] == "merchant_settlement"]
    for order_id, amount in zip(settlements["order_id"], settlements["amount_cents"], strict=True):
        price, promo = orders.loc[order_id, ["amount_cents", "promo_discount_cents"]]
        assert -amount == settlement_cents(int(price), int(promo), TERMS)
    outstanding = tables["plan_writeoffs"].set_index("event_id")["outstanding_cents"]
    recoveries = cash.loc[cash["kind"] == "recovery"]
    assert len(recoveries) > 0
    for ref, amount in zip(recoveries["ref_event_id"], recoveries["amount_cents"], strict=True):
        assert amount == recovery_cents(int(outstanding[ref]), TERMS)


@pytest.mark.parametrize(
    ("cents", "bps", "expected"),
    [
        (1, 5000, 1),  # 0.5 -> 1
        (3, 5000, 2),  # 1.5 -> 2
        (1, 4999, 0),  # 0.4999 -> 0
        (10, 1250, 1),  # 1.25 -> 1
        (10, 1500, 2),  # 1.5 -> 2
        (10, 1550, 2),  # 1.55 -> 2
        (1990, 500, 100),  # 99.5 -> 100
        (1989, 500, 99),  # 99.45 -> 99
        (0, 500, 0),
        (12345, 10000, 12345),
        (12345, 0, 0),
    ],
)
def test_round_bps(cents: int, bps: int, expected: int) -> None:
    assert round_bps(cents, bps) == expected


def test_round_bps_takes_whole_non_negative_cents() -> None:
    with pytest.raises(ValueError):
        round_bps(-1, 500)
    with pytest.raises(TypeError):
        round_bps(19.99, 500)
    with pytest.raises(TypeError):
        round_bps(True, 500)


def test_discount_that_leaves_no_settlement_is_rejected() -> None:
    with pytest.raises(ValueError):
        settlement_cents(10000, 9500, TERMS)  # 10000 - 500 - 9500 = 0
    world = World()
    repaid(world, amount=10000, promo=9600, payments=(100, 100, 100, 100))
    with pytest.raises(ValueError, match="no merchant settlement"):
        world.cash()


# 12. Validation of cash events.

def _valid() -> pd.DataFrame:
    return cash_rows(("merchant_settlement", -9500), ("customer_payment", 2500))


def test_validate_accepts_a_valid_table() -> None:
    validate_cash_events(_valid())


def test_validate_rejects_a_positive_settlement() -> None:
    events = _valid()
    events.loc[0, "amount_cents"] = 9500
    with pytest.raises(ValueError, match="wrong sign"):
        validate_cash_events(events)


def test_validate_rejects_an_unknown_kind() -> None:
    events = _valid()
    events.loc[0, "kind"] = "writeoff"
    with pytest.raises(ValueError, match="unknown kinds"):
        validate_cash_events(events)


def test_validate_rejects_float_amounts() -> None:
    events = _valid()
    events["amount_cents"] = events["amount_cents"].astype(float)
    with pytest.raises(ValueError, match="integer cents"):
        validate_cash_events(events)


def test_validate_rejects_knowledge_before_occurrence() -> None:
    events = _valid()
    events.loc[1, "known_at"] = events.loc[1, "occurred_at"] - pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="known before"):
        validate_cash_events(events)


def test_validate_rejects_zero_amounts_and_missing_columns() -> None:
    events = _valid()
    events.loc[1, "amount_cents"] = 0
    with pytest.raises(ValueError, match="zero"):
        validate_cash_events(events)
    with pytest.raises(ValueError, match="cause"):
        validate_cash_events(_valid().drop(columns="cause"))


def test_validate_requires_a_cause_on_every_event() -> None:
    events = _valid()
    events["cause"] = [NATURAL, None]
    with pytest.raises(ValueError, match="cause"):
        validate_cash_events(events)
    events["cause"] = [NATURAL, ""]
    with pytest.raises(ValueError, match="cause"):
        validate_cash_events(events)


def test_every_kind_has_a_sign() -> None:
    assert set(CASH_KINDS.values()) == {1, -1}
    assert "writeoff" not in CASH_KINDS


# 13. Whole-world behaviour.

def test_declined_order_produces_no_cash() -> None:
    world = World()
    world.order(approved=False)
    placed = repaid(world)
    cash = world.cash()
    assert set(cash["order_id"]) == {placed.order_id}
    assert net_cents(cash) == 500  # the repaid order alone: -9500 + 4 x 2500


def test_output_columns_and_types() -> None:
    world = World()
    repaid(world)
    cash = world.cash()
    assert cash.columns.tolist() == list(CASH_COLUMNS)
    assert str(cash["event_id"].dtype) == "Int64"
    assert cash["event_id"].isna().all()
    for column in ("order_id", "plan_id", "merchant_id", "amount_cents", "ref_event_id"):
        assert str(cash[column].dtype) == "int64", column
    for column in ("occurred_at", "known_at"):
        assert str(cash[column].dtype) == "datetime64[s]", column
    assert set(cash["cause"]) == {NATURAL}


def test_empty_world_has_no_cash() -> None:
    cash = World().cash()
    assert cash.empty
    assert cash.columns.tolist() == list(CASH_COLUMNS)


def test_dispute_fee_of_zero_adds_no_event() -> None:
    world = World()
    placed = repaid(world)
    world.dispute(placed, "unauthorized", 10000, FILED, NOTIFIED)
    cash = world.cash(dataclasses.replace(TERMS, dispute_fee_cents=0))
    assert "dispute_fee" not in set(cash["kind"])
    assert net_cents(cash) == -9500  # -9500 + 10000 - 10000


def test_rows_naming_unknown_plans_or_payments_are_errors() -> None:
    world = World()
    placed = repaid(world)
    tables = world.tables()
    stray = tables["payment_attempts"].copy()
    stray.loc[0, "plan_id"] = 9999
    with pytest.raises(ValueError, match="unknown plan_id"):
        derive_cash_events({**tables, "payment_attempts": stray}, TERMS)

    failed = world.pay(placed, 1, DUE_1, 2500, ok=False)
    world.reverse(placed, failed, DUE_2, DUE_2, 2500)
    with pytest.raises(ValueError, match="successful payment"):
        world.cash()


def test_approved_order_without_plan_is_an_error() -> None:
    world = World()
    repaid(world)
    tables = world.tables()
    with pytest.raises(ValueError, match="no plan"):
        derive_cash_events({**tables, "plans": tables["plans"].iloc[0:0]}, TERMS)


def _rich_world() -> World:
    world = World()
    world.merchant(2)
    world.merchant(3, pd.Timestamp("2026-04-01 00:00:00"))
    world.order(approved=False)
    repaid(world)
    repaid(world, amount=10000, promo=1000, payments=(2250, 2250, 2250, 2250))
    for merchant, outcome in ((2, "lost"), (3, "lost"), (1, "won")):
        placed = repaid(world, amount=4000, merchant=merchant, payments=(1000, 1000, 1000, 1000))
        dispute_id, _ = world.dispute(placed, "not_as_described", 4000, FILED, NOTIFIED)
        world.resolve(dispute_id, outcome, DECIDED, DECISION_KNOWN)
    placed = defaulted(world)
    world.write_off(placed, 7500)
    placed = world.order()
    down = world.pay(placed, 0, T0, 2500)
    world.reverse(placed, down, DUE_1, DUE_1, 2500)
    world.fulfil(placed)
    return world


def test_derivation_is_deterministic_and_ignores_row_order() -> None:
    tables = _rich_world().tables()
    first = derive_cash_events(tables, TERMS)
    pd.testing.assert_frame_equal(first, derive_cash_events(tables, TERMS))
    shuffled = {
        name: frame.sample(frac=1, random_state=7).reset_index(drop=True)
        for name, frame in tables.items()
    }
    pd.testing.assert_frame_equal(first, derive_cash_events(shuffled, TERMS))
    order = list(zip(first["known_at"], first["kind"].map(list(CASH_KINDS).index),
                     first["ref_event_id"], strict=True))
    assert order == sorted(order)


def synthetic_world(n_orders: int, seed: int = 416) -> dict[str, pd.DataFrame]:
    """A large world built with array operations: every order approved but 4%,
    four payment attempts per plan, and a sprinkling of every other event."""
    rng = np.random.default_rng(seed)
    ids = itertools.count(1)

    def event_ids(n: int) -> np.ndarray:
        start = next(ids)
        for _ in range(n - 1):
            next(ids)
        return np.arange(start, start + n, dtype=np.int64)

    def stamps(base: np.ndarray, days: np.ndarray) -> np.ndarray:
        return (base + (days * 86400).astype("timedelta64[s]")).astype("datetime64[s]")

    order_id = np.arange(1, n_orders + 1, dtype=np.int64)
    ordered = np.datetime64("2026-01-01T00:00:00", "s") + rng.integers(
        0, 180 * 86400, n_orders
    ).astype("timedelta64[s]")
    amount = rng.integers(1000, 80000, n_orders).astype(np.int64)
    promo = np.where(rng.random(n_orders) < 0.1, amount // 10, 0).astype(np.int64)
    approved = rng.random(n_orders) >= 0.04
    merchant_id = (order_id % 150 + 1).astype(np.int64)
    orders = pd.DataFrame({
        "event_id": event_ids(n_orders), "order_id": order_id, "user_id": order_id,
        "merchant_id": merchant_id, "occurred_at": ordered, "known_at": ordered,
        "amount_cents": amount, "promo_id": pd.array(np.where(promo > 0, 1, 0), dtype="Int64"),
        "promo_discount_cents": promo,
        "processor_result": np.where(approved, "approved", "declined"),
    })
    plan_order = order_id[approved]
    n_plans = len(plan_order)
    plan_id = plan_order + 10_000_000
    plans = pd.DataFrame({"plan_id": plan_id, "order_id": plan_order})
    plan_time = ordered[approved]
    plan_amount = amount[approved] - promo[approved]

    seq = np.tile(np.arange(4, dtype=np.int64), n_plans)
    attempt_plan = np.repeat(plan_id, 4)
    attempt_time = stamps(np.repeat(plan_time, 4), seq * 14)
    attempts = pd.DataFrame({
        "event_id": event_ids(4 * n_plans), "plan_id": attempt_plan, "seq": seq,
        "attempt_no": np.ones(4 * n_plans, dtype=np.int64), "occurred_at": attempt_time,
        "known_at": attempt_time, "amount_cents": np.repeat(plan_amount // 4 + 1, 4),
        "result": np.where(rng.random(4 * n_plans) < 0.95, "success", "failed"),
    })
    collected = attempts.loc[attempts["result"] == "success"]
    returned = collected.sample(frac=0.01, random_state=seed)
    reversals = pd.DataFrame({
        "event_id": event_ids(len(returned)),
        "payment_event_id": returned["event_id"].to_numpy(),
        "plan_id": returned["plan_id"].to_numpy(),
        "occurred_at": stamps(returned["occurred_at"].to_numpy(), np.full(len(returned), 3)),
        "known_at": stamps(returned["occurred_at"].to_numpy(), np.full(len(returned), 4)),
        "amount_cents": returned["amount_cents"].to_numpy(), "reason": "bank_return",
    })
    shipped_at = plan_time + np.timedelta64(12 * 3600, "s")
    fulfilments = pd.DataFrame({
        "event_id": event_ids(n_plans), "order_id": plan_order,
        "occurred_at": shipped_at, "known_at": shipped_at,
    })
    disputed = rng.random(n_plans) < 0.02
    n_disputes = int(disputed.sum())
    notified = stamps(plan_time[disputed], np.full(n_disputes, 30))
    openings = pd.DataFrame({
        "event_id": event_ids(n_disputes), "dispute_id": np.arange(1, n_disputes + 1),
        "order_id": plan_order[disputed],
        "reason": np.array(["unauthorized", "item_not_received", "not_as_described"])[
            np.arange(n_disputes) % 3
        ],
        "amount_cents": plan_amount[disputed], "occurred_at": notified, "known_at": notified,
    })
    decided = stamps(notified, np.full(n_disputes, 40))
    resolutions = pd.DataFrame({
        "event_id": event_ids(n_disputes), "dispute_id": np.arange(1, n_disputes + 1),
        "outcome": np.where(np.arange(n_disputes) % 2 == 0, "lost", "won"),
        "occurred_at": decided, "known_at": decided,
    })
    unpaid = rng.random(n_plans) < 0.03
    written = stamps(plan_time[unpaid], np.full(int(unpaid.sum()), 56))
    writeoffs = pd.DataFrame({
        "event_id": event_ids(int(unpaid.sum())), "plan_id": plan_id[unpaid],
        "outstanding_cents": plan_amount[unpaid] * 3 // 4, "occurred_at": written,
        "known_at": written,
    })
    closed = np.where(np.arange(150) % 10 == 0, np.datetime64("2026-04-01T00:00:00", "s"),
                      np.datetime64("NaT", "s"))
    merchants = pd.DataFrame({"merchant_id": np.arange(1, 151, dtype=np.int64),
                              "closed_at": closed.astype("datetime64[s]")})
    return {
        "order_attempts": orders, "plans": plans, "payment_attempts": attempts,
        "payment_reversals": reversals, "fulfilments": fulfilments,
        "dispute_openings": openings, "dispute_resolutions": resolutions,
        "plan_writeoffs": writeoffs, "merchants": merchants,
    }


def test_whole_world_derivation_is_vectorised() -> None:
    tables = synthetic_world(52_000)
    started = time.perf_counter()
    cash = derive_cash_events(tables, TERMS)
    elapsed = time.perf_counter() - started
    assert elapsed < 20, f"derivation took {elapsed:.1f} s"

    approved = tables["order_attempts"]["processor_result"] == "approved"
    assert (cash["kind"] == "merchant_settlement").sum() == approved.sum()
    attempts = tables["payment_attempts"]
    collected = attempts.loc[attempts["result"] == "success", "amount_cents"].sum()
    assert cash.loc[cash["kind"] == "customer_payment", "amount_cents"].sum() == collected
    assert cash.loc[cash["kind"] == "payment_reversal", "amount_cents"].sum() == -(
        tables["payment_reversals"]["amount_cents"].sum()
    )
    assert len(cash.loc[cash["kind"] == "dispute_fee"]) == len(tables["dispute_openings"])


# Product terms.

def test_terms_from_world_config() -> None:
    assert TERMS.down_payment_bps == 2500
    assert TERMS.n_installments == 3
    assert TERMS.installment_interval_days == 14
    assert TERMS.merchant_discount_bps == 500
    assert TERMS.dispute_fee_cents == 1500
    assert TERMS.liability == {
        "unauthorized": "platform",
        "item_not_received": "merchant",
        "not_as_described": "merchant",
    }
    assert TERMS.writeoff_after_days == 14
    assert TERMS.recovery_lag_days == 30
    assert TERMS.recovery_rate_bps == 2000
    assert hash(TERMS) == hash(ProductTerms.from_config())


def _product(**changes: object) -> dict:
    product = {
        "down_payment_bps": 2500, "n_installments": 3, "installment_interval_days": 14,
        "merchant_discount_bps": 500, "dispute_fee_cents": 1500,
        "liability": {"unauthorized": "platform", "item_not_received": "merchant",
                      "not_as_described": "merchant"},
        "writeoff_after_days": 14, "recovery_lag_days": 30, "recovery_rate_bps": 2000,
        "remainder_rule": "earliest_installments_first", "promotions_funded_by": "platform",
    }
    product.update(changes)
    return {"product": product}


@pytest.mark.parametrize(
    "changes",
    [
        {"down_payment_bps": 10001},
        {"merchant_discount_bps": -1},
        {"recovery_rate_bps": 2000.0},
        {"n_installments": 0},
        {"dispute_fee_cents": "1500"},
        {"liability": {"unauthorized": "platform", "item_not_received": "merchant"}},
        {"liability": {"unauthorized": "platform", "item_not_received": "merchant",
                       "not_as_described": "issuer"}},
        {"remainder_rule": "last_installment"},
        {"promotions_funded_by": "merchant"},
        {"recovery_lag_days": None},
    ],
)
def test_terms_reject_bad_values(changes: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        ProductTerms.from_config(_product(**changes))


def test_terms_require_every_value() -> None:
    world = _product()
    del world["product"]["recovery_rate_bps"]
    with pytest.raises(ValueError, match="recovery_rate_bps"):
        ProductTerms.from_config(world)
