"""What happens after an approved order, for every actor alike.

Shipment and delivery, the installment collections (on time, late, a
zero-effort default or a partial default), reversed collections, write-offs
and disputes are dated events. The world holds each order's outcomes as they
would be if the platform approved every order (approve-all); policies act only
in the replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simulator.builder import DAY, Actor, Builder, Order
from simulator.timing import HOUR, Clock


@dataclass(frozen=True)
class OutcomeParams:
    fulfilment_sigma: float
    delivery_median_days: float
    delivery_sigma: float
    retry_days: tuple[int, int]
    second_retry_fail: float
    final_retry_success: float
    reversal_days: tuple[float, float]
    reversal_retry_success: float
    notice_days: tuple[float, float]
    decide_days: tuple[float, float]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> OutcomeParams:
        c = cfg["outcomes"]
        return cls(
            fulfilment_sigma=float(cfg["fulfilment"]["order_lag_sigma"]),
            delivery_median_days=float(c["delivery_median_days"]),
            delivery_sigma=float(c["delivery_sigma"]),
            retry_days=tuple(c["retry_days"]),
            second_retry_fail=float(c["second_retry_fail"]),
            final_retry_success=float(c["final_retry_success"]),
            reversal_days=tuple(c["reversal_days"]),
            reversal_retry_success=float(c["reversal_retry_success"]),
            notice_days=tuple(c["dispute_notice_days"]),
            decide_days=tuple(c["dispute_decision_days"]),
        )


class Outcomes:
    def __init__(self, b: Builder, clock: Clock, params: OutcomeParams) -> None:
        self.b, self.clock, self.p = b, clock, params
        self.lag_from: int | None = None
        self.lag_factor = 1.0

    def scale_lag(self, start: int, factor: float) -> None:
        """Multiply the drawn fulfilment lag of every order placed at or after ``start``
        by ``factor`` (a sensitivity family); the draws themselves are unchanged."""
        self.lag_from, self.lag_factor = start, factor

    # ------------------------------------------------------- fulfilment
    def ship_time(self, a: Actor, o: Order) -> int:
        median = self.b.merchants[o.merchant]["median"]
        hours = float(np.exp(a.rng.normal(np.log(median), self.p.fulfilment_sigma)))
        if self.lag_from is not None and o.t >= self.lag_from:
            hours *= self.lag_factor
        return o.t + int(max(0.25, hours) * HOUR)

    def vanishing(self, o: Order) -> bool:
        """Whether the order was placed while its merchant was busting out."""
        start = self.b.merchants[o.merchant]["bustout_from"]
        return start is not None and o.t >= start

    def fulfil(self, a: Actor, o: Order, *, deliver: bool = True) -> None:
        """The merchant reports shipment; the carrier confirms delivery unless ``deliver``
        is false. A merchant busting out reports shipments before it disappears and
        delivers nothing, whoever the buyer is."""
        shipped = self.ship_time(a, o)
        if self.vanishing(o):
            closed = self.b.merchants[o.merchant]["closed"]
            shipped = max(o.t + 600, min(shipped, closed - HOUR))
            deliver = False
        self.b.ship(a, o, shipped)
        if deliver:
            days = float(np.exp(a.rng.normal(np.log(self.p.delivery_median_days),
                                              self.p.delivery_sigma)))
            self.b.deliver(a, o, shipped + int(max(0.5, days) * DAY))

    # ------------------------------------------------------- repayment
    def collect(self, a: Actor, o: Order, plan: str = "paid", *, late_p: float = 0.0,
                reversal_p: float = 0.0, stop_seq: int = 1, stop_at: int | None = None) -> None:
        """Collect installments 1..n. ``plan``: ``paid`` (every installment paid,
        some late), ``zero_effort`` (nothing after the checkout payment) or
        ``partial`` (installments from ``stop_seq`` on are never paid). Installments
        due at or after ``stop_at`` (a card blocked, a payer who stops) are never
        paid either. Unpaid installments fail on their due date and on one retry;
        the plan is written off when a balance remains."""
        b, rng = self.b, a.rng
        lo, hi = self.p.retry_days
        writeoff = b.writeoff_time(o)
        for seq in range(1, len(o.schedule)):
            due = o.due(seq)
            if plan == "zero_effort" or (plan == "partial" and seq >= stop_seq) \
                    or (stop_at is not None and due >= stop_at):
                b.pay(a, o, seq, due, success=False)
                b.pay(a, o, seq, due + lo * DAY, success=False, attempt_no=2)
                continue
            attempt, when = 1, due
            if rng.random() < late_p:
                b.pay(a, o, seq, when, success=False)
                attempt, when = 2, due + lo * DAY
                if rng.random() < self.p.second_retry_fail:
                    b.pay(a, o, seq, when, success=False, attempt_no=2)
                    attempt, when = 3, due + hi * DAY
                    if rng.random() >= self.p.final_retry_success:
                        b.pay(a, o, seq, when, success=False, attempt_no=3)
                        continue
            payment = b.pay(a, o, seq, when, attempt_no=attempt)
            if payment is not None and rng.random() < reversal_p:
                back = when + int(rng.uniform(*self.p.reversal_days) * DAY)
                retry = back + 2 * DAY
                if retry < writeoff:
                    b.reverse(a, o, payment, back)
                    b.pay(a, o, seq, retry, success=rng.random() < self.p.reversal_retry_success,
                          attempt_no=attempt + 1)
        b.write_off(a, o)

    # --------------------------------------------------------- disputes
    def dispute(self, a: Actor, o: Order, reason: str, filed: int, outcome: str) -> None:
        rng = a.rng
        notified = filed + int(rng.uniform(*self.p.notice_days) * DAY)
        decided = notified + int(rng.uniform(*self.p.decide_days) * DAY)
        known = decided + int(rng.uniform(0, 2) * DAY)
        self.b.dispute(a, o, reason, filed, notified, outcome, decided, known)

    def file_time(self, a: Actor, after: int, lo_days: float, hi_days: float) -> int:
        return self.clock.after(a.rng, after, lo_days, hi_days)
