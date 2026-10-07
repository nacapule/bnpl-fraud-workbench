"""Policies: how an order is routed at checkout and where it waits in the review queue.

A policy routes each order once, at its checkout, from the order's as-of context row
alone, with thresholds fixed before the replay starts: no ranking within a day, no
look at other orders (a top-k selection would let tomorrow's volume decide today's
review). An order goes to ``auto_decline`` when its decline score reaches the
decline threshold, otherwise to ``review`` when its review score reaches the review
threshold, otherwise to ``approve``. A threshold of ``None`` switches that route off.

Detectors come from the fitted models as :class:`Scorer` objects (the rules score,
the depth-3 tree, logistic regression, boosting), each scoring a row on its own.
:func:`policy_set` builds the seven policies of the protocol from them:

* ``approve_all`` (reference), ``incumbent_rules``, ``tree_depth3``, ``logistic``,
  ``boosting``: one scorer for both thresholds;
* ``hybrid``: the rules score auto-declines (a high threshold), boosting routes to
  review and orders the queue;
* ``expected_loss``: the calibrated probability times the order's exposure is the
  review score; an order is declined when that expected loss exceeds the expected cost
  of declining a legitimate order (merchant fee plus the LTV proxy), so its decline
  threshold is fixed at zero rather than tuned.

Queue priority (FP-2 §7.1) is assigned mechanically at queue entry from facts known
then and never changes; the review score orders each priority's queue.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from core import evidence, ledger
from core.actions import CheckoutRoute

PRIORITIES = ("P0", "P1", "P2", "P3")


@runtime_checkable
class Scorer(Protocol):
    """A fitted detector (model.train.fit): scores each context row on its own."""

    name: str
    version: str
    columns: tuple[str, ...]

    def score(self, rows: pd.DataFrame) -> np.ndarray: ...

    def probability(self, rows: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class Signal:
    """One number per row that a threshold is compared with."""

    name: str
    version: str
    columns: tuple[str, ...]
    compute: Callable[[pd.DataFrame], np.ndarray] = field(compare=False, repr=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.name, self.version)

    def __call__(self, rows: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.compute(rows), dtype=np.float64)
        if values.shape != (len(rows),):
            raise ValueError(f"{self.name} returned {values.shape} for {len(rows)} rows")
        return values


def score_of(scorer: Scorer) -> Signal:
    return Signal(f"{scorer.name}.score", scorer.version, tuple(scorer.columns), scorer.score)


def expected_loss_cents(scorer: Scorer) -> Signal:
    """Calibrated probability times the order's exposure (cents at risk if approved)."""
    columns = tuple(dict.fromkeys((*scorer.columns, "order_exposure_cents")))
    return Signal(f"{scorer.name}.expected_loss", scorer.version, columns,
                  lambda rows: scorer.probability(rows)
                  * rows["order_exposure_cents"].to_numpy(dtype=float))


def decline_benefit_cents(scorer: Scorer, terms: ledger.ProductTerms, ltv_cents: int) -> Signal:
    """Expected loss avoided by declining minus the expected cost of a false decline."""
    columns = tuple(dict.fromkeys((*scorer.columns, "order_exposure_cents", "amount_cents")))

    def compute(rows: pd.DataFrame) -> np.ndarray:
        p = scorer.probability(rows)
        exposure = rows["order_exposure_cents"].to_numpy(dtype=float)
        fee = rows["amount_cents"].to_numpy(dtype=float) * terms.merchant_discount_bps / 10_000
        return p * exposure - (1 - p) * (fee + ltv_cents)

    return Signal(f"{scorer.name}.decline_benefit@{ltv_cents}", scorer.version, columns,
                  compute)


@dataclass(frozen=True)
class Policy:
    """Checkout routing: signals, thresholds fixed in advance, and the queue order."""

    name: str
    review: Signal | None = None
    review_threshold: float | None = None
    decline: Signal | None = None
    decline_threshold: float | None = None
    tunable: tuple[str, ...] = ("review", "decline")  # thresholds the tuning may set

    @property
    def columns(self) -> tuple[str, ...]:
        signals = [s for s in (self.review, self.decline) if s is not None]
        return tuple(dict.fromkeys(c for s in signals for c in s.columns))

    @property
    def version(self) -> str:
        """Short hash of everything that decides a route: alert id = order id + this."""
        spec = {
            "name": self.name,
            "review": None if self.review is None else [self.review.name, self.review.version],
            "review_threshold": self.review_threshold,
            "decline": None if self.decline is None else [self.decline.name,
                                                          self.decline.version],
            "decline_threshold": self.decline_threshold,
        }
        text = json.dumps(spec, sort_keys=True, default=repr)
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def with_thresholds(self, review: float | None, decline: float | None) -> Policy:
        return replace(self, review_threshold=review, decline_threshold=decline)

    def route(self, rows: pd.DataFrame, *,
              known: Mapping[tuple[str, str], pd.Series] | None = None,
              fresh: np.ndarray | None = None) -> pd.DataFrame:
        """Each row's route and the review score that orders the queue.

        Every row is routed on its own values: the result for one order never depends
        on which other rows are passed. ``known`` may hold each signal's scores of the
        world-level checkout rows (by order id); they are reused for rows not marked
        ``fresh`` (rows whose context differs from the world-level row are rescored).
        """
        n = len(rows)

        def values(signal: Signal | None) -> np.ndarray:
            if signal is None or not n:
                return np.full(n, np.nan)
            if known is None or signal.key not in known:
                return signal(rows)
            out = known[signal.key].reindex(rows["order_id"].to_numpy()).to_numpy(
                dtype=float, copy=True)
            redo = np.isnan(out) if fresh is None else (np.isnan(out) | fresh)
            if redo.any():
                out[redo] = signal(rows.loc[redo])
            return out

        review = values(self.review)
        decline = review if self.decline is self.review else values(self.decline)
        to_review = (np.zeros(n, dtype=bool) if self.review_threshold is None
                     else review >= self.review_threshold)
        to_decline = (np.zeros(n, dtype=bool) if self.decline_threshold is None
                      else decline >= self.decline_threshold)
        route = np.where(to_decline, CheckoutRoute.AUTO_DECLINE.value,
                         np.where(to_review, CheckoutRoute.REVIEW.value,
                                  CheckoutRoute.APPROVE.value))
        return pd.DataFrame({"order_id": rows["order_id"].to_numpy(dtype=np.int64),
                             "route": route, "review_score": review,
                             "decline_score": decline}, index=rows.index)


def approve_all() -> Policy:
    return Policy("approve_all", tunable=())


def single_scorer(name: str, scorer: Scorer) -> Policy:
    signal = score_of(scorer)
    return Policy(name, review=signal, decline=signal)


def hybrid(rules: Scorer, boosting: Scorer) -> Policy:
    return Policy("hybrid", review=score_of(boosting), decline=score_of(rules))


def expected_loss(scorer: Scorer, terms: ledger.ProductTerms, ltv_cents: int) -> Policy:
    return Policy("expected_loss", review=expected_loss_cents(scorer),
                  decline=decline_benefit_cents(scorer, terms, ltv_cents),
                  decline_threshold=0.0, tunable=("review",))


def policy_set(scorers: Mapping[str, Scorer], terms: ledger.ProductTerms,
               ltv_cents: int) -> dict[str, Policy]:
    """The protocol's seven policies, untuned (thresholds set by tuning)."""
    return {
        "approve_all": approve_all(),
        "incumbent_rules": single_scorer("incumbent_rules", scorers["rules"]),
        "tree_depth3": single_scorer("tree_depth3", scorers["tree"]),
        "logistic": single_scorer("logistic", scorers["logistic"]),
        "boosting": single_scorer("boosting", scorers["boosting"]),
        "hybrid": hybrid(scorers["rules"], scorers["boosting"]),
        "expected_loss": expected_loss(scorers["boosting"], terms, ltv_cents),
    }


def priority(rows: pd.DataFrame, entered_at: np.ndarray, approved_at: np.ndarray,
             *, shipped: np.ndarray | None = None,
             cancelled: np.ndarray | None = None) -> np.ndarray:
    """FP-2 §7.1 priority at queue entry, from facts known then (rules as they hold).

    P3 if the order had shipped or been cancelled at entry; otherwise P0 if the
    merchant's median fulfilment time leaves under 2 hours before shipment
    (approval + median < entry + 2 h) or R05 or R07 holds; P1 if the amount is at least
    $500 or R02, R08 or R10 holds; otherwise P2.
    """
    n = len(rows)
    held = evidence.conditions(rows)
    entered = np.asarray(entered_at, dtype="datetime64[s]")
    approved = np.asarray(approved_at, dtype="datetime64[s]")
    ships = approved + (rows["merchant_fulfilment_median_hours"].to_numpy(dtype=float)
                        * 3600).astype("timedelta64[s]")
    urgent = (ships < entered + np.timedelta64(2 * 3600, "s")) \
        | held["R05"].to_numpy() | held["R07"].to_numpy()
    large = (rows["amount_cents"].to_numpy() >= 50_000) | held["R02"].to_numpy() \
        | held["R08"].to_numpy() | held["R10"].to_numpy()
    late = np.zeros(n, dtype=bool)
    for flags in (shipped, cancelled):
        if flags is not None:
            late |= np.asarray(flags, dtype=bool)
    return np.where(late, "P3", np.where(urgent, "P0", np.where(large, "P1", "P2")))


PRIORITY_COLUMNS = ("merchant_fulfilment_median_hours", "amount_cents")


def alert_id(order_ids: Any, version: str) -> np.ndarray:
    """Stable alert identity: the order id plus the policy version."""
    return np.array([f"{int(order)}:{version}" for order in np.atleast_1d(order_ids)],
                    dtype=object)
