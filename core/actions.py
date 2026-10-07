"""Actions and their consequences: what one policy's decisions do to the world's orders.

The world holds every attempted order with the outcomes it would have if approved;
policies act only in the replay (``queue_sim``), which records what it decided for
each order as a row of :data:`FATE_COLUMNS` (a *fate*). This module turns fates into
the policy's realized observations, its cash and its :class:`core.asof.PolicyState`,
so every consumer applies the same effects. Vocabulary (fraud policy FP-2 §4):

Checkout routing, decided at arrival from the as-of context with thresholds fixed in
advance (no within-day ranking):

* ``approve``: the order proceeds; its potential outcomes are realized.
* ``review``: the order proceeds and enters the review queue (reviewer minutes);
  fulfilment continues unless the reviewer holds or declines before shipment.
* ``auto_decline``: no order, no cash events, no reviewer minutes; records no fraud
  finding and blocks no account. A legitimate decline loses the merchant fee (the
  ledger shows it: the order's cash never happens) and an LTV proxy, and counts as
  friction.
* ``blocked`` (route only): the account was blocked by an earlier decline or
  escalation, so the order is declined at checkout as ``auto_decline`` is, and counts
  as friction when legitimate.

Analyst dispositions, applied when the review or a check completes:

* ``clear``: nothing changes.
* ``hold``: runs the verification checks. Before shipment it pauses shipment and
  merchant settlement up to ``config/policy.yaml`` ``actions.hold_max_hours``. A
  customer who verifies is released: the shipment and everything after it move by
  the time the hold paused it, and the repayment schedule starts at the release.
  A customer who does not verify in time is cancelled with the checkout payment
  refunded and no block. After shipment nothing is paused and no response changes
  nothing.
* ``decline``: before shipment, voids the order (cash already moved stays and is
  refunded); after shipment the loss stands (the order's cash is unchanged).
  Either way the account is blocked and its later orders are declined.
* ``escalate``: a decline that also blocks the accounts linked to the order
  (FP-2 §2.5, ``core.asof.linked_accounts`` as known at the decision), plus senior
  review minutes of fraud-queue work (``queue_sim.replay``).

Verification checks (at most two per order, FP-2 §5.1): ``contact`` and ``id_check``,
with outcomes ``passed``, ``failed`` or ``no_response``. The memo drafter may suggest
``needs_check`` (a hold with a named check); it never acts.

Effects on the realized tables, per order (``tables`` in the world schema):

=====================  ===================================================================
fate                   realized
=====================  ===================================================================
approve, review        the world's rows, unchanged (also after a post-shipment hold or a
(cleared or after      decline after shipment: the loss stands)
shipment)
auto_decline, blocked  the attempt only: no plan, schedule, payments, shipment, disputes,
                       reports or write-off
voided at ``v``        rows that occurred by ``v`` (the checkout payment); the plan and its
(before shipment)      whole schedule stay (the columns stop counting the plan from the
                       void), every later event goes; the ledger refunds each payment
                       still standing at ``v``
held at ``h`` before   what occurred by ``h`` stands; everything after ``h`` waited for
shipment, released     the release: installments not yet due at ``h`` (schedule rows,
at ``r``               their payments and reversals after ``h``) and a write-off after
                       ``h`` move by ``r - checkout`` (the schedule starts at the
                       release); shipment, delivery, disputes, victim reports and the
                       payments and reversals of installments already due at ``h`` (the
                       checkout payment's included) move by ``r - h``, and so does cash
                       that rows kept at their time lead to after ``h`` (a dispute
                       notified during the hold, a recovery); events moved past the
                       observation end are dropped
held before shipment,  what occurred by ``h`` stands, nothing after it has happened yet
                       (no cash after ``h`` either),
still pending          and installments not yet due at ``h`` are not scheduled (they do
                       not fall due while the order is paused); neither approved nor
                       voided in the state
=====================  ===================================================================

So an action never changes what was known before it: realizing the fates as decided
at any moment gives the same rows up to that moment as realizing the final fates.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from core import ledger
from core.asof import PolicyState


class CheckoutRoute(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    AUTO_DECLINE = "auto_decline"


BLOCKED = "blocked"  # declined at checkout because the account was blocked
ROUTES = (*(route.value for route in CheckoutRoute), BLOCKED)
DECLINED_AT_CHECKOUT = (CheckoutRoute.AUTO_DECLINE.value, BLOCKED)


class Disposition(StrEnum):
    CLEAR = "clear"
    HOLD = "hold"
    DECLINE = "decline"
    ESCALATE = "escalate"


MEMO_DISPOSITIONS = (*(d.value for d in Disposition), "needs_check")


class Check(StrEnum):
    CONTACT = "contact"
    ID_CHECK = "id_check"


class CheckOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NO_RESPONSE = "no_response"


# How a hold ended (core.asof.PolicyState.held outcome vocabulary).
HOLD_OUTCOMES = ("cleared", "cancelled", "declined")
# Why an order was voided before shipment (the ledger's refund cause).
VOID_CAUSES = ("decline", "escalate", "hold_cancelled")

# One row per order the policy saw (processor-approved attempts in its window).
FATE_COLUMNS: dict[str, str] = {
    "order_id": "int64",
    "user_id": "int64",
    "checkout_at": "datetime64[s]",
    "route": "str",
    "hold_at": "datetime64[s]",  # NaT: never held
    "hold_before_shipment": "bool",
    "released_at": "datetime64[s]",  # a pre-shipment hold's release; NaT otherwise
    "hold_outcome": "object",  # HOLD_OUTCOMES, or None while pending / never held
    "hold_ended_at": "datetime64[s]",  # when the outcome came; NaT while pending / never held
    "void_at": "datetime64[s]",  # voided before shipment; NaT otherwise
    "void_cause": "object",  # VOID_CAUSES or None
}


def empty_fates() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in FATE_COLUMNS.items()})


def approve_all_fates(orders: pd.DataFrame) -> pd.DataFrame:
    """Every processor-approved order approved at checkout (the reference policy)."""
    approved = orders.loc[orders["processor_result"] == "approved"]
    fates = empty_fates().reindex(range(len(approved)))
    fates["order_id"] = approved["order_id"].to_numpy(dtype=np.int64)
    fates["user_id"] = approved["user_id"].to_numpy(dtype=np.int64)
    fates["checkout_at"] = approved["known_at"].to_numpy().astype("datetime64[s]")
    fates["route"] = CheckoutRoute.APPROVE.value
    fates["hold_before_shipment"] = False
    return typed_fates(fates)


def typed_fates(fates: pd.DataFrame) -> pd.DataFrame:
    """``fates`` with :data:`FATE_COLUMNS` in order and type, checked for consistency."""
    missing = [name for name in FATE_COLUMNS if name not in fates.columns]
    if missing:
        raise ValueError(f"fates lack columns {missing}")
    out = pd.DataFrame({
        name: (fates[name].astype(object).where(fates[name].notna(), None)
               if dtype == "object" else fates[name].astype(dtype))
        for name, dtype in FATE_COLUMNS.items()
    }).reset_index(drop=True)
    problems = []
    if out["order_id"].duplicated().any():
        problems.append("an order has two fates")
    if not out["route"].isin(ROUTES).all():
        problems.append(f"unknown routes {sorted(set(out['route']) - set(ROUTES))}")
    checkout_declined = out["route"].isin(DECLINED_AT_CHECKOUT)
    acted = out["hold_at"].notna() | out["void_at"].notna()
    if (checkout_declined & acted).any():
        problems.append("an order declined at checkout was also held or voided")
    if not out["hold_outcome"].dropna().isin(HOLD_OUTCOMES).all():
        problems.append("unknown hold outcomes")
    if not out["void_cause"].dropna().isin(VOID_CAUSES).all():
        problems.append("unknown void causes")
    if (out["void_at"].notna() != out["void_cause"].notna()).any():
        problems.append("void_at and void_cause must be set together")
    held = out["hold_at"].notna()
    ended = out["hold_ended_at"].notna()
    if (~held & (out["hold_before_shipment"] | out["hold_outcome"].notna() | ended)).any():
        problems.append("hold details without a hold")
    if (ended != out["hold_outcome"].notna()).any():
        problems.append("hold_outcome and hold_ended_at must be set together")
    if (ended & (out["hold_ended_at"] < out["hold_at"])).any():
        problems.append("a hold that ends before it starts")
    released = out["released_at"].notna()
    cleared_before = held & out["hold_before_shipment"] & (out["hold_outcome"] == "cleared")
    if (released != cleared_before).any():
        problems.append("only a cleared pre-shipment hold has a release time, and it must")
    if (released & (out["released_at"] != out["hold_ended_at"])).any():
        problems.append("a release must be when the hold ended")
    stopped = held & out["hold_before_shipment"] & out["hold_outcome"].isin(
        ["cancelled", "declined"])
    if (stopped != (held & out["hold_before_shipment"] & out["void_at"].notna())).any() or (
            stopped & (out["void_at"] != out["hold_ended_at"])).any():
        problems.append("a pre-shipment hold that is cancelled or declined is voided when it ends")
    if (out["hold_at"].notna() & (out["hold_at"] < out["checkout_at"])).any():
        problems.append("a hold before checkout")
    if (out["void_at"].notna() & (out["void_at"] < out["checkout_at"])).any():
        problems.append("a void before checkout")
    if (released & out["void_at"].notna()).any():
        problems.append("an order both released and voided")
    if problems:
        raise ValueError("invalid fates: " + "; ".join(problems))
    return out


def fates_as_of(fates: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """What had been decided strictly before ``at``: later orders absent, later outcomes pending.

    A hold placed before ``at`` whose outcome came at or after it is pending (no
    release, outcome, end or void); a void at or after ``at`` has not happened yet.
    """
    at = pd.Timestamp(at)
    out = fates.loc[fates["checkout_at"] < at].copy()
    if not any((out[c] >= at).any() for c in ("hold_at", "hold_ended_at", "void_at")):
        return typed_fates(out)  # nothing was decided at or after ``at``
    late_hold = out["hold_at"] >= at
    out.loc[late_hold, ["hold_at", "released_at", "hold_ended_at", "void_at"]] = pd.NaT
    out.loc[late_hold, ["hold_outcome", "void_cause"]] = None
    out.loc[late_hold, "hold_before_shipment"] = False
    pending = out["hold_at"].notna() & ~(out["hold_ended_at"] < at)
    out.loc[pending, ["released_at", "hold_ended_at"]] = pd.NaT
    out.loc[pending, "hold_outcome"] = None
    paused = pending & out["hold_before_shipment"]
    out.loc[paused, "void_at"] = pd.NaT
    out.loc[paused, "void_cause"] = None
    late_void = out["void_at"] >= at
    out.loc[late_void, "void_at"] = pd.NaT
    out.loc[late_void, "void_cause"] = None
    return typed_fates(out)


# ------------------------------------------------------------------ realized tables

# Tables whose rows belong to one order, how each row finds it, and which clock moves
# it under a released pre-shipment hold ("ship": by the pause; "pay": by the schedule's
# start; "fixed": never).
_SHIP_TABLES = ("fulfilments", "deliveries", "dispute_openings", "dispute_resolutions",
                "victim_reports")
_PAY_TABLES = ("payment_attempts", "payment_reversals", "plan_writeoffs")
_TIME_COLUMNS = ("occurred_at", "known_at")


def _order_of_rows(tables: Mapping[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """For each order-bound table, the order id of every row."""
    plan_order = tables["plans"].set_index("plan_id")["order_id"]
    dispute_order = tables["dispute_openings"].set_index("dispute_id")["order_id"]
    rows = {
        name: tables[name]["order_id"].to_numpy(dtype=np.int64)
        for name in ("order_attempts", "plans", "fulfilments", "deliveries",
                     "dispute_openings", "victim_reports", "cash_events")
    }
    for name in ("installment_schedule", "payment_attempts", "payment_reversals",
                 "plan_writeoffs"):
        rows[name] = tables[name]["plan_id"].map(plan_order).to_numpy(dtype=np.int64)
    rows["dispute_resolutions"] = (
        tables["dispute_resolutions"]["dispute_id"].map(dispute_order).to_numpy(dtype=np.int64))
    return rows


def _installment_rows(tables: Mapping[str, pd.DataFrame], name: str) -> np.ndarray:
    """Whether each row belongs to installments 1..n (not the checkout payment)."""
    frame = tables[name]
    if name in ("installment_schedule", "payment_attempts"):
        return frame["seq"].to_numpy() >= 1
    if name == "payment_reversals":
        seq = tables["payment_attempts"].set_index("event_id")["seq"]
        return frame["payment_event_id"].map(seq).to_numpy() >= 1
    return np.ones(len(frame), dtype=bool)  # plan_writeoffs


def _due_of_rows(tables: Mapping[str, pd.DataFrame], name: str) -> np.ndarray:
    """The due time of each row's installment (NaT for a write-off)."""
    schedule = tables["installment_schedule"]
    due = pd.Series(schedule["due_at"].to_numpy(dtype="datetime64[s]"),
                    index=pd.MultiIndex.from_frame(schedule[["plan_id", "seq"]]))
    frame = tables[name]
    if name in ("installment_schedule", "payment_attempts"):
        keys = frame[["plan_id", "seq"]]
    elif name == "payment_reversals":
        payments = tables["payment_attempts"].set_index("event_id")
        keys = pd.DataFrame({"plan_id": frame["plan_id"].to_numpy(),
                             "seq": frame["payment_event_id"].map(payments["seq"]).to_numpy()})
    else:
        return np.full(len(frame), np.datetime64("NaT", "s"), dtype="datetime64[s]")
    return due.reindex(pd.MultiIndex.from_frame(keys.astype("int64"))).to_numpy(
        dtype="datetime64[s]")


def realize(
    tables: Mapping[str, pd.DataFrame],
    fates: pd.DataFrame,
    terms: ledger.ProductTerms,
    *,
    observed_until: pd.Timestamp | None = None,
    memo: dict[Any, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """The policy's realized observations in the world schema (see the module table).

    ``tables`` are the world's tables; ``fates`` say what the policy did, for the
    orders it saw (:func:`fates_as_of` for a moment inside the replay); orders without
    a fate keep their world rows. Events a released hold moves past
    ``observed_until`` (known after it) are dropped, and so is the cash they lead to.
    Entity tables, account events and the attempts themselves are unchanged;
    ``cash_events`` is the policy's ledger
    (:func:`policy_cash`, which ``memo`` serves).
    """
    typed = typed_fates(fates)
    out = _realize_rows(tables, typed, observed_until)
    out["cash_events"] = _policy_cash(tables, typed, terms, observed_until, memo)
    return out


def _realize_rows(
    tables: Mapping[str, pd.DataFrame],
    fates: pd.DataFrame,
    observed_until: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    owner = _order_of_rows(tables)
    by_order = fates.set_index("order_id")
    declined = set(by_order.index[by_order["route"].isin(DECLINED_AT_CHECKOUT)])
    cut_at = _cut_times(by_order)
    pending = by_order.loc[by_order["hold_at"].notna() & by_order["hold_before_shipment"]
                           & by_order["hold_outcome"].isna(), "hold_at"]
    released = by_order.loc[by_order["released_at"].notna()]
    held_at = released["hold_at"]
    ship_shift = released["released_at"] - released["hold_at"]
    pay_shift = released["released_at"] - released["checkout_at"]
    limit = None if observed_until is None else np.datetime64(pd.Timestamp(observed_until), "s")

    def per_row(order: np.ndarray, values: pd.Series, dtype: str) -> np.ndarray:
        return pd.Series(order).map(values).to_numpy(dtype=dtype)

    out: dict[str, pd.DataFrame] = {}
    for name, frame in tables.items():
        if name not in owner or name in ("order_attempts", "cash_events"):
            out[name] = frame
            continue
        order = owner[name]
        keep = ~np.isin(order, list(declined))
        if name == "plans":
            out[name] = frame.loc[keep].reset_index(drop=True)
            continue
        frame = frame.copy()
        schedule = name == "installment_schedule"
        due = (_due_of_rows(tables, name) if name == "installment_schedule"
               or name in _PAY_TABLES else None)
        if schedule:
            if len(pending):  # not yet due at a pending pre-shipment hold: not scheduled
                hold = per_row(order, pending, "datetime64[s]")
                keep &= ~(~np.isnat(hold) & _installment_rows(tables, name) & (due > hold))
        elif len(cut_at):
            cut = per_row(order, cut_at, "datetime64[s]")
            happened = frame["occurred_at"].to_numpy(dtype="datetime64[s]") <= cut
            keep &= np.isnat(cut) | happened
        if not len(released):
            out[name] = frame.loc[keep].reset_index(drop=True)
            continue
        hold = per_row(order, held_at, "datetime64[s]")
        if schedule:  # installments not yet due at the hold start at the release
            moved = ~np.isnat(hold) & _installment_rows(tables, name) & (due > hold)
            delta = per_row(order, pay_shift, "timedelta64[s]")
            values = frame["due_at"].to_numpy(dtype="datetime64[s]").copy()
            values[moved] = values[moved] + delta[moved]
            frame["due_at"] = values
            out[name] = frame.loc[keep].reset_index(drop=True)
            continue
        after = frame["occurred_at"].to_numpy(dtype="datetime64[s]") > hold
        moved = ~np.isnat(hold) & after  # what followed the hold waited for the release
        pause = per_row(order, ship_shift, "timedelta64[s]")
        if name in _SHIP_TABLES:
            delta = pause
        elif name == "plan_writeoffs":
            delta = per_row(order, pay_shift, "timedelta64[s]")
        else:  # installments not yet due at the hold start at the release; the rest pause
            not_due = _installment_rows(tables, name) & (due > hold)
            delta = np.where(not_due, per_row(order, pay_shift, "timedelta64[s]"), pause)
        for column in _TIME_COLUMNS:
            values = frame[column].to_numpy(dtype="datetime64[s]").copy()
            values[moved] = values[moved] + delta[moved]
            frame[column] = values
        if limit is not None:  # observed by the end: known by then (and so occurred)
            keep &= ~(moved & (frame["known_at"].to_numpy(dtype="datetime64[s]") > limit))
        out[name] = frame.loc[keep].reset_index(drop=True)
    return out


def _shifted_ids(before: Mapping[str, pd.DataFrame],
                 after: Mapping[str, pd.DataFrame]) -> set[int]:
    """Event ids whose time realization moved."""
    out: set[int] = set()
    for name in (*_SHIP_TABLES, *_PAY_TABLES):
        frame = after[name]
        if not len(frame):
            continue
        was = before[name].set_index("event_id")["occurred_at"].reindex(frame["event_id"])
        now = frame["occurred_at"].to_numpy(dtype="datetime64[s]")
        out |= set(frame.loc[now != was.to_numpy(dtype="datetime64[s]"), "event_id"].tolist())
    return out


def _cut_times(by_order: pd.DataFrame) -> pd.Series:
    """When each voided or still-held order stopped: nothing after it happened.

    A hold placed before shipment pauses the order at the hold, so a later void
    (cancellation or decline) keeps only what happened by the hold; a void without
    such a hold keeps what happened by the void; a pending pre-shipment hold keeps
    what happened by the hold. A hold after shipment pauses nothing.
    """
    pre_hold = by_order["hold_at"].notna() & by_order["hold_before_shipment"]
    pending = pre_hold & by_order["hold_outcome"].isna()
    cut = by_order["void_at"].where(~pre_hold, by_order["hold_at"])
    cut = cut.where(by_order["void_at"].notna() | pending)
    return cut.dropna()


# ------------------------------------------------------------------ cash


def policy_cash(
    tables: Mapping[str, pd.DataFrame],
    fates: pd.DataFrame,
    terms: ledger.ProductTerms,
    *,
    observed_until: pd.Timestamp | None = None,
    memo: dict[Any, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """The ledger of the world's orders under the policy (natural events plus refunds).

    Orders the policy left alone (or without a fate) keep the world's cash events;
    orders declined at checkout have none; an order voided before shipment keeps the
    cash realized by the void and gets a refund of each payment still standing
    (:func:`core.ledger.cancel_before_fulfilment`, cause = the void's cause); a
    released hold's cash is derived from its moved events; a pre-shipment hold still
    pending keeps the cash realized by the hold and nothing after it (a hold after
    shipment changes nothing). Natural events keep the
    world's event ids (matched by kind and reference); a refund's id is the negative
    of the payment event it returns, so ids stay unique and the same in every policy.

    ``memo`` (a dict kept by the caller for one world) remembers each voided order's
    cash by its fate, for callers that realize growing sets of fates on the same world.
    """
    return _policy_cash(tables, typed_fates(fates), terms, observed_until, memo)


def _policy_cash(
    tables: Mapping[str, pd.DataFrame],
    fates: pd.DataFrame,
    terms: ledger.ProductTerms,
    observed_until: pd.Timestamp | None,
    memo: dict[Any, pd.DataFrame] | None,
) -> pd.DataFrame:
    world_cash = tables["cash_events"]
    pending = fates["hold_at"].notna() & fates["hold_outcome"].isna() \
        & fates["hold_before_shipment"]
    declined = fates.loc[fates["route"].isin(DECLINED_AT_CHECKOUT), "order_id"]
    voided = fates.loc[fates["void_at"].notna()]
    held = fates.loc[pending]
    released = fates.loc[fates["released_at"].notna()]
    changed = set(declined) | set(voided["order_id"]) | set(released["order_id"]) \
        | set(held["order_id"])
    parts = [world_cash.loc[~world_cash["order_id"].isin(changed)]]

    touched = world_cash.loc[world_cash["order_id"].isin(set(voided["order_id"])
                                                         | set(held["order_id"]))]
    by_order = dict(tuple(touched.groupby("order_id", sort=False)))
    cut_at = _cut_times(fates.set_index("order_id"))
    reversals = tables["payment_reversals"]
    for order, at, cause in voided[["order_id", "void_at", "void_cause"]].itertuples(index=False):
        events = by_order.get(order)
        if events is None or not len(events):
            continue
        key = (int(order), at, cut_at[order], cause)
        if memo is not None and key in memo:
            parts.append(memo[key])
            continue
        events = events.loc[events["occurred_at"] <= cut_at[order]]
        cash = ledger.cancel_before_fulfilment(events, at, cause, reversals=reversals)
        parts.append(cash)
        if memo is not None:
            memo[key] = cash
    for order, at in held[["order_id", "hold_at"]].itertuples(index=False):
        events = by_order.get(order)
        if events is not None and len(events):
            parts.append(events.loc[events["occurred_at"] <= at])

    if len(released):
        parts += _released_cash(tables, released, terms, observed_until, memo)
    cash = pd.concat([part for part in parts if len(part)] or [world_cash.iloc[0:0]],
                     ignore_index=True)
    refunds = cash["kind"].eq("refund") & cash["event_id"].isna()
    if refunds.any():
        cash["event_id"] = pd.array(cash["event_id"], dtype="Int64")
        cash.loc[refunds, "event_id"] = -cash.loc[refunds, "ref_event_id"].to_numpy()
    return cash.reset_index(drop=True)


def _released_cash(
    tables: Mapping[str, pd.DataFrame],
    released: pd.DataFrame,
    terms: ledger.ProductTerms,
    observed_until: pd.Timestamp | None,
    memo: dict[Any, pd.DataFrame] | None,
) -> list[pd.DataFrame]:
    """The cash of orders released from a hold, derived from their moved events.

    With ``memo``, each order's cash is kept by its fate and reused; only orders whose
    attempt ``tables`` hold are kept (their rows are whole in every view the replay
    builds)."""
    keys = [("released", int(order), checkout, held, at, observed_until)
            for order, checkout, held, at in released[
                ["order_id", "checkout_at", "hold_at", "released_at"]].itertuples(index=False)]
    missing = [k for k in keys if memo is None or k not in memo]
    if not missing:
        return [memo[k] for k in keys]
    orders = released.loc[released["order_id"].isin({k[1] for k in missing})]
    subset = _order_subset(tables, orders["order_id"])
    moved = _realize_rows(subset, orders, observed_until)
    derived = ledger.derive_cash_events(moved, terms)
    # cash that a row kept at its time leads to after the hold (a dispute filed before
    # it and notified during it, a recovery of an earlier write-off) waits for the
    # release too: by the pause, as everything else that followed the hold
    by_order = orders.set_index("order_id")
    hold = derived["order_id"].map(by_order["hold_at"]).to_numpy(dtype="datetime64[s]")
    pause = derived["order_id"].map(by_order["released_at"] - by_order["hold_at"]).to_numpy(
        dtype="timedelta64[s]")
    late = (derived["occurred_at"].to_numpy(dtype="datetime64[s]") > hold) & ~derived[
        "ref_event_id"].isin(_shifted_ids(subset, moved)).to_numpy()
    for column in _TIME_COLUMNS:
        values = derived[column].to_numpy(dtype="datetime64[s]").copy()
        values[late] = values[late] + pause[late]
        derived[column] = values
    if observed_until is not None:  # cash a moved event leads to only once it is known
        derived = derived.loc[derived["known_at"] <= pd.Timestamp(observed_until)]
    ids = tables["cash_events"].set_index(["kind", "ref_event_id"])["event_id"]
    derived["event_id"] = pd.array(
        ids.reindex(pd.MultiIndex.from_frame(derived[["kind", "ref_event_id"]])).to_numpy(),
        dtype="Int64")
    if memo is None:
        return [derived]
    known = set(tables["order_attempts"]["order_id"])
    by_order = dict(tuple(derived.groupby("order_id", sort=False)))
    fresh = {k: by_order.get(k[1], derived.iloc[0:0]) for k in missing}
    memo.update({k: part for k, part in fresh.items() if k[1] in known})
    return [fresh[k] if k in fresh else memo[k] for k in keys]


def _order_subset(tables: Mapping[str, pd.DataFrame], orders: pd.Series) -> dict[str, pd.DataFrame]:
    """The rows of ``tables`` that belong to ``orders`` (merchants kept whole)."""
    wanted = set(orders)
    owner = _order_of_rows(tables)
    out = {}
    for name in (*ledger.INPUT_COLUMNS, "installment_schedule", "deliveries",
                 "victim_reports", "cash_events"):
        frame = tables[name]
        out[name] = frame if name not in owner else frame.loc[np.isin(owner[name], list(wanted))]
    return out


# ------------------------------------------------------------------ policy state


def policy_state(fates: pd.DataFrame, blocks: pd.DataFrame) -> PolicyState:
    """The :class:`core.asof.PolicyState` the fates and blocks describe.

    ``approved``: orders that went through, at checkout, or at the release of a hold
    placed before shipment (an order held before shipment is neither approved nor
    voided until its hold ends); ``held``: every hold with its outcome (null while
    pending); ``voided``: voids before shipment (declines, escalations, cancelled
    holds); ``blocked``: ``blocks`` (user_id, at).
    """
    return _state(typed_fates(fates), blocks)


def realize_with_state(
    tables: Mapping[str, pd.DataFrame],
    fates: pd.DataFrame,
    blocks: pd.DataFrame,
    terms: ledger.ProductTerms,
    *,
    observed_until: pd.Timestamp | None = None,
    memo: dict[Any, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], PolicyState]:
    """:func:`realize` and :func:`policy_state` of the same fates, checked once."""
    typed = typed_fates(fates)
    out = _realize_rows(tables, typed, observed_until)
    out["cash_events"] = _policy_cash(tables, typed, terms, observed_until, memo)
    return out, _state(typed, blocks)


def _state(fates: pd.DataFrame, blocks: pd.DataFrame) -> PolicyState:
    went_through = fates["route"].isin([CheckoutRoute.APPROVE.value, CheckoutRoute.REVIEW.value])
    pre_hold = fates["hold_at"].notna() & fates["hold_before_shipment"]
    approved_at = fates["checkout_at"].where(~pre_hold, fates["released_at"])
    approved = pd.DataFrame({"order_id": fates["order_id"], "approved_at": approved_at})
    approved = approved.loc[went_through & approved["approved_at"].notna()]
    held = fates.loc[fates["hold_at"].notna()]
    held = pd.DataFrame({
        "order_id": held["order_id"].astype("int64"),
        "held_at": held["hold_at"].astype("datetime64[s]"),
        "released_at": held["released_at"].astype("datetime64[s]"),
        "outcome": held["hold_outcome"].astype(object),
        "before_shipment": held["hold_before_shipment"].astype(bool),
    })
    voided = fates.loc[fates["void_at"].notna(), ["order_id", "void_at"]].rename(
        columns={"void_at": "at"})
    blocked = blocks[["user_id", "at"]].astype({"user_id": "int64", "at": "datetime64[s]"})
    blocked = blocked.sort_values(["at", "user_id"]).drop_duplicates("user_id")
    return PolicyState(
        approved=approved.reset_index(drop=True).astype(
            {"order_id": "int64", "approved_at": "datetime64[s]"}),
        voided=voided.reset_index(drop=True).astype({"order_id": "int64", "at": "datetime64[s]"}),
        blocked=blocked.reset_index(drop=True),
        held=held.reset_index(drop=True),
    )
