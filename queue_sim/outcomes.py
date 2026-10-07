"""What a replay did, in integer counts and cents: one row per world, policy and staffing.

Outcomes are measured on the orders whose checkout fell in the replay's window, with
their cash followed to the end of observation. Money comes from the ledger
(``core.actions.policy_cash`` against the world's own cash under approve-all). Which
orders were fraud is the adjudicated label as known at the end of observation
(``core.world.labels_as_of``): positive bases are fraud, ``no_finding`` and
``credit_loss`` are legitimate, and an order with no label yet is unknown, never
legitimate. Latent truth appears only in the separate diagnostic table.

:data:`OUTCOME_COLUMNS` are the stage's row (all integers except the key fields);
ratios such as utilization (``review_minutes_used / available_minutes``) or loss in
basis points of GMV (``loss_cents / gmv_cents``) are left to the reader of the row so
numerator and denominator stay visible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from core import actions, world
from queue_sim.policies import PRIORITIES
from queue_sim.replay import ReplayResult, World

KEY_COLUMNS = ("seed", "family", "policy", "capacity_level", "layout", "history", "reviewer",
               "policy_version")
FRAUD, LEGITIMATE, UNKNOWN = "fraud", "legitimate", "unknown"
SLA_TARGETS = {"P0": 1, "P1": 4, "P2": 8, "P3": 24}

OUTCOME_COLUMNS = (
    "orders", "gmv_cents", "approved_orders", "approved_gmv_cents",
    "net_cents", "approve_all_net_cents", "net_vs_approve_all_cents",
    "fraud_orders", "fraud_net_cents", "approve_all_fraud_net_cents", "loss_cents",
    "prevented_loss_cents", "credit_loss_orders", "credit_loss_net_cents",
    "legitimate_orders", "legitimate_held", "legitimate_cancelled",
    "legitimate_declined_checkout", "legitimate_blocked_checkout",
    "legitimate_declined_review", "legitimate_declined", "friction_cost_cents",
    "unknown_orders",
    "fraud_declined_checkout", "fraud_stopped_before_shipping", "fraud_declined_after_shipping",
    "reviews", "reviews_decided", "review_minutes_offered", "review_minutes_used",
    "available_minutes",
    "coverage_minutes", "senior_minutes", "decided_after_shipping", "holds",
    "holds_before_shipping", "checks_run", "escalations", "accounts_blocked",
    "wait_p50_minutes", "wait_p90_minutes", "max_backlog",
    *(f"reviews_{p}" for p in PRIORITIES), *(f"sla_met_{p}" for p in PRIORITIES),
)


def truth(tables: Mapping[str, pd.DataFrame], observed_until: pd.Timestamp) -> pd.DataFrame:
    """Each order's adjudicated class at the end of observation: order_id, truth, basis."""
    known = world.labels_as_of(tables["labels"], observed_until)
    label = known.set_index("order_id")
    orders = tables["order_attempts"]["order_id"]
    out = pd.DataFrame({"order_id": orders.to_numpy(np.int64)})
    out["label"] = out["order_id"].map(label["label"])
    out["basis"] = out["order_id"].map(label["basis"])
    out["truth"] = np.where(out["label"] == 1, FRAUD,
                            np.where(out["label"] == 0, LEGITIMATE, UNKNOWN))
    return out[["order_id", "truth", "basis"]]


def _net_by_order(cash: pd.DataFrame) -> pd.Series:
    return cash.groupby("order_id")["amount_cents"].sum()


def outcome_row(result: ReplayResult, world_: World, *, keys: Mapping[str, Any],
                ltv_cents: int, classes: pd.DataFrame | None = None) -> dict[str, Any]:
    """The stage row for one replay: ``keys`` (KEY_COLUMNS) plus OUTCOME_COLUMNS."""
    fates = result.fates
    orders = fates["order_id"]
    classes = truth(world_.tables, world_.observed_until) if classes is None else classes
    cls = orders.map(classes.set_index("order_id")["truth"]).fillna(UNKNOWN).to_numpy()
    basis = orders.map(classes.set_index("order_id")["basis"])
    amounts = orders.map(world_.tables["order_attempts"].set_index("order_id")["amount_cents"])

    policy_cash = actions.policy_cash(world_.tables, fates, world_.terms,
                                      observed_until=world_.observed_until)
    world_cash = world_.tables["cash_events"]
    world_cash = world_cash.loc[world_cash["order_id"].isin(orders)]
    net = orders.map(_net_by_order(policy_cash)).fillna(0).to_numpy(np.int64)
    base = orders.map(_net_by_order(world_cash)).fillna(0).to_numpy(np.int64)

    route = fates["route"].to_numpy()
    voided = fates["void_at"].notna().to_numpy()
    cause = fates["void_cause"].to_numpy()
    held = fates["hold_at"].notna().to_numpy()
    reviews = result.reviews.set_index("order_id")
    final = orders.map(reviews["final"]).to_numpy()
    after_ship_decline = np.isin(final, ["decline", "escalate"]) & ~voided
    went_through = np.isin(route, ["approve", "review"]) & ~voided

    fraud, legit = cls == FRAUD, cls == LEGITIMATE
    credit = (basis == "credit_loss").to_numpy()
    declined_checkout = route == actions.CheckoutRoute.AUTO_DECLINE.value
    blocked_checkout = route == actions.BLOCKED
    declined_review = (voided & np.isin(cause, ["decline", "escalate"])) | after_ship_decline
    cancelled = voided & (cause == "hold_cancelled")
    legit_declined = legit & (declined_checkout | blocked_checkout | declined_review)

    r = result.reviews
    decided = r["decided_at"].notna()
    wait = ((r.loc[decided, "decided_at"] - r.loc[decided, "entered_at"]).dt.total_seconds()
            / 60).to_numpy()
    started = r["started_at"].notna()
    row: dict[str, Any] = dict(keys)
    row.update({
        "orders": len(fates), "gmv_cents": int(amounts.sum()),
        "approved_orders": int(went_through.sum()),
        "approved_gmv_cents": int(amounts[went_through].sum()),
        "net_cents": int(net.sum()), "approve_all_net_cents": int(base.sum()),
        "net_vs_approve_all_cents": int(net.sum() - base.sum()),
        "fraud_orders": int(fraud.sum()), "fraud_net_cents": int(net[fraud].sum()),
        "approve_all_fraud_net_cents": int(base[fraud].sum()),
        "loss_cents": int(-net[fraud].sum()),
        "prevented_loss_cents": int(net[fraud].sum() - base[fraud].sum()),
        "credit_loss_orders": int(credit.sum()), "credit_loss_net_cents": int(net[credit].sum()),
        "legitimate_orders": int(legit.sum()), "legitimate_held": int((legit & held).sum()),
        "legitimate_cancelled": int((legit & cancelled).sum()),
        "legitimate_declined_checkout": int((legit & declined_checkout).sum()),
        "legitimate_blocked_checkout": int((legit & blocked_checkout).sum()),
        "legitimate_declined_review": int((legit & declined_review).sum()),
        "legitimate_declined": int(legit_declined.sum()),
        "friction_cost_cents": int(ltv_cents * (legit_declined | (legit & cancelled)).sum()),
        "unknown_orders": int((cls == UNKNOWN).sum()),
        "fraud_declined_checkout": int((fraud & (declined_checkout | blocked_checkout)).sum()),
        "fraud_stopped_before_shipping": int((fraud & voided).sum()),
        "fraud_declined_after_shipping": int((fraud & after_ship_decline).sum()),
        "reviews": len(r), "reviews_decided": int(decided.sum()),
        "review_minutes_offered": int(round(r["service_seconds"].sum() / 60)),
        "review_minutes_used": int(round(r.loc[started, "service_seconds"].sum() / 60)),
        "available_minutes": int(round(result.available_minutes)),
        "coverage_minutes": int(round(result.coverage_minutes)),
        "senior_minutes": int(round(r["senior_minutes"].sum())),
        "decided_after_shipping": int(r["shipped_at_decision"].fillna(False).astype(bool).sum()),
        "holds": int(held.sum()), "holds_before_shipping": int(
            fates["hold_before_shipment"].to_numpy()[held].sum()),
        "checks_run": int(r["checks_started"].sum()),
        "escalations": int((final == "escalate").sum()),
        "accounts_blocked": int(result.blocks["user_id"].nunique()),
        "wait_p50_minutes": int(round(np.percentile(wait, 50))) if len(wait) else 0,
        "wait_p90_minutes": int(round(np.percentile(wait, 90))) if len(wait) else 0,
        "max_backlog": int(result.log["backlog"].max()) if len(result.log) else 0,
    })
    for p in PRIORITIES:
        in_p = r["priority"] == p
        met = decided & in_p & (r["service_hours_to_decision"] <= SLA_TARGETS[p])
        row[f"reviews_{p}"] = int(in_p.sum())
        row[f"sla_met_{p}"] = int(met.sum())
    return row


def confusion(result: ReplayResult, classes: pd.DataFrame, *, by: str = "basis") -> pd.DataFrame:
    """The reviewer's decisions by truth, decision point and evidence strength.

    ``by`` names the truth column of ``classes`` (``basis``: the adjudicated label's
    basis, with ``unknown`` for orders not yet labelled; or a latent pattern column for
    the diagnostic). Decision point: whether the order had shipped when the review
    decided; evidence strength: the adverse families present at the review (0, 1, 2+).
    Counts of each final outcome (clear, hold cleared, decline, escalate, cancelled,
    unchanged, undecided).
    """
    r = result.reviews
    truth_of = classes.set_index("order_id")[by]
    frame = pd.DataFrame({
        "truth": r["order_id"].map(truth_of).fillna("unknown").astype(str).to_numpy(),
        "decision_point": np.where(r["shipped_at_decision"].fillna(False).astype(bool),
                                   "after_shipping", "before_shipping"),
        "strength": pd.cut(r["families"].fillna(-1), [-2, -0.5, 0.5, 1.5, 99],
                           labels=["undecided", "0", "1", "2+"]).astype(str),
        "final": r["final"].fillna("undecided").astype(str).to_numpy(),
    })
    table = frame.groupby(["truth", "decision_point", "strength", "final"]).size()
    return table.rename("orders").reset_index()


def prevented_by_pattern(result: ReplayResult, world_: World,
                         classes: pd.DataFrame) -> pd.DataFrame:
    """Per label basis: N orders and net cash under approve-all and the policy."""
    fates = result.fates
    policy_cash = actions.policy_cash(world_.tables, fates, world_.terms,
                                      observed_until=world_.observed_until)
    world_cash = world_.tables["cash_events"]
    orders = fates[["order_id"]].copy()
    orders["basis"] = orders["order_id"].map(classes.set_index("order_id")["basis"]).fillna(
        "unknown")
    orders["policy_net_cents"] = orders["order_id"].map(_net_by_order(policy_cash)).fillna(0)
    orders["approve_all_net_cents"] = orders["order_id"].map(
        _net_by_order(world_cash.loc[world_cash["order_id"].isin(fates["order_id"])])).fillna(0)
    table = orders.groupby("basis").agg(
        orders=("order_id", "size"), policy_net_cents=("policy_net_cents", "sum"),
        approve_all_net_cents=("approve_all_net_cents", "sum")).astype(np.int64)
    table["prevented_cents"] = table["policy_net_cents"] - table["approve_all_net_cents"]
    return table.reset_index()
