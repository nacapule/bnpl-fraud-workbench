"""The case packet: what the memo drafter and the referee see for one review decision.

A packet describes one order at one decision time: the review, or the completion of a
verification check (fraud policy §2.4, §6.6). Its facts come from the shared as-of
context (:mod:`core.asof`) at that time, so the packet obeys the same knowledge rules
as the rules, the models and the simulated reviewer: order-anchored facts as at the
order, linkage and earlier outcomes as known at the decision. Nothing here computes
history; the order's own attributes come from its attempt row and the entities it
names.

Entity ids are replaced by placeholders (``O1``, ``U1``, ``M1``, ``D1``, ``C1``,
``A1`` by default; :func:`placeholders` draws fresh ones for invariance probes), so
identifiers carry no information about how the world was generated. The mapping
stays with the caller and never enters the packet. Labels and latent tables are
never read (:data:`FORBIDDEN_KEYS` guards the output).
"""

from __future__ import annotations

import math
import random
import string
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from core.actions import Check, CheckOutcome
from core.asof import COLUMNS
from core.evidence import CheckResult

PACKET_VERSION = "fp2-1"
DECISION_POINTS = ("review", "check_completed")

# Every context column the packet carries: those the specification gives to packets or
# the referee, in specification order.
CONTEXT_FIELDS: tuple[str, ...] = tuple(
    column.name for column in COLUMNS if {"packets", "referee"} & set(column.used_by))
ENTITIES = ("order", "account", "merchant", "device", "card", "ship_address")
DEFAULT_PREFIX = {"order": "O", "account": "U", "merchant": "M", "device": "D", "card": "C",
                  "ship_address": "A"}
FORBIDDEN_KEYS = {"label", "labels", "label_known_at", "basis", "pattern_id", "episode_id",
                  "intent", "mimic", "actor", "profile", "is_fraud", "story_id"}
INTEGER_DTYPES = ("int8", "int64")


def placeholders(seed: int | None = None) -> dict[str, str]:
    """Placeholder names for the packet's entities: ``O1``… by default, or fresh random
    names (a letter, a hyphen, three characters) for an invariance probe."""
    if seed is None:
        return {entity: f"{DEFAULT_PREFIX[entity]}1" for entity in ENTITIES}
    rng = random.Random(seed)
    alphabet = string.ascii_uppercase + string.digits
    names: dict[str, str] = {}
    for entity in ENTITIES:
        name = ""
        while not name or name in names.values():
            name = f"{DEFAULT_PREFIX[entity]}-" + "".join(rng.choice(alphabet) for _ in range(3))
        names[entity] = name
    return names


def _timestamp(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("a packet timestamp is missing")
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _value(name: str, value: Any) -> int | float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return None
    dtype = next(column.dtype for column in COLUMNS if column.name == name)
    if dtype in INTEGER_DTYPES:
        if float(value) != int(value):
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        return int(value)
    return float(value)  # full precision: rounding could move a value across a threshold


def build_packet(
    context_row: Mapping[str, Any],
    order: Mapping[str, Any],
    *,
    merchant_category: str,
    card_bin_country: str,
    home_country: str,
    checks: Sequence[CheckResult] = (),
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One packet from the order's context row at its decision time.

    ``context_row`` has core.asof KEY_COLUMNS and at least :data:`CONTEXT_FIELDS`;
    ``decision_at`` is the decision time. ``order`` is the order's ``order_attempts``
    row (its own attributes: when it was placed, IP country, AVS and CVV results,
    promotion). ``checks`` are the verification checks completed by the decision
    time, at most one of each; with any, the decision point is the completion of the
    last. ``names`` are the placeholders (:func:`placeholders`).
    """
    names = dict(names or placeholders())
    if set(names) != set(ENTITIES) or len(set(names.values())) != len(ENTITIES):
        raise ValueError("names must give a distinct placeholder for each entity")
    if int(context_row["order_id"]) != int(order["order_id"]):
        raise ValueError("the context row and the order attempt describe different orders")
    decision_at = pd.Timestamp(context_row["decision_at"])
    placed_at = pd.Timestamp(order["known_at"])
    if decision_at < placed_at:
        raise ValueError("a decision cannot precede its order")
    checks = sorted(checks, key=lambda result: result.completed_at)
    if len({result.check for result in checks}) != len(checks):
        raise ValueError("each check runs at most once (fraud policy §5.1)")
    for result in checks:
        if not placed_at <= pd.Timestamp(result.completed_at) <= decision_at:
            raise ValueError("a check must complete between the order and the decision")
    missing = [field for field in CONTEXT_FIELDS if field not in context_row]
    if missing:
        raise KeyError(f"context row lacks {missing}")

    packet = {
        "packet_version": PACKET_VERSION,
        "decision": {
            "point": "check_completed" if checks else "review",
            "decision_at": _timestamp(decision_at),
            "checks": [{"check": str(result.check), "outcome": str(result.outcome),
                        "completed_at": _timestamp(result.completed_at)} for result in checks],
        },
        "order": {
            **{entity: names[entity] for entity in ENTITIES},
            "placed_at": _timestamp(placed_at),
            "merchant_category": str(merchant_category),
            "promotion_used": bool(pd.notna(order.get("promo_id"))),
            "ip_country": str(order["ip_country"]),
            "card_bin_country": str(card_bin_country),
            "account_home_country": str(home_country),
            "avs_result": str(order["avs_result"]),
            "cvv_result": str(order["cvv_result"]),
        },
        "context": {field: _value(field, context_row[field]) for field in CONTEXT_FIELDS},
    }
    assert_no_forbidden(packet)
    return packet


def build_packets(
    tables: Mapping[str, pd.DataFrame],
    context: pd.DataFrame,
    *,
    checks: Mapping[tuple[int, datetime], Sequence[CheckResult]] | None = None,
    name_seeds: Mapping[tuple[int, datetime], int] | None = None,
) -> dict[tuple[int, pd.Timestamp], dict[str, Any]]:
    """Packets for every row of ``context`` (core.asof rows at their decision times),
    keyed by (order_id, decision_at). ``tables`` supply the order attempts, accounts,
    cards and merchants; ``checks`` and ``name_seeds`` are optional per decision."""
    attempts = tables["order_attempts"].set_index("order_id")
    accounts = tables["accounts"].set_index("user_id")
    cards = tables["cards"].set_index("card_id")
    merchants = tables["merchants"].set_index("merchant_id")
    packets: dict[tuple[int, pd.Timestamp], dict[str, Any]] = {}
    for row in context.to_dict("records"):
        key = (int(row["order_id"]), pd.Timestamp(row["decision_at"]))
        order = {"order_id": key[0], **attempts.loc[key[0]].to_dict()}
        seed = (name_seeds or {}).get(key)
        packets[key] = build_packet(
            row, order,
            merchant_category=merchants.loc[order["merchant_id"], "category"],
            card_bin_country=cards.loc[order["card_id"], "bin_country"],
            home_country=accounts.loc[order["user_id"], "home_country"],
            checks=(checks or {}).get(key, ()),
            names=placeholders(seed),
        )
    return packets


def assert_no_forbidden(obj: Any, path: str = "") -> None:
    """Raise if a label or simulation-truth key appears anywhere in ``obj``."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise AssertionError(f"forbidden key {key!r} at {path or '.'}")
            assert_no_forbidden(value, f"{path}.{key}")
    elif isinstance(obj, list | tuple):
        for index, value in enumerate(obj):
            assert_no_forbidden(value, f"{path}[{index}]")


def context_row(packet: Mapping[str, Any]) -> dict[str, Any]:
    """The context values a packet carries (what the referee classifies), with missing
    values as NaN so that a missing fact never makes a condition hold."""
    return {field: (math.nan if value is None else value)
            for field, value in packet["context"].items()}


def checks_of(packet: Mapping[str, Any]) -> tuple[CheckResult, ...]:
    """The completed checks a packet records."""
    return tuple(CheckResult(Check(item["check"]), CheckOutcome(item["outcome"]),
                             datetime.fromisoformat(item["completed_at"]))
                 for item in packet["decision"]["checks"])
