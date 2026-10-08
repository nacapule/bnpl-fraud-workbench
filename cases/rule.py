"""The protocol's case-selection rule and its application to the incumbent's alerts.

The rule is ``experiments/protocol.yaml`` ``cases``, written before any final world
existed. What it gives as data is read from there: the world, the replay cell, the hash
prefix, the slots in their order with their predicates and tiers, and the files. What
it states in words (the population and the duplicate, mimic, reviewed and tier rules)
is implemented here, and :func:`load_rule` compares those words with :data:`TEXTS`, so a
protocol whose rule reads differently is refused rather than applied as this one.

A predicate is terms joined by ``and``; each term is one of

* ``latent_orders.<column> == <value>``: the order's latent row;
* ``<column> is null``: a ``latent_orders`` column;
* ``mimic token <token>``: the mimic rule, exact membership in the order's
  ``+``-separated mimic tokens;
* ``legitimate_rule``: the selection's legitimate rule, itself a predicate;
* ``<column> <op> <number> at checkout``: the context row the checkout routing read.

Candidates are the incumbent's alerts (the replay's routing frame) in ascending
``(SHA-256 hex digest of prefix + order id, order id)``. Labels, cash, later outcomes
and the reviewer's dispositions never enter: :func:`select` is not given them.
"""

from __future__ import annotations

import hashlib
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.actions import CheckoutRoute

TIERS = ("reviewed_alert", "checkout_alert")
ALERT_BANDS = (CheckoutRoute.REVIEW.value, CheckoutRoute.AUTO_DECLINE.value)

# The rule's words this module implements, as the protocol states them.
TEXTS: Mapping[tuple[str, ...], str] = {
    ("replay", "thresholds"): "canonical seed's validation-tuned incumbent",
    ("population",): "queue_sim.stage.routing_frame from this replay: processor-approved "
                     "checkouts with band review or auto_decline; blocked accounts' orders "
                     "excluded",
    ("selection", "duplicate_rule"): "an order already selected for an earlier slot is "
                                     "skipped",
    ("selection", "mimic_rule"): 'exact token membership in (mimic or "").split("+")',
    ("selection", "reviewed_rule"): "present in review_decisions with non-null taken_up_at "
                                    "and decided_at; no condition on disposition, final "
                                    "decision or verification results",
    ("selection", "tier_rule"): "the first non-empty tier, then its first candidate in the "
                                "selection order",
    ("facts",): "reviewed slots use the saved review_decisions row of the first analyst "
                "decision (evidence_at = decision_at, analyst_started_at = taken_up_at, "
                "action_at = decided_at, recorded action = disposition); checkout slots use "
                "the replay's saved checkout context row for the order "
                "(ReplayResult.alert_rows; evidence_at = action_at = checkout, recorded "
                "action = band); later checks and final dispositions are reported "
                "separately",
    ("later_events",): "from the incumbent replay's realized tables through the end of "
                       "observation; the world's approve-all outcomes, where they differ, "
                       "are labelled as such",
    ("missing",): "all five files are kept; a missing slot is reported with its eligible "
                  "alert and reviewed counts; never another seed, window, family or policy",
}
ORDER = re.compile(r'ascending \(SHA-256 hex digest of UTF-8 "([^"]*)" \+ decimal order_id, '
                   r'order_id\)')
OPERATORS: Mapping[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq, ">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}


class RuleError(ValueError):
    """The protocol's case rule cannot be read, or the run does not fit it."""


@dataclass(frozen=True)
class Term:
    """One term of a predicate: ``source`` is ``latent``, ``checkout`` or ``mimic``."""

    source: str
    column: str
    op: str
    value: Any


@dataclass(frozen=True)
class Slot:
    name: str
    predicate: str
    terms: tuple[Term, ...]
    tiers: tuple[str, ...]


@dataclass(frozen=True)
class CaseFile:
    name: str
    slots: tuple[str, ...]
    length: str
    primary: str  # the file's primary alert (its only slot, or primary_alert)


@dataclass(frozen=True)
class Rule:
    seed: int
    family: str
    replay: Mapping[str, str]
    prefix: str
    slot_order: tuple[str, ...]
    slots: Mapping[str, Slot]
    files: Mapping[str, CaseFile]

    @property
    def world(self) -> str:
        return f"{self.seed}-{self.family}"

    def file_of(self, slot: str) -> str:
        return next(name for name, entry in self.files.items() if slot in entry.slots)


def _words(text: Any) -> str:
    return " ".join(str(text).split())


def _at(section: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = section
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise RuleError(f"cases.{'.'.join(path)} is missing")
        node = node[key]
    return node


def parse_predicate(text: str, legitimate_rule: str | None = None) -> tuple[Term, ...]:
    """The terms of a slot predicate (module docstring); refuses anything else."""
    terms: list[Term] = []
    for part in (p.strip() for p in _words(text).split(" and ")):
        if part == "legitimate_rule":
            if legitimate_rule is None:
                raise RuleError("legitimate_rule may not refer to itself")
            terms += parse_predicate(legitimate_rule)
        elif m := re.fullmatch(r"latent_orders\.(\w+) == ([\w-]+)", part):
            terms.append(Term("latent", m[1], "==", m[2]))
        elif m := re.fullmatch(r"(?:latent_orders\.)?(\w+) is null", part):
            terms.append(Term("latent", m[1], "is null", None))
        elif m := re.fullmatch(r"mimic token (\w+)", part):
            terms.append(Term("mimic", "mimic", "has token", m[1]))
        elif m := re.fullmatch(r"(\w+) (==|>=|<=|>|<) (-?\d+(?:\.\d+)?) at checkout", part):
            number = float(m[3]) if "." in m[3] else int(m[3])
            terms.append(Term("checkout", m[1], m[2], number))
        else:
            raise RuleError(f"cannot read the predicate term {part!r}")
    return tuple(terms)


def load_rule(protocol_raw: Mapping[str, Any]) -> Rule:
    """The ``cases`` section of a protocol (``core.protocol.Protocol.raw``)."""
    cases = protocol_raw.get("cases")
    if not isinstance(cases, Mapping):
        raise RuleError("the protocol has no cases section")
    for path, expected in TEXTS.items():
        found = _words(_at(cases, path))
        if found != _words(expected):
            raise RuleError(f"cases.{'.'.join(path)} reads {found!r}; this code implements "
                            f"{_words(expected)!r}")
    order = ORDER.fullmatch(_words(_at(cases, ("selection", "order"))))
    if order is None:
        raise RuleError("cases.selection.order is not the SHA-256 order this code implements")
    legitimate = str(_at(cases, ("selection", "legitimate_rule")))
    slots = {}
    for name, entry in _at(cases, ("slots",)).items():
        tiers = tuple(entry["tiers"])
        if not tiers or any(tier not in TIERS for tier in tiers):
            raise RuleError(f"cases.slots.{name}: tiers must be from {list(TIERS)}")
        slots[name] = Slot(name, _words(entry["predicate"]),
                           parse_predicate(entry["predicate"], legitimate), tiers)
    slot_order = tuple(_at(cases, ("selection", "slot_order")))
    if sorted(slot_order) != sorted(slots) or len(set(slot_order)) != len(slot_order):
        raise RuleError("cases.selection.slot_order must list every slot once")
    files = {}
    for name, entry in _at(cases, ("files",)).items():
        named = tuple(entry["slots"])
        primary = entry.get("primary_alert", named[0])
        if any(slot not in slots for slot in named) or primary not in named:
            raise RuleError(f"cases.files.{name} names slots that do not exist")
        files[name] = CaseFile(name, named, str(entry["length"]), str(primary))
    used = sorted(slot for entry in files.values() for slot in entry.slots)
    if used != sorted(slots):
        raise RuleError("cases.files must use every slot exactly once")
    world = _at(cases, ("world",))
    return Rule(seed=int(world["seed"]), family=str(world["family"]),
                replay=dict(_at(cases, ("replay",))), prefix=order[1], slot_order=slot_order,
                slots=slots, files=files)


def selection_hash(order_id: int, prefix: str) -> str:
    """The SHA-256 hex digest of UTF-8 ``prefix`` + the decimal order id."""
    return hashlib.sha256(f"{prefix}{int(order_id)}".encode()).hexdigest()


# --------------------------------------------------------------------- the population


def population(fates: pd.DataFrame, checkout_rows: pd.DataFrame,
               policy_version: str) -> pd.DataFrame:
    """The replay's routing frame from the frames the pipeline kept: one row per order the
    incumbent routed to review or auto-declined at checkout (``order_id``, ``alert_id``,
    ``user_id``, ``checkout_at``, ``band``), in the order of the kept checkout rows.

    Orders of blocked accounts carry the route ``blocked`` and are not alerts. The kept
    checkout rows are the rows of exactly these orders, in the same order; anything else
    means the frames do not come from one run and is refused.
    """
    alerts = fates.loc[fates["route"].isin(ALERT_BANDS)]
    if alerts["order_id"].tolist() != checkout_rows["order_id"].tolist():
        raise RuleError("the kept checkout rows are not the incumbent's alerts")
    return pd.DataFrame({
        "order_id": alerts["order_id"].to_numpy(np.int64),
        "alert_id": [f"{int(order)}:{policy_version}" for order in alerts["order_id"]],
        "user_id": alerts["user_id"].to_numpy(np.int64),
        "checkout_at": alerts["checkout_at"].to_numpy(),
        "band": alerts["route"].to_numpy(),
    })


def reviewed_orders(reviews: pd.DataFrame) -> set[int]:
    """Orders an analyst took up and decided (the reviewed rule)."""
    done = reviews["taken_up_at"].notna() & reviews["decided_at"].notna()
    return {int(order) for order in reviews.loc[done, "order_id"]}


# --------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Pick:
    """One slot's outcome: the chosen alert, or none, and the candidate counts.

    ``eligible``: alerts meeting the predicate; ``reviewed``: of those, reviewed;
    ``available``: eligible alerts not chosen for an earlier slot; ``tiers``: available
    candidates per tier of the slot.
    """

    slot: str
    order_id: int | None
    alert_id: str | None
    selection_hash: str | None
    tier: str | None
    eligible: int
    reviewed: int
    available: int
    tiers: Mapping[str, int]

    @property
    def selected(self) -> bool:
        return self.order_id is not None


def _holds(term: Term, frame: pd.DataFrame) -> np.ndarray:
    if term.source == "mimic":
        tokens = frame["mimic"].astype(object).where(frame["mimic"].notna(), "")
        return np.array([term.value in str(value).split("+") for value in tokens], dtype=bool)
    column = ("latent." if term.source == "latent" else "checkout.") + term.column
    if column not in frame.columns:
        raise RuleError(f"no {term.source} column {term.column!r}")
    values = frame[column]
    if term.op == "is null":
        return values.isna().to_numpy()
    return (OPERATORS[term.op](values, term.value) & values.notna()).to_numpy(bool)


def candidates(rule: Rule, alerts: pd.DataFrame, reviews: pd.DataFrame,
               checkout_rows: pd.DataFrame, latent_orders: pd.DataFrame) -> pd.DataFrame:
    """The alerts in selection order with what the predicates read: ``latent.<column>``
    from ``latent_orders``, ``checkout.<column>`` from the kept checkout rows, ``mimic``,
    ``reviewed`` and ``selection_hash``."""
    latent = latent_orders.set_index("order_id")
    rows = checkout_rows.set_index("order_id")
    frame = alerts.copy()
    for column in latent.columns:
        frame[f"latent.{column}"] = frame["order_id"].map(latent[column]).to_numpy()
    for column in rows.columns:
        frame[f"checkout.{column}"] = frame["order_id"].map(rows[column]).to_numpy()
    frame["mimic"] = frame["latent.mimic"] if "latent.mimic" in frame.columns else None
    frame["reviewed"] = frame["order_id"].isin(reviewed_orders(reviews))
    frame["selection_hash"] = [selection_hash(order, rule.prefix) for order in frame["order_id"]]
    return frame.sort_values(["selection_hash", "order_id"], kind="stable").reset_index(drop=True)


def select(rule: Rule, alerts: pd.DataFrame, reviews: pd.DataFrame,
           checkout_rows: pd.DataFrame, latent_orders: pd.DataFrame) -> dict[str, Pick]:
    """Each slot's pick, in the rule's slot order (:class:`Pick`)."""
    frame = candidates(rule, alerts, reviews, checkout_rows, latent_orders)
    chosen: set[int] = set()
    picks: dict[str, Pick] = {}
    for name in rule.slot_order:
        slot = rule.slots[name]
        mask = np.ones(len(frame), dtype=bool)
        for term in slot.terms:
            mask &= _holds(term, frame)
        eligible = frame.loc[mask]
        available = eligible.loc[~eligible["order_id"].isin(chosen)]
        tiers = {"reviewed_alert": available.loc[available["reviewed"]],
                 "checkout_alert": available}
        counts = {tier: len(tiers[tier]) for tier in slot.tiers}
        tier = next((t for t in slot.tiers if len(tiers[t])), None)
        if tier is None:
            picks[name] = Pick(name, None, None, None, None, len(eligible),
                               int(eligible["reviewed"].sum()), len(available), counts)
            continue
        first = tiers[tier].iloc[0]
        chosen.add(int(first["order_id"]))
        picks[name] = Pick(name, int(first["order_id"]), str(first["alert_id"]),
                           str(first["selection_hash"]), tier, len(eligible),
                           int(eligible["reviewed"].sum()), len(available), counts)
    return picks
