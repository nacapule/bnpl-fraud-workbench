"""Directional claims in the published documents and the tests behind them.

``report/claims.yaml`` lists every directional sentence in the README and the
operating review (``report/lint.py`` flags comparative sentences that are not
listed). Each claim names the document, an exact piece of its rendered text,
the result key and the paired test that supports it:

``sign``
    the metric's per-seed paired differences (``seeds``); ``positive`` or
    ``negative`` needs the majority of signs that way and an exact two-sided
    sign test at or below ``alpha``; ``no_difference`` needs ``p > alpha``;
``interval``
    the metric's interval: above zero, below zero, or containing zero;
``mcnemar``
    two count metrics (``keys``: cases only the first method got right, cases
    only the second did), tested with the exact McNemar test.

A claim fails when its text is no longer in the document or the result no
longer supports it, so changed results cannot leave a stale sentence behind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.results import metric
from core.stats import mcnemar_exact, sign_test
from report.render import REPO

CLAIMS = REPO / "report" / "claims.yaml"
DIRECTIONS = ("positive", "negative", "no_difference")
TESTS = ("sign", "interval", "mcnemar")


@dataclass(frozen=True)
class Claim:
    id: str
    document: str
    text: str
    test: str
    direction: str
    key: str | None = None
    keys: tuple[str, str] | None = None
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.test not in TESTS:
            raise ValueError(f"claim {self.id}: test must be one of {TESTS}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"claim {self.id}: direction must be one of {DIRECTIONS}")
        if not self.text.strip():
            raise ValueError(f"claim {self.id}: text is empty")
        if self.test == "mcnemar":
            if self.keys is None or len(self.keys) != 2 or self.key is not None:
                raise ValueError(f"claim {self.id}: the mcnemar test takes two keys, not key")
        elif self.key is None or self.keys is not None:
            raise ValueError(f"claim {self.id}: the {self.test} test takes one key")
        if not 0 < self.alpha < 1:
            raise ValueError(f"claim {self.id}: alpha must be between 0 and 1")


def parse_claims(data: Any) -> list[Claim]:
    entries = (data or {}).get("claims") or []
    claims = []
    for entry in entries:
        entry = dict(entry)
        if "keys" in entry:
            entry["keys"] = tuple(entry["keys"])
        claims.append(Claim(**entry))
    ids = [claim.id for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("claim ids must be unique")
    return claims


def load_claims(path: Path = CLAIMS) -> list[Claim]:
    return parse_claims(yaml.safe_load(path.read_text()))


def _significant(p_value: float, claim: Claim) -> bool:
    return p_value <= claim.alpha


def support(claim: Claim, summary: Mapping[str, Any]) -> str | None:
    """Why the results do not support ``claim``, or ``None`` when they do."""
    if claim.test == "sign":
        item = metric(summary, claim.key)
        if item.seeds is None:
            return f"{claim.key} has no per-seed values"
        test = sign_test(item.seeds.per_seed.values())
        observed = f"{test.positive} positive, {test.negative} negative, p = {test.p_value:.4f}"
        if claim.direction == "no_difference":
            return None if not _significant(test.p_value, claim) else f"significant: {observed}"
        if test.direction != claim.direction or not _significant(test.p_value, claim):
            return f"not {claim.direction} at alpha {claim.alpha}: {observed}"
        return None
    if claim.test == "interval":
        item = metric(summary, claim.key)
        if item.interval is None:
            return f"{claim.key} has no interval"
        low, high = item.interval.low, item.interval.high
        holds = {
            "positive": low > 0,
            "negative": high < 0,
            "no_difference": low <= 0 <= high,
        }[claim.direction]
        return None if holds else f"interval [{low}, {high}] is not {claim.direction}"
    first, second = (metric(summary, key) for key in claim.keys)
    for key, item in zip(claim.keys, (first, second), strict=True):
        if item.unit != "count" or not item.evaluated:
            return f"{key} must be an evaluated count"
    only_first, only_second = int(first.value), int(second.value)
    p_value = mcnemar_exact(only_first, only_second)
    observed = f"{only_first} vs {only_second} discordant, p = {p_value:.4g}"
    if claim.direction == "no_difference":
        return None if not _significant(p_value, claim) else f"significant: {observed}"
    leads = (
        "positive" if only_first > only_second
        else "negative" if only_second > only_first
        else None
    )
    if leads != claim.direction or not _significant(p_value, claim):
        return f"not {claim.direction} at alpha {claim.alpha}: {observed}"
    return None


def check_claims(claims: list[Claim], summary: Mapping[str, Any],
                 read_document: Callable[[str], str | None]) -> list[str]:
    """Problems with each claim: its text missing from the document, or no support."""
    problems = []
    for claim in claims:
        text = read_document(claim.document)
        if text is None:
            problems.append(f"{claim.id}: document {claim.document} does not exist")
        elif " ".join(claim.text.split()) not in " ".join(text.split()):
            problems.append(f"{claim.id}: its text is not in {claim.document}")
        try:
            reason = support(claim, summary)
        except KeyError as error:
            reason = str(error.args[0]) if error.args else str(error)
        if reason:
            problems.append(f"{claim.id}: {reason}")
    return problems


def texts_by_document(claims: list[Claim]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for claim in claims:
        out.setdefault(claim.document, []).append(claim.text)
    return out
