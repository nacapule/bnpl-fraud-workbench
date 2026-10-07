"""Directional claims in the published documents and the tests behind them.

``report/claims.yaml`` lists every directional sentence in the README and the
operating review (``report/lint.py`` flags any comparative sentence that is
not listed). A claim names its document, the complete sentence as it appears
in the rendered document's visible text, and one check per comparison the
sentence makes. Each check names a result and a test:

``sign``
    a paired difference over seeds (a metric whose key has a ``vs_<reference>``
    segment, with per-seed values): ``positive`` or ``negative`` needs most
    seeds that way and an exact two-sided sign test at or below ``alpha``;
``interval``
    a paired difference's interval, at a confidence level of at least
    ``1 - alpha``: above zero, or below zero;
``mcnemar``
    two count metrics (``keys``: cases only the first method got right, cases
    only the second did), tested with the exact McNemar test.

Two further directions describe the absence of a difference, and say
different things: ``no_detected_difference`` (the test is not significant:
the data cannot tell, which is not evidence of equality) and ``equivalent``
(every seed's difference, or the whole interval, lies within ``±margin``, a
tolerance in the metric's own unit fixed in the claim; not for ``mcnemar``).

A claim fails when its sentence is no longer in the document or a check is no
longer supported, so changed results cannot leave a stale sentence behind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.results import metric
from core.stats import mcnemar_exact, sign_test
from report.lint import normalize, sentences
from report.render import REPO, is_contrast

CLAIMS = REPO / "report" / "claims.yaml"
DIRECTIONS = ("positive", "negative", "no_detected_difference", "equivalent")
TESTS = ("sign", "interval", "mcnemar")


@dataclass(frozen=True)
class Check:
    test: str
    direction: str
    key: str | None = None
    keys: tuple[str, str] | None = None
    alpha: float = 0.05
    margin: float | None = None

    def __post_init__(self) -> None:
        if self.test not in TESTS:
            raise ValueError(f"test must be one of {TESTS}, got {self.test!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.test == "mcnemar":
            if self.keys is None or len(self.keys) != 2 or self.key is not None:
                raise ValueError("the mcnemar test takes two keys, not key")
            if self.direction == "equivalent":
                raise ValueError("the mcnemar test cannot show equivalence")
        elif self.key is None or self.keys is not None:
            raise ValueError(f"the {self.test} test takes one key")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if (self.direction == "equivalent") != (self.margin is not None):
            raise ValueError("a margin is required for equivalent, and only for it")
        if self.margin is not None and not self.margin > 0:
            raise ValueError("the margin must be positive")


@dataclass(frozen=True)
class Claim:
    id: str
    document: str
    sentence: str
    checks: tuple[Check, ...]

    def __post_init__(self) -> None:
        if not normalize(self.sentence):
            raise ValueError(f"claim {self.id}: the sentence is empty")
        if not self.checks:
            raise ValueError(f"claim {self.id}: needs at least one check")


def parse_claims(data: Any) -> list[Claim]:
    claims = []
    for entry in (data or {}).get("claims") or []:
        entry = dict(entry)
        checks = []
        for item in entry.pop("checks", None) or []:
            item = dict(item)
            if "keys" in item:
                item["keys"] = tuple(item["keys"])
            try:
                checks.append(Check(**item))
            except (TypeError, ValueError) as error:
                raise ValueError(f"claim {entry.get('id')!r}: {error}") from error
        claims.append(Claim(checks=tuple(checks), **entry))
    ids = [claim.id for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("claim ids must be unique")
    return claims


def load_claims(path: Path = CLAIMS) -> list[Claim]:
    return parse_claims(yaml.safe_load(path.read_text()))


def support(check: Check, summary: Mapping[str, Any]) -> str | None:
    """Why the results do not support ``check``, or ``None`` when they do."""
    if check.test == "mcnemar":
        return _mcnemar(check, summary)
    if not is_contrast(check.key):
        return (f"{check.key} is not a paired difference (its key needs a vs_<reference> "
                "segment); a comparison needs a contrast, not a level")
    item = metric(summary, check.key)
    if not item.evaluated:
        return f"{check.key} was not evaluated"
    if check.test == "sign":
        if item.seeds is None:
            return f"{check.key} has no per-seed values"
        values = list(item.seeds.per_seed.values())
        test = sign_test(values)
        observed = f"{test.positive} positive, {test.negative} negative, p = {test.p_value:.4f}"
        if check.direction == "equivalent":
            widest = max(abs(value) for value in values)
            return None if widest <= check.margin else (
                f"a seed differs by {widest}, beyond the margin {check.margin}")
        if check.direction == "no_detected_difference":
            return None if test.p_value > check.alpha else f"significant: {observed}"
        if test.direction != check.direction or test.p_value > check.alpha:
            return f"not {check.direction} at alpha {check.alpha}: {observed}"
        return None
    if item.interval is None:
        return f"{check.key} has no interval"
    if item.interval.level < 1 - check.alpha - 1e-9:
        return f"the interval's level {item.interval.level} is below 1 - alpha"
    low, high = item.interval.low, item.interval.high
    holds = {
        "positive": low > 0,
        "negative": high < 0,
        "no_detected_difference": low <= 0 <= high,
        "equivalent": check.margin is not None and -check.margin <= low and high <= check.margin,
    }[check.direction]
    return None if holds else f"interval [{low}, {high}] is not {check.direction}"


def _mcnemar(check: Check, summary: Mapping[str, Any]) -> str | None:
    first, second = (metric(summary, key) for key in check.keys)
    for key, item in zip(check.keys, (first, second), strict=True):
        if item.unit != "count" or not item.evaluated:
            return f"{key} must be an evaluated count"
    only_first, only_second = int(first.value), int(second.value)
    p_value = mcnemar_exact(only_first, only_second)
    observed = f"{only_first} vs {only_second} discordant, p = {p_value:.4g}"
    if check.direction == "no_detected_difference":
        return None if p_value > check.alpha else f"significant: {observed}"
    leads = (
        "positive" if only_first > only_second
        else "negative" if only_second > only_first
        else None
    )
    if leads != check.direction or p_value > check.alpha:
        return f"not {check.direction} at alpha {check.alpha}: {observed}"
    return None


def check_claims(claims: list[Claim], summary: Mapping[str, Any],
                 read_document: Callable[[str], str | None]) -> list[str]:
    """Problems with each claim: its sentence missing from the document, or a check failing."""
    problems = []
    for claim in claims:
        text = read_document(claim.document)
        if text is None:
            problems.append(f"{claim.id}: document {claim.document} does not exist")
        elif normalize(claim.sentence) not in {sentence for _, sentence in sentences(text)}:
            problems.append(f"{claim.id}: its sentence is not a visible sentence of "
                            f"{claim.document}")
        for index, check in enumerate(claim.checks):
            try:
                reason = support(check, summary)
            except KeyError as error:
                reason = str(error.args[0]) if error.args else str(error)
            if reason:
                problems.append(f"{claim.id}, check {index + 1}: {reason}")
    return problems


def sentences_by_document(claims: list[Claim]) -> dict[str, dict[str, int]]:
    """For the lint: each document's claim sentences and how many checks each has."""
    out: dict[str, dict[str, int]] = {}
    for claim in claims:
        out.setdefault(claim.document, {})[normalize(claim.sentence)] = len(claim.checks)
    return out
