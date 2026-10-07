"""Directional claims in the published documents, written from the results that test them.

A published comparison is never typed by hand. ``report/claims.yaml`` holds each
claim as a record: the policy, the reference it is compared against, the metric,
the world family and capacity level (the defaults when not given), the test and
the direction. A template places a claim with ``{{ claim:<id> }}``, and the
renderer writes its sentence from the record and the results, with the evidence:

    The hybrid policy earned more net contribution than the incumbent rules on 9
    of 10 seeds (exact sign test, p = 0.021).

The sentence says exactly what the test shows, and rendering fails when the
results no longer support the record, so a stale claim cannot be published. Free
prose in the README and the operating review carries no comparison: the lint
(``report/lint.py``) flags comparative words in a template's own text, except in
the sentences listed as stating no result.

The tests, each on a paired difference (a key with a ``vs_<reference>`` segment):

``sign``
    over seeds: ``positive`` or ``negative`` needs most seeds that way and an
    exact two-sided sign test at or below ``alpha``. The sentence speaks of
    seeds ("on 9 of 10 seeds"), never of the mean;
``interval``
    the mean difference's interval, at a confidence level of at least
    ``1 - alpha``: above zero, or below zero. The sentence speaks of the mean;
``mcnemar``
    two counts of cases (``keys``) from one paired comparison, the same but for
    one segment: ``only_policy`` in the first (cases where the metric's event
    happened for the policy and not the reference) and ``only_reference`` in
    the second.

``no_detected_difference`` (the test is not significant: the data cannot tell,
which is not evidence of equality) and ``equivalent`` (every seed's difference,
or the whole interval, lies within ``±margin``, in the metric's own unit; not
for ``mcnemar``) are written as exactly that.

Keys default to the evaluation's
``evaluate.<metric>.vs_<reference>.<family>.<capacity>.<policy>``; a record
naming another ``key`` (or ``keys``) must name its policy, reference and metric
(and any family or capacity) as segments of it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.results import Metric, metric
from core.stats import mcnemar_exact, sign_test
from report import formats
from report.formats import FormatError, Value

REPO = Path(__file__).resolve().parent.parent
CLAIMS = REPO / "report" / "claims.yaml"
DIRECTIONS = ("positive", "negative", "no_detected_difference", "equivalent")
TESTS = ("sign", "interval", "mcnemar")
MCNEMAR_SEGMENTS = ("only_policy", "only_reference")
NAME = re.compile(r"^[a-z0-9_]+$")
CLAIM_PLACEHOLDER = re.compile(r"\{\{\s*claim:([a-z0-9_-]+)\s*\}\}")


class ClaimError(ValueError):
    """A claim cannot be written: malformed, or not supported by the results."""


def is_contrast(key: str) -> bool:
    """A paired difference against a reference: a key with a ``vs_<reference>`` segment."""
    return any(part.startswith("vs_") for part in key.split("."))


@dataclass(frozen=True)
class Check:
    """A test on results: one key (sign, interval) or two (mcnemar), and a direction."""

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
class Wording:
    """How sentences name things (``wording`` in ``report/claims.yaml``)."""

    policies: Mapping[str, str]
    metrics: Mapping[str, Mapping[str, str]]  # name -> more, less, noun, format
    families: Mapping[str, str]
    capacities: Mapping[str, str]
    default_family: str
    default_capacity: str

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> Wording:
        def texts(group: str) -> dict[str, str]:
            entries = dict(data.get(group) or {})
            for name, text in entries.items():
                if not NAME.fullmatch(str(name)) or not isinstance(text, str) or not text:
                    raise ValueError(f"wording.{group}.{name}: give the words as text")
            return entries

        metrics = {}
        for name, words in (data.get("metrics") or {}).items():
            words = dict(words or {})
            missing = {"more", "less", "noun", "format"} - set(words)
            if missing or not all(isinstance(text, str) and text for text in words.values()):
                raise ValueError(f"wording.metrics.{name}: needs more, less, noun and format "
                                 f"as text (missing {sorted(missing)})")
            if words["format"] not in formats.FORMATS:
                raise ValueError(f"wording.metrics.{name}: unknown format {words['format']!r}")
            metrics[name] = words
        groups = [set(data.get(group) or {}) for group in
                  ("policies", "metrics", "families", "capacities")]
        if shared := set.union(*[a & b for i, a in enumerate(groups) for b in groups[i + 1:]]):
            raise ValueError(f"wording: {sorted(shared)} name two kinds of thing")
        defaults = dict(data.get("defaults") or {})
        wording = cls(policies=texts("policies"), metrics=metrics, families=texts("families"),
                      capacities=texts("capacities"), default_family=defaults.get("family", ""),
                      default_capacity=defaults.get("capacity", ""))
        if wording.default_family not in wording.families or \
                wording.default_capacity not in wording.capacities:
            raise ValueError("wording.defaults: name a family and a capacity level from the "
                             "wording")
        return wording


@dataclass(frozen=True)
class Claim:
    """One published comparison: what is compared, how it is tested, and its key."""

    id: str
    policy: str
    reference: str
    metric: str
    test: str
    direction: str
    family: str | None = None
    capacity: str | None = None
    key: str | None = None
    keys: tuple[str, str] | None = None
    alpha: float = 0.05
    margin: float | None = None

    def check(self, wording: Wording) -> Check:
        """The test on results this claim stands on; raises ``ClaimError`` if malformed."""
        for what, name, known in (("policy", self.policy, wording.policies),
                                  ("reference", self.reference, wording.policies),
                                  ("metric", self.metric, wording.metrics),
                                  ("family", self.family, wording.families),
                                  ("capacity", self.capacity, wording.capacities)):
            if name is not None and name not in known:
                raise ClaimError(f"its {what} {name!r} has no wording in report/claims.yaml")
        if self.policy == self.reference:
            raise ClaimError("it compares a policy with itself")
        try:
            if self.test == "mcnemar":
                if self.keys is None or self.key is not None:
                    raise ClaimError("the mcnemar test takes keys, the two discordant counts")
                _mcnemar_pair(self.keys)
                self._names_in(self.keys[0], wording)
                return Check(self.test, self.direction, keys=tuple(self.keys), alpha=self.alpha,
                             margin=self.margin)
            if self.keys is not None:
                raise ClaimError(f"the {self.test} test takes one key")
            key = self.key or self.derived_key(wording)
            self._names_in(key, wording)
            return Check(self.test, self.direction, key=key, alpha=self.alpha,
                         margin=self.margin)
        except ClaimError:
            raise
        except ValueError as error:
            raise ClaimError(str(error)) from error

    def derived_key(self, wording: Wording) -> str:
        family = self.family or wording.default_family
        capacity = self.capacity or wording.default_capacity
        return (f"evaluate.{self.metric}.vs_{self.reference}.{family}.{capacity}."
                f"{self.policy}")

    def _names_in(self, key: str, wording: Wording) -> None:
        """The key compares exactly what the record says, in the scope it says."""
        plain = [part for part in key.split(".") if not part.startswith("vs_")]
        versus = [part[len("vs_"):] for part in key.split(".") if part.startswith("vs_")]
        wanted = (
            ("policy", [p for p in plain if p in wording.policies], [self.policy]),
            ("reference", [p for p in versus if p in wording.policies], [self.reference]),
            ("metric", [p for p in plain if p in wording.metrics], [self.metric]),
            ("family", [p for p in plain if p in wording.families],
             [self.family or wording.default_family]),
            ("capacity", [p for p in plain if p in wording.capacities],
             [self.capacity or wording.default_capacity]),
        )
        for what, found, expected in wanted:
            optional = what in ("family", "capacity") and not found and \
                getattr(self, what) is None
            if found != expected and not optional:
                raise ClaimError(f"its key {key} names {what} {found or 'none'}, but the claim "
                                 f"says {expected[0]}")


def _mcnemar_pair(keys: tuple[str, str]) -> None:
    first, second = (key.split(".") for key in keys)
    differing = [(a, b) for a, b in zip(first, second, strict=False) if a != b]
    if len(first) != len(second) or differing != [MCNEMAR_SEGMENTS]:
        raise ClaimError(f"its keys {list(keys)} must be one paired comparison, the same but "
                         "for only_policy in the first and only_reference in the second")


def parse_claims(data: Any) -> tuple[list[Claim], Wording]:
    data = data or {}
    wording = Wording.from_data(data.get("wording") or {})
    claims = []
    for entry in data.get("claims") or []:
        entry = dict(entry)
        if "keys" in entry:
            entry["keys"] = tuple(entry["keys"])
        try:
            claim = Claim(**entry)
        except TypeError as error:
            raise ValueError(f"claim {entry.get('id')!r}: {error}") from error
        if not isinstance(claim.id, str) or not re.fullmatch(r"[a-z0-9_-]+", claim.id):
            raise ValueError(f"claim id {claim.id!r}: use lower-case letters, digits, - and _")
        try:
            claim.check(wording)
        except ClaimError as error:
            raise ValueError(f"claim {claim.id}: {error}") from error
        claims.append(claim)
    ids = [claim.id for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("claim ids must be unique")
    return claims, wording


def load_claims(path: Path = CLAIMS) -> tuple[list[Claim], Wording]:
    return parse_claims(yaml.safe_load(path.read_text()))


# ---------------------------------------------------------------- whether results support a check
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


def _counts(check: Check, summary: Mapping[str, Any]) -> tuple[int, int] | str:
    first, second = (metric(summary, key) for key in check.keys)
    for key, item in zip(check.keys, (first, second), strict=True):
        if item.unit != "count" or not item.evaluated:
            return f"{key} must be an evaluated count"
        # A count of cases is an int with no spread over seeds; a mean of counts is neither.
        if type(item.value) is not int or item.value < 0 or item.seeds is not None:
            return f"{key} must be a whole number of cases, not a mean, got {item.value!r}"
    return int(first.value), int(second.value)


def _mcnemar(check: Check, summary: Mapping[str, Any]) -> str | None:
    counts = _counts(check, summary)
    if isinstance(counts, str):
        return counts
    only_first, only_second = counts
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


# ---------------------------------------------------------------- the sentence
def _p(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def _amount(number: float, item: Metric, format_name: str, *, signed: bool = True) -> str:
    text = formats.apply(format_name, Value(plain=number, unit=item.unit, contrast=True),
                         ["signed"] if signed else [])
    return text if signed else text.lstrip("+")


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def sentence(claim: Claim, summary: Mapping[str, Any], wording: Wording) -> str:
    """The claim's sentence, written from its record and the results.

    Raises ``ClaimError`` when the claim is malformed or the results do not
    support it, so a stale claim stops the render.
    """
    check = claim.check(wording)
    try:
        reason = support(check, summary)
    except KeyError as error:
        reason = str(error.args[0]) if error.args else str(error)
    if reason:
        raise ClaimError(f"the results do not support it: {reason}")
    words = wording.metrics[claim.metric]
    subject, reference = wording.policies[claim.policy], wording.policies[claim.reference]
    scope = []
    if claim.family and claim.family != wording.default_family:
        scope.append(f"in {wording.families[claim.family]}")
    if claim.capacity and claim.capacity != wording.default_capacity:
        scope.append(f"at {wording.capacities[claim.capacity]}")
    lead = f"{' '.join(scope)}, " if scope else ""
    try:
        if check.test == "mcnemar":
            body = _mcnemar_sentence(check, summary, subject, reference, words)
        else:
            write = _sign_sentence if check.test == "sign" else _interval_sentence
            body = write(check, metric(summary, check.key), subject, reference, words)
    except FormatError as error:
        raise ClaimError(f"wording.metrics.{claim.metric}: {error}") from error
    return _capital(lead + body)


def _sign_sentence(check: Check, item: Metric, subject: str, reference: str,
                   words: Mapping[str, str]) -> str:
    values = list(item.seeds.per_seed.values())
    test = sign_test(values)
    seeds = len(values)
    if check.direction == "positive":
        return (f"{subject} {words['more']} {reference} on {test.positive} of {seeds} seeds "
                f"(exact sign test, {_p(test.p_value)}).")
    if check.direction == "negative":
        return (f"{subject} {words['less']} {reference} on {test.negative} of {seeds} seeds "
                f"(exact sign test, {_p(test.p_value)}).")
    if check.direction == "no_detected_difference":
        return (f"an exact sign test over {seeds} seeds detected no difference in "
                f"{words['noun']} between {subject} and {reference} ({test.positive} higher, "
                f"{test.negative} lower, {_p(test.p_value)}).")
    widest = max(abs(value) for value in values)
    form = words["format"]
    return (f"on each of the {seeds} seeds, the {words['noun']} of {subject} was within "
            f"{_amount(check.margin, item, form, signed=False)} of that of {reference} "
            f"(largest difference {_amount(widest, item, form, signed=False)}).")


def _interval_sentence(check: Check, item: Metric, subject: str, reference: str,
                       words: Mapping[str, str]) -> str:
    form = words["format"]
    level = f"{item.interval.level * 100:g}%"
    low, high = (_amount(bound, item, form) for bound in (item.interval.low, item.interval.high))
    if check.direction in ("positive", "negative"):
        more = words["more" if check.direction == "positive" else "less"]
        return (f"{subject} {more} {reference} on average: a mean difference of "
                f"{_amount(item.value, item, form)} ({level} interval {low} to {high}).")
    where = ("includes zero" if check.direction == "no_detected_difference" else
             f"lies within ±{_amount(check.margin, item, form, signed=False)}")
    return (f"the {level} interval of the mean difference in {words['noun']} between "
            f"{subject} and {reference}, {low} to {high}, {where}.")


def _mcnemar_sentence(check: Check, summary: Mapping[str, Any], subject: str, reference: str,
                      words: Mapping[str, str]) -> str:
    only_policy, only_reference = _counts(check, summary)
    evidence = (f"{only_policy} against {only_reference} discordant cases, exact McNemar test, "
                f"{_p(mcnemar_exact(only_policy, only_reference))}")
    if check.direction == "no_detected_difference":
        return (f"an exact McNemar test detected no difference in {words['noun']} between "
                f"{subject} and {reference} ({evidence}).")
    more = words["more" if check.direction == "positive" else "less"]
    return f"{subject} {more} {reference} ({evidence})."


# ---------------------------------------------------------------- the repository's claims
def placed(templates: Iterable[str]) -> set[str]:
    """The claim ids placed by ``{{ claim:<id> }}`` in the given template texts."""
    return {match.group(1) for text in templates for match in CLAIM_PLACEHOLDER.finditer(text)}


def check_claims(claims: list[Claim], summary: Mapping[str, Any], wording: Wording,
                 used: set[str]) -> list[str]:
    """Problems with the claims: not supported by the results, or placed by no template
    (``used`` holds the placed ids; a placed id with no claim fails the render)."""
    problems = []
    for claim in claims:
        if claim.id not in used:
            problems.append(f"{claim.id}: no template places it ({{{{ claim:{claim.id} }}}})")
        try:
            sentence(claim, summary, wording)
        except ClaimError as error:
            problems.append(f"{claim.id}: {error}")
    return problems
