"""Directional claims in the published documents and the tests behind them.

``report/claims.yaml`` lists every directional sentence in the README and the
operating review (``report/lint.py`` flags any comparative sentence that is
not listed). A claim names its document, the complete sentence as it appears
in the rendered document's visible text, and its checks. Each check names the
clause of the sentence it supports (an exact piece of it), a result and a
test. Every comparative word must fall inside a check's clause (the lint),
and a clause holds at most one, so each comparison has its own check.

The names in ``report/lint.yaml`` (policies, metrics, world families and
capacity levels, each a result-key segment with the phrases documents use)
tie clauses to results:

* a name inside a check's clause must be part of that check's key, so "had
  lower loss" cannot be supported by a net-contribution result;
* every name in the sentence must be inside the clause of a check whose key
  includes it, so "more than the incumbent rules and approve-all" needs a
  check against each;
* the policies and the metric a check's key tests must be named in the
  sentence, so a check cannot test a policy the sentence does not mention.

A policy mentioned without being compared ("Hybrid, which ranks the queue with
gradient boosting, earned more ...") is declared in the claim's ``context``: a
phrase of the sentence set off by commas, parentheses or dashes, holding no
comparative word and no reference policy. Names inside it are not compared and
count for none of the rules above. The tests:

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

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.results import metric
from core.stats import mcnemar_exact, sign_test
from report.lint import clause_spans, inside, normalize, phrase_spans, sentences
from report.render import REPO, is_contrast

CLAIMS = REPO / "report" / "claims.yaml"
DIRECTIONS = ("positive", "negative", "no_detected_difference", "equivalent")
TESTS = ("sign", "interval", "mcnemar")


@dataclass(frozen=True)
class Check:
    clause: str
    test: str
    direction: str
    key: str | None = None
    keys: tuple[str, str] | None = None
    alpha: float = 0.05
    margin: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.clause, str) or not normalize(self.clause):
            raise ValueError("a check needs the clause of the sentence it supports")
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
    context: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not normalize(self.sentence):
            raise ValueError(f"claim {self.id}: the sentence is empty")
        if not self.checks:
            raise ValueError(f"claim {self.id}: needs at least one check")
        if any(not isinstance(phrase, str) or not normalize(phrase) for phrase in self.context):
            raise ValueError(f"claim {self.id}: a context phrase is empty")


@dataclass(frozen=True)
class Vocabulary:
    """How documents name what results test: ``names``, ``references`` and the
    comparative words of ``report/lint.yaml``."""

    names: Mapping[str, tuple[str, ...]]  # result-key segment -> its phrases
    must_name: frozenset[str]  # policies and metrics: named whenever a key tests them
    references: frozenset[str]
    comparatives: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Vocabulary:
        groups = config.get("names") or {}
        names: dict[str, tuple[str, ...]] = {}
        owner: dict[str, str] = {}
        for entries in groups.values():
            for name, phrases in entries.items():
                if name in names:
                    raise ValueError(f"names: {name} is listed twice")
                names[name] = tuple(phrases)
                for phrase in phrases:
                    key = normalize(phrase).lower()
                    if key in owner:
                        raise ValueError(f"names: {phrase!r} names both {owner[key]} and {name}")
                    owner[key] = name
        policies = set(groups.get("policies") or {})
        references = frozenset(config.get("references") or ())
        if not references <= policies:
            raise ValueError(f"references must be policies: {sorted(references - policies)}")
        return cls(names=names,
                   must_name=frozenset(policies | set(groups.get("metrics") or {})),
                   references=references,
                   comparatives=tuple(config.get("directional_words") or ()))


NO_VOCABULARY = Vocabulary(names={}, must_name=frozenset(), references=frozenset(),
                           comparatives=())


def parse_claims(data: Any) -> list[Claim]:
    claims = []
    for entry in (data or {}).get("claims") or []:
        entry = dict(entry)
        if "context" in entry:
            entry["context"] = tuple(entry["context"] or ())
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
        # A count of cases is an int with no spread over seeds; a mean of counts is neither.
        if type(item.value) is not int or item.value < 0 or item.seeds is not None:
            return f"{key} must be a whole number of cases, not a mean, got {item.value!r}"
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


def key_names(check: Check) -> set[str]:
    """The segments of a check's keys, with ``vs_<reference>`` read as the reference."""
    names = set()
    for key in (check.key,) if check.key else check.keys:
        for part in key.split("."):
            names.add(part[len("vs_"):] if part.startswith("vs_") else part)
    return names


def mentions(names: Mapping[str, Sequence[str]], text: str) -> list[tuple[tuple[int, int], str]]:
    """Where ``text`` names each name: whole words, case-insensitive, longest phrase first
    (so "expected loss" is the policy, not the metric)."""
    lookup = {normalize(phrase).lower(): name for name, phrases in names.items()
              for phrase in phrases if normalize(phrase)}
    if not lookup:
        return []
    pattern = re.compile(r"\b(" + "|".join(re.escape(phrase) for phrase in
                                           sorted(lookup, key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
    return [(match.span(), lookup[normalize(match.group(0)).lower()])
            for match in pattern.finditer(text)]


OPENERS, CLOSERS = ",(—–;:", ",)—–;"
CONNECTIVES = {"and", "or", "nor", "but", "than", "versus", "vs", "vs."}


def _context_problems(phrase: str, sentence: str, start: int,
                      vocabulary: Vocabulary) -> list[str]:
    text = sentence[start:start + len(normalize(phrase))]
    before, after = sentence[:start].rstrip(), sentence[start + len(text):].lstrip()
    opened = text[0] in OPENERS or (before and before[-1] in OPENERS)
    closed = text[-1] in CLOSERS or (after and after[0] in CLOSERS)
    problems = []
    if not (opened and closed) or text.split()[0].lower() in CONNECTIVES:
        problems.append(f"context {phrase!r} is not an aside set off by commas, parentheses "
                        "or dashes")
    if words := [text[a:b] for a, b in phrase_spans(vocabulary.comparatives, text)]:
        problems.append(f"context {phrase!r} makes a comparison ({', '.join(words)})")
    if references := sorted({name for _, name in mentions(vocabulary.names, text)}
                            & vocabulary.references):
        problems.append(f"context {phrase!r} names {', '.join(references)}, which comparisons "
                        "are against")
    return problems


def name_problems(claim: Claim, vocabulary: Vocabulary) -> list[str]:
    """How a claim's clauses fail to match what their checks test (see the module doc)."""
    sentence = normalize(claim.sentence)
    problems: list[str] = []
    asides = []
    for phrase in claim.context:
        start = sentence.find(normalize(phrase))
        if start < 0:
            problems.append(f"context {phrase!r} is not in the sentence")
            continue
        problems += _context_problems(phrase, sentence, start, vocabulary)
        asides.append((start, start + len(normalize(phrase))))
    named = [(span, name) for span, name in mentions(vocabulary.names, sentence)
             if not inside(span, asides)]
    clauses = []
    for index, check in enumerate(claim.checks, start=1):
        clause = normalize(check.clause)
        if clause not in sentence:
            problems.append(f"check {index}: its clause is not in the sentence")
            continue
        where = clause_spans(sentence, [clause])[0]
        tested = key_names(check)
        clauses.append((where, tested))
        words = [clause[a:b] for a, b in phrase_spans(vocabulary.comparatives, clause)]
        if len(words) > 1:
            problems.append(f"check {index}: its clause makes {len(words)} comparisons "
                            f"({', '.join(words)}); give each its own check")
        for name in sorted({name for span, name in named if inside(span, [where])} - tested):
            problems.append(f"check {index}: its clause names {name}, which its key "
                            "does not test")
        for name in sorted((tested & vocabulary.must_name) - {name for _, name in named}):
            problems.append(f"check {index}: its key tests {name}, which the sentence "
                            "does not name")
    for span, name in named:
        if not any(inside(span, [where]) and name in tested for where, tested in clauses):
            problems.append(f"{sentence[span[0]:span[1]]!r} ({name}) is in no check's "
                            "clause that tests it")
    return problems


def check_claims(claims: list[Claim], summary: Mapping[str, Any],
                 read_document: Callable[[str], str | None],
                 vocabulary: Vocabulary = NO_VOCABULARY) -> list[str]:
    """Problems with each claim: its sentence missing from the document, clauses
    that do not match what their checks test, or a check the results do not support."""
    problems = []
    for claim in claims:
        text = read_document(claim.document)
        if text is None:
            problems.append(f"{claim.id}: document {claim.document} does not exist")
        elif normalize(claim.sentence) not in {sentence for _, sentence in sentences(text)}:
            problems.append(f"{claim.id}: its sentence is not a visible sentence of "
                            f"{claim.document}")
        problems += [f"{claim.id}: {problem}" for problem in name_problems(claim, vocabulary)]
        for index, check in enumerate(claim.checks):
            try:
                reason = support(check, summary)
            except KeyError as error:
                reason = str(error.args[0]) if error.args else str(error)
            if reason:
                problems.append(f"{claim.id}, check {index + 1}: {reason}")
    return problems


def clauses_by_document(claims: list[Claim]) -> dict[str, dict[str, list[str]]]:
    """For the lint: each document's claim sentences and their checks' clauses."""
    out: dict[str, dict[str, list[str]]] = {}
    for claim in claims:
        out.setdefault(claim.document, {})[normalize(claim.sentence)] = [
            check.clause for check in claim.checks
        ]
    return out
