"""Directional claims in the published documents and the tests behind them.

``report/claims.yaml`` lists every directional sentence in the README and the
operating review (``report/lint.py`` flags any comparative sentence that is
not listed). A claim names its document, the complete sentence as it appears
in the rendered document's visible text, and the checks behind it.

A claim's sentence makes exactly one comparison: it holds one comparative word
(``comparatives`` in ``report/lint.yaml``). A sentence comparing two things is
written as two sentences, each a claim. Everything the sentence names takes
part in that comparison, using the names in ``report/lint.yaml`` (policies,
metrics, world families and capacity levels, each a result-key segment with the
phrases documents use):

* the policies named before the comparative word are compared against the
  policies named after it, on every metric named, in every family and at every
  capacity level named (the defaults when none is named). There must be a check
  for each such comparison, and no check for anything else. "Hybrid earned more
  than the incumbent rules and approve-all" needs two checks; "in the
  acquisition surge" binds every check to that family; a policy after the
  comparative word that is not a reference cannot be supported, because results
  only compare against the references;
* the comparative word sets the direction: ``up`` words (more, higher) need a
  positive difference and ``down`` words (fewer, lower) a negative one;
  ``good`` and ``bad`` words (better, beats, worse) follow the metric's
  ``polarity``;
* a negated sentence ("did not earn more", "no higher than") is tested as
  ``no_detected_difference`` or ``equivalent``, and only a negated one is.

A check's key names what it tests: for ``sign`` and ``interval``, one policy, one
``vs_<reference>`` segment, one metric, and at most one family and one capacity
level; for ``mcnemar``, the first key holds the cases only the compared policy got
right and the second those only the reference got right, each naming its own
policy, the same metric and the same family and capacity level. The tests:

``sign``
    a paired difference over seeds (a metric whose key has a ``vs_<reference>``
    segment, with per-seed values): ``positive`` or ``negative`` needs most
    seeds that way and an exact two-sided sign test at or below ``alpha``;
``interval``
    a paired difference's interval, at a confidence level of at least
    ``1 - alpha``: above zero, or below zero;
``mcnemar``
    two counts of cases (``keys``), tested with the exact McNemar test.

``no_detected_difference`` (the test is not significant: the data cannot tell,
which is not evidence of equality) and ``equivalent`` (every seed's difference,
or the whole interval, lies within ``±margin``, a tolerance in the metric's own
unit fixed in the claim; not for ``mcnemar``) say different things.

A claim fails when its sentence is no longer in the document, its document has
Markdown the sentence reader does not follow, its checks do not match its
sentence, or a check is no longer supported, so changed results cannot leave a
stale sentence behind.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from core.results import metric
from core.stats import mcnemar_exact, sign_test
from report.lint import SIGNS, markdown_problems, normalize, phrase_spans, sentences
from report.render import REPO, is_contrast

CLAIMS = REPO / "report" / "claims.yaml"
DIRECTIONS = ("positive", "negative", "no_detected_difference", "equivalent")
TESTS = ("sign", "interval", "mcnemar")
GROUPS = ("policies", "metrics", "families", "capacities")


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


@dataclass(frozen=True)
class Vocabulary:
    """How documents name what results test (``report/lint.yaml``)."""

    groups: Mapping[str, Mapping[str, tuple[str, ...]]]  # group -> name -> phrases
    references: frozenset[str]
    comparatives: Mapping[str, str]  # word -> up, down, good or bad
    negations: tuple[str, ...]
    polarity: Mapping[str, int]  # metric -> +1 when higher is better, -1 when lower is
    defaults: Mapping[str, str]  # families / capacities -> the name meant when none is named

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Vocabulary:
        groups: dict[str, dict[str, tuple[str, ...]]] = {group: {} for group in GROUPS}
        owner: dict[str, str] = {}
        for group, entries in (config.get("names") or {}).items():
            if group not in GROUPS:
                raise ValueError(f"names: unknown group {group}; use {GROUPS}")
            for name, phrases in (entries or {}).items():
                if any(name in known for known in groups.values()):
                    raise ValueError(f"names: {name} is listed twice")
                groups[group][name] = tuple(phrases)
                for phrase in phrases:
                    key = normalize(phrase).lower()
                    if key in owner:
                        raise ValueError(f"names: {phrase!r} names both {owner[key]} and {name}")
                    owner[key] = name
        references = frozenset(config.get("references") or ())
        if missing := sorted(references - set(groups["policies"])):
            raise ValueError(f"references must be policies: {missing}")
        comparatives = {}
        for sign, words in (config.get("comparatives") or {}).items():
            if sign not in SIGNS:
                raise ValueError(f"comparatives: unknown group {sign}; use {SIGNS}")
            comparatives.update({normalize(word).lower(): sign for word in words})
        polarity = {}
        for name, better in (config.get("polarity") or {}).items():
            if name not in groups["metrics"] or better not in ("higher", "lower"):
                raise ValueError(f"polarity: {name}: {better!r} (a metric: higher or lower)")
            polarity[name] = 1 if better == "higher" else -1
        defaults = dict(config.get("defaults") or {})
        for group, name in defaults.items():
            if group not in ("families", "capacities") or name not in groups[group]:
                raise ValueError(f"defaults: {group}: {name} is not one of the {group}")
        return cls(groups=groups, references=references, comparatives=comparatives,
                   negations=tuple(config.get("negations") or ()), polarity=polarity,
                   defaults=defaults)

    @property
    def names(self) -> dict[str, tuple[str, ...]]:
        return {name: phrases for entries in self.groups.values()
                for name, phrases in entries.items()}

    def group_of(self, name: str) -> str | None:
        return next((group for group, entries in self.groups.items() if name in entries), None)


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


# ---------------------------------------------------------------- sentence and checks
def mentions(names: Mapping[str, Iterable[str]], text: str,
             between: Iterable[str] = ()) -> list[tuple[tuple[int, int], str]]:
    """Where ``text`` names each name: whole words, case-insensitive, the longest
    phrase first (so "expected loss" is the policy, not the metric). A word of
    ``between`` may sit between a phrase's words: "held fewer legitimate orders"
    names "held legitimate orders"."""
    between = sorted({normalize(word) for word in between if normalize(word)}, key=len,
                     reverse=True)
    gap = r"\s+" if not between else (
        r"\s+(?:(?:" + "|".join(re.escape(word) for word in between) + r")\s+)?")
    phrases = sorted(
        {(normalize(phrase), name) for name, items in names.items() for phrase in items
         if normalize(phrase)},
        key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
    )
    if not phrases:
        return []
    alternatives = "|".join(
        f"(?P<g{index}>" + gap.join(re.escape(word) for word in phrase.split()) + ")"
        for index, (phrase, _) in enumerate(phrases)
    )
    pattern = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)
    return [(match.span(), phrases[int(match.lastgroup[1:])][1])
            for match in pattern.finditer(text)]


class Tested(NamedTuple):
    """What one check compares: a policy against a reference on a metric."""

    policy: str
    reference: str
    metric: str
    family: str | None
    capacity: str | None


def _segments(key: str) -> tuple[list[str], list[str]]:
    plain, versus = [], []
    for part in key.split("."):
        (versus if part.startswith("vs_") else plain).append(part.removeprefix("vs_"))
    return plain, versus


def _one(names: list[str], what: str, key: str, optional: bool = False) -> str | None:
    if len(names) > 1 or (not names and not optional):
        raise ValueError(f"{key} names {len(names)} {what} from report/lint.yaml names, "
                         f"not {'at most ' if optional else ''}one")
    return names[0] if names else None


def tested(check: Check, vocabulary: Vocabulary) -> Tested:
    """The comparison a check's key tests, read through the vocabulary's names."""
    def within(group: str, names: list[str]) -> list[str]:
        return [name for name in names if name in vocabulary.groups[group]]

    keys = (check.key,) if check.key else check.keys
    shared = []
    for key in keys:
        plain, _ = _segments(key)
        shared.append(tuple(_one(within(group, plain), group, key, optional=group != "metrics")
                            for group in ("metrics", "families", "capacities")))
    if len(set(shared)) > 1:
        raise ValueError(f"its keys {list(keys)} differ in metric, family or capacity")
    metric_name, family, capacity = shared[0]
    if check.test == "mcnemar":
        policy = _one(within("policies", _segments(keys[0])[0]), "policies", keys[0])
        reference = _one(within("policies", _segments(keys[1])[0]), "policies", keys[1])
    else:
        plain, versus = _segments(check.key)
        policy = _one(within("policies", plain), "policies", check.key)
        reference = _one(within("policies", versus), "vs_<reference> policies", check.key)
    return Tested(policy, reference, metric_name, family, capacity)


def _phrase_list(names: Iterable[str]) -> str:
    return ", ".join(sorted(names)) or "none"


def claim_problems(claim: Claim, vocabulary: Vocabulary) -> list[str]:
    """How a claim's checks fail to match its sentence (see the module doc)."""
    sentence = normalize(claim.sentence)
    words = [(span, sentence[span[0]:span[1]].lower())
             for span in phrase_spans(vocabulary.comparatives, sentence)]
    if len(words) != 1:
        found = ", ".join(word for _, word in words) or "none"
        return [f"its sentence makes {len(words)} comparisons ({found}); a claim's sentence "
                "makes exactly one"]
    (start, end), word = words[0]
    named: dict[str, set[str]] = {"before": set(), "after": set(), "metrics": set(),
                                  "families": set(), "capacities": set()}
    for (_, last), name in mentions(vocabulary.names, sentence, vocabulary.comparatives):
        group = vocabulary.group_of(name)
        if group == "policies":
            named["before" if last <= start else "after"].add(name)
        else:
            named[group].add(name)
    problems = []
    if not named["before"]:
        problems.append(f"its sentence names no policy before {word!r}")
    if not named["after"]:
        problems.append(f"its sentence names no policy after {word!r} to compare against")
    if not named["metrics"]:
        problems.append("its sentence names no metric")
    negated = bool(phrase_spans(vocabulary.negations, sentence))
    sign = vocabulary.comparatives[word]
    found: list[Tested] = []
    for index, check in enumerate(claim.checks, start=1):
        try:
            item = tested(check, vocabulary)
        except ValueError as error:
            problems.append(f"check {index}: {error}")
            continue
        found.append(item)
        problems += [f"check {index}: {problem}"
                     for problem in _match_problems(item, named, word, vocabulary)]
        if negated != (check.direction in ("no_detected_difference", "equivalent")):
            problems.append(
                f"check {index}: a negated comparison is tested as no_detected_difference or "
                "equivalent, and only a negated one" if negated else
                f"check {index}: {word!r} states a direction; test it as positive or negative")
        elif not negated:
            expected = {"up": 1, "down": -1}.get(sign)
            if expected is None:
                polarity = vocabulary.polarity.get(item.metric)
                if polarity is None:
                    problems.append(f"check {index}: {word!r} needs a metric where higher or "
                                    f"lower is better; {item.metric} has none, so say more or "
                                    "less")
                    continue
                expected = polarity if sign == "good" else -polarity
            if expected != (1 if check.direction == "positive" else -1):
                problems.append(f"check {index}: {word!r} means a "
                                f"{'positive' if expected > 0 else 'negative'} difference in "
                                f"{item.metric}, but the check tests {check.direction}")
    problems += _coverage_problems(found, named, vocabulary)
    return problems


def _match_problems(item: Tested, named: Mapping[str, set[str]], word: str,
                    vocabulary: Vocabulary) -> list[str]:
    problems = []
    if item.policy not in named["before"]:
        problems.append(f"its key compares {item.policy}, which the sentence does not name "
                        f"before {word!r}")
    if item.reference not in named["after"]:
        problems.append(f"its key compares against {item.reference}, which the sentence does "
                        f"not name after {word!r}")
    if item.metric not in named["metrics"]:
        problems.append(f"its key measures {item.metric}, which the sentence does not name")
    for group, value in (("families", item.family), ("capacities", item.capacity)):
        allowed = named[group] or {vocabulary.defaults.get(group), None}
        if value not in allowed:
            what = {"families": "family", "capacities": "capacity level"}[group]
            problems.append(f"its key is for the {what} {value}, but the sentence means "
                            f"{_phrase_list(name for name in allowed if name)}" if value else
                            f"its key names no {what}, but the sentence names "
                            f"{_phrase_list(allowed)}")
    return problems


def _covers(item: Tested, wanted: tuple[str, str, str, str | None, str | None],
            vocabulary: Vocabulary) -> bool:
    """Whether a check tests one wanted comparison; no family or capacity named
    means the default (or a key without that dimension)."""
    policy, reference, metric_name, family, capacity = wanted
    return (item.policy == policy and item.reference == reference
            and item.metric == metric_name
            and (item.family == family if family else
                 item.family in (vocabulary.defaults.get("families"), None))
            and (item.capacity == capacity if capacity else
                 item.capacity in (vocabulary.defaults.get("capacities"), None)))


def _coverage_problems(found: list[Tested], named: Mapping[str, set[str]],
                       vocabulary: Vocabulary) -> list[str]:
    problems = []
    for wanted in itertools.product(
            sorted(named["before"]), sorted(named["after"]), sorted(named["metrics"]),
            sorted(named["families"]) or [None], sorted(named["capacities"]) or [None]):
        policy, reference, metric_name, family, capacity = wanted
        if policy != reference and not any(_covers(item, wanted, vocabulary)
                                           for item in found):
            where = "".join(f" in {name}" for name in (family, capacity) if name)
            problems.append(f"no check tests {policy} against {reference} on "
                            f"{metric_name}{where}")
    return problems


def check_claims(claims: list[Claim], summary: Mapping[str, Any],
                 read_document: Callable[[str], str | None],
                 vocabulary: Vocabulary) -> list[str]:
    """Problems with each claim: its sentence missing from its document, unreadable
    Markdown in that document, checks that do not match the sentence, or a check
    the results do not support."""
    problems = []
    documents: dict[str, str | None] = {}
    for claim in claims:
        if claim.document not in documents:
            text = documents[claim.document] = read_document(claim.document)
            if text is not None:
                problems += [f"{claim.document}:{line}: the claims check cannot read {reason}; "
                             "rewrite it as top-level Markdown"
                             for line, reason in markdown_problems(text)]
        text = documents[claim.document]
        if text is None:
            problems.append(f"{claim.id}: document {claim.document} does not exist")
        elif normalize(claim.sentence) not in {sentence for _, sentence in sentences(text)}:
            problems.append(f"{claim.id}: its sentence is not a visible sentence of "
                            f"{claim.document}")
        problems += [f"{claim.id}: {problem}" for problem in claim_problems(claim, vocabulary)]
        for index, check in enumerate(claim.checks):
            try:
                reason = support(check, summary)
            except KeyError as error:
                reason = str(error.args[0]) if error.args else str(error)
            if reason:
                problems.append(f"{claim.id}, check {index + 1}: {reason}")
    return problems


def sentences_by_document(claims: list[Claim]) -> dict[str, set[str]]:
    """For the lint: each document's claim sentences."""
    out: dict[str, set[str]] = {}
    for claim in claims:
        out.setdefault(claim.document, set()).add(normalize(claim.sentence))
    return out
