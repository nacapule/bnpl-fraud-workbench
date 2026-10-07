"""Unmatched-token check: concrete tokens in memo text, looked up in the packet.

A memo's text is scanned for concrete tokens: timestamps, money amounts, entity
ids (a short letter prefix, an underscore and digits, such as ``a_2899019``) and
plain numbers. A token is matched when the packet shown to the model contains it:

* an entity id only as a whole identifier, with no letter, digit or underscore
  on either side, so ``DEV_42`` does not match ``OTHERDEV_42X``;
* a number at numeric token boundaries (``7`` does not match ``1370.55``), or
  when some packet number rounds to it at the precision the text uses
  (``$101.12`` matches ``101.11857``);
* a negative number only by a negative packet value: ``-$99`` does not match
  ``99``, while ``-99`` in the packet matches ``-$99``. A number written without
  a sign is matched by its digits whatever the packet value's sign. A hyphen
  after a letter, digit or point is not a sign (``1-3``, ``R03-R06``).

Tokens still missing are classified as derived when every one of them is a
count or a sum over a packet list the text names ("4 approved orders in
last_orders"); otherwise the text is unmatched.

This is a token check, not semantic truth. A real packet number can be used in
a false relationship ("2 prior chargebacks" passes when only the tenure is 2),
and statements without concrete tokens ("no prior orders") are not checked.

Policy citations are checked separately by rule id (:func:`check_citations`).
Section references such as ``FP-1 §999`` are not checked here: FP-1, which the
archived memos cite, had no clause table to check them against.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

FP1_RULE_IDS = frozenset(f"R{i:02d}" for i in range(1, 13))  # archived memos: R12 was valid then
FP2_RULE_IDS = frozenset(f"R{i:02d}" for i in range(1, 12))  # R12 retired under FP-2

CLASSIFICATIONS = ("verbatim", "derived", "unmatched")

_MINUS = "-\u2212"
_MASK = "\x01"  # stands in for a token already taken, so a hyphen after it is not a sign
# What may not precede a minus sign: a hyphen inside a word, a number or a range.
_NOT_BEFORE_SIGN = rf"(?<![\w.)\]{_MASK}])"
_SIGN = rf"{_NOT_BEFORE_SIGN}[{_MINUS}]"
_TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b")
# Policy references ("FP-1 §6.1", "§6.6(b)") name the policy, not packet facts.
_POLICY_REFERENCE = re.compile(
    r"\bFP-\d+(?:\s*§\s?\d+(?:\.\d+)*(?:\([a-z]\))?)?|§\s?\d+(?:\.\d+)*(?:\([a-z]\))?"
)
_MONEY = re.compile(rf"(?:{_SIGN})?\$\s?[{_MINUS}]?\d[\d,]*(?:\.\d+)?")
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]{1,8}_\d+(?![A-Za-z0-9_])")
_NUMBER = re.compile(rf"(?:{_SIGN})?\b\d+(?:\.\d+)?\b")
_NEGATIVE = re.compile(rf"{_SIGN}\d+(?:\.\d+)?")
_PATTERNS = (
    ("timestamp", _TIMESTAMP),
    ("money", _MONEY),
    ("id", _IDENTIFIER),
    ("number", _NUMBER),
)

# The check as the August 2026 results computed it (see archived_check_texts).
_ARCHIVED_PATTERNS = (
    ("timestamp", _TIMESTAMP),
    ("money", re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")),
    ("id", re.compile(r"\b[A-Za-z]{1,8}_\d+\b")),
    ("number", re.compile(r"\b\d+(?:\.\d+)?\b")),
)
_PACKET_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_NUMERIC = re.compile(r"-?\d+(?:\.\d+)?")
_RULE_ID = re.compile(r"\bR\d{2}\b")
_LIST_ALIASES = {
    "last_orders": ("last_orders", "last orders", "orders"),
    "account_events_90d": ("account_events_90d", "account events", "events"),
}


@dataclass(frozen=True)
class TokenCheck:
    """The tokens found in one text and how the packet accounts for them."""

    text: str
    tokens: tuple[str, ...]
    missing: tuple[str, ...]
    derived: tuple[str, ...]
    classification: str  # "verbatim" | "derived" | "unmatched"


@dataclass(frozen=True)
class _Packet:
    packet: Mapping[str, Any]
    haystack: str
    numbers: tuple[float, ...]  # every digit run, signed when a hyphen precedes it
    negatives: tuple[float, ...]  # only values written with a minus sign


def _flatten(obj: Any, out: list[str]) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            out.append(str(key))
            _flatten(value, out)
    elif isinstance(obj, list | tuple):
        for value in obj:
            _flatten(value, out)
    elif obj is not None:
        out.append(str(obj))


def _index(packet: Mapping[str, Any]) -> _Packet:
    parts: list[str] = []
    _flatten(packet, parts)
    haystack = "\x00".join(parts)
    return _Packet(
        packet,
        haystack,
        tuple(float(match.group()) for match in _PACKET_NUMBER.finditer(haystack)),
        tuple(float(_normalize(match.group())) for match in _NEGATIVE.finditer(haystack)),
    )


def _normalize(token: str) -> str:
    """Drop the currency sign and thousands separators; ``-$ 1,200.00`` becomes ``-1200``."""
    text = token.replace("$", "").replace(",", "").replace("\u2212", "-").strip()
    if text.startswith("-"):
        text = "-" + text[1:].lstrip()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return text


def _tokens(text: str, archived: bool) -> list[tuple[str, str]]:
    """Concrete tokens in order of kind, without counting a timestamp's or id's digits twice."""
    masked = text if archived else _POLICY_REFERENCE.sub(_MASK, text)
    found: list[tuple[str, str]] = []
    for kind, pattern in _ARCHIVED_PATTERNS if archived else _PATTERNS:
        found.extend((kind, match.group()) for match in pattern.finditer(masked))
        masked = pattern.sub(" " if archived else _MASK, masked)
    return found


@lru_cache(maxsize=65536)
def _boundary(kind: str, normalized: str, archived: bool) -> re.Pattern[str]:
    escaped = re.escape(normalized)
    if archived:
        pattern = rf"(?<![0-9.]){escaped}(?![0-9.])"
    elif kind == "id":
        pattern = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    elif normalized.startswith("-"):
        pattern = rf"{_NOT_BEFORE_SIGN}{escaped}(?![0-9.])"
    else:
        pattern = rf"(?<![0-9.]){escaped}(?![0-9.])"
    return re.compile(pattern, re.IGNORECASE)


def _present(kind: str, token: str, packet: _Packet, archived: bool) -> bool:
    normalized = _normalize(token)
    if not normalized:
        return True
    return _boundary(kind, normalized, archived).search(packet.haystack) is not None


def _numeric_match(token: str, numbers: Iterable[float], archived: bool) -> bool:
    """Some number rounds to the token at the precision the token is written with.

    ``$99.00`` is written to two places, so 99.49 does not round to it. The
    archived check dropped trailing zeros first, so there it matched.
    """
    normalized = _normalize(token)
    if not _NUMERIC.fullmatch(normalized):
        return False
    value = float(normalized)
    written = normalized if archived else token
    fraction = re.search(r"\.(\d+)", written)
    places = len(fraction.group(1)) if fraction else 0
    return any(abs(round(number, places) - value) < 10 ** -(places + 6) for number in numbers)


def _iter_lists(obj: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, list[Any]]]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from _iter_lists(value, (*path, str(key)))
    elif isinstance(obj, list):
        yield ".".join(path), obj
        for index, value in enumerate(obj):
            yield from _iter_lists(value, (*path, str(index)))


def _list_named(text: str, path: str) -> bool:
    lower = text.lower()
    name = path.rsplit(".", 1)[-1]
    aliases = _LIST_ALIASES.get(name, (name, name.replace("_", " ")))
    return any(re.search(rf"\b{re.escape(alias)}\b", lower) for alias in aliases)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _derived_numbers(text: str, packet: Mapping[str, Any]) -> list[float]:
    """Counts and sums over packet lists that the text names."""
    candidates: list[float] = []
    lower = text.lower()
    for path, rows in _iter_lists(packet):
        if not rows or not _list_named(text, path):
            continue
        candidates.append(float(len(rows)))
        if not all(isinstance(row, Mapping) for row in rows):
            values = [float(value) for value in rows if isinstance(value, int | float)]
            if values:
                candidates.append(sum(values))
            continue
        keys = {str(key) for row in rows for key in row}
        for key in keys:
            column = [row.get(key) for row in rows]
            values = [float(value) for value in column if _is_number(value)]
            if values and (key.lower() in lower or "total" in lower):
                candidates.append(sum(values))
            # A value named in the text anchors a filtered row count, as in
            # "4 approved orders in last_orders".
            for value in {str(value).lower() for value in column if value is not None}:
                if value and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", lower):
                    candidates.append(float(sum(str(item).lower() == value for item in column)))
    return candidates


def _check(text: str, packet: _Packet, archived: bool) -> TokenCheck:
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    found = _tokens(text, archived)
    missing = []
    for kind, token in found:
        # A signed token needs a value written with a minus sign; the archived
        # check had no signed tokens.
        numbers = packet.negatives if _normalize(token).startswith("-") else packet.numbers
        if not (
            _present(kind, token, packet, archived) or _numeric_match(token, numbers, archived)
        ):
            missing.append(token)
    derived: list[str] = []
    if missing and all(_NUMERIC.fullmatch(_normalize(token)) for token in missing):
        candidates = _derived_numbers(text, packet.packet)
        if candidates and all(_numeric_match(token, candidates, archived) for token in missing):
            derived, missing = missing, []
    if missing:
        classification = "unmatched"
    elif derived:
        classification = "derived"
    else:
        classification = "verbatim"
    return TokenCheck(
        text=text,
        tokens=tuple(token for _, token in found),
        missing=tuple(missing),
        derived=tuple(derived),
        classification=classification,
    )


def check_text(text: str, packet: Mapping[str, Any]) -> TokenCheck:
    """Check the concrete tokens of one text against the packet."""
    return _check(text, _index(packet), archived=False)


def check_texts(texts: Iterable[str], packet: Mapping[str, Any]) -> tuple[TokenCheck, ...]:
    """:func:`check_text` for several texts of one packet, indexing the packet once."""
    index = _index(packet)
    return tuple(_check(text, index, archived=False) for text in texts)


def archived_check_texts(texts: Iterable[str], packet: Mapping[str, Any]) -> tuple[TokenCheck, ...]:
    """The check exactly as the August 2026 results computed it.

    Kept only so the archived numbers can be reproduced. It matched an entity
    id inside a longer identifier (``DEV_42`` in ``OTHERDEV_42X``) and dropped
    the sign of a negative amount (``-$99`` matched ``99``). Its
    ``"unmatched"`` texts are the archived results' "unsupported" ones.
    """
    index = _index(packet)
    return tuple(_check(text, index, archived=True) for text in texts)


def summarize(checks: Sequence[TokenCheck]) -> dict[str, int]:
    """Counts over checked texts: all, with any token, derived and unmatched."""
    return {
        "n_texts": len(checks),
        "n_with_tokens": sum(bool(check.tokens) for check in checks),
        "n_derived": sum(check.classification == "derived" for check in checks),
        "n_unmatched": sum(check.classification == "unmatched" for check in checks),
    }


def rule_ids(text: str) -> tuple[str, ...]:
    """The rule ids (``R01`` style) named in a citation, in order."""
    return tuple(_RULE_ID.findall(text))


def check_citations(
    citations: Iterable[str], valid_rule_ids: Collection[str], fired: Collection[str]
) -> list[dict[str, Any]]:
    """One entry per rule id cited: whether the policy has it and whether it fired.

    A citation naming no rule id (``"FP-1 §6.1"``) gets one entry with
    ``rule_id`` None and ``status`` "no rule id": it is neither valid nor
    invalid, and ``valid`` and ``fired`` are None.
    """
    entries: list[dict[str, Any]] = []
    for citation in citations:
        ids = rule_ids(citation)
        if not ids:
            entries.append(
                {
                    "citation": citation,
                    "rule_id": None,
                    "status": "no rule id",
                    "valid": None,
                    "fired": None,
                }
            )
        for rule_id in ids:
            valid = rule_id in valid_rule_ids
            entries.append(
                {
                    "citation": citation,
                    "rule_id": rule_id,
                    "status": "valid" if valid else "invalid",
                    "valid": valid,
                    "fired": rule_id in fired,
                }
            )
    return entries
