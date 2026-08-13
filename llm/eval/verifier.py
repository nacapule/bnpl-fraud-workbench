"""Mechanical grounding check over concrete tokens in ``signals_observed``.

The verifier checks numeric, entity-id, timestamp, and money tokens against the
packet shown to the model.  It uses token boundaries, accepts legitimate numeric
rounding, and recognizes a narrow class of derived list counts/sums.  It does
not establish semantic truth: a real packet token can still be used in a false
relationship, and token-free prose is outside this mechanical check.

Citation validity is separate: cited rule ids must exist (R01-R12), and a valid
rule is additionally marked ``fired`` when it appears in the packet alert.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

VALID_RULE_IDS = {f"R{i:02d}" for i in range(1, 13)}

_TOKEN_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"),
    re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\b[A-Za-z]{1,8}_\d+\b"),
    re.compile(r"\b\d+(?:\.\d+)?\b"),
]
_NUMBER_RE = re.compile(r"^\$?\s?\d[\d,]*(?:\.\d+)?$")
_LIST_ALIASES = {
    "last_orders": ("last_orders", "last orders", "orders"),
    "account_events_90d": ("account_events_90d", "account events", "events"),
}


def _flatten(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.append(str(key))
            _flatten(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _flatten(value, out)
    elif obj is not None:
        out.append(str(obj))


def packet_haystack(packet: dict[str, Any]) -> str:
    parts: list[str] = []
    _flatten(packet, parts)
    return "\x00".join(parts)


def _normalize_number(token: str) -> str:
    normalized = token.replace("$", "").replace(",", "").strip()
    if re.fullmatch(r"\d+\.0+", normalized):
        normalized = normalized.split(".")[0]
    return normalized


def claim_tokens(claim: str) -> list[str]:
    """Extract concrete tokens without double-counting timestamp/id components."""
    masked = claim
    tokens: list[str] = []
    for pattern in _TOKEN_PATTERNS:
        for match in pattern.finditer(masked):
            tokens.append(match.group())
        masked = pattern.sub(" ", masked)
    return tokens


def _numeric_match(token: str, numbers: Iterable[float]) -> bool:
    """Accept packet numbers rounded to the precision used by the claim."""
    normalized = _normalize_number(token)
    try:
        value = float(normalized)
    except ValueError:
        return False
    decimal_places = len(normalized.split(".")[1]) if "." in normalized else 0
    return any(
        abs(round(number, decimal_places) - value) < 10 ** -(decimal_places + 6)
        for number in numbers
    )


def _packet_numbers(haystack: str) -> list[float]:
    numbers = []
    for match in re.finditer(r"-?\d+(?:\.\d+)?", haystack):
        try:
            numbers.append(float(match.group()))
        except ValueError:  # pragma: no cover
            pass
    return numbers


def _token_present(token: str, haystack: str) -> bool:
    normalized = _normalize_number(token)
    if not normalized:
        return True
    boundary = re.compile(rf"(?<![0-9.]){re.escape(normalized)}(?![0-9.])", re.IGNORECASE)
    return boundary.search(haystack) is not None


def _iter_packet_lists(obj: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, list[Any]]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_packet_lists(value, (*path, str(key)))
    elif isinstance(obj, list):
        yield ".".join(path), obj
        for index, value in enumerate(obj):
            yield from _iter_packet_lists(value, (*path, str(index)))


def _list_is_referenced(claim: str, path: str) -> bool:
    claim_lower = claim.lower()
    name = path.rsplit(".", 1)[-1]
    aliases = _LIST_ALIASES.get(name, (name, name.replace("_", " ")))
    return any(re.search(rf"\b{re.escape(alias)}\b", claim_lower) for alias in aliases)


def _derived_numbers(claim: str, packet: dict[str, Any]) -> list[float]:
    """Return count/sum candidates from packet lists explicitly anchored in a claim."""
    candidates: list[float] = []
    claim_lower = claim.lower()
    for path, rows in _iter_packet_lists(packet):
        if not rows or not _list_is_referenced(claim, path):
            continue
        candidates.append(float(len(rows)))
        if not all(isinstance(row, dict) for row in rows):
            numeric_values = [float(value) for value in rows if isinstance(value, int | float)]
            if numeric_values:
                candidates.append(sum(numeric_values))
            continue

        keys = {str(key) for row in rows for key in row}
        for key in keys:
            values = [row.get(key) for row in rows]
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, int | float) and not isinstance(value, bool)
            ]
            if numeric_values and (key.lower() in claim_lower or "total" in claim_lower):
                candidates.append(sum(numeric_values))

            # A value named in the claim can anchor a filtered row count, e.g.
            # "4 approved orders in last_orders".
            for value in {str(value).lower() for value in values if value is not None}:
                if value and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", claim_lower):
                    candidates.append(float(sum(str(item).lower() == value for item in values)))
    return candidates


def check_claim(
    claim: str,
    haystack: str,
    numbers: list[float] | None = None,
    *,
    packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens = claim_tokens(claim)
    numbers = numbers if numbers is not None else _packet_numbers(haystack)
    missing: list[str] = []
    for token in tokens:
        if _token_present(token, haystack) or _numeric_match(token, numbers):
            continue
        missing.append(token)

    classification = "verbatim"
    derived_tokens: list[str] = []
    if missing and packet is not None:
        candidates = _derived_numbers(claim, packet)
        if candidates and all(_NUMBER_RE.fullmatch(token.strip()) for token in missing):
            if all(_numeric_match(token, candidates) for token in missing):
                derived_tokens = missing.copy()
                missing = []
                classification = "derived"
    if missing:
        classification = "unsupported"
    return {
        "claim": claim,
        "tokens": tokens,
        "missing": missing,
        "derived_tokens": derived_tokens,
        "classification": classification,
        "supported": classification != "unsupported",
    }


def verify_memo(memo: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    haystack = packet_haystack(packet)
    numbers = _packet_numbers(haystack)
    checks = [
        check_claim(claim, haystack, numbers, packet=packet)
        for claim in memo.get("signals_observed", [])
    ]
    fired = {
        str(rule.get("id"))
        for rule in (packet.get("alert", {}).get("fired_rules") or [])
        if isinstance(rule, dict)
    }
    citations = []
    for citation in memo.get("policy_citations", []):
        ids = re.findall(r"\bR\d{2}\b", str(citation))
        for rule_id in ids or [None]:
            citations.append(
                {
                    "citation": citation,
                    "rule_id": rule_id,
                    "valid": rule_id in VALID_RULE_IDS if rule_id else True,
                    "fired": rule_id in fired if rule_id else None,
                }
            )
    unsupported = [check for check in checks if check["classification"] == "unsupported"]
    derived = [check for check in checks if check["classification"] == "derived"]
    invalid_citations = [citation for citation in citations if not citation["valid"]]
    checked = len(checks)
    return {
        "n_claims": checked,
        "n_checked_fields": checked,
        "n_verbatim_claims": checked - len(derived) - len(unsupported),
        "n_derived_claims": len(derived),
        "n_unsupported_claims": len(unsupported),
        "unsupported_claim_rate": len(unsupported) / checked if checked else None,
        "unsupported": unsupported,
        "derived": derived,
        "has_unsupported_claim": bool(unsupported),
        "hallucinated": bool(unsupported),  # compatibility for memo consumers
        "citations": citations,
        "n_invalid_citations": len(invalid_citations),
        "citation_fired_rate": (
            sum(1 for citation in citations if citation["fired"])
            / len([citation for citation in citations if citation["rule_id"]])
            if any(citation["rule_id"] for citation in citations)
            else None
        ),
        "scope": "signals_observed concrete numeric/id/timestamp/money tokens; not semantic truth",
    }
