"""Checks a memo's structured claims against its packet.

Each claim names a packet field by its path, gives the field's value and states the
fact. A plain claim is verified when the field exists and the value matches it (a
number may be rounded to the precision the claim writes). A derived claim declares
its operation and input fields; the verifier recomputes it from the packet and
compares. The factual-error rate is the share of claims that fail: an unknown field,
a wrong value, or a derived value that does not recompute.

What this does not check: whether the claim's sentence says what its field means.
The sentence, the hypotheses' reasoning and the memo text are covered only by the
coarser unmatched-token check (:mod:`llm.eval.tokens`): numbers and identifiers in
them that appear nowhere in the packet.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm.eval import tokens

PATH_PART = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)((?:\[\d{1,9}\])*)")
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d{1,4})?")
MAX_PLACES = 30  # finer than any float carries
TRUE = {"1", "true", "yes"}
FALSE = {"0", "false", "no"}
MISSING = object()


def resolve(packet: Mapping[str, Any], path: str) -> Any:
    """The value at a dotted packet path with optional list indexes, or MISSING."""
    node: Any = packet
    for part in path.strip().split("."):
        match = PATH_PART.fullmatch(part)
        if match is None or not isinstance(node, Mapping) or match.group(1) not in node:
            return MISSING
        node = node[match.group(1)]
        for index in re.findall(r"\[(\d+)\]", match.group(2)):
            if not isinstance(node, list) or int(index) >= len(node):
                return MISSING
            node = node[int(index)]
    return node


def _number(text: str) -> tuple[float, int] | None:
    """A claimed number and the decimal places it is written to ('$1,234.50' ->
    (1234.5, 2), '1.5e-8' -> (1.5e-08, 9)); None for anything else, infinity and
    numbers too large for a float included."""
    cleaned = text.strip().replace(",", "").replace("$", "").rstrip("%").strip()
    if not NUMBER.fullmatch(cleaned):
        return None
    mantissa, _, exponent = cleaned.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    value = float(cleaned)
    if not math.isfinite(value):
        return None
    return value, min(MAX_PLACES, max(0, decimals - int(exponent or 0)))


def _as_float(value: int | float) -> float | None:
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _numbers_match(claimed: Any, actual: int | float) -> bool:
    if isinstance(claimed, int) and not isinstance(claimed, bool) and isinstance(actual, int):
        return claimed == actual
    exact = _as_float(actual)
    if exact is None:
        return False
    if isinstance(claimed, bool):
        return exact == float(claimed)
    text = repr(claimed) if isinstance(claimed, float) else str(claimed).strip()
    parsed = _number(text)
    if parsed is None:
        lowered = text.lower()
        return (lowered in TRUE and actual == 1) or (lowered in FALSE and actual == 0)
    value, places = parsed
    return (abs(round(exact, places) - value) < 10 ** -(places + 6)
            or abs(exact - value) <= 1e-9 * max(1.0, abs(value)))


def value_matches(claimed: Any, actual: Any) -> bool:
    """Whether a claimed value states the packet's value."""
    if actual is None:
        return claimed is None or str(claimed).strip().lower() in {"", "null", "none"}
    if isinstance(actual, list):
        if isinstance(claimed, str):
            try:
                claimed = json.loads(claimed)
            except (json.JSONDecodeError, RecursionError):
                return False
        return claimed == actual
    if claimed is None or isinstance(claimed, dict | list):
        return False
    if isinstance(actual, bool):
        return str(claimed).strip().lower() in (TRUE if actual else FALSE)
    if isinstance(actual, int | float):
        return _numbers_match(claimed, actual)
    return str(claimed).strip().casefold() == str(actual).strip().casefold()


def recompute(operation: str, values: Sequence[Any]) -> float | None:
    """The derived value from its inputs' packet values, or None when undefined."""
    if operation == "count":
        if len(values) == 1 and isinstance(values[0], list):
            return float(len(values[0]))
        return float(sum(bool(value) for value in values))
    if not values or not all(isinstance(value, int | float) and not isinstance(value, bool)
                             for value in values):
        return None
    numbers = [float(value) for value in values]
    if operation == "sum":
        return sum(numbers)
    if operation == "min":
        return min(numbers)
    if operation == "max":
        return max(numbers)
    if len(numbers) != 2:
        return None
    if operation == "difference":
        return numbers[0] - numbers[1]
    if operation == "ratio":
        return numbers[0] / numbers[1] if numbers[1] else None
    return None


@dataclass(frozen=True)
class ClaimCheck:
    field: str
    status: str  # verified, wrong_value, unknown_field, derived_ok, derived_wrong,
    #              derived_invalid, unverifiable
    expected: Any = None

    @property
    def error(self) -> bool:
        return self.status not in ("verified", "derived_ok")


def check_claim(claim: Mapping[str, Any], packet: Mapping[str, Any]) -> ClaimCheck:
    """One claim's check; a claim the verifier cannot evaluate is an error, never an
    exception."""
    try:
        return _check_claim(claim, packet)
    except (ArithmeticError, ValueError, TypeError, RecursionError):
        return ClaimCheck(str(claim.get("field")), "unverifiable")


def _check_claim(claim: Mapping[str, Any], packet: Mapping[str, Any]) -> ClaimCheck:
    derived = claim.get("derived")
    if derived:
        values = [resolve(packet, path) for path in derived["inputs"]]
        if any(value is MISSING for value in values):
            return ClaimCheck(claim["field"], "derived_invalid")
        expected = recompute(derived["operation"], values)
        if expected is None:
            return ClaimCheck(claim["field"], "derived_invalid")
        status = "derived_ok" if value_matches(claim["value"], expected) else "derived_wrong"
        return ClaimCheck(claim["field"], status, expected)
    actual = resolve(packet, claim["field"])
    if actual is MISSING:
        return ClaimCheck(claim["field"], "unknown_field")
    if isinstance(actual, dict):
        return ClaimCheck(claim["field"], "unknown_field")  # a group of facts, not one
    status = "verified" if value_matches(claim["value"], actual) else "wrong_value"
    return ClaimCheck(claim["field"], status, actual)


def verify_memo(memo: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    """Claim checks and the unmatched-token check for one validated memo."""
    checks = [check_claim(claim, packet) for claim in memo["claims"]]
    prose = ([claim["statement"] for claim in memo["claims"]]
             + [hypothesis["reasoning"] for hypothesis in memo["hypotheses"]]
             + [memo["memo"]])
    token_checks = [tokens.check_text(text, packet) for text in prose]
    errors = [check for check in checks if check.error]
    return {
        "n_claims": len(checks),
        "n_claim_errors": len(errors),
        "n_derived_claims": sum(check.status.startswith("derived") for check in checks),
        "claim_errors": [{"field": c.field, "status": c.status, "expected": c.expected}
                         for c in errors],
        "has_claim_error": bool(errors),
        "unmatched_tokens": tokens.summarize(token_checks),
        "has_unmatched_token": any(c.classification == "unmatched" for c in token_checks),
    }
