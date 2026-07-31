"""Mechanical hallucination check: every concrete token a memo cites must exist
in the case packet the model was shown.

A "concrete token" is anything that looks like data rather than prose:
timestamps, money amounts, bare numbers, and entity ids (``d_123``, ``u_4``,
order/device/card identifiers). A claim with at least one concrete token that
does NOT appear in the packet is unsupported; the memo-level hallucination flag
is "any unsupported claim". This is deliberately strict — an LLM memo earns
trust by quoting the packet, not by paraphrasing plausibly (FP-1 §2.2).

Citation validity: cited rule ids must exist (R01–R12) and are additionally
marked "fired" if present in the packet's fired-rule list.
"""

from __future__ import annotations

import re
from typing import Any

VALID_RULE_IDS = {f"R{i:02d}" for i in range(1, 13)}

_TOKEN_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"),  # timestamps
    re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),  # money
    re.compile(r"\b[a-z]{1,8}_\d+\b"),  # entity ids like d_123, u_88
    re.compile(r"\b\d+(?:\.\d+)?\b"),  # bare numbers (incl. counts)
]


def _flatten(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _flatten(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten(v, out)
    elif obj is not None:
        out.append(str(obj))


def packet_haystack(packet: dict[str, Any]) -> str:
    parts: list[str] = []
    _flatten(packet, parts)
    return "\x00".join(parts)


def _normalize_number(tok: str) -> str:
    t = tok.replace("$", "").replace(",", "").strip()
    if re.fullmatch(r"\d+\.0+", t):
        t = t.split(".")[0]
    return t


def claim_tokens(claim: str) -> list[str]:
    """Extract concrete tokens; bare numbers inside timestamps/ids/money are not
    double-counted (longest patterns run first and mask their span)."""
    masked = claim
    tokens: list[str] = []
    for pat in _TOKEN_PATTERNS:
        for m in pat.finditer(masked):
            tokens.append(m.group())
        masked = pat.sub(" ", masked)
    return tokens


def _numeric_match(tok: str, numbers: list[float]) -> bool:
    """A numeric token matches if some packet number rounds to it at the
    token's own precision — models legitimately round (101.11857 → "101.12");
    they may NOT invent or compute new values (sums/averages stay unsupported)."""
    t = _normalize_number(tok)
    try:
        val = float(t)
    except ValueError:
        return False
    dp = len(t.split(".")[1]) if "." in t else 0
    return any(abs(round(n, dp) - val) < 10 ** -(dp + 6) for n in numbers)


def _packet_numbers(haystack: str) -> list[float]:
    nums = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", haystack):
        try:
            nums.append(float(m.group()))
        except ValueError:  # pragma: no cover
            pass
    return nums


def check_claim(claim: str, haystack: str, numbers: list[float] | None = None
                ) -> dict[str, Any]:
    toks = claim_tokens(claim)
    numbers = numbers if numbers is not None else _packet_numbers(haystack)
    missing = []
    for tok in toks:
        norm = _normalize_number(tok)
        if not norm or norm in haystack:
            continue
        if _numeric_match(tok, numbers):
            continue
        missing.append(tok)
    return {"claim": claim, "tokens": toks, "missing": missing, "supported": not missing}


def verify_memo(memo: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    hay = packet_haystack(packet)
    nums = _packet_numbers(hay)
    checks = [check_claim(c, hay, nums) for c in memo.get("signals_observed", [])]
    fired = {
        str(r.get("id"))
        for r in (packet.get("alert", {}).get("fired_rules") or [])
        if isinstance(r, dict)
    }
    citations = []
    for cit in memo.get("policy_citations", []):
        ids = re.findall(r"\bR\d{2}\b", str(cit))
        for rid in ids or [None]:
            citations.append(
                {
                    "citation": cit,
                    "rule_id": rid,
                    "valid": rid in VALID_RULE_IDS if rid else True,  # sections FP-1 §x ok
                    "fired": rid in fired if rid else None,
                }
            )
    unsupported = [c for c in checks if not c["supported"]]
    invalid_citations = [c for c in citations if not c["valid"]]
    return {
        "n_claims": len(checks),
        "n_unsupported_claims": len(unsupported),
        "unsupported": unsupported,
        "hallucinated": bool(unsupported),
        "citations": citations,
        "n_invalid_citations": len(invalid_citations),
        "citation_fired_rate": (
            sum(1 for c in citations if c["fired"]) / len([c for c in citations if c["rule_id"]])
            if any(c["rule_id"] for c in citations)
            else None
        ),
    }
