"""The memo role: an advisory drafter for one review decision.

The system prompt is the drafter's instructions (``llm/prompts/<version>.md``)
followed by the full fraud policy; the user message is the case packet
(:mod:`llm.packet`). The drafter returns one JSON object: structured claims tied to
packet fields, competing hypotheses including a benign one, a suggested disposition
(``needs_check`` included), the clauses and rules it rests on, the cheapest next
check, and a short memo. It sets no queue priority and triggers no action (fraud
policy §10).

:func:`parse` turns a response into a memo or a list of problems. Validation
checks every field's type before any vocabulary, so malformed output is a counted
failure and never an exception.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.actions import MEMO_DISPOSITIONS

REPO = Path(__file__).resolve().parent.parent
PROMPTS = REPO / "llm" / "prompts"
POLICY_PATH = REPO / "policy" / "fraud-policy.md"
PROMPT_VERSION = "memo_fp2_v1"

FIELDS = ("claims", "hypotheses", "disposition", "citations", "next_check", "memo")
CLAIM_FIELDS = ("field", "value", "statement", "derived")
HYPOTHESIS_FIELDS = ("explanation", "likelihood", "reasoning")
OPERATIONS = ("difference", "sum", "ratio", "count", "min", "max")
FRAUD_EXPLANATIONS = ("account_takeover", "stolen_card", "synthetic_identity", "never_pay",
                      "inr_abuse", "promo_abuse", "merchant_bustout")
BENIGN_EXPLANATIONS = ("legitimate_customer", "new_customer", "household", "traveller",
                       "mover", "new_device", "gift_buyer", "credit_loss")
LIKELIHOODS = ("low", "medium", "high")
NEXT_CHECKS = ("contact", "id_check", "none")
FENCE = re.compile(r"\A```(?:json)?\s*\n(.*)\n```\Z", re.DOTALL)


def system_prompt(version: str = PROMPT_VERSION) -> str:
    """The drafter's instructions followed by the full policy text."""
    instructions = (PROMPTS / f"{version}.md").read_text().rstrip()
    return f"{instructions}\n\n---\n\n{POLICY_PATH.read_text().rstrip()}\n"


def user_prompt(packet: Mapping[str, Any], *, shuffle_seed: int | None = None) -> str:
    """The packet as the user message. ``shuffle_seed`` reorders the context facts (an
    invariance probe: the order of the facts carries no information)."""
    if shuffle_seed is not None:
        context = list(packet["context"].items())
        random.Random(shuffle_seed).shuffle(context)
        packet = {**packet, "context": dict(context)}
    return json.dumps(packet, indent=1, ensure_ascii=False)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_keys(obj: Mapping[str, Any], keys: tuple[str, ...], what: str) -> list[str]:
    problems = [f"{what} lacks {key}" for key in keys if key not in obj]
    problems += [f"{what} has unexpected field {key!r}" for key in obj if key not in keys]
    return problems


def _claim_problems(index: int, claim: Any) -> list[str]:
    what = f"claims[{index}]"
    if not isinstance(claim, dict):
        return [f"{what} must be an object"]
    problems = _exact_keys(claim, CLAIM_FIELDS, what)
    if problems:
        return problems
    if not _is_text(claim["field"]):
        problems.append(f"{what}.field must be a non-empty string")
    if not _is_text(claim["statement"]):
        problems.append(f"{what}.statement must be a non-empty string")
    derived = claim["derived"]
    if derived is not None:
        if not isinstance(derived, dict):
            problems.append(f"{what}.derived must be null or an object")
        elif sorted(derived) != ["inputs", "operation"]:
            problems.append(f"{what}.derived must have exactly operation and inputs")
        elif not isinstance(derived["operation"], str) or not isinstance(derived["inputs"], list):
            problems.append(f"{what}.derived has wrong types")
        elif not derived["inputs"] or not all(_is_text(item) for item in derived["inputs"]):
            problems.append(f"{what}.derived.inputs must be non-empty strings")
        elif derived["operation"] not in OPERATIONS:
            problems.append(f"{what}.derived.operation {derived['operation']!r} is unknown")
    return problems


def _hypothesis_problems(index: int, hypothesis: Any) -> list[str]:
    what = f"hypotheses[{index}]"
    if not isinstance(hypothesis, dict):
        return [f"{what} must be an object"]
    problems = _exact_keys(hypothesis, HYPOTHESIS_FIELDS, what)
    if problems:
        return problems
    if not all(isinstance(hypothesis[key], str) for key in HYPOTHESIS_FIELDS):
        return [f"{what} fields must be strings"]
    if hypothesis["explanation"] not in FRAUD_EXPLANATIONS + BENIGN_EXPLANATIONS:
        problems.append(f"{what}.explanation {hypothesis['explanation']!r} is unknown")
    if hypothesis["likelihood"] not in LIKELIHOODS:
        problems.append(f"{what}.likelihood {hypothesis['likelihood']!r} is unknown")
    if not hypothesis["reasoning"].strip():
        problems.append(f"{what}.reasoning is empty")
    return problems


def validate(memo: Any) -> list[str]:
    """Problems with a parsed memo (empty when valid): types first, then vocabularies."""
    if not isinstance(memo, dict):
        return ["the response must be a JSON object"]
    problems = _exact_keys(memo, FIELDS, "memo")
    if problems:
        return problems
    types = {"claims": list, "hypotheses": list, "disposition": str, "citations": list,
             "next_check": str, "memo": str}
    problems = [f"{key} must be a {kind.__name__}" for key, kind in types.items()
                if not isinstance(memo[key], kind)]
    if problems:
        return problems
    if not memo["claims"]:
        problems.append("claims is empty")
    for index, claim in enumerate(memo["claims"]):
        problems += _claim_problems(index, claim)
    for index, hypothesis in enumerate(memo["hypotheses"]):
        problems += _hypothesis_problems(index, hypothesis)
    if len(memo["hypotheses"]) < 2:
        problems.append("fewer than two hypotheses")
    elif not problems and not any(h["explanation"] in BENIGN_EXPLANATIONS
                                  for h in memo["hypotheses"]):
        problems.append("no benign hypothesis")
    if not all(isinstance(citation, str) for citation in memo["citations"]):
        problems.append("citations must be strings")
    if memo["disposition"] not in MEMO_DISPOSITIONS:
        problems.append(f"disposition {memo['disposition']!r} is unknown")
    if memo["next_check"] not in NEXT_CHECKS:
        problems.append(f"next_check {memo['next_check']!r} is unknown")
    if not memo["memo"].strip():
        problems.append("memo text is empty")
    return problems


def parse(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """The memo in a response, or the problems that make it a failure.

    The response must be one JSON object, optionally inside a single ```json fence;
    anything else (prose around it, a second object, a truncated object) fails.
    """
    body = text.strip()
    fenced = FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        memo = json.loads(body)
    except json.JSONDecodeError as error:
        return None, [f"not one JSON object: {error.msg} at {error.pos}"]
    problems = validate(memo)
    return (memo if not problems else None), problems
