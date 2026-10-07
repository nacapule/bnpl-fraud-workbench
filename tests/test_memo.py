"""The memo role's prompt and output contract: malformed output is a counted failure,
never an exception, and validation checks types before vocabularies."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from llm import memo
from llm.packet import CONTEXT_FIELDS

REPO = Path(__file__).resolve().parents[1]

VALID = {
    "claims": [
        {"field": "context.bin_ip_country_mismatch", "value": "1",
         "statement": "The card was issued in another country than the IP.", "derived": None},
        {"field": "context.installments_paid_user", "value": "12",
         "statement": "Twelve installments were paid.",
         "derived": None},
        {"field": "unpaid installments", "value": "0",
         "statement": "None of the due installments is unpaid.",
         "derived": {"operation": "difference",
                     "inputs": ["context.installments_due_user",
                                "context.installments_paid_user"]}},
    ],
    "hypotheses": [
        {"explanation": "stolen_card", "likelihood": "medium", "reasoning": "R03 holds."},
        {"explanation": "traveller", "likelihood": "medium", "reasoning": "Could be abroad."},
    ],
    "disposition": "needs_check",
    "citations": ["FP-2 §6.6(b)", "R03"],
    "next_check": "id_check",
    "memo": "Card evidence only; an identity check would decide it.",
}


def with_change(path: tuple, value) -> dict:
    changed = copy.deepcopy(VALID)
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return changed


def test_a_valid_memo_parses() -> None:
    parsed, problems = memo.parse(json.dumps(VALID))
    assert problems == [] and parsed == VALID
    fenced, problems = memo.parse("```json\n" + json.dumps(VALID) + "\n```")
    assert problems == [] and fenced == VALID


@pytest.mark.parametrize("text", [
    "", "not json", json.dumps(VALID)[:-5], "Here is the memo: " + json.dumps(VALID),
    json.dumps(VALID) + "\n" + json.dumps(VALID), json.dumps([VALID]),
])
def test_anything_but_one_json_object_fails(text: str) -> None:
    parsed, problems = memo.parse(text)
    assert parsed is None and problems


@pytest.mark.parametrize("path,value", [
    (("claims",), [9]),                       # a claim that is not an object
    (("claims",), []),                        # no claims
    (("claims", 0, "field"), 7),
    (("claims", 2, "derived"), {"operation": "average", "inputs": ["context.x"]}),
    (("claims", 2, "derived"), {"operation": "sum", "inputs": []}),
    (("hypotheses",), ["stolen_card"]),        # strings instead of objects
    (("hypotheses",), [VALID["hypotheses"][0]]),  # one hypothesis only
    (("hypotheses", 1, "explanation"), "account_takeover"),  # no benign hypothesis
    (("hypotheses", 0, "likelihood"), ["high"]),  # wrong type before the vocabulary
    (("hypotheses", 0, "reasoning"), ""),
    (("disposition",), ["hold"]),              # a list where a string belongs
    (("disposition",), "hold_contact"),        # the old vocabulary
    (("citations",), "R03"),
    (("citations",), [3]),
    (("next_check",), None),
    (("next_check",), "phone_call"),
    (("memo",), ""),
    (("priority",), "P0"),                     # the drafter sets no queue priority
])
def test_malformed_output_is_a_failure_not_an_exception(path, value) -> None:
    parsed, problems = memo.parse(json.dumps(with_change(path, value)))
    assert parsed is None
    assert problems


def test_missing_fields_are_named() -> None:
    broken = {key: value for key, value in VALID.items() if key != "next_check"}
    assert memo.validate(broken) == ["memo lacks next_check"]


def test_system_prompt_carries_the_instructions_and_the_whole_policy() -> None:
    prompt = memo.system_prompt()
    policy = (REPO / "policy" / "fraud-policy.md").read_text().rstrip()
    assert prompt.rstrip().endswith(policy)
    assert "FP-2" in prompt and "priority" in prompt
    instructions = (REPO / "llm" / "prompts" / f"{memo.PROMPT_VERSION}.md").read_text()
    assert "R12" not in instructions and "FP-1" not in instructions
    for name in ("vendor", "hold_contact", "decline_block"):
        assert name not in instructions


def test_prompt_describes_exactly_the_packet_fields() -> None:
    instructions = (REPO / "llm" / "prompts" / f"{memo.PROMPT_VERSION}.md").read_text()
    described = {field for field in CONTEXT_FIELDS if f"`{field}`" in instructions}
    assert described == set(CONTEXT_FIELDS)
    for explanation in memo.FRAUD_EXPLANATIONS + memo.BENIGN_EXPLANATIONS:
        assert f"`{explanation}`" in instructions


def test_shuffled_probe_reorders_facts_without_changing_them() -> None:
    packet = {"decision": {}, "order": {}, "context": {name: i for i, name in
                                                       enumerate(CONTEXT_FIELDS)}}
    shuffled = json.loads(memo.user_prompt(packet, shuffle_seed=3))
    assert shuffled["context"] == packet["context"]
    assert list(shuffled["context"]) != list(packet["context"])


@pytest.mark.parametrize("text", [
    json.dumps(VALID).replace('"12"', "NaN"),
    json.dumps(VALID).replace('"12"', "Infinity"),
    json.dumps(VALID).replace('"12"', "-Infinity"),
    "[" * 100000 + "]" * 100000,
])
def test_non_standard_or_pathological_json_is_a_failure(text: str) -> None:
    parsed, problems = memo.parse(text)
    assert parsed is None and problems
