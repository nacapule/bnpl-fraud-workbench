"""Draft an investigation memo for one alert packet via the configured Claude model.

The packet (built by ``llm/packet.py``) is the ONLY information the model sees.
Output is schema-validated JSON (see ``llm/prompts/memo_*.md``); parse failures
retry once with an error hint, then raise. Advisory by design: analysts own
decisions (FP-1 §7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.client import LLMResponse, complete_cached, load_config

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

ACTIONS = {"clear", "hold_contact", "decline_block", "escalate"}
PRIORITIES = {"P0", "P1", "P2", "P3"}
PATTERNS = {
    "account_takeover",
    "stolen_card",
    "synthetic_ring",
    "never_pay",
    "inr_abuse",
    "promo_abuse",
    "merchant_bustout",
    "benign",
}
REQUIRED_FIELDS = [
    "signals_observed",
    "hypotheses",
    "policy_citations",
    "recommended_action",
    "priority",
    "evidence_gaps",
    "memo_markdown",
]


def render_prompt(packet: dict[str, Any], prompt_version: str | None = None) -> str:
    version = prompt_version or load_config()["llm"]["tasks"]["triage_memo"]["prompt_version"]
    template = (PROMPTS_DIR / f"{version}.md").read_text()
    return template.replace("{packet_json}", json.dumps(packet, indent=1, sort_keys=True))


def validate_memo(memo: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty = valid)."""
    problems = [f"missing field {f}" for f in REQUIRED_FIELDS if f not in memo]
    if not problems:
        if memo["recommended_action"] not in ACTIONS:
            problems.append(f"bad action {memo['recommended_action']!r}")
        if memo["priority"] not in PRIORITIES:
            problems.append(f"bad priority {memo['priority']!r}")
        if not isinstance(memo["signals_observed"], list) or not memo["signals_observed"]:
            problems.append("signals_observed empty")
        for h in memo.get("hypotheses", []):
            if h.get("pattern") not in PATTERNS:
                problems.append(f"bad hypothesis pattern {h.get('pattern')!r}")
            if h.get("likelihood") not in {"low", "med", "high"}:
                problems.append(f"bad likelihood {h.get('likelihood')!r}")
    return problems


def draft_memo(
    packet: dict[str, Any],
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    offline: bool = False,
) -> tuple[dict[str, Any], LLMResponse]:
    """Returns (validated memo dict, raw LLMResponse)."""
    prompt = render_prompt(packet, prompt_version)
    resp = complete_cached(prompt, task="triage_memo", model=model, offline=offline)
    try:
        memo = resp.parsed_json()
        problems = validate_memo(memo)
    except (ValueError, json.JSONDecodeError) as e:
        memo, problems = {}, [str(e)]
    if problems and not resp.cached and not offline:
        retry_prompt = (
            prompt
            + "\n\nYour previous reply was rejected: "
            + "; ".join(problems)
            + "\nReturn ONLY the corrected JSON object."
        )
        resp = complete_cached(retry_prompt, task="triage_memo", model=model, offline=offline)
        memo = resp.parsed_json()
        problems = validate_memo(memo)
    if problems:
        raise ValueError(f"memo failed validation: {problems}")
    return memo, resp
