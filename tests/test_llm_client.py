from __future__ import annotations

import json
from types import SimpleNamespace

from llm.client import ClaudeCLIClient, _api_equivalent_cost
from llm.eval.harness import _decision_metrics, evaluate_arm
from llm.triage import validate_memo


def _event(value: dict) -> str:
    return json.dumps(value)


def test_cli_prefers_authoritative_result_usage(monkeypatch):
    stdout = "\n".join(
        [
            _event(
                {
                    "type": "assistant",
                    "message": {
                        "id": "m1",
                        "usage": {"input_tokens": 2, "output_tokens": 4},
                    },
                }
            ),
            _event(
                {
                    "type": "result",
                    "result": '{"ok": true}',
                    "duration_ms": 12,
                    "usage": {"input_tokens": 1200, "output_tokens": 300},
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "llm.client.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )
    response = ClaudeCLIClient("unused").complete("prompt", "claude-sonnet-5")
    assert response.input_tokens == 1200
    assert response.output_tokens == 300
    assert response.cost_usd == 0.0081


def test_cli_sums_non_overlapping_message_usage(monkeypatch):
    stdout = "\n".join(
        [
            _event(
                {
                    "type": "assistant",
                    "message": {
                        "id": "m1",
                        "usage": {"input_tokens": 400, "output_tokens": 50},
                    },
                }
            ),
            _event(
                {
                    "type": "assistant",
                    "message": {
                        "id": "m2",
                        "usage": {
                            "input_tokens": 200,
                            "cache_read_input_tokens": 100,
                            "output_tokens": 25,
                        },
                    },
                }
            ),
            _event({"type": "result", "result": '{"ok": true}'}),
        ]
    )
    monkeypatch.setattr(
        "llm.client.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )
    response = ClaudeCLIClient("unused").complete("prompt", "claude-sonnet-5")
    assert response.input_tokens == 700
    assert response.output_tokens == 75


def test_cost_is_suppressed_below_input_sanity_floor():
    assert _api_equivalent_cost("claude-sonnet-5", 2, 900) is None


def test_memo_validator_enforces_nested_output_types():
    memo = {
        "signals_observed": ["amount 10"],
        "hypotheses": ["never_pay"],
        "policy_citations": [9],
        "recommended_action": "clear",
        "priority": "P2",
        "evidence_gaps": [],
        "memo_markdown": ["not", "text"],
    }
    problems = validate_memo(memo)
    assert "each hypothesis must be an object" in problems
    assert "policy_citations must be a list of strings" in problems
    assert "memo_markdown must be a string" in problems


def test_schema_failure_inclusive_decision_denominator():
    rows = [
        {
            "action": "clear",
            "truth_action": "clear",
            "action_ok": True,
        }
    ]
    failure = {"truth_action": "decline_block"}
    valid = _decision_metrics(rows, [failure], include_schema_failures=False)
    inclusive = _decision_metrics(rows, [failure], include_schema_failures=True)
    assert valid["action_accuracy"] == 1.0
    assert inclusive["action_accuracy"] == 0.5
    assert valid["n"] == 1
    assert inclusive["n"] == 2


def test_offline_cache_miss_is_a_counted_skip(monkeypatch):
    monkeypatch.setattr("llm.eval.harness._packet", lambda case: {"alert": {}})

    def missing(*args, **kwargs):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr("llm.eval.harness.draft_memo", missing)
    result = evaluate_arm(
        [
            {
                "alert_id": 1,
                "truth_action": "clear",
                "truth_actions": ["clear"],
                "truth_pattern": "benign",
            }
        ],
        model="test",
        prompt_version="memo_v2",
        offline=True,
        consistency_cases=0,
        consistency_runs=0,
    )
    assert result["cache_misses"] == 1
    assert result["n_cases"] == 0
