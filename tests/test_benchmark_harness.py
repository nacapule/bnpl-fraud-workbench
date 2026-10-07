"""The memo benchmark harness: call caps and retries, quarantine, cache identity, the
coverage gate and scoring, run end to end against a stand-in backend."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llm import client, memo
from llm.eval import harness
from llm.eval.select_cases import case_id, write_benchmark
from llm.packet import build_packet

REPO = Path(__file__).resolve().parents[1]
CASE = json.loads((REPO / "tests" / "fixtures" / "llm" / "quiet_case.json").read_text())
R03 = {"bin_ip_country_mismatch": 1, "avs_mismatch": 1}
SETTLED = {**R03, "unauthorized_disputes_lost_user": 1}
ARMS = {"a": {"backend": "codex", "model": "gpt-6.1-sol", "effort": "high"},
        "b": {"backend": "claude", "model": "claude-opus-5-5", "effort": "high"}}
PRIVATE = ["Jane Example"]


def make_benchmark(directory: Path) -> list[str]:
    rows, packets = [], {}
    for index, (changes, stratum, latent) in enumerate([
            ({}, "legitimate:traveller", "legitimate"),
            (R03, "P-STOLEN", "fraud"),
            (SETTLED, "P-STOLEN", "fraud")]):
        decision_at = pd.Timestamp("2025-07-03 14:05:00") + pd.Timedelta(minutes=index)
        cid = case_id(1041, "baseline", 5012 + index, decision_at)
        context = {**CASE["context_row"], **changes, "order_id": 5012 + index,
                   "decision_at": decision_at}
        order = {**CASE["order"], "order_id": 5012 + index}
        packets[cid] = build_packet(context, order, merchant_category="electronics",
                                    card_bin_country="US", home_country="US")
        rows.append({"case_id": cid, "seed": 1041, "family": "baseline", "stratum": stratum,
                     "cluster": f"account:1041:{377 + index}", "weight": 1.0,
                     "account_key": f"1041:{377 + index}", "episode_key": None,
                     "latent_class": latent,
                     "pattern_id": None if latent == "legitimate" else "P-STOLEN"})
    ids = sorted(packets)
    write_benchmark(directory, benchmark_id="test-bench", phase="development",
                    cases=pd.DataFrame(rows), packets=packets, arms=ARMS,
                    probes={"shuffled": [ids[0]], "renamed": [ids[0]]})
    return ids


def good_memo(packet: dict) -> str:
    context = packet["context"]
    if context["unauthorized_disputes_lost_user"]:
        disposition, citations, check = "decline", ["FP-2 §6.6(a)", "R03"], "none"
    elif context["bin_ip_country_mismatch"]:
        disposition, citations, check = "hold", ["FP-2 §6.6(b)", "R03"], "id_check"
    else:
        disposition, citations, check = "clear", ["FP-2 §6.6(c)"], "none"
    return json.dumps({
        "claims": [{"field": "context.bin_ip_country_mismatch",
                    "value": str(context["bin_ip_country_mismatch"]),
                    "statement": "Card and IP countries.", "derived": None}],
        "hypotheses": [{"explanation": "stolen_card", "likelihood": "low", "reasoning": "r"},
                       {"explanation": "traveller", "likelihood": "low", "reasoning": "r"}],
        "disposition": disposition, "citations": citations, "next_check": check,
        "memo": "Short memo.",
    })


class StandIn:
    """A backend that answers from the packet; arm b misbehaves in three ways."""

    def __init__(self, name: str, failures: dict | None = None):
        self.name, self.calls, self.failures = name, [], dict(failures or {})

    def complete(self, request: client.Request) -> client.Response:
        self.calls.append(request)
        packet = json.loads(request.prompt)
        key = packet["context"]["unauthorized_disputes_lost_user"], packet[
            "context"]["bin_ip_country_mismatch"]
        if self.failures.get(key, 0) > 0:
            self.failures[key] -= 1
            raise client.BackendError("connection reset")
        text, tool_events = good_memo(packet), 0
        if request.backend == "claude":
            if key == (0, 0):
                text = text[:-10]  # truncated
            elif key == (0, 1):
                tool_events = 1  # a tool call in the log
            else:
                text = text.replace("Short memo.", "Memo for Jane Example.")
        summary = client.EventSummary(
            backend=request.backend, cli_version="cli 1", model=request.model,
            effort=request.effort, n_events=3, event_counts={"x": 3},
            tools_offered=0, n_tool_events=tool_events, n_file_events=0,
            n_error_events=0, input_tokens=100, output_tokens=50)
        return client.Response(text=text, summary=summary, duration_ms=5,
                               events=({"type": "result"},))


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    directory = tmp_path / "test-bench"
    make_benchmark(directory)
    return directory


def run(bench: Path, arm: str, tmp_path: Path, **kwargs) -> tuple[dict, StandIn]:
    backend = StandIn(arm, kwargs.pop("failures", None))
    used = harness.run_live(bench, arm, log_dir=tmp_path / "logs", private_terms=PRIVATE,
                            max_calls=kwargs.pop("max_calls", 50),
                            backend_factory=lambda name: backend, **kwargs)
    return used, backend


def test_end_to_end_scores_and_failures(bench: Path, tmp_path: Path) -> None:
    used_a, backend_a = run(bench, "a", tmp_path)
    used_b, _ = run(bench, "b", tmp_path)
    assert used_a["calls"] == used_b["calls"] == 5  # 3 cases and 2 probes
    assert backend_a.calls[0].system_prompt == memo.system_prompt()
    results = harness.evaluate(bench)
    a, b = results["arms"]["a"]["summary"], results["arms"]["b"]["summary"]
    assert a["acceptable"] == {"numerator": 3, "denominator": 3, "value": 1.0}
    assert a["invariance"] == {"renamed": {"numerator": 1, "denominator": 1, "value": 1.0},
                               "shuffled": {"numerator": 1, "denominator": 1, "value": 1.0}}
    assert b["outcomes"] == {"format_failure": 1, "protocol_failure": 2}
    assert b["acceptable"]["numerator"] == 0
    assert results["arms"]["a"]["latent_diagnostic"]["fraud_cleared"]["numerator"] == 0
    assert results["arms"]["a"]["spend"] == {"calls": 5, "input_tokens": 500,
                                             "output_tokens": 250}
    paired = results["statistics"]["paired"]["a vs b"]
    assert (paired["only_first"], paired["only_second"]) == (3, 0)


def test_quarantined_responses_never_enter_the_cache(bench: Path, tmp_path: Path) -> None:
    used, _ = run(bench, "b", tmp_path)
    assert used["quarantined"] == 2  # the tool call and the private name
    for path in (bench / "cache").glob("*.json"):
        record = json.loads(path.read_text())
        text = path.read_text()
        assert "Jane Example" not in text
        if record["outcome"] == "protocol_failure":
            assert "text" not in record and record["quarantined"]
    quarantined = list((tmp_path / "logs" / "quarantine").iterdir())
    assert len(quarantined) == 2
    assert any("Jane Example" in path.read_text() for path in quarantined)
    assert list((tmp_path / "logs" / "test-bench" / "b").glob("*.events.jsonl"))


def test_live_runs_need_the_private_terms(bench: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        harness.run_live(bench, "a", log_dir=tmp_path, private_terms=[], max_calls=5,
                         backend_factory=lambda name: StandIn(name))


def test_call_cap_includes_retries(bench: Path, tmp_path: Path) -> None:
    used, backend = run(bench, "a", tmp_path, max_calls=2, failures={(0, 0): 1})
    assert used["calls"] == 2 and len(backend.calls) == 2
    with pytest.raises(harness.CoverageError):
        harness.evaluate(bench)


def test_one_transport_retry_then_a_recorded_failure(bench: Path, tmp_path: Path) -> None:
    used, _ = run(bench, "a", tmp_path, failures={(0, 1): 1, (1, 1): 2})
    assert used["transport_failures"] == 1
    records = [json.loads(p.read_text()) for p in (bench / "cache").glob("*.json")]
    retried = [r for r in records if r["transport_errors"] and r["outcome"] == "response"]
    failed = [r for r in records if r["outcome"] == "transport_failure"]
    assert len(retried) == 1 and len(failed) == 1 and len(failed[0]["transport_errors"]) == 2


def test_coverage_gate_fails_an_arm_with_missing_cases(bench: Path, tmp_path: Path,
                                                       monkeypatch) -> None:
    run(bench, "a", tmp_path)
    with pytest.raises(harness.CoverageError):  # arm b never ran
        harness.evaluate(bench)
    monkeypatch.setattr(harness, "BENCHMARKS", bench.parent)
    assert harness.main(["--benchmark", bench.name]) == 1
    run(bench, "b", tmp_path)
    assert harness.main(["--benchmark", bench.name]) == 0
    assert (bench / "results.json").exists()


def test_cache_identity_separates_every_setting(bench: Path) -> None:
    definition = harness.load_benchmark(bench)
    arm = definition["arms"]["a"]
    base = harness.call_identity(definition, arm, "c", "primary", "P", "S")
    variants = [
        harness.call_identity(definition, harness.Arm("a", "codex", "gpt-6.1-sol", "xhigh"),
                              "c", "primary", "P", "S"),
        harness.call_identity(definition, harness.Arm("a", "claude", "gpt-6.1-sol", "high"),
                              "c", "primary", "P", "S"),
        harness.call_identity(definition, harness.Arm("a", "codex", "gpt-6-sol", "high"),
                              "c", "primary", "P", "S"),
        harness.call_identity(definition, arm, "c", "primary", "P", "S2"),
        harness.call_identity(definition, arm, "c", "primary", "P2", "S"),
        harness.call_identity(definition, arm, "c", "shuffled", "P", "S"),
        harness.call_identity(definition, arm, "c2", "primary", "P", "S"),
        harness.call_identity({**definition, "code_sha256": {"referee.py": "x"}}, arm, "c",
                              "primary", "P", "S"),
    ]
    keys = {harness.cache_key(identity) for identity in [base, *variants]}
    assert len(keys) == len(variants) + 1


def test_a_changed_referee_or_tampered_record_is_refused(bench: Path, tmp_path: Path) -> None:
    run(bench, "a", tmp_path)
    run(bench, "b", tmp_path)
    views = json.loads((bench / "referee.json").read_text())
    first = sorted(views)[0]
    views[first]["standard"] = ["hold"]
    (bench / "referee.json").write_text(json.dumps(views))
    with pytest.raises(RuntimeError):
        harness.evaluate(bench)


def test_case_ids_identify_an_order_and_decision_whatever_the_bands() -> None:
    first = case_id(416, "baseline", 77, "2025-06-02 10:00:00")
    assert first == case_id(416, "baseline", 77, pd.Timestamp("2025-06-02 10:00:00"))
    assert first != case_id(416, "baseline", 77, "2025-06-02 10:00:01")
    assert first != case_id(416, "fraud_mix_shift", 77, "2025-06-02 10:00:00")
