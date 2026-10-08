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
PRIVATE = ["Jane Example", "jane@example.invalid", "Zoë Q"]
SIZES = {"development_cases": 3, "final_cases": 3, "min_final_seeds": 1,
         "max_cases_per_cluster": 2, "probe_cases": 1, "probe_kinds": ["shuffled", "renamed"],
         "max_output_tokens": 500, "input_overhead_tokens": 0, "context_tokens": 200000,
         "token_stop_margin": 0}
BOUND = 1000  # every stand-in call's token bound
CALL_BOUND = harness.call_bound
CHECK_ISOLATION = harness.check_isolation


@pytest.fixture(autouse=True)
def small_benchmarks(monkeypatch) -> None:
    monkeypatch.setattr(harness, "SIZES", SIZES)
    monkeypatch.setattr(harness, "CAPS", {})
    monkeypatch.setattr(harness, "call_bound", lambda system, user: BOUND)
    monkeypatch.setattr(harness, "check_isolation", lambda *args, **kwargs: None)


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
                     "cluster": f"group 1041:{377 + index}", "weight": 1.0,
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
                    "statement": "Card and IP countries.", "derived": None},
                   {"field": "order.device", "value": packet["order"]["device"],
                    "statement": "The device the order came from.", "derived": None}],
        "hypotheses": [{"explanation": "stolen_card", "likelihood": "low", "reasoning": "r"},
                       {"explanation": "traveller", "likelihood": "low", "reasoning": "r"}],
        "disposition": disposition, "citations": citations, "next_check": check,
        "memo": f"Short memo about device {packet['order']['device']}.",
    })


def summary(request: client.Request, **changes) -> client.EventSummary:
    fields = dict(backend=request.backend, cli_version="cli 1", model=request.model,
                  effort=request.effort, n_events=3, event_counts={"x": 3},
                  tools_offered=0, n_tool_events=0, n_file_events=0, n_error_events=0,
                  input_tokens=100, output_tokens=50, isolation="iso 1")
    return client.EventSummary(**{**fields, **changes})


class StandIn:
    """A backend that answers from the packet; arm b misbehaves in three ways.

    ``failures`` maps a case's (settled, R03) facts to transport failures to raise;
    ``answers`` replaces the response for a case with (text, summary changes)."""

    def __init__(self, name: str, failures: dict | None = None, answers: dict | None = None,
                 version: str = "cli 1", error: str = "connection reset",
                 bounds_output: bool = True, interrupt_after: int | None = None,
                 check: tuple[str, dict] | BaseException = ("NO", {})):
        self.name, self.calls, self.failures = name, [], dict(failures or {})
        self.answers, self.version, self.error = dict(answers or {}), version, error
        self.bounds_output, self.interrupt_after = bounds_output, interrupt_after
        self.check, self.checks = check, []

    def fingerprint(self, model: str, effort: str, max_output_tokens=None) -> dict[str, str]:
        assert (max_output_tokens is not None) is self.bounds_output
        return {"cli_version": self.version, "isolation": "iso 1"}

    def complete(self, request: client.Request) -> client.Response:
        if request.system_prompt == harness.CANARY_SYSTEM:  # the run's isolation check
            self.checks.append(request)
            if isinstance(self.check, BaseException):
                raise self.check
            text, changes = self.check
            return client.Response(text=text, summary=summary(request, **changes),
                                   duration_ms=5, events=({"type": "result"},))
        self.calls.append(request)
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            raise KeyboardInterrupt  # the run is stopped while the model works
        packet = json.loads(request.prompt)
        key = packet["context"]["unauthorized_disputes_lost_user"], packet[
            "context"]["bin_ip_country_mismatch"]
        if self.failures.get(key, 0) > 0:
            self.failures[key] -= 1
            raise client.BackendError(self.error)
        text, changes = good_memo(packet), {}
        if request.backend == "claude":
            if key == (0, 0):
                text = text[:-10]  # truncated
            elif key == (0, 1):
                changes = {"n_tool_events": 1}  # a tool call in the log
            else:
                text = text.replace("Short memo", "Memo for Jane Example")
        if key in self.answers:
            answer = self.answers[key]
            if isinstance(answer, list):
                answer = answer.pop(0) if len(answer) > 1 else answer[0]
            text, changes = answer[0] or text, answer[1]
        return client.Response(text=text, summary=summary(request, **changes), duration_ms=5,
                               events=({"type": "result"},))


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    directory = tmp_path / "test-bench"
    make_benchmark(directory)
    return directory


def run(bench: Path, arm: str, tmp_path: Path, backend: StandIn | None = None,
        **kwargs) -> tuple[dict, StandIn]:
    backend = backend or StandIn(arm, kwargs.pop("failures", None))
    used = harness.run_live(bench, arm, log_dir=tmp_path / "logs", private_terms=PRIVATE,
                            max_calls=kwargs.pop("max_calls", 50),
                            ledger_path=kwargs.pop("ledger_path", tmp_path / "ledger.json"),
                            backend_factory=lambda name: backend, **kwargs)
    return used, backend


def records(bench: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in (bench / "cache").glob("*.json")]


def test_end_to_end_scores_and_failures(bench: Path, tmp_path: Path) -> None:
    used_a, backend_a = run(bench, "a", tmp_path)
    used_b, _ = run(bench, "b", tmp_path)
    assert used_a["calls"] == used_b["calls"] == 5  # 3 cases and 2 probes
    assert backend_a.calls[0].system_prompt == memo.system_prompt()
    results = harness.evaluate(bench)
    a, b = results["arms"]["a"]["summary"], results["arms"]["b"]["summary"]
    assert a["acceptable"] == {"numerator": 3, "denominator": 3, "value": 1.0}
    every = {"numerator": 1, "denominator": 1, "value": 1.0}
    assert a["invariance"] == {probe: dict.fromkeys(("disposition", "action", "next_check",
                                                     "complete_pass", "functional"), every)
                               for probe in ("renamed", "shuffled")}
    assert a["complete_pass"] == {"numerator": 3, "denominator": 3, "value": 1.0}
    assert a["complete_pass_natural"] == a["acceptable_natural"] == 1.0
    assert b["complete_pass"]["numerator"] == 0
    assert b["components"]["format_valid"]["numerator"] == 0
    assert a["claim_errors"]["numerator"] == 0
    assert b["outcomes"] == {"format_failure": 1, "protocol_failure": 2}
    assert b["acceptable"]["numerator"] == 0
    assert results["arms"]["a"]["latent_diagnostic"]["fraud_cleared"]["numerator"] == 0
    assert results["arms"]["a"]["spend"] == {"calls": 5, "input_tokens": 500,
                                             "output_tokens": 250, "charged_tokens": 750,
                                             "unknown_usage_calls": 0}
    for measure in ("complete_pass", "acceptable"):
        paired = results["statistics"]["paired"]["a vs b"][measure]
        assert (paired["only_first"], paired["only_second"]) == (3, 0)
        assert paired["clusters_favouring_first"] == 3 and paired["cluster_sign_test_p"] == 0.25
    assert results["pins"]["a"]["cli_version"] == "cli 1"


def test_renamed_probes_are_verified_against_the_packet_the_model_saw(bench: Path,
                                                                      tmp_path: Path) -> None:
    run(bench, "a", tmp_path)
    run(bench, "b", tmp_path)
    cases = harness.evaluate(bench)["arms"]["a"]["cases"]
    renamed = [row for key, row in cases.items() if key.endswith("/renamed")]
    assert renamed and all(row["verification"]["n_claim_errors"] == 0 for row in renamed)
    assert all(not row["verification"]["has_unmatched_token"] for row in renamed)
    # and a renamed memo that names the original placeholder is a wrong value
    ids = sorted(case["case_id"] for case in harness.load_benchmark(bench)["cases"])
    packet = json.loads((bench / "packets" / f"{ids[0]}.json").read_text())
    shown, _ = harness.probe_input(packet, "renamed", ids[0])
    assert shown["order"]["device"] != packet["order"]["device"]
    stale = json.loads(good_memo(packet))
    view = harness.referee.view(shown)
    record = {"outcome": "response", "text": json.dumps(stale)}
    assert harness.score_record(record, shown, view)["verification"]["n_claim_errors"] == 1


def test_quarantined_responses_never_enter_the_cache(bench: Path, tmp_path: Path) -> None:
    used, _ = run(bench, "b", tmp_path)
    assert used["quarantined"] == 2  # the tool call and the private name
    for path in (bench / "cache").glob("*.json"):
        record = json.loads(path.read_text())
        assert "Jane Example" not in path.read_text()
        if record["outcome"] == "protocol_failure":
            assert "text" not in record and "summary" not in record and record["quarantined"]
    quarantined = list((tmp_path / "logs" / "quarantine").iterdir())
    assert len(quarantined) == 2
    assert any("Jane Example" in path.read_text() for path in quarantined)
    assert list((tmp_path / "logs" / "test-bench" / "b").glob("*.events.jsonl"))


def test_private_terms_in_errors_or_metadata_never_reach_the_cache(bench: Path,
                                                                   tmp_path: Path) -> None:
    backend = StandIn("a", failures={(0, 0): 2}, error="auth failed for jane@example.invalid",
                      answers={(0, 1): (None, {"event_counts": {"Jane Example": 1}}),
                               (1, 1): (None, {"model": "gpt-6.1-sol (Jane Example)"}),
                               (0, 0): (None, {"event_counts": {"Zoë Q": 1}})})
    used, _ = run(bench, "a", tmp_path, backend=backend)
    # the first case fails in transport twice; its two probes then name a private term
    assert used["quarantined"] == 4 and used["transport_failures"] == 1
    for path in (bench / "cache").glob("*.json"):
        text = path.read_text().lower()
        assert "jane" not in text and "auth failed" not in text and "zoë" not in text
    errors = list((tmp_path / "logs" / "test-bench" / "a").glob("*.errors.txt"))
    assert errors and "jane@example.invalid" in errors[0].read_text()


def test_escaped_or_decomposed_private_terms_are_found(bench: Path, tmp_path: Path) -> None:
    shown = {"context": {"unauthorized_disputes_lost_user": 0, "bin_ip_country_mismatch": 1},
             "order": {"device": "D1"}}
    answer = {**json.loads(good_memo(shown)), "memo": "Ask jane@example.invalid."}
    text = json.dumps(answer).replace("@", "\\u0040")  # the address only as a JSON escape
    assert "@" not in text and json.loads(text)["memo"] == "Ask jane@example.invalid."
    backend = StandIn("a", answers={(0, 1): (text, {}),
                                    (1, 1): (None, {"event_counts": {"Zoe\u0308 Q": 1}})})
    used, _ = run(bench, "a", tmp_path, backend=backend)
    assert used["quarantined"] == 2
    assert all(r["outcome"] != "response" or "jane" not in r["text"] for r in records(bench))


def test_a_private_term_in_the_cli_version_is_refused_before_pinning(bench: Path,
                                                                     tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private term"):
        run(bench, "a", tmp_path, backend=StandIn("a", version="cli 1 (jane@example.invalid)"))
    assert not (bench / "pins.json").exists()


def test_live_runs_need_the_private_terms(bench: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        harness.run_live(bench, "a", log_dir=tmp_path, private_terms=[" "], max_calls=5,
                         backend_factory=lambda name: StandIn(name))


def test_call_cap_includes_retries(bench: Path, tmp_path: Path) -> None:
    used, backend = run(bench, "a", tmp_path, max_calls=2, failures={(0, 0): 1})
    assert used["calls"] == 2 and len(backend.calls) == 2 and used["stopped"]
    with pytest.raises(harness.CoverageError):
        harness.evaluate(bench)


def test_the_arm_cap_holds_across_runs(bench: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 3}})
    first, _ = run(bench, "a", tmp_path, max_calls=2)
    second, backend = run(bench, "a", tmp_path, max_calls=2)
    assert first["calls"] == 2 and second["calls"] == 1 and len(backend.calls) == 1
    assert "3 calls" in second["stopped"]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["arms"]["a"]["calls"] == 3
    assert ledger["arms"]["a"]["by_benchmark"]["test-bench"]["calls"] == 3


def test_a_call_runs_only_if_its_bound_fits_and_unknown_usage_is_charged_it(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 100, "tokens": 1500}})
    used, backend = run(bench, "a", tmp_path)
    assert len(backend.calls) == 4  # 4 x 150 tokens; a fifth's bound of 1000 would not fit
    assert all(call.max_output_tokens == 500 for call in backend.calls)
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert ledger["charged_tokens"] == 600 and ledger["pending"] == {}
    unknown = StandIn("a", answers={key: (None, {"output_tokens": None})
                                    for key in [(0, 0), (0, 1), (1, 1)]})
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 100}})
    fresh = tmp_path / "fresh" / "test-bench"
    make_benchmark(fresh)
    run(fresh, "a", tmp_path, backend=unknown, ledger_path=tmp_path / "other.json")
    other = json.loads((tmp_path / "other.json").read_text())["arms"]["a"]
    assert other["unknown_usage_calls"] == 5 and other["charged_tokens"] == 5 * BOUND


def test_a_token_cap_needs_a_backend_that_bounds_its_output(bench: Path, tmp_path: Path,
                                                            monkeypatch) -> None:
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 100, "tokens": 10 ** 9}})
    with pytest.raises(ValueError, match="bounds its output"):
        run(bench, "a", tmp_path, backend=StandIn("a", bounds_output=False))
    monkeypatch.setattr(harness, "CAPS", {})
    used, backend = run(bench, "a", tmp_path, backend=StandIn("a", bounds_output=False))
    assert used["calls"] == 5 and all(call.max_output_tokens is None for call in backend.calls)


def test_an_interrupted_call_stays_charged_in_full(bench: Path, tmp_path: Path,
                                                   monkeypatch) -> None:
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 100, "tokens": 10 ** 6}})
    with pytest.raises(KeyboardInterrupt):
        run(bench, "a", tmp_path, backend=StandIn("a", interrupt_after=2))
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert ledger["calls"] == 2 and len(ledger["pending"]) == 1  # reserved before the call
    run(bench, "a", tmp_path)
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert ledger["interrupted_calls"] == 1 and ledger["pending"] == {}
    assert ledger["calls"] == 2 + 1 + 3 and ledger["charged_tokens"] == 5 * 150 + BOUND
    # the records account for the same spend: the lost attempt is in its case's record
    usage = [record["usage"] for record in records(bench)]
    assert sum(u["calls"] for u in usage) == ledger["calls"]
    assert sum(u["charged_tokens"] for u in usage) == ledger["charged_tokens"]


def test_a_case_stopped_before_its_retry_keeps_its_attempt(bench: Path, tmp_path: Path,
                                                           monkeypatch) -> None:
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 1}})
    used, _ = run(bench, "a", tmp_path, failures={(0, 0): 1})
    assert used["calls"] == 1 and used["stopped"]
    receipts = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]["receipts"]
    assert [r["charged_tokens"] for r in receipts.values()] == [BOUND]  # kept by case
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 100}})
    _, backend = run(bench, "a", tmp_path, backend=StandIn("a", failures={(0, 0): 1}))
    [first] = [r for r in records(bench) if r["identity"]["case"].endswith("5012-20250703T140500")
               and r["identity"]["probe"] == "primary"]
    assert first["outcome"] == "transport_failure"  # its one retry also failed
    assert first["attempts"] == 2 and first["usage"]["charged_tokens"] == 2 * BOUND
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert sum(r["usage"]["charged_tokens"] for r in records(bench)) == ledger["charged_tokens"]


def test_a_setup_failure_stops_the_run_and_is_not_charged(bench: Path, tmp_path: Path) -> None:
    class Unready(StandIn):
        def complete(self, request):
            raise client.BackendError("no credentials", called=False)

    used, _ = run(bench, "a", tmp_path, backend=Unready("a"))
    assert used["calls"] == 0 and "before asking the model" in used["stopped"]
    assert not list((bench / "cache").glob("*.json"))
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert ledger["calls"] == 0 and ledger["pending"] == {}


def test_two_arms_pinning_at_once_keep_both_pins(bench: Path, monkeypatch) -> None:
    import threading
    import time

    read = harness.read_pins

    def slow(directory):
        pins = read(directory)
        time.sleep(0.2)  # both runs read before either writes, without the lock
        return pins

    monkeypatch.setattr(harness, "read_pins", slow)
    definition = harness.load_benchmark(bench)
    threads = [threading.Thread(target=harness.pin_arm, args=(
        bench, definition["arms"][name], {"cli_version": "cli 1", "isolation": "iso 1"}))
        for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert set(read(bench)) == {"a", "b"}


def test_a_call_is_bounded_over_every_request_the_cli_may_send() -> None:
    # 3 prompt bytes; output 500 per request; request k carries k earlier outputs and the
    # CLI's message before it; ten requests: 1 + 3 continuations + a retry that restarts
    # the count + 3 more + a nudge + a retry after a refusal
    assert harness.REQUESTS_PER_CALL == 10
    later = 500 + harness.NUDGE_TOKENS
    assert CALL_BOUND("ab", "c") == sum(3 + k * later + 500 for k in range(10)) == 72530
    assert harness.largest_input("ab", "c") == 3 + 9 * later


def test_the_real_bound_keeps_every_request_inside_the_context_window() -> None:
    sizes = harness.load("llm")["benchmark"]
    packet = json.loads(json.dumps(build_packet(
        {**CASE["context_row"], **R03}, CASE["order"], merchant_category="electronics",
        card_bin_country="GB", home_country="US"), default=str))
    prompts = memo.system_prompt(), memo.user_prompt(packet)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(harness, "SIZES", sizes)
        assert harness.largest_input(*prompts) <= sizes["context_tokens"]


def test_a_call_that_could_outgrow_the_context_window_is_not_made(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "SIZES", {**SIZES, "context_tokens": 1000})
    with pytest.raises(harness.BudgetError, match="context window"):
        run(bench, "a", tmp_path)
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert ledger["calls"] == 0 and ledger["pending"] == {}
    used, _ = run(bench, "a", tmp_path, backend=StandIn("a", bounds_output=False))
    assert used["calls"] == 5  # no output bound, so nothing to keep inside the window


def test_token_caps_stop_admitting_calls_a_margin_below_them(bench: Path, tmp_path: Path,
                                                            monkeypatch) -> None:
    monkeypatch.setattr(harness, "SIZES", {**SIZES, "token_stop_margin": 500})
    monkeypatch.setattr(harness, "CAPS", {"a": {"calls": 100, "tokens": 1500}})
    used, backend = run(bench, "a", tmp_path)
    assert len(backend.calls) == 1  # 150 + the next bound of 1000 + 500 > 1500
    assert "short of the cap" in used["stopped"]
    monkeypatch.setattr(harness, "CAPS", {})
    fresh = tmp_path / "fresh" / "test-bench"
    make_benchmark(fresh)
    used, backend = run(fresh, "a", tmp_path, max_tokens=1600,
                        ledger_path=tmp_path / "other.json")
    assert len(backend.calls) == 1 and "short of" in used["stopped"]


def test_one_live_run_per_arm_and_no_lost_updates_between_arms(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    with harness.Ledger(path, "a", calls=1, tokens=None) as first:
        with pytest.raises(harness.BudgetError):
            harness.Ledger(path, "a", calls=1, tokens=None)
        with harness.Ledger(path, "b", calls=None, tokens=None) as other:
            key_a = first.reserve("x", 100, case="case a")
            key_b = other.reserve("x", 100, case="case b")
            assert first.refusal(100)  # the pending call counts against the cap of one
            first.settle(key_a, 10, 5)
            other.settle(key_b, 20, 5)
    arms = json.loads(path.read_text())["arms"]
    assert (arms["a"]["calls"], arms["a"]["charged_tokens"]) == (1, 15)
    assert (arms["b"]["calls"], arms["b"]["charged_tokens"]) == (1, 25)
    with harness.Ledger(path, "a", calls=1, tokens=None):  # the lock was released
        pass


def test_an_escaped_surrogate_is_a_format_failure_and_results_are_written(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    broken = good_memo({"context": {"unauthorized_disputes_lost_user": 0,
                                    "bin_ip_country_mismatch": 0}, "order": {"device": "D1"}})
    broken = broken.replace("context.bin_ip_country_mismatch", "context.\\ud800")
    run(bench, "a", tmp_path, backend=StandIn("a", answers={(0, 0): (broken, {})}))
    run(bench, "b", tmp_path)
    outcomes = harness.evaluate(bench)["arms"]["a"]["summary"]["outcomes"]
    assert outcomes.get("format_failure", 0) >= 1
    monkeypatch.setattr(harness, "BENCHMARKS", bench.parent)
    assert harness.main(["--benchmark", bench.name]) == 0


def test_a_call_over_its_bound_parks_the_arm(bench: Path, tmp_path: Path) -> None:
    greedy = StandIn("a", answers={(0, 0): (None, {"output_tokens": 2000})})
    with pytest.raises(harness.BudgetError):
        run(bench, "a", tmp_path, backend=greedy)
    path = tmp_path / "ledger.json"
    assert "more than its bound" in json.loads(path.read_text())["arms"]["a"]["parked"]
    used, backend = run(bench, "a", tmp_path)  # a later run makes no call
    assert used["calls"] == 0 and not backend.calls and "parked" in used["stopped"]
    data = json.loads(path.read_text())
    del data["arms"]["a"]["parked"]  # resolved by hand
    path.write_text(json.dumps(data))
    used, _ = run(bench, "a", tmp_path)
    assert used["recorded"] == 5
    spent = sorted(record["usage"]["charged_tokens"] for record in records(bench))
    assert spent[-1] == 2100 + 150  # the case keeps the call over its bound and its retry


@pytest.fixture
def checked(monkeypatch) -> None:
    monkeypatch.setattr(harness, "check_isolation", CHECK_ISOLATION)


@pytest.mark.usefixtures("checked")
def test_each_run_checks_its_isolation_once_before_its_first_call(bench: Path,
                                                                  tmp_path: Path) -> None:
    used, backend = run(bench, "a", tmp_path)
    assert len(backend.checks) == 1 and len(backend.calls) == 5 and used["calls"] == 6
    request = backend.checks[0]
    assert (request.model, request.effort, request.max_output_tokens) == (
        "gpt-6.1-sol", "high", 500)  # the arm's own model, effort and output bound
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert ledger["calls"] == 6 and "parked" not in ledger
    [entry] = json.loads((bench / "canaries.json").read_text())
    assert entry == {"arm": "a", "cli_version": "cli 1", "isolation": "iso 1",
                     "verdict": "no", "outcome": "passed", "input_tokens": 100,
                     "output_tokens": 50}
    again, backend = run(bench, "a", tmp_path)  # nothing left to call: no check either
    assert again["calls"] == 0 and not backend.checks
    assert list((tmp_path / "logs" / "test-bench" / "a").glob("canary-*.text.txt"))


@pytest.mark.usefixtures("checked")
@pytest.mark.parametrize("check", [("YES", {}), ("Yes, the style notes.", {}),
                                   ("NO", {"n_hook_events": 1}),
                                   ("NO", {"n_unrecognized_events": 1}),
                                   ("NO", {"model": "another-model"})])
def test_a_failed_isolation_check_parks_the_arm_before_any_memo_call(
        bench: Path, tmp_path: Path, check) -> None:
    with pytest.raises(harness.IsolationError):
        run(bench, "a", tmp_path, backend=StandIn("a", check=check))
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert ledger["calls"] == 1 and "isolation check" in ledger["parked"]
    [entry] = json.loads((bench / "canaries.json").read_text())
    assert entry["outcome"] == "failed" and "style" not in json.dumps(entry)
    used, backend = run(bench, "a", tmp_path)
    assert used["calls"] == 0 and not backend.calls and not backend.checks
    assert not list((bench / "cache").glob("*.json"))


@pytest.mark.usefixtures("checked")
def test_an_isolation_check_that_fails_in_transport_stops_without_parking(
        bench: Path, tmp_path: Path) -> None:
    used, backend = run(bench, "a", tmp_path,
                        backend=StandIn("a", check=client.BackendError("reset")))
    assert used["calls"] == 1 and not backend.calls and "isolation check" in used["stopped"]
    for answer in ("NO", "I cannot determine that."):  # an error, whatever the answer
        used, backend = run(bench, "a", tmp_path, backend=StandIn("a", check=(answer, {
            "n_error_events": 1})))
        assert not backend.calls and "error events" in used["stopped"]
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert "parked" not in ledger and ledger["calls"] == 3
    outcomes = [e["outcome"] for e in json.loads((bench / "canaries.json").read_text())]
    assert outcomes == ["transport failure", "error events", "error events"]
    used, _ = run(bench, "a", tmp_path)
    assert used["recorded"] == 5


def test_only_a_plain_no_passes_the_isolation_check() -> None:
    answers = {"NO": "no", "No.": "no", " no\n": "no", "YES": "yes", "yes!": "yes",
               "Nope": "unclear", "NO, but": "unclear", "": "unclear", "NO 123": "unclear",
               "N0O": "unclear", "NO ＹＥＳ": "unclear", "ＮＯ": "unclear", "**NO**": "unclear",
               "yeſ": "unclear"}
    assert {text: harness.canary_verdict(text) for text in answers} == answers


@pytest.mark.usefixtures("checked")
def test_an_isolation_check_over_its_bound_parks_the_arm_before_any_log(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    def unwritable(path, text):
        raise OSError("no space left on the log volume")

    monkeypatch.setattr(harness, "_write_log", unwritable)
    with pytest.raises(harness.BudgetError):
        run(bench, "a", tmp_path, backend=StandIn("a", check=("NO", {"output_tokens": 2000})))
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert "more than its bound" in ledger["parked"] and ledger["charged_tokens"] == 2100
    [entry] = json.loads((bench / "canaries.json").read_text())
    assert entry["outcome"] == "over its bound" and entry["output_tokens"] == 2000


def test_a_memo_call_over_its_bound_parks_the_arm_whatever_the_logs_do(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    def unwritable(path, text):
        raise OSError("no space left on the log volume")

    monkeypatch.setattr(harness, "_write_log", unwritable)
    greedy = StandIn("a", answers={(0, 0): (None, {"output_tokens": 2000})})
    with pytest.raises(harness.BudgetError):
        run(bench, "a", tmp_path, backend=greedy)
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert "more than its bound" in ledger["parked"]


@pytest.mark.parametrize("stop", ["call", "settle", "log", "record"])
def test_records_match_the_ledger_wherever_a_run_stops(bench: Path, tmp_path: Path,
                                                       monkeypatch, stop) -> None:
    # the first call's run ends while the model works, just before the ledger settles a
    # 150-token answer, just after it, or as the record is written
    target, name = {"call": (StandIn, "complete"), "settle": (harness.Ledger, "settle"),
                    "log": (harness, "_write_log"), "record": (harness, "private_terms_in")
                    }[stop]
    real = getattr(target, name)

    def interrupted(*args, **kwargs):
        if stop == "record" and len(args[0]) < 3:  # the fingerprint and arm checks
            return real(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(target, name, interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(bench, "a", tmp_path)
    monkeypatch.setattr(target, name, real)
    used, _ = run(bench, "a", tmp_path)
    assert used["recorded"] == 5
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    usage = [record["usage"] for record in records(bench)]
    assert sum(u["calls"] for u in usage) == ledger["calls"]
    assert sum(u["charged_tokens"] for u in usage) == ledger["charged_tokens"]
    lost = {"call": BOUND, "settle": BOUND, "log": 150, "record": 150}[stop]
    assert ledger["calls"] == 6 and ledger["charged_tokens"] == 5 * 150 + lost


def test_records_match_the_ledger_after_a_stop_just_after_a_reservation(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    real = harness.Ledger.reserve

    def reserve_then_stop(self, *args, **kwargs):
        real(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(harness.Ledger, "reserve", reserve_then_stop)
    with pytest.raises(KeyboardInterrupt):
        run(bench, "a", tmp_path)
    monkeypatch.setattr(harness.Ledger, "reserve", real)
    run(bench, "a", tmp_path)
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    usage = [record["usage"] for record in records(bench)]
    assert sum(u["charged_tokens"] for u in usage) == ledger["charged_tokens"] == 5 * 150 + BOUND
    assert sum(u["calls"] for u in usage) == ledger["calls"] == 6


def test_records_match_the_ledger_after_a_stop_just_after_a_release(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    class Unready(StandIn):
        def complete(self, request):
            raise client.BackendError("no credentials", called=False)

    run(bench, "a", tmp_path, backend=StandIn("a", failures={(0, 0): 1}), max_calls=1)
    real = harness.Ledger.release

    def release_then_stop(self, key):
        real(self, key)
        raise KeyboardInterrupt

    monkeypatch.setattr(harness.Ledger, "release", release_then_stop)
    with pytest.raises(KeyboardInterrupt):
        run(bench, "a", tmp_path, backend=Unready("a"))
    monkeypatch.setattr(harness.Ledger, "release", real)
    run(bench, "a", tmp_path)
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    usage = [record["usage"] for record in records(bench)]
    assert sum(u["charged_tokens"] for u in usage) == ledger["charged_tokens"] == 5 * 150 + BOUND
    assert sum(u["calls"] for u in usage) == ledger["calls"] == 6


def test_a_setup_failure_after_an_attempt_leaves_records_matching_the_ledger(
        bench: Path, tmp_path: Path) -> None:
    class UnreadyAfterOne(StandIn):
        def complete(self, request):
            if self.calls:
                raise client.BackendError("no credentials", called=False)
            return super().complete(request)

    run(bench, "a", tmp_path, backend=StandIn("a", failures={(0, 0): 1}), max_calls=1)
    used, _ = run(bench, "a", tmp_path, backend=UnreadyAfterOne("a"))
    assert "before asking the model" in used["stopped"]
    run(bench, "a", tmp_path)
    assert len(records(bench)) == 5
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    usage = [record["usage"] for record in records(bench)]
    assert sum(u["charged_tokens"] for u in usage) == ledger["charged_tokens"]
    assert sum(u["calls"] for u in usage) == ledger["calls"]


@pytest.mark.usefixtures("checked")
def test_a_failed_isolation_check_is_parked_even_if_nothing_after_it_can_be_written(
        bench: Path, tmp_path: Path, monkeypatch) -> None:
    def unwritable(*args, **kwargs):
        raise OSError("read-only benchmark folder")

    monkeypatch.setattr(harness, "_record_canary", unwritable)
    monkeypatch.setattr(harness, "_write_log", unwritable)
    with pytest.raises(OSError):
        run(bench, "a", tmp_path, backend=StandIn("a", check=("YES", {})))
    ledger = json.loads((tmp_path / "ledger.json").read_text())["arms"]["a"]
    assert "isolation check failed" in ledger["parked"] and ledger["calls"] == 1


def test_one_retry_for_transport_or_error_events_then_a_recorded_failure(
        bench: Path, tmp_path: Path) -> None:
    errored = (None, {"n_error_events": 1})
    backend = StandIn("a", failures={(0, 1): 1, (1, 1): 2},
                      answers={(0, 0): [errored, (None, {})]})
    used, _ = run(bench, "a", tmp_path, backend=backend)
    assert used["transport_failures"] == 1
    retried = [r for r in records(bench) if r["attempts"] == 2 and r["outcome"] == "response"]
    failed = [r for r in records(bench) if r["outcome"] == "transport_failure"]
    assert len(retried) == 2  # one transport retry, one error-event retry
    assert len(failed) == 1 and failed[0]["failed_attempts"] == 2
    assert all("connection reset" not in json.dumps(r) for r in records(bench))
    usages = sorted((r["usage"]["charged_tokens"], r["usage"]["unknown_usage_calls"])
                    for r in retried)
    assert usages == [(300, 0), (BOUND + 150, 1)]  # every attempt's spend is kept
    assert failed[0]["usage"] == {"calls": 2, "input_tokens": 0, "output_tokens": 0,
                                  "charged_tokens": 2 * BOUND, "unknown_usage_calls": 2}


def test_consecutive_transport_failures_stop_the_run(bench: Path, tmp_path: Path) -> None:
    backend = StandIn("a", failures={(0, 0): 9, (0, 1): 9, (1, 1): 9})
    used, _ = run(bench, "a", tmp_path, backend=backend)
    assert used["transport_failures"] == 3 and "in a row" in used["stopped"]
    assert used["calls"] == 6


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
    pin = {"cli_version": "cli 1", "isolation": "iso 1"}
    base = harness.call_identity(definition, arm, pin, "c", "primary", "P", "S")
    variants = [
        harness.call_identity(definition, harness.Arm("a", "codex", "gpt-6.1-sol", "xhigh"),
                              pin, "c", "primary", "P", "S"),
        harness.call_identity(definition, harness.Arm("a", "claude", "gpt-6.1-sol", "high"),
                              pin, "c", "primary", "P", "S"),
        harness.call_identity(definition, harness.Arm("a", "codex", "gpt-6-sol", "high"),
                              pin, "c", "primary", "P", "S"),
        harness.call_identity(definition, harness.Arm("a2", "codex", "gpt-6.1-sol", "high"),
                              pin, "c", "primary", "P", "S"),
        harness.call_identity(definition, arm, {**pin, "cli_version": "cli 2"}, "c",
                              "primary", "P", "S"),
        harness.call_identity(definition, arm, {**pin, "isolation": "iso 2"}, "c",
                              "primary", "P", "S"),
        harness.call_identity(definition, arm, pin, "c", "primary", "P", "S2"),
        harness.call_identity(definition, arm, pin, "c", "primary", "P2", "S"),
        harness.call_identity(definition, arm, pin, "c", "shuffled", "P", "S"),
        harness.call_identity(definition, arm, pin, "c2", "primary", "P", "S"),
        harness.call_identity({**definition, "code_sha256": {"llm/referee.py": "x"}}, arm,
                              pin, "c", "primary", "P", "S"),
    ]
    keys = {harness.cache_key(identity) for identity in [base, *variants]}
    assert len(keys) == len(variants) + 1


def test_a_changed_cli_or_isolation_is_refused(bench: Path, tmp_path: Path) -> None:
    run(bench, "a", tmp_path, max_calls=2)
    with pytest.raises(harness.FrozenError):  # the pinned CLI was "cli 1"
        run(bench, "a", tmp_path, backend=StandIn("a", version="cli 2"))
    drifting = StandIn("a", answers={(0, 1): (None, {"cli_version": "cli 2"}),
                                     (1, 1): (None, {"cli_version": "cli 2"})})
    with pytest.raises(harness.FrozenError):  # the CLI changed during a run
        run(bench, "a", tmp_path, backend=drifting)


def test_a_record_whose_summary_does_not_fit_the_arm_is_refused(bench: Path,
                                                                tmp_path: Path) -> None:
    run(bench, "a", tmp_path)
    run(bench, "b", tmp_path)
    path = next(p for p in (bench / "cache").glob("*.json")
                if json.loads(p.read_text())["outcome"] == "response")
    record = json.loads(path.read_text())
    record["summary"]["isolation"] = "iso 0"
    path.write_text(json.dumps(record))
    with pytest.raises(harness.FrozenError):
        harness.evaluate(bench)


def test_recorded_hashes_are_enforced_before_calls_and_scores(bench: Path, tmp_path: Path,
                                                              monkeypatch) -> None:
    run(bench, "a", tmp_path)
    run(bench, "b", tmp_path)
    definition = json.loads((bench / "benchmark.json").read_text())
    assert set(definition["code_sha256"]) >= {"core/evidence.py", "llm/eval/harness.py",
                                              "core/stats.py", "llm/referee.py"}
    stale = {**definition, "code_sha256": {name: "obsolete"
                                           for name in definition["code_sha256"]}}
    (bench / "benchmark.json").write_text(json.dumps(stale))
    with pytest.raises(harness.FrozenError):
        harness.evaluate(bench)
    with pytest.raises(harness.FrozenError):
        run(bench, "a", tmp_path)
    (bench / "benchmark.json").write_text(json.dumps(definition))
    original = harness.code_sha256
    current = original()
    monkeypatch.setattr(harness, "code_sha256",
                        lambda: {**current, "llm/referee.py": "edited after the calls"})
    with pytest.raises(harness.FrozenError):
        harness.evaluate(bench)
    amended = harness.evaluate(bench, amend="verifier fix")
    assert amended["scoring_amendment"] == {"changed_code": ["llm/referee.py"],
                                            "reason": "verifier fix"}
    monkeypatch.setattr(harness, "code_sha256", original)
    packet = sorted((bench / "packets").glob("*.json"))[0]
    packet.write_text(packet.read_text().replace("electronics", "groceries"))
    with pytest.raises(harness.FrozenError):
        harness.evaluate(bench, amend="anything")  # packets cannot be amended


def test_a_changed_referee_view_is_refused(bench: Path, tmp_path: Path) -> None:
    run(bench, "a", tmp_path)
    run(bench, "b", tmp_path)
    views = json.loads((bench / "referee.json").read_text())
    first = sorted(views)[0]
    views[first]["standard"] = ["hold"]
    (bench / "referee.json").write_text(json.dumps(views))
    with pytest.raises(harness.FrozenError):
        harness.evaluate(bench)


def definition_of(bench: Path) -> dict:
    return json.loads((bench / "benchmark.json").read_text())


@pytest.mark.parametrize("change", [
    lambda d: d["cases"].pop(),                                        # too few cases
    lambda d: d["cases"][0].update(seed=35244829),                      # a final seed
    lambda d: d["cases"][0].update(family="fraud_mix_shift"),           # not baseline
    lambda d: [case.update(account="1041:1") for case in d["cases"]],   # three on one account
    lambda d: d["cases"][0].update(weight=0),
    lambda d: d["cases"][0].pop("packet_sha256"),
    lambda d: d["probes"].update(other=[d["cases"][0]["case_id"]]),
    lambda d: d["probes"].update(renamed=["not-a-case"]),
])
def test_a_development_benchmark_must_have_its_shape(bench: Path, change) -> None:
    definition = definition_of(bench)
    harness.check_shape(definition)
    change(definition)
    with pytest.raises(harness.ShapeError):
        harness.check_shape(definition)


def test_a_final_benchmark_needs_final_seeds_and_every_probe(bench: Path) -> None:
    definition = definition_of(bench)
    definition["phase"] = "final"
    for case in definition["cases"]:
        case["seed"] = 35244829
    harness.check_shape(definition)
    without = {**definition, "probes": {"shuffled": definition["probes"]["shuffled"]}}
    with pytest.raises(harness.ShapeError):
        harness.check_shape(without)
    with pytest.raises(harness.ShapeError):
        harness.check_shape(definition, {**SIZES, "min_final_seeds": 3})
    with pytest.raises(harness.ShapeError, match="baseline worlds only"):
        harness.check_shape({**definition, "cases": [{**case, "family": "lag_half"}
                                                     for case in definition["cases"]]})
    definition["cases"][0]["seed"] = 1041
    with pytest.raises(harness.ShapeError):
        harness.check_shape(definition)


def test_paired_inference_counts_clusters_not_linked_cases() -> None:
    def arms_for(pairs):
        cases = [{"case_id": f"c{i}", "cluster": cluster, "weight": 1.0}
                 for i, (cluster, _a, _b) in enumerate(pairs)]
        arms = {name: {"cases": {f"c{i}/primary": {"acceptable": row[1 + index],
                                                   "complete_pass": row[1 + index]}
                                 for i, row in enumerate(pairs)}}
                for index, name in enumerate(("a", "b"))}
        return harness.statistics({"cases": cases}, arms)["paired"]["a vs b"]["acceptable"]

    linked = arms_for([(f"g{i // 2}", True, False) for i in range(6)])
    independent = arms_for([(f"g{i}", True, False) for i in range(3)])
    assert linked["only_first"] == 6 and linked["clusters_favouring_first"] == 3
    assert linked["cluster_sign_test_p"] == independent["cluster_sign_test_p"] == 0.25
    assert linked["difference"] == 1.0


def test_case_ids_identify_an_order_and_decision_whatever_the_bands() -> None:
    first = case_id(416, "baseline", 77, "2025-06-02 10:00:00")
    assert first == case_id(416, "baseline", 77, pd.Timestamp("2025-06-02 10:00:00"))
    assert first != case_id(416, "baseline", 77, "2025-06-02 10:00:01")
    assert first != case_id(416, "fraud_mix_shift", 77, "2025-06-02 10:00:00")


# ------------------------------------------------- headline, components and diagnostics


def checked_packet(outcome: str | None = None) -> dict:
    """The R03 case's packet at its review, or after its id_check answered."""
    from core.actions import Check, CheckOutcome
    from core.evidence import CheckResult

    at = pd.Timestamp("2025-07-03 16:05:00")
    context = {**CASE["context_row"], **R03, "decision_at": at}
    checks = () if outcome is None else (
        CheckResult(Check("id_check"), CheckOutcome(outcome), at.to_pydatetime()),)
    return build_packet(context, CASE["order"], merchant_category="electronics",
                        card_bin_country="US", home_country="US", checks=checks)


def test_needs_check_is_the_hold_only_with_a_required_check_not_yet_run() -> None:
    from llm import referee

    review = checked_packet()
    view = referee.view(review)
    assert harness.canonical_action("needs_check", "id_check", view, review) == "hold"
    assert harness.canonical_action("needs_check", "none", view, review) == "needs_check"
    assert harness.canonical_action("needs_check", "contact", view, review) == "needs_check"
    assert harness.canonical_action("clear", "none", view, review) == "clear"
    after = checked_packet("passed")  # §5.3(a): the check has run, every hold prohibited
    assert harness.canonical_action("needs_check", "id_check", referee.view(after),
                                    after) == "needs_check"


def test_the_single_most_likely_hypothesis() -> None:
    def h(explanation: str, likelihood: str) -> dict:
        return {"explanation": explanation, "likelihood": likelihood, "reasoning": "r"}

    assert harness.top_hypothesis([h("stolen_card", "high"), h("traveller", "low")]) == (
        "stolen_card", False)
    assert harness.top_hypothesis([h("stolen_card", "medium"), h("traveller", "medium")]) == (
        None, True)
    from core.world import PATTERNS

    assert set(harness.PATTERN_EXPLANATION) == set(PATTERNS)
    assert set(harness.PATTERN_EXPLANATION.values()) <= set(memo.FRAUD_EXPLANATIONS)


@pytest.mark.parametrize(("standard", "clauses", "families", "row"), [
    (["clear"], {"clear": "§6.6(c)"}, [], "§6.6(c)"),
    (["hold"], {"hold": "§6.6(b)"}, ["Card"], "§6.6(b), one family"),
    (["hold"], {"hold": "§6.6(b)"}, ["Card", "Linkage"], "§6.6(b), two or more families"),
    (["decline"], {"decline": "§6.6(a)"}, ["Card"], "§6.6(a)"),
    (["clear"], {"clear": "§5.3(a)"}, ["Card"], "§5.3(a)"),
    (["decline"], {"decline": "§5.3(b)"}, ["Card"], "§5.3(b), without Linkage"),
    (["escalate"], {"escalate": "§5.3(b)"}, ["Linkage"], "§5.3(b), with Linkage"),
])
def test_cases_are_reported_by_policy_row(standard, clauses, families, row) -> None:
    assert harness.referee_row({"standard": standard, "clauses": clauses,
                                "families": families}) == row


def test_the_natural_mix_weights_each_axis_by_its_share() -> None:
    definition = {"cases": [{"axis": "review", "weight": 1.0}, {"axis": "review", "weight": 3.0},
                            {"axis": "check_completed", "weight": 2.0}],
                  "axes": {"review": {"cases": 2, "eligible": 75, "share": 0.75},
                           "check_completed": {"cases": 1, "eligible": 25, "share": 0.25}}}
    # review: (1*1 + 3*0) / 4 = 0.25; checks: 1.0; mix 0.75 * 0.25 + 0.25 * 1.0
    assert harness.natural_rate(definition, [1, 0, 1]) == pytest.approx(0.4375)
    assert harness.natural_rate(definition, [1, 0, 1], [0, 0, 2]) == pytest.approx(1.0)
    # a resample without the check axis: the review axis alone, (1 + 1 + 0) / (1 + 1 + 3)
    assert harness.natural_rate(definition, [1, 0, 1], [0, 0, 1]) == pytest.approx(0.4)
    plain = {"cases": [{"case_id": "x", "weight": 2.0}, {"case_id": "y", "weight": 2.0}]}
    assert harness.natural_rate(plain, [1, 0]) == pytest.approx(0.5)  # review decisions only


def test_exact_bounds_count_clusters_and_stay_informative_without_failures() -> None:
    clusters = [f"g{i // 2}" for i in range(200)]  # 100 clusters of two
    bound = harness.cluster_bound([1] * 200, clusters)
    assert bound["clusters"] == 100 and bound["clusters_with_failure"] == 0
    assert bound["failure_share_upper_one_sided"] == round(1 - 0.05 ** (1 / 100), 4)
    assert bound["pass_share_interval"][1] == 1.0
    two_in_one = harness.cluster_bound([0, 0] + [1] * 198, clusters)
    assert two_in_one["clusters_with_failure"] == 1  # two linked failures count once
    assert two_in_one["failure_share_upper_one_sided"] > bound["failure_share_upper_one_sided"]


def test_when_every_case_passes_no_bootstrap_interval_is_reported_and_a_tie_is_named() -> None:
    cases = [{"case_id": f"c{i}", "cluster": f"g{i}", "weight": 1.0} for i in range(4)]
    arms = {name: {"cases": {f"c{i}/primary": {"acceptable": True, "complete_pass": True}
                             for i in range(4)}} for name in ("a", "b")}
    stats = harness.statistics({"cases": cases}, arms)
    first = stats["arms"]["a"]["complete_pass"]
    assert first["cluster_bootstrap"] is None and first["natural_cluster_bootstrap"] is None
    assert first["degenerate"] == "every case passed"
    assert first["exact_clusters"]["failure_share_upper_one_sided"] == round(
        1 - 0.05 ** (1 / 4), 4)
    paired = stats["paired"]["a vs b"]["complete_pass"]
    assert paired["difference"] == 0 and paired["difference_cluster_bootstrap"] is None
    assert "not evidence that they are equivalent" in paired["tie"]
    alone = harness.statistics({"cases": cases}, {"a": arms["a"]}, paired=False)
    assert alone["paired"] == {} and "every arm" in alone["paired_note"]


def test_one_arm_is_scored_alone_under_its_own_coverage_gate(bench: Path, tmp_path: Path) -> None:
    run(bench, "a", tmp_path)
    with pytest.raises(harness.CoverageError):
        harness.evaluate(bench)  # arm b has no records
    alone = harness.evaluate(bench, arms=["a"])
    assert set(alone["arms"]) == {"a"} and alone["statistics"]["paired"] == {}
    with pytest.raises(harness.CoverageError):
        harness.evaluate(bench, arms=["b"])
    with pytest.raises(harness.ShapeError):
        harness.evaluate(bench, arms=["c"])


def test_the_headline_and_its_components_per_case(bench: Path, tmp_path: Path) -> None:
    run(bench, "a", tmp_path)
    results = harness.evaluate(bench, arms=["a"])
    summary = results["arms"]["a"]["summary"]
    assert summary["components"] == {name: {"numerator": 3, "denominator": 3, "value": 1.0}
                                     for name in harness.COMPONENTS}
    assert summary["standard_action"]["numerator"] == 3
    rows = summary["per_referee_row"]
    assert list(rows) == list(harness.REFEREE_ROWS)
    assert rows["§6.6(c)"]["complete_pass"]["denominator"] == 1
    assert rows["§6.6(b), one family"]["acceptable"]["denominator"] == 1
    assert rows["§6.6(a)"]["acceptable"]["denominator"] == 1
    assert rows["§5.3(c)"]["n"] == 0 and "expires" in rows["§5.3(c)"]["not_evaluated"]
    assert rows["§5.3(a)"] == {"n": 0, "not_evaluated": "no case in the cohort"}
    latent = results["arms"]["a"]["latent_diagnostic"]
    assert latent["top_hypothesis_tied"] == 3  # the stand-in's two hypotheses are both low
    assert latent["top_hypothesis_names_latent"] == {"numerator": 0, "denominator": 3,
                                                    "value": 0.0}


def test_a_claim_error_fails_the_memo_but_not_its_disposition() -> None:
    from llm import referee

    packet = checked_packet()
    view = referee.view(packet)
    text = json.dumps({
        "claims": [{"field": "context.bin_ip_country_mismatch", "value": "0",
                    "statement": "Card and IP countries.", "derived": None}],
        "hypotheses": [{"explanation": "stolen_card", "likelihood": "high", "reasoning": "r"},
                       {"explanation": "traveller", "likelihood": "low", "reasoning": "r"}],
        "disposition": "needs_check", "citations": ["FP-2 §6.6(b)", "R03"],
        "next_check": "id_check", "memo": "Short memo."})
    row = harness.score_record({"outcome": "response", "text": text}, packet, view)
    assert row["acceptable"] and row["action"] == "hold" and row["standard_action"]
    assert row["components"]["no_claim_error"] is False and row["complete_pass"] is False
    assert row["top_hypothesis"] == "stolen_card" and not row["top_hypothesis_tied"]


def test_the_shape_holds_each_axis_to_its_configured_count() -> None:
    def definition(phase: str, axes: dict[str, int], recorded: dict | None = None) -> dict:
        cases = [{"case_id": f"{axis}-{i}", "axis": axis, "seed": 1041, "family": "baseline",
                  "account": f"1:{axis}{i}", "episode": None, "cluster": f"g{axis}{i}",
                  "weight": 1.0, "packet_sha256": "x"}
                 for axis, n in axes.items() for i in range(n)]
        out = {"phase": phase, "cases": cases, "arms": {"a": {}}, "probes": {}}
        if recorded is not None:
            out["axes"] = recorded
        return out

    sizes = {**SIZES, "development_check_cases": 2,
             "development_combined": {"review": 1, "check_completed": 1}}
    for axes in ({"review": 3}, {"check_completed": 2}, {"review": 1, "check_completed": 1}):
        harness.check_shape(definition("development", axes), sizes)
    with pytest.raises(harness.ShapeError, match="by decision point"):
        harness.check_shape(definition("development", {"check_completed": 3}), sizes)
    shares = {"review": {"cases": 1, "eligible": 3, "share": 0.75},
              "check_completed": {"cases": 1, "eligible": 1, "share": 0.25}}
    harness.check_shape(definition("development", {"review": 1, "check_completed": 1},
                                   shares), sizes)
    with pytest.raises(harness.ShapeError, match="add up to 1"):
        harness.check_shape(definition("development", {"review": 1, "check_completed": 1},
                                       {**shares, "review": {**shares["review"],
                                                             "share": 0.5}}), sizes)
    assert harness.final_axes({**SIZES, "final_cases": 5,
                               "final_axes": {"review": 3, "check_completed": 2}}) == {
        "review": 3, "check_completed": 2}
    with pytest.raises(harness.ShapeError, match="add up"):
        harness.final_axes({**SIZES, "final_axes": {"review": 1}})
