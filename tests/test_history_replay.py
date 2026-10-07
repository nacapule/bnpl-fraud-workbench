"""Offline replay of the archived August 2026 memo study (llm/eval/history.py)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from llm.eval import history
from llm.eval.history import (
    BENCH_DIR,
    Archive,
    CoverageError,
    historical_problems,
    replay_arm,
    score_arm,
    strict_problems,
)

CORRECTED = BENCH_DIR / "corrected"
LUNA = "memo_v2__gpt-5.6-luna"
TERRA = "memo_v2__gpt-5.6-terra"
V1, V2 = "memo_v1__claude-sonnet-5", "memo_v2__claude-sonnet-5"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, str]:
    out = tmp_path_factory.mktemp("corrected")
    paths = history.write(out)
    return {name: path.read_text(encoding="utf-8") for name, path in paths.items()}


@pytest.fixture(scope="module")
def stats(built) -> dict:
    return json.loads(built["statistics.json"])


@pytest.fixture(scope="module")
def archive() -> Archive:
    return Archive()


def _arm(archive: Archive, arm_id: str):
    return next(arm for arm in archive.arms if arm.id == arm_id)


def _counts(item: dict) -> tuple[int, int]:
    return item["numerator"], item["denominator"]


# Reproduction ----------------------------------------------------------------------
def test_the_replay_equals_the_committed_corrected_files(built):
    for name in history.GENERATED_FILES:
        assert built[name] == (CORRECTED / name).read_text(encoding="utf-8"), name


def test_every_archived_result_file_is_reproduced(archive, stats):
    for arm in archive.arms:
        expected = json.loads((archive.original / arm.result_file).read_text())
        assert history.archived_result(archive, replay_arm(archive, arm)) == expected, arm.id
    assert all(stats["replay"]["archived_results_reproduced"].values())


def test_the_archived_headline_numbers(archive):
    luna = history.archived_result(archive, replay_arm(archive, _arm(archive, LUNA)))
    assert (luna["n_cases"], luna["schema_failures"], luna["action_accuracy"]) == (199, 1, 0.598)
    v2 = history.archived_result(archive, replay_arm(archive, _arm(archive, V2)))
    assert (v2["action_accuracy"], v2["unsupported_claim_rate_memo"]) == (0.735, 0.045)
    assert (v2["consistency_action_agreement"], v2["consistency_n"]) == (0.84, 50)


def test_prompts_and_cache_names_follow_the_historical_protocol(archive):
    case = archive.cases[0]
    packet = archive.packets[case["alert_id"]]
    prompt = history.render_prompt(archive.templates["memo_v2"], packet)
    record = archive.record(history.cache_file_name("claude-sonnet-5", prompt))
    assert record is not None
    assert record["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()


# The attempt chain -------------------------------------------------------------------
def test_luna_alert_6188_has_an_invalid_first_attempt_and_a_valid_stored_retry(archive):
    replay = replay_arm(archive, _arm(archive, LUNA))
    [chain] = [chain for chain in replay.primary if chain.alert_id == 6188]
    first, retry = chain.attempts
    assert first.present and not first.valid
    assert first.problems[0].startswith("Unterminated string")
    assert retry.present and retry.valid
    assert chain.first_memo is None
    assert chain.memo["recommended_action"] == "decline_block"
    assert [c.alert_id for c in replay.primary if len(c.attempts) > 1] == [6188]


def test_luna_first_attempt_and_final_results_are_reported_separately(stats):
    luna = stats["arms"][LUNA]
    first, final = luna["first_attempt"], luna["final"]
    assert _counts(first["valid_outputs"]) == (199, 200)
    assert _counts(first["action_correct_valid_only"]) == (119, 199)
    assert _counts(first["action_correct"]) == (119, 200)
    assert _counts(first["decline_recall"]) == (21, 57)
    assert _counts(final["action_correct"]) == (120, 200)
    assert _counts(final["decline_recall"]) == (22, 57)
    assert final["failures"] == [] and first["failures"] == [6188]
    assert luna["retries"] == {"cases": [6188], "stored": 1, "valid": 1}


def test_arms_without_retries_score_the_same_at_both_endpoints(stats):
    for arm_id in (V1, V2, TERRA):
        arm = stats["arms"][arm_id]
        assert arm["first_attempt"] == arm["final"], arm_id


def test_the_chains_file_records_every_attempt(built):
    chains = json.loads(built["attempt_chains.json"])["arms"]
    luna = [
        c for c in chains[LUNA]["chains"] if c["alert_id"] == 6188 and c["variant"] == "primary"
    ]
    assert [a["valid"] for a in luna[0]["attempts"]] == [False, True]
    assert luna[0]["final"]["recommended_action"] == "decline_block"
    assert len(chains[TERRA]["chains"]) == 86
    assert len(chains[V2]["chains"]) == 200 + 50 * 3


# Strict validation -----------------------------------------------------------------
def test_strict_validation_rejects_no_accepted_historical_output(stats):
    assert stats["replay"]["strict_validation"] == {"accepted_outputs": 979, "rejected": 0}


MALFORMED = [
    None,
    [],
    "text",
    {"recommended_action": []},
    {
        "signals_observed": [9],
        "hypotheses": "benign",
        "policy_citations": "R03",
        "recommended_action": ["clear"],
        "priority": {},
        "evidence_gaps": 99,
        "memo_markdown": None,
    },
    {
        "signals_observed": ["a"],
        "hypotheses": [{"pattern": [], "likelihood": {}, "reasoning": 3}],
        "policy_citations": [],
        "recommended_action": "clear",
        "priority": "P3",
        "evidence_gaps": [],
        "memo_markdown": "",
    },
]


@pytest.mark.parametrize("memo", MALFORMED)
def test_malformed_output_is_a_list_of_problems_not_an_exception(memo):
    assert strict_problems(memo)
    assert isinstance(historical_problems(memo), list)


def test_strict_validation_requires_what_the_historical_validator_skipped():
    memo = {
        "signals_observed": ["amount 10"],
        "hypotheses": [],
        "policy_citations": ["FP-1 §999"],
        "recommended_action": "decline_block",
        "priority": "P0",
        "evidence_gaps": None,
        "memo_markdown": "",
    }
    assert historical_problems(memo) == []
    problems = strict_problems(memo)
    assert "hypotheses must be a non-empty list" in problems
    assert "evidence_gaps must be a list of strings" in problems
    assert "memo_markdown must be a non-empty string" in problems


def _copy_archive(tmp_path: Path) -> Path:
    bench = tmp_path / "2026-08-dev"
    shutil.copytree(BENCH_DIR / "original", bench / "original")
    shutil.copy(BENCH_DIR / "MANIFEST.json", bench / "MANIFEST.json")
    return bench


def _replace_response(archive: Archive, arm_id: str, alert_id: int, memo: object) -> None:
    arm = _arm(archive, arm_id)
    prompt = history.render_prompt(archive.templates[arm.prompt_version], archive.packets[alert_id])
    path = archive.cache_dir / history.cache_file_name(arm.model, prompt)
    record = json.loads(path.read_text())
    record["response"]["text"] = json.dumps(memo)
    path.write_text(json.dumps(record))


def test_a_malformed_historical_output_counts_as_a_failure(tmp_path):
    bench = _copy_archive(tmp_path)
    archive = Archive(bench)
    first, second = archive.cases[0]["alert_id"], archive.cases[1]["alert_id"]
    # Unhashable enumerated values: the historical validator used to raise TypeError.
    _replace_response(archive, TERRA, first, MALFORMED[4])
    # Passes the historical validator, fails the strict one (a number among the signals).
    valid = json.loads(json.dumps(MALFORMED[5]))
    valid["hypotheses"] = [{"pattern": "benign", "likelihood": "high", "reasoning": "r"}]
    valid["memo_markdown"] = "memo"
    valid["signals_observed"] = ["amount 10", 9]
    assert historical_problems(valid) == []
    _replace_response(archive, TERRA, second, valid)

    archive = Archive(bench)
    replay = replay_arm(archive, _arm(archive, TERRA))
    scored = score_arm(archive, replay)["final"]
    by_alert = {row.alert_id: row for row in scored}
    assert not by_alert[first].accepted and not by_alert[first].strict_rejected
    assert not by_alert[second].accepted and by_alert[second].strict_rejected
    assert history.archived_result(archive, replay)["schema_failures"] == 2
    [chain] = [c for c in replay.primary if c.alert_id == first]
    assert not chain.attempts[0].valid and not chain.attempts[1].present


def test_a_stored_record_without_text_is_an_invalid_attempt(tmp_path):
    bench = _copy_archive(tmp_path)
    archive = Archive(bench)
    terra = _arm(archive, TERRA)
    alert = archive.cases[0]["alert_id"]
    prompt = history.render_prompt(archive.templates["memo_v2"], archive.packets[alert])
    path = archive.cache_dir / history.cache_file_name(terra.model, prompt)
    record = json.loads(path.read_text())
    record["response"]["text"] = None
    path.write_text(json.dumps(record))
    [chain] = [c for c in replay_arm(Archive(bench), terra).primary if c.alert_id == alert]
    assert chain.attempts[0].problems == ("stored response has no text",)
    assert chain.memo is None


def test_the_summary_renders_an_arm_without_valid_outputs(stats):
    broken = json.loads(json.dumps(stats))
    broken["arms"][TERRA]["final"]["latency_p50_ms"] = None
    summary = history.render_summary(broken)
    assert f"| {TERRA} | 14/15 (93.3%)" in summary and "| \u2013 |" in summary


def test_an_arm_without_stored_responses_fails(tmp_path):
    bench = _copy_archive(tmp_path)
    archive = Archive(bench)
    terra = _arm(archive, TERRA)
    for case in archive.cases[: terra.case_limit]:
        prompt = history.render_prompt(
            archive.templates["memo_v2"], archive.packets[case["alert_id"]]
        )
        (archive.cache_dir / history.cache_file_name(terra.model, prompt)).unlink()
    with pytest.raises(CoverageError, match="86 of 86"):
        replay_arm(Archive(bench), terra)
    with pytest.raises(CoverageError):
        history.build(bench, CORRECTED / history.WORLD_CHECKS)


def test_one_missing_case_also_fails(tmp_path):
    bench = _copy_archive(tmp_path)
    archive = Archive(bench)
    luna = _arm(archive, LUNA)
    alert = archive.cases[-1]["alert_id"]
    prompt = history.render_prompt(archive.templates["memo_v2"], archive.packets[alert])
    (archive.cache_dir / history.cache_file_name(luna.model, prompt)).unlink()
    with pytest.raises(CoverageError, match=f"1 of 200 .* alert {alert}"):
        replay_arm(Archive(bench), luna)


# Consistency and paired comparisons ------------------------------------------------
def test_consistency_uses_the_complete_probe_sets(stats):
    arms = stats["consistency"]["arms"]
    assert _counts(arms[V2]["probe_agreement"]) == (42, 50)
    assert _counts(arms[V2]["agreement_with_original"]) == (40, 50)
    assert arms[V2]["unanimous_probes_differing_from_original"] == [4069, 7909]
    assert _counts(arms[V1]["complete_probe_sets"]) == (47, 50)
    assert _counts(arms[V1]["probe_agreement"]) == (41, 47)
    assert arms[V2]["probe_agreement"]["wilson_95"] == pytest.approx([0.715, 0.917], abs=5e-4)
    common = stats["consistency"]["common_complete_cases"]
    assert common["cases"] == 47
    assert _counts(common["arms"][V1]["probe_agreement"]) == (41, 47)
    assert _counts(common["arms"][V2]["probe_agreement"]) == (39, 47)


def test_luna_has_no_stored_probes(stats):
    luna = stats["consistency"]["arms"][LUNA]
    assert _counts(luna["complete_probe_sets"]) == (0, 50)
    assert luna["probe_agreement"]["value"] is None


@pytest.mark.parametrize(
    ("name", "cases", "only_first", "only_second", "first", "second"),
    [
        ("prompt_v1_vs_v2_action", 200, 8, 33, 122, 147),
        ("prompt_v1_vs_v2_unmatched_tokens", 200, 1, 60, 132, 191),
        ("prompt_v1_vs_v2_unmatched_tokens_archived_check", 200, 1, 59, 133, 191),
        ("sonnet_vs_luna_action", 200, 32, 5, 147, 120),
        ("sonnet_vs_terra_action", 86, 3, 4, 65, 66),
    ],
)
def test_paired_comparisons_on_matched_cases(
    stats, name, cases, only_first, only_second, first, second
):
    item = stats["paired"][name]
    assert (item["cases"], item["only_first"], item["only_second"]) == (
        cases,
        only_first,
        only_second,
    )
    assert _counts(item["first_rate"]) == (first, cases)
    assert _counts(item["second_rate"]) == (second, cases)
    for side in ("first_rate", "second_rate"):
        low, high = item[side]["user_bootstrap_95"]
        assert low <= item[side]["value"] <= high
    assert item["difference_second_minus_first"]["user_clusters"] <= cases


def test_mcnemar_p_values(stats):
    assert stats["paired"]["prompt_v1_vs_v2_action"]["exact_mcnemar_p"] == pytest.approx(
        1.1222e-4, rel=1e-3
    )
    assert stats["paired"]["sonnet_vs_terra_action"]["exact_mcnemar_p"] == 1.0


def test_terra_is_scored_on_its_86_cases_only(stats):
    terra = stats["arms"][TERRA]
    assert terra["cases"] == 86
    assert _counts(terra["final"]["action_correct"]) == (66, 86)
    assert terra["primary_responses_beyond_case_limit"] == 8
    assert stats["case_set"]["terra_prefix"]["cases"] == 86
    assert _counts(stats["case_set"]["terra_prefix"]["before_cutoff"]) == (68, 86)


# Case set and limitations --------------------------------------------------------------
def test_the_case_set_is_described_as_drawn_from_all_dates(stats):
    case_set = stats["case_set"]
    assert case_set["description"] == "development set drawn from all dates"
    assert case_set["alert_dates"] == {"first": "2025-07-18", "last": "2026-06-21"}
    assert _counts(case_set["before_cutoff"]) == (163, 200)
    assert case_set["users"] == 190
    assert case_set["user_or_story_groups"] == 137
    assert _counts(case_set["labelled_fraud"]) == (140, 200)
    assert case_set["benign_min_tenure_days"] == 78


def test_limitations_from_the_archive_and_the_world(stats):
    limits = stats["limitations"]
    assert limits["negative_tenure_packets"]["count"] == 5
    assert limits["future_linkage_packets"]["count"] == 59
    assert limits["payments_after_alert"]["alert_ids"] == [3682, 5279, 2587]
    assert _counts(limits["category_ratio"]["equals_full_period_mean"]) == (200, 200)
    assert _counts(limits["r06_rationale_future_accounts"]) == (32, 58)
    ranges = limits["order_id_ranges_in_world"]
    assert len(ranges["pure_patterns"]) == 6
    assert _counts(ranges["selected_cases_labelled_by_id_range"]) == (118, 140)
    assert _counts(limits["order_id_ranges_in_sample"]["cases_labelled_by_id_range"]) == (118, 140)


def test_world_checks_are_tied_to_the_archived_world():
    world = json.loads((CORRECTED / history.WORLD_CHECKS).read_text())
    manifest = json.loads((BENCH_DIR / "MANIFEST.json").read_text())
    assert world["alerts_csv_sha256"] == manifest["world"]["alerts_csv_sha256"]


def test_world_checks_from_another_world_are_refused(tmp_path):
    world = json.loads((CORRECTED / history.WORLD_CHECKS).read_text())
    world["alerts_csv_sha256"] = "0" * 64
    other = tmp_path / "world_checks.json"
    other.write_text(json.dumps(world))
    with pytest.raises(ValueError, match="different world"):
        history.build(BENCH_DIR, other)


def _archived_world() -> Path | None:
    data = BENCH_DIR.parents[3] / "data"
    manifest = json.loads((BENCH_DIR / "MANIFEST.json").read_text())
    alerts = data / "alerts.csv"
    if alerts.exists():
        digest = hashlib.sha256(alerts.read_bytes()).hexdigest()
        if digest == manifest["world"]["alerts_csv_sha256"]:
            return data
    return None


@pytest.mark.skipif(_archived_world() is None, reason="needs the superseded world in data/")
def test_world_checks_reproduce_from_the_archived_world():
    fresh = history._dump(history.world_checks(_archived_world()))
    assert fresh == (CORRECTED / history.WORLD_CHECKS).read_text()


# The stage and the command ---------------------------------------------------------------
def test_check_reports_a_stale_committed_file(tmp_path, monkeypatch, built):
    bench = tmp_path / "2026-08-dev"
    shutil.copytree(CORRECTED, bench / "corrected")
    (bench / "corrected" / "summary.md").write_text("stale\n")
    monkeypatch.setattr(history, "build", lambda *args, **kwargs: built)
    differences = history.check(bench)
    assert len(differences) == 1 and differences[0].startswith("summary.md: differs")
    (bench / "corrected" / "summary.md").write_text(built["summary.md"])
    assert history.check(bench) == []


def test_the_command_fails_when_corrected_is_stale(monkeypatch, capsys):
    monkeypatch.setattr(history, "check", lambda *args, **kwargs: ["summary.md: differs"])
    assert history.main(["--check"]) == 1
    assert "out of date" in capsys.readouterr().out


def test_the_llm_stage_writes_the_files_and_returns_headline_metrics(tmp_path, monkeypatch, built):
    monkeypatch.setattr(history, "build", lambda *args, **kwargs: built)
    result = history.run(tmp_path)
    assert result.stage == "llm"
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(history.GENERATED_FILES)
    luna = result.metrics["llm.history.luna_v2.final.action_correct"]
    assert (luna.numerator, luna.denominator, luna.value) == (120, 200, 0.6)
    first = result.metrics["llm.history.luna_v2.first_attempt.action_correct"]
    assert (first.numerator, first.denominator) == (119, 200)
    probes = result.metrics["llm.history.sonnet_v2.probe_agreement"]
    assert (probes.numerator, probes.denominator) == (42, 50)
    assert result.metrics["llm.history.paired.sonnet_vs_terra_action.only_second"].value == 4
    assert history.replay(tmp_path).keys() == result.metrics.keys()
