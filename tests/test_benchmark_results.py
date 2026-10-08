"""The committed results.json files: rescoring the committed records reproduces each one
byte for byte, offline, and the named endpoints are copies of the harness's values."""
import json
import socket
from pathlib import Path

import pytest

from llm.eval import harness, results

COMMITTED = sorted(path.parent for path in harness.BENCHMARKS.glob("*/results.json"))


def _committed(directory: Path) -> str:
    return (directory / "results.json").read_bytes().decode("utf-8")


def _leaves(value, path=()):
    """Every endpoint entry: a mapping with a ``source`` pointer."""
    if isinstance(value, dict):
        if "source" in value and "value" in value:
            yield path, value
            return
        for key, item in value.items():
            yield from _leaves(item, (*path, key))


def test_every_benchmark_with_records_has_results() -> None:
    with_records = sorted({path.parent.parent
                           for path in harness.BENCHMARKS.glob("*/cache/*.json")})
    assert with_records and with_records == COMMITTED


@pytest.mark.parametrize("directory", COMMITTED, ids=lambda path: path.name)
def test_rescoring_reproduces_the_committed_results(directory: Path) -> None:
    committed = (directory / "results.json").read_bytes()
    amendment = json.loads(committed).get("scoring_amendment") or {}
    rebuilt = results.dumps(results.build(directory, amend=amendment.get("reason")))
    assert rebuilt.encode("utf-8") == committed


@pytest.mark.parametrize("directory", COMMITTED, ids=lambda path: path.name)
def test_the_results_hold_no_path_host_or_run_time(directory: Path) -> None:
    text = _committed(directory)
    for trace in ("/Users/", "/home/", "/private/", "/tmp/", str(Path.home()),
                  str(harness.REPO), socket.gethostname()):
        assert trace not in text


@pytest.mark.parametrize("directory", COMMITTED, ids=lambda path: path.name)
def test_each_endpoint_is_the_value_its_pointer_names(directory: Path) -> None:
    value = json.loads(_committed(directory))
    leaves = list(_leaves(value["endpoints"]))
    assert leaves
    for path, entry in leaves:
        assert results.resolve(value, entry["source"]) == entry["value"], path
    primary = value["endpoints"]["primary"]
    assert primary["reported_as"] == "natural_mix"
    assert set(primary["arms"]) == set(value["arms"])
    paired = value["endpoints"]["secondary"]["paired_comparison"]
    definition = harness.load_benchmark(directory)
    if set(value["arms"]) == set(definition["arms"]) and len(value["arms"]) > 1:
        assert paired["source"] == "/statistics/paired" and paired["value"]
    else:
        assert paired == {"not_made": results.NOT_PAIRED}
    for arm in value["arms"]:
        bound = value["endpoints"]["diagnostics"][arm]["failing_cluster_bound"]
        assert set(bound) == {"clusters", "clusters_with_failure",
                              "failure_share_upper_one_sided", "level"}


def test_the_case_memos_carry_their_case_file_memo_and_score() -> None:
    directory = harness.BENCHMARKS / "2026-10-cases"
    value = json.loads(_committed(directory))
    definition = harness.load_benchmark(directory)
    assert [m["case_id"] for m in value["case_memos"]] == [
        case["case_id"] for case in definition["cases"]]
    for entry, case in zip(value["case_memos"], definition["cases"], strict=True):
        assert f"{entry['file']}:{entry['slot']}" == case["stratum"]
        for arm, scored in entry["arms"].items():
            assert scored["score"] == value["arms"][arm]["cases"][f"{case['case_id']}/primary"]
            if scored["score"]["format_valid"]:
                assert scored["problems"] == [] and scored["memo"]["disposition"] == (
                    scored["score"]["disposition"])
            else:
                assert scored["memo"] is None and scored["problems"]


def test_a_pointer_escapes_slashes_and_tildes() -> None:
    value = {"a/b": {"c~d": 1}}
    pointer = results._pointer("a/b", "c~d")
    assert pointer == "/a~1b/c~0d" and results.resolve(value, pointer) == 1
