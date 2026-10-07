"""The pre-registered protocol and the freeze gate for final seeds."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from core import protocol as proto

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load((REPO / "experiments" / "protocol.yaml").read_text())


def test_committed_protocol_is_valid(raw: dict) -> None:
    p = proto.parse_protocol(raw)
    assert p.windows["test"].start == pd.Timestamp("2025-06-01")
    assert p.observed_until == pd.Timestamp("2025-12-29 23:59:59")
    assert 416 in p.development_seeds and len(p.final_seeds) == 10


def test_gaps_cover_the_label_horizon(raw: dict) -> None:
    p = proto.parse_protocol(raw)
    for name in proto.GAPS:
        assert p.windows[name].days >= p.label_horizon_days
    # freeze instants: fitted window end + horizon <= freeze <= next window start
    assert p.freezes["classifier"] >= pd.Timestamp("2024-08-01") + pd.Timedelta(days=60)
    assert p.freezes["policy"] <= p.windows["test"].start


def test_window_assignment(raw: dict) -> None:
    p = proto.parse_protocol(raw)
    stamps = pd.Series(pd.to_datetime([
        "2024-03-31 23:59:59", "2024-04-01 00:00:00", "2025-08-31 12:00:00",
        "2026-01-01 00:00:00",
    ]))
    assert p.window_of(stamps).tolist() == ["warm_up", "fit", "test", pd.NA]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("windows", "gap_1", "end"), "2024-09-01", "not contiguous"),
        (("label_horizon_days",), 90, "shorter than the label horizon"),
        (("freezes", "policy"), "2025-05-15", "before validation labels can mature"),
        (("freezes", "calibrator"), "2025-01-15", "after validation starts"),
        (("windows", "follow_up", "end"), "2025-10-01", "follow-up is shorter"),
        (("families", "fraud_mix_shift", "starts"), "2025-05-01", "must start at the test"),
    ],
)
def test_broken_protocols_are_rejected(raw: dict, path: tuple, value, message: str) -> None:
    broken = copy.deepcopy(raw)
    node = broken
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(proto.ProtocolError, match=message):
        proto.parse_protocol(broken)


def test_the_fulfilment_lag_families_are_sensitivities_from_the_test_window(raw: dict) -> None:
    p = proto.parse_protocol(raw)
    assert p.base_only_families == ("lag_half", "lag_double")
    assert all(p.family_starts[f] == p.windows["test"].start for f in p.base_only_families)
    assert raw["sensitivity"]["fulfilment_lag"]["families"] == {"lag_half": 0.5,
                                                                 "lag_double": 2.0}
    open_entry = copy.deepcopy(raw)
    open_entry["sensitivity"]["fulfilment_lag"] = proto.PLACEHOLDER
    assert proto.parse_protocol(open_entry).base_only_families == ()
    for families, message in (({"lag_triple": 3.0}, "is not a family"),
                              ({"baseline": 0.5}, "must start at the test window"),
                              ({"lag_half": 0}, "positive number"),
                              ({}, "must map each family")):
        broken = copy.deepcopy(raw)
        broken["sensitivity"]["fulfilment_lag"]["families"] = families
        with pytest.raises(proto.ProtocolError, match=message):
            proto.parse_protocol(broken)


def test_the_protocol_families_are_the_generators(raw: dict) -> None:
    from core import config
    from simulator import generate

    p = proto.parse_protocol(raw)
    assert set(p.family_starts) == set(generate.FAMILIES)
    assert set(p.base_only_families) == set(generate.LAG_FAMILIES)
    world = config.load("world")["families"]
    for name, factor in raw["sensitivity"]["fulfilment_lag"]["families"].items():
        assert world[name]["fulfilment_lag_factor"] == factor


def test_the_recommendation_rule_and_its_code_are_frozen(raw: dict) -> None:
    assert "core/recommendation.py" in raw["freeze"]["files"]
    assert proto.PLACEHOLDER not in yaml.safe_dump(raw["reporting"])


def test_the_protocol_states_the_configured_capacity(raw: dict) -> None:
    from core import config

    capacity = config.load("policy")["capacity"]
    if not capacity.get("levels"):
        pytest.skip("the capacity levels are not configured yet")
    stated = raw["capacity"]["levels"]
    assert set(stated) == set(capacity["levels"])
    for name, entry in capacity["levels"].items():  # one analyst, the stated minutes, per shift
        assert set(entry["review_minutes_per_shift"].values()) == {stated[name]}
        assert set(entry["analysts_per_shift"].values()) == {1}
    redesigned = capacity["redesigned"]
    assert raw["capacity"]["redesigned_layout"].startswith(f"{redesigned['layout']}, ")
    assert set(redesigned["review_minutes_per_shift"].values()) == {stated["base"]}


def test_final_seeds_must_not_overlap_development(raw: dict) -> None:
    broken = copy.deepcopy(raw)
    broken["seeds"]["final"][0] = 1041
    with pytest.raises(proto.ProtocolError, match="overlap"):
        proto.parse_protocol(broken)


def _frozen_copy(tmp_path: Path, raw: dict, *, complete: bool) -> Path:
    data = copy.deepcopy(raw)
    if complete:
        data = yaml.safe_load(yaml.safe_dump(data).replace(proto.PLACEHOLDER, "set"))
    else:
        data["reporting"]["recommendation_rule"] = proto.PLACEHOLDER
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "protocol.yaml").write_text(yaml.safe_dump(data))
    for name in data["freeze"]["files"]:
        if name == "experiments/protocol.yaml":
            continue
        target = tmp_path / name
        source = REPO / name
        if name.endswith("/"):  # a directory entry: a small stand-in tree
            (target / "sub").mkdir(parents=True, exist_ok=True)
            (target / "module.py").write_text(f"# stand-in for {name}\n")
            (target / "sub" / "deeper.py").write_text("# nested\n")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy(source, target)
        else:
            target.write_text(f"stand-in for {name}\n")
    return tmp_path


def test_final_seed_refused_while_placeholders_remain(tmp_path: Path, raw: dict) -> None:
    root = _frozen_copy(tmp_path, raw, complete=False)
    with pytest.raises(proto.FreezeError, match="not complete"):
        proto.check_seed(raw["seeds"]["final"][0], root)
    proto.check_seed(416, root)  # development seeds are always open


def test_final_seed_refused_without_marker(tmp_path: Path, raw: dict) -> None:
    root = _frozen_copy(tmp_path, raw, complete=True)
    with pytest.raises(proto.FreezeError, match="no freeze marker"):
        proto.check_seed(raw["seeds"]["final"][0], root)


def _commit(root: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c",
                        "user.email=test@example.com", "-c", "commit.gpgsign=false", *args],
                       check=True, capture_output=True)

    if not (root / ".git").exists():
        git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "freeze")


def test_final_seed_allowed_after_the_freeze_commit_and_refused_after_a_change(
    tmp_path: Path, raw: dict
) -> None:
    root = _frozen_copy(tmp_path, raw, complete=True)
    proto.write_freeze_marker(root)
    _commit(root)
    proto.check_seed(raw["seeds"]["final"][0], root)
    (root / "config" / "world.yaml").write_text("changed: true\n")
    with pytest.raises(proto.FreezeError, match="changed since the freeze"):
        proto.check_seed(raw["seeds"]["final"][0], root)


def test_final_seed_refused_before_the_freeze_is_committed(tmp_path: Path, raw: dict) -> None:
    root = _frozen_copy(tmp_path, raw, complete=True)
    _commit(root)  # the inputs are committed, the marker is not
    proto.write_freeze_marker(root)
    with pytest.raises(proto.FreezeError, match="freeze not committed"):
        proto.check_seed(raw["seeds"]["final"][0], root)


def test_final_seed_refused_outside_a_repository(tmp_path: Path, raw: dict) -> None:
    root = _frozen_copy(tmp_path, raw, complete=True)
    proto.write_freeze_marker(root)
    with pytest.raises(proto.FreezeError, match="freeze commit"):
        proto.check_seed(raw["seeds"]["final"][0], root)


def test_a_directory_entry_freezes_every_file_under_it(tmp_path: Path, raw: dict) -> None:
    root = _frozen_copy(tmp_path, raw, complete=True)
    (root / "rules" / "__pycache__").mkdir()
    (root / "rules" / "__pycache__" / "module.cpython-314.pyc").write_bytes(b"cache")
    (root / ".gitignore").write_text("*.log\n")
    proto.write_freeze_marker(root)
    marker = json.loads((root / "experiments" / "FREEZE.json").read_text())
    assert {"rules/module.py", "rules/sub/deeper.py"} <= set(marker["files"])
    assert not any("__pycache__" in name for name in marker["files"])
    _commit(root)
    final = raw["seeds"]["final"][0]
    proto.check_seed(final, root)
    (root / "rules" / "ignored.log").write_text("ignored by git\n")  # not frozen
    proto.check_seed(final, root)
    (root / "rules" / "sub" / "added.py").write_text("x = 1\n")
    with pytest.raises(proto.FreezeError, match="does not cover.*rules/sub/added.py"):
        proto.check_seed(final, root)
    _commit(root)  # committing the addition does not make it frozen
    with pytest.raises(proto.FreezeError, match="does not cover"):
        proto.check_seed(final, root)
    (root / "rules" / "sub" / "added.py").unlink()
    (root / "rules" / "sub" / "deeper.py").unlink()  # a frozen file removed
    _commit(root)
    with pytest.raises(proto.FreezeError, match="changed since the freeze.*deeper.py"):
        proto.check_seed(final, root)
    shutil.rmtree(root / "model")
    with pytest.raises(proto.FreezeError, match="model/ does not exist"):
        proto.check_seed(final, root)


def test_a_fix_after_the_freeze_is_logged_and_re_frozen(tmp_path: Path, raw: dict) -> None:
    root = _frozen_copy(tmp_path, raw, complete=True)
    proto.write_freeze_marker(root)
    _commit(root)
    final = raw["seeds"]["final"][0]
    with pytest.raises(proto.FreezeError, match="nothing to log"):
        proto.log_fix("a bug", "none", root)
    (root / "queue_sim" / "module.py").write_text("# fixed\n")
    with pytest.raises(proto.FreezeError, match="changed since the freeze"):
        proto.check_seed(final, root)
    with pytest.raises(proto.FreezeError, match="before/after"):
        proto.log_fix("a bug", " ", root)
    assert proto.log_fix("the queue skipped P0", "net +$3 on seed 1, 0 elsewhere", root) == \
        ["queue_sim/module.py"]
    _commit(root)
    proto.check_seed(final, root)
    marker = json.loads((root / "experiments" / "FREEZE.json").read_text())
    assert marker["fixes"] == [{"files": ["queue_sim/module.py"],
                                "what": "the queue skipped P0",
                                "effect": "net +$3 on seed 1, 0 elsewhere"}]


def test_the_frozen_list_is_what_decides_results_and_nothing_later_stages_add(
    raw: dict
) -> None:
    entries = raw["freeze"]["files"]
    files = proto.frozen_files(REPO, entries)
    for needed in ("experiments/protocol.yaml", "config/world.yaml", "config/policy.yaml",
                   "policy/fraud-policy.md", "simulator/generate.py", "core/world.py",
                   "core/ledger.py", "core/actions.py", "core/evidence.py", "core/asof.py",
                   "core/recommendation.py", "core/stats.py", "core/results.py",
                   "core/protocol.py", "queue_sim/replay.py", "rules/tuning.py",
                   "model/train.py", "llm/referee.py", "pipeline.py", "requirements.lock"):
        assert needed in files
    later = ("docs/", "report/", "cases/", "llm/eval/", "results/", "reports/", "README")
    assert not [name for name in files if name.startswith(later)]
    assert not [entry for entry in entries if any(place.startswith(entry) for place in later)]


def test_the_protocol_states_the_tuning_rule_the_code_runs(raw: dict) -> None:
    from core import config

    stage = pytest.importorskip("queue_sim.stage")
    tuning = config.load("policy")["tuning"]
    assert raw["tuning"]["shortlist_best_k"] == tuning["shortlist_best_k"]
    assert raw["tuning"]["history"] in stage.TUNING_HISTORIES
    assert raw["tuning"]["history"] == "shortlist"
    assert set(tuning["friction_guardrail"].values()) == {None}  # no cap inside tuning
    rule = raw["tuning"]["rule"]
    assert {"world", "grid", "screen", "shortlist", "choice", "objective", "feasible", "ties",
            "boundary", "infeasible"} <= set(rule)


def test_the_case_rule_is_checked_against_the_replay(raw: dict) -> None:
    from core import config

    stage = pytest.importorskip("queue_sim.stage")
    p = proto.parse_protocol(raw)
    staffing = stage.staffing(config.load("policy"))
    levels, layouts = {s.level for s in staffing}, {s.layout for s in staffing}
    proto.check_case_replay(p, stage.VARIANTS, levels, layouts)
    assert raw["cases"]["world"] == {"seed": p.canonical_seed, "family": "baseline"}
    for key, value, message in (("verification", "verification_weak", "main variant"),
                                ("capacity_level", "medium", "configured level"),
                                ("layout", "night", "configured layout"),
                                ("policy", "rules", "protocol policy")):
        broken = copy.deepcopy(raw)
        broken["cases"]["replay"][key] = value
        with pytest.raises(proto.ProtocolError, match=message):
            proto.check_case_replay(proto.parse_protocol(broken), stage.VARIANTS, levels,
                                    layouts)


@pytest.mark.parametrize(("change", "message"), [
    (lambda c: c["selection"]["slot_order"].pop(), "slot_order must list every slot"),
    (lambda c: c["slots"]["ring"].update(tiers=["any_alert"]), "tiers from"),
    (lambda c: c["files"]["ring"].update(slots=["rings"]), "must name slots"),
    (lambda c: c["files"].pop("ring"), "use every slot exactly once"),
    (lambda c: c["files"]["traveller"].update(length="long"), "full or short"),
    (lambda c: c.pop("facts"), "lacks"),
    (lambda c: c["replay"].pop("window"), "exactly"),
    (lambda c: c.update(extra=1), "unknown keys"),
])
def test_a_malformed_case_rule_is_rejected(raw: dict, change, message: str) -> None:
    broken = copy.deepcopy(raw)
    change(broken["cases"])
    with pytest.raises(proto.ProtocolError, match=message):
        proto.parse_protocol(broken)


def test_the_committed_repository_is_frozen_once_its_marker_exists() -> None:
    final = proto.load_protocol().final_seeds[0]
    assert proto.load_protocol().placeholders() == []
    if not (REPO / "experiments" / "FREEZE.json").exists():
        with pytest.raises(proto.FreezeError, match="no freeze marker"):
            proto.check_seed(final)
    else:  # the frozen files match the marker and are committed
        proto.check_seed(final)
