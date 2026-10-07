"""The pre-registered protocol and the freeze gate for final seeds."""

from __future__ import annotations

import copy
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


def test_final_seeds_must_not_overlap_development(raw: dict) -> None:
    broken = copy.deepcopy(raw)
    broken["seeds"]["final"][0] = 1041
    with pytest.raises(proto.ProtocolError, match="overlap"):
        proto.parse_protocol(broken)


def _frozen_copy(tmp_path: Path, raw: dict, *, complete: bool) -> Path:
    data = copy.deepcopy(raw)
    if complete:
        data = yaml.safe_load(yaml.safe_dump(data).replace(proto.PLACEHOLDER, "set"))
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "protocol.yaml").write_text(yaml.safe_dump(data))
    for name in data["freeze"]["files"]:
        if name == "experiments/protocol.yaml":
            continue
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = REPO / name
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


def test_committed_repository_is_not_frozen_yet() -> None:
    final = proto.load_protocol().final_seeds[0]
    with pytest.raises(proto.FreezeError):
        proto.check_seed(final)
