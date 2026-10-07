"""The pre-registered evaluation protocol (``experiments/protocol.yaml``).

:func:`load_protocol` reads the file and checks its structure: contiguous
windows, every gap at least the label horizon, freeze dates after each fitted
window's labels mature and before the next phase starts, a follow-up of at
least 90 days, and final seeds disjoint from the development seeds.

:func:`require_freeze` is what the final driver calls before generating any
final-seed world: it refuses unless the freeze marker exists, every file it
lists still has its recorded SHA-256, and no ``TO_COMPLETE_AT_FREEZE``
placeholder remains in the protocol. :func:`check_seed` applies that rule to a
seed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = REPO / "experiments" / "protocol.yaml"
PLACEHOLDER = "TO_COMPLETE_AT_FREEZE"
WINDOW_ORDER = (
    "warm_up", "fit", "gap_1", "calibration", "gap_2", "validation", "embargo", "test",
    "follow_up",
)
GAPS = ("gap_1", "gap_2", "embargo")
MIN_FOLLOW_UP_DAYS = 90


class ProtocolError(ValueError):
    """The protocol file breaks one of its own rules."""


class FreezeError(RuntimeError):
    """A final-seed world was requested before a valid freeze."""


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def contains(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts < self.end


@dataclass(frozen=True)
class Protocol:
    raw: dict[str, Any]
    label_horizon_days: int
    windows: dict[str, Window]
    freezes: dict[str, pd.Timestamp]
    family_starts: dict[str, pd.Timestamp | None]
    development_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    canonical_seed: int
    order_start: pd.Timestamp
    order_end: pd.Timestamp
    observed_end: pd.Timestamp  # exclusive

    @property
    def observed_until(self) -> pd.Timestamp:
        """Last observable instant (inclusive), for core.world.adjudicate and manifests."""
        return self.observed_end - pd.Timedelta(seconds=1)

    def window_of(self, timestamps: pd.Series) -> pd.Series:
        """Window name for each timestamp (missing outside the horizon)."""
        names = pd.Series(pd.NA, index=timestamps.index, dtype="object")
        for window in self.windows.values():
            names[(timestamps >= window.start) & (timestamps < window.end)] = window.name
        return names

    def placeholders(self) -> list[str]:
        return sorted(_placeholders(self.raw))


def _ts(value: Any) -> pd.Timestamp:
    if isinstance(value, dt.date | dt.datetime | str):
        return pd.Timestamp(value)
    raise ProtocolError(f"expected a date, got {value!r}")


def _placeholders(node: Any, path: str = "") -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _placeholders(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _placeholders(value, f"{path}[{index}]")
    elif node == PLACEHOLDER:
        yield path


def parse_protocol(raw: dict[str, Any]) -> Protocol:
    """Build a :class:`Protocol` from the parsed YAML and check its rules."""
    try:
        windows = {
            name: Window(name, _ts(raw["windows"][name]["start"]), _ts(raw["windows"][name]["end"]))
            for name in WINDOW_ORDER
        }
        horizon = raw["world_horizon"]
        protocol = Protocol(
            raw=raw,
            label_horizon_days=int(raw["label_horizon_days"]),
            windows=windows,
            freezes={name: _ts(value) for name, value in raw["freezes"].items()},
            family_starts={
                name: None if spec["starts"] is None else _ts(spec["starts"])
                for name, spec in raw["families"].items()
            },
            development_seeds=tuple(int(s) for s in raw["seeds"]["development"]),
            final_seeds=tuple(int(s) for s in raw["seeds"]["final"]),
            canonical_seed=int(raw["seeds"]["canonical"]),
            order_start=_ts(horizon["order_start"]),
            order_end=_ts(horizon["order_end"]),
            observed_end=_ts(horizon["observed_end"]),
        )
    except KeyError as error:
        raise ProtocolError(f"protocol is missing {error}") from error
    if extra := set(raw["windows"]) - set(WINDOW_ORDER):
        raise ProtocolError(f"unknown windows {sorted(extra)}")
    _check(protocol)
    return protocol


def _check(p: Protocol) -> None:
    problems: list[str] = []
    horizon = pd.Timedelta(days=p.label_horizon_days)
    ordered = [p.windows[name] for name in WINDOW_ORDER]
    for window in ordered:
        if window.end <= window.start:
            problems.append(f"{window.name} ends before it starts")
    for before, after in zip(ordered, ordered[1:], strict=False):
        if before.end != after.start:
            problems.append(f"{before.name} and {after.name} are not contiguous")
    for name in GAPS:
        if p.windows[name].end - p.windows[name].start < horizon:
            problems.append(f"{name} is shorter than the label horizon")
    if p.windows["follow_up"].days < MIN_FOLLOW_UP_DAYS:
        problems.append(f"follow-up is shorter than {MIN_FOLLOW_UP_DAYS} days")
    rules = (
        ("classifier", "fit", "calibration"),
        ("calibrator", "calibration", "validation"),
        ("policy", "validation", "test"),
    )
    for freeze, fitted, following in rules:
        at = p.freezes.get(freeze)
        if at is None:
            problems.append(f"missing {freeze} freeze")
            continue
        if at < p.windows[fitted].end + horizon:
            problems.append(f"{freeze} freeze is before {fitted} labels can mature")
        if at > p.windows[following].start:
            problems.append(f"{freeze} freeze is after {following} starts")
    if p.order_start != p.windows["warm_up"].start or p.order_end != p.windows["test"].end:
        problems.append("order horizon must run from warm-up start to test end")
    if p.observed_end != p.windows["follow_up"].end:
        problems.append("observation must end with the follow-up window")
    for family, starts in p.family_starts.items():
        if starts is not None and starts != p.windows["test"].start:
            problems.append(f"family {family} must start at the test window")
    minimum = int(p.raw["seeds"].get("minimum_final", 8))
    if len(set(p.final_seeds)) != len(p.final_seeds) or len(p.final_seeds) < minimum:
        problems.append(f"need at least {minimum} distinct final seeds")
    if set(p.final_seeds) & set(p.development_seeds):
        problems.append("final seeds overlap development seeds")
    if p.canonical_seed not in p.development_seeds:
        problems.append("the canonical seed must be a development seed")
    if problems:
        raise ProtocolError("; ".join(problems))


def load_protocol(path: Path | None = None) -> Protocol:
    with (path or PROTOCOL_PATH).open() as handle:
        return parse_protocol(yaml.safe_load(handle))


# ------------------------------------------------------------------- freeze
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_freeze_marker(root: Path = REPO, protocol_path: Path | None = None) -> Path:
    """Record the SHA-256 of every file the protocol freezes (run in the freeze commit)."""
    protocol_path = protocol_path or root / "experiments" / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    if protocol.placeholders():
        raise FreezeError(f"placeholders remain: {protocol.placeholders()}")
    spec = protocol.raw["freeze"]
    marker = root / spec["marker"]
    files = {name: file_sha256(root / name) for name in sorted(spec["files"])}
    marker.write_text(json.dumps({"files": files, "fixes": []}, indent=2, sort_keys=True) + "\n")
    return marker


def require_freeze(root: Path = REPO, protocol_path: Path | None = None) -> dict[str, Any]:
    """Refuse (FreezeError) unless the protocol is complete and frozen; return the marker."""
    protocol_path = protocol_path or root / "experiments" / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    if remaining := protocol.placeholders():
        raise FreezeError(f"protocol not complete: {remaining}")
    spec = protocol.raw["freeze"]
    marker_path = root / spec["marker"]
    if not marker_path.exists():
        raise FreezeError(f"no freeze marker at {spec['marker']}")
    marker = json.loads(marker_path.read_text())
    recorded = marker.get("files", {})
    missing = sorted(set(spec["files"]) - set(recorded))
    if missing:
        raise FreezeError(f"freeze marker does not cover {missing}")
    changed = sorted(
        name for name, digest in recorded.items()
        if not (root / name).exists() or file_sha256(root / name) != digest
    )
    if changed:
        raise FreezeError(f"files changed since the freeze: {changed}")
    return marker


def check_seed(seed: int, root: Path = REPO, protocol_path: Path | None = None) -> None:
    """Allow any seed except a final seed before a valid freeze."""
    protocol = load_protocol(protocol_path or root / "experiments" / "protocol.yaml")
    if int(seed) in protocol.final_seeds:
        require_freeze(root, protocol_path)
