"""The pre-registered evaluation protocol (``experiments/protocol.yaml``).

:func:`load_protocol` reads the file and checks its structure: contiguous
windows, every gap at least the label horizon, freeze dates after each fitted
window's labels mature and before the next phase starts, a follow-up of at
least 90 days, and final seeds disjoint from the development seeds.

:func:`require_freeze` is what the final driver calls before generating any
final-seed world: it refuses unless the freeze marker exists, every file it
lists still has its recorded SHA-256, and no ``TO_COMPLETE_AT_FREEZE``
placeholder remains in the protocol. :func:`check_seed` applies that rule to a
seed. An entry of ``freeze.files`` ending in ``/`` is a directory: it freezes
every file under it that git does not ignore (``__pycache__`` never counts), so a
file added there after the freeze is refused like a changed one.

The protocol's ``cases`` section (the pre-registered case-selection rule) is
checked for its keys and slot names here, and its replay cell against the
replay's variants and capacity levels by :func:`check_case_replay`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
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
    # Sensitivity families replayed at the base allotment only (``sensitivity.fulfilment_lag``)
    base_only_families: tuple[str, ...] = ()

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


def _base_only(raw: dict[str, Any]) -> tuple[str, ...]:
    """The fulfilment-lag families (``sensitivity.fulfilment_lag.families``: name to
    lag factor), or none while the entry is a placeholder."""
    entry = (raw.get("sensitivity") or {}).get("fulfilment_lag", PLACEHOLDER)
    if entry == PLACEHOLDER:
        return ()
    families = entry.get("families") if isinstance(entry, dict) else None
    if not isinstance(families, dict) or not families:
        raise ProtocolError("sensitivity.fulfilment_lag.families must map each family to its "
                            "lag factor")
    for name, factor in families.items():
        if isinstance(factor, bool) or not isinstance(factor, int | float) or not factor > 0:
            raise ProtocolError(f"sensitivity.fulfilment_lag: {name}'s lag factor must be a "
                                f"positive number, got {factor!r}")
    return tuple(families)


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
            base_only_families=_base_only(raw),
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
    for family in p.base_only_families:
        if family not in p.family_starts:
            problems.append(f"sensitivity family {family} is not a family")
        elif p.family_starts[family] is None:
            problems.append(f"sensitivity family {family} must start at the test window")
    minimum = int(p.raw["seeds"].get("minimum_final", 8))
    if len(set(p.final_seeds)) != len(p.final_seeds) or len(p.final_seeds) < minimum:
        problems.append(f"need at least {minimum} distinct final seeds")
    if set(p.final_seeds) & set(p.development_seeds):
        problems.append("final seeds overlap development seeds")
    if p.canonical_seed not in p.development_seeds:
        problems.append("the canonical seed must be a development seed")
    problems += _case_problems(p.raw.get("cases", PLACEHOLDER))
    if problems:
        raise ProtocolError("; ".join(problems))


CASE_KEYS = ("world", "replay", "population", "selection", "slots", "files",
             "selection_limits", "facts", "later_events", "missing", "publication")
CASE_REPLAY_KEYS = ("policy", "thresholds", "capacity_level", "layout", "history", "reviewer",
                    "verification", "window")
CASE_TIERS = ("reviewed_alert", "checkout_alert")


def _case_problems(cases: Any) -> list[str]:
    """Structural problems of the ``cases`` section (none while it is a placeholder)."""
    if cases == PLACEHOLDER:
        return []
    if not isinstance(cases, dict):
        return ["cases must be a mapping"]
    problems = []
    if missing := [key for key in CASE_KEYS if key not in cases]:
        problems.append(f"cases lacks {missing}")
    if extra := sorted(set(cases) - set(CASE_KEYS)):
        problems.append(f"cases has unknown keys {extra}")
    replay = cases.get("replay")
    if not isinstance(replay, dict) or sorted(replay) != sorted(CASE_REPLAY_KEYS):
        problems.append(f"cases.replay must give exactly {list(CASE_REPLAY_KEYS)}")
    world = cases.get("world")
    if not isinstance(world, dict) or set(world) != {"seed", "family"}:
        problems.append("cases.world must give its seed and family")
    selection, slots, files = (cases.get(key) or {} for key in ("selection", "slots", "files"))
    order = selection.get("slot_order") if isinstance(selection, dict) else None
    if not isinstance(slots, dict) or not slots:
        problems.append("cases.slots must name the slots")
        slots = {}
    if not isinstance(order, list) or sorted(order) != sorted(slots) or \
            len(set(order)) != len(order):
        problems.append("cases.selection.slot_order must list every slot once")
    for name, slot in slots.items():
        tiers = slot.get("tiers") if isinstance(slot, dict) else None
        if not isinstance(slot, dict) or not slot.get("predicate") or not tiers or \
                any(tier not in CASE_TIERS for tier in tiers):
            problems.append(f"cases.slots.{name} needs a predicate and tiers from "
                            f"{list(CASE_TIERS)}")
    used: list[str] = []
    for name, entry in (files.items() if isinstance(files, dict) else []):
        named = entry.get("slots") if isinstance(entry, dict) else None
        if not isinstance(named, list) or not named or any(s not in slots for s in named):
            problems.append(f"cases.files.{name} must name slots of cases.slots")
            continue
        if entry.get("length") not in ("full", "short"):
            problems.append(f"cases.files.{name}.length must be full or short")
        if "primary_alert" in entry and entry["primary_alert"] not in named:
            problems.append(f"cases.files.{name}.primary_alert must be one of its slots")
        used += named
    if sorted(used) != sorted(slots):
        problems.append("cases.files must use every slot exactly once")
    return problems


def check_case_replay(protocol: Protocol, variants: Iterable[tuple[str, str, str]],
                      levels: Iterable[str], layouts: Iterable[str]) -> None:
    """Refuse (ProtocolError) a ``cases`` replay cell the replay does not run: its world
    must be a protocol family and the canonical seed, its policy a protocol policy, its
    (history, reviewer, verification) the replay's main variant, and its capacity level
    and layout configured ones."""
    cases = protocol.raw.get("cases", PLACEHOLDER)
    if cases == PLACEHOLDER:
        return
    replay, world = cases["replay"], cases["world"]
    problems = []
    if world["family"] not in protocol.family_starts:
        problems.append(f"cases.world.family {world['family']!r} is not a protocol family")
    if world["seed"] != protocol.canonical_seed:
        problems.append("cases.world.seed must be the canonical seed")
    if replay["policy"] not in protocol.raw["policies"]:
        problems.append(f"cases.replay.policy {replay['policy']!r} is not a protocol policy")
    main = tuple(variants)[0]
    if (replay["history"], replay["reviewer"], replay["verification"]) != tuple(main):
        problems.append(f"cases.replay must be the replay's main variant {tuple(main)}")
    if replay["capacity_level"] not in set(levels):
        problems.append(f"cases.replay.capacity_level {replay['capacity_level']!r} is not "
                        "a configured level")
    if replay["layout"] not in set(layouts):
        problems.append(f"cases.replay.layout {replay['layout']!r} is not a configured layout")
    if replay["window"] not in protocol.windows:
        problems.append(f"cases.replay.window {replay['window']!r} is not a window")
    if problems:
        raise ProtocolError("; ".join(problems))


def load_protocol(path: Path | None = None) -> Protocol:
    with (path or PROTOCOL_PATH).open() as handle:
        return parse_protocol(yaml.safe_load(handle))


# ------------------------------------------------------------------- freeze
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files(root: Path, entries: Iterable[str]) -> list[str]:
    """The files ``freeze.files`` names under ``root``: a file entry as given; a directory
    entry (ending in ``/``) as every file under it that git does not ignore (all files
    outside a repository), never ``__pycache__``. A missing entry or an empty directory
    raises ``FreezeError``."""
    out: set[str] = set()
    for entry in entries:
        if not entry.endswith("/"):
            if not (root / entry).is_file():
                raise FreezeError(f"frozen file {entry} does not exist")
            out.add(entry)
            continue
        if not (root / entry).is_dir():
            raise FreezeError(f"frozen directory {entry} does not exist")
        try:
            listed = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--cached", "--others",
                 "--exclude-standard", "-z", "--", entry],
                capture_output=True, text=True, check=True).stdout.split("\0")
        except (OSError, subprocess.CalledProcessError):
            listed = [path.relative_to(root).as_posix()
                      for path in (root / entry).rglob("*") if path.is_file()]
        found = {path for path in listed if path and (root / path).is_file()
                 and "__pycache__" not in Path(path).parts}
        if not found:
            raise FreezeError(f"frozen directory {entry} holds no files")
        out |= found
    return sorted(out)


def write_freeze_marker(root: Path = REPO, protocol_path: Path | None = None) -> Path:
    """Record the SHA-256 of every file the protocol freezes (run in the freeze commit)."""
    protocol_path = protocol_path or root / "experiments" / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    if protocol.placeholders():
        raise FreezeError(f"placeholders remain: {protocol.placeholders()}")
    spec = protocol.raw["freeze"]
    marker = root / spec["marker"]
    files = {name: file_sha256(root / name) for name in frozen_files(root, spec["files"])}
    marker.write_text(json.dumps({"files": files, "fixes": []}, indent=2, sort_keys=True) + "\n")
    return marker


def log_fix(what: str, effect: str, root: Path = REPO,
            protocol_path: Path | None = None) -> list[str]:
    """Record a bug fix made after the freeze: re-hash the frozen files that changed (or
    were added or removed) and append to the marker's ``fixes`` the files, what was
    wrong (``what``) and the fix's before/after effect on the results (``effect``).
    Refuses when nothing frozen changed, or without a marker. Returns the files."""
    protocol_path = protocol_path or root / "experiments" / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    if not what.strip() or not effect.strip():
        raise FreezeError("a logged fix says what was wrong and its before/after effect")
    spec = protocol.raw["freeze"]
    marker_path = root / spec["marker"]
    if not marker_path.exists():
        raise FreezeError(f"no freeze marker at {spec['marker']}: freeze first")
    marker = json.loads(marker_path.read_text())
    recorded = marker.get("files", {})
    files = {name: file_sha256(root / name) for name in frozen_files(root, spec["files"])}
    touched = sorted(name for name in set(files) | set(recorded)
                     if files.get(name) != recorded.get(name))
    if not touched:
        raise FreezeError("no frozen file changed: nothing to log")
    marker["files"] = files
    marker.setdefault("fixes", []).append({"files": touched, "what": what.strip(),
                                           "effect": effect.strip()})
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return touched


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
    try:
        current = frozen_files(root, spec["files"])
    except FreezeError as error:
        raise FreezeError(f"files changed since the freeze: {error}") from error
    missing = sorted(set(current) - set(recorded))
    if missing:
        raise FreezeError(f"freeze marker does not cover {missing} (a file added since the "
                          "freeze, or a marker written for another list)")
    changed = sorted(
        name for name, digest in recorded.items()
        if not (root / name).exists() or file_sha256(root / name) != digest
    )
    if changed:
        raise FreezeError(f"files changed since the freeze: {changed}")
    if uncommitted := _uncommitted(root, [spec["marker"], *current]):
        raise FreezeError(f"freeze not committed: {uncommitted}")
    return marker


def _uncommitted(root: Path, paths: list[str]) -> list[str]:
    """Paths that are not committed, unchanged, at the repository's HEAD."""
    def git(*args: str) -> str:
        try:
            done = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                  text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise FreezeError(f"cannot read the freeze commit: {error}") from error
        return done.stdout

    in_head = set(git("ls-tree", "-r", "--name-only", "HEAD", "--", *paths).splitlines())
    dirty = {line[3:] for line in
             git("status", "--porcelain", "--untracked-files=all", "--", *paths).splitlines()}
    return sorted(path for path in paths if path not in in_head or path in dirty)


def check_seed(seed: int, root: Path = REPO, protocol_path: Path | None = None) -> None:
    """Allow any seed except a final seed before a valid freeze."""
    protocol = load_protocol(protocol_path or root / "experiments" / "protocol.yaml")
    if int(seed) in protocol.final_seeds:
        require_freeze(root, protocol_path)
