"""Structured results: metric objects, one result file per stage, and the summary.

Every number the documents publish comes from a result file. Each pipeline
stage (``world``, ``context``, ``detection``, ``replay``, ``llm``, ...) writes
one :class:`StageResult` to ``results/<stage>.json`` with :func:`write_result`.
The pipeline then assembles ``results/summary.json`` with :func:`write_summary`,
and the renderer looks numbers up by key with :func:`metric`, which raises
``KeyError`` for a missing key so rendering fails instead of printing a gap.
Metric keys are dotted lower-case names such as
``replay.net_contribution_vs_approve_all.baseline``.

A :class:`Metric` is a value together with the question it answers: its unit,
the population that was counted and the protocol window it covers (``test``,
``validation``, ... or ``all``). Rates and shares carry their numerator and
denominator, and any metric that has them must equal their quotient (``bps``
is 10,000 times the quotient). A difference of two rates or shares is not a
quotient: record it in ``bps`` (10,000 times the difference) with no
numerator or denominator, and documents print it in percentage points. Money
is in cents, and a money figure is published beside the count it covers,
recorded as a metric of its own. An :class:`Interval` covers the variation its
method resamples within one world; the spread over seeds (one world per seed)
is a :class:`SeedSpread`, which also counts how many seeds are positive,
negative and zero for paired differences.

A zero denominator is not a metric. When there is nothing to measure, or a
figure is withheld (for example below a minimum support), the stage records
:meth:`Metric.not_evaluated`: the value is ``None``, the note says why, and a
rate or share still carries its numerator and denominator, so documents can
show "N=0, not evaluated".

The summary records the versions all stages agree on (world, features, policy,
benchmark, ...) and each stage's inputs. It refuses stages that disagree on a
version or on the SHA-256 of an input they both name (:class:`VersionMismatch`),
and it refuses duplicate stage names, metric keys and table names.

Files are deterministic: sorted keys, two-space indent, a trailing newline, no
timestamps, integers kept as integers, and NaN or infinity rejected wherever
they appear. Invalid content raises ``ValueError``. Formatting numbers for
documents (percentages, dollars) belongs to the renderer, not to this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 1
UNITS = ("count", "cents", "rate", "share", "bps", "minutes", "hours", "ratio", "score")
QUOTIENT_UNITS = ("rate", "share")
TOLERANCE = 1e-12
KEY_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUMMARY_FILE = "summary.json"

Number = int | float
Cell = str | int | float | bool | None


class VersionMismatch(ValueError):
    """Stages disagree on a version, or on the hash of an input they both name.

    ``kind`` is ``"version"`` or ``"input"``, ``key`` the version or input
    name, and ``stages_by_value`` maps each reported value to its stages.
    """

    def __init__(self, kind: str, key: str, stages_by_value: Mapping[str, Iterable[str]]):
        self.kind = kind
        self.key = key
        self.stages_by_value = {
            value: tuple(sorted(stages)) for value, stages in sorted(stages_by_value.items())
        }
        detail = "; ".join(
            f"{value!r} in {', '.join(stages)}" for value, stages in self.stages_by_value.items()
        )
        super().__init__(f"stages disagree on {kind} {key!r}: {detail}")


def _number(value: Any, what: str) -> Number:
    """Return a finite int or float; numpy scalars become plain Python numbers."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return number


def _optional_number(value: Any, what: str) -> Number | None:
    return None if value is None else _number(value, what)


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{what} must be a non-empty string without outer spaces, got {value!r}")
    return value


def _matching(value: Any, pattern: re.Pattern[str], what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{what} {value!r} does not match {pattern.pattern}")
    return value


def _mapping(value: Any, what: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _fields(data: Any, names: Iterable[str], what: str) -> Mapping[str, Any]:
    """Check that a serialized object has exactly the expected fields."""
    data = _mapping(data, what)
    expected = set(names)
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected, key=str)
    if missing or unknown:
        raise ValueError(f"{what}: missing fields {missing}, unknown fields {unknown}")
    return data


@dataclass(frozen=True)
class Interval:
    """A confidence interval from ``method`` (e.g. ``wilson``, ``cluster_bootstrap``,
    ``exact``) at ``level``, with ``low <= high``."""

    low: Number
    high: Number
    method: str
    level: float = 0.95

    def __post_init__(self) -> None:
        low = _number(self.low, "interval low")
        high = _number(self.high, "interval high")
        if low > high:
            raise ValueError(f"interval low {low!r} is above high {high!r}")
        level = float(_number(self.level, "interval level"))
        if not 0 < level < 1:
            raise ValueError(f"interval level must be between 0 and 1, got {level!r}")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "method", _matching(self.method, NAME_PATTERN, "method"))
        object.__setattr__(self, "level", level)

    def to_dict(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high, "method": self.method, "level": self.level}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Interval:
        data = _fields(data, ("low", "high", "method", "level"), "interval")
        return cls(data["low"], data["high"], data["method"], data["level"])


class SignCount(NamedTuple):
    """How many seeds are above, below and at zero."""

    positive: int
    negative: int
    zero: int


@dataclass(frozen=True)
class SeedSpread:
    """One value per seed (for example a paired difference per final seed).

    Seeds are non-negative ints, kept in ascending order. In files the seeds
    are written as decimal strings, since JSON object keys are strings.
    """

    per_seed: dict[int, Number]

    def __post_init__(self) -> None:
        per_seed = _mapping(self.per_seed, "per_seed")
        if not per_seed:
            raise ValueError("a seed spread needs at least one seed")
        values: dict[int, Number] = {}
        for seed, value in per_seed.items():
            if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
                raise ValueError(f"seed must be a non-negative int, got {seed!r}")
            values[int(seed)] = _number(value, f"value for seed {seed}")
        object.__setattr__(self, "per_seed", dict(sorted(values.items())))

    @property
    def n(self) -> int:
        return len(self.per_seed)

    @property
    def mean(self) -> float:
        return math.fsum(self.per_seed.values()) / self.n

    @property
    def min(self) -> Number:
        return min(self.per_seed.values())

    @property
    def max(self) -> Number:
        return max(self.per_seed.values())

    @property
    def sign_count(self) -> SignCount:
        values = self.per_seed.values()
        return SignCount(
            positive=sum(value > 0 for value in values),
            negative=sum(value < 0 for value in values),
            zero=sum(value == 0 for value in values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"per_seed": {str(seed): value for seed, value in self.per_seed.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeedSpread:
        per_seed = _mapping(_fields(data, ("per_seed",), "seeds")["per_seed"], "per_seed")
        seeds: dict[int, Any] = {}
        for key, value in per_seed.items():
            if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
                raise ValueError(f"seed key must be a decimal integer string, got {key!r}")
            seeds[int(key)] = value
        return cls(seeds)


def _quotient_scale(unit: str) -> int:
    return 10_000 if unit == "bps" else 1


@dataclass(frozen=True, kw_only=True)
class Metric:
    """One published number with its unit, population and window.

    ``value`` is an int or float, or ``None`` for a metric that was not
    evaluated (see :meth:`not_evaluated`). ``numerator`` and ``denominator`` come
    together; they are required for ``rate`` and ``share``. When both are given
    and the value is set, the denominator is positive and ``value`` equals
    ``numerator / denominator`` (times 10,000 for ``bps``) within ``TOLERANCE``
    (relative above 1). ``interval``, ``seeds`` and ``note`` are optional.
    """

    value: Number | None
    unit: str
    population: str
    window: str
    numerator: Number | None = None
    denominator: Number | None = None
    interval: Interval | None = None
    seeds: SeedSpread | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {', '.join(UNITS)}, got {self.unit!r}")
        _text(self.population, "population")
        _matching(self.window, NAME_PATTERN, "window")
        if self.note is not None:
            _text(self.note, "note")
        if self.interval is not None and not isinstance(self.interval, Interval):
            raise ValueError(f"interval must be an Interval, got {self.interval!r}")
        if self.seeds is not None and not isinstance(self.seeds, SeedSpread):
            raise ValueError(f"seeds must be a SeedSpread, got {self.seeds!r}")
        value = _optional_number(self.value, "value")
        numerator = _optional_number(self.numerator, "numerator")
        denominator = _optional_number(self.denominator, "denominator")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

        if (numerator is None) != (denominator is None):
            raise ValueError("numerator and denominator must be given together")
        if numerator is None and self.unit in QUOTIENT_UNITS:
            raise ValueError(f"a {self.unit} needs its numerator and denominator")
        if value is None:
            if self.note is None:
                raise ValueError("a metric without a value needs a note saying why")
            if self.interval is not None or self.seeds is not None:
                raise ValueError("a metric without a value cannot carry an interval or seeds")
        if numerator is None or denominator is None:
            return
        if denominator < 0:
            raise ValueError(f"denominator must not be negative, got {denominator!r}")
        if denominator == 0:
            if value is not None:
                raise ValueError(
                    "a zero denominator is not a metric; record Metric.not_evaluated(...)"
                )
            if numerator != 0:
                raise ValueError(
                    f"numerator must be 0 when the denominator is 0, got {numerator!r}"
                )
            return
        if value is not None:
            expected = _quotient_scale(self.unit) * numerator / denominator
            if abs(value - expected) > TOLERANCE * max(1.0, abs(expected)):
                raise ValueError(
                    f"value {value!r} does not equal {numerator!r}/{denominator!r} "
                    f"({expected!r}) for unit {self.unit}"
                )

    @property
    def evaluated(self) -> bool:
        return self.value is not None

    @classmethod
    def from_ratio(
        cls,
        numerator: Number,
        denominator: Number,
        *,
        population: str,
        window: str,
        unit: str = "rate",
        interval: Interval | None = None,
        seeds: SeedSpread | None = None,
        note: str | None = None,
    ) -> Metric:
        """A metric whose value is ``numerator / denominator`` (times 10,000 for bps).

        A zero denominator raises; record :meth:`not_evaluated` instead.
        """
        numerator = _number(numerator, "numerator")
        denominator = _number(denominator, "denominator")
        if denominator == 0:
            raise ValueError("a zero denominator is not a metric; record Metric.not_evaluated(...)")
        return cls(
            value=_quotient_scale(unit) * numerator / denominator,
            unit=unit,
            population=population,
            window=window,
            numerator=numerator,
            denominator=denominator,
            interval=interval,
            seeds=seeds,
            note=note,
        )

    @classmethod
    def not_evaluated(
        cls,
        *,
        unit: str,
        population: str,
        window: str,
        reason: str,
        numerator: Number | None = None,
        denominator: Number | None = None,
    ) -> Metric:
        """A metric with no value: ``value`` is ``None`` and ``note`` is the reason.

        Rates and shares still give their numerator and denominator (``0`` and
        ``0`` when nothing was there to count), so the documents can show N.
        """
        return cls(
            value=None,
            unit=unit,
            population=population,
            window=window,
            numerator=numerator,
            denominator=denominator,
            note=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "population": self.population,
            "window": self.window,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "interval": None if self.interval is None else self.interval.to_dict(),
            "seeds": None if self.seeds is None else self.seeds.to_dict(),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Metric:
        names = (
            "value",
            "unit",
            "population",
            "window",
            "numerator",
            "denominator",
            "interval",
            "seeds",
            "note",
        )
        data = _fields(data, names, "metric")
        return cls(
            value=data["value"],
            unit=data["unit"],
            population=data["population"],
            window=data["window"],
            numerator=data["numerator"],
            denominator=data["denominator"],
            interval=None if data["interval"] is None else Interval.from_dict(data["interval"]),
            seeds=None if data["seeds"] is None else SeedSpread.from_dict(data["seeds"]),
            note=data["note"],
        )


def _cell(value: Any, what: str) -> Cell:
    if value is None or isinstance(value, (str, bool)):
        return value
    return _number(value, what)


def _table(name: str, rows: Any) -> list[dict[str, Cell]]:
    if not isinstance(rows, list):
        raise ValueError(f"table {name!r} must be a list of rows")
    copied: list[dict[str, Cell]] = []
    for index, row in enumerate(rows):
        row = _mapping(row, f"table {name!r} row {index}")
        if copied and set(row) != set(copied[0]):
            raise ValueError(f"table {name!r} row {index} has different columns from row 0")
        copied.append(
            {
                _matching(column, NAME_PATTERN, f"table {name!r} column"): _cell(
                    cell, f"table {name!r} row {index} column {column!r}"
                )
                for column, cell in row.items()
            }
        )
    return copied


@dataclass(frozen=True, kw_only=True)
class StageResult:
    """The result file of one pipeline stage, written to ``<root>/<stage>.json``.

    ``versions`` names what the stage depends on (``world``, ``features``,
    ``policy``, ``benchmark``, ...); ``inputs`` maps each input name to the
    SHA-256 of that file or artifact (see :func:`file_sha256`); ``metrics`` and
    ``tables`` are keyed by dotted lower-case names. Table rows share one set of
    columns and hold plain values (str, number, bool or None); column order is
    not kept, so templates name the columns they show.
    """

    stage: str
    versions: dict[str, str]
    inputs: dict[str, str]
    metrics: dict[str, Metric]
    tables: dict[str, list[dict[str, Cell]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _matching(self.stage, NAME_PATTERN, "stage")
        if self.stage == Path(SUMMARY_FILE).stem:
            raise ValueError(f"{self.stage!r} is reserved for the summary file")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {self.schema_version!r}")
        versions = {
            _matching(name, NAME_PATTERN, "version name"): _text(value, f"version {name!r}")
            for name, value in _mapping(self.versions, "versions").items()
        }
        inputs = {
            _text(name, "input name"): _matching(digest, SHA256_PATTERN, f"input {name!r} sha256")
            for name, digest in _mapping(self.inputs, "inputs").items()
        }
        metrics = {}
        for key, item in _mapping(self.metrics, "metrics").items():
            if not isinstance(item, Metric):
                raise ValueError(f"metric {key!r} must be a Metric, got {item!r}")
            metrics[_matching(key, KEY_PATTERN, "metric key")] = item
        tables = {
            _matching(name, KEY_PATTERN, "table name"): _table(name, rows)
            for name, rows in _mapping(self.tables, "tables").items()
        }
        if not isinstance(self.notes, list):
            raise ValueError("notes must be a list of strings")
        notes = [_text(note, "note") for note in self.notes]
        object.__setattr__(self, "versions", dict(sorted(versions.items())))
        object.__setattr__(self, "inputs", dict(sorted(inputs.items())))
        object.__setattr__(self, "metrics", dict(sorted(metrics.items())))
        object.__setattr__(self, "tables", dict(sorted(tables.items())))
        object.__setattr__(self, "notes", notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "schema_version": self.schema_version,
            "versions": dict(self.versions),
            "inputs": dict(self.inputs),
            "metrics": {key: item.to_dict() for key, item in self.metrics.items()},
            "tables": {name: [dict(row) for row in rows] for name, rows in self.tables.items()},
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageResult:
        names = ("stage", "schema_version", "versions", "inputs", "metrics", "tables", "notes")
        data = _fields(data, names, "stage result")
        return cls(
            stage=data["stage"],
            schema_version=data["schema_version"],
            versions=data["versions"],
            inputs=data["inputs"],
            metrics={
                key: Metric.from_dict(item)
                for key, item in _mapping(data["metrics"], "metrics").items()
            },
            tables=data["tables"],
            notes=data["notes"],
        )


def canonical_json(data: Any) -> str:
    """Deterministic JSON: sorted keys, two-space indent, trailing newline, no NaN."""
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(data).encode("utf-8"))
    return path


def _read(path: Path) -> Any:
    def reject(constant: str) -> None:
        raise ValueError(f"{path}: {constant} is not allowed in a result file")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def file_sha256(path: Path) -> str:
    """The SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_result(result: StageResult, root: Path) -> Path:
    """Write ``result`` to ``root/<stage>.json`` and return the path."""
    return _write(Path(root) / f"{result.stage}.json", result.to_dict())


def read_result(path: Path) -> StageResult:
    """Read and validate a stage result file; its name must be ``<stage>.json``."""
    path = Path(path)
    result = StageResult.from_dict(_read(path))
    if path.name != f"{result.stage}.json":
        raise ValueError(f"{path} holds stage {result.stage!r}; expected {result.stage}.json")
    return result


def _agreed(results: list[StageResult], kind: str, attribute: str) -> dict[str, str]:
    stages_by_value: dict[str, dict[str, list[str]]] = {}
    for result in results:
        for name, value in getattr(result, attribute).items():
            stages_by_value.setdefault(name, {}).setdefault(value, []).append(result.stage)
    for name, by_value in sorted(stages_by_value.items()):
        if len(by_value) > 1:
            raise VersionMismatch(kind, name, by_value)
    return {name: next(iter(by_value)) for name, by_value in sorted(stages_by_value.items())}


def assemble_summary(results: Iterable[StageResult]) -> dict[str, Any]:
    """Combine stage results into the summary dictionary.

    Raises ``ValueError`` on no results, a repeated stage, or a metric key or
    table name reported by two stages, and :class:`VersionMismatch` when stages
    disagree on a version or on the hash of a shared input.
    """
    results = list(results)
    if not results:
        raise ValueError("no stage results to summarize")
    stages: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    owners: dict[tuple[str, str], str] = {}
    for result in results:
        if not isinstance(result, StageResult):
            raise ValueError(f"expected a StageResult, got {result!r}")
        if result.stage in stages:
            raise ValueError(f"stage {result.stage!r} appears more than once")
        data = result.to_dict()
        for kind, items, target in (
            ("metric key", data["metrics"], metrics),
            ("table name", data["tables"], tables),
        ):
            for name, item in items.items():
                if (kind, name) in owners:
                    raise ValueError(
                        f"{kind} {name!r} is reported by both "
                        f"{owners[kind, name]!r} and {result.stage!r}"
                    )
                owners[kind, name] = result.stage
                target[name] = item
        stages[result.stage] = {
            "versions": data["versions"],
            "inputs": data["inputs"],
            "metrics": sorted(data["metrics"]),
            "tables": sorted(data["tables"]),
            "notes": data["notes"],
        }
    versions = _agreed(results, "version", "versions")
    _agreed(results, "input", "inputs")
    return {
        "schema_version": SCHEMA_VERSION,
        "versions": versions,
        "stages": stages,
        "metrics": metrics,
        "tables": tables,
    }


def write_summary(results: Iterable[StageResult], root: Path) -> Path:
    """Assemble the summary of ``results`` and write it to ``root/summary.json``."""
    return _write(Path(root) / SUMMARY_FILE, assemble_summary(results))


def read_summary(path: Path) -> dict[str, Any]:
    """Load a summary file written by :func:`write_summary`."""
    summary = _read(Path(path))
    if not isinstance(summary, dict) or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} is not a schema version {SCHEMA_VERSION} summary")
    return summary


def metric(summary: Mapping[str, Any], key: str) -> Metric:
    """The metric stored under ``key``; a missing key raises ``KeyError`` naming it."""
    metrics = summary.get("metrics", {})
    if key not in metrics:
        raise KeyError(f"result key {key!r} is not in the summary")
    return Metric.from_dict(metrics[key])


def table(summary: Mapping[str, Any], name: str) -> list[dict[str, Cell]]:
    """The rows of table ``name``; a missing name raises ``KeyError`` naming it."""
    tables = summary.get("tables", {})
    if name not in tables:
        raise KeyError(f"result table {name!r} is not in the summary")
    return [dict(row) for row in tables[name]]
