"""The study's pipeline: one linear run from world generation to rendered documents.

    python pipeline.py run --profile dev            # development seeds, every family
    python pipeline.py run --profile ci             # one small world, end to end
    python pipeline.py run --profile canonical      # seed 416 at full size, loaded into MySQL
    python pipeline.py run --profile final          # final seeds; refused until the freeze
    python pipeline.py run --world DIR              # an existing world instead of generating one
    python pipeline.py stages                       # list the stages

Stages run in this order, each in the same process: world, validate, load,
context, fit, tune, replay, alerts, evaluate, llm, then the summary and the
documents. Options ``--from`` and ``--until`` run part of the list on an
existing run directory.

A run lives in ``runs/<name>/`` (not committed): the generated worlds under
``worlds/<seed>-<family>/``, each stage's artifacts under its own directory,
and ``lineage.json`` (the code commit, timings, and the SHA-256 of every
stage's inputs and outputs). Each stage writes one result file
(``core.results.StageResult``) with the versions it depends on (world,
features, policy, protocol, benchmark) and the hashes of its inputs; the
summary is assembled only from a complete set of stage results that agree on
those versions. Only the ``final`` profile writes the committed ``results/``
and documents, so no other run can overwrite them. Result files carry no
timestamps or commit ids, so regenerating them from the same inputs leaves
``git status`` clean; the lineage file holds the rest.

The ``final`` profile refuses to generate any world unless the protocol freeze
check passes (``core.protocol.require_freeze``), and every profile refuses a
final seed before the freeze.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core import protocol as protocol_module  # noqa: E402
from core import world as world_module  # noqa: E402
from core.config import db_settings, mysql_reachable  # noqa: E402
from core.results import (  # noqa: E402
    SUMMARY_FILE,
    Metric,
    SeedSpread,
    StageResult,
    canonical_json,
    file_sha256,
    read_result,
    read_summary,
    write_result,
    write_summary,
)
from core.stats import paired_seed_differences  # noqa: E402

RUNS = REPO / "runs"
RESULTS = REPO / "results"
POLICY_TABLES_SQL = REPO / "db" / "policy_tables.sql"
CI_SCALE = 0.1  # the CI world: small enough to generate within two minutes
BASELINE = "baseline"
REFERENCES = ("approve_all", "incumbent_rules")
VERSION_FILES = {
    "features": ("core/asof.py",),
    "policy": ("config/policy.yaml", "policy/fraud-policy.md", "core/actions.py",
               "core/evidence.py"),
    "protocol": ("experiments/protocol.yaml",),
}


class PipelineError(RuntimeError):
    """The run cannot continue (bad arguments, missing inputs, an unmet precondition)."""


class StageUnavailable(PipelineError):
    """A stage's implementation does not exist yet."""


# ---------------------------------------------------------------- run setup
@dataclass(frozen=True, order=True)
class WorldRef:
    seed: int
    family: str

    @property
    def name(self) -> str:
        return f"{self.seed}-{self.family}"


@dataclass(frozen=True)
class Profile:
    name: str
    seeds: str  # "development", "final" or "canonical": which protocol seeds
    families: tuple[str, ...] | None  # None: every family in the protocol
    scale: float
    publish: bool  # writes the committed results/ and documents


PROFILES = {
    "ci": Profile("ci", "canonical", (BASELINE,), CI_SCALE, publish=False),
    "dev": Profile("dev", "development", None, 1.0, publish=False),
    "canonical": Profile("canonical", "canonical", (BASELINE,), 1.0, publish=False),
    "final": Profile("final", "final", None, 1.0, publish=True),
}


@dataclass
class Run:
    """One pipeline run: its directory, worlds and where results go."""

    name: str
    directory: Path
    results_dir: Path
    docs_dir: Path
    protocol: protocol_module.Protocol
    seeds: tuple[int, ...]
    families: tuple[str, ...]
    scale: float
    database_world: WorldRef | None
    source: Path | None = None
    memory: dict[str, Any] = field(default_factory=dict)

    @property
    def worlds(self) -> list[WorldRef]:
        """The evaluation worlds: every seed in every family."""
        return [WorldRef(seed, family) for seed in self.seeds for family in self.families]

    @property
    def fit_seeds(self) -> tuple[int, ...]:
        """Seeds whose models are fitted: the evaluation seeds and the database world's."""
        return tuple(sorted({ref.seed for ref in self.all_worlds}))

    @property
    def all_worlds(self) -> list[WorldRef]:
        """Evaluation worlds plus the database world when it is not one of them."""
        worlds = list(self.worlds)
        if self.database_world is not None and self.database_world not in worlds:
            worlds.append(self.database_world)
        return sorted(worlds)

    def world_dir(self, ref: WorldRef) -> Path:
        return self.directory / "worlds" / ref.name

    def stage_dir(self, stage: str) -> Path:
        path = self.directory / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest(self, ref: WorldRef) -> dict[str, Any]:
        path = self.world_dir(ref) / "manifest.json"
        if not path.exists():
            raise PipelineError(f"world {ref.name} has not been generated (run the world stage)")
        return json.loads(path.read_text())

    def tables(self, ref: WorldRef) -> dict[str, pd.DataFrame]:
        cache = self.memory.setdefault("tables", {})
        if ref not in cache:
            self.manifest(ref)
            cache[ref] = world_module.read_world(self.world_dir(ref))
        return cache[ref]


def make_run(profile: Profile, *, name: str | None = None, seeds: list[int] | None = None,
             families: list[str] | None = None, world: Path | None = None,
             scale: float | None = None, runs: Path = RUNS, results: Path = RESULTS,
             docs: Path = REPO, protocol_path: Path | None = None) -> Run:
    """Resolve a profile and overrides into a :class:`Run`."""
    protocol = protocol_module.load_protocol(protocol_path)
    if world is not None:
        manifest_path = Path(world) / "manifest.json"
        if not manifest_path.exists():
            raise PipelineError(f"{world} holds no manifest.json")
        manifest = json.loads(manifest_path.read_text())
        ref = WorldRef(int(manifest["seed"]), str(manifest["family"]))
        chosen_seeds, chosen_families, database = (ref.seed,), (ref.family,), ref
    else:
        default_seeds = {
            "development": protocol.development_seeds,
            "final": protocol.final_seeds,
            "canonical": (protocol.canonical_seed,),
        }[profile.seeds]
        chosen_seeds = tuple(seeds) if seeds else tuple(default_seeds)
        chosen_families = tuple(families or profile.families or protocol.family_starts)
        unknown = sorted(set(chosen_families) - set(protocol.family_starts))
        if unknown:
            raise PipelineError(f"unknown families {unknown}")
        if BASELINE not in chosen_families:
            raise PipelineError("every run needs the baseline family (models are fitted on it)")
        canonical = WorldRef(protocol.canonical_seed, BASELINE)
        database = canonical if profile.name != "ci" else WorldRef(chosen_seeds[0], BASELINE)
    if len(set(chosen_seeds)) != len(chosen_seeds):
        raise PipelineError("seeds repeat")
    directory = Path(runs) / (name or profile.name)
    publish = profile.publish and world is None and not seeds and not families
    return Run(
        name=name or profile.name,
        directory=directory,
        results_dir=Path(results) if publish else directory / "results",
        docs_dir=Path(docs) if publish else directory / "docs",
        protocol=protocol,
        seeds=chosen_seeds,
        families=chosen_families,
        scale=profile.scale if scale is None else scale,
        database_world=database,
        source=Path(world) if world is not None else None,
    )


# ---------------------------------------------------------------- stage contract
@dataclass
class StageOutput:
    """What a stage hands back: published metrics and tables, plus lineage.

    ``inputs`` maps a stable name (a repository path, or a run-relative path)
    to a file whose SHA-256 goes into the result file; ``outputs`` lists files
    or directories the stage wrote, hashed into the run's lineage only.
    """

    metrics: dict[str, Metric] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    inputs: dict[str, Path] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class Stage:
    name: str
    function: Callable[[Run], StageOutput]
    versions: tuple[str, ...]
    description: str


def entry(stage: str, module: str, name: str) -> Callable[..., Any]:
    """The function ``module.name``, or :class:`StageUnavailable` if it does not exist yet."""
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name and (module == error.name or module.startswith(error.name + ".")):
            raise StageUnavailable(f"{stage}: {module}.{name} is not available yet") from error
        raise
    function = getattr(loaded, name, None)
    if not callable(function):
        raise StageUnavailable(f"{stage}: {module}.{name} is not available yet")

    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except NotImplementedError as error:
            raise StageUnavailable(f"{stage}: {module}.{name} is not implemented yet") from error

    return call


def _repo_inputs(*paths: str) -> dict[str, Path]:
    return {path: REPO / path for path in paths if (REPO / path).exists()}


def _world_inputs(run: Run, worlds: list[WorldRef]) -> dict[str, Path]:
    return {
        f"worlds/{ref.name}/manifest.json": run.world_dir(ref) / "manifest.json"
        for ref in worlds
    }


def _count(value: int, population: str, window: str = "all") -> Metric:
    return Metric(value=int(value), unit="count", population=population, window=window)


# ---------------------------------------------------------------- stages
def stage_world(run: Run) -> StageOutput:
    """Generate (or copy) every world of the run into its directory."""
    worlds = run.all_worlds
    for ref in worlds:  # refuse before generating anything
        protocol_module.check_seed(ref.seed)
    generate = None if run.source else entry("world", "simulator.generate", "generate_world")
    run.memory.clear()
    rows = []
    for ref in worlds:
        target = run.world_dir(ref)
        if target.exists():
            shutil.rmtree(target)
        if run.source is not None:
            shutil.copytree(run.source, target, ignore=shutil.ignore_patterns("*.py", "*.md"))
        else:
            generate(ref.seed, ref.family, target, scale=run.scale)
        manifest = run.manifest(ref)
        if (int(manifest["seed"]), manifest["family"]) != (ref.seed, ref.family):
            raise PipelineError(f"world {ref.name}: the manifest names another seed or family")
        tables = world_module.read_world(target, ["order_attempts", "labels", "accounts"])
        orders = tables["order_attempts"]
        rows.append({
            "seed": ref.seed,
            "family": ref.family,
            "accounts": len(tables["accounts"]),
            "order_attempts": len(orders),
            "processor_approved": int(orders["processor_result"].eq("approved").sum()),
            "positive_labels": int(tables["labels"].loc[tables["labels"]["label"].eq(1),
                                                        "order_id"].nunique()),
        })
    return StageOutput(
        metrics={"world.worlds": _count(len(worlds), "worlds generated for the run")},
        tables={"world.worlds": rows},
        inputs=_repo_inputs("config/world.yaml", "experiments/protocol.yaml"),
        outputs=[run.world_dir(ref) for ref in worlds],
    )


def stage_validate(run: Run) -> StageOutput:
    """Check every world against its manifest and the world contract (chronology included)."""
    rows = []
    for ref in run.all_worlds:
        tables = run.tables(ref)
        world_module.verify_manifest(tables, run.manifest(ref))
        world_module.validate_world(tables)
        rows.append({"seed": ref.seed, "family": ref.family, "tables": len(tables),
                     "rows": int(sum(len(frame) for frame in tables.values()))})
    return StageOutput(
        metrics={"validate.worlds_valid": _count(len(rows), "worlds that passed validation")},
        tables={"validate.worlds": rows},
        inputs=_world_inputs(run, run.all_worlds),
    )


def _load_world_module() -> Any:
    spec = importlib.util.spec_from_file_location("load_world", REPO / "db" / "load_world.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_mysql() -> None:
    settings = db_settings()
    if not mysql_reachable(settings):
        raise PipelineError(
            f"MySQL is not reachable at {settings.host}:{settings.port}; start it (make up) "
            "or point BNPL_DB_HOST/BNPL_DB_PORT at a disposable database"
        )


def stage_load(run: Run) -> StageOutput:
    """Load the run's database world into the configured MySQL database."""
    ref = run.database_world
    if ref is None:
        return StageOutput(notes=["no world is loaded into MySQL in this run"])
    _require_mysql()
    counts = _load_world_module().load_world(run.world_dir(ref))
    return StageOutput(
        metrics={"load.rows": _count(sum(counts.values()),
                                     f"rows loaded into MySQL, world {ref.name}")},
        tables={"load.tables": [{"table": name, "rows": int(rows)}
                                for name, rows in sorted(counts.items())]},
        inputs=_world_inputs(run, [ref]),
    )


def context_of(run: Run, ref: WorldRef) -> pd.DataFrame:
    """The world's as-of context, built once per process."""
    cache = run.memory.setdefault("context", {})
    if ref not in cache:
        build = entry("context", "core.asof", "build_context")
        cache[ref] = build(run.tables(ref))
    return cache[ref]


def stage_context(run: Run) -> StageOutput:
    """Build the as-of context of every world."""
    rows = []
    for ref in run.all_worlds:
        frame = context_of(run, ref)
        rows.append({"seed": ref.seed, "family": ref.family, "rows": len(frame),
                     "columns": len(frame.columns)})
    return StageOutput(tables={"context.worlds": rows},
                       inputs=_world_inputs(run, run.all_worlds) | _repo_inputs("core/asof.py"))


def pool_over_seeds(per_seed: Mapping[int, Metric], what: str) -> Metric:
    """One metric from per-seed metrics: pooled numerator over denominator, or the mean.

    The per-seed values become the metric's seed spread. All seeds must share
    unit, population and window.
    """
    metrics = list(per_seed.values())
    first = metrics[0]
    identity = (first.unit, first.population, first.window)
    for item in metrics:
        if (item.unit, item.population, item.window) != identity:
            raise PipelineError(f"{what}: seeds disagree on unit, population or window")
    missing = sorted(seed for seed, item in per_seed.items() if not item.evaluated)
    if missing:
        return Metric.not_evaluated(unit=first.unit, population=first.population,
                                    window=first.window,
                                    reason=f"not evaluated on seeds {missing}",
                                    numerator=None if first.numerator is None else 0,
                                    denominator=None if first.numerator is None else 0)
    spread = SeedSpread({seed: item.value for seed, item in per_seed.items()})
    if first.numerator is not None:
        numerator = sum(item.numerator for item in metrics)
        denominator = sum(item.denominator for item in metrics)
        if denominator == 0:
            return Metric.not_evaluated(unit=first.unit, population=first.population,
                                        window=first.window, reason="zero denominator",
                                        numerator=numerator, denominator=0)
        return Metric.from_ratio(numerator, denominator, population=first.population,
                                 window=first.window, unit=first.unit, seeds=spread,
                                 note=first.note)
    return Metric(value=spread.mean, unit=first.unit, population=first.population,
                  window=first.window, seeds=spread, note=first.note)


def stage_fit(run: Run) -> StageOutput:
    """Fit the classifiers of each seed on the baseline world's pre-test history."""
    fit = entry("fit", "model.train", "fit")
    per_key: dict[str, dict[int, Metric]] = {}
    outputs = []
    for seed in run.fit_seeds:
        ref = WorldRef(seed, BASELINE)
        out_dir = run.stage_dir("fit") / str(seed)
        result = fit(run.tables(ref), context_of(run, ref), run.protocol, out_dir)
        run.memory.setdefault("scorers", {})[seed] = result.scorers
        if seed in run.seeds:  # published metrics cover the evaluation seeds only
            for key, item in result.metrics.items():
                per_key.setdefault(key, {})[seed] = item
        outputs.append(out_dir)
    metrics = {f"fit.{key}": pool_over_seeds(items, key) for key, items in per_key.items()}
    return StageOutput(metrics=metrics, outputs=outputs,
                       inputs=_world_inputs(run, [WorldRef(s, BASELINE) for s in run.fit_seeds]))


def stage_tune(run: Run) -> StageOutput:
    """Choose each policy's thresholds on the validation window through the replay."""
    tune = entry("tune", "queue_sim.replay", "tune")
    output = tune(run)
    return output


REVIEW_DECISIONS = "review_decisions.pkl"


def stage_replay(run: Run) -> StageOutput:
    """Replay every policy on every world at every capacity level.

    Also keeps, per world, the incumbent's review decisions at base capacity
    (with the context rows and completed checks behind them) in
    ``worlds/<seed>-<family>/review_decisions.pkl`` for case selection.
    """
    replay = entry("replay", "queue_sim.replay", "replay")
    decisions = entry("replay", "queue_sim.replay", "review_decisions")
    output = replay(run)
    if "replay.outcomes" not in output.tables:
        raise PipelineError("replay: no replay.outcomes table")
    for ref in run.all_worlds:
        path = run.world_dir(ref) / REVIEW_DECISIONS
        frame = decisions(run, ref)
        if not isinstance(frame, pd.DataFrame):
            raise PipelineError(f"replay: review decisions of {ref.name} are not a data frame")
        frame.to_pickle(path)
        output.outputs.append(path)
    return output


def review_decisions(run: Run, ref: WorldRef) -> pd.DataFrame:
    """The incumbent's review decisions on one world, as the replay stage kept them."""
    path = run.world_dir(ref) / REVIEW_DECISIONS
    if not path.exists():
        raise PipelineError(f"no review decisions for {ref.name}; run the replay stage")
    return pd.read_pickle(path)


def stage_alerts(run: Run) -> StageOutput:
    """Write the incumbent policy's routing decisions on the database world to MySQL."""
    ref = run.database_world
    if ref is None:
        return StageOutput(notes=["no world is loaded into MySQL in this run"])
    routing = entry("alerts", "queue_sim.replay", "routing_frame")
    frame = routing(run, ref)
    _require_mysql()
    written = write_alerts(frame)
    return StageOutput(
        metrics={"alerts.rows": _count(written, f"routing decisions, world {ref.name}")},
        inputs=_repo_inputs("db/policy_tables.sql"),
    )


ALERT_COLUMNS = ("alert_id", "order_id", "user_id", "ts", "score", "band", "fired_rules",
                 "policy", "policy_version")


def write_alerts(frame: pd.DataFrame, settings=None) -> int:
    """Recreate MySQL table ``alerts`` (``db/policy_tables.sql``) and insert ``frame``.

    ``fired_rules`` is a list of rule ids per row, stored as JSON. Runs after
    the world is loaded: the table refers to its orders and accounts.
    """
    import pymysql

    missing = [column for column in ALERT_COLUMNS if column not in frame.columns]
    if missing:
        raise PipelineError(f"routing frame lacks columns {missing}")
    rows = [
        (
            str(row.alert_id), int(row.order_id), int(row.user_id),
            pd.Timestamp(row.ts).strftime(world_module.TS_FORMAT), float(row.score), str(row.band),
            json.dumps(list(row.fired_rules)), str(row.policy), str(row.policy_version),
        )
        for row in frame.loc[:, list(ALERT_COLUMNS)].itertuples(index=False)
    ]
    loader = _load_world_module()
    connection = pymysql.connect(autocommit=False, **(settings or db_settings()).pymysql_kwargs())
    try:
        with connection.cursor() as cursor:
            for statement in loader.statements(POLICY_TABLES_SQL.read_text()):
                cursor.execute(statement)
            cursor.executemany(
                f"INSERT INTO alerts ({', '.join(ALERT_COLUMNS)}) "
                f"VALUES ({', '.join(['%s'] * len(ALERT_COLUMNS))})",
                rows,
            )
            cursor.execute("SELECT COUNT(*) FROM alerts")
            count = cursor.fetchone()[0]
        if count != len(rows):
            raise PipelineError(f"alerts: loaded {count} of {len(rows)} rows")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return count


# How evaluate turns the replay's outcome rows (one per seed, family, policy
# and capacity level) into metrics: name -> (unit, numerator column,
# denominator column or None, population). Ratios are pooled over seeds for
# the value and computed per seed for the spread.
OUTCOME_METRICS: dict[str, tuple[str, str, str | None, str]] = {
    "net_contribution": ("cents", "net_cents", None, "platform net cash, test-window orders"),
    "loss_bps_of_gmv": ("bps", "loss_cents", "gmv_cents",
                        "fraud and abuse loss per 10,000 of GMV, test-window orders"),
    "legitimate_held_per_10k": ("bps", "legit_held", "legit_orders",
                                "legitimate orders held per 10,000 legitimate orders"),
    "legitimate_declined_per_10k": ("bps", "legit_declined", "legit_orders",
                                    "legitimate orders declined per 10,000 legitimate orders"),
    "review_minutes_used_share": ("share", "review_minutes_used", "review_minutes_available",
                                  "review minutes used out of minutes available"),
    "decided_after_shipping": ("count", "decided_after_shipping", None,
                               "orders decided after they shipped"),
}
OUTCOME_KEYS = ("seed", "family", "capacity", "policy")


def _per_seed(rows: pd.DataFrame, numerator: str, denominator: str | None,
              unit: str) -> dict[int, float | int]:
    values = {}
    for row in rows.itertuples(index=False):
        top = getattr(row, numerator)
        if denominator is None:
            values[int(row.seed)] = int(top) if float(top).is_integer() else float(top)
        else:
            bottom = getattr(row, denominator)
            if bottom == 0:
                raise PipelineError(f"evaluate: zero {denominator} on seed {row.seed}")
            values[int(row.seed)] = (10_000 if unit == "bps" else 1) * top / bottom
    return values


def evaluate_outcomes(outcomes: list[dict[str, Any]], seeds: tuple[int, ...],
                      window: str = "test") -> tuple[dict[str, Metric], list[dict[str, Any]]]:
    """Per-seed paired metrics from the replay's outcome rows.

    For each family, capacity level and policy: each metric pooled over seeds
    with its per-seed spread, and the per-seed paired difference against
    approve-all and against the incumbent rules. Differences of shares and
    rates are in basis points. Every (family, capacity, policy) must cover
    exactly the run's seeds.
    """
    frame = pd.DataFrame(outcomes)
    needed = set(OUTCOME_KEYS) | {c for _, n, d, _ in OUTCOME_METRICS.values() for c in (n, d) if c}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise PipelineError(f"evaluate: the replay's outcome rows lack columns {missing}")
    if frame.duplicated(list(OUTCOME_KEYS)).any():
        raise PipelineError("evaluate: repeated outcome rows for a seed, family, capacity, policy")
    metrics: dict[str, Metric] = {}
    table_rows: list[dict[str, Any]] = []
    for (family, capacity), group in frame.groupby(["family", "capacity"], sort=True):
        policies = sorted(group["policy"].unique())
        values: dict[str, dict[str, dict[int, float | int]]] = {}
        for policy in policies:
            rows = group[group["policy"].eq(policy)]
            if tuple(sorted(rows["seed"].astype(int))) != tuple(sorted(seeds)):
                raise PipelineError(
                    f"evaluate: {family}/{capacity}/{policy} covers seeds "
                    f"{sorted(rows['seed'].astype(int))}, not {sorted(seeds)}"
                )
            values[policy] = {}
            summary_row: dict[str, Any] = {"family": family, "capacity": capacity,
                                           "policy": policy, "seeds": len(seeds)}
            for name, (unit, numerator, denominator, population) in OUTCOME_METRICS.items():
                per_seed = _per_seed(rows, numerator, denominator, unit)
                values[policy][name] = per_seed
                spread = SeedSpread(per_seed)
                key = f"evaluate.{name}.{family}.{capacity}.{policy}"
                if denominator is None:
                    item = Metric(value=spread.mean, unit=unit, population=population,
                                  window=window, seeds=spread)
                else:
                    item = Metric.from_ratio(
                        int(rows[numerator].sum()), int(rows[denominator].sum()),
                        population=population, window=window, unit=unit, seeds=spread,
                    )
                metrics[key] = item
                summary_row[name] = item.value
        for policy in policies:
            for reference in REFERENCES:
                if reference not in values or reference == policy:
                    continue
                for name, (unit, _, _, population) in OUTCOME_METRICS.items():
                    spread = paired_seed_differences(values[policy][name], values[reference][name])
                    difference_unit = "bps" if unit in ("share", "rate") else unit
                    if unit in ("share", "rate"):
                        spread = SeedSpread({s: 10_000 * v for s, v in spread.per_seed.items()})
                    key = f"evaluate.{name}.vs_{reference}.{family}.{capacity}.{policy}"
                    metrics[key] = Metric(
                        value=spread.mean, unit=difference_unit, window=window, seeds=spread,
                        population=f"{population}: {policy} minus {reference}, paired by seed",
                    )
        table_rows += [row for row in _summary_rows(metrics, family, capacity, policies)]
    return metrics, table_rows


def _summary_rows(metrics: dict[str, Metric], family: str, capacity: str,
                  policies: list[str]) -> list[dict[str, Any]]:
    rows = []
    for policy in policies:
        row: dict[str, Any] = {"family": family, "capacity": capacity, "policy": policy}
        for name in OUTCOME_METRICS:
            row[name] = metrics[f"evaluate.{name}.{family}.{capacity}.{policy}"].value
            for reference in REFERENCES:
                key = f"evaluate.{name}.vs_{reference}.{family}.{capacity}.{policy}"
                item = metrics.get(key)
                row[f"{name}_vs_{reference}"] = None if item is None else item.value
                row[f"{name}_vs_{reference}_positive_seeds"] = (
                    None if item is None else item.seeds.sign_count.positive
                )
        rows.append(row)
    return rows


def stage_evaluate(run: Run) -> StageOutput:
    """Pair the replay's outcomes by seed: each policy against approve-all and the incumbent."""
    replay_file = run.results_dir / "replay.json"
    if not replay_file.exists():
        raise PipelineError("evaluate: run the replay stage first")
    outcomes = read_result(replay_file).tables.get("replay.outcomes")
    if outcomes is None:
        raise PipelineError("evaluate: the replay result has no replay.outcomes table")
    metrics, rows = evaluate_outcomes(outcomes, run.seeds)
    return StageOutput(metrics=metrics, tables={"evaluate.policies": rows},
                       inputs={"results/replay.json": replay_file})


def stage_llm(run: Run) -> StageOutput:
    """Replay the LLM benchmarks offline from their stored responses."""
    replay = entry("llm", "llm.eval.history", "replay")
    output = replay(run.stage_dir("llm"))
    if isinstance(output, StageOutput):
        return output
    return StageOutput(metrics=dict(output))


STAGES = (
    Stage("world", stage_world, ("world", "protocol"), "generate every world of the run"),
    Stage("validate", stage_validate, ("world",), "check worlds against the contract"),
    Stage("load", stage_load, ("world",), "load the database world into MySQL"),
    Stage("context", stage_context, ("world", "features"), "build the as-of context"),
    Stage("fit", stage_fit, ("world", "features", "protocol"), "fit classifiers per seed"),
    Stage("tune", stage_tune, ("world", "features", "policy", "protocol"),
          "choose thresholds on validation"),
    Stage("replay", stage_replay, ("world", "features", "policy", "protocol"),
          "replay every policy on every world"),
    Stage("alerts", stage_alerts, ("world", "policy"), "routing decisions into MySQL"),
    Stage("evaluate", stage_evaluate, ("world", "features", "policy", "protocol"),
          "paired comparisons across seeds"),
    Stage("llm", stage_llm, ("benchmark", "policy"), "offline LLM benchmark replay"),
)
STAGE_NAMES = tuple(stage.name for stage in STAGES)


# ---------------------------------------------------------------- versions and lineage
def _short(digest: Any) -> str:
    return digest.hexdigest()[:16]


def files_version(paths: tuple[str, ...], root: Path = REPO) -> str:
    """A short hash over the named files' paths and bytes (absent files count as absent)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode() + b"\0")
        target = root / path
        digest.update(target.read_bytes() if target.exists() else b"<absent>")
        digest.update(b"\0")
    return _short(digest)


def world_version(run: Run) -> str:
    """A short hash over the run's world manifests (the code commit left out)."""
    digest = hashlib.sha256()
    for ref in run.all_worlds:
        manifest = dict(run.manifest(ref))
        manifest.pop("code_commit", None)
        digest.update(canonical_json(manifest).encode())
    return _short(digest)


def versions(run: Run, names: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for name in names:
        if name == "world":
            out[name] = world_version(run)
        elif name == "benchmark":
            manifests = sorted(
                path.relative_to(REPO).as_posix()
                for path in (REPO / "llm" / "eval" / "benchmarks").glob("*/manifest.json")
            )
            out[name] = files_version(tuple(manifests)) if manifests else "none"
        else:
            out[name] = files_version(VERSION_FILES[name])
    return out


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def code_identity() -> dict[str, Any]:
    """The commit the run used, and whether the working tree had changes."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    dirty = None if status is None else bool(status)
    return {"commit": _git("rev-parse", "HEAD"), "dirty": dirty}


def _hash_outputs(paths: list[Path], run: Run) -> dict[str, str]:
    hashes = {}
    for path in paths:
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for item in files:
            try:
                name = item.relative_to(run.directory).as_posix()
            except ValueError:
                name = item.as_posix()
            hashes[name] = file_sha256(item)
    return hashes


class Lineage:
    """``runs/<name>/lineage.json``: what each stage read and wrote, and when."""

    def __init__(self, run: Run):
        self.path = run.directory / "lineage.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"run": run.name, "stages": {}}
        self.data["seeds"] = list(run.seeds)
        self.data["families"] = list(run.families)
        self.data["scale"] = run.scale
        self.data["results_dir"] = str(run.results_dir)
        self.data["python"] = platform.python_version()

    def record(self, stage: str, entry_: dict[str, Any]) -> None:
        self.data["stages"][stage] = entry_
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(canonical_json(self.data))


# ---------------------------------------------------------------- execution
def select(start: str | None = None, until: str | None = None) -> list[Stage]:
    names = list(STAGE_NAMES)
    for value in (start, until):
        if value is not None and value not in names:
            raise PipelineError(f"unknown stage {value!r}; stages: {', '.join(names)}")
    first = names.index(start) if start else 0
    last = names.index(until) if until else len(names) - 1
    if first > last:
        raise PipelineError(f"--from {start} comes after --until {until}")
    return list(STAGES[first:last + 1])


def run_stage(run: Run, stage: Stage, lineage: Lineage) -> Path:
    started = dt.datetime.now(dt.UTC)
    clock = time.monotonic()
    output = stage.function(run)
    result = StageResult(
        stage=stage.name,
        versions=versions(run, stage.versions),
        inputs={name: file_sha256(path) for name, path in output.inputs.items()},
        metrics=output.metrics,
        tables=output.tables,
        notes=output.notes,
    )
    path = write_result(result, run.results_dir)
    lineage.record(stage.name, {
        "started_at": started.isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - clock, 3),
        "code": code_identity(),
        "versions": result.versions,
        "inputs": result.inputs,
        "outputs": _hash_outputs(output.outputs, run),
        "result": {"path": str(path), "sha256": file_sha256(path)},
    })
    return path


def summarize(run: Run) -> Path:
    """Assemble ``summary.json`` from one result file per stage; refuse an incomplete set."""
    missing = [name for name in STAGE_NAMES if not (run.results_dir / f"{name}.json").exists()]
    if missing:
        raise PipelineError(f"no results from stages {missing}; the run is not complete")
    results = [read_result(run.results_dir / f"{name}.json") for name in STAGE_NAMES]
    return write_summary(results, run.results_dir)


def render_documents(run: Run) -> list[Path]:
    from report.render import Sources, render_all

    sources = Sources.from_repo(read_summary(run.results_dir / SUMMARY_FILE))
    return render_all(sources, run.docs_dir)


def execute(run: Run, stages: list[Stage], *, log: Callable[[str], None] = print) -> None:
    """Run ``stages`` in order; assemble the summary and render when the last stage ran."""
    from report import figures

    figures.pin()
    run.directory.mkdir(parents=True, exist_ok=True)
    lineage = Lineage(run)
    for stage in stages:
        clock = time.monotonic()
        log(f"[{stage.name}] {stage.description}")
        path = run_stage(run, stage, lineage)
        log(f"[{stage.name}] wrote {path} ({time.monotonic() - clock:.1f} s)")
    if stages and stages[-1].name == STAGE_NAMES[-1]:
        log(f"[summary] wrote {summarize(run)}")
        for path in render_documents(run):
            log(f"[render] wrote {path}")


def preflight(profile: Profile, run: Run) -> None:
    """Checks before anything is written."""
    if profile.name == "final":
        protocol_module.require_freeze()
        if code_identity()["dirty"]:
            raise PipelineError("the final run needs a clean working tree (commit first)")
    for ref in run.all_worlds:
        protocol_module.check_seed(ref.seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python pipeline.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stages", help="list the stages in order")
    runner = commands.add_parser("run", help="run the pipeline")
    runner.add_argument("--profile", choices=sorted(PROFILES), default="dev")
    runner.add_argument("--name", help="run directory name under runs/ (default: the profile)")
    runner.add_argument("--seeds", type=int, nargs="+", help="override the profile's seeds")
    runner.add_argument("--families", nargs="+", help="override the profile's families")
    runner.add_argument("--world", type=Path, help="use this world directory instead of generating")
    runner.add_argument("--scale", type=float, help="override the profile's world scale")
    runner.add_argument("--from", dest="start", help="first stage to run")
    runner.add_argument("--until", help="last stage to run")
    args = parser.parse_args(argv)
    if args.command == "stages":
        for stage in STAGES:
            print(f"{stage.name:9s} {stage.description}")
        return 0
    profile = PROFILES[args.profile]
    try:
        run = make_run(profile, name=args.name, seeds=args.seeds, families=args.families,
                       world=args.world, scale=args.scale)
        preflight(profile, run)
        execute(run, select(args.start, args.until))
    except (PipelineError, protocol_module.FreezeError, world_module.WorldError) as error:
        print(f"pipeline stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
