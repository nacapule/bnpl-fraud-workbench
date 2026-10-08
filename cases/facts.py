"""Each case file's facts, from one pipeline run's canonical world.

:func:`load_run` reads what the run kept: the world's tables, the incumbent's review
decisions, the context rows its checkout routing read and every test-window order's fate
(``pipeline.KEPT_FRAMES``), and the incumbent's tuned thresholds, each checked against the
run's lineage (the replay stage's world, tuning result and kept files) and the world's
manifest. :func:`build` writes, per file of the protocol's ``cases.files``, and per alert:

* ``publication``: order id, alert id, policy version, selection hash, tier and the
  candidate counts (:class:`cases.rule.Pick`);
* ``decision``: the checkout, when the evidence was assembled (``evidence_at``), when the
  analyst started (reviewed alerts), when the action was taken and the recorded action
  (the first disposition of a reviewed alert, the band of a checkout alert); and
  ``same_day_orders``, the account's orders checked out earlier on the evidence's replay
  day with the route the policy gave each: known when the evidence was assembled, but not
  yet in its outcome-derived columns, which the replay rebuilds from the policy's
  decisions before the start of each day (they count such an order as approved);
* ``evidence``: the saved row the replay decided on, as it had it, with the §6.2
  conditions that hold on it and the rule score (computed from the row alone);
  ``routing``: the same for the checkout row; ``reviewer``: for a reviewed alert, how the
  decision table reads the row (it must give the recorded disposition);
* ``policy_view``: what FP-2 supports on that evidence, as the memo referee states it for
  the memo drafter's packet of the same row (``llm.referee.view``): the decision-table row,
  the standard, also-permitted (together: ``acceptable``) and prohibited dispositions, the
  checks still required and the clause each rests on (for a reviewed alert it must agree
  with the reviewer's reading);
* ``later``: the review's later checks and final disposition, the incumbent's fate, the
  order's realized events and cash under the incumbent to the end of observation, the
  world's approve-all events and cash where they differ, the adjudicated label, and the
  account's later orders under the incumbent;
* ``latent``: the simulation's truth for the order, its account and episode, a diagnostic
  kept apart from everything above.

A missing slot is written with its counts (``selected`` false). Times are
``YYYY-MM-DD HH:MM:SS`` strings, money integer cents, other numbers rounded to
:data:`DECIMALS` places, keys sorted, so a rebuild writes the same bytes.
:func:`memo_inputs` gives the selected alerts' saved rows to the memo drafter's packets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cases.rule import Pick, Rule, RuleError, load_rule, population, select
from core import actions, asof, config, evidence, ledger
from core import protocol as protocol_module
from core import world as world_module
from core.results import canonical_json, file_sha256
from llm import referee
from llm.packet import build_packets
from pipeline import KEPT_FRAMES, manifest_identity
from queue_sim import stage
from rules import engine

DECIMALS = 4
ROW_COLUMNS = (*asof.KEY_COLUMNS, *asof.COLUMN_NAMES)
KEPT = {"reviews": KEPT_FRAMES["review_decisions"], "checkout_rows": KEPT_FRAMES["checkout_rows"],
        "fates": KEPT_FRAMES["incumbent_fates"]}
ILLUSTRATION = ("seed {seed} is a development seed: the case illustrates how the operating "
                "policy behaves on one world; it is not evidence for the policy comparison")
EVENT_RANK = {name: rank for rank, name in enumerate((
    "checkout", "payment", "shipped", "delivered", "payment reversed", "dispute opened",
    "dispute resolved", "victim report", "written off"))}


# --------------------------------------------------------------------- the run


@dataclass
class Run:
    """What the case files read from one pipeline run (:func:`load_run`)."""

    rule: Rule
    protocol: protocol_module.Protocol
    world_dir: Path
    manifest: Mapping[str, Any]
    fates: pd.DataFrame
    reviews: pd.DataFrame
    checkout_rows: pd.DataFrame
    policy_version: str
    review_threshold: float | None
    decline_threshold: float | None
    run_dir: Path
    results_dir: Path
    _tables: dict[str, pd.DataFrame] | None = field(default=None, repr=False)

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        if self._tables is None:
            self._tables = world_module.read_world(self.world_dir)
        return self._tables

    @property
    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        window = self.protocol.windows[self.rule.replay["window"]]
        return window.start, window.end

    def alerts(self) -> pd.DataFrame:
        return population(self.fates, self.checkout_rows, self.policy_version)

    def picks(self) -> dict[str, Pick]:
        latent = (self.tables["latent_orders"] if self._tables is not None else
                  world_module.read_world(self.world_dir, ["latent_orders"])["latent_orders"])
        return select(self.rule, self.alerts(), self.reviews, self.checkout_rows, latent)


def _check_cell(rule: Rule, protocol: protocol_module.Protocol) -> None:
    """The kept frames are the incumbent's main run (base level, current layout, main
    variant, test window): refuse a case cell they are not."""
    policy_cfg = config.load("policy")
    staffing = stage.staffing(policy_cfg)
    try:
        protocol_module.check_case_replay(protocol, stage.VARIANTS,
                                          {s.level for s in staffing},
                                          {s.layout for s in staffing})
    except protocol_module.ProtocolError as error:
        raise RuleError(str(error)) from error
    replay = rule.replay
    kept = {"policy": stage.INCUMBENT, "capacity_level": stage.base_staffing(policy_cfg).level,
            "layout": policy_cfg["roster"]["layout"], "window": "test"}
    if wrong := {key: replay[key] for key, value in kept.items() if replay[key] != value}:
        raise RuleError(f"the replay keeps frames only for {kept}; the case cell names {wrong}")


def _check_lineage(run_dir: Path, results_dir: Path, world: str,
                   manifest: Mapping[str, Any]) -> None:
    """The kept files, the tuning result and the world are those the run's replay stage
    recorded, and the world's tables match its manifest."""
    replay = json.loads((run_dir / "lineage.json").read_text())["stages"]["replay"]
    expected = {f"worlds/{world}/{name}": replay["outputs"].get(f"worlds/{world}/{name}")
                for name in KEPT.values()}
    problems = [path for path, digest in expected.items()
                if digest is None or file_sha256(run_dir / path) != digest]
    if file_sha256(results_dir / "tune.json") != replay["inputs"].get("results/tune.json"):
        problems.append(str(results_dir / "tune.json"))
    if manifest_identity(manifest) != replay["inputs"].get(f"world:{world}"):
        problems.append(f"world {world}")
    world_dir = run_dir / "worlds" / world
    problems += [f"{name}.csv" for name, entry in manifest["tables"].items()
                 if file_sha256(world_dir / f"{name}.csv") != entry["sha256"]]
    if problems:
        raise RuleError(f"not what the run's replay used or its manifest names: {problems}")


def load_run(run_dir: Path, protocol: protocol_module.Protocol | None = None,
             results_dir: Path | None = None) -> Run:
    """The canonical world of a pipeline run directory (``runs/<name>``) and what its
    replay kept, for the protocol's case cell. ``results_dir`` is where the run wrote its
    results (default: ``<run>/results``, or for a run that published them the directory
    its lineage names)."""
    run_dir = Path(run_dir)
    protocol = protocol or protocol_module.load_protocol()
    rule = load_rule(protocol.raw)
    if rule.seed != protocol.canonical_seed:
        raise RuleError("the case world must be the canonical seed")
    _check_cell(rule, protocol)
    world_dir = run_dir / "worlds" / rule.world
    missing = [name for name in ("manifest.json", *KEPT.values())
               if not (world_dir / name).exists()]
    if not (run_dir / "lineage.json").exists():
        missing.append("lineage.json")
    if missing:
        raise RuleError(f"{run_dir} lacks {missing}: run the pipeline's replay stage")
    if results_dir is None:
        results_dir = run_dir / "results"
        if not (results_dir / "tune.json").exists():
            results_dir = Path(json.loads((run_dir / "lineage.json").read_text())
                               .get("results_dir") or results_dir)
    results_dir = Path(results_dir)
    manifest = json.loads((world_dir / "manifest.json").read_text())
    _check_lineage(run_dir, results_dir, rule.world, manifest)
    frames = {key: pd.read_pickle(world_dir / name) for key, name in KEPT.items()}
    tune = json.loads((results_dir / "tune.json").read_text())
    chosen = [row for row in tune["tables"]["tune.chosen"]
              if row["seed"] == rule.seed and row["policy"] == rule.replay["policy"]]
    if len(chosen) != 1 or chosen[0]["chosen_version"] is None:
        raise RuleError(f"the run has no tuned {rule.replay['policy']} for seed {rule.seed}")
    version = str(chosen[0]["chosen_version"])
    recorded = set(frames["reviews"]["policy_version"].dropna())
    if recorded and recorded != {version}:
        raise RuleError(f"review decisions name policy versions {sorted(recorded)}, the "
                        f"tuned incumbent is {version}")
    return Run(rule=rule, protocol=protocol, world_dir=world_dir, manifest=manifest,
               fates=frames["fates"], reviews=frames["reviews"],
               checkout_rows=frames["checkout_rows"], policy_version=version,
               review_threshold=_number(chosen[0]["review_threshold"]),
               decline_threshold=_number(chosen[0]["decline_threshold"]), run_dir=run_dir,
               results_dir=results_dir)


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


# --------------------------------------------------------------------- plain values


def plain(value: Any) -> Any:
    """A JSON value: times as text, numpy scalars as Python, floats rounded, NaN null."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [plain(item) for item in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp | np.datetime64):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(value, "isoformat") and not isinstance(value, str):  # datetime
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else round(float(value), DECIMALS)
    if pd.isna(value):
        return None
    return str(value)


def dumps(facts: Mapping[str, Any]) -> str:
    """The file's text: indented, keys sorted, no NaN."""
    return canonical_json(plain(facts))


def _row(frame: pd.DataFrame, key: int, column: str = "order_id") -> dict[str, Any]:
    rows = frame.loc[frame[column] == key]
    if len(rows) != 1:
        raise RuleError(f"{column} {key}: {len(rows)} rows where one was expected")
    return rows.iloc[0].to_dict()


def _context(row: Mapping[str, Any]) -> dict[str, Any]:
    return {column: row[column] for column in ROW_COLUMNS}


def _rules(row: Mapping[str, Any]) -> dict[str, Any]:
    """The §6.2 conditions that hold on a context row and the rule score (FP-2 §4.1)."""
    frame = pd.DataFrame([dict(row)])
    held = evidence.conditions(frame)
    return {"rules_held": [rule for rule in evidence.RULE_FAMILY if bool(held[rule].iloc[0])],
            "rule_score": float(engine.score(frame)[0])}


def policy_view(tables: Mapping[str, pd.DataFrame], row: Mapping[str, Any]) -> dict[str, Any]:
    """FP-2's answer on a saved row with no checks completed: the memo referee's view of
    the packet the memo drafter gets for it (``llm.packet.build_packets``)."""
    frame = pd.DataFrame([{column: row[column] for column in ROW_COLUMNS}])
    (packet,) = build_packets(tables, frame).values()
    view = referee.view(packet)
    return {**view.as_dict(), "acceptable": sorted(view.standard | view.permitted)}


def _reviewer(row: Mapping[str, Any]) -> dict[str, Any]:
    """How the decision table (FP-2 §6.6) reads the row with no checks completed."""
    found = evidence.classify(row, [])
    allowed = evidence.permitted_actions(found)
    (standard,) = allowed.standard
    return {"families_present": sorted(str(f) for f in found.families_present),
            "decision_table_row": allowed.row, "standard_disposition": standard,
            "clause": allowed.clauses[standard],
            "checks_required": sorted(str(c) for c in allowed.required_checks),
            "household_exceptions": [c.clause for c in found.household_exceptions],
            "benign": [c.clause for c in found.benign],
            "settlements": [c.clause for c in found.settlements]}


# --------------------------------------------------------------------- what happened


def _order_tables(tables: Mapping[str, pd.DataFrame], order_id: int) -> dict[str, pd.DataFrame]:
    """The rows of one order in the order-bound tables."""
    out = {name: tables[name].loc[tables[name]["order_id"] == order_id]
           for name in ("order_attempts", "plans", "fulfilments", "deliveries",
                        "dispute_openings", "victim_reports", "cash_events")}
    plans = out["plans"]["plan_id"].to_numpy()
    for name in ("payment_attempts", "payment_reversals", "plan_writeoffs"):
        out[name] = tables[name].loc[tables[name]["plan_id"].isin(plans)]
    disputes = out["dispute_openings"]["dispute_id"].to_numpy()
    out["dispute_resolutions"] = tables["dispute_resolutions"].loc[
        tables["dispute_resolutions"]["dispute_id"].isin(disputes)]
    return out


def events(tables: Mapping[str, pd.DataFrame], order_id: int,
           until: Any = None) -> list[dict[str, Any]]:
    """The order's dated observations in ``tables`` (the world's, or a policy's realized
    ones) known by ``until`` (all when None): ``event``, ``detail``, ``occurred_at``,
    ``known_at``, ``amount_cents``."""
    t = _order_tables(tables, order_id)
    rows: list[dict[str, Any]] = []

    def add(frame: pd.DataFrame, event: str, detail: Any = None,
            amount: str | None = None) -> None:
        for record in frame.to_dict("records"):
            rows.append({"event": event,
                         "detail": detail(record) if callable(detail) else detail,
                         "occurred_at": record["occurred_at"], "known_at": record["known_at"],
                         "amount_cents": None if amount is None else int(record[amount])})

    add(t["order_attempts"], "checkout", lambda r: f"processor {r['processor_result']}",
        "amount_cents")
    add(t["payment_attempts"], "payment", lambda r: (
        "down payment" if int(r["seq"]) == 0 else f"installment {int(r['seq'])}")
        + f", attempt {int(r['attempt_no'])}: {r['result']}", "amount_cents")
    add(t["fulfilments"], "shipped")
    add(t["deliveries"], "delivered")
    add(t["payment_reversals"], "payment reversed", lambda r: str(r["reason"]), "amount_cents")
    add(t["dispute_openings"], "dispute opened", lambda r: str(r["reason"]), "amount_cents")
    add(t["dispute_resolutions"], "dispute resolved", lambda r: str(r["outcome"]))
    add(t["victim_reports"], "victim report")
    add(t["plan_writeoffs"], "written off", None, "outstanding_cents")
    if until is not None:
        rows = [r for r in rows if pd.Timestamp(r["known_at"]) <= pd.Timestamp(until)]
    rows.sort(key=lambda r: (pd.Timestamp(r["known_at"]), pd.Timestamp(r["occurred_at"]),
                             EVENT_RANK[r["event"]], str(r["detail"])))
    return rows


def cash(ledger_events: pd.DataFrame, order_id: int, until: Any = None) -> dict[str, Any]:
    """The order's cash events in a ledger known by ``until`` (all when None), as the
    results count cash, with their sum by kind and in total."""
    frame = ledger_events.loc[ledger_events["order_id"] == order_id]
    if until is not None:
        frame = frame.loc[frame["known_at"] <= pd.Timestamp(until)]
    frame = frame.sort_values(["occurred_at", "known_at", "event_id"], kind="stable")
    listed = [{"event_id": int(r["event_id"]), "kind": str(r["kind"]),
               "amount_cents": int(r["amount_cents"]), "cause": str(r["cause"]),
               "occurred_at": r["occurred_at"], "known_at": r["known_at"]}
              for r in frame.to_dict("records")]
    by_kind: dict[str, int] = {}
    for item in listed:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + item["amount_cents"]
    return {"cash": listed, "cash_by_kind": by_kind,
            "net_cents": sum(item["amount_cents"] for item in listed)}


def _fate(fates: pd.DataFrame, order_id: int) -> dict[str, Any]:
    row = _row(fates, order_id)
    return {key: row[key] for key in actions.FATE_COLUMNS if key not in ("order_id", "user_id")}


def _label(labels: pd.DataFrame, order_id: int) -> dict[str, Any] | None:
    rows = labels.loc[labels["order_id"] == order_id]
    if not len(rows):
        return None
    row = rows.iloc[0]
    return {"label": int(row["label"]), "basis": str(row["basis"]),
            "label_known_at": row["label_known_at"]}


def _review(reviews: pd.DataFrame, order_id: int) -> dict[str, Any] | None:
    rows = reviews.loc[reviews["order_id"] == order_id]
    if not len(rows):
        return None
    row = rows.iloc[0]
    return {"taken_up_at": row["taken_up_at"], "decided_at": row["decided_at"],
            "first_disposition": row["disposition"], "final": row["final"],
            "checks": [{"check": str(check), "outcome": str(outcome), "completed_at": at}
                       for check, outcome, at in row["checks_later"]]}


def _latent(tables: Mapping[str, pd.DataFrame], order_id: int, user_id: int) -> dict[str, Any]:
    order = _row(tables["latent_orders"], order_id)
    account = _row(tables["latent_accounts"], user_id, "user_id")
    out: dict[str, Any] = {
        "order": {key: order[key] for key in ("pattern_id", "episode_id", "intent", "mimic")},
        "account": {key: account[key] for key in ("actor", "episode_id", "profile")},
        "episode": None}
    episode = order["episode_id"]
    if pd.notna(episode):
        row = _row(tables["latent_episodes"], int(episode), "episode_id")
        out["episode"] = {
            "episode_id": int(episode), "pattern_id": row["pattern_id"],
            "started_at": row["started_at"], "ended_at": row["ended_at"],
            "orders": int((tables["latent_orders"]["episode_id"] == episode).sum()),
            "accounts": int((tables["latent_accounts"]["episode_id"] == episode).sum())}
    return out


# --------------------------------------------------------------------- the facts


def _decision(run: Run, pick: Pick) -> tuple[dict[str, Any], dict[str, Any]]:
    """The decision block and the saved row it rests on (the protocol's ``facts`` rule)."""
    order = int(pick.order_id)
    fate = _row(run.fates, order)
    checkout_row = _row(run.checkout_rows, order)
    if pick.tier == "reviewed_alert":
        saved = _row(run.reviews, order)
        row = _context(saved)
        decision = {"kind": "review", "evidence_at": saved["decision_at"],
                    "analyst_started_at": saved["taken_up_at"], "action_at": saved["decided_at"],
                    "recorded_action": saved["disposition"]}
    else:
        row = _context(checkout_row)
        decision = {"kind": "checkout", "evidence_at": checkout_row["decision_at"],
                    "analyst_started_at": None, "action_at": checkout_row["decision_at"],
                    "recorded_action": fate["route"]}
    decision.update(checkout_at=fate["checkout_at"], band=fate["route"],
                    same_day_orders=same_day_orders(run.fates, order, decision["evidence_at"]))
    _check_times(order, decision, row)
    return decision, row


def same_day_orders(fates: pd.DataFrame, order: int, evidence_at: Any) -> list[dict[str, Any]]:
    """The order's account's other orders checked out on the replay day of ``evidence_at``
    before it, with their routes (the decisions the day's evidence rows do not reflect)."""
    at = pd.Timestamp(evidence_at)
    user = _row(fates, order)["user_id"]
    same = fates.loc[(fates["user_id"] == user) & (fates["order_id"] != order)
                     & (fates["checkout_at"] >= at.normalize()) & (fates["checkout_at"] < at)]
    return [{"order_id": int(r["order_id"]), "checkout_at": r["checkout_at"], "route": r["route"]}
            for r in same.sort_values(["checkout_at", "order_id"]).to_dict("records")]


def _check_times(order: int, decision: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    """The evidence is the row assembled at ``evidence_at``, between the checkout and the
    action; a reviewed alert's analyst starts before acting."""
    at = {key: pd.Timestamp(decision[key]) for key in ("checkout_at", "evidence_at", "action_at")}
    problems = []
    if pd.Timestamp(row["decision_at"]) != at["evidence_at"]:
        problems.append("the evidence row was not assembled at evidence_at")
    if not at["checkout_at"] <= at["evidence_at"] <= at["action_at"]:
        problems.append("evidence_at is not between the checkout and the action")
    started = decision["analyst_started_at"]
    if started is not None and pd.Timestamp(started) > at["action_at"]:
        problems.append("the analyst started after the action")
    if decision["kind"] == "checkout" and at["evidence_at"] != at["checkout_at"]:
        problems.append("a checkout alert's evidence is the checkout row")
    if problems:
        raise RuleError(f"order {order}: " + "; ".join(problems))


def _routing(run: Run, order: int, band: str) -> dict[str, Any]:
    """The checkout row's rules and score; the band must follow from the thresholds."""
    routing = _rules(_row(run.checkout_rows, order))
    score = routing["rule_score"]
    decline = run.decline_threshold is not None and score >= run.decline_threshold
    review = run.review_threshold is not None and score >= run.review_threshold
    expected = "auto_decline" if decline else "review" if review else "approve"
    if expected != band:
        raise RuleError(f"order {order}: a rule score of {score} routes to {expected}, the "
                        f"replay recorded {band}")
    return routing


def alert_facts(run: Run, pick: Pick, realized: Mapping[str, pd.DataFrame],
                labels: pd.DataFrame) -> dict[str, Any]:
    """One selected alert's facts (module docstring)."""
    tables = run.tables
    order = int(pick.order_id)
    decision, row = _decision(run, pick)
    facts: dict[str, Any] = {
        "selected": True, "slot": pick.slot,
        "publication": _publication(run, pick),
        "decision": decision,
        "evidence": {"row": row, **_rules(row)},
        "routing": _routing(run, order, decision["band"]),
    }
    facts["policy_view"] = policy_view(tables, row)
    if decision["kind"] == "review":
        facts["reviewer"] = _reviewer(row)
        reading = facts["reviewer"]
        if reading["standard_disposition"] != decision["recorded_action"]:
            raise RuleError(f"order {order}: the saved row gives "
                            f"{reading['standard_disposition']}, the replay recorded "
                            f"{decision['recorded_action']}")
        view = facts["policy_view"]
        if (view["row"], view["standard"]) != (reading["decision_table_row"],
                                               [reading["standard_disposition"]]):
            raise RuleError(f"order {order}: the packet's policy view differs from the "
                            "reviewer's reading of the same row")
    cut = run.protocol.observed_until
    incumbent = {"events": events(realized, order, cut),
                 **cash(realized["cash_events"], order, cut)}
    world_view = {"events": events(tables, order, cut), **cash(tables["cash_events"], order, cut)}
    approve_all = {"differs": (incumbent["events"] != world_view["events"]
                               or incumbent["cash"] != world_view["cash"]), **world_view}
    user = int(row["user_id"])
    fates = run.fates
    later_orders = fates.loc[(fates["user_id"] == user)
                             & (fates["checkout_at"] > decision["checkout_at"])]
    facts["later"] = {
        "review": _review(run.reviews, order),
        "fate": _fate(fates, order),
        "incumbent": incumbent,
        "approve_all": approve_all,
        "prevented_cents": incumbent["net_cents"] - world_view["net_cents"],
        "label": _label(labels, order),
        "account": {"later_orders": len(later_orders),
                    "later_routes": {str(k): int(v) for k, v in
                                     later_orders["route"].value_counts().sort_index().items()}},
    }
    facts["latent"] = _latent(tables, order, user)
    return facts


def _publication(run: Run, pick: Pick) -> dict[str, Any]:
    return {"order_id": pick.order_id, "alert_id": pick.alert_id,
            "policy_version": run.policy_version, "selection_hash": pick.selection_hash,
            "tier": pick.tier,
            "candidates": {"eligible": pick.eligible, "reviewed": pick.reviewed,
                           "available": pick.available, "tiers": dict(pick.tiers)}}


def _source(run: Run) -> dict[str, Any]:
    rule, manifest = run.rule, run.manifest
    start, end = run.window
    return {
        "world": {"seed": manifest["seed"], "family": manifest["family"],
                  "generator_version": manifest["generator_version"],
                  "manifest_identity": manifest_identity(manifest),
                  "code_commit": manifest.get("code_commit")},
        "replay": {**rule.replay, "review_threshold": run.review_threshold,
                   "decline_threshold": run.decline_threshold,
                   "policy_version": run.policy_version, "window_start": start,
                   "window_end": end, "observed_until": run.protocol.observed_until},
        "development_seed": rule.seed in run.protocol.development_seeds,
        "illustration": ILLUSTRATION.format(seed=rule.seed)
        if rule.seed in run.protocol.development_seeds else None,
    }


def build(run: Run, picks: Mapping[str, Pick] | None = None) -> dict[str, dict[str, Any]]:
    """Every case file's facts, by file name."""
    picks = run.picks() if picks is None else picks
    tables = run.tables
    chosen = [p.order_id for p in picks.values() if p.selected]
    observed = run.protocol.observed_until
    realized = actions.realize(tables, run.fates.loc[run.fates["order_id"].isin(chosen)],
                               ledger.ProductTerms.from_config(), observed_until=observed)
    labels = world_module.labels_as_of(tables["labels"], observed)
    source = _source(run)
    out = {}
    for name, entry in run.rule.files.items():
        alerts = {}
        for slot in entry.slots:
            pick = picks[slot]
            alerts[slot] = (alert_facts(run, pick, realized, labels) if pick.selected else
                            {"selected": False, "slot": slot,
                             "publication": _publication(run, pick)})
        out[name] = plain({"file": name, "length": entry.length, "slots": list(entry.slots),
                           "primary_alert": entry.primary, "source": source,
                           "alerts": alerts})
    return out


def write(facts: Mapping[str, Mapping[str, Any]], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in sorted(facts.items()):
        path = out_dir / f"{name}.json"
        path.write_text(dumps(content))
        written.append(path)
    return written


# --------------------------------------------------------------------- memo packets


def memo_inputs(run_dir: Path, protocol: protocol_module.Protocol | None = None,
                results_dir: Path | None = None) -> pd.DataFrame:
    """The selected alerts as the memo drafter's packets need them: one row per selected
    alert with ``file``, ``slot``, ``tier``, ``alert_id``, ``policy_version``,
    ``evidence_at`` and the saved row the replay decided on (``core.asof`` KEY_COLUMNS +
    COLUMN_NAMES, ``decision_at`` = ``evidence_at``), in the protocol's slot order. No
    verification check had completed at these decisions (checks start with a hold), so
    the packets carry none. ``llm.packet.build_packets(tables, frame[ROW_COLUMNS])``
    builds them."""
    run = load_run(run_dir, protocol, results_dir)
    records = []
    for slot, pick in run.picks().items():
        if not pick.selected:
            continue
        decision, row = _decision(run, pick)
        records.append({"file": run.rule.file_of(slot), "slot": slot, "tier": pick.tier,
                        "alert_id": pick.alert_id, "policy_version": run.policy_version,
                        "evidence_at": decision["evidence_at"], **row})
    return pd.DataFrame(records, columns=["file", "slot", "tier", "alert_id", "policy_version",
                                          "evidence_at", *ROW_COLUMNS])
