"""Selects the memo benchmark's cases and writes the benchmark before any call.

A case is one analyst decision on an order the incumbent policy sent to review, in one
world, on one of two decision-point axes:

* ``review``: the first decision, when an analyst takes the order up, before any check;
* ``check_completed``: the decision when the checks a hold ran have answered, at the
  last completion, with every completed check, on the row the replay's reviewer read
  then (:mod:`llm.eval.check_points`). These are the rows of fraud policy §5.3: after a
  passed check every hold is prohibited once every required check has passed (until then
  the hold continues under §6.6(b)); after a failed one the order is declined, or
  escalated with Linkage. A hold that no check answers expires without an analyst
  decision, so §5.3(c) has no cases.

A case's id joins the world's seed and family, the order id and the decision time, so
it keeps its identity whatever the score bands. Selection is the one place on the memo
side that reads simulation truth, and only to stratify and to report a diagnostic; the
packets never carry it.

* Phases: development cases come from the test windows of the development seeds'
  baseline worlds; final cases from the test windows of the baseline worlds of at
  least three final seeds (:data:`PHASES`). A world outside its phase's seeds or
  families is refused.
* Strata: on the review axis, every fraud pattern among the review decisions, and the
  legitimate orders by benign trait: of the traits in an order's mimic label (or,
  without one, its account's behaviour profile: new customers, households, travellers,
  movers, hardship, ...), the one rarest among the pool's legitimate orders
  (:func:`stratify`). On the check axis, the completed checks with their outcomes, the
  referee's standard disposition and the latent class (:func:`check_stratum`). More
  strata among phase one's cases than an axis's cases are refused, so every stratum
  phase one reaches has cases; a stratum whose every case lost its group's draw in phase
  one has none, as two-phase sampling allows, and the definition names it
  (``strata_not_drawn``).
* Sizes: the final cohort takes 120 review decisions and 80 decisions at check
  completions (``config/llm.yaml`` ``final_axes``; 80 is about the development pools'
  share of analyst decisions that follow a check, 283 of 743). Each check outcome in the
  final pool gets at least ``min_cases_per_check_outcome`` cases, or the selection is
  refused.
* Independence: cases linked through an account or an episode form one group (the
  cluster for intervals), and at most two cases per group are eligible, over both axes
  together; development and final cohorts share no account and no episode.
* Balance and weights: selection is two-phase sampling. Phase one draws at most two
  cases at random from each linked group, so a case is drawn with probability
  (cases drawn from its group) / (its group's size); phase two draws, within each axis,
  an equal share per stratum (a challenge set) at random from phase one's cases, so,
  given phase one, a case is kept with probability (its stratum's quota) / (its
  stratum's phase-one cases). A case's weight is the inverse of the product of the two.
  These are sequential two-phase weights, not marginal inclusion probabilities (phase
  two's denominators depend on phase one); within an axis the weighted rate is the
  two-phase Hájek estimator of the rate over the axis's eligible decisions, and the
  natural-mix rate weights each axis by its share of the eligible decisions
  (``axes`` in the definition). A pool that cannot fill the requested size is refused.

:func:`write_benchmark` then fixes the benchmark: packets, the referee's view of each,
the definition with every hash (policy, prompt, scoring code, packets, views) and the
arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.actions import Check, CheckOutcome
from core.config import load
from core.evidence import POLICY, CheckResult
from core.protocol import load_protocol
from core.world import read_world
from llm import client, memo, referee
from llm.eval import check_points, harness
from llm.eval.harness import BENCHMARKS, code_sha256, sha256_file
from llm.packet import assert_no_forbidden, build_packets

_SIZES = load("llm")["benchmark"]
MAX_PER_CLUSTER = int(_SIZES["max_cases_per_cluster"])
MIN_FINAL_SEEDS = int(_SIZES["min_final_seeds"])
AXES = ("review", "check_completed")
# Where each phase's cases come from: protocol seeds, world families, window. Final cases
# too come from baseline worlds only: a lag-sensitivity world holds its baseline world's
# reviewed orders again, and the shifted futures would blend other populations into the
# natural-mix rate.
PHASES: dict[str, dict[str, Any]] = {
    "development": {"seeds": "development_seeds", "families": ("baseline",),
                    "window": "test"},
    "final": {"seeds": "final_seeds", "families": ("baseline",), "window": "test"},
}


def case_id(seed: int, family: str, order_id: int, decision_at: Any) -> str:
    return f"{seed}-{family}-{int(order_id)}-{pd.Timestamp(decision_at):%Y%m%dT%H%M%S}"


def candidates(seed: int, family: str, decisions: pd.DataFrame,
               tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Review decisions of one world with their stratum, account and episode.

    ``decisions`` has order_id and decision_at (other columns are not used here);
    ``tables`` must include order_attempts, latent_orders and latent_accounts.
    """
    attempts = tables["order_attempts"][["order_id", "user_id"]]
    orders = tables["latent_orders"][["order_id", "pattern_id", "episode_id", "intent",
                                      "mimic"]]
    accounts = tables["latent_accounts"][["user_id", "profile"]]
    frame = (decisions[["order_id", "decision_at"]]
             .merge(attempts, on="order_id", how="left", validate="many_to_one")
             .merge(orders, on="order_id", how="left", validate="many_to_one")
             .merge(accounts, on="user_id", how="left", validate="many_to_one"))
    if frame["user_id"].isna().any() or frame["intent"].isna().any():
        raise ValueError("every decision needs its order attempt and latent order row")
    frame["benign_traits"] = [_traits(mimic) or _traits(profile) for mimic, profile
                              in zip(frame["mimic"], frame["profile"], strict=True)]
    frame["stratum"] = stratify(frame)
    frame["latent_class"] = np.where(frame["intent"].eq("legitimate"), "legitimate", "fraud")
    frame["seed"], frame["family"] = int(seed), str(family)
    frame["case_id"] = [case_id(seed, family, o, t)
                        for o, t in zip(frame["order_id"], frame["decision_at"], strict=True)]
    frame["account_key"] = [f"{seed}:{u}" for u in frame["user_id"]]
    frame["episode_key"] = [f"{seed}:{int(e)}" if pd.notna(e) else None
                            for e in frame["episode_id"]]
    frame["axis"], frame["check_outcome"] = "review", None
    return frame


def check_outcome(checks: Sequence[CheckResult]) -> str:
    """The outcome a decision after checks rests on: failed if any check failed."""
    outcomes = {str(result.outcome) for result in checks}
    for outcome in ("failed", "no_response", "passed"):
        if outcome in outcomes:
            return outcome
    raise ValueError("a decision after checks needs a completed check")


def check_stratum(checks: Sequence[CheckResult], standard: Iterable[str],
                  latent_class: str) -> str:
    """``after_check:<check>=<outcome>[+...]:<standard>:<latent class>``."""
    done = "+".join(f"{result.check}={result.outcome}"
                    for result in sorted(checks, key=lambda r: str(r.check)))
    return f"after_check:{done}:{'|'.join(sorted(standard))}:{latent_class}"


def check_candidates(seed: int, family: str, completions: pd.DataFrame,
                     tables: Mapping[str, pd.DataFrame]
                     ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Decisions at check completions of one world (rows of
    :func:`check_points.completion_decisions`) as candidates on the check axis, with
    their packets by case id."""
    if completions.empty:
        return candidates(seed, family, completions, tables).assign(
            axis="check_completed"), {}
    keys = [(int(o), pd.Timestamp(t)) for o, t in zip(completions["order_id"],
                                                      completions["decision_at"], strict=True)]
    built = build_packets(tables, completions.reset_index(drop=True),
                          checks=dict(zip(keys, completions["checks"], strict=True)))
    frame = candidates(seed, family, completions, tables)
    frame["axis"] = "check_completed"
    frame["check_outcome"] = [check_outcome(checks) for checks in completions["checks"]]
    frame["stratum"] = [
        check_stratum(checks, referee.view(built[key]).standard, latent)
        for checks, key, latent in zip(completions["checks"], keys, frame["latent_class"],
                                       strict=True)]
    packets = {case: built[key] for case, key in zip(frame["case_id"], keys, strict=True)}
    return frame, packets


def _traits(value: Any) -> tuple[str, ...]:
    """The traits of a ``+``-joined mimic or profile label."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    return tuple(sorted({trait.strip() for trait in str(value).split("+") if trait.strip()}))


def stratify(frame: pd.DataFrame) -> list[str]:
    """Each candidate's stratum: its fraud pattern, or for a legitimate order
    ``legitimate:<trait>``, the rarest among the frame's legitimate orders of the order's
    benign traits (its mimic's, else its account profile's), so a combined label such as
    ``gift+new_customer_first_order`` falls in one stratum per trait and rare traits keep
    a stratum of their own; ``legitimate:other`` when it has none."""
    legitimate = frame["pattern_id"].isna().to_numpy()
    counts = Counter(trait for traits, plain in zip(frame["benign_traits"], legitimate,
                                                    strict=True) if plain for trait in traits)
    return [pattern if not plain else
            "legitimate:" + (min(traits, key=lambda t: (counts[t], t)) if traits else "other")
            for pattern, traits, plain in zip(frame["pattern_id"], frame["benign_traits"],
                                              legitimate, strict=True)]


def linked_groups(frame: pd.DataFrame) -> list[str]:
    """Each row's group: rows sharing an account or an episode, directly or through
    other rows, are one group, named after its first account key."""
    parent: dict[str, str] = {}

    def root(node: str) -> str:
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for account, episode in zip(frame["account_key"], frame["episode_key"], strict=True):
        if pd.notna(episode):
            first, second = root(f"account {account}"), root(f"episode {episode}")
            if first != second:
                parent[max(first, second)] = min(first, second)
    names: dict[str, str] = {}
    for account in sorted(frame["account_key"]):
        names.setdefault(root(f"account {account}"), f"group {account}")
    return [names[root(f"account {account}")] for account in frame["account_key"]]


def _allocation(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Equal shares per stratum, capped by its size, the remainder spread in order."""
    quota = dict.fromkeys(sizes, 0)
    remaining = total
    open_strata = sorted(stratum for stratum, size in sizes.items() if size > 0)
    while remaining > 0 and open_strata:
        share = max(1, remaining // len(open_strata))
        for stratum in list(open_strata):
            take = min(share, sizes[stratum] - quota[stratum], remaining)
            quota[stratum] += take
            remaining -= take
            if quota[stratum] >= sizes[stratum]:
                open_strata.remove(stratum)
            if remaining == 0:
                break
    return quota


def _with_minimum(quota: Mapping[str, int], sizes: Mapping[str, int],
                  group_of: Mapping[str, str], minimum: int,
                  required: Iterable[str] = ()) -> dict[str, int]:
    """``quota`` with at least ``minimum`` cases in every group of strata (and in every
    group named in ``required``, the outcomes present before phase one): cases move one
    at a time to the group's stratum with the fewest, from the largest stratum of a
    group above the minimum (never leaving a stratum empty). Raises ValueError when a
    group has fewer than ``minimum`` cases available or none can move."""
    quota = dict(quota)
    for group in sorted(set(group_of.values()) | set(required)):
        members = sorted(s for s in sizes if group_of[s] == group)
        available = sum(sizes[s] for s in members)
        if available < minimum:
            raise ValueError(f"{available} cases after a {group} check under the cluster "
                             f"caps; the cohort needs {minimum}")
        while sum(quota[s] for s in members) < minimum:
            totals = Counter()
            for stratum, count in quota.items():
                totals[group_of[stratum]] += count
            donors = [s for s in sizes if group_of[s] != group and quota[s] > 1
                      and totals[group_of[s]] > minimum]
            open_members = [s for s in members if quota[s] < sizes[s]]
            if not donors or not open_members:
                raise ValueError(f"the axis's cases cannot give {minimum} to every check "
                                 "outcome")
            donor = max(donors, key=lambda s: (quota[s], s))
            taker = min(open_members, key=lambda s: (quota[s], s))
            quota[donor] -= 1
            quota[taker] += 1
    return quota


def select(pool: pd.DataFrame, total: int | Mapping[str, int], *, rng_seed: int,
           exclude_accounts: Iterable[str] = (), exclude_episodes: Iterable[str] = (),
           min_per_outcome: int = 0) -> pd.DataFrame:
    """A stratified sample from ``pool`` (rows of :func:`candidates` and
    :func:`check_candidates`) of exactly ``total`` cases, or of each axis's count when
    ``total`` maps axes to counts (an int means review decisions only), with
    ``cluster`` (the linked group), ``first_phase`` and ``second_phase`` (the case's
    phase-one probability and its phase-two probability given phase one) and ``weight``
    (the inverse of their product).

    Rows of other axes, and excluded accounts and episodes (another cohort's), are
    removed first. Phase one draws at most :data:`MAX_PER_CLUSTER` cases at random from
    every linked group, over the axes together; phase two draws each stratum's equal
    share of its axis's count at random from phase one's cases, with at least
    ``min_per_outcome`` cases per check outcome on the check axis
    (:func:`_with_minimum`). Raises ValueError when the pool cannot fill a count.
    """
    totals = ({"review": int(total)} if isinstance(total, int | np.integer)
              else {str(axis): int(count) for axis, count in total.items()})
    if unknown := set(totals) - set(AXES):
        raise ValueError(f"unknown decision-point axes {sorted(unknown)}")
    eligible = _eligible(pool, set(totals), exclude_accounts, exclude_episodes)
    eligible["cluster"] = linked_groups(eligible)
    rng = np.random.default_rng(rng_seed)
    group_size = eligible["cluster"].value_counts()
    taken: dict[str, int] = {}
    first_phase = []
    for position in rng.permutation(len(eligible)):
        row = eligible.index[position]
        group = eligible.at[row, "cluster"]
        if taken.get(group, 0) < MAX_PER_CLUSTER:
            first_phase.append(row)
            taken[group] = taken.get(group, 0) + 1
    drawn = eligible.loc[first_phase]
    drawn = drawn.iloc[rng.permutation(len(drawn))]  # an independent order for phase two
    available = drawn["stratum"].value_counts().to_dict()
    axis_of = dict(zip(drawn["stratum"], drawn["axis"], strict=True))
    outcome_of = dict(zip(drawn["stratum"], drawn["check_outcome"], strict=True))
    quota: dict[str, int] = {}
    for axis, count in sorted(totals.items()):
        sizes = {stratum: size for stratum, size in available.items()
                 if axis_of[stratum] == axis}
        if len(sizes) > count:
            raise ValueError(f"{len(sizes)} strata for {count} cases: some strata would "
                             "get no case and the weights could not reach them")
        share = _allocation(sizes, count)
        if axis == "check_completed" and min_per_outcome:
            present = eligible.loc[eligible["axis"] == axis, "check_outcome"].dropna()
            share = _with_minimum(share, sizes, {s: str(outcome_of[s]) for s in sizes},
                                  min_per_outcome, required=set(present.astype(str)))
        if sum(share.values()) < count:
            raise ValueError(f"the pool yields {sum(share.values())} of the {count} "
                             f"{axis.replace('_', ' ')} cases asked for under the cluster "
                             "caps")
        quota.update(share)
    chosen = pd.concat([drawn[drawn["stratum"] == stratum].head(count)
                        for stratum, count in sorted(quota.items()) if count > 0])
    groups = chosen["cluster"]
    first = groups.map(lambda group: min(MAX_PER_CLUSTER, group_size[group])
                       / group_size[group])
    second = chosen["stratum"].map(lambda stratum: quota[stratum] / available[stratum])
    chosen = chosen.assign(first_phase=first, second_phase=second,
                           weight=1.0 / (first * second))
    return chosen.sort_values("case_id").reset_index(drop=True)


def _eligible(pool: pd.DataFrame, axes: set[str], exclude_accounts: Iterable[str],
              exclude_episodes: Iterable[str]) -> pd.DataFrame:
    """The pool's rows on ``axes`` (rows without an axis are review decisions), less the
    excluded accounts and episodes."""
    frame = pool.copy()
    if "axis" not in frame:
        frame["axis"] = "review"
    if "check_outcome" not in frame:
        frame["check_outcome"] = None
    keep = (frame["axis"].isin(axes) & ~frame["account_key"].isin(set(exclude_accounts))
            & ~frame["episode_key"].isin(set(exclude_episodes)))
    return frame[keep.to_numpy()].copy()


def eligible_axes(pool: pd.DataFrame, totals: Mapping[str, int], *,
                exclude_accounts: Iterable[str] = (), exclude_episodes: Iterable[str] = ()
                ) -> dict[str, dict[str, Any]]:
    """Per axis: its cases, its eligible decisions and its share of them, the weight the
    natural-mix rate gives the axis."""
    eligible = _eligible(pool, set(totals), exclude_accounts, exclude_episodes)
    counts = eligible["axis"].value_counts()
    whole = int(sum(int(counts.get(axis, 0)) for axis in totals))
    return {axis: {"cases": int(count), "eligible": int(counts.get(axis, 0)),
                   "share": (int(counts.get(axis, 0)) / whole) if whole else 0.0}
            for axis, count in sorted(totals.items())}


def check_cohorts(development: pd.DataFrame, final: pd.DataFrame,
                  final_seeds: Sequence[int]) -> None:
    """Development and final cohorts share no account or episode; final cases come
    from at least three of the protocol's final seeds and from no other seed."""
    shared_accounts = set(development["account_key"]) & set(final["account_key"])
    shared_episodes = (set(development["episode_key"].dropna())
                       & set(final["episode_key"].dropna()))
    if shared_accounts or shared_episodes:
        raise ValueError(f"cohorts share accounts {sorted(shared_accounts)[:3]} or "
                         f"episodes {sorted(shared_episodes)[:3]}")
    seeds = set(final["seed"])
    if not seeds <= set(final_seeds) or len(seeds) < MIN_FINAL_SEEDS:
        raise ValueError(f"final cases must come from at least {MIN_FINAL_SEEDS} final "
                         f"seeds, got {sorted(seeds)}")
    for cohort in (development, final):
        for key in ("account_key", "episode_key"):
            if cohort[key].dropna().value_counts().max() > MAX_PER_CLUSTER:
                raise ValueError(f"more than {MAX_PER_CLUSTER} cases share a {key}")


def probe_cases(cases: pd.DataFrame, n: int, *, rng_seed: int) -> list[str]:
    """``n`` cases for the invariance probes, spread across strata."""
    rng = np.random.default_rng(rng_seed)
    seen: dict[str, int] = {}
    ranked = []
    for row in cases.iloc[rng.permutation(len(cases))].itertuples():
        seen[row.stratum] = seen.get(row.stratum, 0) + 1
        ranked.append((seen[row.stratum], row.stratum, row.case_id))
    return sorted(case for _, _, case in sorted(ranked)[:n])


def write_benchmark(directory: Path, *, benchmark_id: str, phase: str,
                    cases: pd.DataFrame, packets: Mapping[str, Mapping[str, Any]],
                    arms: Mapping[str, Mapping[str, str]],
                    probes: Mapping[str, Sequence[str]] | None = None,
                    prompt_version: str = memo.PROMPT_VERSION,
                    axes: Mapping[str, Mapping[str, Any]] | None = None,
                    drawn_from: Mapping[str, Any] | None = None,
                    strata_not_drawn: Sequence[str] = ()) -> dict[str, Any]:
    """Write the packets, the referee's view of each and the definition. ``axes`` gives
    each decision-point axis's cases, eligible decisions and share (:func:`eligible_axes`;
    by default the cases' own axes, each with its share of the cases); ``drawn_from``
    names the benchmarks a subset was drawn from and the rule."""
    if "axis" not in cases:
        cases = cases.assign(axis="review")
    if axes is None:
        counts = cases["axis"].value_counts()
        axes = {axis: {"cases": int(n), "eligible": int(n), "share": int(n) / len(cases)}
                for axis, n in sorted(counts.items())}
    if phase not in harness.PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    if set(packets) != set(cases["case_id"]):
        raise ValueError("every case needs exactly one packet")
    (directory / "packets").mkdir(parents=True, exist_ok=True)
    views, packet_hashes = {}, {}
    for case in sorted(packets):
        assert_no_forbidden(packets[case])
        path = directory / "packets" / f"{case}.json"
        path.write_text(json.dumps(packets[case], indent=1, ensure_ascii=False) + "\n")
        packet_hashes[case] = sha256_file(path)
        views[case] = referee.view(packets[case]).as_dict()
    (directory / "referee.json").write_text(json.dumps(views, indent=1, sort_keys=True) + "\n")
    definition = {
        "id": benchmark_id,
        "phase": phase,
        "policy": {"id": POLICY, "sha256": sha256_file(memo.POLICY_PATH)},
        "prompt": {"version": prompt_version,
                   "sha256": client.sha256_text(memo.system_prompt(prompt_version))},
        "code_sha256": code_sha256(),
        "referee_sha256": sha256_file(directory / "referee.json"),
        "arms": {name: dict(arm) for name, arm in sorted(arms.items())},
        "cases": [
            {"case_id": row.case_id, "seed": int(row.seed), "family": row.family,
             "account": row.account_key,
             "episode": None if pd.isna(row.episode_key) else row.episode_key,
             "axis": row.axis, "stratum": row.stratum, "cluster": row.cluster,
             "weight": float(row.weight),
             "phase_probabilities": [float(getattr(row, "first_phase", 1.0)),
                                     float(getattr(row, "second_phase", 1.0))],
             "packet_sha256": packet_hashes[row.case_id],
             "latent": {"class": row.latent_class,
                        "pattern": None if pd.isna(row.pattern_id) else row.pattern_id}}
            for row in cases.sort_values("case_id").itertuples()
        ],
        "probes": {name: sorted(ids) for name, ids in sorted((probes or {}).items())},
        "axes": {axis: dict(values) for axis, values in sorted(axes.items())},
    }
    if drawn_from is not None:
        definition["drawn_from"] = dict(drawn_from)
    if strata_not_drawn:
        definition["strata_not_drawn"] = sorted(strata_not_drawn)
    (directory / "benchmark.json").write_text(
        json.dumps(definition, indent=1, sort_keys=True) + "\n")
    return definition


def _checks(value: Any) -> tuple[CheckResult, ...]:
    """Completed checks from a review-decision row: a list (or its JSON text) of check
    results, mappings with ``check``, ``outcome`` and ``completed_at``, or
    ``(check, outcome, completed_at)`` triples as the replay keeps them."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return ()
    results = []
    for item in json.loads(value) if isinstance(value, str) else value:
        if isinstance(item, CheckResult):
            results.append(item)
            continue
        check, outcome, completed_at = (
            (item["check"], item["outcome"], item["completed_at"])
            if isinstance(item, Mapping) else item)
        results.append(CheckResult(Check(check), CheckOutcome(outcome),
                                   pd.Timestamp(completed_at).to_pydatetime()))
    return tuple(results)


def _read_decisions(world_dir: Path) -> pd.DataFrame:
    for name in ("review_decisions.pkl", "review_decisions.csv"):
        path = world_dir / name
        if path.exists():
            frame = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_csv(path)
            frame["decision_at"] = pd.to_datetime(frame["decision_at"])
            return frame
    raise FileNotFoundError(f"no review decisions in {world_dir}")


def phase_axes(phase: str, axis: str | None = None,
               sizes: Mapping[str, Any] | None = None) -> dict[str, int]:
    """The cases a benchmark of ``phase`` takes on each axis: the final cohort's split,
    or a development set of review decisions or of decisions at check completions."""
    sizes = harness.SIZES if sizes is None else sizes
    if phase == "final":
        if axis is not None:
            raise ValueError("a final cohort takes both axes")
        return harness.final_axes(sizes)
    if axis in (None, "review"):
        return {"review": int(sizes["development_cases"])}
    if axis == "check_completed":
        return {"check_completed": int(sizes["development_check_cases"])}
    raise ValueError(f"unknown decision-point axis {axis!r}")


def build(benchmark_id: str, phase: str, world_dirs: Sequence[Path], *, rng_seed: int,
          development: str | Sequence[str] | None = None, axis: str | None = None
          ) -> dict[str, Any]:
    """Select and write a benchmark of the phase's configured size (and, for the final
    phase, its invariance probes) from worlds whose directories hold the tables, the
    manifest and the incumbent's review decisions (with their context rows); decisions
    at check completions also need the run's tuning results and fitted models
    (:mod:`llm.eval.check_points`). A development set takes one axis (``axis``,
    review decisions by default); the final cohort takes both and excludes the
    accounts and episodes of every development benchmark named. Worlds outside the
    phase's seeds or families are refused, and so is a definition that does not have
    the configured shape (:func:`harness.check_shape`)."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    rules, protocol = PHASES[phase], load_protocol()
    sizes = harness.SIZES
    totals = phase_axes(phase, axis, sizes)
    allowed_seeds = set(getattr(protocol, rules["seeds"]))
    pools, worlds, check_packets = [], {}, {}
    for world_dir in world_dirs:
        manifest = json.loads((world_dir / "manifest.json").read_text())
        seed, family = int(manifest["seed"]), str(manifest["family"])
        if seed not in allowed_seeds:
            raise ValueError(f"seed {seed} is not a {phase} seed")
        if rules["families"] is not None and family not in rules["families"]:
            raise ValueError(f"{phase} cases come from {rules['families']} worlds, not {family}")
        tables = read_world(world_dir)
        decisions = _read_decisions(world_dir)
        placed = decisions[["order_id"]].merge(
            tables["order_attempts"][["order_id", "known_at"]], on="order_id", how="left")
        in_window = protocol.window_of(pd.to_datetime(placed["known_at"])).eq(rules["window"])
        decisions = decisions[in_window.fillna(False).to_numpy(dtype=bool)].reset_index(
            drop=True)
        if "review" in totals:
            pools.append(candidates(seed, family, decisions, tables))
        if "check_completed" in totals:
            kept = world_dir / "review_decisions.pkl"
            if not kept.exists():
                raise FileNotFoundError(f"decisions at check completions need {kept}")
            completions = check_points.completion_decisions(
                world_dir, pd.read_pickle(kept), seed=seed, tables=tables)
            completions = completions[completions["order_id"].isin(
                decisions["order_id"])].reset_index(drop=True)
            frame, packets = check_candidates(seed, family, completions, tables)
            pools.append(frame)
            check_packets.update(packets)
        worlds[(seed, family)] = (tables, decisions)
    pool = pd.concat(pools, ignore_index=True)
    review = pool["axis"].eq("review").to_numpy()
    if review.any():  # trait rarity over every world of the build
        pool.loc[review, "stratum"] = stratify(pool.loc[review])
    exclude: dict[str, set[str]] = {"accounts": set(), "episodes": set()}
    if phase == "final":
        earlier_ids = [development] if isinstance(development, str) else list(development or ())
        if not earlier_ids:
            raise ValueError("a final cohort is checked against its development cohorts")
        for earlier_id in earlier_ids:
            earlier = json.loads((BENCHMARKS / earlier_id / "benchmark.json").read_text())
            if earlier["phase"] != "development":
                raise ValueError(f"{earlier_id} is not a development benchmark")
            exclude["accounts"] |= {case["account"] for case in earlier["cases"]}
            exclude["episodes"] |= {case["episode"] for case in earlier["cases"]
                                    if case["episode"]}
    minimum = int(sizes.get("min_cases_per_check_outcome", 0)) if phase == "final" else 0
    chosen = select(pool, totals, rng_seed=rng_seed, exclude_accounts=exclude["accounts"],
                    exclude_episodes=exclude["episodes"], min_per_outcome=minimum)
    axes = eligible_axes(pool, totals, exclude_accounts=exclude["accounts"],
                         exclude_episodes=exclude["episodes"])
    eligible = _eligible(pool, set(totals), exclude["accounts"], exclude["episodes"])
    not_drawn = sorted(set(eligible["stratum"]) - set(chosen["stratum"]))
    if phase == "final":
        earlier_frame = pd.DataFrame({"account_key": sorted(exclude["accounts"]),
                                      "episode_key": None})
        check_cohorts(earlier_frame, chosen, protocol.final_seeds)
    packets: dict[str, dict[str, Any]] = {
        case: check_packets[case]
        for case in chosen.loc[chosen["axis"] == "check_completed", "case_id"]}
    first = chosen[chosen["axis"] == "review"]
    for (seed, family), (tables, decisions) in worlds.items():
        mine = first[(first["seed"] == seed) & (first["family"] == family)]
        if mine.empty:
            continue
        rows = decisions.merge(mine[["order_id", "decision_at", "case_id"]],
                               on=["order_id", "decision_at"], how="inner")
        checks: dict[tuple[int, datetime], tuple[CheckResult, ...]] = {
            (int(r.order_id), pd.Timestamp(r.decision_at)): _checks(getattr(r, "checks", None))
            for r in rows.itertuples()}
        built = build_packets(tables, rows, checks=checks)
        for r in rows.itertuples():
            packets[r.case_id] = built[(int(r.order_id), pd.Timestamp(r.decision_at))]
    arms = load("llm")["arms"]
    probes = ({kind: probe_cases(chosen, int(sizes["probe_cases"]), rng_seed=rng_seed)
               for kind in sizes["probe_kinds"]} if phase == "final" else {})
    return _write(benchmark_id, phase=phase, cases=chosen, packets=packets, arms=arms,
                  probes=probes, axes=axes, strata_not_drawn=not_drawn)


def _write(benchmark_id: str, **fields: Any) -> dict[str, Any]:
    """Write a benchmark in a staging folder, check its shape, then move it in place."""
    staging = BENCHMARKS / f".{benchmark_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    definition = write_benchmark(staging, benchmark_id=benchmark_id, **fields)
    try:
        harness.check_shape(definition, harness.SIZES, load_protocol())
    except harness.ShapeError:
        shutil.rmtree(staging)
        raise
    target = BENCHMARKS / benchmark_id
    if target.exists():
        shutil.rmtree(staging)
        raise FileExistsError(f"benchmark {benchmark_id} already exists")
    staging.rename(target)
    return definition


def _rank(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


SUBSET_RULE = ("from each source, its strata in order of the SHA-256 of their names and "
               "each stratum's cases in order of the SHA-256 of their ids, one case per "
               "stratum in turn until the source's count is reached; a case that would put "
               "more than the cluster cap of cases in one linked group of the subset is "
               "passed over")


def subset_cases(sources: Mapping[str, Sequence[Mapping[str, Any]]],
                 counts: Mapping[str, int]) -> list[tuple[str, Mapping[str, Any]]]:
    """The cases :data:`SUBSET_RULE` takes, as (source id, case) pairs: ``counts[id]``
    from each source's cases."""
    chosen: list[tuple[str, Mapping[str, Any]]] = []
    for source in sorted(sources):
        by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for case in sorted(sources[source], key=lambda c: _rank(c["case_id"])):
            by_stratum[case["stratum"]].append(case)
        order = sorted(by_stratum, key=_rank)
        queue = [by_stratum[stratum] for stratum in order]
        taken = 0
        while taken < counts[source] and any(queue):
            for cases in queue:
                if taken >= counts[source]:
                    break
                while cases:
                    case = cases.pop(0)
                    trial = [*(c for _, c in chosen), case]
                    frame = pd.DataFrame({"account_key": [c["account"] for c in trial],
                                          "episode_key": [c["episode"] for c in trial]})
                    if max(Counter(linked_groups(frame)).values()) <= MAX_PER_CLUSTER:
                        chosen.append((source, case))
                        taken += 1
                        break
        if taken < counts[source]:
            raise ValueError(f"{source} gives {taken} of the {counts[source]} cases asked for")
    return chosen


def combine(benchmark_id: str, sources: Sequence[str], *, arms: Sequence[str]
            ) -> dict[str, Any]:
    """A development benchmark of cases drawn from other development benchmarks by
    :data:`SUBSET_RULE` (``development_combined`` of ``config/llm.yaml``: so many review
    decisions and so many decisions at check completions), for the arms named. Packets
    are copied as they are; the subset is not a probability sample, so its reweighted
    rates estimate nothing (its axes' shares are the sources' eligible decisions when every
    source recorded them, else its own case shares) and its clusters are its own linked
    groups."""
    wanted = {str(axis): int(n) for axis, n in harness.SIZES["development_combined"].items()}
    definitions = {source: json.loads((BENCHMARKS / source / "benchmark.json").read_text())
                   for source in sources}
    counts: dict[str, int] = {}
    for source, definition in definitions.items():
        if definition["phase"] != "development":
            raise ValueError(f"{source} is not a development benchmark")
        found = {case.get("axis", "review") for case in definition["cases"]}
        if len(found) != 1 or next(iter(found)) not in wanted:
            raise ValueError(f"{source} must hold one axis of {sorted(wanted)}")
        counts[source] = wanted.pop(next(iter(found)))
    if wanted:
        raise ValueError(f"no source for the {sorted(wanted)} cases")
    picked = subset_cases({source: definition["cases"]
                           for source, definition in definitions.items()}, counts)
    rows = []
    packets: dict[str, dict[str, Any]] = {}
    for source, case in picked:
        packets[case["case_id"]] = json.loads(
            (BENCHMARKS / source / "packets" / f"{case['case_id']}.json").read_text())
        rows.append({"case_id": case["case_id"], "seed": case["seed"],
                     "family": case["family"], "account_key": case["account"],
                     "episode_key": case["episode"], "axis": case.get("axis", "review"),
                     "stratum": case["stratum"], "weight": case["weight"],
                     "first_phase": case["phase_probabilities"][0],
                     "second_phase": case["phase_probabilities"][1],
                     "latent_class": case["latent"]["class"],
                     "pattern_id": case["latent"]["pattern"]})
    cases = pd.DataFrame(rows)
    cases["cluster"] = linked_groups(cases)
    configured = load("llm")["arms"]
    if unknown := set(arms) - set(configured):
        raise ValueError(f"unknown arms {sorted(unknown)}")
    # each axis's eligible decisions as its source recorded them; the shares are the
    # natural mix's when every source recorded them, else the subset's own case shares
    axes = {}
    for source, definition in definitions.items():
        (axis,) = {case.get("axis", "review") for case in definition["cases"]}
        eligible = (definition.get("axes") or {}).get(axis, {}).get("eligible")
        axes[axis] = {"cases": counts[source],
                      "eligible": None if eligible is None else int(eligible)}
    known = all(item["eligible"] for item in axes.values())
    whole = sum(item["eligible" if known else "cases"] for item in axes.values())
    for item in axes.values():
        item["share"] = item["eligible" if known else "cases"] / whole
    prompts = {definition["prompt"]["version"] for definition in definitions.values()}
    if len(prompts) != 1:
        raise ValueError(f"the sources use different prompts {sorted(prompts)}")
    return _write(benchmark_id, phase="development", cases=cases, packets=packets,
                  arms={name: configured[name] for name in arms}, axes=axes,
                  prompt_version=prompts.pop(),
                  drawn_from={"benchmarks": sorted(sources), "counts": counts,
                              "rule": SUBSET_RULE})


def case_memos(benchmark_id: str, run_dir: Path, *, arms: Sequence[str] = ("sol",),
               inputs: pd.DataFrame | None = None) -> dict[str, Any]:
    """The memos for the case files' alerts as a benchmark of phase ``cases``: one packet
    per selected alert of the protocol's case world, from the row the replay decided on
    (``cases.facts.memo_inputs``, given as ``inputs`` or read from the run), with no
    completed check. A case's stratum is its case file and slot; it carries weight one."""
    if inputs is None:
        from cases.facts import memo_inputs

        inputs = memo_inputs(Path(run_dir))
    world = load_protocol().raw["cases"]["world"]
    seed, family = int(world["seed"]), str(world["family"])
    tables = read_world(Path(run_dir) / "worlds" / f"{seed}-{family}")
    rows = inputs[list(check_points.ROW_COLUMNS)].reset_index(drop=True)
    built = build_packets(tables, rows)
    frame = candidates(seed, family, rows, tables)
    frame["stratum"] = [f"{name}:{slot}" for name, slot
                        in zip(inputs["file"], inputs["slot"], strict=True)]
    frame["cluster"] = linked_groups(frame)
    frame["weight"] = 1.0
    packets = {case: built[(int(order), pd.Timestamp(at))] for case, order, at
               in zip(frame["case_id"], rows["order_id"], rows["decision_at"], strict=True)}
    configured = load("llm")["arms"]
    if unknown := set(arms) - set(configured):
        raise ValueError(f"unknown arms {sorted(unknown)}")
    return _write(benchmark_id, phase="cases", cases=frame, packets=packets,
                  arms={name: configured[name] for name in arms})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and write a memo benchmark.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES))
    parser.add_argument("--world", type=Path, action="append",
                        help="a world directory holding review_decisions.csv or .pkl")
    parser.add_argument("--axis", choices=AXES,
                        help="a development set's decision points (default: review)")
    parser.add_argument("--development", action="append",
                        help="a development benchmark's id (final phase; repeat for each)")
    parser.add_argument("--combine", action="append", metavar="ID",
                        help="draw a development subset from these benchmarks instead")
    parser.add_argument("--case-memos", type=Path, metavar="RUN",
                        help="the memos for the case files' alerts from this pipeline run")
    parser.add_argument("--arm", action="append",
                        help="the arms of a subset (--combine) or of the case memos")
    parser.add_argument("--rng-seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    if args.combine:
        if not args.arm:
            parser.error("--combine needs --arm")
        definition = combine(args.id, args.combine, arms=args.arm)
    elif args.case_memos:
        definition = case_memos(args.id, args.case_memos, arms=args.arm or ("sol",))
    else:
        if not (args.phase and args.world):
            parser.error("--phase and --world are required")
        definition = build(args.id, args.phase, args.world, rng_seed=args.rng_seed,
                           development=args.development, axis=args.axis)
    strata = pd.Series([case["stratum"] for case in definition["cases"]]).value_counts()
    print(f"{args.id}: {len(definition['cases'])} cases")
    print(strata.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
