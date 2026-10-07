"""Selects the memo benchmark's cases and writes the benchmark before any call.

A case is one review decision (an order the incumbent policy sent to review, at the
time an analyst takes it up, optionally after a completed check) in one world. Its id
joins the world's seed and family, the order id and the decision time, so it keeps
its identity whatever the score bands. Selection is the one place on the memo side
that reads simulation truth, and only to stratify and to report a diagnostic; the
packets never carry it.

* Phases: development cases come from the test windows of the development seeds'
  baseline worlds; final cases from the test windows of at least three final seeds
  (:data:`PHASES`). A world outside its phase's seeds or families is refused.
* Strata: every fraud pattern among the review decisions, and the legitimate orders by
  their benign mimic or behaviour profile (legitimate new customers, households,
  travellers, movers, hardship, ...), each a stratum of its own.
* Independence: cases linked through an account or an episode form one group (the
  cluster for intervals), and at most two cases per group are eligible; development
  and final cohorts share no account and no episode.
* Balance and weights: selection has two phases with known probabilities. First, at
  most two cases are drawn at random from each linked group; then an equal share per
  stratum (a challenge set) is drawn at random from those. A case's weight is the
  inverse of its inclusion probability (its group's size over the cases drawn from it,
  times its stratum's drawn cases over its quota), so weighted rates estimate the rate
  over all eligible review decisions. A pool that cannot fill the requested size is
  refused.

:func:`write_benchmark` then fixes the benchmark: packets, the referee's view of each,
the definition with every hash (policy, prompt, scoring code, packets, views) and the
arms.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from llm.eval import harness
from llm.eval.harness import BENCHMARKS, code_sha256, sha256_file
from llm.packet import assert_no_forbidden, build_packets

_SIZES = load("llm")["benchmark"]
MAX_PER_CLUSTER = int(_SIZES["max_cases_per_cluster"])
MIN_FINAL_SEEDS = int(_SIZES["min_final_seeds"])
# Where each phase's cases come from: protocol seeds, world families (None: any), window.
PHASES: dict[str, dict[str, Any]] = {
    "development": {"seeds": "development_seeds", "families": ("baseline",),
                    "window": "test"},
    "final": {"seeds": "final_seeds", "families": None, "window": "test"},
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
    benign = frame["mimic"].fillna(frame["profile"]).fillna("other")
    frame["stratum"] = np.where(frame["pattern_id"].notna(), frame["pattern_id"],
                                "legitimate:" + benign.astype(str))
    frame["latent_class"] = np.where(frame["intent"].eq("legitimate"), "legitimate", "fraud")
    frame["seed"], frame["family"] = int(seed), str(family)
    frame["case_id"] = [case_id(seed, family, o, t)
                        for o, t in zip(frame["order_id"], frame["decision_at"], strict=True)]
    frame["account_key"] = [f"{seed}:{u}" for u in frame["user_id"]]
    frame["episode_key"] = [f"{seed}:{int(e)}" if pd.notna(e) else None
                            for e in frame["episode_id"]]
    return frame


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


def select(pool: pd.DataFrame, total: int, *, rng_seed: int,
           exclude_accounts: Iterable[str] = (), exclude_episodes: Iterable[str] = ()
           ) -> pd.DataFrame:
    """A stratified sample of exactly ``total`` cases from ``pool`` (rows of
    :func:`candidates`), with ``cluster`` (the linked group), ``inclusion`` (the
    probability the design gave the case) and ``weight`` (its inverse).

    Excluded accounts and episodes (another cohort's) are removed first. Phase one
    draws at most :data:`MAX_PER_CLUSTER` cases at random from every linked group;
    phase two draws each stratum's equal share at random from phase one's cases.
    Raises ValueError when the pool cannot fill ``total``.
    """
    excluded_accounts, excluded_episodes = set(exclude_accounts), set(exclude_episodes)
    eligible = pool[~pool["account_key"].isin(excluded_accounts)
                    & ~pool["episode_key"].isin(excluded_episodes)].copy()
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
    quota = _allocation(available, total)
    if sum(quota.values()) < total:
        raise ValueError(f"the pool yields {sum(quota.values())} of the {total} cases "
                         "asked for under the cluster caps")
    chosen = pd.concat([drawn[drawn["stratum"] == stratum].head(count)
                        for stratum, count in sorted(quota.items()) if count > 0])
    groups = chosen["cluster"]
    first = groups.map(lambda group: min(MAX_PER_CLUSTER, group_size[group])
                       / group_size[group])
    second = chosen["stratum"].map(lambda stratum: quota[stratum] / available[stratum])
    chosen = chosen.assign(inclusion=first * second, weight=1.0 / (first * second))
    return chosen.sort_values("case_id").reset_index(drop=True)


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
                    prompt_version: str = memo.PROMPT_VERSION) -> dict[str, Any]:
    """Write the packets, the referee's view of each and the definition."""
    if phase not in ("development", "final"):
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
             "stratum": row.stratum, "cluster": row.cluster, "weight": float(row.weight),
             "inclusion": float(getattr(row, "inclusion", 1.0 / float(row.weight))),
             "packet_sha256": packet_hashes[row.case_id],
             "latent": {"class": row.latent_class,
                        "pattern": None if pd.isna(row.pattern_id) else row.pattern_id}}
            for row in cases.sort_values("case_id").itertuples()
        ],
        "probes": {name: sorted(ids) for name, ids in sorted((probes or {}).items())},
    }
    (directory / "benchmark.json").write_text(
        json.dumps(definition, indent=1, sort_keys=True) + "\n")
    return definition


def _checks(value: Any) -> tuple[CheckResult, ...]:
    """Completed checks from a review-decision row (a JSON list or a list of mappings)."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return ()
    items = json.loads(value) if isinstance(value, str) else value
    return tuple(CheckResult(Check(item["check"]), CheckOutcome(item["outcome"]),
                             pd.Timestamp(item["completed_at"]).to_pydatetime())
                 for item in items)


def _read_decisions(world_dir: Path) -> pd.DataFrame:
    for name in ("review_decisions.pkl", "review_decisions.csv"):
        path = world_dir / name
        if path.exists():
            frame = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_csv(path)
            frame["decision_at"] = pd.to_datetime(frame["decision_at"])
            return frame
    raise FileNotFoundError(f"no review decisions in {world_dir}")


def build(benchmark_id: str, phase: str, world_dirs: Sequence[Path], *, rng_seed: int,
          development: str | None = None) -> dict[str, Any]:
    """Select and write a benchmark of the phase's configured size (and, for the final
    phase, its invariance probes) from worlds whose directories hold the tables, the
    manifest and the incumbent's review decisions (with their context rows). Worlds
    outside the phase's seeds or families are refused, and so is a definition that
    does not have the configured shape (:func:`harness.check_shape`)."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    rules, protocol = PHASES[phase], load_protocol()
    sizes = harness.SIZES
    allowed_seeds = set(getattr(protocol, rules["seeds"]))
    pools, worlds = [], {}
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
        pools.append(candidates(seed, family, decisions, tables))
        worlds[(seed, family)] = (tables, decisions)
    pool = pd.concat(pools, ignore_index=True)
    exclude: dict[str, set[str]] = {"accounts": set(), "episodes": set()}
    if phase == "final":
        if development is None:
            raise ValueError("a final cohort is checked against its development cohort")
        earlier = json.loads((BENCHMARKS / development / "benchmark.json").read_text())
        if earlier["phase"] != "development":
            raise ValueError(f"{development} is not a development benchmark")
        exclude = {"accounts": {case["account"] for case in earlier["cases"]},
                   "episodes": {case["episode"] for case in earlier["cases"]
                                if case["episode"]}}
    n_cases = int(sizes["development_cases" if phase == "development" else "final_cases"])
    chosen = select(pool, n_cases, rng_seed=rng_seed, exclude_accounts=exclude["accounts"],
                    exclude_episodes=exclude["episodes"])
    if phase == "final":
        earlier_frame = pd.DataFrame({"account_key": sorted(exclude["accounts"]),
                                      "episode_key": None})
        check_cohorts(earlier_frame, chosen, protocol.final_seeds)
    packets: dict[str, dict[str, Any]] = {}
    for (seed, family), (tables, decisions) in worlds.items():
        mine = chosen[(chosen["seed"] == seed) & (chosen["family"] == family)]
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
    staging = BENCHMARKS / f".{benchmark_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    definition = write_benchmark(staging, benchmark_id=benchmark_id, phase=phase,
                                 cases=chosen, packets=packets, arms=arms, probes=probes)
    try:
        harness.check_shape(definition, sizes, protocol)
    except harness.ShapeError:
        shutil.rmtree(staging)
        raise
    target = BENCHMARKS / benchmark_id
    if target.exists():
        shutil.rmtree(staging)
        raise FileExistsError(f"benchmark {benchmark_id} already exists")
    staging.rename(target)
    return definition


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and write a memo benchmark.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--world", type=Path, action="append", required=True,
                        help="a world directory holding review_decisions.csv or .pkl")
    parser.add_argument("--development", help="the development benchmark's id (final phase)")
    parser.add_argument("--rng-seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    definition = build(args.id, args.phase, args.world, rng_seed=args.rng_seed,
                       development=args.development)
    strata = pd.Series([case["stratum"] for case in definition["cases"]]).value_counts()
    print(f"{args.id}: {len(definition['cases'])} cases")
    print(strata.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
