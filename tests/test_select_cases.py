"""Benchmark case selection: strata for every pattern and benign behaviour, at most two
cases per linked group of accounts and episodes, two-phase sampling weights, phases
held to their seeds, and development and final cohorts kept apart."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from core.actions import Check, CheckOutcome
from core.evidence import CheckResult
from llm.eval import harness, select_cases
from llm.eval.select_cases import (
    MAX_PER_CLUSTER,
    candidates,
    check_cohorts,
    linked_groups,
    probe_cases,
    select,
)


def world(seed: int, n_accounts: int = 30) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Review decisions in a small synthetic world: three orders per account; accounts
    0-9 a stolen-card ring (episodes of five accounts), 10-14 travellers, 15-19 new
    customers, the rest ordinary customers."""
    attempts, latent_orders, latent_accounts, decisions = [], [], [], []
    order_id = 0
    for user in range(n_accounts):
        profile = "traveller" if 10 <= user < 15 else "new_customer" if 15 <= user < 20 else None
        latent_accounts.append({"user_id": user, "profile": profile})
        for k in range(3):
            order_id += 1
            fraud = user < 10
            attempts.append({"order_id": order_id, "user_id": user})
            latent_orders.append({"order_id": order_id,
                                  "pattern_id": "P-STOLEN" if fraud else None,
                                  "episode_id": user // 5 if fraud else None,
                                  "intent": "fraud" if fraud else "legitimate",
                                  "mimic": None})
            decisions.append({"order_id": order_id,
                              "decision_at": pd.Timestamp("2025-06-01") + pd.Timedelta(hours=k)})
    tables = {"order_attempts": pd.DataFrame(attempts),
              "latent_orders": pd.DataFrame(latent_orders),
              "latent_accounts": pd.DataFrame(latent_accounts)}
    return pd.DataFrame(decisions), tables


def pool(seed: int, family: str = "baseline") -> pd.DataFrame:
    decisions, tables = world(seed)
    return candidates(seed, family, decisions, tables)


def test_strata_name_patterns_and_benign_behaviour() -> None:
    frame = pool(1041)
    assert set(frame["stratum"]) == {"P-STOLEN", "legitimate:traveller",
                                     "legitimate:new_customer", "legitimate:other"}
    groups = pd.Series(linked_groups(frame), index=frame.index)
    stolen = groups[frame["stratum"] == "P-STOLEN"]
    assert stolen.nunique() == 2  # two episodes of five accounts each, not ten accounts
    assert groups[frame["stratum"] != "P-STOLEN"].nunique() == 20  # one per account
    assert set(frame.loc[frame["stratum"] != "P-STOLEN", "latent_class"]) == {"legitimate"}
    assert frame["case_id"].is_unique


def test_benign_strata_are_the_rarest_trait_of_combined_labels() -> None:
    # simulated worlds join several traits in one mimic or profile label
    labels = ([("profile", "existing")] * 6 + [("profile", "existing+traveller")] * 3
              + [("profile", "new_customer")] * 4 + [("profile", "household+new_customer")] * 2
              + [("mimic", "gift+new_customer_first_order")] * 2
              + [("mimic", "new_customer_first_order")] * 5 + [("profile", None)])
    attempts, latent_orders, latent_accounts, decisions = [], [], [], []
    for order_id, (source, label) in enumerate(labels, start=1):
        attempts.append({"order_id": order_id, "user_id": order_id})
        latent_accounts.append({"user_id": order_id,
                                "profile": label if source == "profile" else "existing"})
        latent_orders.append({"order_id": order_id, "pattern_id": None, "episode_id": None,
                              "intent": "legitimate",
                              "mimic": label if source == "mimic" else None})
        decisions.append({"order_id": order_id, "decision_at": pd.Timestamp("2025-06-01")})
    frame = candidates(7, "baseline", pd.DataFrame(decisions), {
        "order_attempts": pd.DataFrame(attempts), "latent_orders": pd.DataFrame(latent_orders),
        "latent_accounts": pd.DataFrame(latent_accounts)})
    assert list(frame["stratum"].str.removeprefix("legitimate:")) == (
        ["existing"] * 6 + ["traveller"] * 3 + ["new_customer"] * 4 + ["household"] * 2
        + ["gift"] * 2 + ["new_customer_first_order"] * 5 + ["other"])


def test_more_strata_than_cases_are_refused() -> None:
    frame = pd.DataFrame({"case_id": [f"c{i}" for i in range(5)],
                          "stratum": [f"s{i}" for i in range(5)],
                          "account_key": [f"1:{i}" for i in range(5)],
                          "episode_key": [None] * 5})
    with pytest.raises(ValueError, match="strata"):
        select(frame, 3, rng_seed=1)
    assert len(select(frame, 5, rng_seed=1)) == 5


def test_an_account_in_two_episodes_links_them() -> None:
    frame = pd.DataFrame({"account_key": ["1:1", "1:1", "1:2", "1:3"],
                          "episode_key": ["1:7", "1:8", "1:8", None]})
    assert linked_groups(frame) == ["group 1:1", "group 1:1", "group 1:1", "group 1:3"]


def test_selection_caps_linked_groups_and_balances_strata() -> None:
    chosen = select(pool(1041), 8, rng_seed=0)
    assert len(chosen) == 8
    assert chosen["account_key"].value_counts().max() <= MAX_PER_CLUSTER
    assert chosen["episode_key"].dropna().value_counts().max() <= MAX_PER_CLUSTER
    assert chosen["cluster"].value_counts().max() <= MAX_PER_CLUSTER
    assert set(chosen["stratum"].value_counts()) == {2}  # equal shares of four strata
    everything = select(pool(1041), 44, rng_seed=0)  # every case phase one can draw
    counts = everything["stratum"].value_counts()
    assert counts["P-STOLEN"] == 2 * MAX_PER_CLUSTER  # two episodes, two cases each
    assert counts["legitimate:traveller"] == 5 * MAX_PER_CLUSTER  # five accounts
    # weights restore the natural mix: weighted stratum shares equal the eligible pool's
    shares = chosen.groupby("stratum")["weight"].sum() / chosen["weight"].sum()
    natural = pool(1041)["stratum"].value_counts(normalize=True)
    assert (shares - natural.loc[shares.index]).abs().max() < 1e-6
    assert select(pool(1041), 8, rng_seed=0).equals(chosen)


def test_a_pool_that_cannot_fill_the_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="44 of the 45"):
        select(pool(1041), 45, rng_seed=0)


def test_weights_undo_the_cluster_cap_inside_a_stratum() -> None:
    """One stratum: 100 failing cases on one account and 100 passing single-case
    accounts. The cap keeps 2 of the 100 failures; their weights stand for all 100."""
    rows = [{"case_id": f"f{i}", "account_key": "9:0", "episode_key": None,
             "stratum": "s", "passes": False} for i in range(100)]
    rows += [{"case_id": f"p{i}", "account_key": f"9:{i + 1}", "episode_key": None,
              "stratum": "s", "passes": True} for i in range(100)]
    chosen = select(pd.DataFrame(rows), 102, rng_seed=3)
    assert (~chosen["passes"]).sum() == 2
    weighted = (chosen["weight"] * chosen["passes"]).sum() / chosen["weight"].sum()
    assert abs(weighted - 0.5) < 1e-12  # the population rate, not 100/102
    product = chosen["first_phase"] * chosen["second_phase"]
    assert (1 / product - chosen["weight"]).abs().max() < 1e-12


def test_two_phase_weights_estimate_totals_when_a_group_spans_strata() -> None:
    """A1, A2 and B1 share an account (capped at two), B2 is alone; two cases are drawn,
    one per stratum. Over many draws, the weighted counts average the true counts."""
    rows = [{"case_id": name, "account_key": "5:1" if name != "B2" else "5:2",
             "episode_key": None, "stratum": name[0]} for name in ("A1", "A2", "B1", "B2")]
    frame = pd.DataFrame(rows)
    totals = {"A": [], "B": []}
    for seed in range(3000):
        chosen = select(frame, 2, rng_seed=seed)
        for stratum in totals:
            totals[stratum].append(chosen.loc[chosen["stratum"] == stratum, "weight"].sum())
    assert abs(sum(totals["A"]) / 3000 - 2) < 0.1  # two A cases in the pool
    assert abs(sum(totals["B"]) / 3000 - 2) < 0.1


def test_excluded_accounts_and_episodes_never_appear() -> None:
    frame = pool(1041)
    chosen = select(frame, 12, rng_seed=1, exclude_accounts={"1041:10", "1041:11"},
                    exclude_episodes={"1041:0"})
    assert not {"1041:10", "1041:11"} & set(chosen["account_key"])
    assert "1041:0" not in set(chosen["episode_key"].dropna())


def test_cohorts_must_not_share_accounts_or_episodes() -> None:
    development = select(pool(1041), 6, rng_seed=0)
    finals = pd.concat([select(pool(seed), 6, rng_seed=0) for seed in (11, 12, 13)])
    check_cohorts(development, finals, final_seeds=(11, 12, 13, 14))
    leaked = pd.concat([finals, development.head(1)])
    with pytest.raises(ValueError):
        check_cohorts(development, leaked, final_seeds=(11, 12, 13, 1041))


def test_final_cases_come_from_three_final_seeds() -> None:
    development = select(pool(1041), 6, rng_seed=0)
    two_seeds = pd.concat([select(pool(seed), 6, rng_seed=0) for seed in (11, 12)])
    with pytest.raises(ValueError):
        check_cohorts(development, two_seeds, final_seeds=(11, 12, 13))
    not_final = pd.concat([select(pool(seed), 6, rng_seed=0) for seed in (11, 12, 99)])
    with pytest.raises(ValueError):
        check_cohorts(development, not_final, final_seeds=(11, 12, 13))


def test_probe_cases_spread_across_strata() -> None:
    chosen = select(pool(1041), 12, rng_seed=0)
    probes = probe_cases(chosen, 4, rng_seed=0)
    assert len(probes) == 4
    assert chosen.set_index("case_id").loc[probes, "stratum"].nunique() == 4


MINI = Path(__file__).resolve().parent / "fixtures" / "mini_world"
QUIET = json.loads((Path(__file__).resolve().parent / "fixtures" / "llm" /
                    "quiet_case.json").read_text())["context_row"]


FINAL_SEEDS = (35244829, 931592514, 2066792625)


def final_run(root: Path, seeds=FINAL_SEEDS, generated=None) -> list[Path]:
    """A final run folder: the baseline world of each seed and a lineage naming the seeds
    the run generated (``generated``, default ``seeds``)."""
    root.mkdir(parents=True)
    (root / "lineage.json").write_text(json.dumps({"seeds": list(generated or seeds)}))
    return [mini_world_with_reviews(root / "worlds" / f"{seed}-baseline", seed)
            for seed in seeds]


def mini_world_with_reviews(directory: Path, seed: int) -> Path:
    """The mini world as a run directory: its tables, a manifest for ``seed`` and review
    decisions an hour after each approved order (context values from the fixture)."""
    shutil.copytree(MINI, directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    (directory / "manifest.json").write_text(json.dumps({**manifest, "seed": seed}))
    attempts = pd.read_csv(directory / "order_attempts.csv")
    approved = attempts[attempts["processor_result"] == "approved"]
    rows = []
    for order in approved.itertuples():
        decision_at = pd.Timestamp(order.known_at) + pd.Timedelta(hours=1)
        checks = ([{"check": "id_check", "outcome": "passed",
                    "completed_at": str(decision_at - pd.Timedelta(minutes=5))}]
                  if order.order_id % 2 else [])
        rows.append({**QUIET, "order_id": order.order_id, "user_id": order.user_id,
                     "merchant_id": order.merchant_id, "decision_at": decision_at,
                     "checks": json.dumps(checks)})
    pd.DataFrame(rows).to_csv(directory / "review_decisions.csv", index=False)
    return directory


@pytest.fixture
def mini_phases(tmp_path, monkeypatch):
    """The mini world's review decisions fall in the validation window; the sizes are
    small enough for it and at least its number of strata."""
    phases = {name: {**rules, "window": "validation"}
              for name, rules in select_cases.PHASES.items()}
    monkeypatch.setattr(select_cases, "PHASES", phases)
    monkeypatch.setattr(select_cases, "BENCHMARKS", tmp_path / "benchmarks")
    protocol = select_cases.load_protocol()  # three final seeds are enough here
    small = replace(protocol, raw={**protocol.raw,
                                   "seeds": {**protocol.raw["seeds"], "minimum_final": 3}})
    monkeypatch.setattr(select_cases, "load_protocol", lambda: small)
    monkeypatch.setattr(harness, "SIZES", {**harness.SIZES, "development_cases": 12,
                                           "final_cases": 18, "final_axes": {"review": 18},
                                           "probe_cases": 3})
    (tmp_path / "benchmarks").mkdir()


def test_benchmarks_are_built_from_worlds_and_their_review_decisions(tmp_path, mini_phases):
    development = select_cases.build(
        "t-dev", "development", [mini_world_with_reviews(tmp_path / "dev", 1041)],
        rng_seed=1)
    bench = tmp_path / "benchmarks" / "t-dev"
    assert len(development["cases"]) == 12
    assert {p.stem for p in (bench / "packets").glob("*.json")} == {
        case["case_id"] for case in development["cases"]}
    views = json.loads((bench / "referee.json").read_text())
    assert set(views) == {case["case_id"] for case in development["cases"]}
    for case in development["cases"]:
        packet = json.loads((bench / "packets" / f"{case['case_id']}.json").read_text())
        assert packet["context"]["account_age_days"] == QUIET["account_age_days"]
        assert "pattern" not in json.dumps(packet) and "P-" not in json.dumps(packet)
    finals = final_run(tmp_path / "final")
    final = select_cases.build("t-final", "final", finals, rng_seed=2, development="t-dev")
    assert len({case["seed"] for case in final["cases"]}) == 3
    assert set(final["probes"]) == {"shuffled", "renamed"}
    assert len(final["probes"]["shuffled"]) == 3
    checked = [json.loads((tmp_path / "benchmarks" / "t-final" / "packets" /
                           f"{case['case_id']}.json").read_text())
               for case in final["cases"]]
    assert any(packet["decision"]["point"] == "check_completed" for packet in checked)

    assert all(case["packet_sha256"] for case in final["cases"])
    assert not list((tmp_path / "benchmarks").glob(".*staging"))
    # every final seed of the run, or nothing
    partial = final_run(tmp_path / "partial", seeds=FINAL_SEEDS[:2], generated=FINAL_SEEDS)
    with pytest.raises(ValueError, match="every final seed"):
        select_cases.build("t-partial", "final", partial, rng_seed=2, development="t-dev")
    with pytest.raises(ValueError, match="one run"):
        select_cases.build("t-mixed", "final", [*partial, finals[2]], rng_seed=2,
                           development="t-dev")


@pytest.mark.parametrize("form", ["triples", "results"])
def test_review_decisions_in_the_pipelines_pickle_are_read(tmp_path, mini_phases, form):
    # the pipeline keeps review decisions as a pickled frame whose checks are lists:
    # (check, outcome, completed_at) triples as the replay keeps them, or check results
    world = mini_world_with_reviews(tmp_path / "dev", 1041)
    frame = pd.read_csv(world / "review_decisions.csv")
    (world / "review_decisions.csv").unlink()

    def item(entry: dict):
        at = pd.Timestamp(entry["completed_at"])
        if form == "triples":
            return (entry["check"], entry["outcome"], at)
        return CheckResult(Check(entry["check"]), CheckOutcome(entry["outcome"]),
                           at.to_pydatetime())

    frame["checks"] = [[item(entry) for entry in json.loads(text)] for text in frame["checks"]]
    frame["decision_at"] = pd.to_datetime(frame["decision_at"])
    frame.to_pickle(world / "review_decisions.pkl")
    development = select_cases.build("t-dev", "development", [world], rng_seed=1)
    bench = tmp_path / "benchmarks" / "t-dev"
    packets = [json.loads((bench / "packets" / f"{case['case_id']}.json").read_text())
               for case in development["cases"]]
    checks = {case["case_id"]: packet["decision"]["checks"]
              for case, packet in zip(development["cases"], packets, strict=True)}
    assert any(checks.values()) and not all(checks.values())
    assert all(check["outcome"] == "passed" for done in checks.values() for check in done)


@pytest.mark.parametrize("phase,seed,family", [
    ("development", 35244829, "baseline"),      # a final seed in development
    ("development", 1041, "fraud_mix_shift"),   # development uses baseline worlds only
    ("final", 1041, "baseline"),                # a development seed in the final cohort
    ("final", 35244829, "lag_half"),            # final cases come from baseline worlds too
])
def test_a_phase_refuses_worlds_outside_its_seeds_and_families(tmp_path, mini_phases,
                                                               phase, seed, family):
    world_dir = mini_world_with_reviews(tmp_path / "w", seed)
    manifest = json.loads((world_dir / "manifest.json").read_text())
    (world_dir / "manifest.json").write_text(json.dumps({**manifest, "family": family}))
    with pytest.raises(ValueError):
        select_cases.build("t-x", phase, [world_dir], rng_seed=1, development="t-dev")
    assert not (tmp_path / "benchmarks" / "t-x").exists()


# ------------------------------------------------------- decisions at check completions


def checked(outcome: str, check: str = "id_check") -> tuple[CheckResult, ...]:
    return (CheckResult(Check(check), CheckOutcome(outcome),
                        pd.Timestamp("2025-06-01 12:00").to_pydatetime()),)


def two_axis_pool() -> pd.DataFrame:
    """The synthetic world's review decisions, and for every second order a decision
    after its id_check: failed for the ring, passed for the others."""
    review = pool(1041)
    later = review.iloc[::2].copy()
    later["axis"] = "check_completed"
    later["case_id"] = later["case_id"] + "-after"
    fraud = later["latent_class"].eq("fraud")
    later["check_outcome"] = fraud.map({True: "failed", False: "passed"})
    later["stratum"] = [
        select_cases.check_stratum(checked(outcome), {"decline" if outcome == "failed"
                                                      else "clear"}, latent)
        for outcome, latent in zip(later["check_outcome"], later["latent_class"], strict=True)]
    return pd.concat([review, later], ignore_index=True)


def test_check_strata_name_the_checks_the_standard_and_the_latent_class() -> None:
    both = checked("passed") + checked("failed", "contact")
    assert select_cases.check_stratum(both, {"escalate"}, "fraud") == \
        "after_check:contact=failed+id_check=passed:escalate:fraud"
    assert select_cases.check_outcome(both) == "failed"
    assert select_cases.check_outcome(checked("passed")) == "passed"
    with pytest.raises(ValueError):
        select_cases.check_outcome(())


def test_the_axes_share_the_cluster_cap_and_each_fills_its_count() -> None:
    frame = two_axis_pool()
    chosen = select(frame, {"review": 8, "check_completed": 4}, rng_seed=0)
    assert chosen["axis"].value_counts().to_dict() == {"review": 8, "check_completed": 4}
    for key in ("account_key", "cluster"):  # two per group over both axes
        assert chosen[key].value_counts().max() <= MAX_PER_CLUSTER
    assert set(chosen.loc[chosen["axis"] == "review", "stratum"]) == {
        "P-STOLEN", "legitimate:traveller", "legitimate:new_customer", "legitimate:other"}
    assert set(chosen.loc[chosen["axis"] == "check_completed", "check_outcome"]) == {
        "passed", "failed"}
    # an int asks for review decisions only, and the other axis's rows play no part
    assert select(frame, 8, rng_seed=0).equals(select(pool(1041), 8, rng_seed=0))
    with pytest.raises(ValueError, match="unknown decision-point axes"):
        select(frame, {"later": 2}, rng_seed=0)


def test_every_check_outcome_gets_its_minimum_or_the_selection_is_refused() -> None:
    sizes = {"p1": 20, "p2": 20, "p3": 20, "f1": 30}
    group = {"p1": "passed", "p2": "passed", "p3": "passed", "f1": "failed"}
    quota = select_cases._allocation(sizes, 20)
    assert quota == {"f1": 5, "p1": 5, "p2": 5, "p3": 5}
    moved = select_cases._with_minimum(quota, sizes, group, 8)
    assert moved["f1"] == 8 and sum(moved.values()) == 20
    assert min(moved.values()) >= 1
    with pytest.raises(ValueError, match="cases after a failed check"):
        select_cases._with_minimum(quota, {**sizes, "f1": 6}, group, 8)
    with pytest.raises(ValueError, match="cannot give"):
        select_cases._with_minimum({"p1": 1, "f1": 1}, {"p1": 5, "f1": 5},
                                   {"p1": "passed", "f1": "failed"}, 2)
    frame = two_axis_pool()
    chosen = select(frame, {"review": 8, "check_completed": 6}, rng_seed=0, min_per_outcome=3)
    assert (chosen.loc[chosen["axis"] == "check_completed", "check_outcome"]
            .value_counts().min() >= 3)


def test_the_axes_are_weighted_by_their_share_of_eligible_decisions() -> None:
    frame = two_axis_pool()
    axes = select_cases.eligible_axes(frame, {"review": 8, "check_completed": 4},
                                      exclude_accounts={"1041:0"})
    # account 0's three review decisions and two decisions after a check are excluded
    assert axes["review"] == {"cases": 8, "eligible": 87, "share": 87 / 130}
    assert axes["check_completed"] == {"cases": 4, "eligible": 43, "share": 43 / 130}
    assert abs(sum(item["share"] for item in axes.values()) - 1) < 1e-12


def subset_source(axis: str, n: int, strata: int, accounts: int) -> list[dict]:
    return [{"case_id": f"{axis}-{i}", "stratum": f"{axis}-s{i % strata}",
             "account": f"1:{i % accounts}", "episode": None, "axis": axis}
            for i in range(n)]


def test_the_subset_rule_goes_round_the_strata_in_hash_order_within_the_cap() -> None:
    sources = {"dev": subset_source("review", 12, 4, 12),
               "dev-checks": subset_source("check_completed", 9, 3, 3)}
    picked = select_cases.subset_cases(sources, {"dev": 4, "dev-checks": 3})
    assert picked == select_cases.subset_cases(sources, {"dev": 4, "dev-checks": 3})
    by_source = {name: [case for source, case in picked if source == name] for name in sources}
    assert len({case["stratum"] for case in by_source["dev"]}) == 4  # one per stratum
    assert len({case["stratum"] for case in by_source["dev-checks"]}) == 3
    accounts = pd.Series([case["account"] for _, case in picked]).value_counts()
    assert accounts.max() <= MAX_PER_CLUSTER
    crowded = {"dev": subset_source("review", 6, 3, 1)}  # one account: two cases at most
    with pytest.raises(ValueError, match="gives 2 of the 3"):
        select_cases.subset_cases(crowded, {"dev": 3})


R03 = {"bin_ip_country_mismatch": 1, "avs_mismatch": 1}  # one Card condition: row (b)


def with_completions(directory: Path, monkeypatch) -> None:
    """Keep the mini world's review decisions as the pipeline's pickle, and stand in for
    the replay: every odd order's id_check answers two hours after its first decision,
    failed for orders divisible by three, passed otherwise."""
    frame = pd.read_csv(directory / "review_decisions.csv")
    frame["decision_at"] = pd.to_datetime(frame["decision_at"])
    frame["checks"] = [[] for _ in range(len(frame))]
    frame.to_pickle(directory / "review_decisions.pkl")

    def completions(world_dir, kept, *, seed, tables=None, **_):
        rows = []
        for row in pd.read_csv(Path(world_dir) / "review_decisions.csv").itertuples():
            if row.order_id % 2 == 0:
                continue
            at = pd.Timestamp(row.decision_at) + pd.Timedelta(hours=2)
            outcome = "failed" if row.order_id % 3 == 0 else "passed"
            rows.append({**QUIET, **R03, "order_id": row.order_id, "user_id": row.user_id,
                         "merchant_id": row.merchant_id, "decision_at": at,
                         "assembled_at": pd.Timestamp(row.decision_at),
                         "checks": (CheckResult(Check("id_check"), CheckOutcome(outcome),
                                                at.to_pydatetime()),),
                         "disposition": "decline" if outcome == "failed" else "clear",
                         "first_decision_at": pd.Timestamp(row.decision_at)})
        return pd.DataFrame(rows)

    monkeypatch.setattr(select_cases.check_points, "completion_decisions", completions)


def test_a_final_cohort_takes_both_axes(tmp_path, mini_phases, monkeypatch) -> None:
    monkeypatch.setattr(harness, "SIZES", {**harness.SIZES, "final_cases": 18,
                                           "final_axes": {"review": 12, "check_completed": 6},
                                           "min_cases_per_check_outcome": 2,
                                           "development_check_cases": 6})
    dev = mini_world_with_reviews(tmp_path / "dev", 1041)
    with_completions(dev, monkeypatch)
    select_cases.build("t-dev", "development", [dev], rng_seed=1)
    select_cases.build("t-dev-checks", "development", [dev], rng_seed=1,
                       axis="check_completed")
    finals = final_run(tmp_path / "final")
    for final in finals:
        with_completions(final, monkeypatch)
    definition = select_cases.build("t-final", "final", finals, rng_seed=2,
                                    development=["t-dev", "t-dev-checks"])
    axes = pd.Series([case["axis"] for case in definition["cases"]]).value_counts()
    assert axes.to_dict() == {"review": 12, "check_completed": 6}
    assert set(definition["axes"]) == {"review", "check_completed"}
    assert abs(sum(item["share"] for item in definition["axes"].values()) - 1) < 1e-12
    bench = tmp_path / "benchmarks" / "t-final"
    views = json.loads((bench / "referee.json").read_text())
    outcomes = []
    for case in definition["cases"]:
        packet = json.loads((bench / "packets" / f"{case['case_id']}.json").read_text())
        if case["axis"] == "check_completed":
            assert packet["decision"]["point"] == "check_completed"
            (done,) = packet["decision"]["checks"]
            outcomes.append(done["outcome"])
            assert case["stratum"] == (f"after_check:id_check={done['outcome']}:"
                                       f"{views[case['case_id']]['standard'][0]}:"
                                       f"{case['latent']['class']}")
        else:
            assert packet["decision"]["point"] == "review"
    assert min(Counter(outcomes).values()) >= 2
    earlier = {case["account"] for name in ("t-dev", "t-dev-checks") for case in json.loads(
        (tmp_path / "benchmarks" / name / "benchmark.json").read_text())["cases"]}
    assert not earlier & {case["account"] for case in definition["cases"]}


def test_a_development_subset_draws_from_both_sets(tmp_path, mini_phases, monkeypatch) -> None:
    monkeypatch.setattr(harness, "SIZES", {**harness.SIZES, "development_check_cases": 6,
                                           "development_combined": {"review": 4,
                                                                    "check_completed": 3}})
    dev = mini_world_with_reviews(tmp_path / "dev", 1041)
    with_completions(dev, monkeypatch)
    select_cases.build("t-dev", "development", [dev], rng_seed=1)
    checks = select_cases.build("t-dev-checks", "development", [dev], rng_seed=1,
                                axis="check_completed")
    assert checks["axes"] == {"check_completed": {"cases": 6, "eligible": 10, "share": 1.0}}
    subset = select_cases.combine("t-dev-both", ["t-dev", "t-dev-checks"], arms=["opus"])
    assert set(subset["arms"]) == {"opus"}
    assert subset["drawn_from"]["counts"] == {"t-dev": 4, "t-dev-checks": 3}
    assert {axis: item["cases"] for axis, item in subset["axes"].items()} == {
        "review": 4, "check_completed": 3}
    reviews = json.loads((tmp_path / "benchmarks" / "t-dev" / "benchmark.json").read_text())[
        "axes"]["review"]["eligible"]
    assert subset["axes"]["check_completed"]["share"] == 10 / (reviews + 10)  # both recorded
    old = tmp_path / "benchmarks" / "t-dev" / "benchmark.json"  # a set from before the axes
    old.write_text(json.dumps({key: value for key, value in json.loads(old.read_text())
                               .items() if key != "axes"}))
    shares = select_cases.combine("t-dev-old", ["t-dev", "t-dev-checks"], arms=["opus"])
    assert shares["axes"]["review"] == {"cases": 4, "eligible": None, "share": 4 / 7}
    root = tmp_path / "benchmarks"
    for case in subset["cases"]:
        source = "t-dev-checks" if case["axis"] == "check_completed" else "t-dev"
        assert (root / "t-dev-both" / "packets" / f"{case['case_id']}.json").read_bytes() == \
            (root / source / "packets" / f"{case['case_id']}.json").read_bytes()
    with pytest.raises(ValueError, match="unknown arms"):
        select_cases.combine("t-dev-other", ["t-dev", "t-dev-checks"], arms=["nobody"])


def test_the_case_memos_are_a_benchmark_of_the_case_worlds_alerts(tmp_path, mini_phases) -> None:
    world_dir = mini_world_with_reviews(tmp_path / "run" / "worlds" / "416-baseline", 416)
    decisions = pd.read_csv(world_dir / "review_decisions.csv").head(3)
    columns = dict.fromkeys(select_cases.check_points.ROW_COLUMNS, 0)  # what the run saved
    inputs = pd.DataFrame([{**columns, **QUIET, "order_id": row.order_id,
                            "user_id": row.user_id, "merchant_id": row.merchant_id,
                            "decision_at": pd.Timestamp(row.decision_at),
                            "file": f"CASE-0{index + 1}", "slot": f"slot{index}"}
                           for index, row in enumerate(decisions.itertuples())])
    definition = select_cases.case_memos("t-cases", tmp_path / "run", inputs=inputs)
    assert definition["phase"] == "cases" and set(definition["arms"]) == {"sol"}
    assert [case["stratum"] for case in definition["cases"]] == sorted(
        f"CASE-0{index + 1}:slot{index}" for index in range(3))
    assert all(case["seed"] == 416 and case["weight"] == 1.0 for case in definition["cases"])
    packet = json.loads((tmp_path / "benchmarks" / "t-cases" / "packets" /
                         f"{definition['cases'][0]['case_id']}.json").read_text())
    assert packet["decision"]["point"] == "review" and not packet["decision"]["checks"]
    shape = {**definition, "cases": [{**case, "seed": 1041}
                                     for case in definition["cases"]]}
    with pytest.raises(harness.ShapeError, match="case world"):
        harness.check_shape(shape)
    with pytest.raises(harness.ShapeError, match="without probes"):
        harness.check_shape({**definition,
                             "probes": {"shuffled": [definition["cases"][0]["case_id"]]}})
    with pytest.raises(harness.ShapeError, match="number 1 to"):
        harness.check_shape(definition, {**harness.SIZES, "case_memos": 2})


def test_an_outcome_that_phase_one_drops_still_counts_for_the_minimum() -> None:
    # three passed and one failed decision share one account: phase one keeps two
    rows = [{"case_id": f"c{i}", "account_key": "3:1", "episode_key": None,
             "axis": "check_completed", "check_outcome": outcome,
             "stratum": f"after_check:id_check={outcome}:x:fraud"}
            for i, outcome in enumerate(["passed", "passed", "passed", "failed"])]
    frame = pd.DataFrame(rows)
    dropped = [seed for seed in range(40)
               if "failed" not in set(select(frame, {"check_completed": 2}, rng_seed=seed)
                                      ["check_outcome"])]
    assert dropped  # some draws keep no failed case
    with pytest.raises(ValueError, match="after a failed check"):
        select(frame, {"check_completed": 2}, rng_seed=dropped[0], min_per_outcome=1)
