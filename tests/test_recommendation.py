"""The pre-registered recommendation rule on synthetic outcome rows: eligibility, the
fallback when the incumbent misses, the hurdle, ties, the analyst allotment, the LTV
cells and the flip table."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from core import config
from core import recommendation as rec
from core.results import StageResult

REPO = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(1, 11))
POLICIES = ("approve_all", "incumbent_rules", "tree_depth3", "logistic", "boosting",
            "hybrid", "expected_loss")
WHERE = {"capacity_level": "base", "layout": "current", "history": "policy",
         "reviewer": "evidence", "verification": "verification"}
LOW = WHERE | {"capacity_level": "low"}
PRIMARY = rec.OperatingCell("primary", "primary", "baseline", "base", WHERE)
RULE = rec.Rule(
    lost=rec.Cap(Fraction(100), Fraction(200)), held=rec.Cap(Fraction(300), Fraction(600)),
    service_share=Fraction(9, 10), service_min_entries=50, hurdle_cents=Fraction(10_000),
    positive_seeds={10: 9, 9: 8, 8: 8}, simpler=POLICIES,
    analyst_cents_per_hour=Fraction(3_500), ltv_cents=1_500, ltv_sensitivity_cents=(500, 4_500))
INCUMBENT_NET = 1_000_000  # 600 allotted minutes cost 35,000: rule net 965,000


def row(seed: int, policy: str, net: int, *, where=WHERE, family: str = "baseline",
        orders: int = 10_000, legit: int = 9_000, declined: int = 0, cancelled: int = 0,
        held: int = 0, band: int = 1, available: int = 600, minutes: int = 300,
        entries=(0, 0, 0, 0), met=None) -> dict:
    """One outcome row as queue_sim.outcomes writes it, at the configured $15 LTV proxy."""
    met = entries if met is None else met
    return {
        "seed": seed, "family": family, "policy": policy, **where, "evaluated": True,
        "orders": orders, "net_cents": net, "friction_cost_cents": 1_500 * (declined + cancelled),
        "legitimate_orders": legit, "legitimate_held": held, "legitimate_declined": declined,
        "legitimate_cancelled": cancelled, "available_minutes": available,
        "review_band": band, "review_minutes_used": minutes,
        **{f"reviews_p{i}": n for i, n in enumerate(entries)},
        **{f"sla_met_p{i}": n for i, n in enumerate(met)},
    }


def rows_for(spec: dict[str, dict], seeds=SEEDS, **common) -> list[dict]:
    """``spec``: policy -> row arguments, each a constant or one value per seed."""
    out = []
    for i, seed in enumerate(seeds):
        for policy, arguments in spec.items():
            values = {k: v[i] if isinstance(v, list) else v for k, v in arguments.items()}
            out.append(row(seed, policy, **(common | values)))
    return out


def judge(rows: list[dict], cell=PRIMARY, policies=None) -> rec.CellResult:
    present = tuple(dict.fromkeys(r["policy"] for r in rows))
    return rec.apply_rule(rows, [cell], RULE, policies or present)[0]


def with_incumbent(**challengers) -> list[dict]:
    """The incumbent at 1,000,000 cents on every seed, and challengers at the given nets
    (an int, or a dict of row arguments)."""
    spec = {"incumbent_rules": {"net": INCUMBENT_NET}}
    for policy, given in challengers.items():
        spec[policy] = given if isinstance(given, dict) else {"net": given}
    return rows_for(spec)


# ---------------------------------------------------------------- the protocol
def test_the_protocol_states_the_rule_as_pre_registered() -> None:
    raw = yaml.safe_load((REPO / "experiments" / "protocol.yaml").read_text())
    rule = rec.Rule.from_protocol(raw, config.load("policy"))
    assert rule == RULE
    assert rule.needed(10) == 9 and rule.needed(9) == 8 and rule.needed(8) == 8
    assert rule.needed(3) == 3  # fewer seeds than the rule lists: every seed
    with pytest.raises(rec.RuleError, match="no sign bar"):
        rule.needed(11)
    costly = config.load("policy")
    costly["costs"]["analyst_loaded_hourly_usd"] = 40.0
    with pytest.raises(rec.RuleError, match="analyst hour"):
        rec.Rule.from_protocol(raw, costly)
    incomplete = yaml.safe_load(yaml.safe_dump(raw))
    del incomplete["reporting"]["recommendation_rule"]["hurdle"]
    with pytest.raises(rec.RuleError, match="incomplete"):
        rec.Rule.from_protocol(incomplete, config.load("policy"))
    garbled = yaml.safe_load(yaml.safe_dump(raw))
    garbled["reporting"]["recommendation_rule"]["hurdle"]["positive_seeds"] = "nine of ten"
    with pytest.raises(rec.RuleError, match="incomplete"):
        rec.Rule.from_protocol(garbled, config.load("policy"))
    loose = yaml.safe_load(yaml.safe_dump(raw))
    loose["reporting"]["recommendation_rule"]["eligibility"]["service_share_at_least"] = 1.5
    with pytest.raises(rec.RuleError, match="service floor"):
        rec.Rule.from_protocol(loose, config.load("policy"))


# ---------------------------------------------------------------- the choice
def test_a_challenger_that_clears_the_hurdle_is_recommended() -> None:
    result = judge(with_incumbent(hybrid=1_150_000, boosting=1_120_000))
    assert (result.outcome, result.chosen, result.best) == (rec.RECOMMEND, "hybrid", "hybrid")
    hybrid = result.verdicts["hybrid"]
    assert hybrid.mean == 15_000  # 150,000 cents more on 10,000 orders: $150 per 1,000
    assert (hybrid.positive, hybrid.needed, hybrid.clears, hybrid.eligible) == (10, 9, True, True)
    assert result.incumbent_misses == ()


def test_the_analyst_allotment_is_charged_unless_the_policy_has_no_review_route() -> None:
    # approve-all releases the allotment: 1,000,000 against the incumbent's 865,000
    rows = rows_for({"incumbent_rules": {"net": 900_000},
                     "approve_all": {"net": 1_000_000, "band": 0, "minutes": 0}})
    result = judge(rows)
    assert result.chosen == "approve_all" and result.verdicts["approve_all"].mean == 13_500
    assert result.standings["incumbent_rules"].net[1] == 900_000 - 35_000
    # unused minutes count: the allotment, not the minutes used
    rows = rows_for({"incumbent_rules": {"net": INCUMBENT_NET, "minutes": 10},
                     "hybrid": {"net": INCUMBENT_NET, "band": 0, "minutes": 0}})
    assert judge(rows).verdicts["hybrid"].mean == 3_500  # the 35,000 released, per 1,000


def test_approve_all_is_eligible_and_has_no_queue_to_assess() -> None:
    rows = rows_for({"incumbent_rules": {"net": 900_000, "entries": (10, 10, 10, 10)},
                     "approve_all": {"net": 1_000_000, "band": 0}})
    result = judge(rows)
    approve = result.standings["approve_all"]
    assert all(not s.assessed and s.share is None for s in approve.service.values())
    assert result.verdicts["approve_all"].eligible and result.chosen == "approve_all"


def test_ties_go_to_fewer_lost_customers_then_fewer_minutes_then_the_simpler_policy() -> None:
    # equal improvement; boosting loses fewer customers (tree's extra net offsets its LTV)
    rows = with_incumbent(tree_depth3={"net": 1_150_000 + 15_000, "declined": 10},
                          boosting={"net": 1_150_000})
    result = judge(rows)
    assert result.verdicts["tree_depth3"].mean == result.verdicts["boosting"].mean
    assert result.chosen == "boosting"
    # customers come before minutes: fewer minutes do not make up for more lost customers
    rows = with_incumbent(tree_depth3={"net": 1_150_000 + 15_000, "declined": 10,
                                       "minutes": 100},
                          boosting={"net": 1_150_000, "minutes": 900})
    assert judge(rows).chosen == "boosting"
    # equal customers: fewer review minutes
    rows = with_incumbent(tree_depth3={"net": 1_150_000, "minutes": 400},
                          boosting={"net": 1_150_000, "minutes": 350})
    assert judge(rows).chosen == "boosting"
    # all equal: the simpler policy
    rows = with_incumbent(boosting=1_150_000, tree_depth3=1_150_000, hybrid=1_150_000)
    assert judge(rows).chosen == "tree_depth3"
    # a strictly higher mean wins whatever the tie-breaks say
    rows = with_incumbent(tree_depth3=1_150_000, boosting={"net": 1_150_001, "minutes": 999})
    assert judge(rows).chosen == "boosting"


# ---------------------------------------------------------------- eligibility
def test_a_challenger_over_a_guardrail_is_not_recommended() -> None:
    # hybrid clears the hurdle but loses 101.1 legitimate customers per 10,000 on average
    rows = with_incumbent(hybrid={"net": 1_300_000, "declined": 91}, boosting=1_050_000)
    result = judge(rows)
    assert result.verdicts["hybrid"].clears and result.verdicts["hybrid"].failed == ("lost_mean",)
    assert (result.outcome, result.chosen) == (rec.STAYS, None)
    assert result.best == "boosting"  # the best eligible challenger is named
    # exactly at the cap is within it
    rows = with_incumbent(hybrid={"net": 1_300_000, "declined": 90})
    assert judge(rows).chosen == "hybrid"
    # one seed over the per-seed cap, though the mean is far below
    rows = with_incumbent(hybrid={"net": 1_400_000, "declined": [0] * 9 + [181]})
    assert judge(rows).verdicts["hybrid"].failed == ("lost_any_seed",)
    # cancelled after an unanswered hold counts as lost
    rows = with_incumbent(hybrid={"net": 1_300_000, "cancelled": 91})
    assert judge(rows).verdicts["hybrid"].failed == ("lost_mean",)
    # held: 301.1 per 10,000 on the mean, then 601.1 on one seed
    rows = with_incumbent(hybrid={"net": 1_300_000, "held": 271})
    assert judge(rows).verdicts["hybrid"].failed == ("held_mean",)
    rows = with_incumbent(hybrid={"net": 1_300_000, "held": [0] * 9 + [541]})
    assert judge(rows).verdicts["hybrid"].failed == ("held_any_seed",)
    # when no challenger is eligible the best of them is still named
    rows = with_incumbent(hybrid={"net": 1_300_000, "held": 271})
    assert judge(rows).best == "hybrid"


def test_service_is_the_mean_share_over_seeds_and_assessed_from_fifty_entries() -> None:
    queue = {"entries": (10, 10, 10, 10)}
    late = {"net": 1_200_000, "entries": (10, 10, 10, 10), "met": (8, 10, 10, 10)}
    rows = rows_for({"incumbent_rules": {"net": INCUMBENT_NET, **queue}, "hybrid": late})
    result = judge(rows)
    assert result.verdicts["hybrid"].failed == ("service_p0",)
    # pooled 95 of 100 in time, but the mean over seeds of each seed's share is 0.5
    uneven = {"net": 1_200_000, "entries": [(1, 0, 0, 0)] * 5 + [(19, 0, 0, 0)] * 5,
              "met": [(0, 0, 0, 0)] * 5 + [(19, 0, 0, 0)] * 5}
    rows = rows_for({"incumbent_rules": {"net": INCUMBENT_NET, **queue}, "hybrid": uneven})
    service = judge(rows).standings["hybrid"].service["p0"]
    assert (service.entries, service.share, service.assessed) == (100, Fraction(1, 2), True)
    # 49 entries pooled: reported, not assessed; 50: assessed
    few = {"net": 1_200_000, "entries": [(5, 0, 0, 0)] * 9 + [(4, 0, 0, 0)],
           "met": (0, 0, 0, 0)}
    rows = rows_for({"incumbent_rules": {"net": INCUMBENT_NET, **queue}, "hybrid": few})
    result = judge(rows)
    assert not result.standings["hybrid"].service["p0"].assessed
    assert result.chosen == "hybrid"
    rows = rows_for({"incumbent_rules": {"net": INCUMBENT_NET, **queue},
                     "hybrid": few | {"entries": (5, 0, 0, 0)}})
    assert judge(rows).verdicts["hybrid"].failed == ("service_p0",)


def test_a_criterion_the_incumbent_fails_becomes_no_worse_than_the_incumbent() -> None:
    # the incumbent loses 120 customers per 10,000 and decides 80% of P1 in time
    incumbent = {"net": INCUMBENT_NET, "declined": 108, "entries": (0, 10, 0, 0),
                 "met": (0, 8, 0, 0)}
    rows = rows_for({
        "incumbent_rules": incumbent,
        "hybrid": {"net": 1_300_000, "declined": 99, "entries": (0, 20, 0, 0),
                   "met": (0, 17, 0, 0)},  # 110 lost, 85% in time: no worse
        "boosting": {"net": 1_400_000, "declined": 117, "entries": (0, 10, 0, 0),
                     "met": (0, 8, 0, 0)},  # 130 lost: worse than the incumbent
        "logistic": {"net": 1_400_000, "declined": 99, "entries": (0, 20, 0, 0),
                     "met": (0, 15, 0, 0)},  # 75% in time: worse than the incumbent
    })
    result = judge(rows)
    assert result.incumbent_misses == ("lost_mean", "service_p1")
    assert result.verdicts["hybrid"].eligible
    assert result.verdicts["boosting"].failed == ("lost_mean",)
    assert result.verdicts["logistic"].failed == ("service_p1",)
    assert result.chosen == "hybrid"


# ---------------------------------------------------------------- the hurdle
def test_the_hurdle_needs_the_mean_and_strictly_positive_seeds() -> None:
    def verdict(nets: list[int], seeds=SEEDS) -> rec.Verdict:
        rows = rows_for({"incumbent_rules": {"net": INCUMBENT_NET}, "hybrid": {"net": nets}},
                        seeds=seeds)
        return judge(rows).verdicts["hybrid"]

    exactly = verdict([INCUMBENT_NET + 100_000] * 10)
    assert exactly.mean == RULE.hurdle_cents and exactly.clears
    assert not verdict([INCUMBENT_NET + 99_990] * 10).clears  # $99.99 per 1,000
    # 9 of 10 positive at a mean of exactly the hurdle
    nine = verdict([INCUMBENT_NET + 112_000] * 9 + [INCUMBENT_NET - 8_000])
    assert (nine.positive, nine.mean, nine.clears) == (9, 10_000, True)
    eight = verdict([INCUMBENT_NET + 200_000] * 8 + [INCUMBENT_NET - 1] * 2)
    assert (eight.positive, eight.clears) == (8, False)
    zeros = verdict([INCUMBENT_NET + 150_000] * 8 + [INCUMBENT_NET] * 2)  # zero: not positive
    assert (zeros.positive, zeros.clears) == (8, False)
    # fewer seeds than the rule lists: every one positive
    three = verdict([INCUMBENT_NET + 150_000] * 2 + [INCUMBENT_NET - 1], seeds=(1, 2, 3))
    assert (three.needed, three.clears) == (3, False)
    with pytest.raises(rec.RuleError, match="no sign bar"):
        verdict([INCUMBENT_NET] * 11, seeds=tuple(range(1, 12)))


def test_a_seed_without_a_feasible_point_leaves_fewer_seeds_to_pair() -> None:
    rows = [r for r in with_incumbent(hybrid=[INCUMBENT_NET + 150_000] * 8
                                      + [INCUMBENT_NET - 1] * 2)
            if not (r["policy"] == "hybrid" and r["seed"] == 10)]
    verdict = judge(rows).verdicts["hybrid"]
    assert (len(verdict.improvement), verdict.needed, verdict.positive, verdict.clears) == \
        (9, 8, 8, True)
    # a policy unfit on every seed is listed as absent; an absent incumbent leaves the
    # cell unassessed
    result = rec.apply_rule(with_incumbent(hybrid=1_200_000), [PRIMARY], RULE, POLICIES)[0]
    assert set(result.absent) == set(POLICIES) - {"incumbent_rules", "hybrid"}
    no_incumbent = [r for r in with_incumbent(hybrid=1_200_000)
                    if r["policy"] != "incumbent_rules"]
    result = rec.apply_rule(no_incumbent, [PRIMARY], RULE, POLICIES)[0]
    assert result.outcome == rec.NOT_ASSESSED and "incumbent" in result.note


def test_rows_the_rule_cannot_read_are_refused() -> None:
    rows = with_incumbent(hybrid=1_200_000)
    with pytest.raises(rec.RuleError, match="friction_cost_cents"):
        judge([r | {"friction_cost_cents": 1} for r in rows])
    with pytest.raises(rec.RuleError, match="lack columns"):
        judge([{k: v for k, v in r.items() if k != "review_band"} for r in rows])
    with pytest.raises(rec.RuleError, match="different orders"):
        judge(rows[:-1] + [rows[-1] | {"orders": 9_999}])
    with pytest.raises(rec.RuleError, match="two rows"):
        judge(rows + rows[-1:])
    with pytest.raises(rec.RuleError, match="non-negative integer"):
        judge([r | {"review_minutes_used": 1.5} for r in rows])
    with pytest.raises(rec.RuleError, match="in time"):
        judge([r | {"reviews_p0": 1, "sla_met_p0": 2} for r in rows])
    with pytest.raises(rec.RuleError, match="not a protocol policy"):
        rec.apply_rule(rows, [PRIMARY], RULE, ("incumbent_rules",))


# ---------------------------------------------------------------- the operating cells
def test_ltv_cells_recompute_the_friction_cost_from_the_same_rows() -> None:
    rows = with_incumbent(hybrid={"net": 1_170_000, "declined": 45})
    cells = [PRIMARY,
             rec.OperatingCell("ltv_5_usd", "ltv", "baseline", "base", WHERE, 500),
             rec.OperatingCell("ltv_45_usd", "ltv", "baseline", "base", WHERE, 4_500)]
    primary, cheap, dear = rec.apply_rule(rows, cells, RULE, ("incumbent_rules", "hybrid"))
    # 1,170,000 - LTV x 45 - 35,000 against 965,000, per 1,000 of 10,000 orders
    assert primary.verdicts["hybrid"].mean == (1_170_000 - 67_500 - 35_000 - 965_000) / 10
    assert cheap.verdicts["hybrid"].mean == (1_170_000 - 22_500 - 35_000 - 965_000) / 10
    assert dear.verdicts["hybrid"].mean == (1_170_000 - 202_500 - 35_000 - 965_000) / 10
    assert (primary.chosen, cheap.chosen, dear.chosen) == ("hybrid", "hybrid", None)
    assert rec.flip(primary, cheap) == (True, rec.HOLDS)
    assert rec.flip(primary, dear) == (False, rec.NONE_CLEARED)


def test_the_flip_table_names_the_reason() -> None:
    def at(name: str) -> rec.OperatingCell:
        return rec.OperatingCell(name, "allotment", "baseline", name, WHERE | {
            "capacity_level": name})

    def cell_rows(name: str, **challengers) -> list[dict]:
        return [r | {"capacity_level": name} for r in with_incumbent(**challengers)]

    rows = (with_incumbent(hybrid=1_200_000, boosting=1_150_000)
            + cell_rows("same", hybrid=1_200_000)
            + cell_rows("guardrail", hybrid={"net": 1_400_000, "held": 271},
                        boosting=1_150_000)
            + cell_rows("other", hybrid=1_120_000, boosting=1_150_000)
            + cell_rows("nobody", hybrid=1_050_000, boosting=1_050_000))
    cells = [PRIMARY, at("same"), at("guardrail"), at("other"), at("nobody")]
    results = rec.apply_rule(rows, cells, RULE, ("incumbent_rules", "boosting", "hybrid"))
    primary = results[0]
    assert [rec.flip(primary, r) for r in results[1:]] == [
        (True, rec.HOLDS), (False, rec.WINNER_FAILS), (False, rec.OTHER_POLICY),
        (False, rec.NONE_CLEARED)]
    metrics, tables = rec.outputs(results, RULE)
    holds = metrics["evaluate.recommendation.holds"]
    assert (holds.numerator, holds.denominator) == (1, 4)
    flips = {r["cell"]: r for r in tables["evaluate.flips"]}
    assert flips["primary"]["reason"] == "primary" and flips["primary"]["holds"] is None
    assert (flips["guardrail"]["recommended"], flips["guardrail"]["reason"]) == \
        ("boosting", rec.WINNER_FAILS)
    assert flips["nobody"]["outcome"] == rec.STAYS
    assert flips["nobody"]["best_challenger"] == "boosting"  # tied with hybrid: simpler
    # when the incumbent stays in the primary cell, a challenger elsewhere is a flip
    stays = rec.apply_rule(cell_rows("nobody", hybrid=1_050_000)
                           + cell_rows("same", hybrid=1_200_000),
                           [rec.OperatingCell("primary", "primary", "baseline", "nobody",
                                              WHERE | {"capacity_level": "nobody"}),
                            at("same")], RULE, ("incumbent_rules", "hybrid"))
    assert rec.flip(*stays) == (False, rec.OTHER_POLICY)


def test_the_outputs_are_a_valid_result_with_no_p_value() -> None:
    rows = with_incumbent(hybrid=1_200_000, boosting={"net": 1_100_000, "declined": 91},
                          approve_all={"net": 900_000, "band": 0})
    cells = [PRIMARY, rec.OperatingCell("ltv_45_usd", "ltv", "baseline", "base", WHERE, 4_500)]
    results = rec.apply_rule(rows, cells, RULE, POLICIES)
    metrics, tables = rec.outputs(results, RULE)
    StageResult(stage="evaluate", versions={}, inputs={}, metrics=metrics, tables=tables)
    gain = metrics["evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.hybrid"]
    assert gain.value == 20_000 and gain.unit == "cents" and gain.seeds.n == 10
    assert "evaluate.rule_net_per_1000_orders_ltv_45_usd.vs_incumbent_rules.baseline.base." \
        "hybrid" in metrics
    lost = metrics["evaluate.rule_lost_legitimate_per_10k.baseline.base.boosting"]
    assert lost.value == pytest.approx(910 / 9) and lost.unit == "bps"
    assert not any(m.interval is not None for m in metrics.values())
    assert not any("p_value" in key for key in metrics)
    table = tables["evaluate.recommendation"]
    assert len(table) == 2 * len(POLICIES)  # absent policies listed too
    hybrid = next(r for r in table if r["policy"] == "hybrid" and r["cell"] == "primary")
    assert (hybrid["recommended"], hybrid["eligible"], hybrid["clears_hurdle"]) == \
        (True, True, True)
    assert (hybrid["positive_seeds_count"], hybrid["positive_seeds_needed_count"]) == (10, 9)
    boosting = next(r for r in table if r["policy"] == "boosting" and r["cell"] == "primary")
    assert boosting["fails"] == "lost_mean" and boosting["eligible"] is False
    absent = next(r for r in table if r["policy"] == "logistic")
    assert absent["seeds_count"] == 0 and absent["eligible"] is None
    with pytest.raises(rec.RuleError, match="primary"):
        rec.outputs(results[::-1], RULE)
