"""Generated tables from selected rows of a result table: a cell kept by ``where``,
counts and money added up over seeds, levels averaged, one column spread into several,
and a loud failure on a column that does not exist, a selection that keeps nothing, or a
rate or share added up or averaged instead of pooled."""

from __future__ import annotations

import pytest

from report.render import (
    OUTCOME_LEVELS,
    OUTCOME_TOTALS,
    RenderError,
    Sources,
    label,
    parse_selection,
    render,
)

# Per-seed rows of one replay table, as the pipeline writes them: two seeds, two
# policies, two families; seed 2 of the hybrid policy was not evaluated (no values).
ROWS = [
    {"seed": 1, "family": "baseline", "policy": "incumbent_rules", "evaluated": True,
     "reviews": 10, "net_cents": 1_000, "wait_p50_minutes": 30, "held_share": 0.5,
     "max_backlog": 4, "net_vs_approve_all_cents": 100},
    {"seed": 2, "family": "baseline", "policy": "incumbent_rules", "evaluated": True,
     "reviews": 14, "net_cents": -250, "wait_p50_minutes": 50, "held_share": 0.25,
     "max_backlog": 7, "net_vs_approve_all_cents": -40},
    {"seed": 1, "family": "baseline", "policy": "hybrid", "evaluated": True,
     "reviews": 6, "net_cents": 2_000, "wait_p50_minutes": 20, "held_share": 0.1,
     "max_backlog": 2, "net_vs_approve_all_cents": 300},
    {"seed": 2, "family": "baseline", "policy": "hybrid", "evaluated": False,
     "reviews": None, "net_cents": None, "wait_p50_minutes": None, "held_share": None,
     "max_backlog": None, "net_vs_approve_all_cents": None},
    {"seed": 1, "family": "surge", "policy": "incumbent_rules", "evaluated": True,
     "reviews": 99, "net_cents": 9_999, "wait_p50_minutes": 99, "held_share": 0.9,
     "max_backlog": 9, "net_vs_approve_all_cents": 999},
]
# One confusion table: outcome by label, per seed and evidence strength.
CONFUSION = [
    {"seed": 1, "policy": "incumbent_rules", "truth": "account_takeover", "final": "decline",
     "strength": "1", "orders": 3},
    {"seed": 1, "policy": "incumbent_rules", "truth": "account_takeover", "final": "decline",
     "strength": "2+", "orders": 2},
    {"seed": 2, "policy": "incumbent_rules", "truth": "account_takeover", "final": "clear",
     "strength": "0", "orders": 1},
    {"seed": 2, "policy": "incumbent_rules", "truth": "no_finding", "final": "clear",
     "strength": "0", "orders": 40},
    {"seed": 1, "policy": "hybrid", "truth": "no_finding", "final": "escalate",
     "strength": "1", "orders": 8},
]
PRIMARY = "where family=baseline"


@pytest.fixture
def sources() -> Sources:
    return Sources(summary={"metrics": {}, "tables": {"replay.outcomes": ROWS,
                                                      "replay.confusion": CONFUSION}},
                   labels={"incumbent_rules": "incumbent rules", "hybrid": "hybrid",
                           "service_p1": "P1 service"})


def table(spec: str, sources: Sources) -> list[str]:
    return render("{{ table:" + spec + " }}", sources).split("\n")


def test_where_keeps_the_rows_whose_cells_read_as_given(sources):
    assert table("replay.outcomes where family=baseline policy=incumbent_rules | seed, "
                 "reviews count, evaluated", sources) == [
        "| seed | reviews | evaluated |", "|---|---:|---|",
        "| 1 | 10 | yes |", "| 2 | 14 | yes |"]
    assert table("replay.outcomes where evaluated=false | policy, net_cents usd",
                 sources) == ["| policy | net_cents |", "|---|---:|", "| hybrid | n/a |"]
    assert parse_selection(["where", "seed=1"]).where == (("seed", "1"),)


def test_sum_adds_counts_and_money_over_the_rows_of_each_group(sources):
    # oracle by hand: incumbent 10 + 14 reviews, $10.00 - $2.50, $1.00 - $0.40
    assert table(f"replay.outcomes {PRIMARY} policy=incumbent_rules sum reviews, net_cents, "
                 "net_vs_approve_all_cents by policy | policy, reviews count, net_cents usd:2, "
                 "net_vs_approve_all_cents usd:2, rows_count count", sources) == [
        "| policy | reviews | net_cents | net_vs_approve_all_cents | rows_count |",
        "|---|---:|---:|---:|---:|",
        "| incumbent_rules | 24 | $7.50 | $0.60 | 2 |"]
    assert table("replay.outcomes where seed=1 sum net_cents by family | family, net_cents usd, "
                 "rows_count count", sources)[2:] == [
        "| baseline | $30 | 2 |", "| surge | $100 | 1 |"]


def test_a_total_never_leaves_out_a_world_the_replay_did_not_evaluate(sources):
    # hybrid has no operating point on seed 2: a total over its worlds would be seed 1's
    # alone, so any aggregation the hybrid's seed-2 world falls inside fails, whatever
    # else it filters on; rows of other tables for that world simply do not exist
    for spec in (f"replay.outcomes {PRIMARY} sum reviews by policy | policy",
                 "replay.outcomes where family=baseline evaluated=true sum net_cents by policy "
                 "| policy",
                 "replay.confusion sum orders by truth across final | truth",
                 "replay.confusion where policy=hybrid sum orders by truth | truth"):
        with pytest.raises(RenderError, match="'hybrid' was not evaluated on seed 2"):
            table(spec, sources)
    # a selection that excludes the world is fine, and so is listing rows
    assert table("replay.outcomes where family=baseline seed=1 sum reviews by policy | "
                 "policy, reviews count", sources)[2:] == ["| incumbent_rules | 10 |",
                                                           "| hybrid | 6 |"]
    assert table("replay.outcomes where evaluated=false | policy, reviews", sources)[2:] == [
        "| hybrid | n/a |"]


def test_mean_averages_levels_with_the_rows_it_covers(sources):
    # (30 + 50) / 2 minutes and (4 + 7) / 2 orders waiting at most
    assert table(f"replay.outcomes {PRIMARY} policy=incumbent_rules mean wait_p50_minutes, "
                 "max_backlog by policy | policy, wait_p50_minutes count, max_backlog num:1, "
                 "rows_count", sources)[2:] == ["| incumbent_rules | 40 | 5.5 | 2 |"]


@pytest.mark.parametrize("clause, refused", [
    ("sum held_share", "is not a count or money column"),
    ("mean held_share", "is not a count or money column"),
    ("sum wait_p50_minutes", "is a per-world level, which does not add up"),
    ("sum max_backlog", "is a per-world level, which does not add up"),
    ("mean reviews", "adds up over worlds: use sum"),
    ("mean net_cents", "adds up over worlds: use sum"),
])
def test_rates_shares_and_quantiles_are_never_added_up_or_averaged(sources, clause, refused):
    with pytest.raises(RenderError, match=refused):
        table(f"replay.outcomes {PRIMARY} {clause} by policy | policy", sources)


def test_tables_already_pooled_over_seeds_are_never_aggregated(sources):
    # per-cell figures (a mean over seeds, a pooled rate) look like money and counts but
    # adding them up again would count each world's cash a second time
    cells = [{"policy": "hybrid", "capacity": "base", "net_contribution_cents": 2_500,
              "decided_after_shipping_count": 3.5},
             {"policy": "hybrid", "capacity": "high", "net_contribution_cents": 2_500,
              "decided_after_shipping_count": 2.5}]
    pooled = Sources(summary={"metrics": {}, "tables": {"evaluate.policies": cells}})
    for clause in ("sum net_contribution_cents", "mean decided_after_shipping_count"):
        with pytest.raises(RenderError, match="already pooled or averaged over seeds"):
            table(f"evaluate.policies {clause} by policy | policy", pooled)
    assert table("evaluate.policies where capacity=high | policy, net_contribution_cents usd",
                 pooled)[2:] == ["| hybrid | $25 |"]


def test_the_aggregable_columns_are_the_replay_rows_columns():
    from queue_sim.outcomes import OUTCOME_COLUMNS

    assert set(OUTCOME_TOTALS) | set(OUTCOME_LEVELS) == set(OUTCOME_COLUMNS)
    assert not set(OUTCOME_TOTALS) & set(OUTCOME_LEVELS)


def test_across_spreads_one_column_into_columns_holding_its_sum(sources):
    # oracle by hand: takeover 3 + 2 declined (two strengths) and 1 cleared; no finding
    # 40 cleared; an absent combination is zero
    assert table("replay.confusion where policy=incumbent_rules sum orders by truth "
                 "across final | truth \"Label\" label, decline count, clear count",
                 sources) == [
        "| Label | decline | clear |", "|---|---:|---:|",
        "| account takeover | 5 | 1 |", "| no finding | 0 | 40 |"]
    with pytest.raises(RenderError, match="no column 'escalate'"):  # not among the kept rows
        table("replay.confusion where policy=incumbent_rules sum orders by truth across final "
              "| truth, escalate", sources)
    # each spread row says how many rows it covers
    assert table("replay.confusion where seed=1 sum orders by policy across final | policy, "
                 "decline count, escalate count, rows_count count", sources)[2:] == [
        "| incumbent_rules | 5 | 0 | 2 |", "| hybrid | 0 | 8 | 1 |"]


@pytest.mark.parametrize("rows, message", [
    # two values that print alike would put two totals in one column
    ([{"k": "a", "strength": 1, "orders": 3}, {"k": "a", "strength": "1", "orders": 5}],
     "print as the same column name"),
    ([{"k": "a", "strength": "2+", "orders": 3}], r"\['2\+'\] cannot name a column"),
    ([{"k": "a", "strength": "rows_count", "orders": 3}], "clash with the groups"),
    ([{"k": "a", "strength": "k", "orders": 3}], "clash with the groups"),
])
def test_spread_columns_must_be_distinct_names(rows, message):
    spread = Sources(summary={"metrics": {}, "tables": {"replay.confusion": rows}})
    with pytest.raises(RenderError, match=message):
        table("replay.confusion sum orders by k across strength | k", spread)


@pytest.mark.parametrize("spec, message", [
    ("replay.outcomes where family=nowhere | policy", "no rows match where family=nowhere"),
    ("replay.outcomes where familly=baseline | policy", "no column 'familly'"),
    ("replay.outcomes where family | policy", "is not column=value"),
    ("replay.outcomes sum reviews | policy", "sum needs by"),
    ("replay.outcomes by policy | policy", "by and across follow sum or mean"),
    ("replay.outcomes sum reviews by policy where family=baseline | policy",
     "cannot follow 'by'"),
    ("replay.outcomes sum reviews mean net_cents by policy | policy", "cannot follow 'sum'"),
    ("replay.outcomes mean reviews by policy across seed | policy", "across spreads a sum"),
    ("replay.outcomes sum reviews, net_cents by policy across seed | policy",
     "sums one column"),
    ("replay.outcomes sum policy by seed | seed", "'policy' is not a count or money column"),
    ("replay.outcomes sum reviews by reviews | reviews", "both group and are aggregated"),
    ("replay.outcomes where seed=1 sum reviews by policy | policy, seed", "no column 'seed'"),
    ("replay.outcomes filter family=baseline | policy", "is not a clause"),
])
def test_a_selection_that_cannot_mean_what_it_says_fails(sources, spec, message):
    with pytest.raises((RenderError, ValueError), match=message):
        table(spec, sources)


def test_labels_print_identifiers_in_words(sources):
    assert label("incumbent_rules", sources.labels) == "incumbent rules"
    assert label("tree_depth3", {}) == "tree depth3"
    assert label("service_p1, service_p2", sources.labels) == "P1 service, service p2"
    assert label("", sources.labels) == "none"
    with pytest.raises(RenderError, match="prints identifiers, not numbers"):
        table("replay.outcomes | reviews label", sources)


def test_a_table_without_clauses_is_unchanged(sources):
    assert parse_selection(["where", "a=b"]).where == (("a", "b"),)
    assert table("replay.outcomes | policy", sources) == [
        "| policy |", "|---|", *[f"| {row['policy']} |" for row in ROWS]]


def test_a_cell_that_is_not_a_number_fails_the_sum():
    rows = [{"policy": "hybrid", "reviews": "many"}]
    odd = Sources(summary={"metrics": {}, "tables": {"replay.outcomes": rows}})
    with pytest.raises(RenderError, match="holds 'many', not a number"):
        table("replay.outcomes sum reviews by policy | policy", odd)


def test_identifiers_print_as_they_are(sources):
    seeds = [{"seed": 35244829, "policy": "hybrid", "net_cents": 10, "reviews": 3}]
    ids = Sources(summary={"metrics": {}, "tables": {"replay.outcomes": seeds}})
    assert table("replay.outcomes | seed id, policy id", ids)[2:] == ["| 35244829 | hybrid |"]
    for column in ("net_cents", "seed id:0"):
        with pytest.raises(RenderError, match="the id format prints an identifier"):
            table(f"replay.outcomes | {column if ' ' in column else column + ' id'}", ids)


def test_groups_keep_one_order_across_selections_from_a_table():
    rows = [{"policy": "a", "truth": "x", "final": "clear", "orders": 1},
            {"policy": "a", "truth": "y", "final": "clear", "orders": 2},
            {"policy": "b", "truth": "y", "final": "clear", "orders": 3},
            {"policy": "b", "truth": "x", "final": "clear", "orders": 4}]
    ordered = Sources(summary={"metrics": {}, "tables": {"replay.confusion": rows}})
    for policy, first, second in (("a", 1, 2), ("b", 4, 3)):
        lines = table(f"replay.confusion where policy={policy} sum orders by truth "
                      "| truth, orders count", ordered)
        assert lines[2:] == [f"| x | {first} |", f"| y | {second} |"]
