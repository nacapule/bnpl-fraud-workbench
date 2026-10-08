"""Generated tables from selected rows of a result table: a cell kept by ``where``,
counts and money added up over seeds, levels averaged, one column spread into several,
and a loud failure on a column that does not exist, a selection that keeps nothing, or a
rate or share added up or averaged instead of pooled."""

from __future__ import annotations

import pytest

from report.render import RenderError, Sources, label, parse_selection, render

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


def test_sum_adds_counts_and_money_over_the_rows_of_each_group(sources):
    # oracle by hand: incumbent 10 + 14 reviews, $10.00 - $2.50; hybrid's second seed
    # has nothing measured, so its totals are unknown, not the first seed's alone
    assert table(f"replay.outcomes {PRIMARY} sum reviews, net_cents, net_vs_approve_all_cents "
                 "by policy | policy, reviews count, net_cents usd:2, "
                 "net_vs_approve_all_cents usd:2, rows_count count", sources) == [
        "| policy | reviews | net_cents | net_vs_approve_all_cents | rows_count |",
        "|---|---:|---:|---:|---:|",
        "| incumbent_rules | 24 | $7.50 | $0.60 | 2 |",
        "| hybrid | n/a | n/a | n/a | 2 |"]
    # without the not-evaluated seed, the group shows the one seed it covers
    assert table("replay.outcomes where family=baseline evaluated=true sum net_cents by policy "
                 "| policy, net_cents usd, rows_count", sources)[2:] == [
        "| incumbent_rules | $8 | 2 |", "| hybrid | $20 | 1 |"]


def test_mean_averages_levels_with_the_rows_it_covers(sources):
    # (30 + 50) / 2 minutes and (4 + 7) / 2 orders waiting at most
    assert table(f"replay.outcomes {PRIMARY} policy=incumbent_rules mean wait_p50_minutes, "
                 "max_backlog by policy | policy, wait_p50_minutes count, max_backlog num:1, "
                 "rows_count", sources)[2:] == ["| incumbent_rules | 40 | 5.5 | 2 |"]


@pytest.mark.parametrize("clause, refused", [
    ("sum held_share", "is a rate, share or mean"),
    ("mean held_share", "is a rate, share or mean"),
    ("sum wait_p50_minutes", "is a quantile or an extreme"),
    ("sum max_backlog", "is a quantile or an extreme"),
])
def test_rates_shares_and_quantiles_are_never_added_up_or_averaged(sources, clause, refused):
    with pytest.raises(RenderError, match=refused):
        table(f"replay.outcomes {PRIMARY} {clause} by policy | policy", sources)


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
    ("replay.outcomes sum policy by seed | seed", "holds 'incumbent_rules', not a number"),
    ("replay.outcomes sum reviews by reviews | reviews", "both group and are aggregated"),
    ("replay.outcomes sum reviews by policy | policy, seed", "no column 'seed'"),
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
