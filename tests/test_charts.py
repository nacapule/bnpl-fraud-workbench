"""The figures: every plotted value is a summary key, a rebuild writes the same bytes, and
the repository's figures come only from the committed results."""

from __future__ import annotations

import copy
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib
import pytest
import yaml
from matplotlib.axes import Axes

from core import recommendation as rec
from core.results import (
    Metric,
    SeedSpread,
    StageResult,
    assemble_summary,
    canonical_json,
    read_summary,
)
from report import charts
from report.charts import CHARTS, FIGURES, REPO, SUMMARY, main, out_of_sync, render_all

matplotlib.use("Agg")

SEEDS = (1, 2, 3)
POLICIES = ("approve_all", "incumbent_rules", "tree_depth3", "logistic", "boosting", "hybrid",
            "expected_loss")
CHALLENGERS = tuple(p for p in POLICIES if p != "incumbent_rules")
# name, what varies, family, capacity, LTV proxy in cents
CELLS = (
    ("primary", "primary", "baseline", "base", 1500),
    ("acquisition_surge", "family", "acquisition_surge", "base", 1500),
    ("low", "allotment", "baseline", "low", 1500),
    ("high", "allotment", "baseline", "high", 1500),
    ("redesigned_layout", "layout", "baseline", "redesigned_layout", 1500),
    ("ltv_5_usd", "ltv", "baseline", "base", 500),
)
ABSENT = {("low", "hybrid")}  # no feasible operating point on any seed
UNPAIRED = {("acquisition_surge", "logistic")}  # evaluated, but never on a seed with the incumbent
INELIGIBLE = {"logistic": "lost_mean", "boosting": "service_p1"}
RECOMMENDED = {"high": "expected_loss"}
PROTOCOL = {
    "policies": {p: p for p in POLICIES},
    "reporting": {"recommendation_rule": {
        "eligibility": {"lost_legitimate_per_10000": {"mean_at_most": 100}},
        "hurdle": {"mean_improvement_usd_per_1000_orders": 100},
    }},
}


def _summary() -> dict[str, Any]:
    """A summary with the keys and tables the charts read, in the evaluate stage's shape."""
    counter = itertools.count(1)

    def number() -> float:
        return 10.0 + 3.5 * next(counter)

    def spread(value: float) -> SeedSpread:
        return SeedSpread({seed: value + (seed - 2) * 1.25 for seed in SEEDS})

    def plain(unit: str, what: str, with_seeds: bool = True) -> Metric:
        value = number()
        return Metric(value=value, unit=unit, population=what, window="test",
                      seeds=spread(value) if with_seeds else None)

    metrics: dict[str, Metric] = {}
    recommendation, flips = [], []
    for name, varies, family, capacity, ltv in CELLS:
        scope = f"{family}.{capacity}"
        improvement = "rule_net_per_1000_orders" if ltv == 1500 else \
            f"rule_net_per_1000_orders_ltv_{ltv // 100}_usd"
        for policy in POLICIES:
            absent = (name, policy) in ABSENT
            row = {"cell": name, "varies": varies, "family": family, "capacity": capacity,
                   "ltv_cents": ltv, "policy": policy, "seeds_count": 0 if absent else 3,
                   "eligible": None if absent or policy == "incumbent_rules"
                   else policy not in INELIGIBLE,
                   "fails": INELIGIBLE.get(policy, "") if not absent else "",
                   "recommended": RECOMMENDED.get(name) == policy,
                   "positive_seeds_count": None if absent else 2,
                   "paired_seeds_count": None if absent else 3}
            recommendation.append(row)
            if absent:
                continue
            metrics[f"evaluate.rule_lost_legitimate_per_10k.{scope}.{policy}"] = plain(
                "bps", "lost legitimate customers per 10,000")
            metrics[f"evaluate.rule_held_legitimate_per_10k.{scope}.{policy}"] = plain(
                "bps", "held legitimate orders per 10,000")
            metrics[f"evaluate.loss_of_gmv.{scope}.{policy}"] = plain(
                "bps", "loss per 10,000 of GMV", with_seeds=False)
            if policy == "incumbent_rules":
                continue
            key = f"evaluate.{improvement}.vs_incumbent_rules.{scope}.{policy}"
            if (name, policy) in UNPAIRED:
                metrics[key] = Metric.not_evaluated(
                    unit="cents", population="paired improvement", window="test",
                    reason="no seed on which both have a feasible operating point")
            else:
                metrics[key] = plain("cents", "paired improvement per 1,000 orders")
        flips.append({"cell": name, "varies": varies, "family": family, "capacity": capacity,
                      "ltv_cents": ltv,
                      "outcome": "recommend" if name in RECOMMENDED else "incumbent_stays",
                      "recommended": RECOMMENDED.get(name), "best_challenger": "tree_depth3",
                      "incumbent_misses": "service_p1, service_p2" if name == "low" else "",
                      "holds": None if varies == "primary" else name not in RECOMMENDED,
                      "reason": "primary" if varies == "primary" else "holds", "note": None})
    stage = StageResult(stage="evaluate", versions={"world": "w1"}, inputs={}, metrics=metrics,
                        tables={"evaluate.recommendation": recommendation,
                                "evaluate.flips": flips})
    return assemble_summary([stage])


@pytest.fixture
def summary() -> dict[str, Any]:
    return _summary()


# ---------------------------------------------------------------- reading the JSON back
def _references(node: Any, field: str | None = None) -> list[tuple[str | None, dict]]:
    """Every value record in a chart's JSON, with the field holding it: a summary key
    (with a seed for per-seed values), a result-table cell or a protocol setting."""
    found = []
    if isinstance(node, dict):
        if "value" in node and ("key" in node or "setting" in node or "table" in node):
            found.append((field, node))
        for name, item in node.items():
            found.extend(_references(item, name))
    elif isinstance(node, list):
        for item in node:
            found.extend(_references(item, field))
    return found


def _looked_up(reference: dict[str, Any], summary: dict[str, Any], protocol: dict[str, Any]
               ) -> Any:
    """The value the reference names, read from the summary and protocol as plain data."""
    if "key" in reference:
        item = summary["metrics"][reference["key"]]
        if "seed" in reference:
            return item["seeds"]["per_seed"][str(reference["seed"])]
        return item["value"]
    if "setting" in reference:
        node = protocol
        for part in reference["setting"].split("."):
            node = node[part]
        return node
    rows = [row for row in summary["tables"][reference["table"]]
            if all(row[column] == value for column, value in reference["row"].items())]
    assert len(rows) == 1, reference
    return rows[0][reference["column"]]


def _plotted(monkeypatch) -> list[tuple[float, float]]:
    """Spy on ``Axes.plot``: every (x, y) pair drawn, markers, lines and dots alike."""
    pairs: list[tuple[float, float]] = []
    original = Axes.plot

    def plot(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], (list, tuple)) \
                and isinstance(args[1], (list, tuple)):
            pairs.extend(zip(map(float, args[0]), map(float, args[1]), strict=True))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    return pairs


def _row_of(pairs: list[tuple[float, float]], xs: list[float]) -> float:
    """The single y at which every x in ``xs`` was drawn."""
    rows = {y for x, y in pairs if x == xs[0]}
    rows = {y for y in rows if all((x, y) in pairs for x in xs)}
    assert len(rows) == 1, (xs, rows)
    return rows.pop()


# ---------------------------------------------------------------- reproducibility
def test_a_rebuild_writes_every_figure_byte_for_byte(tmp_path: Path, summary: dict) -> None:
    first = render_all(summary, PROTOCOL, tmp_path / "a")
    second = render_all(summary, PROTOCOL, tmp_path / "b")
    assert [p.name for p in first] == [p.name for p in second]
    assert {p.name for p in first} == {f"{name}.{kind}" for name in CHARTS
                                       for kind in ("svg", "json")}
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name
    for svg in (p for p in first if p.suffix == ".svg"):
        text = svg.read_bytes()
        assert text.startswith(b"<?xml") and b"<dc:date>" not in text
        assert b"#ffffff" in text  # an opaque surface, legible in a dark theme


def test_a_changed_result_changes_the_figure_and_its_values(tmp_path: Path,
                                                             summary: dict) -> None:
    before = {p.name: p.read_bytes() for p in render_all(summary, PROTOCOL, tmp_path / "a")}
    changed = copy.deepcopy(summary)
    key = "evaluate.loss_of_gmv.baseline.base.tree_depth3"
    changed["metrics"][key]["value"] += 1.0
    after = {p.name: p.read_bytes() for p in render_all(changed, PROTOCOL, tmp_path / "b")}
    assert before["frontier.svg"] != after["frontier.svg"]
    assert before["frontier.json"] != after["frontier.json"]
    assert before["seed_spread.svg"] == after["seed_spread.svg"]  # the key is not drawn there


# ---------------------------------------------------------------- values
def test_every_plotted_value_is_the_summary_key_it_names(tmp_path: Path, summary: dict) -> None:
    """Checked against the summary as plain data, not through the chart code."""
    render_all(summary, PROTOCOL, tmp_path)
    for name in CHARTS:
        data = json.loads((tmp_path / f"{name}.json").read_text())
        references = [r for _, r in _references(data)]
        assert len([r for r in references if "key" in r]) >= 6, name
        assert any("setting" in r for r in references), name
        assert any("table" in r for r in references), name
        for reference in references:
            assert reference["value"] == _looked_up(reference, summary, PROTOCOL), reference


def test_the_frontier_draws_each_policy_at_its_lost_and_loss(summary: dict, monkeypatch) -> None:
    pairs = _plotted(monkeypatch)
    data = charts.collect_frontier(summary, PROTOCOL)
    charts.draw_frontier(data)
    shown = [p for p in data["policies"] if p["evaluated"]]
    assert len(shown) == len(POLICIES)
    for p in shown:
        assert (p["lost"]["value"], p["loss"]["value"]) in pairs, p["policy"]
    assert all(pair in pairs for pair in {(p["lost"]["value"], p["loss"]["value"])
                                          for p in shown})


def test_held_customers_are_written_into_the_frontier_labels(summary: dict,
                                                             monkeypatch) -> None:
    texts: list[str] = []
    original = Axes.annotate

    def annotate(self, text, *args, **kwargs):
        texts.append(text)
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", annotate)
    data = charts.collect_frontier(summary, PROTOCOL)
    charts.draw_frontier(data)
    for policy in data["policies"]:
        held = policy["held"]["value"]
        assert any(t.startswith(policy["label"]) and f"{held:,.0f} held" in t for t in texts)


def test_the_seed_spread_draws_each_policy_on_one_row(summary: dict, monkeypatch) -> None:
    pairs = _plotted(monkeypatch)
    data = charts.collect_seed_spread(summary, PROTOCOL)
    charts.draw_seed_spread(data)
    rows = {}
    for item in data["policies"]:
        values = [s["value"] for s in item["seeds"]] + [item["mean"]["value"]]
        rows[item["policy"]] = _row_of(pairs, values)
    assert len(set(rows.values())) == len(CHALLENGERS)
    by_mean = sorted(data["policies"], key=lambda p: p["mean"]["value"], reverse=True)
    assert [rows[p["policy"]] for p in by_mean] == sorted(rows.values(), reverse=True)


def test_the_operating_cells_chart_draws_each_cell_on_one_row(summary: dict,
                                                               monkeypatch) -> None:
    pairs = _plotted(monkeypatch)
    data = charts.collect_operating_cells(summary, PROTOCOL)
    figure = charts.draw_operating_cells(data)
    rows = {}
    for cell in data["cells"]:
        values = [p["value"] for p in cell["policies"] if p.get("value") is not None]
        rows[cell["name"]] = _row_of(pairs, values)
    assert len(set(rows.values())) == len(CELLS)
    assert rows["primary"] == max(rows.values())  # the first row is the primary cell
    axes = figure.axes[0]
    ticks = dict(zip(axes.get_yticks(), (t.get_text() for t in axes.get_yticklabels()),
                     strict=True))
    for cell in data["cells"]:  # each row is named by its own cell
        assert ticks[rows[cell["name"]]] == cell["label"], cell["name"]


def test_a_policy_the_rule_did_not_evaluate_is_left_out_and_recorded(summary: dict,
                                                                     monkeypatch) -> None:
    pairs = _plotted(monkeypatch)
    data = charts.collect_operating_cells(summary, PROTOCOL)
    low = next(c for c in data["cells"] if c["name"] == "low")
    hybrid = next(p for p in low["policies"] if p["policy"] == "hybrid")
    assert hybrid["evaluated"] is False and hybrid["seeds_count"]["value"] == 0
    assert "key" not in hybrid and "value" not in hybrid
    cell = next(c for c in data["cells"] if c["name"] == "acquisition_surge")
    logistic = next(p for p in cell["policies"] if p["policy"] == "logistic")
    assert logistic["value"] is None and "feasible operating point" in logistic["note"]
    charts.draw_operating_cells(data)
    others = [p["value"] for p in low["policies"] if p.get("value") is not None]
    assert others and len({y for x, y in pairs if x in others}) == 1  # the row is still drawn


def _outcome_row(seed: int, policy: str, net: int, *, where: dict, declined: int = 0) -> dict:
    """One outcome row as the replay writes it, at a $15 LTV proxy."""
    return {"seed": seed, "family": "baseline", "policy": policy, **where, "orders": 10_000,
            "net_cents": net, "friction_cost_cents": 1_500 * declined,
            "legitimate_orders": 9_000, "legitimate_held": 0, "legitimate_declined": declined,
            "legitimate_cancelled": 0, "available_minutes": 600, "review_band": 1,
            "review_minutes_used": 300, **{f"reviews_p{i}": 60 for i in range(4)},
            **{f"sla_met_p{i}": 60 for i in range(4)}}


def test_a_cell_the_rule_could_not_assess_is_left_out_and_recorded(tmp_path: Path) -> None:
    """The rule publishes no improvement key in a cell where the incumbent has no feasible
    point; the charts read the flip table's outcome instead of failing on the key."""
    rule = rec.Rule(
        lost=rec.Cap(Fraction(100), Fraction(200)), held=rec.Cap(Fraction(300), Fraction(600)),
        service_share=Fraction(9, 10), service_min_entries=50, hurdle_cents=Fraction(10_000),
        positive_seeds={10: 9, 9: 8, 8: 8}, simpler=POLICIES,
        analyst_cents_per_hour=Fraction(3_500), ltv_cents=1_500,
        ltv_sensitivity_cents=(500, 4_500))
    where = {"capacity_level": "base", "layout": "current", "history": "policy",
             "reviewer": "evidence", "verification": "verification"}
    cells = [rec.OperatingCell("primary", "primary", "baseline", "base", where),
             rec.OperatingCell("low", "allotment", "baseline", "low",
                               where | {"capacity_level": "low"}),
             rec.OperatingCell("high", "allotment", "baseline", "high",
                               where | {"capacity_level": "high"}),
             rec.OperatingCell("redesigned_layout", "layout", "baseline", "redesigned_layout",
                               where | {"layout": "evening"})]
    rows = []
    for cell in cells:
        for seed in SEEDS:
            if cell.name != "low":  # at the low allotment the incumbent has no feasible point
                rows.append(_outcome_row(seed, "incumbent_rules", 1_000_000, where=cell.where))
            rows.append(_outcome_row(seed, "tree_depth3", 1_300_000, where=cell.where))
            rows.append(_outcome_row(seed, "hybrid", 1_100_000, where=cell.where, declined=9))
    results = rec.apply_rule(rows, cells, rule, POLICIES)
    assert [r.outcome for r in results] == [rec.RECOMMEND, rec.NOT_ASSESSED, rec.RECOMMEND,
                                            rec.RECOMMEND]
    metrics, tables = rec.outputs(results, rule)
    for policy in ("incumbent_rules", "tree_depth3", "hybrid"):
        metrics[f"evaluate.loss_of_gmv.baseline.base.{policy}"] = Metric(
            value=50.0, unit="bps", population="loss", window="test")
    summary = assemble_summary([StageResult(stage="evaluate", versions={"world": "w1"},
                                            inputs={}, metrics=metrics, tables=tables)])
    assert not any(".low." in key for key in metrics if "rule_net" in key)
    render_all(summary, PROTOCOL, tmp_path)
    operating = json.loads((tmp_path / "operating_cells.json").read_text())
    assert [c["outcome"]["assessed"] for c in operating["cells"]] == [True, False, True, True]
    low = next(cell for cell in operating["cells"] if cell["name"] == "low")
    assert low["outcome"]["value"] == rec.NOT_ASSESSED
    for point in low["policies"]:
        assert "key" not in point and "value" not in point
    frontier = json.loads((tmp_path / "frontier.json").read_text())
    assert [p["policy"] for p in frontier["policies"] if p["evaluated"]] == [
        "incumbent_rules", "tree_depth3", "hybrid"]
    for name in CHARTS:
        data = json.loads((tmp_path / f"{name}.json").read_text())
        for _, reference in _references(data):
            assert reference["value"] == _looked_up(reference, summary, PROTOCOL), reference


def test_a_missing_result_key_fails_naming_it(tmp_path: Path, summary: dict, capsys) -> None:
    key = "evaluate.loss_of_gmv.baseline.base.tree_depth3"
    del summary["metrics"][key]
    with pytest.raises(KeyError, match=key.replace(".", r"\.")):
        render_all(summary, PROTOCOL, tmp_path)
    path = tmp_path / "summary.json"
    path.write_text(canonical_json(summary))
    assert main(["--summary", str(path), "--out", str(tmp_path / "out")]) == 1
    assert key in capsys.readouterr().err


def test_crowded_labels_never_reach_the_subtitle(summary: dict) -> None:
    """Seven policies on one point: labels are pushed apart within the plot and the
    small gap above it, never into the subtitle, and never below the plot."""
    crowded = copy.deepcopy(summary)
    for policy in POLICIES:
        for measure in ("rule_lost_legitimate_per_10k", "loss_of_gmv"):
            crowded["metrics"][f"evaluate.{measure}.baseline.base.{policy}"]["value"] = 40.0
    figure = charts.draw_frontier(charts.collect_frontier(crowded, PROTOCOL))
    figure.canvas.draw()
    axes = figure.axes[0]
    subtitle = min(text.get_window_extent().y0 for text in figure.texts)
    bottom = axes.get_window_extent().y0
    labels = [text for text in axes.texts if text.get_text() not in ("", " ")]
    assert len(labels) >= len(POLICIES)
    for text in labels:
        extent = text.get_window_extent()
        assert extent.y1 <= subtitle and extent.y0 >= bottom - 1, text.get_text()


# ---------------------------------------------------------------- the command
def test_the_repository_figures_are_written_only_from_the_committed_results(
    tmp_path: Path, summary: dict, capsys
) -> None:
    other = tmp_path / "summary.json"
    other.write_text(canonical_json(summary))
    before = sorted(FIGURES.glob("*")) if FIGURES.exists() else []
    assert main(["--summary", str(other)]) == 1
    assert main(["--summary", str(other), "--out", str(FIGURES)]) == 1
    assert "results/summary.json" in capsys.readouterr().err
    assert (sorted(FIGURES.glob("*")) if FIGURES.exists() else []) == before
    assert main(["--summary", str(other), "--out", str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        f"{name}.{kind}" for name in CHARTS for kind in ("svg", "json"))
    assert main(["--summary", str(tmp_path / "missing.json"), "--out", str(tmp_path)]) == 1


def test_check_reports_a_figure_that_differs_from_the_results(tmp_path: Path, summary: dict,
                                                              capsys) -> None:
    path = tmp_path / "summary.json"
    path.write_text(canonical_json(summary))
    out = tmp_path / "figures"
    assert main(["--summary", str(path), "--out", str(out)]) == 0
    assert main(["--summary", str(path), "--out", str(out), "--check"]) == 0
    (out / "frontier.svg").write_bytes(b"<svg/>")
    (out / "operating_cells.json").unlink()
    assert main(["--summary", str(path), "--out", str(out), "--check"]) == 1
    err = capsys.readouterr().err
    assert "frontier.svg: differs" in err and "operating_cells.json: not rendered" in err


def test_committed_figures_match_the_committed_results(tmp_path: Path) -> None:
    if not SUMMARY.exists():
        pytest.skip("no final results committed yet")
    protocol = yaml.safe_load((REPO / "experiments" / "protocol.yaml").read_text())
    summary = read_summary(SUMMARY)
    assert out_of_sync(summary, protocol, FIGURES, tmp_path) == []
    for name in CHARTS:
        data = json.loads((FIGURES / f"{name}.json").read_text())
        for _, reference in _references(data):
            assert reference["value"] == _looked_up(reference, summary, protocol), reference
