"""Charts for the operating review and the README, drawn from the results summary.

    python -m report.charts [--summary results/summary.json] [--out reports/figures] [--check]

Each chart reads the summary the pipeline wrote (``results/summary.json``) and the
frozen protocol, and writes two files: ``<name>.svg``, saved through
:func:`report.figures.save_svg` so a rebuild writes the same bytes, and
``<name>.json``, exactly the values the figure shows with the summary key (and
seed), result-table cell or protocol setting each one came from. The repository's
own figures are written only from the committed summary; a summary from another
run renders into ``--out`` elsewhere. ``--check`` writes nothing and fails when a
committed figure differs from a fresh render.

A metric the summary lacks fails the run, naming the key. A policy the rule did not
evaluate in a cell (no feasible operating point on any seed), and a cell the rule could
not assess (no incumbent point), are left out of the figure and recorded in the JSON
with the reason.

The charts, all in money the recommendation rule uses (rule net contribution:
ledger net after the friction cost, less the analyst allotment):

``frontier``
    In the primary cell, each policy's fraud and abuse loss in basis points of GMV
    against the legitimate customers it loses per 10,000 legitimate orders, with
    the rule's customer cap, and hollow markers where a policy fails one of the
    rule's eligibility criteria.
``staffing``
    Each challenger's paired improvement over the incumbent, per 1,000 orders (the
    orders a policy decides), at each allotment level and on the evening layout, with
    the hurdle.
``seed_spread``
    The per-seed paired differences behind the primary cell's means.
``operating_cells``
    The flip table as a picture: every challenger's improvement in every operating
    cell, the best one named, and the cell's outcome.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from core.recommendation import INCUMBENT, NOT_ASSESSED, ltv_name  # noqa: E402
from core.results import canonical_json, metric, read_summary, table  # noqa: E402
from report.figures import save_svg  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "results" / "summary.json"
PROTOCOL = REPO / "experiments" / "protocol.yaml"
FIGURES = REPO / "reports" / "figures"

IMPROVEMENT = "rule_net_per_1000_orders"
RECOMMENDATION = "evaluate.recommendation"
FLIPS = "evaluate.flips"
LOST_CAP = "reporting.recommendation_rule.eligibility.lost_legitimate_per_10000.mean_at_most"
HURDLE = "reporting.recommendation_rule.hurdle.mean_improvement_usd_per_1000_orders"

LABELS = {
    "approve_all": "approve all",
    "incumbent_rules": "incumbent rules",
    "tree_depth3": "tree (depth 3)",
    "logistic": "logistic",
    "boosting": "boosting",
    "hybrid": "hybrid",
    "expected_loss": "expected loss",
}
CELL_LABELS = {
    "primary": "baseline, base allotment",
    "acquisition_surge": "acquisition surge",
    "fraud_mix_shift": "fraud-mix shift",
    "lag_half": "fulfilment lag halved",
    "lag_double": "fulfilment lag doubled",
    "low": "low allotment",
    "high": "high allotment",
    "redesigned_layout": "evening shift layout",
    "base_weak_verification": "weak verification",
}
CRITERIA = {"lost_mean": "over the lost-customer cap",
            "lost_any_seed": "over the lost-customer cap on a seed",
            "held_mean": "over the held-customer cap",
            "held_any_seed": "over the held-customer cap on a seed"}

# Colours: an opaque white surface (GitHub's dark mode leaves the SVG legible), ink
# for text, one accent for the policies in charts where labels carry identity, and
# a fixed hue and marker per policy where lines must be told apart.
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#ffffff", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
ACCENT, REFERENCE = "#2a78d6", "#898781"
HUES = {"tree_depth3": "#2a78d6", "logistic": "#eb6834", "boosting": "#1baf7a",
        "hybrid": "#eda100", "expected_loss": "#e87ba4", "approve_all": REFERENCE}
MARKERS = {"tree_depth3": "o", "logistic": "s", "boosting": "^", "hybrid": "D",
           "expected_loss": "v", "approve_all": "o", "incumbent_rules": "o"}
FONT = 10


# ---------------------------------------------------------------- reading the results
def load_protocol(path: Path = PROTOCOL) -> Mapping[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _label(policy: str) -> str:
    return LABELS.get(policy, policy.replace("_", " "))


def _misses(fails: str) -> str:
    """The rule's criteria a policy fails, in words: ``service_p1, service_p2`` reads
    "misses P1, P2 service"; the customer caps by name."""
    names = [m for m in fails.split(", ") if m]
    service = [m.replace("service_p", "P") for m in names if m.startswith("service_")]
    other = [CRITERIA.get(m, m) for m in names if not m.startswith("service_")]
    parts = ([f"misses {', '.join(service)} service"] if service else []) + other
    return "; ".join(parts)


def _policies(protocol: Mapping[str, Any]) -> list[str]:
    """The protocol's policies in its order; the incumbent is among them."""
    policies = list(protocol["policies"])
    if INCUMBENT not in policies:
        raise KeyError(f"the protocol's policies do not include {INCUMBENT!r}")
    return policies


def _value(summary: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A metric's value with its key; ``value`` is ``None`` when it was not evaluated."""
    item = metric(summary, key)
    return {"key": key, "value": item.value, "note": item.note}


def _seeds(summary: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    """A metric's per-seed values, each with the key and its seed."""
    item = metric(summary, key)
    if item.seeds is None:
        raise KeyError(f"result key {key!r} carries no per-seed values")
    return [{"key": key, "seed": seed, "value": value}
            for seed, value in item.seeds.per_seed.items()]


def _setting(protocol: Mapping[str, Any], dotted: str) -> dict[str, Any]:
    node: Any = protocol
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"the protocol has no {dotted!r}")
        node = node[part]
    return {"setting": dotted, "value": node}


def _rows(summary: Mapping[str, Any], name: str, where: Mapping[str, Any]
          ) -> list[dict[str, Any]]:
    return [row for row in table(summary, name)
            if all(row.get(column) == value for column, value in where.items())]


def _cell_of(summary: Mapping[str, Any], name: str, where: Mapping[str, Any], column: str
             ) -> dict[str, Any]:
    """One table cell: the value in ``column`` of the single row matching ``where``."""
    rows = _rows(summary, name, where)
    if len(rows) != 1:
        raise KeyError(f"result table {name!r} has {len(rows)} rows matching {dict(where)}, "
                       "not one")
    if column not in rows[0]:
        raise KeyError(f"result table {name!r} has no column {column!r}")
    return {"table": name, "row": dict(where), "column": column, "value": rows[0][column]}


def _primary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The primary operating cell (the flip table's first row names it)."""
    [row] = _rows(summary, FLIPS, {"varies": "primary"}) or [None]
    if row is None:
        raise KeyError(f"result table {FLIPS!r} has no primary cell")
    return {"name": row["cell"], "family": row["family"], "capacity": row["capacity"],
            "ltv_cents": row["ltv_cents"]}


def _standing(summary: Mapping[str, Any], cell: str, policy: str) -> dict[str, Any]:
    """Whether the rule evaluated ``policy`` in ``cell``, and how it fared there."""
    where = {"cell": cell, "policy": policy}
    seeds = _cell_of(summary, RECOMMENDATION, where, "seeds_count")
    return {
        "evaluated": seeds["value"] > 0,
        "seeds_count": seeds,
        "eligible": _cell_of(summary, RECOMMENDATION, where, "eligible"),
        "fails": _cell_of(summary, RECOMMENDATION, where, "fails"),
        "recommended": _cell_of(summary, RECOMMENDATION, where, "recommended"),
    }


def _outcome(summary: Mapping[str, Any], cell: str) -> dict[str, Any]:
    """The rule's outcome in a cell: ``not_assessed`` (no incumbent point on any seed)
    publishes no improvement keys, so the charts read none there."""
    outcome = _cell_of(summary, FLIPS, {"cell": cell}, "outcome")
    outcome["assessed"] = outcome["value"] != NOT_ASSESSED
    return outcome


def _improvement_key(primary: Mapping[str, Any], family: str, capacity: str,
                     ltv_cents: int, policy: str) -> str:
    name = IMPROVEMENT if ltv_cents == primary["ltv_cents"] else \
        f"{IMPROVEMENT}_{ltv_name(ltv_cents)}"
    return f"evaluate.{name}.vs_{INCUMBENT}.{family}.{capacity}.{policy}"


# ---------------------------------------------------------------- the data each chart shows
def collect_frontier(summary: Mapping[str, Any], protocol: Mapping[str, Any]
                     ) -> dict[str, Any]:
    cell = _primary(summary)
    scope = f"{cell['family']}.{cell['capacity']}"
    policies = []
    for policy in _policies(protocol):
        item: dict[str, Any] = {"policy": policy, "label": _label(policy),
                                **_standing(summary, cell["name"], policy)}
        if item["evaluated"]:
            for name, measure in (("lost", "rule_lost_legitimate_per_10k"),
                                  ("held", "rule_held_legitimate_per_10k"),
                                  ("loss", "loss_of_gmv")):
                item[name] = _value(summary, f"evaluate.{measure}.{scope}.{policy}")
        policies.append(item)
    return {"chart": "frontier", "cell": cell, "policies": policies,
            "lost_cap": _setting(protocol, LOST_CAP)}


def collect_staffing(summary: Mapping[str, Any], protocol: Mapping[str, Any]
                     ) -> dict[str, Any]:
    primary = _primary(summary)
    levels = protocol["capacity"]["levels"]
    cells = []
    for level in levels:
        where = {"family": primary["family"], "capacity": level,
                 "ltv_cents": primary["ltv_cents"]}
        rows = [row for row in _rows(summary, FLIPS, where)
                if row["varies"] in ("primary", "allotment")]
        if len(rows) != 1:
            raise KeyError(f"result table {FLIPS!r} has {len(rows)} allotment cells for "
                           f"{level!r}, not one")
        cells.append({"name": rows[0]["cell"], "capacity": level,
                      "label": f"{level} allotment",
                      "minutes_per_shift": _setting(protocol, f"capacity.levels.{level}"),
                      "outcome": _outcome(summary, rows[0]["cell"]),
                      "note": _cell_of(summary, FLIPS, {"cell": rows[0]["cell"]}, "note"),
                      "incumbent_misses": _cell_of(summary, FLIPS, {"cell": rows[0]["cell"]},
                                                   "incumbent_misses")})
    [layout] = _rows(summary, FLIPS, {"family": primary["family"], "varies": "layout",
                                      "ltv_cents": primary["ltv_cents"]}) or [None]
    if layout is None:
        raise KeyError(f"result table {FLIPS!r} has no shift-layout cell")
    cells.append({"name": layout["cell"], "capacity": layout["capacity"],
                  "label": "evening layout", "minutes_per_shift": None,
                  "outcome": _outcome(summary, layout["cell"]),
                  "note": _cell_of(summary, FLIPS, {"cell": layout["cell"]}, "note"),
                  "incumbent_misses": _cell_of(summary, FLIPS, {"cell": layout["cell"]},
                                               "incumbent_misses")})
    policies = []
    for policy in _policies(protocol):
        if policy == INCUMBENT:
            continue
        points = []
        for cell in cells:
            point: dict[str, Any] = {"cell": cell["name"],
                                     **_standing(summary, cell["name"], policy)}
            if point["evaluated"] and cell["outcome"]["assessed"]:
                point.update(_value(summary, _improvement_key(
                    primary, primary["family"], cell["capacity"], primary["ltv_cents"], policy)))
            points.append(point)
        policies.append({"policy": policy, "label": _label(policy), "points": points})
    return {"chart": "staffing", "primary": primary, "cells": cells, "policies": policies,
            "hurdle_usd": _setting(protocol, HURDLE)}


def collect_seed_spread(summary: Mapping[str, Any], protocol: Mapping[str, Any]
                        ) -> dict[str, Any]:
    cell = _primary(summary)
    cell["outcome"] = _outcome(summary, cell["name"])
    cell["note"] = _cell_of(summary, FLIPS, {"cell": cell["name"]}, "note")
    policies = []
    for policy in _policies(protocol):
        if policy == INCUMBENT:
            continue
        item: dict[str, Any] = {"policy": policy, "label": _label(policy),
                                **_standing(summary, cell["name"], policy)}
        if item["evaluated"] and cell["outcome"]["assessed"]:
            key = _improvement_key(cell, cell["family"], cell["capacity"], cell["ltv_cents"],
                                   policy)
            item["mean"] = _value(summary, key)
            item["seeds"] = _seeds(summary, key) if item["mean"]["value"] is not None else []
            where = {"cell": cell["name"], "policy": policy}
            item["positive_seeds"] = _cell_of(summary, RECOMMENDATION, where,
                                              "positive_seeds_count")
            item["paired_seeds"] = _cell_of(summary, RECOMMENDATION, where,
                                            "paired_seeds_count")
        policies.append(item)
    return {"chart": "seed_spread", "cell": cell, "policies": policies,
            "hurdle_usd": _setting(protocol, HURDLE)}


def collect_operating_cells(summary: Mapping[str, Any], protocol: Mapping[str, Any]
                            ) -> dict[str, Any]:
    primary = _primary(summary)
    challengers = [policy for policy in _policies(protocol) if policy != INCUMBENT]
    cells = []
    for row in table(summary, FLIPS):
        where = {"cell": row["cell"]}
        cell: dict[str, Any] = {
            "name": row["cell"], "label": CELL_LABELS.get(row["cell"], row["cell"]),
            "family": row["family"], "capacity": row["capacity"],
            "ltv_cents": row["ltv_cents"], "varies": row["varies"],
            "outcome": _outcome(summary, row["cell"]),
            "recommended": _cell_of(summary, FLIPS, where, "recommended"),
            "best_challenger": _cell_of(summary, FLIPS, where, "best_challenger"),
            "holds": _cell_of(summary, FLIPS, where, "holds"),
            "reason": _cell_of(summary, FLIPS, where, "reason"),
            "note": _cell_of(summary, FLIPS, where, "note"),
            "policies": [],
        }
        if row["cell"].startswith("ltv_"):
            cell["label"] = f"customer value ${row['ltv_cents'] / 100:g}"
        for policy in challengers:
            point: dict[str, Any] = {"policy": policy, "label": _label(policy),
                                     **_standing(summary, row["cell"], policy)}
            if point["evaluated"] and cell["outcome"]["assessed"]:
                point.update(_value(summary, _improvement_key(
                    primary, row["family"], row["capacity"], row["ltv_cents"], policy)))
            cell["policies"].append(point)
        cells.append(cell)
    return {"chart": "operating_cells", "primary": primary, "cells": cells,
            "hurdle_usd": _setting(protocol, HURDLE)}


# ---------------------------------------------------------------- drawing
def _dollars(cents: float, _position: Any = None) -> str:
    sign = "−" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:,.0f}"


def _figure(width: float, height: float, title: str, subtitle: str, *, left: float,
            right: float, bottom: float):
    """A figure with its title and wrapped subtitle; the axes start below them."""
    figure, axes = plt.subplots(figsize=(width, height), dpi=100)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    figure.text(0.01, 0.975, title, ha="left", va="top", fontsize=12.5, color=INK,
                fontweight="bold")
    wrapped = textwrap.fill(subtitle, width=int(12.6 * width))
    subtitle_top = 0.975 - 0.3 / height
    figure.text(0.01, subtitle_top, wrapped, ha="left", va="top", fontsize=FONT, color=INK2,
                linespacing=1.4)
    lines = wrapped.count("\n") + 1
    top = subtitle_top - (lines * FONT * 1.4 + 22) / (height * 72)
    figure.subplots_adjust(top=top, left=left, right=right, bottom=bottom)
    return figure, axes


def _style(axes, *, grid: str = "y") -> None:
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=MUTED, labelsize=FONT - 1, length=3, width=0.8)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_color(INK2)
    for label in (axes.xaxis.label, axes.yaxis.label):
        label.set_color(INK2)
        label.set_size(FONT)
    if grid:
        axes.grid(True, axis=grid, color=GRID, linewidth=0.8)
        axes.set_axisbelow(True)


def _marker(axes, x: float, y: float, policy: str, eligible: bool | None, *,
            colour: str | None = None, size: float = 8.0, zorder: int = 4) -> None:
    """A policy's marker: filled when the rule finds it eligible, hollow when it fails
    a criterion, with a surface ring either way."""
    colour = colour or HUES.get(policy, ACCENT)
    hollow = eligible is False
    axes.plot([x], [y], marker=MARKERS.get(policy, "o"), linestyle="none", markersize=size,
              markerfacecolor=SURFACE if hollow else colour,
              markeredgecolor=colour if hollow else SURFACE,
              markeredgewidth=1.6 if hollow else 1.2, zorder=zorder)


def _legend(axes, policies: list[str], *, lines: bool, y: float) -> None:
    handles = [Line2D([], [], marker=MARKERS.get(p, "o"), color=HUES.get(p, ACCENT),
                      linestyle="-" if lines else "none", linewidth=2, markersize=7,
                      markeredgecolor=SURFACE, markeredgewidth=1.0) for p in policies]
    legend = axes.legend(handles, [_label(p) for p in policies], loc="upper center",
                         bbox_to_anchor=(0.5, y), ncol=len(policies), frameon=False,
                         fontsize=FONT - 1.5, handlelength=1.8, columnspacing=1.4,
                         handletextpad=0.5)
    for text in legend.get_texts():
        text.set_color(INK2)


def _place_labels(axes, items: list[tuple[float, float, str]], *,
                  obstacles: list[tuple[float, float]] = (), gap: float = 12.5,
                  size: float = FONT - 0.5, colour: str = INK, flip: bool = True) -> None:
    """Label each point beside it: to its right; to its left when ``flip`` allows it and
    the right is taken; else pushed up until it overlaps neither another label nor a
    marker (``obstacles``, data coordinates), a thin leader then joining it to its point.
    A second line of text (after a newline) is written smaller and muted."""
    figure = axes.figure
    scale = 72 / figure.dpi  # display pixels -> points
    position = axes.get_position()
    x_min = position.x0 * figure.get_figwidth() * 72
    x_max = figure.get_figwidth() * 72 - 3
    # labels may sit a little above the axes (the subtitle is further up), not below
    y_min, y_max = (position.y0 * figure.get_figheight() * 72 - 2,
                    position.y1 * figure.get_figheight() * 72 + 14)
    boxes = []
    for x, y in obstacles:
        px, py = axes.transData.transform((x, y)) * scale
        boxes.append((px - 6, px + 6, py - 6, py + 6))

    def free(x0: float, x1: float, y0: float, y1: float) -> bool:
        return not any(x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1
                       for bx0, bx1, by0, by1 in boxes)

    for x, y, text in sorted(items, key=lambda item: axes.transData.transform(item[:2])[1]):
        px, py = axes.transData.transform((x, y)) * scale
        lines = text.split("\n")
        width = 0.56 * size * max(len(line) for line in lines)
        half = gap * len(lines) / 2
        sides = [1] if px + 9 + width <= x_max else []
        if flip and px - 9 - width >= x_min:
            sides.append(-1)
        sides = sides or [1]
        starts = {1: px + 9, -1: px - 9 - width}
        side, ly = sides[0], py
        for candidate in sides:
            if free(starts[candidate], starts[candidate] + width, py - half, py + half):
                side = candidate
                break
        else:  # pushed up or down inside the axes; when nothing is free, left where it is
            steps = (gap / 2 * (k // 2 + 1) * (1 if k % 2 == 0 else -1)
                     for k in range(2 * int((y_max - y_min) / gap) + 2))
            for step in steps:
                candidate = py + step
                if not y_min + half <= candidate <= y_max - half:
                    continue
                if free(starts[side], starts[side] + width, candidate - half, candidate + half):
                    ly = candidate
                    break
        boxes.append((starts[side], starts[side] + width, ly - half, ly + half))
        displaced = abs(ly - py) > 2
        offsets = [ly - py] if len(lines) == 1 else [ly - py + half / 2, ly - py - half / 2]
        for n, (line, dy) in enumerate(zip(lines, offsets, strict=True)):
            axes.annotate(
                line, xy=(x, y), xycoords="data", xytext=(9 * side, dy),
                textcoords="offset points", ha="left" if side == 1 else "right",
                va="center", fontsize=size if n == 0 else size - 1,
                color=colour if n == 0 else INK2, zorder=6,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.7, shrinkA=1,
                                shrinkB=5) if displaced and n == 0 else None)


def _reference_lines(axes, hurdle: float, *, vertical: bool) -> None:
    """The incumbent (zero) and the hurdle, tagged at the top of the plot."""
    line = axes.axvline if vertical else axes.axhline
    line(0, color=INK, linewidth=0.9, zorder=1)
    line(hurdle, color=AXIS, linewidth=1.0, zorder=1)
    if vertical:
        top = axes.get_ylim()[1]
        axes.text(0, top, "incumbent rules ", ha="right", va="top", fontsize=FONT - 1,
                  color=INK2)
        axes.text(hurdle, top, f" hurdle +{_dollars(hurdle)}", ha="left", va="top",
                  fontsize=FONT - 1, color=INK2)
    else:
        left = axes.get_xlim()[0]
        axes.text(left, 0, " incumbent rules", ha="left", va="top", fontsize=FONT - 1,
                  color=INK2)
        axes.text(left, hurdle, f" hurdle +{_dollars(hurdle)}", ha="left", va="bottom",
                  fontsize=FONT - 1, color=INK2)


def draw_frontier(data: Mapping[str, Any]):
    figure, axes = _figure(
        8.4, 5.2, "Fraud loss and good customers lost, by policy",
        "Baseline family, base allotment, test window: fraud and abuse loss against the "
        "legitimate customers each policy turns away (declined, blocked, or cancelled after "
        "an unanswered hold), with the legitimate orders it holds for verification, both per "
        "10,000 legitimate orders. Hollow: fails an eligibility criterion of the "
        "recommendation rule, named in the label (the customer caps, or the queue's service "
        "target at a priority).",
        left=0.08, right=0.97, bottom=0.11)
    cap = data["lost_cap"]["value"]
    shown = [p for p in data["policies"] if p["evaluated"]
             and p["lost"]["value"] is not None and p["loss"]["value"] is not None]
    xs = [p["lost"]["value"] for p in shown]
    x_max = max([cap * 1.12, *(x * 1.15 for x in xs)])
    axes.set_xlim(-x_max * 0.02, x_max)
    axes.set_ylim(0, max([1.0, *(p["loss"]["value"] * 1.12 for p in shown)]))
    if not shown:
        axes.text(0.5, 0.5, "no policy evaluated in the primary cell", ha="center",
                  va="center", transform=axes.transAxes, fontsize=FONT, color=INK2)
    axes.axvline(cap, color=AXIS, linewidth=1.0, zorder=1)
    axes.text(cap, axes.get_ylim()[1], f"rule's cap: {cap:g} lost per 10,000 (mean) ",
              ha="right", va="top", fontsize=FONT - 1, color=INK2)
    labels = []
    for p in shown:
        colour = INK if p["policy"] == INCUMBENT else \
            REFERENCE if p["policy"] == "approve_all" else ACCENT
        _marker(axes, p["lost"]["value"], p["loss"]["value"], p["policy"],
                p["eligible"]["value"], colour=colour, size=9)
        held = p["held"]["value"]
        text = p["label"] if not held else f"{p['label']}, {held:,.0f} held"
        if p["eligible"]["value"] is False:
            text += f"\n{_misses(p['fails']['value'])}"
        labels.append((p["lost"]["value"], p["loss"]["value"], text))
    _place_labels(axes, labels, obstacles=[(x, y) for x, y, _ in labels])
    axes.set_xlabel("Legitimate customers lost per 10,000 legitimate orders")
    axes.set_ylabel("Fraud and abuse loss, basis points of GMV")
    _style(axes)
    return figure


def draw_staffing(data: Mapping[str, Any]):
    figure, axes = _figure(
        8.4, 6.0, "Net contribution against today's rules, by analyst allotment",
        "Baseline family, test window: each challenger's rule net contribution (ledger net "
        "after the friction cost, less the analyst allotment) minus the incumbent's, per 1,000 "
        "orders decided, with thresholds tuned at the base allotment and held fixed. The "
        "evening layout runs the base minutes at different hours. Hollow: fails an "
        "eligibility criterion in that cell; ringed: the rule's recommendation there.",
        left=0.10, right=0.86, bottom=0.2)
    cells = data["cells"]
    positions = list(range(len(cells)))
    line_end = max(i for i, cell in enumerate(cells) if cell["minutes_per_shift"] is not None)
    hurdle = data["hurdle_usd"]["value"] * 100
    labels, ends = [], []
    for item in data["policies"]:
        colour = HUES.get(item["policy"], ACCENT)
        points = [(i, p) for i, p in enumerate(item["points"])
                  if p["evaluated"] and p.get("value") is not None]
        on_line = [(i, p["value"]) for i, p in points if i <= line_end]
        axes.plot([i for i, _ in on_line], [v for _, v in on_line], color=colour,
                  linewidth=2, solid_capstyle="round", zorder=3)
        for i, p in points:
            _marker(axes, i, p["value"], item["policy"], p["eligible"]["value"], colour=colour)
            if p["recommended"]["value"]:
                axes.plot([i], [p["value"]], marker="o", linestyle="none", markersize=17,
                          markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.0,
                          zorder=5)
        if points:
            labels.append((points[-1][0], points[-1][1]["value"], item["label"]))
            ends.append((points[-1][0], points[-1][1]["value"]))
    ticks = []
    for cell in cells:
        minutes = cell["minutes_per_shift"]
        ticks.append(f"{cell['label']}\n" + ("base minutes, evening hours" if minutes is None
                                             else f"{minutes['value']:g} min per shift"))
    axes.set_xticks(positions, ticks)
    axes.set_xlim(positions[0] - 0.45, positions[-1] + 0.45)
    if line_end < positions[-1]:
        axes.axvline(line_end + 0.5, color=AXIS, linewidth=0.8, zorder=1)
    low, high = axes.get_ylim()
    axes.set_ylim(min(low, -hurdle), max(high, hurdle * 2.2))
    _reference_lines(axes, hurdle, vertical=False)
    _place_labels(axes, labels, obstacles=ends, flip=False)
    axes.yaxis.set_major_formatter(FuncFormatter(_dollars))
    axes.set_ylabel("Net contribution vs today's rules, $ per 1,000 orders")
    _style(axes)
    width_pt = (figure.subplotpars.right - figure.subplotpars.left) * figure.get_figwidth() * 72
    columns = max(1, int(width_pt / len(cells) / (0.56 * (FONT - 2))) - 1)
    captions = []
    for cell in cells:
        misses = cell["incumbent_misses"]["value"]
        note = cell["note"]["value"] if not cell["outcome"]["assessed"] else None
        text = f"incumbent {_misses(misses)}" if misses else note or ""
        captions.append(textwrap.wrap(text, width=columns) if text else [])
    lines = max((len(caption) for caption in captions), default=0)
    caption_pt = lines * (FONT - 2) * 1.3
    figure.subplots_adjust(bottom=(32 + caption_pt + 44) / (figure.get_figheight() * 72))
    for i, caption in enumerate(captions):
        if caption:
            axes.annotate("\n".join(caption), xy=(i, 0), xycoords=("data", "axes fraction"),
                          xytext=(0, -32), textcoords="offset points", ha="center", va="top",
                          fontsize=FONT - 2, color=MUTED, linespacing=1.3)
    axes_pt = (figure.subplotpars.top - figure.subplotpars.bottom) * figure.get_figheight() * 72
    _legend(axes, [item["policy"] for item in data["policies"]], lines=True,
            y=-(32 + caption_pt + 10) / axes_pt)
    return figure


def draw_seed_spread(data: Mapping[str, Any]):
    figure, axes = _figure(
        8.4, 4.6, "Paired difference from the incumbent, seed by seed",
        "Baseline family, base allotment, test window: each dot is one seed's rule net "
        "contribution (ledger net after the friction cost, less the analyst allotment) minus "
        "the incumbent's on the same world, per 1,000 orders decided; the bar is the mean "
        "over seeds, the count at the right how many seeds come out above the incumbent.",
        left=0.16, right=0.80, bottom=0.14)
    shown = [p for p in data["policies"] if p.get("mean", {}).get("value") is not None]
    shown.sort(key=lambda p: p["mean"]["value"], reverse=True)
    hurdle = data["hurdle_usd"]["value"] * 100
    rows = list(range(len(shown)))[::-1]
    if not shown:
        axes.text(0.5, 0.5, "no challenger assessed in the primary cell", ha="center",
                  va="center", transform=axes.transAxes, fontsize=FONT, color=INK2)
    for y, p in zip(rows, shown, strict=True):
        colour = REFERENCE if p["policy"] == "approve_all" else ACCENT
        axes.plot([s["value"] for s in p["seeds"]], [y] * len(p["seeds"]), marker="o",
                  linestyle="none", markersize=7, markerfacecolor=colour,
                  markeredgecolor=SURFACE, markeredgewidth=1.0, alpha=0.75, zorder=3)
        axes.plot([p["mean"]["value"]], [y], marker="|", linestyle="none", markersize=15,
                  markeredgewidth=2.2, color=INK, zorder=4)
        positive, paired = p["positive_seeds"]["value"], p["paired_seeds"]["value"]
        axes.annotate(f"{positive} of {paired} seeds positive", xy=(1.0, y),
                      xycoords=("axes fraction", "data"), xytext=(6, 0),
                      textcoords="offset points", ha="left", va="center", fontsize=FONT - 1,
                      color=INK2)
    axes.set_yticks(rows, [p["label"] for p in shown])
    axes.set_ylim(-0.7, max(len(shown), 1) - 0.3)
    axes.xaxis.set_major_formatter(FuncFormatter(_dollars))
    low, high = axes.get_xlim()
    axes.set_xlim(min(low, -hurdle), max(high, hurdle * 2))
    _reference_lines(axes, hurdle, vertical=True)
    axes.set_xlabel("Net contribution vs today's rules, $ per 1,000 orders")
    _style(axes, grid="x")
    axes.tick_params(axis="y", length=0)
    for label in axes.get_yticklabels():
        label.set_color(INK)
    return figure


def draw_operating_cells(data: Mapping[str, Any]):
    cells = data["cells"]
    figure, axes = _figure(
        8.4, 3.0 + 0.34 * len(cells), "Where the recommendation holds: every operating cell",
        "Each challenger's mean rule net contribution (ledger net after the friction cost, "
        "less the analyst allotment) minus the incumbent's, per 1,000 orders decided, with the "
        "recommendation rule applied in every operating cell on its own (the first row is the "
        "primary cell). Hollow: fails an eligibility criterion in that cell. At the right, the "
        "cell's outcome and the best eligible challenger, or the best of all when none is "
        "eligible.",
        left=0.235, right=0.77, bottom=0.13)
    hurdle = data["hurdle_usd"]["value"] * 100
    rows = list(range(len(cells)))[::-1]
    for y, cell in zip(rows, cells, strict=True):
        for p in cell["policies"]:
            if p["evaluated"] and p.get("value") is not None:
                _marker(axes, p["value"], y, p["policy"], p["eligible"]["value"], size=7.5)
        recommended = cell["recommended"]["value"]
        best = cell["best_challenger"]["value"]
        if not cell["outcome"]["assessed"]:
            lines = ["not assessed", "(no incumbent point)"]
        elif recommended is None:
            eligible = {p["policy"]: p["eligible"]["value"] for p in cell["policies"]}
            which = "best eligible" if eligible.get(best) else "best, none eligible"
            lines = ["incumbent stays", f"{which}: {_label(best)}" if best else ""]
        else:
            lines = ["recommend", _label(recommended)]
        axes.annotate("\n".join(line for line in lines if line), xy=(1.0, y),
                      xycoords=("axes fraction", "data"), xytext=(6, 0),
                      textcoords="offset points", ha="left", va="center", fontsize=FONT - 1.5,
                      color=INK, linespacing=1.25)
    axes.set_yticks(rows, [cell["label"] for cell in cells])
    axes.set_ylim(-0.7, len(cells) - 0.3)
    axes.xaxis.set_major_formatter(FuncFormatter(_dollars))
    low, high = axes.get_xlim()
    axes.set_xlim(min(low, -hurdle), max(high, hurdle * 2))
    _reference_lines(axes, hurdle, vertical=True)
    axes.set_xlabel("Net contribution vs today's rules, $ per 1,000 orders")
    _style(axes, grid="x")
    axes.tick_params(axis="y", length=0)
    primary = [cell["label"] for cell in cells if cell["varies"] == "primary"]
    for label in axes.get_yticklabels():
        label.set_color(INK)
        if label.get_text() in primary:
            label.set_fontweight("bold")
    _legend(axes, [p["policy"] for p in cells[0]["policies"]], lines=False, y=-0.11)
    return figure


# ---------------------------------------------------------------- rendering
Collector = Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]
Drawer = Callable[[Mapping[str, Any]], Any]
CHARTS: dict[str, tuple[Collector, Drawer]] = {
    "frontier": (collect_frontier, draw_frontier),
    "staffing": (collect_staffing, draw_staffing),
    "seed_spread": (collect_seed_spread, draw_seed_spread),
    "operating_cells": (collect_operating_cells, draw_operating_cells),
}


def render_chart(name: str, summary: Mapping[str, Any], protocol: Mapping[str, Any],
                 out_dir: Path) -> tuple[Path, Path]:
    """Write ``<name>.svg`` and ``<name>.json`` into ``out_dir``; returns both paths."""
    collect, draw = CHARTS[name]
    data = collect(summary, protocol)
    figure = draw(data)
    try:
        svg = save_svg(figure, Path(out_dir) / f"{name}.svg")
    finally:
        plt.close(figure)
    values = Path(out_dir) / f"{name}.json"
    values.write_bytes(canonical_json(data).encode("utf-8"))
    return svg, values


def render_all(summary: Mapping[str, Any], protocol: Mapping[str, Any], out_dir: Path
               ) -> list[Path]:
    written: list[Path] = []
    for name in CHARTS:
        written.extend(render_chart(name, summary, protocol, out_dir))
    return written


def out_of_sync(summary: Mapping[str, Any], protocol: Mapping[str, Any], out_dir: Path,
                scratch: Path) -> list[str]:
    """Figures in ``out_dir`` that differ from a fresh render (made in ``scratch``)."""
    problems = []
    for fresh in render_all(summary, protocol, scratch):
        committed = Path(out_dir) / fresh.name
        if not committed.exists():
            problems.append(f"{committed}: not rendered (run python -m report.charts)")
        elif committed.read_bytes() != fresh.read_bytes():
            problems.append(f"{committed}: differs from the results (run python -m report.charts)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m report.charts",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--out", type=Path, default=FIGURES, help="where to write the figures")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; fail when a figure differs from a fresh render")
    args = parser.parse_args(argv)
    if args.out.resolve() == FIGURES.resolve() and args.summary.resolve() != SUMMARY.resolve():
        print("the repository's figures are written only from results/summary.json; "
              "give --out for another summary", file=sys.stderr)
        return 1
    if not args.summary.exists():
        print(f"{args.summary} does not exist; run the pipeline first", file=sys.stderr)
        return 1
    summary, protocol = read_summary(args.summary), load_protocol()
    try:
        if args.check:
            import tempfile

            with tempfile.TemporaryDirectory() as scratch:
                problems = out_of_sync(summary, protocol, args.out, Path(scratch))
            for problem in problems:
                print(problem, file=sys.stderr)
            print(f"figures: {len(CHARTS)} charts, {len(problems)} problems")
            return 1 if problems else 0
        for path in render_all(summary, protocol, args.out):
            print(f"wrote {path}")
    except KeyError as error:
        print(f"cannot draw the figures: {error.args[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
