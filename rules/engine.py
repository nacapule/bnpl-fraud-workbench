"""Rules engine: fire the rule conditions of the fraud policy (FP-2 §6.2) on as-of
context rows and score them.

The engine reads only rows of the as-of context (``core.asof``): routing and the
replay pass each order's row at its decision time; the engine computes no history
of its own. The score is the sum of the weights of the rules that fire; checkout
routing compares it with the bands in ``config/policy.yaml`` (FP-2 §4.1).

Run ``python -m rules.engine <world directory>`` to print how often each condition
fires at checkout and how many processor-approved orders reach each band.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from rules.definitions import COLUMNS, CONDITIONS, RULE_IDS, WEIGHTS


def _require(context: pd.DataFrame) -> None:
    missing = [column for column in COLUMNS if column not in context.columns]
    if missing:
        raise KeyError(f"context rows lack the rule inputs {missing}")


def fired(context: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule id (R01..R11; R06 when either part fires) and per R06
    part ("R06(a)", "R06(b)"), aligned with the rows of ``context``."""
    _require(context)
    parts = {c.id: c.fire(context).fillna(False).to_numpy(bool) for c in CONDITIONS}
    out = {rule: np.logical_or.reduce([parts[c.id] for c in CONDITIONS if c.rule == rule])
           for rule in RULE_IDS}
    out.update({c.id: parts[c.id] for c in CONDITIONS if c.id != c.rule})
    return pd.DataFrame(out, index=context.index)


def score(context: pd.DataFrame) -> np.ndarray:
    """Sum of the weights of the rules that fire, one value per row."""
    flags = fired(context)
    weights = np.array([WEIGHTS[rule] for rule in RULE_IDS])
    return flags[list(RULE_IDS)].to_numpy(np.int64) @ weights


def rationales(context: pd.DataFrame) -> list[list[dict[str, object]]]:
    """For each row, the conditions that fire: id, rule, name, the rule's weight and a
    rationale citing the values it rests on."""
    _require(context)
    out: list[list[dict[str, object]]] = [[] for _ in range(len(context))]
    for condition in CONDITIONS:
        mask = condition.fire(context).fillna(False).to_numpy(bool)
        if not mask.any():
            continue
        texts = condition.explain(context[mask])
        for position, text in zip(np.flatnonzero(mask), texts, strict=True):
            out[position].append({"id": condition.id, "rule": condition.rule,
                                  "name": condition.name, "weight": WEIGHTS[condition.rule],
                                  "rationale": text})
    return out


def main(argv: list[str]) -> None:
    from core import asof, config, world

    if len(argv) != 1:
        raise SystemExit("usage: python -m rules.engine <world directory>")
    tables = world.read_world(argv[0], asof.INPUT_TABLES)
    context = asof.build_context(tables)
    approved = context["order_id"].isin(tables["order_attempts"].loc[
        tables["order_attempts"]["processor_result"] == "approved", "order_id"])
    context = context[approved.to_numpy()]
    flags = fired(context)
    scores = score(context)
    bands = config.load("policy")["rules"]["bands"]
    print(f"{len(context):,} processor-approved orders scored at checkout")
    print("fires:", {name: int(flags[name].sum()) for name in flags.columns})
    print(f"review band (>= {bands['review']}): {int((scores >= bands['review']).sum()):,}; "
          f"decline band (>= {bands['decline']}): {int((scores >= bands['decline']).sum()):,}")


if __name__ == "__main__":
    main(sys.argv[1:])
