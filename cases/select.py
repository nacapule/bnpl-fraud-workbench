"""Choose the case files' alerts and write their facts.

    python -m cases.select --run runs/final [--out cases/facts]

Reads the protocol's ``cases`` rule and the run's canonical world with what its replay
kept (:mod:`cases.facts`), selects one alert per slot (:mod:`cases.rule`), writes one
``<file>.json`` per case file into ``--out`` and prints each slot's pick and candidate
counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cases import facts as facts_module
from cases.rule import Pick

REPO = Path(__file__).resolve().parent.parent
FACTS = REPO / "cases" / "facts"


def describe(pick: Pick) -> str:
    counts = (f"{pick.eligible} eligible, {pick.reviewed} reviewed, "
              + ", ".join(f"{tier} {n}" for tier, n in pick.tiers.items()))
    if not pick.selected:
        return f"{pick.slot}: no alert ({counts})"
    return f"{pick.slot}: order {pick.order_id}, {pick.tier} ({counts})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cases.select",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True,
                        help="a pipeline run directory (runs/<name>)")
    parser.add_argument("--out", type=Path, default=FACTS, help="where the facts go")
    args = parser.parse_args(argv)
    run = facts_module.load_run(args.run)
    picks = run.picks()
    for pick in picks.values():
        print(describe(pick))
    for path in facts_module.write(facts_module.build(run, picks), args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
