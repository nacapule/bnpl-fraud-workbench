"""Render or check the published documents.

    python -m report render [--summary results/summary.json] [--out DIR]
    python -m report check  [--summary results/summary.json]

``render`` writes every document from its template. The repository's own
documents are written only from the committed ``results/summary.json``; a
summary from another run renders into ``--out`` elsewhere. ``check`` changes
nothing: it fails when a committed document differs from a fresh render, a
claim no longer holds or is missing from its document, or the lint finds a
number outside a rendered value or an unlisted directional sentence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.results import read_summary
from report import claims as claims_module
from report import lint as lint_module
from report.render import REPO, SUMMARY, Sources, documents, out_of_sync, render_all


def _sources(summary_path: Path) -> Sources:
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} does not exist; run the pipeline first")
    return Sources.from_repo(read_summary(summary_path))


def check(summary_path: Path = SUMMARY, root: Path = REPO) -> list[str]:
    """Every problem with the committed documents, claims and templates."""
    claims = claims_module.load_claims()
    config = lint_module.load_config()
    problems = lint_module.lint(config, claims_module.sentences_by_document(claims), root)
    if not documents() and not claims:
        return problems
    sources = _sources(summary_path)
    problems += out_of_sync(sources, root)

    def read(name: str) -> str | None:
        path = root / name
        return path.read_text() if path.exists() else None

    vocabulary = claims_module.Vocabulary.from_config(config)
    problems += claims_module.check_claims(claims, sources.summary, read, vocabulary)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m report", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("render", "check"))
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--out", type=Path, default=REPO, help="where to write documents")
    args = parser.parse_args(argv)
    if args.command == "render":
        if args.out.resolve() == REPO and args.summary.resolve() != SUMMARY:
            print("the repository's documents are rendered only from results/summary.json; "
                  "give --out for another summary", file=sys.stderr)
            return 1
        for path in render_all(_sources(args.summary), args.out):
            print(f"wrote {path}")
        return 0
    problems = check(args.summary)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"documents: {len(documents())} templates, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
