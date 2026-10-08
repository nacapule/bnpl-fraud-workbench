"""Render or check the published documents.

    python -m report render [--summary results/summary.json] [--out DIR] [--strict]
    python -m report check  [--summary results/summary.json]

``render`` writes every document from its template. The repository's own
documents are written only from the committed ``results/summary.json``, and a
document that cannot be rendered fails the render; a summary from another run
renders into ``--out`` elsewhere, where a document it cannot render is skipped and
listed in ``NOT_RENDERED.txt`` (``--strict`` fails instead). ``check`` changes
nothing: it fails when a committed document differs from a fresh render, a
claim no longer holds or no template places it, the lint finds a number or
a comparative word typed into a template, or a relative link in a published
document reaches no file or heading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.results import read_summary
from report import claims as claims_module
from report import links as links_module
from report import lint as lint_module
from report.render import (
    REPO,
    SUMMARY,
    Sources,
    documents,
    is_repository,
    out_of_sync,
    render_all,
)


def _sources(summary_path: Path) -> Sources:
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} does not exist; run the pipeline first")
    return Sources.from_repo(read_summary(summary_path))


def check(summary_path: Path = SUMMARY, root: Path = REPO) -> list[str]:
    """Every problem with the committed documents, claims and templates."""
    claims, wording = claims_module.load_claims(root / "report" / "claims.yaml")
    problems = lint_module.lint(lint_module.load_config(), root)
    problems += links_module.broken(root)
    if not documents() and not claims:
        return problems
    sources = _sources(summary_path)
    problems += out_of_sync(sources, root)
    used = claims_module.placed(doc.template.read_text() for doc in documents())
    problems += claims_module.check_claims(claims, sources.summary, wording, used)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m report", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("render", "check"))
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--out", type=Path, default=REPO, help="where to write documents")
    parser.add_argument("--strict", action="store_true",
                        help="fail on a document that cannot be rendered, outside the "
                             "repository too")
    args = parser.parse_args(argv)
    if args.command == "render":
        same_summary = args.summary.resolve() == SUMMARY.resolve() or (
            args.summary.exists() and SUMMARY.exists() and args.summary.samefile(SUMMARY))
        if is_repository(args.out) and not same_summary:
            print("the repository's documents are rendered only from results/summary.json; "
                  "give --out for another summary", file=sys.stderr)
            return 1
        for path in render_all(_sources(args.summary), args.out, strict=args.strict):
            print(f"wrote {path}")
        return 0
    problems = check(args.summary)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"documents: {len(documents())} templates, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
