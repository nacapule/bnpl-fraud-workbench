"""Relative links and images in the published Markdown documents must reach a file
in the repository, and a link to a heading must reach that heading.

The documents are the README, ``docs/``, ``reports/`` and ``cases/``. Inline links and
images (``[text](target)``, ``![alt](target)``), reference definitions
(``[name]: target``) and HTML ``src``/``href`` attributes are checked; targets with a
scheme (``https:``, ``mailto:``) are not, nor is anything inside code. A fragment
(``#heading``) must name a heading of its target document as GitHub writes its anchor:
lower case, punctuation other than hyphens and underscores dropped, spaces as hyphens,
and ``-1``, ``-2``... for repeats.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from report.render import REPO

PUBLISHED = ("README.md", "docs/*.md", "reports/**/*.md", "cases/*.md")
FENCE = re.compile(r"^\s{0,3}(```|~~~)")
CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")
INLINE = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
DEFINITION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
HTML = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
ANCHOR = re.compile(r"""<a\s+(?:name|id)\s*=\s*["']([^"']+)["']""")


def published(root: Path = REPO) -> list[Path]:
    """The Markdown documents a reader opens, under ``root``."""
    return sorted({path for pattern in PUBLISHED for path in root.glob(pattern)
                   if path.is_file()})


def _prose(text: str, spans: bool = True) -> list[tuple[int, str]]:
    """``(line number, line)`` outside fenced code, with code spans blanked unless
    ``spans`` is false."""
    out, fence = [], None
    for number, line in enumerate(text.split("\n"), start=1):
        marker = FENCE.match(line)
        if fence is not None:
            if marker and marker.group(1) == fence:
                fence = None
            continue
        if marker:
            fence = marker.group(1)
            continue
        out.append((number, CODE_SPAN.sub(lambda m: " " * len(m.group(0)), line)
                     if spans else line))
    return out


def targets(text: str) -> list[tuple[int, str]]:
    """Every link and image target in a Markdown text, with its line."""
    found = []
    for number, line in _prose(text):
        for pattern in (INLINE, DEFINITION, HTML):
            for match in pattern.finditer(line):
                target = match.group(1)
                found.append((number, target[1:-1] if target.startswith("<") else target))
    return found


def slug(heading: str) -> str:
    """A heading's anchor as GitHub writes it (without the repeat suffix)."""
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # a link keeps its text
    text = text.replace("`", "").lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors of a Markdown text's headings, and explicit HTML anchors."""
    seen: dict[str, int] = {}
    found = set()
    for _, line in _prose(text, spans=False):
        if heading := HEADING.match(line):
            base = slug(heading.group(2))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
        found.update(ANCHOR.findall(line))
    return found


def broken(root: Path = REPO, documents: list[Path] | None = None) -> list[str]:
    """Each relative link or image in the published documents whose file or heading does
    not exist, as ``<document>:<line>: ...``."""
    root = root.resolve()
    problems = []
    for document in documents if documents is not None else published(root):
        text = document.read_text()
        name = document.resolve().relative_to(root).as_posix()
        for number, target in targets(text):
            if SCHEME.match(target) or target.startswith("//"):
                continue
            path_part, _, fragment = target.partition("#")
            path_part = unquote(path_part)
            if path_part.startswith("/"):
                problems.append(f"{name}:{number}: link {target!r} is absolute; use a path "
                                "relative to the document")
                continue
            resolved = (document.parent / path_part).resolve() if path_part else \
                document.resolve()
            if root not in (resolved, *resolved.parents):
                problems.append(f"{name}:{number}: link {target!r} leaves the repository")
                continue
            if not resolved.exists():
                problems.append(f"{name}:{number}: link {target!r}: no such file")
                continue
            if fragment and resolved.suffix == ".md" and \
                    unquote(fragment) not in anchors(resolved.read_text()):
                problems.append(f"{name}:{number}: link {target!r}: no heading with anchor "
                                f"#{fragment} in {resolved.relative_to(root).as_posix()}")
    return problems
