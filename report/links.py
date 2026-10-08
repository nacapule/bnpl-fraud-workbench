"""Relative links and images in the published Markdown documents must reach a file in
the repository, and a link to a heading must reach that heading.

The documents are the README, ``docs/``, ``reports/`` and ``cases/``. The check reads
every line outside fenced code blocks and collects every link-like destination: whatever
follows ``](``, the target of a reference definition (``[name]: target``), and the
value of an attribute named exactly ``src`` or ``href``. It does not try to tell inline
code, titles or quoted attribute text apart from links, so a construct it misreads can
add a finding but never hide one, as with the number lint; a link-like string in our own
documents that is not a link is exempted in :data:`NOT_LINKS` by file and line text.
Targets with a scheme (``https:``, ``mailto:``) are not checked.

A target, once its Markdown escapes and HTML entities are decoded, must be a file inside
the repository. A fragment (``#heading``) must name an anchor of its Markdown target as
GitHub writes it: the heading's text (a link's label, code kept, tags and emphasis
dropped, entities decoded), lower case, every character but letters, digits, ``_``,
``-`` and spaces dropped, each space a hyphen, and a repeat numbered ``-1``, ``-2``, ...
until it is unused; ``<a id="...">`` and ``<a name="...">`` count too.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import unquote

from report.render import REPO

PUBLISHED = ("README.md", "docs/*.md", "reports/**/*.md", "cases/*.md")
# Link-like strings in the documents that are not links: (document, the line's text).
NOT_LINKS: frozenset[tuple[str, str]] = frozenset()
FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# a destination: <...>, or text without spaces whose parentheses are balanced, one deep
DESTINATION = r"(<[^<>\n]*>|[^\s()]*(?:\([^\s()]*\)[^\s()]*)*)"
INLINE = re.compile(r"\]\(\s*" + DESTINATION)
DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:[ \t]*(\S*)")
ATTRIBUTE = re.compile(r"""(?<![\w.:-])(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>`]+))""",
                       re.IGNORECASE)
ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")  # a backslash before ASCII punctuation
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
ANCHOR = re.compile(r"""<a\s[^<>]*?\b(?:name|id)\s*=\s*["']([^"']+)["']""")
TAG = re.compile(r"<[^<>]+>")
LABEL = r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]"
TARGET = r"\((?:[^()]|\([^()]*\))*\)"


def published(root: Path = REPO) -> list[Path]:
    """The Markdown documents a reader opens, under ``root``."""
    return sorted({path for pattern in PUBLISHED for path in root.glob(pattern)
                   if path.is_file()})


def _outside_fences(text: str) -> list[str]:
    """The text's lines, with fenced code and its fences emptied. A fence closes on a
    line of the same character, at least as long, with nothing after it; an unclosed
    fence runs to the end."""
    kept, fence = [], None
    for line in text.split("\n"):
        if fence is not None:
            char, length = fence
            if re.fullmatch(r" {0,3}" + re.escape(char) + "{" + str(length) + r",}[ \t]*", line):
                fence = None
            kept.append("")
            continue
        opening = FENCE_OPEN.match(line)
        if opening and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            fence = (opening.group(1)[0], len(opening.group(1)))
            kept.append("")
            continue
        kept.append(line)
    return kept


def targets(text: str, name: str = "") -> list[tuple[int, str]]:
    """Every link-like destination in a Markdown text, with its line, in order; lines
    listed in :data:`NOT_LINKS` for the document ``name`` are skipped."""
    lines = _outside_fences(text)
    found = []
    for index, line in enumerate(lines):
        if (name, line.strip()) in NOT_LINKS:
            continue
        number = index + 1
        for match in INLINE.finditer(line):
            target = match.group(1)
            if not target and match.end() == len(line.rstrip()) and index + 1 < len(lines):
                target = re.match(DESTINATION, lines[index + 1].strip()).group(1)
            found.append((match.start(), number, target))
        if definition := DEFINITION.match(line):
            target = definition.group(1)
            if not target and index + 1 < len(lines):
                target = (lines[index + 1].split() or [""])[0]
            found.append((definition.start(), number, target))
        for match in ATTRIBUTE.finditer(line):
            found.append((match.start(), number,
                          next(group for group in match.groups() if group is not None)))
    found.sort(key=lambda item: (item[1], item[0]))
    return [(number, target[1:-1] if target.startswith("<") and target.endswith(">")
             else target) for _, number, target in found]


def _heading_text(raw: str) -> str:
    """A heading's text as it renders: link and image markup dropped (a link keeps its
    label), code spans kept literally, tags and emphasis markers dropped, entities
    decoded."""
    code: list[str] = []

    def hold(match: re.Match[str]) -> str:
        code.append(match.group(2))
        return f"\x00{len(code) - 1}\x00"

    text = re.sub(r"(`+)(.+?)\1", hold, raw)  # code first, so code is never read as markup
    text = re.sub("!" + LABEL + "(?:" + TARGET + r"|\[[^\]]*\])", "", text)
    text = re.sub(LABEL + "(?:" + TARGET + r"|\[[^\]]*\])", r"\1", text)
    text = TAG.sub("", text).replace("*", "")
    text = re.sub(r"(?<![^\W_])_+|_+(?![^\W_])", "", text)  # emphasis, not snake_case
    text = html.unescape(text)
    return re.sub("\x00(\\d+)\x00", lambda match: code[int(match.group(1))], text).strip()


def slug(heading: str) -> str:
    """A heading's anchor as GitHub writes it, before any repeat number."""
    text = _heading_text(heading).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors of a Markdown text's headings, and explicit HTML anchors."""
    occurrences: dict[str, int] = {}
    found = set()
    for line in _outside_fences(text):
        if heading := HEADING.match(line):
            base = anchor = slug(heading.group(2) or "")
            while anchor in occurrences:
                occurrences[base] += 1
                anchor = f"{base}-{occurrences[base]}"
            occurrences[anchor] = 0
            found.add(anchor)
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
        for number, written in targets(text, name):
            target = html.unescape(ESCAPE.sub(r"\1", written))
            if SCHEME.match(target) or target.startswith("//"):
                continue
            path_part, _, fragment = target.partition("#")
            path_part = unquote(path_part)
            if path_part.startswith("/"):
                problems.append(f"{name}:{number}: link {written!r} is absolute; use a path "
                                "relative to the document")
                continue
            resolved = (document.parent / path_part).resolve() if path_part else \
                document.resolve()
            if root not in (resolved, *resolved.parents):
                problems.append(f"{name}:{number}: link {written!r} leaves the repository")
                continue
            if not resolved.is_file():
                problems.append(f"{name}:{number}: link {written!r}: no such file")
                continue
            if fragment and resolved.suffix == ".md" and \
                    unquote(fragment) not in anchors(resolved.read_text()):
                problems.append(f"{name}:{number}: link {written!r}: no heading with anchor "
                                f"#{fragment} in {resolved.relative_to(root).as_posix()}")
    return problems
