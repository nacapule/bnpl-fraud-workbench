"""Relative links and images in the published Markdown documents must reach a file in
the repository, and a link to a heading must reach that heading.

The documents are the README, ``docs/``, ``reports/`` and ``cases/``. Checked: inline
links and images, including an image inside a link's text (``[![alt](src)](target)``),
with a destination in angle brackets or with balanced parentheses and a title in double
quotes, single quotes or parentheses; reference definitions (``[name]: target``, the
target on the same or the next line); and HTML ``src`` and ``href`` attributes. Not
checked: targets with a scheme (``https:``, ``mailto:``) and anything inside fenced code
or a code span. Like the number lint, the reader errs one way only: what it cannot place
for certain as code (indented code, a span it cannot match) is read as text, so unusual
Markdown can add a finding but never hide a link.

A target must be a file inside the repository (not a directory). A fragment
(``#heading``) must name an anchor of its Markdown target as GitHub writes it: the
heading's text (code kept, markup and tags dropped), lower case, every character but
letters, digits, ``_``, ``-`` and spaces dropped, each space a hyphen, and a repeat
numbered ``-1``, ``-2``, ... until it is unused; ``<a id="...">`` and ``<a name="...">``
count too. Only ATX headings (``#`` to ``######``) are read.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import unquote

from report.render import REPO

PUBLISHED = ("README.md", "docs/*.md", "reports/**/*.md", "cases/*.md")
FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
DEFINITION = re.compile(r"^ {0,3}\[(?:[^\[\]\\]|\\.)+\]:[ \t]*\n?[ \t]*(<[^<>\n]*>|\S+)",
                        re.MULTILINE)
HTML_ATTRIBUTE = re.compile(r"""<[A-Za-z][^<>]*?\b(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
ANCHOR = re.compile(r"""<a\s[^<>]*?\b(?:name|id)\s*=\s*["']([^"']+)["']""")
TAG = re.compile(r"<[^<>]+>")
TITLE_CLOSE = {'"': '"', "'": "'", "(": ")"}


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


def _blocks(text: str) -> list[tuple[int, str]]:
    """The text outside fenced code, as ``(first line number, block)``: runs of lines
    separated by blank lines, each with its code spans blanked."""
    blocks, current, start = [], [], 1
    for number, line in enumerate(_outside_fences(text), start=1):
        if not line.strip():
            if current:
                blocks.append((start, _blank_code("\n".join(current))))
            current = []
            continue
        if not current:
            start = number
        current.append(line)
    if current:
        blocks.append((start, _blank_code("\n".join(current))))
    return blocks


def _blank_code(block: str) -> str:
    """``block`` with every code span (a run of backticks closed by a run of the same
    length) blanked, newlines kept; an unmatched run is literal text."""
    runs = [(match.start(), match.end()) for match in re.finditer(r"`+", block)]
    out, index = list(block), 0
    while index < len(runs):
        start, end = runs[index]
        length = end - start
        closing = next((later for later in range(index + 1, len(runs))
                        if runs[later][1] - runs[later][0] == length), None)
        if closing is None:
            index += 1
            continue
        for position in range(start, runs[closing][1]):
            if out[position] != "\n":
                out[position] = " "
        index = closing + 1
    return "".join(out)


def _opening_bracket(block: str, close: int) -> int | None:
    """The ``[`` that the ``]`` at ``close`` closes, if any."""
    depth = 0
    for position in range(close, -1, -1):
        char = block[position]
        if position and block[position - 1] == "\\":
            continue
        if char == "]":
            depth += 1
        elif char == "[":
            depth -= 1
            if depth == 0:
                return position
    return None


def _destination(block: str, start: int) -> str | None:
    """The destination of an inline link whose ``(`` is at ``start - 1``, or ``None``
    when what follows is not a link's destination, title and ``)``."""
    position = start
    while position < len(block) and block[position] in " \t\n":
        position += 1
    if position < len(block) and block[position] == "<":
        end = block.find(">", position)
        if end < 0 or "\n" in block[position:end] or "<" in block[position + 1:end]:
            return None
        target, position = block[position + 1:end], end + 1
    else:
        depth, begin = 0, position
        while position < len(block):
            char = block[position]
            if char == "\\" and position + 1 < len(block):
                position += 2
                continue
            if char in " \t\n" or (char == ")" and depth == 0):
                break
            depth += char == "("
            depth -= char == ")"
            position += 1
        if depth:
            return None
        target = block[begin:position]
    gap = position
    while position < len(block) and block[position] in " \t\n":
        position += 1
    if position < len(block) and block[position] in TITLE_CLOSE and position > gap:
        close = block.find(TITLE_CLOSE[block[position]], position + 1)
        if close < 0:
            return None
        position = close + 1
        while position < len(block) and block[position] in " \t\n":
            position += 1
    if position >= len(block) or block[position] != ")":
        return None
    return target


def targets(text: str) -> list[tuple[int, str]]:
    """Every link and image target in a Markdown text, with its line, in order."""
    found = []
    for first, block in _blocks(text):
        def line(offset: int, first: int = first, block: str = block) -> int:
            return first + block.count("\n", 0, offset)
        for match in re.finditer(r"\]\(", block):
            if _opening_bracket(block, match.start()) is None:
                continue
            target = _destination(block, match.end())
            if target is not None:
                found.append((line(match.start()), match.start(), target))
        for match in DEFINITION.finditer(block):
            target = match.group(1)
            found.append((line(match.start()), match.start(),
                          target[1:-1] if target.startswith("<") else target))
        for match in HTML_ATTRIBUTE.finditer(block):
            found.append((line(match.start()), match.start(),
                          match.group(1) or match.group(2) or ""))
    return [(number, target) for number, _, target in sorted(found)]


def _heading_text(raw: str) -> str:
    """A heading's text as it renders: code spans kept literally, tags, link and image
    markup and emphasis markers dropped, entities decoded."""
    parts = re.split(r"(`+)(.+?)\1", raw)
    text = []
    for index in range(0, len(parts), 3):
        prose = parts[index]
        prose = re.sub(r"!\[([^\]]*)\]\([^)]*\)", "", prose)  # an image adds no text
        prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", prose)  # a link keeps its text
        prose = TAG.sub("", prose).replace("*", "")
        prose = re.sub(r"(?<![^\W_])_+|_+(?![^\W_])", "", prose)  # emphasis, not snake_case
        text.append(html.unescape(prose))
        if index + 2 < len(parts):
            text.append(parts[index + 2].strip() if parts[index + 2].strip() else
                        parts[index + 2])
    return "".join(text).strip()


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
            if not resolved.is_file():
                problems.append(f"{name}:{number}: link {target!r}: no such file")
                continue
            if fragment and resolved.suffix == ".md" and \
                    unquote(fragment) not in anchors(resolved.read_text()):
                problems.append(f"{name}:{number}: link {target!r}: no heading with anchor "
                                f"#{fragment} in {resolved.relative_to(root).as_posix()}")
    return problems
