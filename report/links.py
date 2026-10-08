"""Relative links and images in the published Markdown documents must reach a file in
the repository, and a link to a heading must reach that heading.

The documents are the README, ``docs/``, ``reports/`` and ``cases/``. Where the check
reads Markdown differently from GitHub, it errs toward a finding, as the number lint
does: it may report a link that is not one, but it should not let a broken link pass.

Outside fenced code, it collects whatever follows ``](`` or ``]:`` (a reference
definition, wherever it sits) and the value of an attribute named exactly ``src`` or
``href``, across line breaks. It does not tell inline code, titles or quoted attribute
text apart from links. A destination it cannot read whole (parentheses nested twice,
say) is checked as everything up to the next space. Fences are read two ways: as they
are at the top level, and ending where a list item would end and never opening inside
an HTML block. A line either reading leaves outside code is scanned. A link-like string
in our own documents that is not a link is exempted in :data:`NOT_LINKS` by file and
line text. Targets with a scheme (``https:``, ``mailto:``) are not checked.

A target, once decoded (Markdown escapes and HTML entities in Markdown, entities only
in an attribute), must be a file inside the repository. A fragment (``#heading``) must
name an anchor of its Markdown target as GitHub writes it: the heading's text (a link's
label, code kept, tags and emphasis dropped, entities decoded), lower case, every
character but letters, digits, ``_``, ``-`` and spaces dropped, each space a hyphen,
and a repeat numbered ``-1``, ``-2``, ... until it is unused. Only a heading that both
readings leave outside code, and outside any HTML block, counts; ``<a id="...">`` and
``<a name="...">`` count outside code spans, comments and indented lines. The check may
miss an anchor GitHub makes, but it does not invent one.

GitHub Markdown outside the subset our generated documents use (headings inside lists or
block quotes, setext headings, indented code, nested containers) is not modelled.
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
FENCE_OPEN = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
# a destination: <...>, or text without spaces whose parentheses are balanced, one deep
DESTINATION = r"(<[^<>\n]*>|[^\s()]*(?:\([^\s()]*\)[^\s()]*)*)"
INLINE = re.compile(r"\]\(\s*" + DESTINATION)
DEFINITION = re.compile(r"\]:[ \t]*\n?[ \t]*(\S*)")
ATTRIBUTE = re.compile(r"""(?<![\w.:-])(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>`]+))""",
                       re.IGNORECASE)
RAW = re.compile(r"\S*")
ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")  # a backslash before ASCII punctuation
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
ANCHOR = re.compile(r"""<a\s[^<>]*?\b(?:name|id)\s*=\s*["']([^"']+)["']""")
TAG = re.compile(r"<[^<>]+>")
CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LABEL = r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]"
TARGET = r"\((?:[^()]|\([^()]*\))*\)"
# How an HTML block starts, and the line that ends it (None: the next blank line).
HTML_BLOCKS = (
    (re.compile(r"<!--"), re.compile(r"-->")),
    (re.compile(r"<\?"), re.compile(r"\?>")),
    (re.compile(r"<!\[CDATA\["), re.compile(r"\]\]>")),
    (re.compile(r"<![A-Za-z]"), re.compile(r">")),
    (re.compile(r"<(?:script|pre|style|textarea)(?:[\s>]|$)", re.IGNORECASE),
     re.compile(r"</(?:script|pre|style|textarea)>", re.IGNORECASE)),
    (re.compile(r"</?[A-Za-z]"), None),
)


def published(root: Path = REPO) -> list[Path]:
    """The Markdown documents a reader opens, under ``root``."""
    return sorted({path for pattern in PUBLISHED for path in root.glob(pattern)
                   if path.is_file()})


def _blocks(lines: list[str], strict: bool) -> tuple[set[int], set[int]]:
    """The indices of the lines in fenced code (fences included), and of those in HTML
    blocks. A fence closes on a line of its character, at least as long, with nothing
    after it; unclosed, it runs to the end. Read strictly, a fence also ends at a line
    indented less than its opening fence (where a list item ends), and none opens inside
    an HTML block, whose lines run to the block's end marker or the next blank line."""
    code: set[int] = set()
    raw: set[int] = set()
    fence: tuple[str, int, int] | None = None
    block: list[re.Pattern[str] | None] | None = None
    for index, line in enumerate(lines):
        if fence is not None:
            char, length, indent = fence
            closing = r" {0,3}" + re.escape(char) + "{" + str(length) + r",}[ \t]*"
            if strict and line.strip() and len(line) - len(line.lstrip(" ")) < indent:
                fence = None  # the list item, and its fence, end here; read the line afresh
            elif re.fullmatch(closing, line):
                code.add(index)
                fence = None
                continue
            else:
                code.add(index)
                continue
        if block is not None:
            end = block[0]
            if end is None and not line.strip():
                block = None
                continue
            raw.add(index)
            if end is not None and end.search(line):
                block = None
            continue
        opening = FENCE_OPEN.match(line)
        if opening and not (opening.group(2)[0] == "`" and "`" in opening.group(3)):
            fence = (opening.group(2)[0], len(opening.group(2)), len(opening.group(1)))
            code.add(index)
            continue
        if strict:
            start = len(line) - len(line.lstrip(" "))
            for begins, end in HTML_BLOCKS if start < 4 else ():
                if found := begins.match(line, start):
                    raw.add(index)
                    if end is None or not end.search(line, found.end()):
                        block = [end]
                    break
    return code, raw


def targets(text: str, name: str = "") -> list[tuple[int, str, str]]:
    """Every link-like destination in a Markdown text, in order: its line, the target as
    written and the target decoded. Lines listed in :data:`NOT_LINKS` for the document
    ``name`` are skipped."""
    lines = text.split("\n")
    code = _blocks(lines, strict=False)[0] & _blocks(lines, strict=True)[0]
    prose = "\n".join("" if index in code or (name, line.strip()) in NOT_LINKS else line
                      for index, line in enumerate(lines))
    found = []
    for match in INLINE.finditer(prose):
        start, end = match.start(1), match.end()
        whole = end == len(prose) or prose[end].isspace() or prose[end] == ")"
        found.append((start, match.group(1) if whole else RAW.match(prose, start).group(),
                      True))
    found += [(match.start(1), match.group(1), True) for match in DEFINITION.finditer(prose)]
    for match in ATTRIBUTE.finditer(prose):
        group = next(group for group in (1, 2, 3) if match.group(group) is not None)
        found.append((match.start(group), match.group(group), False))
    read = []
    for start, target, markdown in sorted(found):
        if markdown and target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        decoded = html.unescape(ESCAPE.sub(r"\1", target) if markdown else target)
        read.append((prose.count("\n", 0, start) + 1, target, decoded))
    return read


def _heading_text(raw: str) -> str:
    """A heading's text as it renders: link and image markup dropped (a link keeps its
    label), code spans kept literally, tags and emphasis markers dropped, entities
    decoded."""
    code: list[str] = []

    def hold(match: re.Match[str]) -> str:
        span = match.group(2)
        if span.startswith(" ") and span.endswith(" ") and span.strip():
            span = span[1:-1]  # a code span drops one space from each side
        code.append(span)
        return f"\x00{len(code) - 1}\x00"

    text = CODE.sub(hold, raw)  # code first, so code is never read as markup
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
    lines = text.split("\n")
    code, raw = _blocks(lines, strict=True)
    code |= _blocks(lines, strict=False)[0]
    occurrences: dict[str, int] = {}
    found = set()
    kept = []
    for index, line in enumerate(lines):
        if index in code:
            kept.append("")
            continue
        if index not in raw and (heading := HEADING.match(line)):
            base = anchor = slug(heading.group(2) or "")
            while anchor in occurrences:
                occurrences[base] += 1
                anchor = f"{base}-{occurrences[base]}"
            occurrences[anchor] = 0
            found.add(anchor)
        kept.append("" if line.startswith(("    ", "\t")) else line)  # perhaps indented code
    found.update(ANCHOR.findall(CODE.sub("", COMMENT.sub("", "\n".join(kept)))))
    return found


def broken(root: Path = REPO, documents: list[Path] | None = None) -> list[str]:
    """Each relative link or image in the published documents whose file or heading does
    not exist, as ``<document>:<line>: ...``."""
    root = root.resolve()
    problems = []
    for document in documents if documents is not None else published(root):
        text = document.read_text()
        name = document.resolve().relative_to(root).as_posix()
        for number, written, target in targets(text, name):
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
