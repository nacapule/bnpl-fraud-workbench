"""Lint for numbers and comparisons typed into the published documents' templates.

Two checks on a template's own text, configured in ``report/lint.yaml``:

* **numbers**: no digits, except identifiers such as rule ids (R01), policy
  clauses (FP-2 §6.3), queue priorities (P0), SQL queries (Q12), case ids
  (CASE-01), numbered-list and footnote markers, and the phrases the
  configuration allows. Every published number therefore comes from a rendered
  result key or setting.
* **comparisons**: in the templates of the documents named under
  ``directional``, no comparative word (more, fewer, higher, better, beats,
  ...). A published comparison is a claim placed with ``{{ claim:<id> }}`` and
  written from the results (``report/claims.py``); a sentence that compares
  without stating a result ("a lower threshold holds more orders") is allowed
  by listing it under ``allowed_sentences``.

Documents matching ``documents`` must have a template, except those listed
under ``not_yet_templated``: the hand-written documents that predate the
templates. That list may only shrink: it must stay within
:data:`EXEMPTION_BASELINE`.

A template's own text is everything outside its placeholders, read the way the
published page shows it, except that the lints skip only what they can place
for certain as code or as hidden: fenced code blocks that start at the left
margin, code spans and link targets on one line, HTML comments, tags, and
reference definitions. Anything they cannot place for certain (code inside a
list or a quote, indented code, an unclosed construct, raw HTML) is read as
text, so unusual Markdown can add a finding but never hide one. Put code that
holds digits or comparatives in a fenced block at the left margin.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from report.render import PLACEHOLDER, REPO, TEMPLATES, documents

CONFIG = REPO / "report" / "lint.yaml"
# The hand-written documents that predate the templates; none may be added.
EXEMPTION_BASELINE = frozenset({
    "README.md",
    "cases/CASE-01-account-takeover.md",
    "cases/CASE-02-card-testing-stolen-card.md",
    "cases/CASE-03-synthetic-ring-bustout.md",
    "cases/CASE-04-neverpay-vs-hardship.md",
    "cases/CASE-05-traveler-cleared.md",
})
# An identifier ends where its grammar ends: no letter, digit, underscore, percent
# sign, opening parenthesis or decimal may follow.
END = r"(?![\w%(]|[.,]\d)"
QUALIFIERS = r"(?:\([a-z]+\))*"  # (a), (b)(ii): lowercase letters only
# Each identifier is matched whole (an atomic group) before the boundary is
# checked, so dropping a qualifier the grammar does not take cannot exempt a
# prefix: §84(A)% and §6.3(2) are numbers.
IDENTIFIERS = (
    rf"(?>\bR\d{{2}}{QUALIFIERS}){END}",  # rule ids, R06(b)
    rf"(?>\bFP-\d+){END}",  # policy version
    rf"(?>§ ?\d+(?:\.\d+)*{QUALIFIERS}){END}",  # policy clauses, §6.6(b)
    rf"(?>\bP[0-3]){END}",  # queue priorities
    rf"(?>\bQ\d{{2}}){END}",  # SQL investigation queries
    rf"(?>\bCASE-\d{{2}}){END}",  # case files
)
# A footnote reference, unless escaped, an image, or part of a link (a link's text,
# or followed by its target or label), where its label shows as typed.
FOOTNOTE_MARKER = re.compile(r"(?<![\\\]!])\[\^([^\]\s]+)\](?![(\[])")
FOOTNOTE_DEFINITION = re.compile(r"\[\^([^\]\s]+)\]:")
NUMBER = re.compile(r"\d[\d,.]*")

# ---------------------------------------------------------- reading a template
FILL = "\x1a"  # holds a placeholder's place while the structure is read
PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
BACKTICKS = re.compile(r"`+")
# Link destinations and titles, on one line (CommonMark 0.31 §6.3).
_PLAIN = r"(?:[^\s()\\]|\\\S)"  # a backslash cannot escape a space
_NESTED = rf"\({_PLAIN}*\)"
for _ in range(2):
    _NESTED = rf"\((?:{_PLAIN}|{_NESTED})*\)"
DESTINATION = rf"(?:<(?:[^<>\n\\]|\\.)*>|(?!<)(?:{_PLAIN}|{_NESTED})+)"
TITLE = r"""(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|\((?:[^()\\\n]|\\.)*\))"""
LINK_TARGET = re.compile(rf"\([ \t]*(?:{DESTINATION}(?:[ \t]+{TITLE})?)?[ \t]*\)")
REFERENCE_DEFINITION = re.compile(
    rf" {{0,3}}\[(?!\^)(?![ \t]*\])(?:[^\[\]\\\n]|\\.){{1,999}}\]:[ \t]*{DESTINATION}"
    rf"(?:[ \t]+{TITLE})?[ \t]*")
# Raw HTML on one line (§6.6), as both CommonMark 0.29 and 0.31 read it.
TAG_NAME = r"[A-Za-z][A-Za-z0-9-]*"
ATTRIBUTE = (r"[ \t]+[A-Za-z_:][A-Za-z0-9_.:-]*"
             r"""(?:[ \t]*=[ \t]*(?:[^ \t\n"'=<>`]+|'[^'\n]*'|"[^"\n]*"))?""")
TAG = rf"(?:<{TAG_NAME}(?:{ATTRIBUTE})*[ \t]*/?>|</{TAG_NAME}[ \t]*>)"
# A comment both versions accept and a browser closes at the same place.
INLINE_COMMENT = re.compile(r"<!--(?!-?>)(?:[^-\n]|-(?!-))*(?<!-)-->")
INLINE_HTML = re.compile(
    rf"{TAG}|<\?[^\n]*?\?>|<![A-Z]+[ \t]+[^>\n]*>|<!\[CDATA\[[^\n]*?\]\]>")
AUTOLINK = re.compile(
    r"<[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\x00-\x20<>]*>"
    r"|<[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*>")
# Where a browser ends a comment in raw HTML: at once (<!-->), or at --> or --!>.
COMMENT_END = re.compile(r"--!?>")
# HTML blocks (§4.6): the start conditions, in order, and the line that ends each.
RAW_TAGS = "pre|script|style|textarea"
RAW_END = re.compile(rf"</(?:{RAW_TAGS})>", re.I)
BLOCK_TAGS = (
    "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|"
    "details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|"
    "h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|nav|noframes|ol|"
    "optgroup|option|p|param|section|summary|table|tbody|td|tfoot|th|thead|title|tr|"
    "track|ul")
HTML_BLOCKS = (  # (start, end): the block ends with the first line holding its end
    (re.compile(rf"<(?:{RAW_TAGS})(?:[ \t>]|$)", re.I), RAW_END),
    (re.compile(r"<!--"), re.compile(r"-->")),
    (re.compile(r"<\?"), re.compile(r"\?>")),
    (re.compile(r"<![A-Z]"), re.compile(r">")),
    (re.compile(r"<!\[CDATA\["), re.compile(r"\]\]>")),
    (re.compile(rf"</?(?:{BLOCK_TAGS})(?:[ \t>]|/>|$)", re.I), None),
)
# Lines that CommonMark 0.29 and 0.31 read differently: one of them opens an HTML block.
UNSETTLED = (re.compile(r"<![a-z]"), re.compile(r"</?(?:source|search)(?:[ \t>]|/>|$)", re.I))
TYPE_7 = re.compile(rf"{TAG}[ \t]*")
NOT_TYPE_7 = re.compile(rf"</?(?:{RAW_TAGS})(?![A-Za-z0-9-])", re.I)
# What may start raw HTML wherever a line's content begins, and where it may end.
RAW_REGIONS = (
    (re.compile(rf"<(?:{RAW_TAGS})(?![A-Za-z0-9-])", re.I), RAW_END, 1),
    (re.compile(r"<!--"), re.compile(r"-->"), 4),
    (re.compile(r"<\?"), re.compile(r"\?>"), 2),
    (re.compile(r"<!\[CDATA\["), re.compile(r"\]\]>"), 9),
    (re.compile(r"<![A-Za-z]"), re.compile(r">"), 2),
)
CONTAINER_MARKS = re.compile(r"(?:[ \t]*(?:>[ \t]?|(?:[-+*]|\d{1,9}[.)])(?=[ \t]|$)))*[ \t]*")
# Elements whose content a browser does not read as HTML (shown as plain text, or
# run), or reads as foreign content: a template with one is read with nothing skipped.
TEXT_ELEMENTS = re.compile(
    r"</?(?:textarea|title|xmp|plaintext|noscript|noembed|noframes|iframe|svg|math|script"
    r"|style)"
    r"(?![A-Za-z0-9-])",
    re.I)
TAG_START = re.compile(r"</?[A-Za-z]")
QUOTES = re.compile(r"(?: {0,3}>[ ]?)* {0,3}")  # quote marks a paragraph line can open
SETEXT = re.compile(r" {0,3}(?:=+|-+)[ \t]*")
TABLE_DELIMITER = re.compile(r"[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*")
ORDERED_ITEM = re.compile(r"((?:[ \t]*>[ ]?)*([ \t]*))(\d{1,9})([.)])([ \t]+|$)")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str  # "number" or "comparison"
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.text}"


def _words(value: Any, where: str) -> list[str]:
    """A list of strings, refusing what YAML turned into something else (no: False)."""
    items = list(value or [])
    if bad := [item for item in items if not isinstance(item, str)]:
        raise ValueError(f"{where}: {bad} are not text; quote them in report/lint.yaml")
    return items


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    config = {
        "documents": list(data.get("documents", [])),
        "not_yet_templated": list(data.get("not_yet_templated", [])),
        "allowed_phrases": _words(data.get("allowed_phrases"), "allowed_phrases"),
        "directional": list(data.get("directional", [])),
        "comparatives": _words(data.get("comparatives"), "comparatives"),
        "allowed_sentences": _words(data.get("allowed_sentences"), "allowed_sentences"),
    }
    if unfinished := [text for text in config["allowed_sentences"]
                      if not text.strip() or text.strip()[-1] not in ".!?"]:
        raise ValueError(f"allowed_sentences: {unfinished} must each end with its full stop")
    return config


def _blank(match: re.Match[str]) -> str:
    """Replace a match with spaces, keeping newlines so line numbers hold."""
    return re.sub(r"[^\n]", " ", match.group(0))


def _fill(match: re.Match[str]) -> str:
    """A placeholder held by filler, keeping its closing brace: what it renders is not
    the template's own text, but a sentence after it starts a new one."""
    return re.sub(r"[^\n]", FILL, match.group(0))[:-1] + "}"


def _blank_line(line: str) -> bool:
    return not line.strip(" \t")  # CommonMark's blank line: spaces and tabs only


def _column(line: str) -> int:
    """The column where a line's content starts, with tab stops every four columns."""
    column = 0
    for char in line:
        if char == " ":
            column += 1
        elif char == "\t":
            column += 4 - column % 4
        else:
            break
    return column


def _fence_marker(line: str) -> str | None:
    """The backticks or tildes of a fenced code block this line opens at the margin."""
    match = re.match(r"(`{3,})[^`]*$|(~{3,})", line)
    return match and (match.group(1) or match.group(2))


def _closes(line: str, marker: str) -> bool:
    match = re.fullmatch(r" {0,3}(`+|~+)[ \t]*", line)
    return bool(match) and match.group(1)[0] == marker[0] and len(match.group(1)) >= len(marker)


def _html_block(line: str, paragraph: bool) -> tuple[re.Pattern[str] | None] | str | None:
    """The HTML block a line at the margin opens, as a 1-tuple of the pattern of its
    last line (None: a blank line ends it); "unsettled" if the line cannot be placed
    for certain; None if it opens none."""
    if any(pattern.match(line) for pattern in UNSETTLED):
        return "unsettled"
    for start, end in HTML_BLOCKS:
        if start.match(line):
            return (end,)
    if TYPE_7.fullmatch(line) and not NOT_TYPE_7.match(line):
        # an HTML block only where no paragraph is open, which is not always known
        return "unsettled" if paragraph else (None,)
    return None


def _top_level(lines: list[str]) -> tuple[list[bool], list[bool], dict[int, int], int]:
    """Find the fenced code blocks and HTML blocks that open at the left margin.

    A line at the margin that opens a fence or an HTML block does so at the top level:
    it is not indented into a list item and cannot continue a quote. The reading stops
    at the first line it cannot place for certain (a fence or HTML line indented one to
    three columns, which may sit in a list item; an unclosed block; a line the
    CommonMark versions read differently), and what follows is left as text.
    Returns which lines are fenced code, which belong to an HTML block, the comment
    blocks (first line to last), and the line where the reading stopped.
    """
    fenced, html = [False] * len(lines), [False] * len(lines)
    comments: dict[int, int] = {}
    index, paragraph = 0, False
    while index < len(lines):
        line = lines[index]
        if _blank_line(line):
            index, paragraph = index + 1, False
            continue
        content = line.lstrip(" \t")
        if not content.startswith(("```", "~~~", "<")) or _column(line) >= 4:
            index, paragraph = index + 1, True
            continue
        if _column(line) > 0:
            break
        marker = _fence_marker(line)
        if marker:
            close = next((later for later in range(index + 1, len(lines))
                          if _closes(lines[later], marker)), None)
            if close is None:
                break
            for later in range(index, close + 1):
                fenced[later] = True
            index, paragraph = close + 1, False
            continue
        block = None if content[0] != "<" else _html_block(line, paragraph)
        if block is None:
            index, paragraph = index + 1, True
            continue
        if block == "unsettled":
            break
        (end,) = block
        if end is None:  # to the next blank line
            while index < len(lines) and not _blank_line(lines[index]):
                html[index], index = True, index + 1
            paragraph = False
            continue
        last = next((later for later in range(index, len(lines)) if end.search(lines[later])),
                    None)
        if last is None:
            break
        for later in range(index, last + 1):
            html[later] = True
        if line.startswith("<!--"):
            comments[index] = last
        index, paragraph = last + 1, False
    return fenced, html, comments, index


def _comment_end(text: str, start: int) -> int | None:
    """Where a browser ends the comment opened by the ``<!--`` at ``start``."""
    body = start + 4
    if text.startswith(">", body):
        return body + 1
    if text.startswith("->", body):
        return body + 2
    match = COMMENT_END.search(text, body)
    return match and match.end()


def _tag_end(line: str, start: int) -> int | None:
    """Where a browser ends the tag opened at ``start``: the first ``>`` outside an
    attribute value; None if the tag runs past the line."""
    index = start + 1
    while index < len(line):
        char = line[index]
        if char == ">":
            return index + 1
        index += 1
        if char == "=":
            while index < len(line) and line[index] in " \t\f":
                index += 1
            if index < len(line) and line[index] in "\"'":
                close = line.find(line[index], index + 1)
                if close < 0:
                    return None
                index = close + 1
            else:
                while index < len(line) and line[index] not in " \t\f>":
                    index += 1
    return None


def _raw_comments(line: str, start: int) -> list[tuple[int, int]] | None:
    """The comments a browser reads in one line of raw HTML, from ``start`` with no tag,
    attribute or comment open; None if one of those runs past the line."""
    spans, index = [], start
    while (index := line.find("<", index)) >= 0:
        if line.startswith("<!--", index):
            end = _comment_end(line, index)
            if end is None:
                return None
            spans.append((index, end))
        elif TAG_START.match(line, index):
            end = _tag_end(line, index)
        elif line.startswith(("<!", "<?", "</"), index):  # read up to its > and dropped
            end = line.find(">", index) + 1 or None
        else:
            end = index + 1
        if end is None:
            return None
        index = end
    return spans


def _hide_raw_comments(lines: list[str], source: list[str], raw: list[bool],
                       html: list[bool], comments: dict[int, int]) -> list[str]:
    """Blank the comments of HTML blocks, reading raw HTML in order as a browser does,
    for as long as that reading is certain: from the first tag, attribute or comment
    that may run past its line, nothing more is blanked."""
    out = list(lines)

    def blank(index: int, spans: list[tuple[int, int]]) -> None:
        line = out[index]
        for start, end in spans:
            line = line[:start] + " " * (end - start) + line[end:]
        out[index] = line

    index = 0
    while index < len(source):
        if not raw[index]:
            index += 1
            continue
        first, start = index, 0
        if index in comments:  # a comment block: hidden up to where a browser ends it
            last = comments[index]
            text = "\n".join(source[index:last + 1])
            stop = _comment_end(text, text.index("<!--"))
            if stop is None:
                return out
            offset = 0
            for line in range(index, last + 1):
                begin, end = offset, offset + len(source[line])
                if begin < stop:
                    blank(line, [(0, min(stop, end) - begin)])
                if stop <= end:
                    first, start = line, stop - begin
                    break
                offset = end + 1
        for line in range(first, comments.get(index, index) + 1):
            spans = _raw_comments(source[line], start if line == first else 0)
            if spans is None:
                return out
            if html[line]:
                blank(line, spans)
        index = comments.get(index, index) + 1
    return out


def _raw_lines(lines: list[str], fenced: list[bool]) -> list[bool]:
    """Lines that may be raw HTML, where Markdown is not read: from any line whose
    content (after list and quote marks) starts with a tag, comment or declaration,
    to the end CommonMark gives that kind of block, or to the next blank line."""
    raw, to_blank, ends = [False] * len(lines), False, []
    for index, line in enumerate(lines):
        if fenced[index]:
            continue
        if _blank_line(line):
            to_blank = False
            continue
        raw[index] = to_blank or bool(ends)
        ends = [end for end in ends if not end.search(line)]
        content = line[CONTAINER_MARKS.match(line).end():]
        if re.match(r"<[A-Za-z/!?]", content):
            raw[index] = True
            region = next(((end, skip) for start, end, skip in RAW_REGIONS
                           if start.match(content)), None)
            if region is None:
                to_blank = True
            elif not region[0].search(content, region[1]):
                ends.append(region[0])
    return raw


def _definitions(lines: list[str], skip: list[bool], breaks: list[bool]) -> list[bool]:
    """Link reference definitions on one line, where a paragraph starts. Definitions
    followed by a setext underline or a table's delimiter row are left as text: some
    readers take them as the heading's or the table's text."""
    defined = [False] * len(lines)
    for index, line in enumerate(lines):
        starts = breaks[index] or (index > 0 and defined[index - 1])
        defined[index] = starts and not skip[index] and bool(REFERENCE_DEFINITION.fullmatch(line))
    for index, line in enumerate(lines):
        if SETEXT.fullmatch(line) or TABLE_DELIMITER.fullmatch(line):
            above = index - 1
            while above >= 0 and defined[above]:
                defined[above], above = False, above - 1
    return defined


def _scan(text: str, limit: int) -> str:
    """Blank the code spans, comments and link targets in a run of lines, read
    left to right as CommonMark reads them, up to ``limit`` or to the first thing that
    cannot be read for certain on its own line."""
    out = list(text)

    def blank(start: int, end: int) -> bool:
        if "|" in text[start:end]:
            return False  # a table cell may end inside it
        out[start:end] = [char if char == "\n" else " " for char in text[start:end]]
        return True

    openers: list[list[Any]] = []  # [position, image, active: True, False or None (unknown)]
    label = False  # this [ may be a reference label
    index = 0
    while index < min(limit, len(text)):
        char, follows_label, label = text[index], label, False
        line_end = text.find("\n", index) % (len(text) + 1)
        if char == "\\" and text[index + 1:index + 2] in PUNCTUATION:
            index += 2
        elif char == "`":
            run = BACKTICKS.match(text, index).end() - index
            close = next((match.start() for match in BACKTICKS.finditer(text, index + run)
                          if len(match.group(0)) == run), None)
            if close is None:
                index += run
                continue
            if close > line_end or not blank(index, close + run):
                break
            index = close + run
        elif char == "<":
            if match := AUTOLINK.match(text, index) or INLINE_HTML.match(text, index):
                # read past, as text (an autolink shows; so do some attributes)
                if "|" in match.group(0):
                    break  # a table cell may end inside it
                if match.re is AUTOLINK:
                    for opener in openers:
                        opener[2] = opener[2] and None
                index = match.end()
            elif match := INLINE_COMMENT.match(text, index):
                if not blank(index, match.end()):
                    break
                index = match.end()
            elif re.match(r"<[A-Za-z/!?]", text[index:index + 2]):
                break
            else:
                index += 1
        elif char == "[" or (char == "!" and text.startswith("[", index + 1)):
            position = index + (char == "!")
            unknown = follows_label or text.startswith("^", position + 1)
            openers.append([position, char == "!", None if unknown else True])
            index = position + 1
        elif char == "]" and openers:
            position, image, active = openers.pop()
            target = LINK_TARGET.match(text, index + 1)
            if target and active and "\n" not in text[position:index] \
                    and "|" not in text[position:target.end()]:
                blank(index + 1, target.end())
                if not image:
                    for opener in openers:
                        opener[2] = opener[2] if opener[1] else False
                index = target.end()
                continue
            if target and active is not False:
                break  # a link or not: cannot tell
            if active is not False:  # it may have made a reference link
                for opener in openers:
                    opener[2] = opener[2] and None
                label = text.startswith("[", index + 1)
            index += 1
        else:
            index += 1
    return "".join(out)


def _inline(lines: list[str], raw: list[bool], apart: list[bool]) -> list[str]:
    """Run :func:`_scan` over each run of lines between blank lines and blocks that
    are certainly apart (fences, HTML blocks, definitions), stopping at raw HTML."""
    out = list(lines)
    index = 0
    while index < len(lines):
        if apart[index]:
            index += 1
            continue
        end = index
        while end < len(lines) and not apart[end]:
            end += 1
        stop = next((line for line in range(index, end) if raw[line]), end)
        limit = sum(len(lines[line]) + 1 for line in range(index, stop))
        out[index:end] = _scan("\n".join(lines[index:end]), limit).split("\n")
        index = end
    return out


def _list_numbers(lines: list[str], source: list[str], skip: list[bool],
                  breaks: list[bool]) -> list[str]:
    """Blank the numbers of ordered-list items: an item after a blank line or a block,
    a first item numbered 1, or the next item of a list whose last item is known. A
    number opening a line of running text stays."""
    out, last = list(lines), None  # last: (prefix, delimiter, content column, quoted)
    for index, line in enumerate(source):
        if _blank_line(line) or skip[index]:
            last = None
            continue
        match = ORDERED_ITEM.match(line)
        if match:
            prefix, indent, number, delimiter, space = match.groups()
            text = line[match.end():].strip(" \t")
            # without a blank line before it, an item needs marks that certainly make one
            plain = QUOTES.fullmatch(prefix) is not None
            first = int(number) == 1 and text and plain
            sibling = last is not None and last[:2] == (prefix, delimiter) and plain
            if breaks[index] or first or sibling:
                out[index] = prefix + " " * len(number) + out[index][len(prefix) + len(number):]
                width = len(space) if 0 < len(space) <= 4 and text else 1
                column = len(prefix) + len(number) + 1 + width
                tabs = "\t" in line[:match.end()]
                last = None if tabs else (prefix, delimiter, column, ">" in prefix)
                continue
        if not (last and not last[3] and _column(line) >= last[2]):
            last = None
    return out


def literal_text(template: str, allowed_phrases: Iterable[str] = ()) -> str:
    """The template's own words, with placeholders, identifiers, allowed phrases and
    what is certainly code or hidden blanked (see the module notes). Lines keep their
    length."""
    text = PLACEHOLDER.sub(_fill, template)
    if not TEXT_ELEMENTS.search(text):
        source = text.split("\n")
        fenced, html, comments, certain = _top_level(source)
        lines = [" " * len(line) if fence else line
                 for line, fence in zip(source, fenced, strict=True)]
        raw = [a or b for a, b in zip(_raw_lines(source, fenced), html, strict=True)]
        lines = _hide_raw_comments(lines, source, raw, html, comments)
        # where a paragraph cannot be open: the start, after a blank line or a block
        breaks = [index == 0 or _blank_line(source[index - 1]) or fenced[index - 1]
                  or html[index - 1] for index in range(len(source))]
        defined = _definitions(source, raw, breaks)
        lines = [" " * len(line) if hide else line
                 for line, hide in zip(lines, defined, strict=True)]
        apart = [_blank_line(line) or a or b or c
                 for line, a, b, c in zip(source, fenced, defined, html, strict=True)]
        lines = _inline(lines, raw, apart)
        lines = _list_numbers(lines, source, raw, breaks)
        # footnotes defined at the margin, where the structure is certain (a definition
        # there interrupts a paragraph in GitHub's reading)
        notes = {match.group(1) for index, line in enumerate(source[:certain])
                 if not raw[index] and not fenced[index]
                 and (match := FOOTNOTE_DEFINITION.match(line))}

        def footnote(match: re.Match[str]) -> str:  # shown as an ordinal, if defined
            return _blank(match) if match.group(1) in notes else match.group(0)

        lines = [line if raw[index] else FOOTNOTE_MARKER.sub(footnote, line)
                 for index, line in enumerate(lines)]
        text = "\n".join(lines)
    text = text.replace(FILL, " ")
    for phrase in allowed_phrases:
        text = re.sub(re.escape(phrase), _blank, text)
    for identifier in IDENTIFIERS:
        text = re.sub(identifier, _blank, text)
    return text


def number_findings(name: str, template: str, allowed_phrases: Iterable[str] = ()) -> list[Finding]:
    """Digits in a template's own text, outside placeholders and identifiers."""
    findings = []
    for number, line in enumerate(literal_text(template, allowed_phrases).split("\n"), start=1):
        for match in NUMBER.finditer(line):
            findings.append(Finding(name, number, "number", match.group(0).rstrip(".,")))
    return findings


def normalize(text: str) -> str:
    return " ".join(text.split())


def phrase_spans(phrases: Iterable[str], text: str) -> list[tuple[int, int]]:
    """Where whole-word, case-insensitive ``phrases`` occur in ``text``."""
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return []
    pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(phrases, key=len,
                                                                         reverse=True)) + r")\b",
                         re.IGNORECASE)
    return [match.span() for match in pattern.finditer(text)]


def _sentence_start(text: str, start: int) -> bool:
    """Whether ``start`` begins a sentence: the start of a line (after list, quote or
    heading marks) or after a full stop, question mark, exclamation mark or placeholder."""
    line = text[text.rfind("\n", 0, start) + 1:start]
    if re.fullmatch(r"\s*(?:(?:[-*+>]|\d+[.)]|#+)\s+)*", line):
        return True
    before = text[:start].rstrip()
    return not before or before[-1] in ".!?}"


def directional_findings(name: str, template: str, words: Iterable[str],
                         allowed_phrases: Iterable[str] = (),
                         allowed_sentences: Iterable[str] = ()) -> list[Finding]:
    """Comparative words in a template's own text, outside the allowed sentences.

    An allowed sentence exempts only itself, whole: from the start of a sentence
    to its own full stop (it must end with ``.``, ``?`` or ``!``).
    """
    text = literal_text(template, allowed_phrases)
    for allowed in allowed_sentences:
        allowed = normalize(allowed)
        if not allowed or allowed[-1] not in ".!?":
            raise ValueError(f"allowed sentence {allowed!r} must end with its full stop")
        pattern = r"\s+".join(re.escape(word) for word in allowed.split())

        def blank_whole(match: re.Match[str], text: str = text) -> str:
            return _blank(match) if _sentence_start(text, match.start()) else match.group(0)

        text = re.sub(pattern, blank_whole, text)
    findings = []
    for number, line in enumerate(text.split("\n"), start=1):
        for start, end in phrase_spans(words, line):
            findings.append(Finding(name, number, "comparison",
                                    f"{line[start:end]!r} in the template's own text; place a "
                                    "{{ claim:<id> }} or list the sentence under "
                                    "allowed_sentences"))
    return findings


def missing_templates(config: dict[str, Any], root: Path = REPO,
                      templates: Path = TEMPLATES) -> list[str]:
    """Documents that need a template and have none, and exemptions that must go."""
    templated = {doc.output.as_posix() for doc in documents(templates)}
    exempt = set(config["not_yet_templated"])
    problems = []
    for pattern in config["documents"]:
        for path in sorted(root.glob(pattern)):
            name = path.relative_to(root).as_posix()
            if name not in templated and name not in exempt:
                problems.append(f"{name}: published document without a template")
    for name in sorted(exempt):
        if name not in EXEMPTION_BASELINE:
            problems.append(f"{name}: not_yet_templated may only shrink; write a template")
        elif name in templated:
            problems.append(f"{name}: has a template now; remove it from not_yet_templated")
        elif not (root / name).exists():
            problems.append(f"{name}: listed in not_yet_templated but does not exist")
    return problems


def lint(config: dict[str, Any], root: Path = REPO, templates: Path = TEMPLATES) -> list[str]:
    """Every lint problem in the repository's templates."""
    problems = missing_templates(config, root, templates)
    for doc in documents(templates):
        template = doc.template.read_text()
        problems += [str(finding) for finding in
                     number_findings(doc.name, template, config["allowed_phrases"])]
        if doc.name in config["directional"]:
            problems += [str(finding) for finding in directional_findings(
                doc.name, template, config["comparatives"], config["allowed_phrases"],
                config["allowed_sentences"])]
    return problems
