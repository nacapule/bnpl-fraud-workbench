"""Lint for numbers and comparisons typed into the published documents' templates.

Two checks on a template's own text (everything outside placeholders, code,
comments and link targets), configured in ``report/lint.yaml``:

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
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
# A link reference definition: only its target is blanked; footnotes keep their text.
REFERENCE_TARGET = re.compile(r"^(\s*\[(?!\^)[^\]]+\]:)(\s*\S+.*)$", re.MULTILINE)
FOOTNOTE_MARKER = re.compile(r"\[\^[^\]\s]+\]")
NUMBER = re.compile(r"\d[\d,.]*")
LIST_MARK = re.compile(r"[-+*]|(\d{1,9})[.)]")
THEMATIC_BREAK = re.compile(r" {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
SETEXT_UNDERLINE = re.compile(r" {0,3}(?:=+|-+)[ \t]*$")
COMMENT_BLOCK = re.compile(r" {0,3}<!--")
HEADING = re.compile(r" {0,3}#{1,6}(?:[ \t]|$)")
# A fence opens with three or more backticks (whose info string holds no
# backtick: ```make final``` is inline code) or three or more tildes.
FENCE_OPEN = re.compile(r"^(?:(`{3,})(?![^\n]*`)|(~{3,}))")


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


def literal_text(template: str, allowed_phrases: Iterable[str] = ()) -> str:
    """The template's own words: placeholders, code, comments, links and identifiers blanked."""
    text = template
    text = _visible_blocks(text)
    # A placeholder keeps its closing brace, so a sentence after it still starts one.
    text = PLACEHOLDER.sub(lambda match: _blank(match)[:-1] + "}", text)
    for pattern in (INLINE_CODE, LINK_TARGET, FOOTNOTE_MARKER):
        text = pattern.sub(_blank, text)
    text = REFERENCE_TARGET.sub(lambda m: m.group(1) + " " * len(m.group(2)), text)
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


def _spaces(text: str) -> int:
    return len(text) - len(text.lstrip(" "))


def _fence(content: str) -> str | None:
    match = FENCE_OPEN.match(content)
    return None if match is None else match.group(1) or match.group(2)


def _closes(fence: str, stripped: str) -> bool:
    return stripped.startswith(fence) and set(stripped) == {fence[0]}


def _new_container(rest: str, paragraph: bool) -> tuple[str, int] | None:
    """A block quote or list item opening at the start of ``rest``: (kind, the
    columns its mark takes, up to where its content starts), or None."""
    indent = _spaces(rest)
    if indent > 3:
        return None
    if rest[indent:indent + 1] == ">":
        return "quote", indent + (2 if rest[indent + 1:indent + 2] == " " else 1)
    mark = LIST_MARK.match(rest, indent)
    if mark is None or THEMATIC_BREAK.match(rest):
        return None
    after = rest[mark.end():]
    if after and after[0] != " ":
        return None
    empty = not after.strip()
    if paragraph and (empty or (mark.group(1) is not None and mark.group(1) != "1")):
        return None  # cannot interrupt a paragraph
    gap = _spaces(after)
    return "item", mark.end() + (1 if empty or gap > 4 else gap)


def _mask_inline(text: str, in_comment: bool) -> tuple[str, bool]:
    """``text`` with its inline HTML comments blanked (code spans are not comments);
    returns whether a comment is still open at the end of the line."""
    out = list(text)
    i = 0
    while i < len(text):
        if in_comment:
            end = text.find("-->", i)
            stop = len(text) if end < 0 else end + 3
            out[i:stop] = " " * (stop - i)
            i, in_comment = stop, end < 0
        elif text[i] == "`":
            run = len(text[i:]) - len(text[i:].lstrip("`"))
            closer = re.compile(rf"(?<!`){'`' * run}(?!`)").search(text, i + run)
            i = closer.end() if closer else i + run
        elif text.startswith("<!--", i):
            in_comment = True
        else:
            i += 1
    return "".join(out), in_comment


def _visible_blocks(text: str) -> str:
    """The text with container marks, comments, fenced code and indented code blanked.

    Block quotes and list items are matched line by line as CommonMark does: each
    open container in turn (a quote by its ``>``, a list item by its content
    indentation), then new ones; a line that does not match them all continues a
    paragraph lazily or closes them. Fences open and close at most three columns
    into their container's content and end with their container; lines four
    columns in that do not continue a paragraph are indented code. A comment
    starting a line is a block to its ``-->``; one inside a paragraph is blanked
    without ending the paragraph, and one never closed in its paragraph is text.
    Lines keep their length, so positions and line numbers hold.
    """
    out: list[str] = []
    containers: list[tuple[str, int]] = []  # ("quote", 0) or ("item", content columns)
    fence: tuple[str, int] | None = None  # marker, how many containers it sits in
    comment_block: int | None = None  # how many containers an open comment block sits in
    paragraph = False
    inline: list[tuple[int, str]] = []  # lines of an inline comment not yet closed

    def end_paragraph() -> None:
        nonlocal paragraph
        for index, original in inline:  # an unclosed comment was text after all
            out[index] = original
        inline.clear()
        paragraph = False

    def paragraph_line(prefix: int, rest: str) -> None:
        nonlocal paragraph
        masked, still_open = _mask_inline(rest, bool(inline))
        if still_open:
            inline.append((len(out), " " * prefix + rest))
        else:
            inline.clear()
        out.append(" " * prefix + masked)
        paragraph = True

    for raw in text.expandtabs(4).split("\n"):
        pos, matched = 0, 0
        for kind, width in containers:
            rest = raw[pos:]
            if kind == "quote":
                if _spaces(rest) > 3 or rest[_spaces(rest):_spaces(rest) + 1] != ">":
                    break
                pos += _spaces(rest) + 1 + (1 if rest[_spaces(rest) + 1:_spaces(rest) + 2] == " "
                                            else 0)
            elif rest.strip():
                if _spaces(rest) < width:
                    break
                pos += width
            matched += 1
        rest = raw[pos:]
        if matched < len(containers):
            starts_block = (_new_container(rest, False) is not None
                            or (_spaces(rest) <= 3 and _fence(rest.strip()))
                            or THEMATIC_BREAK.match(rest) or HEADING.match(rest)
                            or COMMENT_BLOCK.match(rest))
            if paragraph and fence is None and comment_block is None and rest.strip() \
                    and not starts_block:
                paragraph_line(pos, rest)  # a paragraph continued without its marks
                continue
            containers = containers[:matched]
            end_paragraph()
            if fence is not None and fence[1] > matched:
                fence = None
            if comment_block is not None and comment_block > matched:
                comment_block = None
        if fence is not None:
            if _spaces(rest) <= 3 and _closes(fence[0], rest.strip()):
                fence = None
            out.append(" " * len(raw))
            continue
        if comment_block is not None:
            if "-->" in rest:
                comment_block = None
            out.append(" " * len(raw))
            continue
        while (opened := _new_container(rest, paragraph)) is not None:
            kind, width = opened
            containers.append((kind, 0 if kind == "quote" else width))
            pos += min(width, len(rest))
            rest = raw[pos:]
            end_paragraph()
        indent = _spaces(rest)
        if not rest.strip() or (indent >= 4 and not paragraph):  # blank, or indented code
            end_paragraph()
            out.append(" " * len(raw))
            continue
        if paragraph and SETEXT_UNDERLINE.match(rest):  # the paragraph was a heading
            end_paragraph()
            out.append(" " * len(raw))
            continue
        if indent <= 3 and (marker := _fence(rest.strip())):
            end_paragraph()
            fence = (marker, len(containers))
            out.append(" " * len(raw))
            continue
        if COMMENT_BLOCK.match(rest):
            end_paragraph()
            if "-->" not in rest[rest.index("<!--") + 4:]:
                comment_block = len(containers)
            out.append(" " * len(raw))
            continue
        if THEMATIC_BREAK.match(rest) or HEADING.match(rest):
            end_paragraph()
            masked, unclosed = _mask_inline(rest, False)  # a heading is one line
            out.append(" " * pos + (rest if unclosed else masked))
            continue
        paragraph_line(pos, rest)
    end_paragraph()
    return "\n".join(out)


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
