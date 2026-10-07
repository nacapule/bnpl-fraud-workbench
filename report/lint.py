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
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LINK_TARGET = re.compile(r"\]\([^)]*\)")
# A link reference definition: only its target is blanked; footnotes keep their text.
REFERENCE_TARGET = re.compile(r"^(\s*\[(?!\^)[^\]]+\]:)(\s*\S+.*)$", re.MULTILINE)
FOOTNOTE_MARKER = re.compile(r"\[\^[^\]\s]+\]")
LIST_MARKER = re.compile(r"^(\s*)\d+[.)](?=\s)", re.MULTILINE)
NUMBER = re.compile(r"\d[\d,.]*")
LIST_ITEM = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])(\s+)")
QUOTE = re.compile(r"^ {0,3}> ?")
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
    return {
        "documents": list(data.get("documents", [])),
        "not_yet_templated": list(data.get("not_yet_templated", [])),
        "allowed_phrases": _words(data.get("allowed_phrases"), "allowed_phrases"),
        "directional": list(data.get("directional", [])),
        "comparatives": _words(data.get("comparatives"), "comparatives"),
        "allowed_sentences": _words(data.get("allowed_sentences"), "allowed_sentences"),
    }


def _blank(match: re.Match[str]) -> str:
    """Replace a match with spaces, keeping newlines so line numbers hold."""
    return re.sub(r"[^\n]", " ", match.group(0))


def literal_text(template: str, allowed_phrases: Iterable[str] = ()) -> str:
    """The template's own words: placeholders, code, comments, links and identifiers blanked."""
    text = template
    text = _blank_fences(COMMENT.sub(_blank, text))
    for pattern in (PLACEHOLDER, INLINE_CODE, LINK_TARGET, FOOTNOTE_MARKER):
        text = pattern.sub(_blank, text)
    text = REFERENCE_TARGET.sub(lambda m: m.group(1) + " " * len(m.group(2)), text)
    text = LIST_MARKER.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), text)
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


def _indent(line: str) -> int:
    expanded = line.replace("\t", "    ")
    return len(expanded) - len(expanded.lstrip(" "))


def _unquote(line: str) -> tuple[int, str, int]:
    """A line without its block-quote marks: (depth, the rest, indent of the first mark)."""
    depth, quote_indent = 0, 0
    while marker := QUOTE.match(line):
        if depth == 0:
            quote_indent = _indent(marker.group(0))
        line = line[marker.end():]
        depth += 1
    return depth, line, quote_indent


def _fence(content: str) -> str | None:
    match = FENCE_OPEN.match(content)
    return None if match is None else match.group(1) or match.group(2)


def _closes(fence: str, stripped: str) -> bool:
    return stripped.startswith(fence) and set(stripped) == {fence[0]}


def _strip_quotes(line: str, most: int) -> tuple[int, str]:
    """``line`` without up to ``most`` block-quote marks: (marks removed, the rest)."""
    removed = 0
    while removed < most and (marker := QUOTE.match(line)):
        line = line[marker.end():]
        removed += 1
    return removed, line


def _blank_fences(text: str) -> str:
    """Fenced code blanked wherever it sits (in lists and quotes too), for the lints.

    An open fence's lines are read as its content first, inside exactly the
    quote marks it opened in; it ends at its closer (indented at most three
    columns past the content column of the list item it is in) or where its
    container ends (fewer quote marks, or a line indented less than that column).
    """
    out: list[str] = []
    fence: tuple[str, int, int] | None = None  # marker, quote depth, content column
    lists: list[int] = []  # content columns of the open list items
    blank = True
    for line in text.split("\n"):
        if fence is not None:
            marker, depth, column = fence
            quotes, rest = _strip_quotes(line, depth)
            ended = quotes < depth or (column and rest.strip() and _indent(rest) < column)
            if not ended:
                if _indent(rest) <= column + 3 and _closes(marker, rest.strip()):
                    fence, blank = None, True
                out.append(" " * len(line))
                continue
            fence = None
        depth, rest, _ = _unquote(line)
        if not rest.strip():
            blank = True
            out.append(line)
            continue
        indent = _indent(rest)
        item = LIST_ITEM.match(rest)
        if blank or item:  # a new block belongs to the items it is indented under
            while lists and lists[-1] > indent:
                lists.pop()
        if item:
            lists.append(len(item.group(0).replace("\t", "    ")))
        blank = False
        content = (rest[item.end():] if item else rest).strip()
        if opened := _fence(content):
            column = lists[-1] if lists and (item or indent >= lists[-1]) else 0
            fence = (opened, depth, column)
            out.append(" " * len(line))
        else:
            out.append(line)
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


def directional_findings(name: str, template: str, words: Iterable[str],
                         allowed_phrases: Iterable[str] = (),
                         allowed_sentences: Iterable[str] = ()) -> list[Finding]:
    """Comparative words in a template's own text, outside the allowed sentences."""
    text = literal_text(template, allowed_phrases)
    for allowed in allowed_sentences:
        words_of = [re.escape(word) for word in normalize(allowed).split()]
        if words_of:
            text = re.sub(r"\s+".join(words_of), _blank, text)
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
