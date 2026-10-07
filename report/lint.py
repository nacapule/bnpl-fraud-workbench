"""Lint for numbers and directional sentences in the published documents.

Two checks, configured in ``report/lint.yaml``:

* **numbers**: a template's own text (everything outside placeholders, code,
  comments and link targets) may hold no digits, except identifiers such as
  rule ids (R01), policy clauses (FP-2 §6.3), queue priorities (P0), SQL
  queries (Q12), case ids (CASE-01), numbered-list and footnote markers, and the
  phrases the configuration allows. Every published number therefore comes
  from a rendered result key or setting.
* **directional sentences**: in the documents named under ``directional``, a
  visible sentence with comparative wording (more, fewer, higher, lower,
  better, beats, ...) must be the sentence of a claim in ``report/claims.yaml``
  and every comparative word must fall inside the clause of one of its checks,
  so every comparison is tested; comparisons that state no result are allowed
  by listing the sentence.

Documents matching ``documents`` must have a template, except those listed
under ``not_yet_templated``: the hand-written documents that predate the
templates. That list may only shrink: it must stay within
:data:`EXEMPTION_BASELINE`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
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
END = r"(?![\d%]|[.,]\d)"  # an identifier is not followed by a digit, a percent or a decimal
# Each identifier is matched whole (an atomic group) before the boundary is
# checked, so a qualifier such as (a) cannot be dropped to exempt a prefix.
IDENTIFIERS = (
    rf"(?>\bR\d{{2}}\b){END}",  # rule ids
    rf"(?>\bFP-\d+\b){END}",  # policy version
    rf"(?>§ ?\d+(?:\.\d+)*(?:\([a-z]\))*){END}",  # policy clauses
    rf"(?>\bP[0-3]\b){END}",  # queue priorities
    rf"(?>\bQ\d{{2}}\b){END}",  # SQL investigation queries
    rf"(?>\bCASE-\d{{2}}\b){END}",  # case files
)
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
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
FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})")
BREAK = re.compile(r"^ {0,3}(?:=+|-{2,}|\*{3,}|_{3,})\s*$")  # heading underline or rule
INLINE = (  # Markdown that is not visible text, and what is shown instead
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # image: its alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # link: its text
    (re.compile(r"\[([^\]]+)\]\[[^\]]*\]"), r"\1"),  # reference link: its text
    (re.compile(r"\[\^[^\]\s]+\]"), ""),  # footnote marker
    (re.compile(r"`([^`]*)`"), r"\1"),  # inline code: its content
    (re.compile(r"</?[A-Za-z][^>\n]*>"), ""),  # inline HTML
    (re.compile(r"(?<!\w)[*_]+|[*_]+(?!\w)"), ""),  # emphasis delimiters
)
BLOCK_START = re.compile(r"^\s*(?:#+|[-*+>]|\d+[.)])\s")
LEAD = re.compile(r"^(?:#+|[-*+>]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.")
HIDDEN_DOT = "․"


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str  # "number" or "directional"
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.text}"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    return {
        "documents": list(data.get("documents", [])),
        "not_yet_templated": list(data.get("not_yet_templated", [])),
        "allowed_phrases": list(data.get("allowed_phrases", [])),
        "directional": list(data.get("directional", [])),
        "directional_words": list(data.get("directional_words", [])),
        "allowed_sentences": list(data.get("allowed_sentences", [])),
        "names": {
            str(group): {str(name): list(phrases) for name, phrases in (entries or {}).items()}
            for group, entries in (data.get("names") or {}).items()
        },
        "references": list(data.get("references", [])),
    }


def _blank(match: re.Match[str]) -> str:
    """Replace a match with spaces, keeping newlines so line numbers hold."""
    return re.sub(r"[^\n]", " ", match.group(0))


def literal_text(template: str, allowed_phrases: Iterable[str] = ()) -> str:
    """The template's own words: placeholders, code, comments, links and identifiers blanked."""
    text = template
    for pattern in (FENCE, COMMENT, PLACEHOLDER, INLINE_CODE, LINK_TARGET, FOOTNOTE_MARKER):
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


def visible_lines(text: str) -> list[tuple[int, str]]:
    """Rendered Markdown's prose lines as (block-quote depth, text).

    Comments, fenced code, indented code (four columns past the enclosing list
    item's content, inside or outside block quotes), table rows, rules and
    heading underlines become empty lines, so line numbers hold. Block-quote
    marks are removed, and a quoted paragraph continued without its mark keeps
    its depth, so a quoted paragraph reads as one.
    """
    text = COMMENT.sub(_blank, text)
    out: list[tuple[int, str]] = []
    # Per block-quote depth: after a blank line or a block's end; inside indented
    # code; the open fence; the content columns of the open list items.
    state: dict[int, dict[str, Any]] = {}
    previous_depth, previous_prose = 0, False
    for line in text.split("\n"):
        depth = 0
        while QUOTE.match(line):
            line = QUOTE.sub("", line, count=1)
            depth += 1
        stripped = line.strip()
        if (depth < previous_depth and previous_prose and stripped
                and not BLOCK_START.match(line) and not FENCE_OPEN.match(stripped)
                and not BREAK.match(line) and not stripped.startswith("|")):
            out.append((previous_depth, line))  # a lazy continuation of the quote
            continue
        for deeper in [key for key in state if key > depth]:
            del state[deeper]
        context = state.setdefault(
            depth, {"blank": True, "code": False, "fence": None, "lists": []})
        previous_depth, previous_prose = depth, False
        if context["fence"]:
            fence = context["fence"]
            if stripped.startswith(fence) and set(stripped) == {fence[0]}:
                context.update(fence=None, blank=True)
            out.append((depth, ""))
            continue
        if not stripped:
            context["blank"] = True
            out.append((depth, ""))
            continue
        indent = _indent(line)
        item = LIST_ITEM.match(line)
        lists = context["lists"]
        if context["blank"] or item:  # a new block belongs to the items it is indented under
            while lists and lists[-1] > indent:
                lists.pop()
        threshold = (lists[-1] if lists else 0) + 4
        if indent >= threshold and (context["code"] or context["blank"]):
            context.update(code=True, blank=False)
            out.append((depth, ""))
            continue
        context["code"] = False
        if FENCE_OPEN.match(stripped):
            context.update(fence=FENCE_OPEN.match(stripped).group(1), blank=False)
            out.append((depth, ""))
            continue
        if BREAK.match(line) or stripped.startswith("|"):
            context["blank"] = True
            out.append((depth, ""))
            continue
        if item:
            lists.append(len(item.group(0).replace("\t", "    ")))
        # A heading ends its block: an indented line after it is code.
        context["blank"] = stripped.startswith("#")
        previous_prose = not context["blank"]
        out.append((depth, line))
    return out


def inline_text(line: str) -> str:
    """A line's visible text: link and image text kept, markup removed."""
    for pattern, replacement in INLINE:
        line = pattern.sub(replacement, line)
    return line


def normalize(text: str) -> str:
    return " ".join(text.split())


def _protect(text: str) -> str:
    for abbreviation in ABBREVIATIONS:
        text = re.sub(re.escape(abbreviation), abbreviation.replace(".", HIDDEN_DOT), text,
                      flags=re.IGNORECASE)
    return text


def sentences(text: str) -> list[tuple[int, str]]:
    """Visible prose sentences with their starting line.

    Paragraphs, headings (each alone), list items and block quotes (lines of
    one quoted paragraph joined) are separate blocks; within a block, sentences
    split at ``.``, ``!`` or ``?`` followed by space, except after common
    abbreviations. Markup is removed first (:func:`inline_text`), so emphasis
    and links read as their text.
    """
    blocks: list[tuple[int, list[str]]] = []
    previous_depth = 0
    for number, (depth, line) in enumerate(visible_lines(text), start=1):
        marker = BLOCK_START.match(line)  # a list mark is not emphasis
        line = line[:marker.end()] + inline_text(line[marker.end():]) if marker \
            else inline_text(line)
        if not line.strip():
            blocks.append((number + 1, []))
        elif (BLOCK_START.match(line) or not blocks or not blocks[-1][1]
              or depth != previous_depth):
            blocks.append((number, [line]))
            if line.lstrip().startswith("#"):
                blocks.append((number + 1, []))  # a heading stands alone
        else:
            blocks[-1][1].append(line)
        previous_depth = depth
    out = []
    for start, lines in blocks:
        if not lines:
            continue
        block = _protect("\n".join(lines))
        marker = BLOCK_START.match(block)  # a list or heading mark is not a sentence
        if marker:
            block = " " * marker.end() + block[marker.end():]
        position = 0
        for piece in SENTENCE_END.split(block):
            offset = block.find(piece, position)
            position = offset + len(piece)
            sentence = LEAD.sub("", normalize(piece)).replace(HIDDEN_DOT, ".")
            if sentence:
                out.append((start + block.count("\n", 0, offset), sentence))
    return out


def phrase_spans(phrases: Iterable[str], text: str) -> list[tuple[int, int]]:
    """Where whole-word, case-insensitive ``phrases`` occur in ``text``."""
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return []
    pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(phrases, key=len,
                                                                         reverse=True)) + r")\b",
                         re.IGNORECASE)
    return [match.span() for match in pattern.finditer(text)]


def clause_spans(sentence: str, clauses: Sequence[str]) -> list[tuple[int, int]]:
    spans = []
    for clause in clauses:
        start = sentence.find(normalize(clause))
        if start >= 0:
            spans.append((start, start + len(normalize(clause))))
    return spans


def inside(span: tuple[int, int], spans: Iterable[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def directional_findings(name: str, rendered: str, claimed: Mapping[str, Sequence[str]],
                         words: Iterable[str], allowed: Iterable[str] = ()) -> list[Finding]:
    """Comparative sentences that are not claims, or whose comparisons no check covers.

    ``claimed`` maps each claim sentence of this document to its checks' clauses.
    """
    words = list(words)
    claims = {normalize(sentence): clauses for sentence, clauses in claimed.items()}
    allowed = {normalize(text) for text in allowed}
    findings = []
    for line, sentence in sentences(rendered):
        comparisons = phrase_spans(words, sentence)
        if not comparisons or sentence in allowed:
            continue
        if sentence not in claims:
            findings.append(Finding(name, line, "directional", sentence))
            continue
        covered = clause_spans(sentence, claims[sentence])
        loose = [sentence[a:b] for a, b in comparisons if not inside((a, b), covered)]
        if loose:
            findings.append(Finding(name, line, "directional",
                                    f"{sentence} (no check covers: {', '.join(loose)})"))
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


def lint(config: dict[str, Any], claimed: Mapping[str, Mapping[str, Sequence[str]]],
         root: Path = REPO, templates: Path = TEMPLATES) -> list[str]:
    """Every lint problem in the repository's templates and rendered documents."""
    problems = missing_templates(config, root, templates)
    for doc in documents(templates):
        problems += [
            str(finding)
            for finding in number_findings(doc.name, doc.template.read_text(),
                                           config["allowed_phrases"])
        ]
        if doc.name in config["directional"] and (root / doc.output).exists():
            problems += [
                str(finding)
                for finding in directional_findings(
                    doc.name, (root / doc.output).read_text(), claimed.get(doc.name, {}),
                    config["directional_words"], config["allowed_sentences"],
                )
            ]
    return problems
