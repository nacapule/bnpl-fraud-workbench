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
  better, beats, ...) must be exactly the sentence of a claim in
  ``report/claims.yaml`` with at least one check per comparative word, so
  every comparison is tested; comparisons that state no result are allowed
  by listing the sentence.

Documents matching ``documents`` must have a template, except those listed
under ``not_yet_templated``: the hand-written documents that predate the
templates. That list may only shrink: it must stay within
:data:`EXEMPTION_BASELINE`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
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
IDENTIFIERS = (
    r"\bR\d{2}\b(?![%.,]\d)",  # rule ids
    r"\bFP-\d+\b(?![%.,]\d)",  # policy version
    r"§ ?\d+(?:\.\d+)*(?:\([a-z]\))*(?![\d%]|[.,]\d)",  # policy clauses
    r"\bP[0-3]\b(?![%.,]\d)",  # queue priorities
    r"\bQ\d{2}\b(?![%.,]\d)",  # SQL investigation queries
    r"\bCASE-\d{2}\b(?![%.,]\d)",  # case files
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
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\s*\n|\n(?=\s*(?:[-*+>]|\d+[.)]|#)\s)")
LEAD = re.compile(r"^(?:#+|[-*+>]|\d+[.)])\s+")


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


def visible_text(text: str) -> str:
    """Rendered Markdown without comments, code blocks and table rows (lines kept)."""
    text = FENCE.sub(_blank, COMMENT.sub(_blank, text))
    return "\n".join("" if line.lstrip().startswith("|") else line for line in text.split("\n"))


def normalize(text: str) -> str:
    return " ".join(text.split())


def sentences(text: str) -> list[tuple[int, str]]:
    """Visible prose sentences with their starting line, list and heading marks removed."""
    text = visible_text(text)
    out = []
    position = 0
    for piece in SENTENCE_END.split(text):
        start = text.find(piece, position)
        position = start + len(piece)
        sentence = LEAD.sub("", normalize(piece))
        if sentence:
            out.append((text.count("\n", 0, start) + 1, sentence))
    return out


def directional_pattern(words: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(word) for word in words) + r")\b",
                      re.IGNORECASE)


def directional_findings(name: str, rendered: str, claimed: Mapping[str, int],
                         words: Iterable[str], allowed: Iterable[str] = ()) -> list[Finding]:
    """Comparative sentences that are not a claim's sentence with enough checks.

    ``claimed`` maps each claim sentence of this document to its number of checks.
    """
    pattern = directional_pattern(words)
    claims = {normalize(sentence): checks for sentence, checks in claimed.items()}
    allowed = {normalize(text) for text in allowed}
    findings = []
    for line, sentence in sentences(rendered):
        comparisons = len(pattern.findall(sentence))
        if not comparisons or sentence in allowed:
            continue
        checks = claims.get(sentence, 0)
        if checks >= comparisons:
            continue
        reason = "" if not checks else f" ({comparisons} comparative words, {checks} checks)"
        findings.append(Finding(name, line, "directional", sentence + reason))
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


def lint(config: dict[str, Any], claimed: Mapping[str, Mapping[str, int]], root: Path = REPO,
         templates: Path = TEMPLATES) -> list[str]:
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
