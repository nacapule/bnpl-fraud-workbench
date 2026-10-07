"""Lint for numbers and directional sentences in the published documents.

Two checks, configured in ``report/lint.yaml``:

* **numbers**: a template's own text (everything outside placeholders, code
  and link targets) may hold no digits, except identifiers such as rule ids
  (R01), policy clauses (FP-2 §6.3), queue priorities (P0), SQL queries (Q12),
  case ids (CASE-01), numbered-list markers and the phrases the configuration
  allows. Every published number therefore comes from a rendered result key.
* **directional sentences**: in the documents named under ``directional``, a
  sentence with comparative wording (more, fewer, higher, lower, better,
  beats, ...) must contain the text of a claim in ``report/claims.yaml`` for
  that document, so every directional claim is tested; non-result comparisons
  are allowed by listing them.

Documents matching ``documents`` must have a template, except those listed
under ``not_yet_templated`` (the older hand-written documents until they are
rewritten as templates; the list only shrinks).
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
IDENTIFIERS = (
    r"\bR\d{2}\b",  # rule ids
    r"\bFP-\d+\b",  # policy version
    r"§\s?\d+(?:\.\d+)*(?:\([a-z]\))*",  # policy clauses
    r"\bP[0-3]\b",  # queue priorities
    r"\bQ\d{2}\b",  # SQL investigation queries
    r"\bCASE-\d{2}\b",  # case files
)
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LINK_TARGET = re.compile(r"\]\([^)]*\)")
REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
LIST_MARKER = re.compile(r"^(\s*)\d+[.)](?=\s)", re.MULTILINE)
NUMBER = re.compile(r"\d[\d,.]*")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\s*\n|\n(?=\s*(?:[-*+]|\d+[.)]|#)\s)")


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
    for pattern in (FENCE, COMMENT, PLACEHOLDER, INLINE_CODE, LINK_TARGET, REFERENCE):
        text = pattern.sub(_blank, text)
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


def sentences(text: str) -> list[tuple[int, str]]:
    """Prose sentences with their starting line; tables, code and comments are skipped."""
    text = FENCE.sub(_blank, COMMENT.sub(_blank, text))
    lines = [("" if line.lstrip().startswith("|") else line) for line in text.split("\n")]
    text = "\n".join(lines)
    out = []
    position = 0
    for piece in SENTENCE_END.split(text):
        start = text.find(piece, position)
        position = start + len(piece)
        sentence = " ".join(piece.split())
        if sentence:
            out.append((text.count("\n", 0, start) + 1, sentence))
    return out


def directional_findings(name: str, rendered: str, claim_texts: Iterable[str],
                         words: Iterable[str], allowed: Iterable[str] = ()) -> list[Finding]:
    """Sentences with comparative wording that no claim for this document covers."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(word) for word in words) + r")\b",
                         re.IGNORECASE)
    claims = [" ".join(text.split()) for text in claim_texts]
    allowed = {" ".join(text.split()) for text in allowed}
    findings = []
    for line, sentence in sentences(rendered):
        if not pattern.search(sentence) or sentence in allowed:
            continue
        if any(claim in sentence or sentence in claim for claim in claims):
            continue
        findings.append(Finding(name, line, "directional", sentence))
    return findings


def missing_templates(config: dict[str, Any], root: Path = REPO,
                      templates: Path = TEMPLATES) -> list[str]:
    """Documents that need a template and have none, and stale exemptions."""
    templated = {doc.output.as_posix() for doc in documents(templates)}
    exempt = set(config["not_yet_templated"])
    problems = []
    for pattern in config["documents"]:
        for path in sorted(root.glob(pattern)):
            name = path.relative_to(root).as_posix()
            if name not in templated and name not in exempt:
                problems.append(f"{name}: published document without a template")
    for name in sorted(exempt):
        if name in templated:
            problems.append(f"{name}: has a template now; remove it from not_yet_templated")
        elif not (root / name).exists():
            problems.append(f"{name}: listed in not_yet_templated but does not exist")
    return problems


def lint(config: dict[str, Any], claim_texts: dict[str, list[str]], root: Path = REPO,
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
                    doc.name, (root / doc.output).read_text(), claim_texts.get(doc.name, []),
                    config["directional_words"], config["allowed_sentences"],
                )
            ]
    return problems
