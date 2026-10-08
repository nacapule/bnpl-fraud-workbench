"""A relative link or image in a published document must reach a file in the
repository, and a link to a heading must reach that heading, as GitHub writes its
anchor; links inside code, and links with a scheme, are left alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from report import links


def _repo(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


BASE = {
    "reports/figures/chart.svg": "<svg/>",
    "reports/memo.md": "# Operating review: which policy to run\n\n## The queue\n\n"
                       "## The queue\n\n## A reviewer (a diagnostic)\n\n"
                       "## `code` and *emphasis*, too\n\n<a id=\"kept\"></a>\n",
    "docs/methods.md": "# Methods\n\n### Upper bound\n",
}


@pytest.mark.parametrize("text", [
    "![chart](reports/figures/chart.svg)",
    "[memo](reports/memo.md) and [methods](docs/methods.md#upper-bound)",
    "[queue](reports/memo.md#the-queue), [again](reports/memo.md#the-queue-1)",
    "[diagnostic](reports/memo.md#a-reviewer-a-diagnostic)",
    "[styled](reports/memo.md#code-and-emphasis-too), [kept](reports/memo.md#kept)",
    "[title](reports/memo.md \"a title\") and [angle](<reports/memo.md>)",
    "[top](#the-question)\n\n## The question",
    '<img src="reports/figures/chart.svg" alt="chart">',
    "[ref]: reports/figures/chart.svg",
    "[site](https://example.org/missing) and [mail](mailto:someone@example.org)",
    "`[code](missing.md)` and\n\n```\n![fenced](missing.svg)\n```",
])
def test_links_that_reach_their_targets_pass(tmp_path: Path, text: str) -> None:
    root = _repo(tmp_path, {**BASE, "README.md": text + "\n"})
    assert links.broken(root) == []


@pytest.mark.parametrize("text, message", [
    ("![chart](reports/figures/gone.svg)", "'reports/figures/gone.svg': no such file"),
    ("[memo](reports/missing.md)", "no such file"),
    ("[methods](docs/methods.md#lower-bound)",
     "no heading with anchor #lower-bound in docs/methods.md"),
    ("[queue](reports/memo.md#the-queue-2)", "no heading with anchor #the-queue-2"),
    ("[here](#nowhere)", "no heading with anchor #nowhere in README.md"),
    ('<img src="figures/chart.svg">', "no such file"),
    ("[ref]: reports/figures/gone.svg", "no such file"),
    ("[out](../elsewhere.md)", "leaves the repository"),
    ("[abs](/reports/memo.md)", "is absolute"),
])
def test_a_link_to_nothing_fails_with_its_line(tmp_path: Path, text: str,
                                               message: str) -> None:
    root = _repo(tmp_path, {**BASE, "README.md": "Intro.\n\n" + text + "\n"})
    problems = links.broken(root)
    assert len(problems) == 1
    assert problems[0].startswith("README.md:3: ")
    assert message in problems[0]


def test_every_published_document_is_read_relative_to_itself(tmp_path: Path) -> None:
    root = _repo(tmp_path, {**BASE,
                            "cases/ring.md": "[memo](../reports/memo.md#the-queue)\n"
                                             "![chart](../reports/figures/gone.svg)\n",
                            "reports/memo.md": BASE["reports/memo.md"]
                            + "\n![chart](figures/chart.svg)\n[methods](../docs/methods.md)\n",
                            "notes/private.md": "[ignored](missing.md)\n"})
    assert [path.relative_to(root).as_posix() for path in links.published(root)] == [
        "cases/ring.md", "docs/methods.md", "reports/memo.md"]
    assert links.broken(root) == [
        "cases/ring.md:2: link '../reports/figures/gone.svg': no such file"]


def test_github_anchors_drop_punctuation_and_keep_hyphens() -> None:
    assert links.slug("Operating review: which review policy to run") == \
        "operating-review-which-review-policy-to-run"
    assert links.slug("A reviewer who always knows the truth (a diagnostic)") == \
        "a-reviewer-who-always-knows-the-truth-a-diagnostic"
    assert links.slug("The analyst's decisions by label") == "the-analysts-decisions-by-label"
    assert links.slug("Pay-in-4 and depth-3, §6.2") == "pay-in-4-and-depth-3-62"


@pytest.mark.parametrize("text, found", [
    # titles in either quote or in parentheses, angle brackets, balanced parentheses
    ("[a](x.md 'title') [b](y.md (title)) [c](<z w.md>) [d](f(1).md)",
     ["x.md", "y.md", "z w.md", "f(1).md"]),
    # an image inside a link's text, and in a table cell
    ("[![chart](img.svg)](page.md)", ["img.svg", "page.md"]),
    ("| a | b |\n|---|---|\n| ![c](c.svg) | [d](d.md) |", ["c.svg", "d.md"]),
    # a link across two lines of one paragraph, and a definition with its target below
    ("[text over\ntwo lines](two.md)", ["two.md"]),
    ("[ref]:\n  below.md", ["below.md"]),
    # what is not a link: a destination with a space, an unclosed title, no bracket
    ("[a](x y.md) [b](x.md 'open) (c](z.md)", []),
    # code: a span across lines, a span of two backticks around one, a fence of four
    # closed by four but not by three, and a fence closed only by a bare line
    ("`[a](in.md)\n[b](still-in.md)` [c](out.md)", ["out.md"]),
    ("``a ` [b](in.md) `` [c](out.md)", ["out.md"]),
    ("````\n[a](in.md)\n```\n[b](in2.md)\n````\n[c](out.md)", ["out.md"]),
    ("```\n[a](in.md)\n``` text\n[b](in2.md)\n```\n[c](out.md)", ["out.md"]),
    # an unmatched backtick is text, so it hides nothing
    ("a ` b [c](seen.md)", ["seen.md"]),
])
def test_links_are_read_as_github_writes_them(text: str, found: list[str]) -> None:
    assert [target for _, target in links.targets(text)] == found


def test_a_directory_is_not_a_file(tmp_path: Path) -> None:
    root = _repo(tmp_path, {**BASE, "README.md": "[docs](docs) ![figures](reports/figures)\n"})
    assert links.broken(root) == ["README.md:1: link 'docs': no such file",
                                  "README.md:1: link 'reports/figures': no such file"]


def test_a_link_out_of_the_repository_is_caught_in_every_form(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo", {**BASE, "README.md":
                                     "[a](../out.md 'title') [![b](../out.svg)](README.md)\n"})
    (tmp_path / "out.md").write_text("outside\n")
    (tmp_path / "out.svg").write_text("<svg/>")
    assert links.broken(root) == [
        "README.md:1: link '../out.md' leaves the repository",
        "README.md:1: link '../out.svg' leaves the repository"]


def test_anchors_follow_githubs_rendered_text_and_repeat_numbers() -> None:
    text = ("# X\n\n# X\n\n# X-1\n\n## _Italics_ and snake_case\n\n"
            "## The `<foo>` tag\n\n## Caf&eacute; &amp; more\n\n"
            "```\n# not a heading\n```\n")
    assert links.anchors(text) == {"x", "x-1", "x-1-1", "italics-and-snake_case",
                                   "the-foo-tag", "café--more"}


@pytest.mark.parametrize("text, found", [
    # headings, list items and table rows are blocks of their own for code spans
    ("# `\n## [out](a.md) `", ["a.md"]),
    ("- `a\n- [b](b.md) `", ["b.md"]),
    # an escaped backtick or bracket is text; an escaped backslash is not an escape
    ("\\`[a](a.md)` [b](b.md)", ["a.md", "b.md"]),
    ("\\\\[x](x.md)", ["x.md"]),
    # a title with escaped delimiters
    ('[a](a.md "a \\"title\\"") [b](b.md \'it\\\'s\') [c](c.md (one \\) two))',
     ["a.md", "b.md", "c.md"]),
    # the exact src and href attributes of a tag, whatever else it holds
    ('<a data-href="x.md" href="a.md">a</a>', ["a.md"]),
    ("<img title=\"href='x.md'\" src='b.svg'>", ["b.svg"]),
])
def test_blocks_escapes_titles_and_attributes(text: str, found: list[str]) -> None:
    assert [target for _, target in links.targets(text)] == found


def test_a_destination_is_resolved_after_its_escapes_and_entities(tmp_path: Path) -> None:
    root = _repo(tmp_path, {**BASE, "README.md": "[a](docs/methods\\.md) [b](docs&#47;methods.md)"
                                                 " [c](docs/methods.md#upper&#45;bound)\n"})
    assert links.broken(root) == []


@pytest.mark.parametrize("heading, anchor", [
    ("[`code`](README.md)", "code"),
    ("[foo](a(b)c.md) bar", "foo-bar"),
    ("[foo][ref] and ![logo](logo.svg) more", "foo-and--more"),
])
def test_a_linked_heading_takes_its_label(heading: str, anchor: str) -> None:
    assert links.slug(heading) == anchor
