"""A relative link or image in a published document must reach a file in the
repository, and a link to a heading must reach that heading, as GitHub writes its
anchor. Every link-like destination outside fenced code is checked, so a construct the
check misreads (code, a title, quoted attribute text) adds a finding but never hides a
link; links with a scheme are left alone."""

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
    "```\n![fenced](missing.svg)\n```",
    "[escaped](docs/methods\\.md) [entity](docs&#47;methods.md#upper&#45;bound)",
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
    ("[docs](docs)", "'docs': no such file"),
    ("![figures](reports/figures)", "no such file"),
])
def test_a_link_to_nothing_fails_with_its_line(tmp_path: Path, text: str,
                                               message: str) -> None:
    root = _repo(tmp_path, {**BASE, "README.md": "Intro.\n\n" + text + "\n"})
    problems = links.broken(root)
    assert len(problems) == 1
    assert problems[0].startswith("README.md:3: ")
    assert message in problems[0]


@pytest.mark.parametrize("text, found", [
    # titles in any form, angle brackets, balanced parentheses
    ("[a](x.md 'title') [b](y.md (title)) [c](<z w.md>) [d](f(1).md) [e](g.md \"a \\\"t\\\"\")",
     ["x.md", "y.md", "z w.md", "f(1).md", "g.md"]),
    # an image inside a link, in a table, in a quote, and a link wrapped over two lines
    ("[![chart](img.svg)](page.md)", ["img.svg", "page.md"]),
    ("| a | b |\n|---|---|\n| ![c](c.svg) | [d](d.md) |", ["c.svg", "d.md"]),
    ("> See [the operating\n> review](quoted.md).", ["quoted.md"]),
    ("[text over\ntwo lines](two.md)", ["two.md"]),
    # a definition with its target below, a destination on the next line
    ("[ref]:\n  below.md", ["below.md"]),
    ("[a](\n  next.md)", ["next.md"]),
    # exact src and href attributes anywhere in a tag, whatever the quotes hold
    ('<a data-href="x.md" href="a.md">a</a>', ["a.md"]),
    ('<img alt="Gain > hurdle" src="b.svg"> <a title="Loss < cap" href="c.md">m</a>',
     ["b.svg", "c.md"]),
    # a fence of four closed by four but not by three, a fence closed only by a bare line
    ("````\n[a](in.md)\n```\n[b](in2.md)\n````\n[c](out.md)", ["out.md"]),
    ("```\n[a](in.md)\n``` text\n[b](in2.md)\n```\n[c](out.md)", ["out.md"]),
])
def test_every_link_like_destination_is_collected(text: str, found: list[str]) -> None:
    assert [target for _, target, _ in links.targets(text)] == found


@pytest.mark.parametrize("text, found", [
    # code, titles and quoted attribute text are read as links: a finding, never a miss
    ("`[a](in-code.md)` and [b](real.md)", ["in-code.md", "real.md"]),
    ("# `\n## [out](a.md) `", ["a.md"]),
    ('<img title="href=\'x.md\'" src="y.svg">', ["x.md", "y.svg"]),
])
def test_what_might_be_a_link_is_checked_as_one(text: str, found: list[str]) -> None:
    assert [target for _, target, _ in links.targets(text)] == found


@pytest.mark.parametrize("text, found", [
    # a destination read only in part is checked whole, up to the next space
    ("[x](README.md(no(such)))", ["README.md(no(such)))"]),
    # a reference definition anywhere: in a quote, after an escaped or wrapped label
    ("[r]\n\n> [r]: quoted.md", ["quoted.md"]),
    ("[a\\]b]: escaped.md and [wrapped\nlabel]: wrapped.md", ["escaped.md", "wrapped.md"]),
    # an attribute over two lines
    ('<a href=\n"split.md">x</a>', ["split.md"]),
    # a fence inside an HTML block is no fence; a list item's fence ends with the item
    ('<div>\n~~~\n<a href="in-html.md">x</a>\n</div>', ["in-html.md"]),
    ("- item\n  ~~~\n  code\n\n[x](after-list.md)", ["after-list.md"]),
])
def test_no_reading_of_a_block_hides_a_link(text: str, found: list[str]) -> None:
    assert [target for _, target, _ in links.targets(text)] == found


def test_attributes_take_entities_but_not_markdown_escapes(tmp_path: Path) -> None:
    root = _repo(tmp_path, {**BASE, "README.md": '<a href="docs/methods\\.md">m</a> '
                            '<a href="docs&#47;methods.md">m</a>\n'})
    assert links.broken(root) == ["README.md:1: link 'docs/methods\\\\.md': no such file"]


@pytest.mark.parametrize("text, missing", [
    ("<div>\n# Fake\n</div>", "fake"),
    ("<!--\n# Commented\n-->", "commented"),
    ('`<a id="ghost"></a>` and <!-- <a id="hidden"></a> -->', "ghost"),
    ('<!-- <a id="hidden"></a> -->', "hidden"),
    ('Text\n\n    <a id="indented"></a>', "indented"),
    ("- item\n  ```\n  code\n\n```\n# Not heading\n```", "not-heading"),
])
def test_no_anchor_is_read_where_github_makes_none(text: str, missing: str) -> None:
    assert missing not in links.anchors(text)


def test_a_known_non_link_is_exempted_by_its_document_and_line(tmp_path: Path,
                                                                monkeypatch) -> None:
    line = "Write `[label](target.md)` for a link."
    root = _repo(tmp_path, {**BASE, "README.md": line + "\n"})
    assert links.broken(root) == ["README.md:1: link 'target.md': no such file"]
    monkeypatch.setattr(links, "NOT_LINKS", frozenset({("README.md", line)}))
    assert links.broken(root) == []
    monkeypatch.setattr(links, "NOT_LINKS", frozenset({("cases/ring.md", line)}))
    assert links.broken(root) == ["README.md:1: link 'target.md': no such file"]


def test_the_repository_exempts_nothing() -> None:
    assert links.NOT_LINKS == frozenset()


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


def test_a_link_out_of_the_repository_is_caught_in_every_form(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo", {**BASE, "README.md":
                                     "[a](../out.md 'title') [![b](../out.svg)](README.md)\n"})
    (tmp_path / "out.md").write_text("outside\n")
    (tmp_path / "out.svg").write_text("<svg/>")
    assert links.broken(root) == [
        "README.md:1: link '../out.md' leaves the repository",
        "README.md:1: link '../out.svg' leaves the repository"]


def test_github_anchors_drop_punctuation_and_keep_hyphens() -> None:
    assert links.slug("Operating review: which review policy to run") == \
        "operating-review-which-review-policy-to-run"
    assert links.slug("A reviewer who always knows the truth (a diagnostic)") == \
        "a-reviewer-who-always-knows-the-truth-a-diagnostic"
    assert links.slug("The analyst's decisions by label") == "the-analysts-decisions-by-label"
    assert links.slug("Pay-in-4 and depth-3, §6.2") == "pay-in-4-and-depth-3-62"
    assert links.slug("[`code`](README.md)") == "code"
    assert links.slug("[foo](a(b)c.md) bar") == "foo-bar"
    assert links.slug("Syntax: `[label](README.md)`") == "syntax-labelreadmemd"
    assert links.slug("a ` foo ` b") == "a-foo-b"
    assert links.slug("``a`b``") == "ab"


def test_anchors_follow_githubs_rendered_text_and_repeat_numbers() -> None:
    text = ("# X\n\n# X\n\n# X-1\n\n## _Italics_ and snake_case\n\n"
            "## The `<foo>` tag\n\n## Caf&eacute; &amp; more\n\n"
            "```\n# not a heading\n```\n")
    assert links.anchors(text) == {"x", "x-1", "x-1-1", "italics-and-snake_case",
                                   "the-foo-tag", "café--more"}
