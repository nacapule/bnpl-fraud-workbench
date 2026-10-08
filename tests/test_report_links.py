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
