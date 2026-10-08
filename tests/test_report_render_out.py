"""Where documents are rendered decides how a document that cannot be rendered is
handled: the repository's documents fail the render, as published numbers must exist;
a development or small run's documents are written where the run has their results and
keys, and the rest are listed with their problems instead of stopping the run."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.results import Metric, StageResult, assemble_summary, write_summary
from report import render
from report.__main__ import main
from report.render import NOT_RENDERED, RenderError, Sources, render_all
from report.render import documents as real_documents


def _sources() -> Sources:
    summary = assemble_summary([StageResult(
        stage="evaluate", versions={}, inputs={},
        metrics={"evaluate.things": Metric(value=7, unit="count", population="things",
                                           window="test")})])
    return Sources(summary=summary)


def _templates(root: Path) -> Path:
    templates = root / "report" / "templates"
    (templates / "reports").mkdir(parents=True)
    (templates / "README.md").write_text("Things: {{ evaluate.things }}.\n")
    (templates / "reports" / "memo.md").write_text(
        "Gain: {{ evaluate.gain.hybrid | usd }}.\nFact: {{ fact:ring:alerts.ring.selected }}\n")
    return templates


def test_the_repository_render_still_fails_on_a_missing_key(tmp_path: Path) -> None:
    templates = _templates(tmp_path)
    with pytest.raises(RenderError, match="evaluate.gain.hybrid"):
        render_all(_sources(), tmp_path, templates, tmp_path)
    assert not (tmp_path / "README.md").exists()  # nothing is written
    assert not (tmp_path / NOT_RENDERED).exists()


def test_a_render_elsewhere_writes_what_it_can_and_lists_the_rest(
    tmp_path: Path, capsys
) -> None:
    root, out = tmp_path / "repo", tmp_path / "runs" / "ci" / "docs"
    templates = _templates(root)
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "memo.md").write_text("an earlier render\n")
    written = render_all(_sources(), out, templates, root)
    assert (out / "README.md").read_text().endswith("Things: 7.\n")
    assert not (out / "reports" / "memo.md").exists()  # no stale copy is left
    notice = (out / NOT_RENDERED).read_text()
    assert written == [out / "README.md", out / NOT_RENDERED]
    assert "reports/memo.md:" in notice
    assert "'evaluate.gain.hybrid' is not in the summary" in notice
    assert "no case facts file 'ring'" in notice
    assert "README" not in notice
    assert "1 of 2 documents not rendered" in capsys.readouterr().err
    # once every document renders, the list goes
    (templates / "reports" / "memo.md").write_text("Things again: {{ evaluate.things }}.\n")
    assert render_all(_sources(), out, templates, root) == [out / "README.md",
                                                             out / "reports" / "memo.md"]
    assert not (out / NOT_RENDERED).exists()


def test_strict_fails_elsewhere_too(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    templates = _templates(root)
    with pytest.raises(RenderError, match="evaluate.gain.hybrid"):
        render_all(_sources(), out, templates, root, strict=True)
    assert not out.exists()


def test_the_command_takes_strict(tmp_path: Path, monkeypatch) -> None:
    templates = _templates(tmp_path / "repo")
    monkeypatch.setattr(render, "documents", lambda _=None: real_documents(templates))
    summary = write_summary([StageResult(stage="evaluate", versions={}, inputs={},
                                         metrics={})], tmp_path / "results")
    out = tmp_path / "docs"
    assert main(["render", "--summary", str(summary), "--out", str(out)]) == 0
    assert (out / NOT_RENDERED).exists()
    with pytest.raises(RenderError):
        main(["render", "--summary", str(summary), "--out", str(tmp_path / "strict"),
              "--strict"])


def test_the_repository_stays_strict_whatever_root_is_given(tmp_path: Path,
                                                            monkeypatch) -> None:
    repo = tmp_path / "repo"
    templates = _templates(repo)
    monkeypatch.setattr(render, "REPO", repo)
    with pytest.raises(RenderError, match="evaluate.gain.hybrid"):
        render_all(_sources(), repo, templates, root=tmp_path / "elsewhere")
    assert not (repo / "README.md").exists() and not (repo / NOT_RENDERED).exists()


def test_nothing_is_written_or_removed_through_a_link_out_of_the_folder(tmp_path: Path) -> None:
    root, out, published = tmp_path / "repo", tmp_path / "docs", tmp_path / "published"
    templates = _templates(root)
    (published / "memo.md").parent.mkdir(parents=True)
    (published / "memo.md").write_text("a published document\n")
    out.mkdir()
    (out / "reports").symlink_to(published, target_is_directory=True)
    with pytest.raises(ValueError, match="outside .* through a link"):
        render_all(_sources(), out, templates, root)
    assert (published / "memo.md").read_text() == "a published document\n"
    assert not (out / "README.md").exists()


def test_another_spelling_of_the_repository_is_the_repository(tmp_path: Path,
                                                              monkeypatch) -> None:
    repo = tmp_path / "repo"
    templates = _templates(repo)
    monkeypatch.setattr(render, "REPO", repo)
    variant = tmp_path / "REPO"
    if not variant.exists():
        pytest.skip("a case-sensitive file system: no other spelling reaches the folder")
    assert render.is_repository(variant)
    with pytest.raises(RenderError, match="evaluate.gain.hybrid"):
        render_all(_sources(), variant, templates, root=tmp_path / "elsewhere")
    assert not (repo / "README.md").exists()


def test_a_hard_link_in_the_folder_is_replaced_not_written_through(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "docs"
    templates = _templates(root)
    (root / "README.md").write_text("the published README\n")
    out.mkdir()
    (out / "README.md").hardlink_to(root / "README.md")
    render_all(_sources(), out, templates, root)  # the memo is skipped, the README rendered
    assert (root / "README.md").read_text() == "the published README\n"
    assert (out / "README.md").read_text().endswith("Things: 7.\n")
    assert (out / NOT_RENDERED).exists()


@pytest.mark.parametrize("value", [
    "replay.outcomes where family=nowhere | reviews",  # no row
    "replay.outcomes where family=baseline | reviews",  # two rows
    "replay.outcomes where family=baseline seed=1 | revews",  # an unknown column
    "replay.outcomes where family=baseline sum held_share by family | held_share",  # refused
])
def test_the_repository_render_fails_on_a_cell_that_is_not_one_value(tmp_path: Path,
                                                                     value: str) -> None:
    templates = tmp_path / "report" / "templates"
    templates.mkdir(parents=True)
    (templates / "README.md").write_text("Reviews: {{ cell:" + value + " }}.\n")
    rows = [{"seed": seed, "family": "baseline", "policy": "hybrid", "evaluated": True,
             "reviews": 3 * seed, "held_share": 0.5} for seed in (1, 2)]
    sources = Sources(summary={"metrics": {}, "tables": {"replay.outcomes": rows}})
    with pytest.raises(RenderError, match=r"line 1: \{\{cell:replay\.outcomes "):
        render_all(sources, tmp_path, templates, tmp_path)
    assert not (tmp_path / "README.md").exists()
    # the same document with one value meant renders
    (templates / "README.md").write_text(
        "Reviews: {{ cell:replay.outcomes where seed=2 | reviews count }}.\n")
    render_all(sources, tmp_path, templates, tmp_path)
    assert (tmp_path / "README.md").read_text().endswith("Reviews: 6.\n")
