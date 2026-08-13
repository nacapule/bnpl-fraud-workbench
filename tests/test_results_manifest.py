from pathlib import Path

from analysis.results_manifest import END, START, render_results, rewrite_readme


def test_results_manifest_is_idempotent_and_matches_readme(tmp_path: Path) -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text()
    committed = text.split(START, 1)[1].split(END, 1)[0].strip()
    assert committed == render_results()

    copy = tmp_path / "README.md"
    copy.write_text(text)
    rewrite_readme(copy)
    first = copy.read_bytes()
    rewrite_readme(copy)
    assert copy.read_bytes() == first
