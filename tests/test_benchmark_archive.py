"""The August 2026 memo benchmark is archived byte-for-byte under original/.

The manifest records every archived file's SHA-256; corrections live beside the
archive (corrected/, HISTORY.md) and never inside it. Where the repository's
history is available, each archived file is also checked against the blob it
came from at the source commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "llm" / "eval" / "benchmarks" / "2026-08-dev"
ORIGINAL = BENCH / "original"
MANIFEST = json.loads((BENCH / "MANIFEST.json").read_text())


def _archived_files() -> dict[str, Path]:
    return {
        path.relative_to(ORIGINAL).as_posix(): path
        for path in sorted(ORIGINAL.rglob("*"))
        if path.is_file()
    }


def test_archive_holds_exactly_the_manifest_files() -> None:
    assert set(_archived_files()) == set(MANIFEST["files"])


def test_archived_bytes_match_the_manifest() -> None:
    changed = [
        name
        for name, path in _archived_files().items()
        if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST["files"].get(name)
    ]
    assert changed == []


def test_manifest_counts_match_the_archive() -> None:
    files = MANIFEST["files"]
    assert sum(name.startswith("cache/") for name in files) == 997
    assert sum(name.startswith("packets/") for name in files) == 200
    cases = json.loads((ORIGINAL / MANIFEST["cases"]["file"]).read_text())
    assert len(cases) == MANIFEST["cases"]["n"] == 200
    for arm in MANIFEST["arms"]:
        result = json.loads((ORIGINAL / arm["result"]).read_text())
        assert result["n_cases_requested"] == arm["case_limit"]
        assert result["model"] == arm["model"]
        assert result["prompt_version"] == arm["prompt_version"]


def _source_path(name: str) -> str:
    for archived, source in MANIFEST["source_paths"].items():
        if archived.endswith("/") and name.startswith(archived):
            return source + name[len(archived):]
        if name == archived:
            return source
    raise KeyError(name)


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _source_tree() -> dict[str, str] | None:
    commit = MANIFEST["source_commit"]
    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO), "ls-tree", "-r", commit],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tree = {}
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        tree[path] = meta.split()[2]
    return tree


def test_archived_files_are_the_source_commit_blobs() -> None:
    tree = _source_tree()
    if tree is None:
        pytest.skip("the source commit is not in this clone's history")
    mismatched = [
        name
        for name, path in _archived_files().items()
        if tree.get(_source_path(name)) != _git_blob(path.read_bytes())
    ]
    assert mismatched == []
