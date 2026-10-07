"""Reproducible figures: SVG files that a rebuild writes byte for byte the same.

Matplotlib stamps each SVG with the current date and its own version, and
gives elements random ids unless ``svg.hashsalt`` is set, so a rebuild with
identical data still changes the file. :func:`save_svg` writes no date or
creator metadata and uses a fixed salt. :func:`pin` applies the same settings
to every figure saved in the process (the pipeline calls it before running
any stage), for producers that save figures themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RC = {"svg.hashsalt": "bnpl-fraud-workbench", "svg.fonttype": "path"}
SOURCE_DATE_EPOCH = "0"
METADATA = {"Date": None, "Creator": None}


def pin() -> None:
    """Fix the SVG id salt and the metadata date for every figure saved from now on."""
    matplotlib.rcParams.update(RC)
    os.environ["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH


def save_svg(figure, path: Path) -> Path:
    """Save ``figure`` as an SVG with no date or creator and stable element ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(RC):
        figure.savefig(path, format="svg", metadata=METADATA)
    return path
