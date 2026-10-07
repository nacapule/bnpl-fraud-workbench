"""History, linkage, repayment state and exposure are computed only in core/.

Consumers (rules, models, the replay, packets, analysis, reports) read the
as-of context from core.asof. Modules listed in LEGACY still compute their own
history and are replaced by the as-of context; the list must only shrink, and
the last test passes once it is empty. The SQL investigation library
(db/queries) is checked against the context by its own tests instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCANNED = ("rules", "model", "queue_sim", "llm", "analysis", "report", "pipeline.py")
HISTORY = re.compile(
    r"merge_asof|searchsorted|\.rolling\(|\.expanding\(|cumcount"
    r"|_asof_event_count|_hours_since_event|_rolling_distinct_accounts|_installment_history"
    r"|_windowed_distinct_users|_trailing_count|\.shift\(|\.cumsum\(|\.diff\(|merge_ordered"
    r"|COUNT\(DISTINCT|FROM installments|FROM payments|JOIN installments|JOIN payments"
    r"|\bprincipal\b|\bcollected\b"
)
# Modules that still compute history, linkage, repayment state or exposure.
LEGACY = {
    "llm/packet.py",
}
# Lines that match the pattern but compute no order history, each read and listed here.
NOT_HISTORY = {
    # the share of scores at or above each value (a score distribution's upper tail)
    "rules/tuning.py": {"share = np.cumsum(counts[::-1])[::-1] / len(values)"},
    # service seconds before each day of a weekly staffing calendar
    "queue_sim/roster.py": {"prefix = np.concatenate([[0], np.cumsum(pattern)])"},
}


def _modules() -> list[Path]:
    paths: list[Path] = []
    for entry in SCANNED:
        path = REPO / entry
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.py")))
    return paths


def _matches(module: str, text: str) -> list[str]:
    """Lines of ``text`` that match the pattern, other than those listed as no history
    (compared without their comments)."""
    allowed = NOT_HISTORY.get(module, set())
    return [line for line in text.splitlines()
            if HISTORY.search(line) and line.split("#")[0].strip() not in allowed]


def _computing_history() -> set[str]:
    return {
        module
        for path in _modules()
        if _matches(module := str(path.relative_to(REPO)), path.read_text())
    }


def test_no_new_module_computes_history_outside_core() -> None:
    assert _computing_history() - LEGACY == set()


@pytest.mark.xfail(
    strict=True,
    reason="legacy history code remains in the modules listed in LEGACY until they read "
    "core.asof; remove this mark when LEGACY is empty",
)
def test_history_is_computed_only_in_core() -> None:
    assert _computing_history() == set()
