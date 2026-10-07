"""Q01–Q12 execute against the loaded DB and respect the labels boundary."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
QUERIES = sorted(glob.glob(str(REPO / "db" / "queries" / "Q*.sql")))


def _without_comments(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))


def _statements(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def test_twelve_queries_exist() -> None:
    assert len(QUERIES) == 12


@pytest.mark.parametrize("path", QUERIES)
def test_no_query_reads_ground_truth(path: str) -> None:
    """Q11 reads alerts (analyst-facing); nothing reads labels/stories."""
    sql = _without_comments(Path(path).read_text()).lower()
    assert "labels" not in sql, f"{path} references the labels table"
    assert "stories" not in sql


@pytest.mark.legacy_world
@pytest.mark.parametrize("path", QUERIES)
def test_query_executes_and_returns_rows(path: str) -> None:
    import pandas as pd
    import sqlalchemy as sa

    from core.config import db_settings

    eng = sa.create_engine(db_settings().sqlalchemy_url())
    # Comments out first (a ';' inside a comment is not a statement boundary),
    # then double % for the driver: exec_driver_sql still hands pymysql an empty
    # parameter tuple, which triggers %-interpolation over the raw SQL.
    statements = _statements(_without_comments(Path(path).read_text()))
    with eng.connect() as c:
        for statement in statements[:-1]:
            c.exec_driver_sql(statement.replace("%", "%%"))
        result = c.exec_driver_sql(statements[-1].replace("%", "%%"))
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    if "Q11" in path:
        # queue-ops view may be empty until the rules engine has run
        return
    assert len(df) >= 1, f"{path} returned no rows on demo data"


def test_q11_uses_calendar_day_window() -> None:
    sql = _without_comments((REPO / "db" / "queries" / "Q11_queue_ops.sql").read_text())
    assert "RANGE BETWEEN INTERVAL 6 DAY PRECEDING" in sql
    assert "ROWS BETWEEN 6 PRECEDING" not in sql
