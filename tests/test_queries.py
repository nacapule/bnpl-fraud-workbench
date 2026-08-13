"""Q01–Q12 execute against the loaded DB and respect the labels boundary."""

from __future__ import annotations

import glob
import socket
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
QUERIES = sorted(glob.glob(str(REPO / "db" / "queries" / "Q*.sql")))


def _without_comments(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))


def _statements(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def _db_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 3306), timeout=2):
            return True
    except OSError:
        return False


def test_twelve_queries_exist() -> None:
    assert len(QUERIES) == 12


@pytest.mark.parametrize("path", QUERIES)
def test_no_query_reads_ground_truth(path: str) -> None:
    """Q11 reads alerts (analyst-facing); nothing reads labels/stories."""
    sql = _without_comments(Path(path).read_text()).lower()
    assert "labels" not in sql, f"{path} references the labels table"
    assert "stories" not in sql


@pytest.mark.skipif(not _db_up(), reason="mysql not running")
@pytest.mark.parametrize("path", QUERIES)
def test_query_executes_and_returns_rows(path: str) -> None:
    import pandas as pd
    import sqlalchemy as sa
    import yaml

    db = yaml.safe_load(open(REPO / "config.yaml"))["db"]
    url = (f"mysql+pymysql://{db['user']}:{db['password']}@{db['host']}:"
           f"{db['port']}/{db['database']}")
    eng = sa.create_engine(url)
    statements = _statements(Path(path).read_text())
    with eng.connect() as c:
        # exec_driver_sql: the files are plain MySQL (literal % in DATE_FORMAT),
        # so they must bypass the DBAPI's parameter interpolation.
        for statement in statements[:-1]:
            c.exec_driver_sql(statement)
        result = c.exec_driver_sql(statements[-1])
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    if "Q11" in path:
        # queue-ops view may be empty until the rules engine has run
        return
    assert len(df) >= 1, f"{path} returned no rows on demo data"


def test_q11_uses_calendar_day_window() -> None:
    sql = _without_comments((REPO / "db" / "queries" / "Q11_queue_ops.sql").read_text())
    assert "RANGE BETWEEN INTERVAL 6 DAY PRECEDING" in sql
    assert "ROWS BETWEEN 6 PRECEDING" not in sql
