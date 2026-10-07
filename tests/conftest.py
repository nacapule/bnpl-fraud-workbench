"""Shared test settings.

MySQL tests use the configured endpoint (``config/world.yaml`` with the
``BNPL_DB_*`` overrides), probed with a real login and ``SELECT 1``. They are
skipped when it is unreachable, or fail instead when ``BNPL_REQUIRE_MYSQL=1``
(CI and the finishing checks set it). Tests marked ``reloads_mysql`` drop and
reload the database, so they run only when ``BNPL_DB_DISPOSABLE=1`` says the
configured database may be overwritten, and after every other test.
"""

from __future__ import annotations

import os
from functools import cache

import pytest

from core.config import db_settings, mysql_reachable


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "mysql: needs the configured MySQL endpoint (see tests/conftest.py)"
    )
    config.addinivalue_line(
        "markers", "reloads_mysql: replaces the database contents; runs after other tests"
    )
    config.addinivalue_line(
        "markers", "legacy_world: needs the current simulator's world loaded into MySQL"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    items.sort(key=lambda item: item.get_closest_marker("reloads_mysql") is not None)


@cache
def _reachable() -> bool:
    return mysql_reachable()


@cache
def _legacy_world_loaded() -> bool:
    import pymysql

    settings = db_settings()
    connection = pymysql.connect(**settings.pymysql_kwargs())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'orders' AND column_name = 'ip'",
                (settings.database,),
            )
            return cursor.fetchone()[0] == 1
    finally:
        connection.close()


def pytest_runtest_setup(item: pytest.Item) -> None:
    needs_db = any(
        item.get_closest_marker(name) for name in ("mysql", "reloads_mysql", "legacy_world")
    )
    if not needs_db:
        return
    if not _reachable():
        settings = db_settings()
        message = f"MySQL not reachable at {settings.host}:{settings.port}"
        if os.environ.get("BNPL_REQUIRE_MYSQL") == "1":
            pytest.fail(f"{message} and BNPL_REQUIRE_MYSQL=1")
        pytest.skip(message)
    if item.get_closest_marker("reloads_mysql") and os.environ.get("BNPL_DB_DISPOSABLE") != "1":
        message = "drops and reloads the configured database; set BNPL_DB_DISPOSABLE=1"
        if os.environ.get("BNPL_REQUIRE_MYSQL") == "1":
            pytest.fail(message)
        pytest.skip(message)
    if item.get_closest_marker("legacy_world") and not _legacy_world_loaded():
        pytest.skip("the current simulator's world is not loaded (python db/load.py)")
