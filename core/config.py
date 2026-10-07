"""Configuration: the world, policy and LLM files, and the database endpoint.

Settings live in three files under ``config/``:

- ``config/world.yaml``: the synthetic world (volumes, product terms, labelling,
  family parameters) and the database the world is loaded into;
- ``config/policy.yaml``: the operating policy (rule bands, costs, capacity,
  roster, service levels), the one policy artifact the engine, the replay and
  the case files read;
- ``config/llm.yaml``: the memo drafter's backends, models and benchmark sizes.

Evaluation windows, families and seeds are pre-registered separately in
``experiments/protocol.yaml`` (see :mod:`core.protocol`).

The database endpoint is read from ``config/world.yaml`` and each field can be
overridden from the environment (``BNPL_DB_HOST``, ``BNPL_DB_PORT``,
``BNPL_DB_USER``, ``BNPL_DB_PASSWORD``, ``BNPL_DB_NAME``), so tests can point at
any MySQL instance without editing a shared file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "config"
CONFIG_NAMES = ("world", "policy", "llm")

DB_ENV = {
    "host": "BNPL_DB_HOST",
    "port": "BNPL_DB_PORT",
    "user": "BNPL_DB_USER",
    "password": "BNPL_DB_PASSWORD",
    "database": "BNPL_DB_NAME",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping")
    return data


@cache
def _load_cached(name: str) -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / f"{name}.yaml")


def load(name: str) -> dict[str, Any]:
    """Return a fresh copy of ``config/<name>.yaml`` (``world``, ``policy`` or ``llm``)."""
    if name not in CONFIG_NAMES:
        raise KeyError(f"unknown configuration {name!r}; expected one of {CONFIG_NAMES}")
    return _deep_copy(_load_cached(name))


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


@dataclass(frozen=True)
class DbSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/"
            f"{self.database}"
        )

    def pymysql_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }


def db_settings(environ: dict[str, str] | None = None) -> DbSettings:
    """The configured MySQL endpoint with any ``BNPL_DB_*`` environment overrides."""
    env = os.environ if environ is None else environ
    values = dict(load("world")["db"])
    for field, variable in DB_ENV.items():
        if env.get(variable):
            values[field] = env[variable]
    return DbSettings(
        host=str(values["host"]),
        port=int(values["port"]),
        user=str(values["user"]),
        password=str(values["password"]),
        database=str(values["database"]),
    )


def mysql_reachable(settings: DbSettings | None = None, timeout: float = 3.0) -> bool:
    """True when the endpoint accepts a login and answers ``SELECT 1``."""
    import pymysql

    settings = settings or db_settings()
    try:
        connection = pymysql.connect(
            connect_timeout=timeout,
            read_timeout=timeout,
            write_timeout=timeout,
            **settings.pymysql_kwargs(),
        )
    except pymysql.err.MySQLError:
        return False
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone() == (1,)
    except pymysql.err.MySQLError:
        return False
    finally:
        connection.close()
