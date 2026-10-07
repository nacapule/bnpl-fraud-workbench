"""Split configuration files and the database endpoint."""

from __future__ import annotations

import socket

from core import config


def test_split_files_load() -> None:
    assert config.load("world")["product"]["down_payment_bps"] == 2500
    assert config.load("policy")["rules"]["bands"]["review"] == 30
    assert "tasks" in config.load("llm")


def test_load_returns_a_copy() -> None:
    first = config.load("policy")
    first["rules"]["bands"]["review"] = -1
    assert config.load("policy")["rules"]["bands"]["review"] == 30


def test_db_settings_take_environment_overrides() -> None:
    base = config.db_settings({})
    moved = config.db_settings({"BNPL_DB_HOST": "db.example", "BNPL_DB_PORT": "3310"})
    assert (moved.host, moved.port) == ("db.example", 3310)
    assert (moved.user, moved.database) == (base.user, base.database)
    assert moved.sqlalchemy_url().endswith("@db.example:3310/" + base.database)


def test_legacy_config_uses_the_configured_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("BNPL_DB_PORT", "3399")
    legacy = config.legacy()
    assert legacy["db"]["port"] == 3399
    assert legacy["seed"] == 416


def test_unreachable_endpoint_is_reported_not_raised() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    settings = config.db_settings({"BNPL_DB_HOST": "127.0.0.1", "BNPL_DB_PORT": str(free_port)})
    assert config.mysql_reachable(settings, timeout=1) is False


def test_reachability_probe_uses_the_configured_endpoint(monkeypatch) -> None:
    """The MySQL gate must probe the configured host and port, not a fixed socket."""
    import pymysql

    calls: list[dict] = []

    def fake_connect(**kwargs):
        calls.append(kwargs)
        raise pymysql.err.OperationalError(2003, "refused")

    monkeypatch.setattr(pymysql, "connect", fake_connect)
    monkeypatch.setenv("BNPL_DB_HOST", "mysql.internal")
    monkeypatch.setenv("BNPL_DB_PORT", "4321")
    assert config.mysql_reachable() is False
    assert (calls[0]["host"], calls[0]["port"]) == ("mysql.internal", 4321)


def test_open_port_that_is_not_mysql_is_unreachable() -> None:
    """A listening socket alone does not count; the probe needs a login and SELECT 1."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        settings = config.db_settings({"BNPL_DB_HOST": "127.0.0.1", "BNPL_DB_PORT": str(port)})
        assert config.mysql_reachable(settings, timeout=1) is False
