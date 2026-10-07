"""Load a world (CSV files in the core.world contract) into MySQL.

Usage:
    python db/load_world.py DIR          # validate, drop, create, load, check
    python db/load_world.py --write-schema

The DDL in ``db/schema.sql`` is generated from the table specifications in
``core/world.py`` (``--write-schema`` rewrites it; a test keeps them equal),
followed by the merchant-risk-screener compatibility views. Loading refuses an
invalid world (core.world.validate_world), keeps foreign keys enforced while
inserting, and checks every table's row count. It drops and recreates the
objects it owns, so point it only at a disposable database.
"""

from __future__ import annotations

import argparse
import sys
from graphlib import TopologicalSorter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import world  # noqa: E402
from core.config import DbSettings, db_settings  # noqa: E402

SCHEMA_PATH = REPO / "db" / "schema.sql"
MYSQL_TYPES = {
    "id": "INT",
    "int": "INT",
    "cents": "BIGINT",
    "float": "DOUBLE",
    "ts": "DATETIME",
    "bool": "BOOLEAN",
    "str": "VARCHAR(64)",
}
VARCHAR = {"email": 120, "name": 80, "ip": 45, "line_hash": 40, "profile": 40, "mimic": 40}
# References the column specs cannot express (composite or to a non-entity table).
EXTRA_FOREIGN_KEYS = {
    "payment_attempts": [("plan_id, seq", "installment_schedule (plan_id, seq)")],
    "payment_reversals": [("payment_event_id", "payment_attempts (event_id)")],
}
# Columns referenced by foreign keys that are not the table's primary key.
UNIQUE = {"order_attempts": ["order_id"], "dispute_openings": ["dispute_id"],
          "plans": ["order_id"], "dispute_resolutions": ["dispute_id"]}
EXTRA_INDEXES = {
    "account_events": ["user_id, known_at"],
    "order_attempts": ["user_id, known_at", "merchant_id, known_at"],
    "payment_attempts": ["plan_id, known_at"],
    "cash_events": ["order_id"],
}
COMPAT_VIEWS = ("users", "orders", "chargebacks", "installments")
# Tables of the earlier schema that share names with the views or tables here.
LEGACY_TABLES = (
    "alerts", "labels", "chargebacks", "promo_redemptions", "promos", "account_events",
    "payments", "installments", "plans", "orders", "merchants", "addresses", "cards",
    "user_devices", "devices", "users",
)

COMPAT_SQL = """
-- Compatibility surface for merchant-risk-screener: its 17 columns over the
-- event tables. `orders.status` is the processor result ('approved' or
-- 'declined'); `chargebacks` has one row per dispute dated when the platform
-- learned of it; `installments` gives each installment's state at the end of
-- observation (paid, late, failed, written_off or pending), the final-state
-- view the screener reads. `plans` is a table above.
CREATE VIEW users AS
SELECT user_id, created_at AS signup_ts FROM accounts;

CREATE VIEW orders AS
SELECT order_id, user_id, merchant_id, occurred_at AS ts,
       CAST(amount_cents / 100 AS DECIMAL(12, 2)) AS amount,
       processor_result AS status
FROM order_attempts;

CREATE VIEW chargebacks AS
SELECT dispute_id AS chargeback_id, order_id, reason, known_at AS opened_ts
FROM dispute_openings;

CREATE VIEW installments AS
SELECT s.plan_id, s.seq, s.due_at AS due_ts,
       CAST(s.amount_cents / 100 AS DECIMAL(12, 2)) AS amount,
       CASE
         WHEN paid.first_paid_at IS NOT NULL AND paid.first_paid_at <= s.due_at THEN 'paid'
         WHEN paid.first_paid_at IS NOT NULL THEN 'late'
         WHEN w.plan_id IS NOT NULL THEN 'written_off'
         WHEN failed.plan_id IS NOT NULL THEN 'failed'
         ELSE 'pending'
       END AS outcome
FROM installment_schedule s
LEFT JOIN (
  SELECT a.plan_id, a.seq, MIN(a.occurred_at) AS first_paid_at
  FROM payment_attempts a
  LEFT JOIN payment_reversals r ON r.payment_event_id = a.event_id
  WHERE a.result = 'success' AND r.event_id IS NULL
  GROUP BY a.plan_id, a.seq
) paid ON paid.plan_id = s.plan_id AND paid.seq = s.seq
LEFT JOIN (SELECT DISTINCT plan_id FROM plan_writeoffs) w ON w.plan_id = s.plan_id
LEFT JOIN (
  SELECT DISTINCT plan_id, seq FROM payment_attempts WHERE result = 'failed'
) failed ON failed.plan_id = s.plan_id AND failed.seq = s.seq
WHERE s.seq >= 1;
"""


def _quote(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "''") + "'"


def _column_sql(column: world.Column) -> str:
    if column.values is not None:
        kind = "ENUM(" + ", ".join(_quote(v) for v in column.values) + ")"
    elif column.type == "str":
        kind = f"VARCHAR({VARCHAR.get(column.name, 64)})"
    else:
        kind = MYSQL_TYPES[column.type]
    null = "NULL" if column.nullable else "NOT NULL"
    return f"  {column.name} {kind} {null} COMMENT {_quote(column.meaning)}"


def load_order() -> list[str]:
    """Tables in an order that satisfies every foreign key."""
    graph: dict[str, set[str]] = {name: set() for name in world.TABLES}
    for name, spec in world.TABLES.items():
        for column in spec.columns:
            if column.references:
                target = column.references.split(".")[0]
                if target != name:
                    graph[name].add(target)
        for _, target in EXTRA_FOREIGN_KEYS.get(name, []):
            graph[name].add(target.split()[0])
    return list(TopologicalSorter(graph).static_order())


def table_sql(name: str) -> str:
    spec = world.TABLES[name]
    lines = [_column_sql(column) for column in spec.columns]
    lines.append(f"  PRIMARY KEY ({', '.join(spec.key)})")
    for column in UNIQUE.get(name, []):
        lines.append(f"  UNIQUE KEY uq_{name}_{column} ({column})")
    if spec.role == "event":
        lines.append(f"  KEY ix_{name}_known_at (known_at)")
    for i, columns in enumerate(EXTRA_INDEXES.get(name, []), start=1):
        lines.append(f"  KEY ix_{name}_{i} ({columns})")
    for column in spec.columns:
        if column.references:
            table, target = column.references.split(".")
            lines.append(
                f"  CONSTRAINT fk_{name}_{column.name} FOREIGN KEY ({column.name}) "
                f"REFERENCES {table} ({target})"
            )
    for i, (columns, target) in enumerate(EXTRA_FOREIGN_KEYS.get(name, []), start=1):
        lines.append(f"  CONSTRAINT fk_{name}_x{i} FOREIGN KEY ({columns}) REFERENCES {target}")
    header = f"-- {spec.layer} {spec.role}: {spec.description}"
    body = ",\n".join(lines)
    return (f"{header}\nCREATE TABLE {name} (\n{body}\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 "
            f"COMMENT {_quote(spec.layer)};\n")


def schema_sql() -> str:
    parts = [
        "-- bnpl-fraud-workbench MySQL 8.4 schema, generated from core/world.py by\n"
        "-- `python db/load_world.py --write-schema`; do not edit by hand.\n"
        "-- Latent tables are simulator truth for offline diagnostics only: analyst-facing\n"
        "-- queries, rules, packets and memos never read them or the labels.\n",
    ]
    parts += [table_sql(name) for name in load_order()]
    parts.append(COMPAT_SQL.lstrip("\n"))
    return "\n".join(parts)


def statements(sql: str) -> list[str]:
    """Split on ';' at line ends (the generated DDL has no ';' inside strings)."""
    lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";\n") if s.strip().rstrip(";")]


def _connect(settings: DbSettings):
    import pymysql

    return pymysql.connect(autocommit=False, **settings.pymysql_kwargs())


def recreate_schema(connection) -> None:
    """Drop the objects this schema and the earlier one own, then create the schema."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name, table_type FROM information_schema.tables "
            "WHERE table_schema = DATABASE()"
        )
        existing = dict(cursor.fetchall())
        owned = set(world.TABLES) | set(COMPAT_VIEWS) | set(LEGACY_TABLES)
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        try:
            for name, kind in existing.items():
                if name in owned:
                    cursor.execute(f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} `{name}`")
        finally:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        for statement in statements(SCHEMA_PATH.read_text()):
            cursor.execute(statement)
    connection.commit()


def _rows(name: str, frame: pd.DataFrame) -> list[tuple]:
    spec = world.TABLES[name]
    out = []
    for record in frame[list(spec.column_names)].itertuples(index=False):
        row = []
        for column, value in zip(spec.columns, record, strict=True):
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                row.append(None)
            elif column.type == "ts":
                row.append(pd.Timestamp(value).to_pydatetime())
            elif column.type in ("id", "int", "cents"):
                row.append(int(value))
            elif column.type == "bool":
                row.append(bool(value))
            elif column.type == "float":
                row.append(float(value))
            else:
                row.append(str(value))
        out.append(tuple(row))
    return out


def load_tables(tables: dict[str, pd.DataFrame], settings: DbSettings | None = None) -> dict:
    """Validate, recreate the schema and insert every table; return row counts loaded."""
    world.validate_world(tables)
    connection = _connect(settings or db_settings())
    counts: dict[str, int] = {}
    try:
        recreate_schema(connection)
        with connection.cursor() as cursor:
            for name in load_order():
                spec = world.TABLES[name]
                columns = ", ".join(spec.column_names)
                marks = ", ".join(["%s"] * len(spec.columns))
                cursor.executemany(f"INSERT INTO {name} ({columns}) VALUES ({marks})",
                                   _rows(name, tables[name]))
                cursor.execute(f"SELECT COUNT(*) FROM {name}")
                counts[name] = cursor.fetchone()[0]
                if counts[name] != len(tables[name]):
                    raise RuntimeError(f"{name}: loaded {counts[name]} of {len(tables[name])} rows")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return counts


def load_world(directory: str | Path, settings: DbSettings | None = None) -> dict:
    return load_tables(world.read_world(directory), settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", help="world directory of <table>.csv files")
    parser.add_argument("--write-schema", action="store_true", help="regenerate db/schema.sql")
    args = parser.parse_args()
    if args.write_schema:
        SCHEMA_PATH.write_text(schema_sql())
        print(f"wrote {SCHEMA_PATH.relative_to(REPO)}")
        return
    if not args.directory:
        parser.error("give a world directory or --write-schema")
    for name, count in load_world(args.directory).items():
        print(f"{name:22s} {count:>9,d}")


if __name__ == "__main__":
    main()
