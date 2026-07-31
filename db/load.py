"""Load data/*.csv into MySQL. Idempotent: drops and recreates the schema.

Usage: python db/load.py   (or `make load`)

Ends with a QA table — CSV row count vs DB row count per table, plus orphan
checks — and exits nonzero on any mismatch (the workspace habit: loads prove
themselves).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pymysql
import yaml

REPO = Path(__file__).resolve().parent.parent

TABLES = [
    "users", "devices", "user_devices", "cards", "addresses", "merchants", "orders",
    "plans", "installments", "payments", "account_events", "promos", "promo_redemptions",
    "chargebacks", "labels",
]

ORPHAN_CHECKS = [
    ("orders→users", "SELECT COUNT(*) FROM orders o LEFT JOIN users u USING(user_id) "
                     "WHERE u.user_id IS NULL"),
    ("installments→plans", "SELECT COUNT(*) FROM installments i LEFT JOIN plans p "
                           "USING(plan_id) WHERE p.plan_id IS NULL"),
    ("payments→plans", "SELECT COUNT(*) FROM payments y LEFT JOIN plans p USING(plan_id) "
                       "WHERE p.plan_id IS NULL"),
    ("labels→orders", "SELECT COUNT(*) FROM labels l LEFT JOIN orders o USING(order_id) "
                      "WHERE o.order_id IS NULL"),
    ("chargebacks→orders", "SELECT COUNT(*) FROM chargebacks c LEFT JOIN orders o "
                           "USING(order_id) WHERE o.order_id IS NULL"),
]


def connect(cfg: dict, retries: int = 60) -> pymysql.Connection:
    last: Exception | None = None
    for _ in range(retries):
        try:
            return pymysql.connect(
                host=cfg["host"], port=cfg["port"], user=cfg["user"],
                password=cfg["password"], database=cfg["database"], autocommit=False,
                local_infile=False,
            )
        except pymysql.err.OperationalError as e:
            last = e
            time.sleep(2)
    raise SystemExit(f"MySQL unreachable after {retries * 2}s: {last}")


def load_table(conn: pymysql.Connection, name: str) -> int:
    path = REPO / "data" / f"{name}.csv"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        placeholders = ",".join(["%s"] * len(header))
        sql = f"INSERT INTO {name} ({','.join(header)}) VALUES ({placeholders})"
        n = 0
        batch: list[list[str | None]] = []
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for row in reader:
                batch.append([v if v != "" else None for v in row])
                if len(batch) >= 5000:
                    cur.executemany(sql, batch)
                    n += len(batch)
                    batch = []
            if batch:
                cur.executemany(sql, batch)
                n += len(batch)
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
    return n


def main() -> None:
    with open(REPO / "config.yaml") as f:
        cfg = yaml.safe_load(f)["db"]
    t0 = time.time()
    conn = connect(cfg)
    ddl = "\n".join(
        line for line in (REPO / "db" / "schema.sql").read_text().splitlines()
        if not line.strip().startswith("--")
    )
    with conn.cursor() as cur:
        for stmt in ddl.split(";"):
            if stmt.strip():
                cur.execute(stmt)
    conn.commit()

    print(f"{'table':<20}{'csv':>10}{'db':>10}  ok")
    failures = 0
    for name in TABLES:
        n_csv = load_table(conn, name)
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {name}")
            n_db = cur.fetchone()[0]
        ok = n_csv == n_db
        failures += 0 if ok else 1
        print(f"{name:<20}{n_csv:>10,d}{n_db:>10,d}  {'✓' if ok else 'MISMATCH'}")

    for label, sql in ORPHAN_CHECKS:
        with conn.cursor() as cur:
            cur.execute(sql)
            n = cur.fetchone()[0]
        ok = n == 0
        failures += 0 if ok else 1
        print(f"orphans {label:<24} {n}  {'✓' if ok else 'FAIL'}")

    print(f"loaded in {time.time() - t0:.1f}s")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
