"""Answer checks for the SQL investigation library (db/queries).

Every query answers as of a cutoff, ``@as_of``, from what the platform knew
then. The MySQL tests load the mini world (tests/fixtures/mini_world, stories in
its README) and check three things:

* expected rows on the mini world, derived by hand from its stories;
* each repair, on rows that reproduce the defect it fixes (added inside a
  transaction that is rolled back);
* prefix invariance: deleting everything the platform learned after the cutoff
  never changes an answer at that cutoff.

They drop and reload the configured database (see tests/conftest.py).
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from core import asof, ledger, world

REPO = Path(__file__).resolve().parent.parent
QUERY_DIR = REPO / "db" / "queries"
QUERIES = sorted(QUERY_DIR.glob("Q*.sql"))
QUERY = {path.name[:3]: path for path in QUERIES}
MINI = REPO / "tests" / "fixtures" / "mini_world"
MINI_END = "2025-06-30 23:59:59"  # the mini world's last observed instant
FIRST10 = 1  # the mini world's first-purchase promotion (10% off)
mysql = pytest.mark.reloads_mysql


# ---------------------------------------------------------------- query text
def _without_comments(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))


def _statements(sql: str) -> list[str]:
    return [part.strip() for part in _without_comments(sql).split(";") if part.strip()]


def test_twelve_queries_exist() -> None:
    assert [path.name[:3] for path in QUERIES] == [f"Q{n:02d}" for n in range(1, 13)]


@pytest.mark.parametrize("path", QUERIES, ids=lambda p: p.name[:3])
def test_no_query_reads_labels_latent_truth_or_final_state_views(path: Path) -> None:
    """Analysts never see labels or simulator truth, and the screener's views
    (final-state installments, orders without knowledge times) cannot be cut at
    a cutoff."""
    sql = _without_comments(path.read_text()).lower()
    assert "labels" not in sql and "latent_" not in sql
    read = set(re.findall(r"\b(?:from|join)\s+([a-z_]+)", sql))
    assert not read & {"orders", "users", "installments", "chargebacks", "payments"}
    assert ("alerts" in read) == (path.name[:3] == "Q11")


@pytest.mark.parametrize("path", QUERIES, ids=lambda p: p.name[:3])
def test_every_query_takes_a_cutoff_defaulting_to_the_end_of_observation(path: Path) -> None:
    from core.protocol import load_protocol

    first = _statements(path.read_text())[0]
    match = re.fullmatch(r"SET @as_of = CAST\(COALESCE\(@as_of, '([^']+)'\) AS DATETIME\)", first)
    assert match, first
    assert pd.Timestamp(match.group(1)) == load_protocol().observed_until


def _q10_centroids() -> dict[str, tuple[float, float]]:
    sql = QUERY["Q10"].read_text()
    pairs = re.findall(r"SELECT '([A-Z]{2})'(?: AS cc)?, (-?[\d.]+)(?: AS lat)?, (-?[\d.]+)", sql)
    return {cc: (float(lat), float(lon)) for cc, lat, lon in pairs}


def test_q10_centroids_are_the_rule_engines() -> None:
    from rules.engine import CENTROIDS

    assert _q10_centroids() == CENTROIDS


# ---------------------------------------------------------------- MySQL
@pytest.fixture(scope="module")
def loaded():
    import importlib.util

    import pymysql

    from core.config import db_settings

    spec = importlib.util.spec_from_file_location("load_world", REPO / "db" / "load_world.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    loader.load_world(MINI)
    connection = pymysql.connect(**db_settings().pymysql_kwargs(),
                                 cursorclass=pymysql.cursors.DictCursor)
    yield connection
    connection.close()


@pytest.fixture
def db(loaded):
    """The loaded mini world in a transaction that is rolled back afterwards."""
    loaded.rollback()
    yield loaded
    loaded.rollback()
    with loaded.cursor() as cursor:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        cursor.execute("DROP TEMPORARY TABLE IF EXISTS alerts")


def _plain(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, Decimal):
        return str(value)
    return value


def run(db, name: str, as_of: str | None = MINI_END, **params) -> list[dict]:
    """Run one query file at a cutoff (None: the file's default)."""
    with db.cursor() as cursor:
        cursor.execute("SET @as_of = %s, @min_cohort = %s, @min_orders = %s",
                       (as_of, params.get("min_cohort"), params.get("min_orders")))
        for statement in _statements(QUERY[name].read_text()):
            cursor.execute(statement)
        rows = cursor.fetchall()
    return [{key: _plain(value) for key, value in row.items()} for row in rows]


def column(rows: list[dict], name: str) -> list:
    return [row[name] for row in rows]


def _mini_ids() -> tuple[dict[str, int], dict[str, int]]:
    """Story names from the mini world's README: accounts by email prefix
    (``eve``) and their attempts numbered in time order (``eve3``)."""
    tables = world.read_world(MINI, ["accounts", "order_attempts"])
    users = {email.split(".")[0]: int(uid)
             for uid, email in zip(tables["accounts"]["user_id"], tables["accounts"]["email"],
                                   strict=True)}
    names = {uid: key for key, uid in users.items()}
    orders: dict[str, int] = {}
    attempts = tables["order_attempts"].sort_values(["known_at", "event_id"])
    seen: dict[int, int] = {}
    for uid, order_id in zip(attempts["user_id"], attempts["order_id"], strict=True):
        seen[uid] = seen.get(uid, 0) + 1
        orders[f"{names[uid]}{seen[uid]}"] = int(order_id)
    return users, orders


USER, ORDER = _mini_ids()


# ------------------------------------------------------------ extra stories
class Stories:
    """Rows added to the loaded mini world to exercise one query. Ids start at
    1000 (entities), 9000 (orders) and 900000 (events), clear of the mini world."""

    def __init__(self, db) -> None:
        self.db = db
        self.next_event = 900000

    def insert(self, table: str, **row) -> None:
        columns = ", ".join(row)
        marks = ", ".join(["%s"] * len(row))
        with self.db.cursor() as cursor:
            cursor.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(row.values()))

    def event(self, table: str, at: str, **row) -> int:
        self.next_event += 1
        self.insert(table, event_id=self.next_event, occurred_at=at, known_at=at, **row)
        return self.next_event

    def account(self, uid: int, created: str, email: str | None = None) -> None:
        email = email or f"u{uid}@example.com"
        self.insert("accounts", user_id=uid, created_at=created, email=email,
                    email_domain=email.split("@")[1].lower(), home_country="US", dob_year=1990)

    def device(self, did: int, created: str) -> None:
        self.insert("devices", device_id=did, created_at=created, fingerprint=f"fp-{did}",
                    ua_family="iOS")

    def link_device(self, uid: int, did: int, created: str, removed: str | None = None) -> None:
        self.insert("device_links", user_id=uid, device_id=did, created_at=created,
                    removed_at=removed)

    def address(self, aid: int, created: str) -> None:
        self.insert("addresses", address_id=aid, created_at=created, line_hash=f"h-{aid}",
                    city="Austin", region="TX", country="US")

    def link_address(self, uid: int, aid: int, created: str, role: str = "home") -> None:
        self.insert("address_links", user_id=uid, address_id=aid, created_at=created,
                    removed_at=None, role=role)

    def card(self, cid: int, uid: int, created: str, bin_country: str = "US") -> None:
        self.insert("cards", card_id=cid, user_id=uid, created_at=created, removed_at=None,
                    bin_country=bin_country, network="visa", last4=f"{cid % 10000:04d}")

    def customer(self, uid: int, created: str, email: str | None = None, *,
                 device: bool = True, address: bool = True) -> None:
        """An account with its own card, and its own device and home address
        (ids equal to the account's) unless told otherwise."""
        self.account(uid, created, email)
        if device:
            self.device(uid, created)
            self.link_device(uid, uid, created)
        if address:
            self.address(uid, created)
            self.link_address(uid, uid, created)
        self.card(uid, uid, created)

    def attempt(self, order_id: int, uid: int, at: str, amount_cents: int, *,
                device: int | None = None, card: int | None = None, address: int | None = None,
                ip_country: str = "US", declined: bool = False, promo: int | None = None,
                merchant: int = 1) -> int:
        return self.event(
            "order_attempts", at, order_id=order_id, user_id=uid, merchant_id=merchant,
            device_id=device or uid, card_id=card or uid, ship_address_id=address or uid,
            amount_cents=amount_cents, promo_id=promo,
            promo_discount_cents=amount_cents // 10 if promo else 0, ip="10.0.0.1",
            ip_country=ip_country, avs_result="Y", cvv_result="M",
            processor_result="declined" if declined else "approved")

    def account_event(self, uid: int, kind: str, at: str, device: int,
                      email: str | None = None) -> None:
        self.event("account_events", at, user_id=uid, kind=kind, device_id=device,
                   ip="10.0.0.1", ip_country="US", email=email)


def velocity_story(s: Stories) -> None:
    # Account 1001 tries five times in two hours; the last two at the same second.
    s.customer(1001, "2025-04-01 09:00:00")
    for order_id, at in ((9001, "10:00:00"), (9002, "10:20:00"), (9003, "10:40:00"),
                         (9004, "12:00:00"), (9005, "12:00:00")):
        s.attempt(order_id, 1001, f"2025-04-02 {at}", 1000, declined=order_id == 9003)
    # Six accounts each try once on one shared device within six hours.
    s.device(2001, "2025-04-03 08:00:00")
    for n in range(6):
        uid = 1002 + n
        s.customer(uid, "2025-04-03 08:00:00", device=False)
        s.link_device(uid, 2001, "2025-04-03 08:00:00")
        s.attempt(9006 + n, uid, f"2025-04-03 {9 + n:02d}:00:00", 2500, device=2001)


def card_testing_story(s: Stories) -> None:
    # Inventory: three cards declined on device 2010, then the third card
    # approved on the account's other device 2011.
    s.customer(1010, "2025-04-10 00:00:00")
    for did in (2010, 2011):
        s.device(did, "2025-04-10 00:00:00")
        s.link_device(1010, did, "2025-04-10 00:00:00")
    for cid in (2010, 2011, 2012):
        s.card(cid, 1010, "2025-04-10 00:00:00")
    s.attempt(9020, 1010, "2025-04-11 10:00:00", 5000, device=2010, card=2010, declined=True)
    s.attempt(9021, 1010, "2025-04-11 10:01:00", 5000, device=2010, card=2011, declined=True)
    s.attempt(9022, 1010, "2025-04-11 10:02:00", 5000, device=2010, card=2012, declined=True)
    s.attempt(9023, 1010, "2025-04-11 11:10:00", 30000, device=2011, card=2012)
    # A decline at the approval's second but after it in event order is not before it.
    s.attempt(9038, 1010, "2025-04-11 11:10:00", 5000, device=2010, card=2012, declined=True)
    # Burst: card 2013 declined three times on device 2012, then approved for $20
    # and an hour later for $900; card 2014 was declined 24 h before the burst
    # and card 2015 a day after it.
    s.customer(1011, "2025-04-20 00:00:00")
    s.device(2012, "2025-04-20 00:00:00")
    s.link_device(1011, 2012, "2025-04-20 00:00:00")
    for cid in (2013, 2014, 2015):
        s.card(cid, 1011, "2025-04-20 00:00:00")
    s.attempt(9024, 1011, "2025-04-21 10:01:00", 5000, device=2012, card=2014, declined=True)
    for order_id, at in ((9025, "10:00:00"), (9026, "10:01:00"), (9027, "10:02:00")):
        s.attempt(order_id, 1011, f"2025-04-22 {at}", 5000, device=2012, card=2013,
                  declined=True)
    s.attempt(9028, 1011, "2025-04-22 10:10:00", 2000, device=2012, card=2013)
    s.attempt(9029, 1011, "2025-04-22 11:10:00", 90000, device=2012, card=2013)
    s.attempt(9030, 1011, "2025-04-23 09:59:00", 5000, device=2012, card=2015, declined=True)


def geo_story(s: Stories) -> None:
    s.customer(1020, "2025-05-01 00:00:00")
    for order_id, at, country in ((9031, "2025-05-02 08:00:00", "US"),
                                  (9032, "2025-05-02 10:00:00", "RO"),
                                  (9033, "2025-05-02 21:00:00", "GB"),
                                  (9034, "2025-05-03 10:00:00", "US")):
        s.attempt(order_id, 1020, at, 4000, ip_country=country)
    s.customer(1021, "2025-05-01 00:00:00")
    for order_id, at, country in ((9035, "2025-05-05 08:00:00", "US"),
                                  (9036, "2025-05-05 18:00:00", "FR"),
                                  (9037, "2025-05-05 18:00:30", "RO")):
        s.attempt(order_id, 1021, at, 4000, ip_country=country)


def takeover_story(s: Stories) -> None:
    # 1030: the full chain, with a second credential change and a second address.
    s.customer(1030, "2024-06-01 09:00:00")
    s.device(2030, "2025-04-05 00:00:00")
    s.link_device(1030, 2030, "2025-04-05 00:00:00")
    s.account_event(1030, "password_change", "2025-04-05 00:05:00", device=2030)
    s.account_event(1030, "email_change", "2025-04-05 00:10:00", device=2030,
                    email="u1030.new@example.com")
    for aid, at in ((2030, "00:30:00"), (2031, "00:40:00")):
        s.address(aid, f"2025-04-05 {at}")
        s.link_address(1030, aid, f"2025-04-05 {at}", role="shipping")
    s.attempt(9040, 1030, "2025-04-05 01:00:00", 50000, device=2030, address=2030)
    # 1031: credential change, order, and only then an address.
    s.customer(1031, "2024-06-01 09:00:00")
    s.device(2032, "2025-04-06 00:00:00")
    s.link_device(1031, 2032, "2025-04-06 00:00:00")
    s.account_event(1031, "password_change", "2025-04-06 00:10:00", device=2032)
    s.attempt(9041, 1031, "2025-04-06 01:00:00", 50000, device=2032)
    s.address(2032, "2025-04-06 02:00:00")
    s.link_address(1031, 2032, "2025-04-06 02:00:00", role="shipping")
    # 1032: the chain on the device the account has used since it opened.
    s.customer(1032, "2024-06-01 09:00:00")
    s.account_event(1032, "password_change", "2025-04-07 00:00:00", device=1032)
    s.address(2033, "2025-04-07 00:30:00")
    s.link_address(1032, 2033, "2025-04-07 00:30:00", role="shipping")
    s.attempt(9042, 1032, "2025-04-07 01:00:00", 50000, address=2033)
    # 1033: the full chain on an account five weeks old.
    s.customer(1033, "2025-03-01 09:00:00")
    s.device(2034, "2025-04-08 00:00:00")
    s.link_device(1033, 2034, "2025-04-08 00:00:00")
    s.account_event(1033, "password_change", "2025-04-08 00:05:00", device=2034)
    s.address(2034, "2025-04-08 00:30:00")
    s.link_address(1033, 2034, "2025-04-08 00:30:00", role="shipping")
    s.attempt(9043, 1033, "2025-04-08 01:00:00", 50000, device=2034, address=2034)
    # 1034: the address was added before the credential change.
    s.customer(1034, "2024-06-01 09:00:00")
    s.device(2035, "2025-04-09 00:00:00")
    s.link_device(1034, 2035, "2025-04-09 00:00:00")
    s.address(2035, "2025-04-09 00:01:00")
    s.link_address(1034, 2035, "2025-04-09 00:01:00", role="shipping")
    s.account_event(1034, "password_change", "2025-04-09 00:05:00", device=2035)
    s.attempt(9044, 1034, "2025-04-09 01:00:00", 50000, device=2035, address=2035)


def promotion_story(s: Stories) -> None:
    # A household: three accounts at one home address, each with its own device
    # and email, each using FIRST10.
    s.address(2040, "2025-04-01 08:00:00")
    for n, uid in enumerate((1040, 1041, 1042)):
        s.customer(uid, f"2025-04-01 09:{10 * n:02d}:00", address=False)
        s.link_address(uid, 2040, f"2025-04-01 09:{10 * n:02d}:00")
        s.attempt(9050 + n, uid, f"2025-04-0{2 + n} 10:00:00", 6000, address=2040, promo=FIRST10)
    # One phone passed on twice: each account held it only after the last let go.
    s.device(2041, "2025-04-05 10:00:00")
    for n, (uid, day) in enumerate(((1043, 5), (1044, 9), (1045, 13))):
        s.customer(uid, f"2025-04-{day:02d} 10:00:00", device=False)
        s.link_device(uid, 2041, f"2025-04-{day:02d} 10:00:00",
                      None if n == 2 else f"2025-04-{day + 3:02d} 10:00:00")
        s.attempt(9053 + n, uid, f"2025-04-{day:02d} 12:00:00", 6000, device=2041,
                  promo=FIRST10)
    # One Gmail mailbox behind three accounts; the first returns at full price.
    for n, (uid, email) in enumerate(((1046, "sam.lee+a@gmail.com"),
                                      (1047, "samlee@googlemail.com"),
                                      (1048, "Sam.Lee+promo@gmail.com"))):
        s.customer(uid, f"2025-04-15 09:{5 * n:02d}:00", email)
        s.attempt(9056 + n, uid, f"2025-04-16 1{n}:00:00", 6000, promo=FIRST10)
    s.attempt(9059, 1046, "2025-05-01 10:00:00", 8000)


def partial_repayment_story(s: Stories) -> None:
    # A November first plan: the first installment bounces and is paid again,
    # the last two fail.
    s.customer(1050, "2024-11-01 10:00:00")
    s.attempt(9060, 1050, "2024-11-15 10:00:00", 20000)
    s.insert("plans", plan_id=1050, order_id=9060, created_at="2024-11-15 10:00:00",
             principal_cents=20000, down_payment_cents=5000, n_installments=3)
    for seq, due in enumerate(("2024-11-15", "2024-11-29", "2024-12-13", "2024-12-27")):
        s.insert("installment_schedule", plan_id=1050, seq=seq, due_at=f"{due} 10:00:00",
                 amount_cents=5000)

    def pay(seq: int, at: str, result: str = "success", attempt_no: int = 1) -> int:
        return s.event("payment_attempts", at, plan_id=1050, seq=seq, attempt_no=attempt_no,
                       amount_cents=5000, result=result)

    pay(0, "2024-11-15 10:00:00")
    bounced = pay(1, "2024-11-29 10:00:00")
    s.event("payment_reversals", "2024-12-02 10:00:00", payment_event_id=bounced, plan_id=1050,
            amount_cents=5000, reason="bank_return")
    retried = pay(1, "2024-12-05 10:00:00", attempt_no=2)
    pay(2, "2024-12-13 10:00:00", "failed")
    pay(3, "2024-12-27 10:00:00", "failed")
    # the retry comes back too, after the cohort has matured
    s.event("payment_reversals", "2025-02-10 10:00:00", payment_event_id=retried, plan_id=1050,
            amount_cents=5000, reason="card_reversal")


def late_delivery_story(s: Stories) -> None:
    # Two item-not-received claims, both rejected; the carrier confirms the
    # second delivery only after the claim was opened.
    s.customer(1060, "2025-04-01 09:00:00")
    for order_id, ordered, delivered, opened in (
            (9070, "2025-05-01 10:00:00", "2025-05-03 10:00:00", "2025-05-15 10:00:00"),
            (9071, "2025-05-10 10:00:00", "2025-05-20 10:00:00", "2025-05-16 10:00:00")):
        s.attempt(order_id, 1060, ordered, 7000)
        s.event("deliveries", delivered, order_id=order_id)
        s.event("dispute_openings", opened, dispute_id=order_id, order_id=order_id,
                reason="item_not_received", amount_cents=1750)
        s.event("dispute_resolutions", "2025-06-01 10:00:00", dispute_id=order_id, outcome="won")


ALERTS = (  # alert id, order, checkout time, score, band
    ("a1", "fay1", "2025-02-01 12:30:00", 30, "review"),
    ("a2", "dan2", "2025-02-03 02:20:00", 60, "review"),
    ("a3", "fay2", "2025-02-04 10:00:00", 40, "review"),
    ("a4", "jay1", "2025-02-12 16:00:00", 20, "review"),
    ("a5", "eve2", "2025-02-14 01:20:00", 70, "auto_decline"),
    ("a6", "eve3", "2025-02-14 01:40:00", 90, "auto_decline"),
)


def alerts_story(s: Stories) -> None:
    """The routing table the pipeline loads beside the world, for Q11."""
    with s.db.cursor() as cursor:
        cursor.execute(
            "CREATE TEMPORARY TABLE alerts (alert_id VARCHAR(64) NOT NULL PRIMARY KEY, "
            "order_id INT NOT NULL, user_id INT NOT NULL, ts DATETIME NOT NULL, "
            "score DOUBLE NOT NULL, band ENUM('review', 'auto_decline') NOT NULL, "
            "fired_rules JSON NOT NULL, policy VARCHAR(64) NOT NULL, "
            "policy_version VARCHAR(64) NOT NULL) ENGINE=InnoDB")
    for alert_id, order, ts, score, band in ALERTS:
        user = USER[re.sub(r"\d+$", "", order)]
        s.insert("alerts", alert_id=f"{ORDER[order]}:{alert_id}", order_id=ORDER[order],
                 user_id=user, ts=ts, score=score, band=band, fired_rules="[]",
                 policy="rules", policy_version="test")


ALL_STORIES = (velocity_story, card_testing_story, geo_story, takeover_story,
               promotion_story, partial_repayment_story, late_delivery_story, alerts_story)


# ------------------------------------------------------- Q01 velocity
@mysql
def test_q01_mini_world(db) -> None:
    # eve's three attempts in 30 minutes: more than two in an hour.
    assert run(db, "Q01") == [{
        "order_id": ORDER["eve3"], "user_id": USER["eve"], "device_id": 14,
        "attempted_at": "2025-02-14 01:40:00", "amount_usd": "650.00",
        "processor_result": "approved", "attempts_user_1h": 3, "attempts_user_24h": 3,
        "attempts_user_7d": 3, "amount_attempted_user_24h_usd": "1949.98",
        "attempts_device_24h": 3,
    }]


@mysql
def test_q01_counts_attempts_in_event_order_including_declines(db) -> None:
    velocity_story(Stories(db))
    rows = {row["order_id"]: row for row in run(db, "Q01")}
    counts = {order: (row["attempts_user_1h"], row["attempts_user_24h"],
                      row["attempts_device_24h"], row["amount_attempted_user_24h_usd"])
              for order, row in rows.items()}
    assert counts == {
        ORDER["eve3"]: (3, 3, 3, "1949.98"),
        9003: (3, 3, 3, "30.00"),  # declined, counted: three in the hour
        9004: (1, 4, 4, "40.00"),  # the same second as 9005 but first in event order
        9005: (2, 5, 5, "50.00"),
        9011: (1, 1, 6, "25.00"),  # sixth account on device 2001 in 24 h
    }
    assert list(rows) == [9011, 9005, 9004, ORDER["eve3"], 9003]


# ------------------------------------------------------- Q02 linkage
@mysql
def test_q02_mini_world(db) -> None:
    # The never-pay phone (nr1-nr3) and the promotion farm's phone (pf1-pf3);
    # ring_score = 3 / (1 + days from the last order to 2025-06-30).
    rows = run(db, "Q02")
    assert rows == [
        {"link_type": "device", "link_value": "16", "n_accounts": 3,
         "first_seen": "2025-03-10 20:20:00", "last_seen": "2025-03-12 21:20:00",
         "ring_score": "0.0270"},  # 3 / (1 + 110)
        {"link_type": "device", "link_value": "15", "n_accounts": 3,
         "first_seen": "2025-03-03 11:20:00", "last_seen": "2025-03-08 16:20:00",
         "ring_score": "0.0261"},  # 3 / (1 + 114)
    ]


@mysql
def test_q02_cutoff_filters_the_sources(db) -> None:
    # Before pf3 orders, the farm's phone has two accounts; before nr3 orders, none.
    rows = run(db, "Q02", "2025-03-12 21:19:59")
    assert [(r["link_value"], r["n_accounts"], r["last_seen"]) for r in rows] == [
        ("15", 3, "2025-03-08 16:20:00")]
    assert run(db, "Q02", "2025-03-08 16:19:59") == []
    for cutoff in ("2025-02-01 00:00:00", "2025-03-10 00:00:00", "2025-04-01 00:00:00"):
        assert all(row["last_seen"] <= cutoff for row in run(db, "Q02", cutoff))


@mysql
def test_q02_keeps_digits_in_email_identities(db) -> None:
    # nr1.98, nr2.98 and nr3.98 at gmail.com are three mailboxes; stripping
    # trailing digits would merge them into one identity (and pf1.0-pf3.0 too).
    assert [row for row in run(db, "Q02") if row["link_type"] == "email_root"] == []


EMAILS = (
    "jameshernandez123@gmail.com", "jameshernandez456@gmail.com", "jameshernandez789@gmail.com",
    "a.b@googlemail.com", "ab@gmail.com", "A.B+c@Gmail.com",
    "J.Doe@outlook.com", "j.doe+x@outlook.com", "jdoe@outlook.com", "j.doe+y@outlook.com",
    "kim+1@yahoo.com", "kim+2@yahoo.com", "k.im@yahoo.com",
)


@mysql
def test_q02_email_identity_is_the_one_normalization_rule(db) -> None:
    s = Stories(db)
    for n, email in enumerate(EMAILS):
        s.account(1100 + n, "2025-04-01 09:00:00", email)
    counts = pd.Series([asof.normalize_email(email) for email in EMAILS]).value_counts()
    expected = {root: n for root, n in counts.items() if n >= 3}
    assert expected == {"ab@gmail.com": 3, "j.doe@outlook.com": 3}
    rows = run(db, "Q02")
    assert {r["link_value"]: r["n_accounts"] for r in rows
            if r["link_type"] == "email_root"} == expected


@mysql
def test_q02_counts_emails_held_at_the_cutoff(db) -> None:
    s = Stories(db)
    for uid in (1110, 1111, 1112):
        s.customer(uid, "2025-04-01 09:00:00", "kit@yahoo.com")
    s.account_event(1112, "email_change", "2025-04-10 09:00:00", device=1112,
                    email="kit.other@yahoo.com")
    held = [(r["link_value"], r["n_accounts"]) for r in run(db, "Q02", "2025-04-10 08:59:59")
            if r["link_type"] == "email_root"]
    assert held == [("kit@yahoo.com", 3)]
    assert [r for r in run(db, "Q02", "2025-04-10 09:00:00")
            if r["link_type"] == "email_root"] == []


# ------------------------------------------------------- Q03 first orders
@mysql
def test_q03_mini_world(db) -> None:
    # First attempts of accounts under a week old: ben's $1,299.99 (electronics
    # p95 at the cutoff is $899.00, the largest of nine approved amounts with
    # percent rank <= 0.95) and eve's first, declined attempt (AVS failed, GB card
    # from a Romanian IP). fay, nr1-nr3 and pf1-pf3 are small and clean.
    rows = run(db, "Q03")
    assert [(r["order_id"], r["amount_vs_category_p95"], r["processor_result"],
             r["bin_ip_mismatch"]) for r in rows] == [
        (ORDER["ben1"], "1.45", "approved", 0),  # 129,999 / 89,900
        (ORDER["eve1"], "0.72", "declined", 1),  # 64,999 / 89,900
    ]
    assert column(rows, "account_age_hours") == [0, 0]


# ------------------------------------------------------- Q04 matured cohorts
MINI_COHORTS = [  # month, band, first plans, fully paid, partial, zero effort
    ("2024-12", "a_under_100", 2, "1.000", "0.000", "0.000"),  # hal1, dan1
    ("2024-12", "b_100_300", 1, "1.000", "0.000", "0.000"),  # gus1
    ("2025-01", "a_under_100", 2, "1.000", "0.000", "0.000"),  # ana1, cara1
    ("2025-01", "b_100_300", 1, "1.000", "0.000", "0.000"),  # ivy1
    ("2025-01", "c_300_700", 1, "1.000", "0.000", "0.000"),  # kim1
    ("2025-01", "d_over_700", 1, "1.000", "0.000", "0.000"),  # ben1
    ("2025-02", "c_300_700", 2, "0.000", "0.000", "1.000"),  # fay1, eve3
    ("2025-02", "d_over_700", 1, "1.000", "0.000", "0.000"),  # jay1
    ("2025-03", "a_under_100", 3, "1.000", "0.000", "0.000"),  # pf1-pf3
    ("2025-03", "b_100_300", 2, "0.500", "0.000", "0.500"),  # lee1, nr2
    ("2025-03", "c_300_700", 2, "0.000", "0.000", "1.000"),  # nr1, nr3
]  # June (mo1) has not matured: its last installment is due 2025-07-22.


def _cohorts(rows: list[dict]) -> list[tuple]:
    return [(r["order_month"], r["first_order_band"], r["n_first_plans"], r["fully_paid_share"],
             r["partial_share"], r["zero_effort_share"]) for r in rows]


@mysql
def test_q04_mini_world(db) -> None:
    assert _cohorts(run(db, "Q04", min_cohort=1)) == MINI_COHORTS
    assert run(db, "Q04") == []  # no cell reaches the default 20 first plans


@mysql
def test_q04_shows_a_cohort_only_once_it_has_matured(db) -> None:
    # January's last first plan (cara1, 2025-01-20 19:00) has its last
    # installment due 2025-03-03 19:00; January matures 30 days later.
    before = _cohorts(run(db, "Q04", "2025-04-02 18:59:59", min_cohort=1))
    after = _cohorts(run(db, "Q04", "2025-04-02 19:00:00", min_cohort=1))
    assert before == MINI_COHORTS[:2]
    assert after == MINI_COHORTS[:6]


@mysql
def test_q04_counts_payments_less_reversals(db) -> None:
    partial_repayment_story(Stories(db))
    partial = ("2024-11", "b_100_300", 1, "0.000", "1.000", "0.000")
    # matured on 2025-01-26 10:00 (last installment due 2024-12-27 10:00)
    assert _cohorts(run(db, "Q04", "2025-01-26 09:59:59", min_cohort=1)) == []
    assert _cohorts(run(db, "Q04", "2025-01-26 10:00:00", min_cohort=1)) == [partial]
    # the first installment stands until its retry is reversed on 2025-02-10
    assert _cohorts(run(db, "Q04", "2025-02-10 09:59:59", min_cohort=1)) == [partial]
    assert _cohorts(run(db, "Q04", min_cohort=1)) == [
        ("2024-11", "b_100_300", 1, "0.000", "0.000", "1.000"), *MINI_COHORTS]


# ------------------------------------------------------- Q05 takeover chain
@mysql
def test_q05_mini_world(db) -> None:
    # dan: new device 02:10, password reset 02:12, drop address 02:15, order
    # 02:20. lee's password change abroad used his own device; no row.
    assert run(db, "Q05") == [{
        "order_id": ORDER["dan2"], "user_id": USER["dan"], "tenure_days": 885,
        "credential_change": "password_reset", "changed_at": "2025-02-03 02:12:00",
        "device_first_used_at": "2025-02-03 02:10:00", "address_added_at": "2025-02-03 02:15:00",
        "ordered_at": "2025-02-03 02:20:00", "change_to_address_minutes": 3,
        "change_to_order_minutes": 8, "amount_usd": "899.00", "processor_result": "approved",
        "ip_country": "RO", "device_id": 13, "changed_on_order_device": 1,
        "ships_to_added_address": 1,
    }]


@mysql
def test_q05_enforces_the_order_of_events_and_a_new_device(db) -> None:
    takeover_story(Stories(db))
    rows = run(db, "Q05")
    # 9041 (address added after the order), 9042 (the account's old device),
    # 9043 (account five weeks old) and 9044 (address added before the
    # credential change) are not takeover chains.
    assert column(rows, "order_id") == [ORDER["dan2"], 9040]
    chain = rows[1]
    assert (chain["credential_change"], chain["changed_at"], chain["address_added_at"]) == (
        "password_change", "2025-04-05 00:05:00", "2025-04-05 00:30:00")
    assert (chain["change_to_address_minutes"], chain["change_to_order_minutes"]) == (25, 55)
    assert chain["tenure_days"] == (datetime(2025, 4, 5, 1) - datetime(2024, 6, 1, 9)).days


# ------------------------------------------------------- Q06 merchant health
MINI_MERCHANTS = [
    # jewelry: kim1 and jay1, jay's dispute; closed 2025-02-20
    {"merchant_id": 3, "closed_at": "2025-02-20 12:00:00", "orders_all": 2, "orders_90d": 0,
     "dispute_rate_all": "0.5000", "dispute_rate_90d": None, "failed_installment_rate": "0.0000",
     "avg_ticket_all_usd": "595.00", "avg_ticket_90d_usd": None, "ticket_drift": None,
     "new_buyer_share": "0.000"},
    # electronics: nine approved orders; ivy's and eve's disputes; fay1, dan2,
    # eve3 and nr1-nr3 unpaid after failures; ben, fay, eve, nr1-nr3 new buyers
    {"merchant_id": 2, "closed_at": None, "orders_all": 9, "orders_90d": 0,
     "dispute_rate_all": "0.2222", "dispute_rate_90d": None, "failed_installment_rate": "0.6667",
     "avg_ticket_all_usd": "557.11", "avg_ticket_90d_usd": None, "ticket_drift": None,
     "new_buyer_share": "0.667"},
    # home: thirteen orders; hal's two disputes; gus2 and fay2 unpaid; mo1 the
    # only order in the 90 days to the cutoff; fay, pf1-pf3, mo new buyers
    {"merchant_id": 1, "closed_at": None, "orders_all": 13, "orders_90d": 1,
     "dispute_rate_all": "0.1538", "dispute_rate_90d": "0.0000",
     "failed_installment_rate": "0.1538", "avg_ticket_all_usd": "96.88",
     "avg_ticket_90d_usd": "30.00", "ticket_drift": "0.31", "new_buyer_share": "0.385"},
]


def _merchants(rows: list[dict]) -> list[dict]:
    keys = MINI_MERCHANTS[0].keys()
    return [{key: row[key] for key in keys} for row in rows]


@mysql
def test_q06_mini_world(db) -> None:
    assert _merchants(run(db, "Q06", min_orders=1)) == MINI_MERCHANTS
    assert run(db, "Q06") == []  # no merchant has the default 20 orders


@mysql
def test_q06_bounds_every_source_by_the_cutoff(db) -> None:
    # 2024-12-14: the jewelry merchant is not onboarded yet; home has gus1 and
    # hal1, with hal's dispute still three weeks away.
    rows = run(db, "Q06", "2024-12-14 23:59:59", min_orders=1)
    assert [(r["merchant_id"], r["orders_all"], r["dispute_rate_all"]) for r in rows] == [
        (1, 2, "0.0000")]
    # 2025-02-01: home has seven orders; hal2's dispute (known 2025-02-12) and
    # gus2's failures (first installment due 2025-02-08) are later facts.
    rows = run(db, "Q06", "2025-02-01 00:00:00", min_orders=1)
    assert [(r["merchant_id"], r["orders_all"], r["dispute_rate_all"],
             r["failed_installment_rate"], r["closed_at"]) for r in rows] == [
        (1, 7, "0.1429", "0.0000", None),
        (2, 2, "0.0000", "0.0000", None),  # ivy's dispute is known on 2025-02-05
        (3, 1, "0.0000", "0.0000", None),  # closes on 2025-02-20
    ]


# ------------------------------------------------------- Q07 promotion clusters
@mysql
def test_q07_mini_world(db) -> None:
    # pf1-pf3 share one phone and all used FIRST10; ben and cara (a household
    # sharing an address and a tablet) are two accounts, not three.
    rows = run(db, "Q07")
    assert [(r["order_id"], r["linked_accounts_on_promo"], r["linked_by"],
             r["full_price_orders_90d"]) for r in rows] == [
        (ORDER["pf11"], 3, "device", 0), (ORDER["pf21"], 3, "device", 0),
        (ORDER["pf31"], 3, "device", 0)]
    # before pf3's use only two accounts had used it
    assert run(db, "Q07", "2025-03-12 21:19:59") == []


@mysql
def test_q07_links_by_device_or_email_held_at_the_same_time_not_by_address(db) -> None:
    promotion_story(Stories(db))
    rows = run(db, "Q07")
    # The household at one address (9050-9052) and the phone passed from hand to
    # hand (9053-9055) are not clusters; one Gmail mailbox behind three accounts is.
    assert [(r["order_id"], r["linked_accounts_on_promo"], r["linked_by"],
             r["full_price_orders_90d"]) for r in rows] == [
        (ORDER["pf11"], 3, "device", 0), (ORDER["pf21"], 3, "device", 0),
        (ORDER["pf31"], 3, "device", 0),
        (9056, 3, "email", 1), (9057, 3, "email", 0), (9058, 3, "email", 0)]


# ------------------------------------------------------- Q08 card testing
@mysql
def test_q08_mini_world(db) -> None:
    # eve's approval followed two declines on two cards: below every threshold.
    assert run(db, "Q08") == []


@mysql
def test_q08_inventory_branch_trailing_window_and_own_amount(db) -> None:
    card_testing_story(Stories(db))
    rows = run(db, "Q08")
    assert [{key: row[key] for key in (
        "order_id", "device_id", "approved_at", "approved_amount_usd", "declines_card_24h",
        "declines_device_24h", "inventory_device_id", "inventory_cards_24h", "first_decline_at",
    )} for row in rows] == [
        # $900 an hour after the burst: its own amount and time; card 2014's
        # decline 25 hours earlier and card 2015's the next day are outside its
        # trailing 24 hours
        {"order_id": 9029, "device_id": 2012, "approved_at": "2025-04-22 11:10:00",
         "approved_amount_usd": "900.00", "declines_card_24h": 3, "declines_device_24h": 3,
         "inventory_device_id": 2012, "inventory_cards_24h": 1,
         "first_decline_at": "2025-04-22 10:00:00"},
        # three cards declined on device 2010, the third approved on device 2011
        {"order_id": 9023, "device_id": 2011, "approved_at": "2025-04-11 11:10:00",
         "approved_amount_usd": "300.00", "declines_card_24h": 1, "declines_device_24h": 0,
         "inventory_device_id": 2010, "inventory_cards_24h": 3,
         "first_decline_at": "2025-04-11 10:00:00"},
        # the first approval after the burst, $20
        {"order_id": 9028, "device_id": 2012, "approved_at": "2025-04-22 10:10:00",
         "approved_amount_usd": "20.00", "declines_card_24h": 3, "declines_device_24h": 3,
         "inventory_device_id": 2012, "inventory_cards_24h": 1,
         "first_decline_at": "2025-04-22 10:00:00"},
    ]


# ------------------------------------------------------- Q09 INR abuse
@mysql
def test_q09_counts_a_delivery_only_once_the_carrier_confirms_it(db) -> None:
    late_delivery_story(Stories(db))
    rows = [(r["user_id"], r["inr_disputes_opened"], r["claims_on_delivered"],
             r["claims_rejected_on_delivered"], r["claims_pending"])
            for r in run(db, "Q09", "2025-05-18 00:00:00")]
    assert rows == [(USER["hal"], 2, 2, 2, 0), (1060, 2, 1, 0, 2)]
    assert [(r["user_id"], r["claims_on_delivered"], r["claims_rejected_on_delivered"])
            for r in run(db, "Q09", "2025-06-01 10:00:00")] == [(USER["hal"], 2, 2), (1060, 2, 2)]


@mysql
def test_q09_mini_world(db) -> None:
    # hal: two item-not-received claims on delivered orders, both rejected; he
    # repaid every installment. ivy and jay disputed once each.
    hal = {"user_id": USER["hal"], "inr_disputes_opened": 2, "claims_on_delivered": 2,
           "claims_rejected_on_delivered": 2, "claims_pending": 0, "disputed_usd": "85.00",
           "first_opened_at": "2025-01-04 10:00:00", "last_opened_at": "2025-02-12 10:00:00",
           "installments_due": 6, "installments_paid_share": "1.000"}
    assert run(db, "Q09") == [hal]
    # On 2025-03-01 the second claim is open: its rejection is known on 03-10.
    assert run(db, "Q09", "2025-03-01 00:00:00") == [
        hal | {"claims_rejected_on_delivered": 1, "claims_pending": 1}]
    assert run(db, "Q09", "2025-02-12 09:59:59") == []


# ------------------------------------------------------- Q10 geo velocity
def _haversine_km(a: str, b: str) -> float:
    from rules.engine import CENTROIDS

    (lat1, lon1), (lat2, lon2) = CENTROIDS[a], CENTROIDS[b]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(h))


@mysql
def test_q10_mini_world(db) -> None:
    # dan's two attempts (US, then Romania) are six weeks apart; lee's French
    # attempt is his only one.
    assert run(db, "Q10") == []


@mysql
def test_q10_flags_impossible_speed_within_twelve_hours(db) -> None:
    geo_story(Stories(db))
    rows = run(db, "Q10")
    # US to Romania in 2 h, and France to Romania in 30 s (counted as 72 s); US
    # to France in 10 h (about 760 km/h), Romania to Britain in 11 h and Britain
    # to the US after 13 h are possible.
    assert [(r["previous_order_id"], r["order_id"], r["gap_hours"]) for r in rows] == [
        (9036, 9037, "0.01"), (9031, 9032, "2.00")]
    for row, hours in zip(rows, (0.02, 2.0), strict=True):
        km = _haversine_km(row["previous_country"], row["ip_country"])
        assert abs(row["km"] - km) <= 1
        assert abs(row["implied_kmh"] - km / hours) <= 1
    assert _haversine_km("US", "FR") / 10 < 900


@mysql
def test_every_ip_country_has_a_centroid(db) -> None:
    with db.cursor() as cursor:
        cursor.execute("SELECT DISTINCT ip_country FROM order_attempts")
        countries = {row["ip_country"] for row in cursor.fetchall()}
    assert countries <= set(_q10_centroids())


# ------------------------------------------------------- Q11 routing volumes
@mysql
def test_q11_daily_volumes_and_seven_calendar_days(db) -> None:
    alerts_story(Stories(db))
    rows = [(r["d"].isoformat(), r["band"], r["n_alerts"], r["avg_score"],
             r["n_7_calendar_day_rolling"]) for r in run(db, "Q11")]
    assert rows == [
        ("2025-02-14", "auto_decline", 2, 80.0, 2),
        ("2025-02-12", "review", 1, 20.0, 1),  # 02-04 is eight days earlier
        ("2025-02-04", "review", 1, 40.0, 3),
        ("2025-02-03", "review", 1, 60.0, 2),
        ("2025-02-01", "review", 1, 30.0, 1),
    ]
    assert run(db, "Q11", "2025-02-14 01:30:00")[0]["avg_score"] == 70.0


# ------------------------------------------------------- Q12 loss accounting
MINI_ORDER_NET = {  # cents at the end of observation, from the mini world README
    "gus1": 600, "hal1": -1125, "dan1": 225, "ana1": 423, "ivy1": -450, "kim1": 2050,
    "ben1": -6500, "hal2": -1025, "cara1": -310, "gus2": -13200, "fay1": -28600,
    "dan2": -49445, "fay2": -9900, "ana2": 2150, "jay1": -36600, "eve3": -53500,
    "lee1": 775, "nr11": -17600, "nr21": -15125, "nr31": -22550, "pf11": -275,
    "pf21": -290, "pf31": -300, "mo1": -1350,
}
# Repaid orders net the 5% merchant fee (gus1 +$6.00); hal2 also paid the $15
# dispute fee on a claim it won; cara1 and the pf orders paid FIRST10's discount.
MINI_LOSS_CELLS = [  # month, category, account age, orders, loss orders, written off, loss
    ("2024-12", "home", "gt_180d", 3, 1, 0, "3.00"),  # gus1, hal1, dan1
    ("2025-01", "home", "gt_180d", 3, 2, 1, "138.02"),  # ana1, hal2, gus2
    ("2025-01", "electronics", "lt_30d", 1, 1, 0, "65.00"),  # ben1
    ("2025-01", "electronics", "gt_180d", 1, 1, 0, "4.50"),  # ivy1
    ("2025-01", "home", "30_180d", 1, 1, 0, "3.10"),  # cara1
    ("2025-01", "jewelry", "gt_180d", 1, 0, 0, "-20.50"),  # kim1
    ("2025-02", "electronics", "lt_30d", 2, 2, 2, "821.00"),  # fay1, eve3
    ("2025-02", "electronics", "gt_180d", 2, 1, 1, "472.95"),  # dan2, ana2
    ("2025-02", "jewelry", "30_180d", 1, 1, 0, "366.00"),  # jay1
    ("2025-02", "home", "lt_30d", 1, 1, 1, "99.00"),  # fay2
    ("2025-03", "electronics", "lt_30d", 3, 3, 3, "552.75"),  # nr1-nr3
    ("2025-03", "home", "lt_30d", 3, 3, 0, "8.65"),  # pf1-pf3
    ("2025-03", "home", "gt_180d", 1, 0, 0, "-7.75"),  # lee1
    ("2025-06", "home", "lt_30d", 1, 1, 0, "13.50"),  # mo1
]


def _cents(text: str) -> int:
    return int(Decimal(text) * 100)


@mysql
def test_q12_mini_world(db) -> None:
    rows = run(db, "Q12")
    assert [(r["order_month"], r["category"], r["account_age_band"], r["n_orders"],
             r["n_loss_orders"], r["n_written_off_plans"], r["loss_usd"])
            for r in rows] == MINI_LOSS_CELLS
    assert sum(_cents(r["loss_usd"]) for r in rows) == -sum(MINI_ORDER_NET.values())
    for row in rows:
        parts = ("merchant_payouts_usd", "collections_usd", "disputes_usd", "recoveries_usd")
        assert _cents(row["loss_usd"]) == -sum(_cents(row[part]) for part in parts)


def _q12_oracle(tables: dict[str, pd.DataFrame], cutoff: str) -> list[tuple]:
    """Q12 recomputed with pandas from the world's files."""
    cash = tables["cash_events"]
    cash = cash.loc[cash["known_at"] <= pd.Timestamp(cutoff)]
    group = {"merchant_settlement": "payout", "promotion_funding": "payout",
             "customer_payment": "collection", "payment_reversal": "collection",
             "refund": "collection", "dispute_debit": "dispute", "dispute_fee": "dispute",
             "dispute_won_credit": "dispute", "merchant_recourse": "dispute",
             "recovery": "recovery"}
    per_order = cash.assign(part=cash["kind"].map(group)).pivot_table(
        index="order_id", columns="part", values="amount_cents", aggfunc="sum", fill_value=0)
    per_order = per_order.reindex(columns=["payout", "collection", "dispute", "recovery"],
                                  fill_value=0)
    per_order["net"] = per_order.sum(axis=1)
    writeoffs = tables["plan_writeoffs"]
    writeoffs = writeoffs.loc[writeoffs["known_at"] <= pd.Timestamp(cutoff)].merge(
        tables["plans"][["plan_id", "order_id"]], on="plan_id")
    orders = (tables["order_attempts"][["order_id", "user_id", "merchant_id", "occurred_at"]]
              .merge(tables["accounts"][["user_id", "created_at"]], on="user_id")
              .merge(tables["merchants"][["merchant_id", "category"]], on="merchant_id")
              .set_index("order_id"))
    frame = per_order.join(orders, how="left")
    frame["written_off"] = writeoffs.groupby("order_id")["outstanding_cents"].sum()
    age = frame["occurred_at"] - frame["created_at"]
    frame["band"] = ["lt_30d" if a < pd.Timedelta(days=30) else
                     "30_180d" if a < pd.Timedelta(days=180) else "gt_180d" for a in age]
    frame["month"] = frame["occurred_at"].dt.strftime("%Y-%m")
    cells = frame.groupby(["month", "category", "band"]).agg(
        n=("net", "size"), losing=("net", lambda s: int((s < 0).sum())),
        written=("written_off", "count"), payout=("payout", "sum"),
        collection=("collection", "sum"), dispute=("dispute", "sum"),
        recovery=("recovery", "sum"), net=("net", "sum"),
        written_cents=("written_off", "sum"))
    return sorted(
        ((month, category, band, int(c.n), int(c.losing), int(c.written), int(c.payout),
          int(c.collection), int(c.dispute), int(c.recovery), -int(c.net),
          int(c.written_cents)) for (month, category, band), c in cells.iterrows()),
        key=lambda cell: (cell[0], -cell[10], cell[1], cell[2]))


@mysql
@pytest.mark.parametrize("cutoff", ["2025-01-31 23:59:59", "2025-03-15 12:00:00",
                                    "2025-04-30 23:59:59", MINI_END])
def test_q12_reconciles_with_the_ledger(db, cutoff: str) -> None:
    tables = world.read_world(MINI)
    rows = run(db, "Q12", cutoff)
    assert [(r["order_month"], r["category"], r["account_age_band"], r["n_orders"],
             r["n_loss_orders"], r["n_written_off_plans"], _cents(r["merchant_payouts_usd"]),
             _cents(r["collections_usd"]), _cents(r["disputes_usd"]),
             _cents(r["recoveries_usd"]), _cents(r["loss_usd"]), _cents(r["written_off_usd"]))
            for r in rows] == _q12_oracle(tables, cutoff)
    net = ledger.net_cents(tables["cash_events"], until=cutoff, clock="known_at")
    assert sum(_cents(r["loss_usd"]) for r in rows) == -net


# ------------------------------------------------------- prefix invariance
def _cut_world(db, cutoff: str) -> None:
    """Delete what the platform learned after the cutoff: events known later,
    entities created later, and later removals and closures."""
    with db.cursor() as cursor:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        for name, spec in world.TABLES.items():
            if spec.layer != "observable":
                continue
            if spec.role == "event":
                cursor.execute(f"DELETE FROM {name} WHERE known_at > %s", (cutoff,))
            elif spec.created:
                cursor.execute(f"DELETE FROM {name} WHERE {spec.created} > %s", (cutoff,))
                for col in spec.columns:
                    if col.type == "ts" and col.nullable:
                        cursor.execute(f"UPDATE {name} SET {col.name} = NULL "
                                       f"WHERE {col.name} > %s", (cutoff,))
        cursor.execute("DELETE s FROM installment_schedule s LEFT JOIN plans p "
                       "ON p.plan_id = s.plan_id WHERE p.plan_id IS NULL")
        cursor.execute("DELETE FROM alerts WHERE ts > %s", (cutoff,))
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")


def _instants(db) -> list[str]:
    """Cutoffs: every fifth distinct event or creation time, and one second before."""
    parts = [f"SELECT known_at AS t FROM {name}" for name, spec in world.TABLES.items()
             if spec.layer == "observable" and spec.role == "event"]
    parts += [f"SELECT {spec.created} FROM {name}" for name, spec in world.TABLES.items()
              if spec.layer == "observable" and spec.created]
    with db.cursor() as cursor:
        cursor.execute(f"SELECT DISTINCT t FROM ({' UNION '.join(parts)}) x ORDER BY t")
        times = [row["t"] for row in cursor.fetchall()]
    picked = times[::5]
    return sorted({str(t) for t in picked} | {str(t - pd.Timedelta(seconds=1)) for t in picked})


@mysql
def test_answers_never_change_when_later_events_are_added(db) -> None:
    stories = Stories(db)
    for story in ALL_STORIES:
        story(stories)
    params = {"min_cohort": 1, "min_orders": 1}
    cutoffs = _instants(db)
    full = {cutoff: {name: run(db, name, cutoff, **params) for name in QUERY}
            for cutoff in cutoffs}
    assert all(any(full[c][name] for c in cutoffs) for name in QUERY), "a query is never exercised"
    with db.cursor() as cursor:
        for cutoff in cutoffs:
            cursor.execute("SAVEPOINT cut")
            _cut_world(db, cutoff)
            for name in QUERY:
                assert run(db, name, cutoff, **params) == full[cutoff][name], (name, cutoff)
            cursor.execute("ROLLBACK TO SAVEPOINT cut")
