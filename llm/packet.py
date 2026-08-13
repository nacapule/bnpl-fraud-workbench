"""Build the case packet for one alert — the ONLY thing the triage model sees.

Hard rule (FP-1 §2.4): nothing here may read the ``labels`` table or
``stories.jsonl``. The packet carries observable facts plus fired-rule
rationales; ``tests/test_packet.py`` walks the JSON to prove label fields are
absent. Derived values the memo may cite (tenure days, ratios, linkage counts)
are materialized as explicit fields so the concrete-token verifier can match
them verbatim.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa

from llm.client import load_config

FORBIDDEN_KEYS = {"label", "labels", "pattern_id", "story_id", "is_fraud"}
REPO = Path(__file__).resolve().parent.parent


def get_engine() -> sa.Engine:
    db = load_config()["db"]
    url = f"mysql+pymysql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
    return sa.create_engine(url)


def _rows(engine: sa.Engine | sa.Connection, sql: str, **params: Any) -> list[dict[str, Any]]:
    connection = nullcontext(engine) if isinstance(engine, sa.Connection) else engine.connect()
    with connection as connected:
        df = pd.read_sql(sa.text(sql), connected, params=params)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return json.loads(df.to_json(orient="records"))


def build_packet(
    alert_id: int, engine: sa.Engine | sa.Connection | None = None
) -> dict[str, Any]:
    engine = engine or get_engine()

    alert = _rows(
        engine,
        """SELECT a.alert_id, a.order_id, a.user_id, a.ts, a.score, a.band, a.fired_rules,
                  o.amount, o.ip, o.ip_country, o.device_id, o.card_id, o.ship_address_id,
                  o.avs_result, o.cvv_result, o.status,
                  m.name AS merchant_name, m.category AS merchant_category
           FROM alerts a
           JOIN orders o ON o.order_id = a.order_id
           JOIN merchants m ON m.merchant_id = o.merchant_id
           WHERE a.alert_id = :aid""",
        aid=alert_id,
    )[0]
    alert["fired_rules"] = json.loads(alert["fired_rules"])
    uid = alert["user_id"]
    order_ts = alert["ts"]

    account = _rows(
        engine,
        """SELECT u.user_id, u.signup_ts, u.email, u.email_domain, u.kyc_country,
                  DATEDIFF(:ts, u.signup_ts) AS tenure_days
           FROM users u WHERE u.user_id = :uid""",
        uid=uid,
        ts=order_ts,
    )[0]

    history = _rows(
        engine,
        """SELECT COUNT(*) AS n_orders_prior,
                  COALESCE(SUM(amount), 0) AS total_spent_prior,
                  COALESCE(AVG(amount), 0) AS avg_amount_prior,
                  SUM(avs_result <> 'Y') AS n_avs_mismatch_prior,
                  SUM(cvv_result <> 'M') AS n_cvv_mismatch_prior
           FROM orders WHERE user_id = :uid AND ts < :ts AND status = 'approved'""",
        uid=uid,
        ts=order_ts,
    )[0]

    amount_ctx = _rows(
        engine,
        """SELECT ROUND(:amt / NULLIF(AVG(ranked.amount), 0), 2)
                     AS amount_over_category_median
           FROM (
             SELECT o2.amount,
                    ROW_NUMBER() OVER (ORDER BY o2.amount) AS rn,
                    COUNT(*) OVER () AS rc
             FROM orders o2
             JOIN merchants m2 ON m2.merchant_id = o2.merchant_id
             WHERE m2.category = :cat AND o2.status = 'approved' AND o2.ts <= :ts
           ) ranked
           WHERE ranked.rn IN (
             FLOOR((ranked.rc + 1) / 2),
             FLOOR((ranked.rc + 2) / 2)
           )""",
        amt=alert["amount"],
        cat=alert["merchant_category"],
        ts=order_ts,
    )[0]

    repayment = _rows(
        engine,
        """SELECT COUNT(*) AS installments_due,
                  SUM(i.paid_ts <= :ts AND i.paid_ts <= i.due_ts) AS paid,
                  SUM(i.paid_ts <= :ts AND i.paid_ts > i.due_ts) AS late,
                  SUM(i.paid_ts IS NULL AND EXISTS (
                    SELECT 1 FROM payments px
                    WHERE px.installment_id = i.installment_id
                      AND px.result = 'fail' AND px.ts <= :ts
                  )) AS failed_or_written_off
           FROM installments i JOIN plans p ON p.plan_id = i.plan_id
           JOIN orders o ON o.order_id = p.order_id
           WHERE o.user_id = :uid AND i.due_ts <= :ts""",
        uid=uid,
        ts=order_ts,
    )[0]

    last_orders = _rows(
        engine,
        """SELECT o.order_id, o.ts, o.amount, o.status, o.ip_country, o.device_id,
                  o.avs_result, o.cvv_result, m.category
           FROM orders o JOIN merchants m ON m.merchant_id = o.merchant_id
           WHERE o.user_id = :uid AND o.ts <= :ts
           ORDER BY o.ts DESC LIMIT 10""",
        uid=uid,
        ts=order_ts,
    )

    events = _rows(
        engine,
        """SELECT ts, kind, ip, device_id FROM account_events
           WHERE user_id = :uid AND ts <= :ts AND ts >= DATE_SUB(:ts, INTERVAL 90 DAY)
           ORDER BY ts DESC LIMIT 40""",
        uid=uid,
        ts=order_ts,
    )

    linkage = _rows(
        engine,
        """SELECT
             (SELECT COUNT(DISTINCT o2.user_id) FROM orders o2
              WHERE o2.device_id = :dev AND o2.user_id <> :uid
                AND o2.ts BETWEEN DATE_SUB(:ts, INTERVAL 30 DAY) AND :ts)
                 AS other_accounts_on_device,
             (SELECT COUNT(DISTINCT o3.user_id) FROM orders o3
              WHERE o3.ship_address_id = :addr AND o3.user_id <> :uid AND o3.ts <= :ts)
                 AS other_accounts_on_ship_address,
             (SELECT COUNT(*) - 1 FROM users u2
              WHERE u2.email_domain = :dom AND SUBSTRING_INDEX(REPLACE(u2.email, '.', ''), '+', 1)
                    = SUBSTRING_INDEX(REPLACE(:email, '.', ''), '+', 1)
                AND u2.signup_ts <= :ts)
                 AS other_accounts_same_email_root""",
        dev=alert["device_id"],
        addr=alert["ship_address_id"],
        uid=uid,
        dom=account["email_domain"],
        email=account["email"],
    )[0]

    card = _rows(
        engine,
        """SELECT c.bin_country, c.network,
                  (SELECT COUNT(*) FROM orders od WHERE od.card_id = c.card_id
                     AND od.status = 'declined'
                     AND od.ts BETWEEN DATE_SUB(:ts, INTERVAL 24 HOUR) AND :ts)
                      AS declines_on_card_24h
           FROM cards c WHERE c.card_id = :cid""",
        cid=alert["card_id"],
        ts=order_ts,
    )[0]

    vendor: dict[str, Any] = {}
    try:
        scores = pd.read_csv(REPO / "vendor" / "fixtures" / "scores.csv")
        for kind, val in (("ip", alert["ip"]), ("email", account["email"])):
            hit = scores[(scores["kind"] == kind) & (scores["value"] == val)]
            if len(hit):
                vendor[f"{kind}_fraud_score"] = int(hit.iloc[0]["fraud_score"])
                vendor["source"] = str(hit.iloc[0]["source"])
    except FileNotFoundError:
        pass

    packet = {
        "alert": alert,
        "account": {**account, **history, **amount_ctx},
        "repayment_history": repayment,
        "card": card,
        "last_orders": last_orders,
        "account_events_90d": events,
        "linkage": linkage,
        "vendor_scores": vendor or None,
    }
    _assert_no_forbidden(packet)
    return packet


def _assert_no_forbidden(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in FORBIDDEN_KEYS:
                raise AssertionError(f"forbidden key {k!r} at {path}")
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_forbidden(v, f"{path}[{i}]")
