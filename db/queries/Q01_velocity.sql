-- Q01 — Attempt velocity: which accounts and devices are trying to buy faster
-- than ordinary shopping?
-- Cutoff: @as_of. Every query in this library sees only what the platform knew
-- at @as_of: events with known_at <= @as_of and entities created by then. Set
-- @as_of in the session to rewind; when it is unset, the end of observation of
-- the canonical world.
-- Counts: order attempts of any processor result (FP-2 R05), in the world's
-- event order (known_at, then event id), up to and including the attempt shown:
-- attempts_user_1h / _24h / _7d on the account, attempts_device_24h on the
-- device by any account, amount_attempted_user_24h_usd summed on the account.
-- Each window covers [attempt time − window, attempt time].
-- Read: rows meet R05 (more than 3 attempts on the account or more than 5 on
-- the device in 24 h) or show more than 2 attempts on the account in an hour.
-- Velocity alone is never grounds for a decline (FP-2 §6.6); bursts that mix
-- processor declines and an approval are card testing (Q08).
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH attempts AS (
  SELECT event_id, order_id, user_id, device_id, known_at, amount_cents, processor_result
  FROM order_attempts
  WHERE known_at <= @as_of
),
windows AS (
  -- RANGE frames include every attempt at the same instant; attempts later in
  -- the event order at that instant (tie_user, tie_device) are taken back out.
  SELECT a.*,
         COUNT(*) OVER user_1h - COUNT(*) OVER tie_user AS attempts_user_1h,
         COUNT(*) OVER user_24h - COUNT(*) OVER tie_user AS attempts_user_24h,
         COUNT(*) OVER user_7d - COUNT(*) OVER tie_user AS attempts_user_7d,
         SUM(amount_cents) OVER user_24h
           - COALESCE(SUM(amount_cents) OVER tie_user, 0) AS amount_attempted_user_24h_cents,
         COUNT(*) OVER device_24h - COUNT(*) OVER tie_device AS attempts_device_24h
  FROM attempts a
  WINDOW
    user_1h AS (PARTITION BY user_id ORDER BY known_at
                RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW),
    user_24h AS (PARTITION BY user_id ORDER BY known_at
                 RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW),
    user_7d AS (PARTITION BY user_id ORDER BY known_at
                RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW),
    device_24h AS (PARTITION BY device_id ORDER BY known_at
                   RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW),
    tie_user AS (PARTITION BY user_id, known_at ORDER BY event_id
                 ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING),
    tie_device AS (PARTITION BY device_id, known_at ORDER BY event_id
                   ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
)
SELECT order_id, user_id, device_id, known_at AS attempted_at,
       CAST(amount_cents / 100 AS DECIMAL(12, 2)) AS amount_usd, processor_result,
       attempts_user_1h, attempts_user_24h, attempts_user_7d,
       CAST(amount_attempted_user_24h_cents / 100 AS DECIMAL(14, 2))
         AS amount_attempted_user_24h_usd,
       attempts_device_24h
FROM windows
WHERE attempts_user_24h > 3 OR attempts_device_24h > 5 OR attempts_user_1h > 2
ORDER BY GREATEST(attempts_user_24h, attempts_device_24h) DESC,
         amount_attempted_user_24h_cents DESC, order_id
LIMIT 200;
