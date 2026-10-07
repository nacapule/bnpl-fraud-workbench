-- Q05 — Takeover chain: on a tenured account, did a credential change, then an
-- address addition, then an order from a newly used device follow within 72
-- hours?
-- Cutoff: @as_of (see Q01); unset means the end of observation.
-- A row is an order attempt (any processor result) on an account at least 90
-- days old where, in this order and within the 72 hours before the attempt:
--   1. the password was changed or reset, or the email changed;
--   2. the account added an address at or after that change;
--   3. the attempt followed, from a device first used on this account in the
--      72 hours before it (FP-2 R01's device condition).
-- When several credential changes qualify, the row shows the earliest, with the
-- first address added after it, so each order appears once.
-- Read: the minutes columns show how tightly scripted the sequence was. R01
-- (FP-2 §6.2) uses a 48-hour credential window; this query looks back 72 hours.
-- New phones, moves, travel and gift addresses explain each step on their own
-- (FP-2 §6.5(b)); the check, not the chain, settles it.
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH device_first_use AS (
  SELECT user_id, device_id, MIN(created_at) AS first_used_at
  FROM device_links
  WHERE created_at <= @as_of
  GROUP BY user_id, device_id
),
recent_device_orders AS (
  SELECT o.event_id, o.order_id, o.user_id, o.device_id, o.ship_address_id, o.known_at AS ordered_at,
         o.amount_cents, o.ip_country, o.processor_result, d.first_used_at AS device_first_used_at,
         TIMESTAMPDIFF(DAY, u.created_at, o.known_at) AS tenure_days
  FROM order_attempts o
  JOIN accounts u ON u.user_id = o.user_id
  JOIN device_first_use d ON d.user_id = o.user_id AND d.device_id = o.device_id
  WHERE o.known_at <= @as_of
    AND u.created_at <= o.known_at - INTERVAL 90 DAY
    AND d.first_used_at >= o.known_at - INTERVAL 72 HOUR
),
chains AS (
  SELECT o.order_id, e.event_id AS credential_event_id, e.kind AS credential_change,
         e.known_at AS changed_at, e.device_id AS changed_on_device,
         MIN(l.created_at) AS address_added_at,
         MAX(l.address_id = o.ship_address_id) AS ships_to_added_address
  FROM recent_device_orders o
  JOIN account_events e
    ON e.user_id = o.user_id
   AND e.kind IN ('password_change', 'password_reset', 'email_change')
   AND e.known_at >= o.ordered_at - INTERVAL 72 HOUR
   AND (e.known_at < o.ordered_at OR (e.known_at = o.ordered_at AND e.event_id < o.event_id))
  JOIN address_links l
    ON l.user_id = o.user_id
   AND l.created_at >= e.known_at
   AND l.created_at <= o.ordered_at
  GROUP BY o.order_id, e.event_id, e.kind, e.known_at, e.device_id
),
first_chain AS (
  SELECT c.*,
         ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY changed_at, credential_event_id) AS pick
  FROM chains c
)
SELECT o.order_id, o.user_id, o.tenure_days, c.credential_change, c.changed_at,
       o.device_first_used_at, c.address_added_at, o.ordered_at,
       TIMESTAMPDIFF(MINUTE, c.changed_at, c.address_added_at) AS change_to_address_minutes,
       TIMESTAMPDIFF(MINUTE, c.changed_at, o.ordered_at) AS change_to_order_minutes,
       CAST(o.amount_cents / 100 AS DECIMAL(12, 2)) AS amount_usd, o.processor_result,
       o.ip_country, o.device_id, (c.changed_on_device = o.device_id) AS changed_on_order_device,
       c.ships_to_added_address
FROM recent_device_orders o
JOIN first_chain c ON c.order_id = o.order_id AND c.pick = 1
ORDER BY o.amount_cents DESC, o.order_id
LIMIT 200;
