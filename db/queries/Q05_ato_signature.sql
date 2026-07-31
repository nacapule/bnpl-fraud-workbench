-- Q05 — ATO signature: tenured accounts where credentials changed, a new device
-- appeared, an address was added, and an order followed — all inside 72 hours.
-- Read: each row is one candidate takeover chain with its timestamps. The
-- benign mimic is a new phone + a move; what a mimic does NOT do is change the
-- password from a new device first (FP-1 §6.6). gap_hours columns let you see
-- how tightly scripted the sequence was.
WITH cred AS (
  SELECT e.user_id, e.ts AS cred_ts, e.device_id AS cred_device, e.ip AS cred_ip
  FROM account_events e
  WHERE e.kind IN ('password_change', 'email_change')
),
addr AS (
  SELECT e.user_id, e.ts AS addr_ts
  FROM account_events e
  WHERE e.kind = 'address_add'
)
SELECT o.order_id, o.user_id,
       TIMESTAMPDIFF(DAY, u.signup_ts, o.ts) AS tenure_days,
       c.cred_ts, a.addr_ts, o.ts AS order_ts,
       ROUND(TIMESTAMPDIFF(MINUTE, c.cred_ts, a.addr_ts) / 60.0, 1) AS cred_to_addr_hours,
       ROUND(TIMESTAMPDIFF(MINUTE, c.cred_ts, o.ts) / 60.0, 1) AS cred_to_order_hours,
       o.amount, o.ip_country, o.device_id,
       (ud.first_seen >= c.cred_ts - INTERVAL 1 HOUR) AS device_is_new
FROM orders o
JOIN users u ON u.user_id = o.user_id
JOIN cred c ON c.user_id = o.user_id
           AND o.ts BETWEEN c.cred_ts AND c.cred_ts + INTERVAL 72 HOUR
JOIN addr a ON a.user_id = o.user_id
           AND a.addr_ts BETWEEN c.cred_ts AND c.cred_ts + INTERVAL 72 HOUR
LEFT JOIN user_devices ud ON ud.user_id = o.user_id AND ud.device_id = o.device_id
WHERE o.status = 'approved'
  AND o.ts >= u.signup_ts + INTERVAL 90 DAY
ORDER BY o.amount DESC
LIMIT 200;
