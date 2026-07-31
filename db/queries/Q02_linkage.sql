-- Q02 — Shared-attribute linkage: how many accounts hang off one device, one
-- ship-to address, or one email root (plus-tags and dots collapsed)?
-- Read: legitimate households share a device across ≤2 accounts; ≥3 accounts on
-- a device/address, or many plus-tag variants of one mailbox, is ring structure.
-- ring_score ranks by breadth × recency so fresh clusters surface first.
WITH device_link AS (
  SELECT o.device_id AS link_value, 'device' AS link_type,
         COUNT(DISTINCT o.user_id) AS n_accounts,
         MIN(o.ts) AS first_seen, MAX(o.ts) AS last_seen
  FROM orders o
  GROUP BY o.device_id
  HAVING COUNT(DISTINCT o.user_id) >= 3
),
addr_link AS (
  SELECT o.ship_address_id AS link_value, 'ship_address' AS link_type,
         COUNT(DISTINCT o.user_id) AS n_accounts,
         MIN(o.ts) AS first_seen, MAX(o.ts) AS last_seen
  FROM orders o
  GROUP BY o.ship_address_id
  HAVING COUNT(DISTINCT o.user_id) >= 3
),
email_root AS (
  SELECT CONCAT(
           REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(u.email, '@', 1), '+', 1), '.', ''),
           '@', u.email_domain) AS root,
         COUNT(*) AS n_accounts,
         MIN(u.signup_ts) AS first_seen, MAX(u.signup_ts) AS last_seen
  FROM users u
  GROUP BY root
  HAVING COUNT(*) >= 3
)
SELECT link_type, CAST(link_value AS CHAR) AS link_value, n_accounts,
       first_seen, last_seen,
       ROUND(n_accounts * (1 / (1 + DATEDIFF('2026-06-30', last_seen))), 4) AS ring_score
FROM device_link
UNION ALL
SELECT link_type, CAST(link_value AS CHAR), n_accounts, first_seen, last_seen,
       ROUND(n_accounts * (1 / (1 + DATEDIFF('2026-06-30', last_seen))), 4)
FROM addr_link
UNION ALL
SELECT 'email_root', root, n_accounts, first_seen, last_seen,
       ROUND(n_accounts * (1 / (1 + DATEDIFF('2026-06-30', last_seen))), 4)
FROM email_root
ORDER BY ring_score DESC, n_accounts DESC
LIMIT 200;
