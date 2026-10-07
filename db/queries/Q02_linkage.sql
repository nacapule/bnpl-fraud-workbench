-- Q02 — Shared-attribute linkage: how many accounts hang off one device, one
-- ship-to address, or one email identity?
-- Cutoff: @as_of (see Q01); unset means the end of observation. Every source is
-- cut there: attempts and account events known by @as_of, accounts and email
-- changes known by then.
-- Links: a device counts the accounts with an order attempt or account event on
-- it (FP-2 R02's evidence, over all history to @as_of); a ship-to address counts
-- the accounts with an attempt shipping to it (R08's); an email identity counts
-- the accounts holding it at @as_of, after the one normalization rule (FP-2
-- R06(b)): lower case, the +tag removed for every provider, dots removed only
-- for Gmail (googlemail.com is gmail.com), digits kept, so
-- jameshernandez123@gmail.com and jameshernandez456@gmail.com stay apart.
-- An account holds its signup email until an email change is known, then the
-- new address.
-- Read: three or more accounts on one device, address or email identity is ring
-- structure or a household (FP-2 §6.4); ring_score = accounts / (1 + days from
-- the latest link to @as_of) puts fresh clusters first.
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH device_seen AS (
  SELECT device_id, user_id, known_at FROM order_attempts WHERE known_at <= @as_of
  UNION ALL
  SELECT device_id, user_id, known_at FROM account_events WHERE known_at <= @as_of
),
device_link AS (
  SELECT 'device' AS link_type, CAST(device_id AS CHAR) AS link_value,
         COUNT(DISTINCT user_id) AS n_accounts,
         MIN(known_at) AS first_seen, MAX(known_at) AS last_seen
  FROM device_seen
  GROUP BY device_id
  HAVING COUNT(DISTINCT user_id) >= 3
),
address_link AS (
  SELECT 'ship_address' AS link_type, CAST(ship_address_id AS CHAR) AS link_value,
         COUNT(DISTINCT user_id) AS n_accounts,
         MIN(known_at) AS first_seen, MAX(known_at) AS last_seen
  FROM order_attempts
  WHERE known_at <= @as_of
  GROUP BY ship_address_id
  HAVING COUNT(DISTINCT user_id) >= 3
),
held AS (
  -- each address an account took, from when it was known: the signup email,
  -- then every email change, in the world's event order
  SELECT user_id, created_at AS since, -1 AS event_order, LOWER(TRIM(email)) AS email
  FROM accounts
  WHERE created_at <= @as_of
  UNION ALL
  SELECT user_id, known_at, event_id, LOWER(TRIM(email))
  FROM account_events
  WHERE kind = 'email_change' AND known_at <= @as_of
),
current_email AS (
  SELECT user_id, since, email
  FROM (
    SELECT h.*, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY since DESC, event_order DESC) AS latest
    FROM held h
  ) ranked
  WHERE latest = 1
),
email_parts AS (
  SELECT user_id, since,
         SUBSTRING_INDEX(LEFT(email, CHAR_LENGTH(email) - CHAR_LENGTH(SUBSTRING_INDEX(email, '@', -1)) - 1),
                         '+', 1) AS local_part,
         SUBSTRING_INDEX(email, '@', -1) AS domain
  FROM current_email
),
email_link AS (
  SELECT 'email_root' AS link_type, email_root AS link_value,
         COUNT(*) AS n_accounts, MIN(since) AS first_seen, MAX(since) AS last_seen
  FROM (
    SELECT user_id, since,
           CASE WHEN domain IN ('gmail.com', 'googlemail.com')
                THEN CONCAT(REPLACE(local_part, '.', ''), '@gmail.com')
                ELSE CONCAT(local_part, '@', domain) END AS email_root
    FROM email_parts
  ) roots
  GROUP BY email_root
  HAVING COUNT(*) >= 3
)
SELECT link_type, link_value, n_accounts, first_seen, last_seen,
       ROUND(n_accounts / (1 + DATEDIFF(@as_of, last_seen)), 4) AS ring_score
FROM (
  SELECT * FROM device_link
  UNION ALL
  SELECT * FROM address_link
  UNION ALL
  SELECT * FROM email_link
) links
ORDER BY ring_score DESC, n_accounts DESC, link_type, link_value
LIMIT 200;
