-- Q07 — Promotion clusters: are first-purchase discounts being farmed by linked
-- accounts?
-- Cutoff: @as_of (see Q01); unset means the end of observation.
-- A row is one use of a first-purchase promotion (a processor-approved order
-- that carried it) whose account is linked to at least two other accounts that
-- used the same promotion by @as_of: linked_accounts_on_promo counts them with
-- this account (FP-2 R10's count, over every use known at the cutoff rather than
-- only the earlier ones). Accounts are linked, as in R10 and FP-2 §9(e), when
-- they shared a device (links to it active at the same time) or a normalized
-- email (held at the same time; the rule in Q02). A shared shipping address does
-- not link them: households share addresses (FP-2 §6.4).
-- full_price_orders_90d counts this account's approved orders without a
-- promotion in the 90 days after the use, known by @as_of.
-- Read: a cluster whose accounts never return at full price is the §9(e)
-- promotion-abuse shape; that determination is known only 90 days after the
-- latest use.
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH uses AS (
  SELECT o.order_id, o.user_id, o.promo_id, o.known_at AS used_at
  FROM order_attempts o
  JOIN promotions p ON p.promo_id = o.promo_id
  WHERE p.first_purchase_only AND o.processor_result = 'approved' AND o.known_at <= @as_of
),
promo_users AS (
  SELECT DISTINCT promo_id, user_id FROM uses
),
device_holdings AS (
  SELECT l.user_id, l.device_id, l.created_at AS since,
         CASE WHEN l.removed_at <= @as_of THEN l.removed_at END AS until
  FROM device_links l
  WHERE l.created_at <= @as_of AND l.user_id IN (SELECT user_id FROM promo_users)
),
held AS (
  SELECT user_id, created_at AS since, -1 AS event_order, LOWER(TRIM(email)) AS email
  FROM accounts
  WHERE created_at <= @as_of AND user_id IN (SELECT user_id FROM promo_users)
  UNION ALL
  SELECT user_id, known_at, event_id, LOWER(TRIM(email))
  FROM account_events
  WHERE kind = 'email_change' AND known_at <= @as_of
    AND user_id IN (SELECT user_id FROM promo_users)
),
email_parts AS (
  SELECT user_id, since,
         LEAD(since) OVER (PARTITION BY user_id ORDER BY since, event_order) AS until,
         SUBSTRING_INDEX(LEFT(email, CHAR_LENGTH(email) - CHAR_LENGTH(SUBSTRING_INDEX(email, '@', -1)) - 1),
                         '+', 1) AS local_part,
         SUBSTRING_INDEX(email, '@', -1) AS domain
  FROM held
),
email_holdings AS (
  SELECT user_id, since, until,
         CASE WHEN domain IN ('gmail.com', 'googlemail.com')
              THEN CONCAT(REPLACE(local_part, '.', ''), '@gmail.com')
              ELSE CONCAT(local_part, '@', domain) END AS email_root
  FROM email_parts
),
pairs AS (
  SELECT a.user_id, b.user_id AS other_id, 'device' AS via
  FROM device_holdings a
  JOIN device_holdings b ON b.device_id = a.device_id AND b.user_id <> a.user_id
  -- linked at the same time: the intervals intersect
  WHERE GREATEST(a.since, b.since) < LEAST(COALESCE(a.until, '9999-12-31'),
                                           COALESCE(b.until, '9999-12-31'))
  UNION
  SELECT a.user_id, b.user_id, 'email'
  FROM email_holdings a
  JOIN email_holdings b ON b.email_root = a.email_root AND b.user_id <> a.user_id
  -- held at the same time: the intervals intersect (an address taken and
  -- dropped in the same second was never held)
  WHERE GREATEST(a.since, b.since) < LEAST(COALESCE(a.until, '9999-12-31'),
                                           COALESCE(b.until, '9999-12-31'))
),
linked AS (
  SELECT u.order_id, p.other_id, MAX(p.via = 'device') AS by_device, MAX(p.via = 'email') AS by_email
  FROM uses u
  JOIN pairs p ON p.user_id = u.user_id
  JOIN promo_users pu ON pu.promo_id = u.promo_id AND pu.user_id = p.other_id
  GROUP BY u.order_id, p.other_id
),
per_use AS (
  SELECT u.order_id, u.user_id, u.promo_id, u.used_at,
         1 + COUNT(l.other_id) AS linked_accounts_on_promo,
         MAX(l.by_device) AS by_device, MAX(l.by_email) AS by_email
  FROM uses u
  LEFT JOIN linked l ON l.order_id = u.order_id
  GROUP BY u.order_id, u.user_id, u.promo_id, u.used_at
)
SELECT pu.order_id, pu.user_id, pr.code AS promo_code, pu.used_at,
       pu.linked_accounts_on_promo,
       CONCAT_WS('+', CASE WHEN pu.by_device THEN 'device' END,
                 CASE WHEN pu.by_email THEN 'email' END) AS linked_by,
       (SELECT COUNT(*) FROM order_attempts o
        WHERE o.user_id = pu.user_id AND o.processor_result = 'approved' AND o.promo_id IS NULL
          AND o.known_at > pu.used_at AND o.known_at <= pu.used_at + INTERVAL 90 DAY
          AND o.known_at <= @as_of) AS full_price_orders_90d
FROM per_use pu
JOIN promotions pr ON pr.promo_id = pu.promo_id
WHERE pu.linked_accounts_on_promo >= 3
ORDER BY pu.linked_accounts_on_promo DESC, pr.code, pu.used_at, pu.order_id
LIMIT 100;
