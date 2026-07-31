-- Q07 — Promo-redemption clustering: are "first purchase" discounts being
-- farmed by linked accounts?
-- Read: a device or address with ≥3 distinct redeeming accounts is a promo
-- farm candidate; email-root clustering catches the same mailbox plus-tagging
-- its way through the promo. Join Q02 for the full linkage picture.
WITH red AS (
  SELECT pr.redemption_id, pr.promo_id, pr.user_id, pr.ts,
         o.device_id, o.ship_address_id,
         CONCAT(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(u.email, '@', 1), '+', 1), '.', ''),
                '@', u.email_domain) AS email_root
  FROM promo_redemptions pr
  JOIN orders o ON o.order_id = pr.order_id
  JOIN users u ON u.user_id = pr.user_id
)
SELECT 'device' AS cluster_type, CAST(device_id AS CHAR) AS cluster_value,
       promo_id, COUNT(DISTINCT user_id) AS n_accounts,
       MIN(ts) AS first_redemption, MAX(ts) AS last_redemption
FROM red GROUP BY device_id, promo_id HAVING COUNT(DISTINCT user_id) >= 3
UNION ALL
SELECT 'ship_address', CAST(ship_address_id AS CHAR), promo_id,
       COUNT(DISTINCT user_id), MIN(ts), MAX(ts)
FROM red GROUP BY ship_address_id, promo_id HAVING COUNT(DISTINCT user_id) >= 3
UNION ALL
SELECT 'email_root', email_root, promo_id,
       COUNT(DISTINCT user_id), MIN(ts), MAX(ts)
FROM red GROUP BY email_root, promo_id HAVING COUNT(DISTINCT user_id) >= 3
ORDER BY n_accounts DESC
LIMIT 100;
