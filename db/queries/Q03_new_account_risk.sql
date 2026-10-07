-- Q03 — New-account first-order risk snapshot: who starts big, verifies badly,
-- and buys from far away?
-- Cutoff: @as_of (see Q01); unset means the end of observation.
-- Rows: each account's first order attempt (any processor result, as FP-2 R04
-- counts it) placed when the account was under 7 days old, that is large
-- (above 0.8 × the category reference), failed address verification, or came
-- from an IP country other than the card's issuing country.
-- amount_vs_category_p95 compares the attempt with the 95th percentile of the
-- processor-approved amounts in its merchant category known at @as_of (the
-- largest amount whose percent rank is at most 0.95). That is the analyst's
-- reference at the cutoff, not R04's, which uses only amounts approved before
-- the attempt.
-- Read: sorting by amount_vs_category_p95 puts "new account, top-of-distribution
-- order" first, the stolen-card and never-pay entry shape. These are context
-- (FP-2 §6.5(a), (c)): AVS failures and large first orders are common among
-- legitimate customers and never grounds alone.
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH attempts AS (
  SELECT o.*, m.category,
         ROW_NUMBER() OVER (PARTITION BY o.user_id ORDER BY o.known_at, o.event_id) AS nth
  FROM order_attempts o
  JOIN merchants m ON m.merchant_id = o.merchant_id
  WHERE o.known_at <= @as_of
),
category_p95 AS (
  SELECT category, MAX(CASE WHEN pct <= 0.95 THEN amount_cents END) AS p95_cents
  FROM (
    SELECT category, amount_cents,
           PERCENT_RANK() OVER (PARTITION BY category ORDER BY amount_cents) AS pct
    FROM attempts
    WHERE processor_result = 'approved'
  ) ranked
  GROUP BY category
)
SELECT f.order_id, f.user_id, f.known_at AS attempted_at,
       CAST(f.amount_cents / 100 AS DECIMAL(12, 2)) AS amount_usd, f.category,
       f.processor_result,
       ROUND(f.amount_cents / p.p95_cents, 2) AS amount_vs_category_p95,
       TIMESTAMPDIFF(HOUR, u.created_at, f.known_at) AS account_age_hours,
       f.avs_result, f.cvv_result, c.bin_country, f.ip_country,
       (c.bin_country <> f.ip_country) AS bin_ip_mismatch
FROM attempts f
JOIN accounts u ON u.user_id = f.user_id
JOIN cards c ON c.card_id = f.card_id
LEFT JOIN category_p95 p ON p.category = f.category
WHERE f.nth = 1
  AND f.known_at < u.created_at + INTERVAL 7 DAY
  AND (f.amount_cents > p.p95_cents * 0.8 OR f.avs_result = 'N'
       OR c.bin_country <> f.ip_country)
ORDER BY amount_vs_category_p95 DESC, f.order_id
LIMIT 200;
