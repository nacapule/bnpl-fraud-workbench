-- Q03 — New-account first-order risk snapshot: who starts big, verifies badly,
-- and buys from far away?
-- Read: each row is a first-ever order by an account < 7 days old. Sorting by
-- amount_vs_p95 puts "new account, top-of-distribution order" first — the
-- stolen-card and never-pay entry shape. AVS/CVV and BIN-vs-IP mismatches are
-- corroboration, not verdicts (benign mismatch noise ≈ 3%).
WITH firsts AS (
  SELECT o.*, ROW_NUMBER() OVER (PARTITION BY o.user_id ORDER BY o.ts) AS rn
  FROM orders o
  WHERE o.status = 'approved'
),
cat_p95 AS (
  SELECT m.category,
         MAX(CASE WHEN q.pct <= 0.95 THEN q.amount END) AS p95
  FROM (
    SELECT o.merchant_id, o.amount,
           PERCENT_RANK() OVER (PARTITION BY m2.category ORDER BY o.amount) AS pct
    FROM orders o JOIN merchants m2 ON m2.merchant_id = o.merchant_id
    WHERE o.status = 'approved'
  ) q JOIN merchants m ON m.merchant_id = q.merchant_id
  GROUP BY m.category
)
SELECT f.order_id, f.user_id, f.ts, f.amount, m.category,
       ROUND(f.amount / p.p95, 2) AS amount_vs_p95,
       TIMESTAMPDIFF(HOUR, u.signup_ts, f.ts) AS account_age_hours,
       f.avs_result, f.cvv_result,
       c.bin_country, f.ip_country,
       (c.bin_country <> f.ip_country) AS bin_ip_mismatch
FROM firsts f
JOIN users u ON u.user_id = f.user_id
JOIN merchants m ON m.merchant_id = f.merchant_id
JOIN cards c ON c.card_id = f.card_id
JOIN cat_p95 p ON p.category = m.category
WHERE f.rn = 1
  AND f.ts < u.signup_ts + INTERVAL 7 DAY
  AND (f.amount > p.p95 * 0.8 OR f.avs_result <> 'Y' OR c.bin_country <> f.ip_country)
ORDER BY amount_vs_p95 DESC
LIMIT 200;
