-- Q01 — Order/amount velocity: which users and devices are transacting faster
-- than organic shopping rhythm?
-- Read: top rows are burst candidates. n_1h > 2 or n_24h > 3 per user is far
-- outside the benign base rate here; device-level velocity catches multi-account
-- activity that user-level windows miss.
WITH w AS (
  SELECT
    o.order_id, o.user_id, o.device_id, o.ts, o.amount,
    COUNT(*)        OVER (PARTITION BY o.user_id  ORDER BY o.ts RANGE BETWEEN INTERVAL 1 HOUR  PRECEDING AND CURRENT ROW) AS user_n_1h,
    COUNT(*)        OVER (PARTITION BY o.user_id  ORDER BY o.ts RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW) AS user_n_24h,
    COUNT(*)        OVER (PARTITION BY o.user_id  ORDER BY o.ts RANGE BETWEEN INTERVAL 7 DAY   PRECEDING AND CURRENT ROW) AS user_n_7d,
    SUM(o.amount)   OVER (PARTITION BY o.user_id  ORDER BY o.ts RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW) AS user_amt_24h,
    COUNT(*)        OVER (PARTITION BY o.device_id ORDER BY o.ts RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW) AS device_n_24h
  FROM orders o
  WHERE o.status = 'approved'
)
SELECT order_id, user_id, device_id, ts, amount,
       user_n_1h, user_n_24h, user_n_7d, ROUND(user_amt_24h, 2) AS user_amt_24h,
       device_n_24h
FROM w
WHERE user_n_24h > 3 OR device_n_24h > 5 OR user_n_1h > 2
ORDER BY GREATEST(user_n_24h, device_n_24h) DESC, user_amt_24h DESC
LIMIT 200;
