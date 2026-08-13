-- Q06 — Merchant health: chargeback and failure trajectory per merchant,
-- label-free (uses only observable operational signals).
-- Read: the bust-out shape is a young merchant whose recent-90d chargeback rate
-- and installment-failure rate both explode while avg ticket drifts up and GMV
-- concentrates in young accounts. Any single metric alone has benign
-- explanations; the conjunction is the alarm.
-- Set this date to the observation cutoff before running the query.
SET @as_of = '2026-06-30';

WITH per_order AS (
  SELECT o.order_id, o.merchant_id, o.ts, o.amount, o.user_id,
         (u.signup_ts > o.ts - INTERVAL 30 DAY) AS buyer_is_new,
         EXISTS (SELECT 1 FROM chargebacks cb WHERE cb.order_id = o.order_id) AS has_cb,
         (SELECT COUNT(*) FROM installments i JOIN plans p ON p.plan_id = i.plan_id
          WHERE p.order_id = o.order_id AND i.outcome IN ('failed', 'written_off')) AS n_failed_inst
  FROM orders o
  JOIN users u ON u.user_id = o.user_id
  WHERE o.status = 'approved'
),
merch_window AS (
  SELECT merchant_id,
         COUNT(*) AS orders_all,
         SUM(ts >= @as_of - INTERVAL 90 DAY) AS orders_90d,
         ROUND(AVG(has_cb), 4) AS cb_rate_all,
         ROUND(AVG(CASE WHEN ts >= @as_of - INTERVAL 90 DAY THEN has_cb END), 4)
             AS cb_rate_90d,
         ROUND(AVG(n_failed_inst > 0), 4) AS inst_fail_rate,
         ROUND(AVG(amount), 2) AS avg_ticket_all,
         ROUND(AVG(CASE WHEN ts >= @as_of - INTERVAL 90 DAY THEN amount END), 2)
             AS avg_ticket_90d,
         ROUND(AVG(buyer_is_new), 3) AS new_buyer_share
  FROM per_order
  GROUP BY merchant_id
)
SELECT m.merchant_id, mch.name, mch.category, mch.risk_tier, mch.onboarded_ts,
       m.orders_all, m.orders_90d, m.cb_rate_all, m.cb_rate_90d,
       m.inst_fail_rate, m.avg_ticket_all, m.avg_ticket_90d,
       ROUND(m.avg_ticket_90d / NULLIF(m.avg_ticket_all, 0), 2) AS ticket_drift,
       m.new_buyer_share
FROM merch_window m
JOIN merchants mch ON mch.merchant_id = m.merchant_id
WHERE m.orders_all >= 20
ORDER BY COALESCE(m.cb_rate_90d, m.cb_rate_all) DESC, m.inst_fail_rate DESC
LIMIT 60;
