-- Q04 — Repayment cohort curves: how does repayment behave by signup month and
-- first-order size, and where does never-pay concentrate?
-- Read: never_pay_share = accounts whose plans collected zero successful
-- installments. Benign hardship shows partial payment (partial_share); a cohort
-- cell with high never-pay AND low partial-pay is fraud concentration, not an
-- economy-wide credit event. First-order-size bands separate "started huge and
-- vanished" from ordinary defaults.
WITH plan_user AS (
  SELECT p.plan_id, o.user_id, o.ts, o.amount,
         DATE_FORMAT(u.signup_ts, '%Y-%m') AS signup_month,
         CASE WHEN o.amount < 100 THEN 'a_under_100'
              WHEN o.amount < 300 THEN 'b_100_300'
              WHEN o.amount < 700 THEN 'c_300_700'
              ELSE 'd_over_700' END AS first_order_band,
         ROW_NUMBER() OVER (PARTITION BY o.user_id ORDER BY o.ts) AS rn
  FROM plans p
  JOIN orders o ON o.order_id = p.order_id
  JOIN users u ON u.user_id = o.user_id
  WHERE o.status = 'approved'
),
inst_stats AS (
  SELECT i.plan_id,
         SUM(i.outcome IN ('paid', 'late')) AS n_paid,
         COUNT(*) AS n_due
  FROM installments i
  GROUP BY i.plan_id
)
SELECT pu.signup_month, pu.first_order_band,
       COUNT(*) AS n_first_orders,
       ROUND(AVG(s.n_paid = s.n_due), 3) AS fully_paid_share,
       ROUND(AVG(s.n_paid > 0 AND s.n_paid < s.n_due), 3) AS partial_share,
       ROUND(AVG(s.n_paid = 0), 3) AS never_pay_share
FROM plan_user pu
JOIN inst_stats s ON s.plan_id = pu.plan_id
WHERE pu.rn = 1
GROUP BY pu.signup_month, pu.first_order_band
HAVING COUNT(*) >= 20
ORDER BY pu.signup_month, pu.first_order_band;
