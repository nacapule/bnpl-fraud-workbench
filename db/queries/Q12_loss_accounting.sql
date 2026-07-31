-- Q12 — Loss accounting: written-off principal by month, category, and account
-- age band — the "know your losses" management view.
-- Read: loss = plan principal minus everything actually collected (down payment
-- + successful installment payments). The account-age split is the credit-vs-
-- fraud lens at portfolio level: losses concentrated in <30d accounts are an
-- acquisition-fraud problem; losses spread across tenured accounts are credit.
WITH collected AS (
  SELECT p.plan_id, p.order_id, p.principal,
         COALESCE(SUM(CASE WHEN pay.result = 'success' THEN pay.amount END), 0) AS collected
  FROM plans p
  LEFT JOIN payments pay ON pay.plan_id = p.plan_id
  GROUP BY p.plan_id, p.order_id, p.principal
),
loss_plans AS (
  SELECT c.plan_id, c.order_id, c.principal, c.collected,
         c.principal - c.collected AS loss
  FROM collected c
  WHERE EXISTS (SELECT 1 FROM installments i
                WHERE i.plan_id = c.plan_id AND i.outcome = 'written_off')
)
SELECT DATE_FORMAT(o.ts, '%Y-%m') AS order_month,
       m.category,
       CASE WHEN o.ts < u.signup_ts + INTERVAL 30 DAY THEN 'lt_30d'
            WHEN o.ts < u.signup_ts + INTERVAL 180 DAY THEN '30_180d'
            ELSE 'gt_180d' END AS account_age_band,
       COUNT(*) AS n_written_off_plans,
       ROUND(SUM(lp.loss), 2) AS written_off_usd,
       ROUND(SUM(lp.collected) / NULLIF(SUM(lp.principal), 0), 3) AS recovery_rate
FROM loss_plans lp
JOIN orders o ON o.order_id = lp.order_id
JOIN users u ON u.user_id = o.user_id
JOIN merchants m ON m.merchant_id = o.merchant_id
GROUP BY order_month, m.category, account_age_band
HAVING SUM(lp.loss) > 500
ORDER BY order_month, written_off_usd DESC
LIMIT 300;
