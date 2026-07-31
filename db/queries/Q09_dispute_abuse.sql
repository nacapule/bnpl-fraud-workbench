-- Q09 — Friendly-fraud / INR abuse: accounts that repeatedly dispute delivered
-- orders as "item not received" while their repayment stays healthy.
-- Read: a genuine non-delivery victim disputes once and often churns; the abuse
-- shape is ≥2 INR disputes with ≥80% of installments paid — they keep the goods,
-- keep the credit line, and let the bank claw back the merchant side. These are
-- account-remedy cases (FP-1 §5), not order declines.
WITH user_inr AS (
  SELECT o.user_id,
         COUNT(*) AS n_inr,
         SUM(o.amount) AS disputed_value,
         MIN(cb.opened_ts) AS first_dispute,
         MAX(cb.opened_ts) AS last_dispute
  FROM chargebacks cb
  JOIN orders o ON o.order_id = cb.order_id
  WHERE cb.reason = 'inr'
  GROUP BY o.user_id
  HAVING COUNT(*) >= 2
),
repay AS (
  SELECT o.user_id,
         SUM(i.outcome IN ('paid', 'late')) / COUNT(*) AS paid_share,
         COUNT(*) AS n_installments
  FROM installments i
  JOIN plans p ON p.plan_id = i.plan_id
  JOIN orders o ON o.order_id = p.order_id
  GROUP BY o.user_id
)
SELECT ui.user_id, ui.n_inr, ROUND(ui.disputed_value, 2) AS disputed_value,
       ui.first_dispute, ui.last_dispute,
       ROUND(r.paid_share, 3) AS paid_share, r.n_installments
FROM user_inr ui
JOIN repay r ON r.user_id = ui.user_id
WHERE r.paid_share >= 0.8
ORDER BY ui.n_inr DESC, ui.disputed_value DESC
LIMIT 100;
