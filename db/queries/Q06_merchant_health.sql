-- Q06 — Merchant health: dispute and repayment-failure trajectory per merchant,
-- from operational signals only (no labels).
-- Cutoff: @as_of (see Q01); unset means the end of observation. Every source has
-- an upper bound there: merchants onboarded by @as_of, processor-approved orders
-- placed by then, disputes the platform knew of by then, failed payment attempts
-- and write-offs known by then. @min_orders (default 20) hides merchants with
-- fewer orders.
-- Rates are order-cohort rates: dispute_rate_all is the share of the merchant's
-- orders with a dispute opened by @as_of; dispute_rate_90d the same share among
-- orders placed in the 90 days to @as_of (young orders have had little time to
-- be disputed, so this rate is a floor); failed_installment_rate the share of
-- orders with an installment unpaid at @as_of after a failed attempt or a
-- write-off; new_buyer_share the share of orders (not of GMV) from accounts
-- under 30 days old. closed_at shows a closure known by @as_of.
-- Read: the bust-out shape is a young merchant whose recent dispute and failure
-- rates rise while its ticket drifts up and its buyers get newer. Each signal
-- alone has benign explanations; merchant evidence belongs to merchant risk, not
-- to order decisions (FP-2 §6.5(e)).
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);
SET @min_orders = COALESCE(@min_orders, 20);

WITH approved AS (
  SELECT o.order_id, o.merchant_id, o.known_at AS ordered_at, o.amount_cents,
         (u.created_at > o.known_at - INTERVAL 30 DAY) AS buyer_is_new
  FROM order_attempts o
  JOIN accounts u ON u.user_id = o.user_id
  WHERE o.processor_result = 'approved' AND o.known_at <= @as_of
),
disputed AS (
  SELECT DISTINCT order_id FROM dispute_openings WHERE known_at <= @as_of
),
paid_in AS (
  SELECT plan_id, seq, SUM(amount_cents) AS cents
  FROM payment_attempts
  WHERE result = 'success' AND seq >= 1 AND known_at <= @as_of
  GROUP BY plan_id, seq
),
reversed AS (
  SELECT a.plan_id, a.seq, SUM(r.amount_cents) AS cents
  FROM payment_reversals r
  JOIN payment_attempts a ON a.event_id = r.payment_event_id
  -- a reversal counts once the payment it reverses is known too
  WHERE a.seq >= 1 AND a.result = 'success' AND a.known_at <= @as_of AND r.known_at <= @as_of
  GROUP BY a.plan_id, a.seq
),
failed_attempts AS (
  SELECT DISTINCT plan_id, seq FROM payment_attempts
  WHERE result = 'failed' AND seq >= 1 AND known_at <= @as_of
),
written_off AS (
  SELECT DISTINCT plan_id FROM plan_writeoffs WHERE known_at <= @as_of
),
failed_orders AS (
  SELECT DISTINCT p.order_id
  FROM installment_schedule s
  JOIN plans p ON p.plan_id = s.plan_id
  LEFT JOIN paid_in pi ON pi.plan_id = s.plan_id AND pi.seq = s.seq
  LEFT JOIN reversed rv ON rv.plan_id = s.plan_id AND rv.seq = s.seq
  LEFT JOIN failed_attempts fa ON fa.plan_id = s.plan_id AND fa.seq = s.seq
  LEFT JOIN written_off w ON w.plan_id = s.plan_id
  WHERE s.seq >= 1
    AND COALESCE(pi.cents, 0) - COALESCE(rv.cents, 0) < s.amount_cents
    AND (fa.plan_id IS NOT NULL OR w.plan_id IS NOT NULL)
),
per_order AS (
  SELECT a.*,
         (d.order_id IS NOT NULL) AS has_dispute,
         (f.order_id IS NOT NULL) AS has_failed_installment,
         (a.ordered_at >= @as_of - INTERVAL 90 DAY) AS recent
  FROM approved a
  LEFT JOIN disputed d ON d.order_id = a.order_id
  LEFT JOIN failed_orders f ON f.order_id = a.order_id
),
merchant_window AS (
  SELECT merchant_id,
         COUNT(*) AS orders_all,
         CAST(SUM(recent) AS SIGNED) AS orders_90d,
         ROUND(AVG(has_dispute), 4) AS dispute_rate_all,
         ROUND(AVG(CASE WHEN recent THEN has_dispute END), 4) AS dispute_rate_90d,
         ROUND(AVG(has_failed_installment), 4) AS failed_installment_rate,
         ROUND(AVG(amount_cents) / 100, 2) AS avg_ticket_all_usd,
         ROUND(AVG(CASE WHEN recent THEN amount_cents END) / 100, 2) AS avg_ticket_90d_usd,
         ROUND(AVG(buyer_is_new), 3) AS new_buyer_share
  FROM per_order
  GROUP BY merchant_id
)
SELECT w.merchant_id, m.name, m.category, m.risk_tier, m.created_at AS onboarded_at,
       CASE WHEN m.closed_at <= @as_of THEN m.closed_at END AS closed_at,
       w.orders_all, w.orders_90d, w.dispute_rate_all, w.dispute_rate_90d,
       w.failed_installment_rate, w.avg_ticket_all_usd, w.avg_ticket_90d_usd,
       ROUND(w.avg_ticket_90d_usd / NULLIF(w.avg_ticket_all_usd, 0), 2) AS ticket_drift,
       w.new_buyer_share
FROM merchant_window w
JOIN merchants m ON m.merchant_id = w.merchant_id
WHERE m.created_at <= @as_of AND w.orders_all >= @min_orders
ORDER BY COALESCE(w.dispute_rate_90d, w.dispute_rate_all) DESC, w.failed_installment_rate DESC,
         w.merchant_id
LIMIT 60;
