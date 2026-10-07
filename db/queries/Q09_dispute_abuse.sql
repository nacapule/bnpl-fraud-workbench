-- Q09 — Item-not-received abuse: which accounts keep claiming that orders never
-- arrived, and how many of those claims were rejected on delivered orders?
-- Cutoff: @as_of (see Q01); unset means the end of observation. Disputes,
-- resolutions, carrier deliveries and payments count only when known by then.
-- A row is an account with two or more item_not_received disputes opened on its
-- orders (FP-2 R09's count). claims_on_delivered counts those whose order has a
-- carrier-confirmed delivery; claims_rejected_on_delivered those also resolved
-- against the customer ('won' for the platform). Two rejected claims on
-- delivered orders are the FP-2 §9(d) item-not-received abuse determination,
-- which settles later orders on the account (§6.3(c)). claims_pending have no
-- resolution yet. installments_paid_share is the share of the account's
-- installments due by @as_of that are paid in full (payments less reversals).
-- Read: R09 alone is context, never grounds for a hold (FP-2 §6.2); a genuine
-- non-delivery victim's claim is upheld ('lost'), or concerns an order with no
-- delivery. Repayment describes the account holder, not the claim (§6.5(f)).
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH claims AS (
  SELECT o.user_id, d.dispute_id, d.known_at AS opened_at, d.amount_cents,
         r.outcome,
         EXISTS (SELECT 1 FROM deliveries v
                 WHERE v.order_id = d.order_id AND v.known_at <= @as_of) AS delivered
  FROM dispute_openings d
  JOIN order_attempts o ON o.order_id = d.order_id
  LEFT JOIN dispute_resolutions r ON r.dispute_id = d.dispute_id AND r.known_at <= @as_of
  WHERE d.reason = 'item_not_received' AND d.known_at <= @as_of
),
user_claims AS (
  SELECT user_id,
         COUNT(*) AS inr_disputes_opened,
         CAST(SUM(delivered) AS SIGNED) AS claims_on_delivered,
         CAST(SUM(delivered AND outcome = 'won') AS SIGNED) AS claims_rejected_on_delivered,
         CAST(SUM(outcome IS NULL) AS SIGNED) AS claims_pending,
         SUM(amount_cents) AS disputed_cents,
         MIN(opened_at) AS first_opened_at, MAX(opened_at) AS last_opened_at
  FROM claims
  GROUP BY user_id
  HAVING COUNT(*) >= 2
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
  WHERE a.seq >= 1 AND r.known_at <= @as_of
  GROUP BY a.plan_id, a.seq
),
repayment AS (
  SELECT o.user_id, COUNT(*) AS installments_due,
         SUM(COALESCE(pi.cents, 0) - COALESCE(rv.cents, 0) >= s.amount_cents) AS installments_paid
  FROM installment_schedule s
  JOIN plans p ON p.plan_id = s.plan_id
  JOIN order_attempts o ON o.order_id = p.order_id
  JOIN user_claims uc ON uc.user_id = o.user_id
  LEFT JOIN paid_in pi ON pi.plan_id = s.plan_id AND pi.seq = s.seq
  LEFT JOIN reversed rv ON rv.plan_id = s.plan_id AND rv.seq = s.seq
  WHERE s.seq >= 1 AND s.due_at <= @as_of AND o.known_at <= @as_of
  GROUP BY o.user_id
)
SELECT uc.user_id, uc.inr_disputes_opened, uc.claims_on_delivered,
       uc.claims_rejected_on_delivered, uc.claims_pending,
       CAST(uc.disputed_cents / 100 AS DECIMAL(12, 2)) AS disputed_usd,
       uc.first_opened_at, uc.last_opened_at,
       COALESCE(rp.installments_due, 0) AS installments_due,
       ROUND(rp.installments_paid / rp.installments_due, 3) AS installments_paid_share
FROM user_claims uc
LEFT JOIN repayment rp ON rp.user_id = uc.user_id
ORDER BY uc.claims_rejected_on_delivered DESC, uc.inr_disputes_opened DESC, uc.disputed_cents DESC,
         uc.user_id
LIMIT 100;
