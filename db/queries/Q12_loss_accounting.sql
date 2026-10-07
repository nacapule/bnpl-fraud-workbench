-- Q12 — Loss accounting: where does the platform lose money, by order month,
-- merchant category and account age at the order?
-- Cutoff: @as_of (see Q01); unset means the end of observation. Only cash
-- events known by then count, so recent order months are still incomplete.
-- Source: the cash ledger (cash_events, core/ledger.py), signed from the
-- platform's side in integer cents. Each order's cash falls in the month of the
-- order. loss_usd = −(sum of the cash events), the ledger's definition, so a
-- cell of profitable orders shows a negative loss. Its parts, signed:
-- merchant_payouts_usd (settlements net of the merchant fee, and promotion
-- funding), collections_usd (customer payments less reversals and refunds),
-- disputes_usd (dispute debits and fees, won credits, merchant recourse) and
-- recoveries_usd; loss_usd = −(the four parts). Write-off is a status, not cash:
-- written_off_usd is the outstanding balance of plans written off by @as_of.
-- Read: losses concentrated in accounts under 30 days old are acquisition fraud
-- or first-party default; losses spread across tenured accounts look like
-- credit (FP-2 §8).
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH per_order AS (
  SELECT order_id,
         SUM(amount_cents) AS net_cents,
         SUM(CASE WHEN kind IN ('merchant_settlement', 'promotion_funding')
                  THEN amount_cents ELSE 0 END) AS payout_cents,
         SUM(CASE WHEN kind IN ('customer_payment', 'payment_reversal', 'refund')
                  THEN amount_cents ELSE 0 END) AS collection_cents,
         SUM(CASE WHEN kind IN ('dispute_debit', 'dispute_fee', 'dispute_won_credit',
                                'merchant_recourse')
                  THEN amount_cents ELSE 0 END) AS dispute_cents,
         SUM(CASE WHEN kind = 'recovery' THEN amount_cents ELSE 0 END) AS recovery_cents
  FROM cash_events
  WHERE known_at <= @as_of
  GROUP BY order_id
),
written_off AS (
  SELECT p.order_id, SUM(w.outstanding_cents) AS outstanding_cents
  FROM plan_writeoffs w
  JOIN plans p ON p.plan_id = w.plan_id
  WHERE w.known_at <= @as_of
  GROUP BY p.order_id
)
SELECT DATE_FORMAT(o.occurred_at, '%Y-%m') AS order_month, m.category,
       CASE WHEN o.occurred_at < u.created_at + INTERVAL 30 DAY THEN 'lt_30d'
            WHEN o.occurred_at < u.created_at + INTERVAL 180 DAY THEN '30_180d'
            ELSE 'gt_180d' END AS account_age_band,
       COUNT(*) AS n_orders,
       CAST(SUM(po.net_cents < 0) AS SIGNED) AS n_loss_orders,
       COUNT(w.order_id) AS n_written_off_plans,
       CAST(SUM(po.payout_cents) / 100 AS DECIMAL(14, 2)) AS merchant_payouts_usd,
       CAST(SUM(po.collection_cents) / 100 AS DECIMAL(14, 2)) AS collections_usd,
       CAST(SUM(po.dispute_cents) / 100 AS DECIMAL(14, 2)) AS disputes_usd,
       CAST(SUM(po.recovery_cents) / 100 AS DECIMAL(14, 2)) AS recoveries_usd,
       CAST(-SUM(po.net_cents) / 100 AS DECIMAL(14, 2)) AS loss_usd,
       CAST(COALESCE(SUM(w.outstanding_cents), 0) / 100 AS DECIMAL(14, 2)) AS written_off_usd
FROM per_order po
JOIN order_attempts o ON o.order_id = po.order_id
JOIN accounts u ON u.user_id = o.user_id
JOIN merchants m ON m.merchant_id = o.merchant_id
LEFT JOIN written_off w ON w.order_id = po.order_id
GROUP BY order_month, m.category, account_age_band
ORDER BY order_month, loss_usd DESC, m.category, account_age_band;
