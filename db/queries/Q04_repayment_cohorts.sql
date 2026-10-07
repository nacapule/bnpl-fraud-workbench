-- Q04 — Matured-cohort repayment: once a cohort's installments have all fallen
-- due, how much of it repaid, by first-order month and first-order size?
-- Cutoff: @as_of (see Q01); unset means the end of observation. @min_cohort
-- (default 20) hides cells with fewer first plans.
-- Cohort: each account's first plan (its first processor-approved order),
-- grouped by the month of that order and its amount band. A cohort is shown only
-- once it has matured: every first plan in it has its last installment due at
-- least 30 days before @as_of (FP-2 §3.2: a missed installment becomes a default
-- 30 days after its due date). A younger cohort would count installments not yet
-- due as unpaid.
-- Shares, from payments and reversals known by @as_of (an installment is paid
-- when its standing cents, successful payments less their reversals, reach the
-- scheduled amount): fully_paid_share, every installment paid; partial_share,
-- something paid after the checkout payment but not every installment;
-- zero_effort_share, nothing standing after the checkout payment (the FP-2 §8.2
-- zero-effort default).
-- Read: a zero-effort default is not never-pay by itself (FP-2 §8.3, §8.4); a
-- cell with many zero-effort plans and few partial ones points at first-party
-- fraud to investigate, while partial payment looks like hardship.
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);
SET @min_cohort = COALESCE(@min_cohort, 20);

WITH first_plans AS (
  SELECT plan_id, order_month, first_order_band
  FROM (
    SELECT p.plan_id, DATE_FORMAT(o.known_at, '%Y-%m') AS order_month,
           CASE WHEN o.amount_cents < 10000 THEN 'a_under_100'
                WHEN o.amount_cents < 30000 THEN 'b_100_300'
                WHEN o.amount_cents < 70000 THEN 'c_300_700'
                ELSE 'd_over_700' END AS first_order_band,
           ROW_NUMBER() OVER (PARTITION BY o.user_id ORDER BY o.known_at, o.event_id) AS nth
    FROM plans p
    JOIN order_attempts o ON o.order_id = p.order_id
    WHERE o.known_at <= @as_of
  ) ranked
  WHERE nth = 1
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
plan_state AS (
  SELECT s.plan_id,
         COUNT(*) AS n_installments,
         SUM(COALESCE(pi.cents, 0) - COALESCE(rv.cents, 0) >= s.amount_cents) AS n_paid,
         SUM(COALESCE(pi.cents, 0) - COALESCE(rv.cents, 0)) AS standing_cents,
         MAX(s.due_at) AS last_due_at
  FROM installment_schedule s
  JOIN first_plans f ON f.plan_id = s.plan_id
  LEFT JOIN paid_in pi ON pi.plan_id = s.plan_id AND pi.seq = s.seq
  LEFT JOIN reversed rv ON rv.plan_id = s.plan_id AND rv.seq = s.seq
  WHERE s.seq >= 1
  GROUP BY s.plan_id
),
matured_months AS (
  SELECT f.order_month
  FROM first_plans f
  JOIN plan_state ps ON ps.plan_id = f.plan_id
  GROUP BY f.order_month
  HAVING MAX(ps.last_due_at) + INTERVAL 30 DAY <= @as_of
)
SELECT f.order_month, f.first_order_band,
       COUNT(*) AS n_first_plans,
       ROUND(AVG(ps.n_paid = ps.n_installments), 3) AS fully_paid_share,
       ROUND(AVG(ps.standing_cents > 0 AND ps.n_paid < ps.n_installments), 3) AS partial_share,
       ROUND(AVG(ps.standing_cents = 0), 3) AS zero_effort_share
FROM first_plans f
JOIN plan_state ps ON ps.plan_id = f.plan_id
JOIN matured_months mm ON mm.order_month = f.order_month
GROUP BY f.order_month, f.first_order_band
HAVING COUNT(*) >= @min_cohort
ORDER BY f.order_month, f.first_order_band;
