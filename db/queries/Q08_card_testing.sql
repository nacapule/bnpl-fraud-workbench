-- Q08 — Card testing: decline bursts followed by an approval on the same
-- card/device, plus devices cycling many distinct cards.
-- Read: fraudsters validate stolen credentials with cheap declined attempts,
-- then spend. A burst of ≥3 declines inside an hour that converts to an
-- approval within 24h is the classic shape. Declines use fixed clock-hour
-- buckets, not a sliding one-hour window; distinct_cards_per_device catches
-- the inventory-of-cards variant even when approvals land elsewhere.
WITH declines AS (
  SELECT o.card_id, o.device_id,
         MIN(o.ts) AS burst_start, MAX(o.ts) AS burst_end,
         COUNT(*) AS n_declines
  FROM orders o
  WHERE o.status = 'declined'
  GROUP BY o.card_id, o.device_id,
           FLOOR(UNIX_TIMESTAMP(o.ts) / 3600)  -- fixed clock-hour buckets
  HAVING COUNT(*) >= 3
),
converted AS (
  SELECT d.card_id, d.device_id, d.burst_start, d.burst_end, d.n_declines,
         MIN(a.ts) AS first_approval_ts,
         MAX(a.amount) AS approved_amount
  FROM declines d
  JOIN orders a ON a.card_id = d.card_id
               AND a.device_id = d.device_id
               AND a.status = 'approved'
               AND a.ts BETWEEN d.burst_end AND d.burst_end + INTERVAL 24 HOUR
  GROUP BY d.card_id, d.device_id, d.burst_start, d.burst_end, d.n_declines
)
SELECT c.*, cards.bin_country,
       (SELECT COUNT(DISTINCT o2.card_id) FROM orders o2
        WHERE o2.device_id = c.device_id
          AND o2.ts BETWEEN c.burst_start - INTERVAL 24 HOUR AND c.burst_start + INTERVAL 24 HOUR)
         AS distinct_cards_on_device_24h
FROM converted c
JOIN cards ON cards.card_id = c.card_id
ORDER BY c.n_declines DESC, approved_amount DESC
LIMIT 100;
