-- Q10 — Impossible geo-velocity: consecutive order attempts on one account from
-- countries farther apart than anyone can travel in between.
-- Cutoff: @as_of (see Q01); unset means the end of observation.
-- A row pairs an order attempt with the account's previous attempt, in the
-- world's event order, when the IP countries differ, the earlier one is under 12
-- hours before, and the implied speed between the countries' centroids exceeds
-- 900 km/h (FP-2 R11). Centroids are approximate and embedded below; they are
-- the as-of context's list (core/asof.py), and every IP country in the world
-- must have one.
-- Gaps under 72 seconds count as 72 seconds.
-- Read: one of the two attempts is probably not the customer, or the customer
-- is behind a proxy. A traveller moves between sessions at airline speed, not
-- within the hour; the check settles it (FP-2 §6.5(b)).
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH centroids AS (
  SELECT 'US' AS cc, 39.8 AS lat, -98.6 AS lon UNION ALL
  SELECT 'CA', 56.1, -106.3 UNION ALL SELECT 'MX', 23.6, -102.6 UNION ALL SELECT 'GB', 54.0, -2.9 UNION ALL
  SELECT 'IE', 53.4, -8.2 UNION ALL SELECT 'DE', 51.2, 10.4 UNION ALL SELECT 'FR', 46.6, 2.4 UNION ALL
  SELECT 'ES', 40.3, -3.7 UNION ALL SELECT 'PT', 39.6, -8.0 UNION ALL SELECT 'IT', 42.8, 12.6 UNION ALL
  SELECT 'NL', 52.2, 5.6 UNION ALL SELECT 'PL', 52.1, 19.4 UNION ALL SELECT 'RO', 45.9, 24.9 UNION ALL
  SELECT 'UA', 49.0, 31.4 UNION ALL SELECT 'RU', 61.5, 105.3 UNION ALL SELECT 'TR', 39.1, 35.2 UNION ALL
  SELECT 'BR', -10.8, -52.9 UNION ALL SELECT 'AR', -35.4, -65.2 UNION ALL SELECT 'CO', 4.1, -72.9 UNION ALL
  SELECT 'IN', 22.9, 79.6 UNION ALL SELECT 'PK', 29.9, 69.4 UNION ALL SELECT 'NG', 9.6, 8.1 UNION ALL
  SELECT 'ZA', -29.0, 25.1 UNION ALL SELECT 'EG', 26.6, 29.8 UNION ALL SELECT 'VN', 16.6, 106.3 UNION ALL
  SELECT 'CN', 36.5, 103.8 UNION ALL SELECT 'JP', 36.6, 138.0 UNION ALL SELECT 'KR', 36.5, 127.8 UNION ALL
  SELECT 'PH', 12.9, 122.9 UNION ALL SELECT 'ID', -2.2, 117.3 UNION ALL SELECT 'AU', -25.7, 134.5
),
pairs AS (
  SELECT user_id, order_id, known_at, ip_country,
         LAG(order_id) OVER by_account AS previous_order_id,
         LAG(known_at) OVER by_account AS previous_at,
         LAG(ip_country) OVER by_account AS previous_country
  FROM order_attempts
  WHERE known_at <= @as_of
  WINDOW by_account AS (PARTITION BY user_id ORDER BY known_at, event_id)
),
distances AS (
  SELECT p.*, TIMESTAMPDIFF(SECOND, p.previous_at, p.known_at) AS gap_seconds,
         6371 * 2 * ASIN(SQRT(
           POW(SIN(RADIANS(c2.lat - c1.lat) / 2), 2) +
           COS(RADIANS(c1.lat)) * COS(RADIANS(c2.lat)) *
           POW(SIN(RADIANS(c2.lon - c1.lon) / 2), 2))) AS km
  FROM pairs p
  JOIN centroids c1 ON c1.cc = p.previous_country
  JOIN centroids c2 ON c2.cc = p.ip_country
  WHERE p.ip_country <> p.previous_country
    AND TIMESTAMPDIFF(SECOND, p.previous_at, p.known_at) < 12 * 3600
)
SELECT user_id, previous_order_id, previous_at, previous_country,
       order_id, known_at AS attempted_at, ip_country,
       ROUND(gap_seconds / 3600, 2) AS gap_hours, ROUND(km, 0) AS km,
       ROUND(km / GREATEST(gap_seconds / 3600, 0.02), 0) AS implied_kmh
FROM distances
WHERE km / GREATEST(gap_seconds / 3600, 0.02) > 900
ORDER BY implied_kmh DESC, order_id
LIMIT 200;
