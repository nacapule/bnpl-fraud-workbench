-- Q10 — Impossible geo-velocity: consecutive events from countries farther
-- apart than travel allows, using approximate country centroids (embedded CTE).
-- Read: implied_kmh > 900 means two actors (or proxying) — one of them is not
-- the customer. Travelers move at airplane speed BETWEEN sessions, not within
-- an hour; the mimic check is whether a plausible flight window separates the
-- sightings (FP-1 §6.1).
WITH centroids AS (
  SELECT 'US' AS cc, 39.8 AS lat, -98.6 AS lon UNION ALL
  SELECT 'CA', 56.1, -106.3 UNION ALL SELECT 'GB', 54.0, -2.9 UNION ALL
  SELECT 'DE', 51.2, 10.4 UNION ALL SELECT 'FR', 46.6, 2.4 UNION ALL
  SELECT 'ES', 40.3, -3.7 UNION ALL SELECT 'BR', -10.8, -52.9 UNION ALL
  SELECT 'IN', 22.9, 79.6 UNION ALL SELECT 'NG', 9.6, 8.1 UNION ALL
  SELECT 'RO', 45.9, 24.9 UNION ALL SELECT 'VN', 16.6, 106.3 UNION ALL
  SELECT 'CN', 36.5, 103.8 UNION ALL SELECT 'RU', 61.5, 105.3 UNION ALL
  SELECT 'ID', -2.2, 117.3
),
sightings AS (
  SELECT user_id, ts, ip_country AS cc FROM orders
),
pairs AS (
  SELECT s.user_id, s.ts, s.cc,
         LAG(s.ts) OVER (PARTITION BY s.user_id ORDER BY s.ts) AS prev_ts,
         LAG(s.cc) OVER (PARTITION BY s.user_id ORDER BY s.ts) AS prev_cc
  FROM sightings s
)
SELECT p.user_id, p.prev_ts, p.prev_cc, p.ts, p.cc,
       ROUND(TIMESTAMPDIFF(MINUTE, p.prev_ts, p.ts) / 60.0, 2) AS gap_hours,
       ROUND(6371 * 2 * ASIN(SQRT(
         POW(SIN(RADIANS(c2.lat - c1.lat) / 2), 2) +
         COS(RADIANS(c1.lat)) * COS(RADIANS(c2.lat)) *
         POW(SIN(RADIANS(c2.lon - c1.lon) / 2), 2))), 0) AS km,
       ROUND(6371 * 2 * ASIN(SQRT(
         POW(SIN(RADIANS(c2.lat - c1.lat) / 2), 2) +
         COS(RADIANS(c1.lat)) * COS(RADIANS(c2.lat)) *
         POW(SIN(RADIANS(c2.lon - c1.lon) / 2), 2)))
         / GREATEST(TIMESTAMPDIFF(MINUTE, p.prev_ts, p.ts) / 60.0, 0.02), 0) AS implied_kmh
FROM pairs p
JOIN centroids c1 ON c1.cc = p.prev_cc
JOIN centroids c2 ON c2.cc = p.cc
WHERE p.prev_cc IS NOT NULL
  AND p.cc <> p.prev_cc
  AND TIMESTAMPDIFF(MINUTE, p.prev_ts, p.ts) < 720
HAVING implied_kmh > 900
ORDER BY implied_kmh DESC
LIMIT 200;
