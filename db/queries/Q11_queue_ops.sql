-- Q11 — Daily routing volumes and scores by band, with a seven-calendar-day
-- rollup.
-- Cutoff: @as_of (see Q01); unset means the end of observation.
-- Reads the alerts table that the pipeline loads beside the world: one row per
-- order the policy routed at checkout to review or auto_decline (FP-2 §4.1),
-- with ts the attempt's checkout time. The rollup sums the band's alerts on the
-- day and the six calendar days before it; days without alerts count as zero.
-- Read: a volume or score spike confined to one band can mean a threshold
-- change or an attack.
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH daily AS (
  SELECT DATE(ts) AS d, band, COUNT(*) AS n_alerts, ROUND(AVG(score), 1) AS avg_score
  FROM alerts
  WHERE ts <= @as_of
  GROUP BY DATE(ts), band
)
SELECT d, band, n_alerts, avg_score,
       CAST(SUM(n_alerts) OVER (PARTITION BY band ORDER BY d
                                RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)
            AS SIGNED) AS n_7_calendar_day_rolling
FROM daily
ORDER BY d DESC, band
LIMIT 200;
