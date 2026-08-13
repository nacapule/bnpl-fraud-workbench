-- Q11 — Daily alert volumes and scores by band, with a seven-calendar-day rollup.
-- Reads the alerts table written by rules/engine.py. Volume or score spikes
-- localized to one band can indicate a threshold change or an attack.
WITH daily AS (
  SELECT DATE(ts) AS d, band, COUNT(*) AS n_alerts,
         ROUND(AVG(score), 1) AS avg_score
  FROM alerts
  GROUP BY DATE(ts), band
)
SELECT d, band, n_alerts, avg_score,
       SUM(n_alerts) OVER (PARTITION BY band ORDER BY d
                           RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)
         AS n_7_calendar_day_rolling
FROM daily
ORDER BY d DESC, band
LIMIT 200;
