-- Q11 — Alert-queue operations view: what is open, how old, and what mix of
-- bands is the team facing? (Reads the alerts table written by rules/engine.py;
-- resolution timestamps come from the queue simulation when present.)
-- Read: this is the standup dashboard — daily alert volume by band, and an age
-- histogram of the backlog. Volume spikes localized to one band usually mean a
-- threshold change or an attack, not organic growth.
WITH daily AS (
  SELECT DATE(ts) AS d, band, COUNT(*) AS n_alerts,
         ROUND(AVG(score), 1) AS avg_score
  FROM alerts
  GROUP BY DATE(ts), band
)
SELECT d, band, n_alerts, avg_score,
       SUM(n_alerts) OVER (PARTITION BY band ORDER BY d
                           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS n_7d_rolling
FROM daily
ORDER BY d DESC, band
LIMIT 200;
