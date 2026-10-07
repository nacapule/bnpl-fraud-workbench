-- Q08 — Card testing: which approvals came after a run of processor declines on
-- the same card, on the same device, or on a device cycling through cards?
-- Cutoff: @as_of (see Q01); unset means the end of observation.
-- A row is one processor-approved order attempt. Its window is the 24 hours
-- before it, in the world's event order (declines at the same instant count
-- when they come first). In that window:
--   declines_card_24h    processor declines with this card, on any device;
--   declines_device_24h  processor declines on this device, with any card
--                        (these two are FP-2 R07's counts);
--   inventory_cards_24h  distinct cards declined on one device, the inventory
--                        device: this attempt's device, or a device where this
--                        attempt's card was declined, whichever cycled through
--                        more cards (lowest id on a tie). It catches a device
--                        testing several cards whose approval lands on another
--                        device.
-- The row is shown when any of the three reaches 3. Its amount is the amount of
-- this approval; each approval after a burst is its own row.
-- Read: fraudsters validate stolen cards with cheap declined attempts, then
-- spend; the earliest decline in the window shows how long the test ran.
SET @as_of = CAST(COALESCE(@as_of, '2025-12-29 23:59:59') AS DATETIME);

WITH attempts AS (
  SELECT event_id, order_id, user_id, card_id, device_id, known_at, amount_cents, processor_result
  FROM order_attempts
  WHERE known_at <= @as_of
),
approvals AS (
  SELECT * FROM attempts WHERE processor_result = 'approved'
),
declines AS (
  SELECT * FROM attempts WHERE processor_result = 'declined'
),
card_window AS (
  SELECT a.order_id, a.event_id AS approval_event_id, a.known_at AS approved_at, d.device_id,
         d.known_at
  FROM approvals a
  JOIN declines d
    ON d.card_id = a.card_id
   AND d.known_at >= a.known_at - INTERVAL 24 HOUR
   AND (d.known_at < a.known_at OR (d.known_at = a.known_at AND d.event_id < a.event_id))
),
candidate_devices AS (
  -- the approval's own device and every device its card was declined on
  SELECT DISTINCT order_id, approval_event_id, approved_at, device_id FROM card_window
  UNION
  SELECT order_id, event_id, known_at, device_id FROM approvals
),
device_window AS (
  SELECT c.order_id, c.device_id, COUNT(*) AS declines, COUNT(DISTINCT d.card_id) AS cards,
         MIN(d.known_at) AS first_decline_at
  FROM candidate_devices c
  JOIN declines d
    ON d.device_id = c.device_id
   AND d.known_at >= c.approved_at - INTERVAL 24 HOUR
   AND (d.known_at < c.approved_at
        OR (d.known_at = c.approved_at AND d.event_id < c.approval_event_id))
  GROUP BY c.order_id, c.device_id
),
inventory AS (
  SELECT order_id, device_id AS inventory_device_id, cards AS inventory_cards_24h
  FROM (
    SELECT w.*, ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY cards DESC, device_id) AS pick
    FROM device_window w
  ) ranked
  WHERE pick = 1
),
card_counts AS (
  SELECT order_id, COUNT(*) AS declines_card_24h, MIN(known_at) AS first_decline_at
  FROM card_window
  GROUP BY order_id
),
flagged AS (
  SELECT a.*,
         COALESCE(cc.declines_card_24h, 0) AS declines_card_24h,
         COALESCE(own.declines, 0) AS declines_device_24h,
         inv.inventory_device_id, COALESCE(inv.inventory_cards_24h, 0) AS inventory_cards_24h,
         LEAST(COALESCE(cc.first_decline_at, a.known_at), COALESCE(own.first_decline_at, a.known_at),
               COALESCE(invw.first_decline_at, a.known_at)) AS first_decline_at
  FROM approvals a
  LEFT JOIN card_counts cc ON cc.order_id = a.order_id
  LEFT JOIN device_window own ON own.order_id = a.order_id AND own.device_id = a.device_id
  LEFT JOIN inventory inv ON inv.order_id = a.order_id
  LEFT JOIN device_window invw
    ON invw.order_id = a.order_id AND invw.device_id = inv.inventory_device_id
)
SELECT f.order_id, f.user_id, f.card_id, f.device_id, f.known_at AS approved_at,
       CAST(f.amount_cents / 100 AS DECIMAL(12, 2)) AS approved_amount_usd,
       f.declines_card_24h, f.declines_device_24h, f.inventory_device_id, f.inventory_cards_24h,
       f.first_decline_at, c.bin_country
FROM flagged f
JOIN cards c ON c.card_id = f.card_id
WHERE f.declines_card_24h >= 3 OR f.declines_device_24h >= 3 OR f.inventory_cards_24h >= 3
ORDER BY GREATEST(f.declines_card_24h, f.declines_device_24h, f.inventory_cards_24h) DESC,
         f.amount_cents DESC, f.order_id
LIMIT 100;
