-- Policy output loaded beside the world (not world data): the incumbent rules
-- policy's routing decisions on the loaded world, written by the pipeline's
-- alerts stage after the world is loaded (python pipeline.py, stage alerts),
-- which replaces the rows in one transaction. Reloading the world drops this
-- table with the world's own objects, so a changed definition takes effect
-- at the next world load.

CREATE TABLE IF NOT EXISTS alerts (
  alert_id VARCHAR(64) NOT NULL COMMENT 'Stable alert id: order id and policy version.',
  order_id INT NOT NULL COMMENT 'Routed order.',
  user_id INT NOT NULL COMMENT 'Ordering account.',
  ts DATETIME NOT NULL COMMENT 'When the routing decision was made: the checkout known_at.',
  score DOUBLE NOT NULL COMMENT 'Policy score at checkout.',
  band ENUM('review', 'auto_decline') NOT NULL COMMENT 'Route taken: review queue or automatic decline.',
  fired_rules JSON NOT NULL COMMENT 'Rule ids that fired at checkout.',
  policy VARCHAR(64) NOT NULL COMMENT 'Policy name.',
  policy_version VARCHAR(64) NOT NULL COMMENT 'Policy version (configuration and scorer hash).',
  PRIMARY KEY (alert_id),
  UNIQUE KEY uq_alerts_order_policy (order_id, policy, policy_version),
  KEY ix_alerts_ts (ts),
  CONSTRAINT fk_alerts_order FOREIGN KEY (order_id) REFERENCES order_attempts (order_id),
  CONSTRAINT fk_alerts_user FOREIGN KEY (user_id) REFERENCES accounts (user_id)
) COMMENT 'Routing decisions of the incumbent rules policy (policy output, not world data).';
