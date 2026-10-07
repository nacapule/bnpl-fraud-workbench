-- bnpl-fraud-workbench MySQL 8.4 schema, generated from core/world.py by
-- `python db/load_world.py --write-schema`; do not edit by hand.
-- Latent tables are simulator truth for offline diagnostics only: analyst-facing
-- queries, rules, packets and memos never read them or the labels.

-- observable entity: One row per customer account.
CREATE TABLE accounts (
  user_id INT NOT NULL COMMENT 'Customer account id.',
  created_at DATETIME NOT NULL COMMENT 'Signup time.',
  email VARCHAR(120) NOT NULL COMMENT 'Signup email as entered (normalised only in core.asof); later addresses are email_change events.',
  email_domain VARCHAR(64) NOT NULL COMMENT 'Domain of the signup email, lower case.',
  home_country VARCHAR(64) NOT NULL COMMENT 'Country of residence from KYC; drives the home IP country.',
  dob_year INT NOT NULL COMMENT 'Year of birth.',
  PRIMARY KEY (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable entity: One row per merchant. A closed merchant takes no orders and cannot be charged back.
CREATE TABLE merchants (
  merchant_id INT NOT NULL COMMENT 'Merchant id.',
  created_at DATETIME NOT NULL COMMENT 'Onboarding time.',
  name VARCHAR(80) NOT NULL COMMENT 'Trading name.',
  category VARCHAR(64) NOT NULL COMMENT 'Merchandise category.',
  risk_tier INT NOT NULL COMMENT 'Onboarding risk tier, 1 (low) to 3 (high).',
  fulfilment_median_hours DOUBLE NOT NULL COMMENT 'Median hours from approval to shipment.',
  closed_at DATETIME NULL COMMENT 'When the merchant stopped trading and became unreachable.',
  PRIMARY KEY (merchant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable entity: One row per physical device.
CREATE TABLE devices (
  device_id INT NOT NULL COMMENT 'Device id.',
  created_at DATETIME NOT NULL COMMENT 'First time the device was seen on the platform.',
  fingerprint VARCHAR(64) NOT NULL COMMENT 'Device fingerprint.',
  ua_family VARCHAR(64) NOT NULL COMMENT 'Browser or OS family.',
  PRIMARY KEY (device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable entity: One row per physical address; households and drops share rows.
CREATE TABLE addresses (
  address_id INT NOT NULL COMMENT 'Physical address id.',
  created_at DATETIME NOT NULL COMMENT 'First time any account registered the address.',
  line_hash VARCHAR(40) NOT NULL COMMENT 'Hash of the normalised street line.',
  city VARCHAR(64) NOT NULL COMMENT 'City.',
  region VARCHAR(64) NOT NULL COMMENT 'State or province.',
  country VARCHAR(64) NOT NULL COMMENT 'Country.',
  PRIMARY KEY (address_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable entity: Platform-funded promotions.
CREATE TABLE promotions (
  promo_id INT NOT NULL COMMENT 'Promotion id.',
  code VARCHAR(64) NOT NULL COMMENT 'Promotion code.',
  discount_bps INT NOT NULL COMMENT 'Discount on the merchant''s price, in basis points.',
  first_purchase_only BOOLEAN NOT NULL COMMENT 'Only an account''s first approved order may use it.',
  valid_from DATETIME NOT NULL COMMENT 'Start of validity (creation).',
  valid_to DATETIME NOT NULL COMMENT 'End of validity (exclusive).',
  PRIMARY KEY (promo_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- latent truth: Simulator-only: fraud episodes (one actor or ring acting together).
CREATE TABLE latent_episodes (
  episode_id INT NOT NULL COMMENT 'Episode id.',
  pattern_id ENUM('P-ATO', 'P-STOLEN', 'P-SYNTH', 'P-NEVERPAY', 'P-INR-ABUSE', 'P-PROMO', 'P-MERCH') NOT NULL COMMENT 'Fraud pattern.',
  started_at DATETIME NOT NULL COMMENT 'First event of the episode.',
  ended_at DATETIME NULL COMMENT 'Last event of the episode.',
  PRIMARY KEY (episode_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'latent';

-- observable entity: Payment cards on accounts.
CREATE TABLE cards (
  card_id INT NOT NULL COMMENT 'Card id (one per account registration).',
  user_id INT NOT NULL COMMENT 'Account that registered the card.',
  created_at DATETIME NOT NULL COMMENT 'When the card was added to the account.',
  removed_at DATETIME NULL COMMENT 'When it was removed.',
  bin_country VARCHAR(64) NOT NULL COMMENT 'Issuing country from the BIN.',
  network VARCHAR(64) NOT NULL COMMENT 'Card network.',
  last4 VARCHAR(64) NOT NULL COMMENT 'Last four digits.',
  PRIMARY KEY (card_id),
  CONSTRAINT fk_cards_user_id FOREIGN KEY (user_id) REFERENCES accounts (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable entity: Device association: an account may use a device in [created_at, removed_at).
CREATE TABLE device_links (
  user_id INT NOT NULL COMMENT 'Account.',
  device_id INT NOT NULL COMMENT 'Device.',
  created_at DATETIME NOT NULL COMMENT 'First use of the device on this account.',
  removed_at DATETIME NULL COMMENT 'When the device stopped being usable on the account.',
  PRIMARY KEY (user_id, device_id, created_at),
  CONSTRAINT fk_device_links_user_id FOREIGN KEY (user_id) REFERENCES accounts (user_id),
  CONSTRAINT fk_device_links_device_id FOREIGN KEY (device_id) REFERENCES devices (device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: Logins and credential changes. Device and address additions are the link tables.
CREATE TABLE account_events (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  user_id INT NOT NULL COMMENT 'Account.',
  kind ENUM('login', 'password_reset', 'password_change', 'email_change', 'phone_change') NOT NULL COMMENT 'Account activity.',
  device_id INT NOT NULL COMMENT 'Device used; must be linked to the account at the time.',
  ip VARCHAR(45) NOT NULL COMMENT 'IP address.',
  ip_country VARCHAR(64) NOT NULL COMMENT 'IP geolocation country.',
  email VARCHAR(120) NULL COMMENT 'The new address of an email_change (null for other kinds).',
  PRIMARY KEY (event_id),
  KEY ix_account_events_known_at (known_at),
  KEY ix_account_events_1 (user_id, known_at),
  CONSTRAINT fk_account_events_user_id FOREIGN KEY (user_id) REFERENCES accounts (user_id),
  CONSTRAINT fk_account_events_device_id FOREIGN KEY (device_id) REFERENCES devices (device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable entity: Address association: an order may ship to an address linked and active at its time.
CREATE TABLE address_links (
  user_id INT NOT NULL COMMENT 'Account.',
  address_id INT NOT NULL COMMENT 'Address.',
  created_at DATETIME NOT NULL COMMENT 'When the account added the address.',
  removed_at DATETIME NULL COMMENT 'When the account removed it (a move ends the old home).',
  role ENUM('home', 'shipping') NOT NULL COMMENT 'home: the account''s residence while active; shipping: an extra delivery address (gift recipient, office, drop).',
  PRIMARY KEY (user_id, address_id, created_at),
  CONSTRAINT fk_address_links_user_id FOREIGN KEY (user_id) REFERENCES accounts (user_id),
  CONSTRAINT fk_address_links_address_id FOREIGN KEY (address_id) REFERENCES addresses (address_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- latent truth: Simulator-only: account truth. A taken-over account stays legitimate here.
CREATE TABLE latent_accounts (
  user_id INT NOT NULL COMMENT 'Account.',
  actor ENUM('legitimate', 'fraudster', 'synthetic_identity') NOT NULL COMMENT 'Who controls the account.',
  episode_id INT NULL COMMENT 'Episode the account belongs to.',
  profile VARCHAR(160) NULL COMMENT 'Behaviour profile or benign mimic (traveller, mover, household, hardship, new_customer, sleeper, ...).',
  PRIMARY KEY (user_id),
  CONSTRAINT fk_latent_accounts_user_id FOREIGN KEY (user_id) REFERENCES accounts (user_id),
  CONSTRAINT fk_latent_accounts_episode_id FOREIGN KEY (episode_id) REFERENCES latent_episodes (episode_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'latent';

-- observable event: Every attempted checkout, with its potential outcomes under approve-all; known at the attempt (known_at = occurred_at).
CREATE TABLE order_attempts (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  order_id INT NOT NULL COMMENT 'Order id, increasing with occurred_at.',
  user_id INT NOT NULL COMMENT 'Ordering account.',
  merchant_id INT NOT NULL COMMENT 'Merchant.',
  device_id INT NOT NULL COMMENT 'Device used.',
  card_id INT NOT NULL COMMENT 'Card charged at checkout.',
  ship_address_id INT NOT NULL COMMENT 'Delivery address.',
  amount_cents BIGINT NOT NULL COMMENT 'Merchant''s price.',
  promo_id INT NULL COMMENT 'Promotion used.',
  promo_discount_cents BIGINT NOT NULL COMMENT 'Platform-funded discount (0 without promotion).',
  ip VARCHAR(45) NOT NULL COMMENT 'Checkout IP address.',
  ip_country VARCHAR(64) NOT NULL COMMENT 'IP geolocation country.',
  avs_result ENUM('Y', 'N') NOT NULL COMMENT 'Address verification: Y match, N mismatch.',
  cvv_result ENUM('M', 'N') NOT NULL COMMENT 'Card verification: M match, N mismatch.',
  processor_result ENUM('approved', 'declined') NOT NULL COMMENT 'Processor decision; only processor declines happen inside the world.',
  PRIMARY KEY (event_id),
  UNIQUE KEY uq_order_attempts_order_id (order_id),
  KEY ix_order_attempts_known_at (known_at),
  KEY ix_order_attempts_1 (user_id, known_at),
  KEY ix_order_attempts_2 (merchant_id, known_at),
  CONSTRAINT fk_order_attempts_user_id FOREIGN KEY (user_id) REFERENCES accounts (user_id),
  CONSTRAINT fk_order_attempts_merchant_id FOREIGN KEY (merchant_id) REFERENCES merchants (merchant_id),
  CONSTRAINT fk_order_attempts_device_id FOREIGN KEY (device_id) REFERENCES devices (device_id),
  CONSTRAINT fk_order_attempts_card_id FOREIGN KEY (card_id) REFERENCES cards (card_id),
  CONSTRAINT fk_order_attempts_ship_address_id FOREIGN KEY (ship_address_id) REFERENCES addresses (address_id),
  CONSTRAINT fk_order_attempts_promo_id FOREIGN KEY (promo_id) REFERENCES promotions (promo_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable entity: One pay-in-4 plan per processor-approved order (the approve-all potential outcome).
CREATE TABLE plans (
  plan_id INT NOT NULL COMMENT 'Plan id.',
  order_id INT NOT NULL COMMENT 'The processor-approved order it finances.',
  created_at DATETIME NOT NULL COMMENT 'Approval time (equals the order''s occurred_at).',
  principal_cents BIGINT NOT NULL COMMENT 'Customer obligation: amount minus promotion discount.',
  down_payment_cents BIGINT NOT NULL COMMENT 'Collected at approval (schedule seq 0).',
  n_installments INT NOT NULL COMMENT 'Installments after the down payment.',
  PRIMARY KEY (plan_id),
  UNIQUE KEY uq_plans_order_id (order_id),
  CONSTRAINT fk_plans_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: Merchant reports shipment; the merchant is settled then.
CREATE TABLE fulfilments (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  order_id INT NOT NULL COMMENT 'Order shipped.',
  PRIMARY KEY (event_id),
  KEY ix_fulfilments_known_at (known_at),
  CONSTRAINT fk_fulfilments_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: Carrier-confirmed delivery. A shipped order without one was not delivered.
CREATE TABLE deliveries (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  order_id INT NOT NULL COMMENT 'Order delivered.',
  PRIMARY KEY (event_id),
  KEY ix_deliveries_known_at (known_at),
  CONSTRAINT fk_deliveries_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: occurred_at: the customer files; known_at: the platform is notified.
CREATE TABLE dispute_openings (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  dispute_id INT NOT NULL COMMENT 'Dispute id, increasing with known_at.',
  order_id INT NOT NULL COMMENT 'Disputed order.',
  reason ENUM('unauthorized', 'item_not_received', 'not_as_described') NOT NULL COMMENT 'Reason given by the customer.',
  amount_cents BIGINT NOT NULL COMMENT 'Collected payments disputed (positive).',
  PRIMARY KEY (event_id),
  UNIQUE KEY uq_dispute_openings_dispute_id (dispute_id),
  KEY ix_dispute_openings_known_at (known_at),
  CONSTRAINT fk_dispute_openings_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: Account owner disowns orders after a takeover; one row per disowned order.
CREATE TABLE victim_reports (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  user_id INT NOT NULL COMMENT 'Account owner reporting.',
  order_id INT NOT NULL COMMENT 'Order the owner says they did not place.',
  PRIMARY KEY (event_id),
  KEY ix_victim_reports_known_at (known_at),
  CONSTRAINT fk_victim_reports_user_id FOREIGN KEY (user_id) REFERENCES accounts (user_id),
  CONSTRAINT fk_victim_reports_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- latent truth: Simulator-only: order truth. Repaid ring warm-ups stay episode members.
CREATE TABLE latent_orders (
  order_id INT NOT NULL COMMENT 'Order.',
  pattern_id ENUM('P-ATO', 'P-STOLEN', 'P-SYNTH', 'P-NEVERPAY', 'P-INR-ABUSE', 'P-PROMO', 'P-MERCH') NULL COMMENT 'Fraud pattern; null for legitimate orders.',
  episode_id INT NULL COMMENT 'Episode.',
  intent ENUM('legitimate', 'fraud', 'abuse') NOT NULL COMMENT 'Intent behind the order.',
  mimic VARCHAR(160) NULL COMMENT 'Benign behaviour that resembles fraud, if any.',
  PRIMARY KEY (order_id),
  CONSTRAINT fk_latent_orders_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id),
  CONSTRAINT fk_latent_orders_episode_id FOREIGN KEY (episode_id) REFERENCES latent_episodes (episode_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'latent';

-- adjudicated label: Adjudicated label history from core.world.adjudicate: at most one negative row (known when the horizon has passed) and one positive row (the first determination). An order has no label before its first row; use labels_as_of().
CREATE TABLE labels (
  order_id INT NOT NULL COMMENT 'Processor-approved order.',
  label INT NOT NULL COMMENT '1 = abusive order as determined, 0 = not.',
  basis ENUM('third_party_fraud', 'account_takeover', 'never_pay', 'inr_abuse', 'promo_abuse', 'merchant_bustout', 'credit_loss', 'no_finding') NOT NULL COMMENT 'Determination behind the label.',
  label_known_at DATETIME NOT NULL COMMENT 'When the determination was known.',
  PRIMARY KEY (order_id, label_known_at),
  CONSTRAINT fk_labels_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'adjudicated';

-- observable schedule: What each plan owes and when; known from the plan's creation.
CREATE TABLE installment_schedule (
  plan_id INT NOT NULL COMMENT 'Plan.',
  seq INT NOT NULL COMMENT '0 = down payment due at approval; 1..n installments.',
  due_at DATETIME NOT NULL COMMENT 'Due time.',
  amount_cents BIGINT NOT NULL COMMENT 'Amount due (core.ledger.split_principal).',
  PRIMARY KEY (plan_id, seq),
  CONSTRAINT fk_installment_schedule_plan_id FOREIGN KEY (plan_id) REFERENCES plans (plan_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: Plans written off after the last installment stays unpaid.
CREATE TABLE plan_writeoffs (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  plan_id INT NOT NULL COMMENT 'Plan.',
  outstanding_cents BIGINT NOT NULL COMMENT 'Unpaid balance written off (a status, not cash).',
  PRIMARY KEY (event_id),
  KEY ix_plan_writeoffs_known_at (known_at),
  CONSTRAINT fk_plan_writeoffs_plan_id FOREIGN KEY (plan_id) REFERENCES plans (plan_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: The platform's cash ledger (core.ledger). Loss over a horizon = -sum(amount_cents).
CREATE TABLE cash_events (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  order_id INT NOT NULL COMMENT 'Order.',
  plan_id INT NOT NULL COMMENT 'Plan.',
  merchant_id INT NOT NULL COMMENT 'Merchant.',
  kind ENUM('merchant_settlement', 'promotion_funding', 'customer_payment', 'payment_reversal', 'refund', 'dispute_debit', 'dispute_fee', 'dispute_won_credit', 'merchant_recourse', 'recovery') NOT NULL COMMENT 'Cash movement (core.ledger.CASH_KINDS).',
  amount_cents BIGINT NOT NULL COMMENT 'Signed cents from the platform''s point of view.',
  ref_event_id INT NOT NULL COMMENT 'World event that caused it (for action-caused cash, the payment it compensates).',
  cause VARCHAR(64) NOT NULL COMMENT '''natural'', or the replay action that produced it.',
  PRIMARY KEY (event_id),
  KEY ix_cash_events_known_at (known_at),
  KEY ix_cash_events_1 (order_id),
  CONSTRAINT fk_cash_events_order_id FOREIGN KEY (order_id) REFERENCES order_attempts (order_id),
  CONSTRAINT fk_cash_events_plan_id FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
  CONSTRAINT fk_cash_events_merchant_id FOREIGN KEY (merchant_id) REFERENCES merchants (merchant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: A dispute without a resolution is pending.
CREATE TABLE dispute_resolutions (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  dispute_id INT NOT NULL COMMENT 'Dispute.',
  outcome ENUM('won', 'lost') NOT NULL COMMENT 'won: resolved for the platform (against the customer); lost: for the customer.',
  PRIMARY KEY (event_id),
  UNIQUE KEY uq_dispute_resolutions_dispute_id (dispute_id),
  KEY ix_dispute_resolutions_known_at (known_at),
  CONSTRAINT fk_dispute_resolutions_dispute_id FOREIGN KEY (dispute_id) REFERENCES dispute_openings (dispute_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: Collection attempts; the down payment is seq 0 at approval.
CREATE TABLE payment_attempts (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  plan_id INT NOT NULL COMMENT 'Plan.',
  seq INT NOT NULL COMMENT 'Schedule entry being paid.',
  attempt_no INT NOT NULL COMMENT '1 for the first try, then retries.',
  amount_cents BIGINT NOT NULL COMMENT 'Amount attempted.',
  result ENUM('success', 'failed') NOT NULL COMMENT 'Collection result.',
  PRIMARY KEY (event_id),
  KEY ix_payment_attempts_known_at (known_at),
  KEY ix_payment_attempts_1 (plan_id, known_at),
  CONSTRAINT fk_payment_attempts_plan_id FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
  CONSTRAINT fk_payment_attempts_x1 FOREIGN KEY (plan_id, seq) REFERENCES installment_schedule (plan_id, seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- observable event: Collections that bounced after succeeding.
CREATE TABLE payment_reversals (
  event_id INT NOT NULL COMMENT 'World-unique event id, assigned in the global event order.',
  occurred_at DATETIME NOT NULL COMMENT 'When the event happened.',
  known_at DATETIME NOT NULL COMMENT 'When the platform could know it; as-of code filters on this.',
  payment_event_id INT NOT NULL COMMENT 'The successful payment attempt reversed.',
  plan_id INT NOT NULL COMMENT 'Plan.',
  amount_cents BIGINT NOT NULL COMMENT 'Amount reversed (positive).',
  reason ENUM('bank_return', 'card_reversal') NOT NULL COMMENT 'Why the collection came back.',
  PRIMARY KEY (event_id),
  KEY ix_payment_reversals_known_at (known_at),
  CONSTRAINT fk_payment_reversals_plan_id FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
  CONSTRAINT fk_payment_reversals_x1 FOREIGN KEY (payment_event_id) REFERENCES payment_attempts (event_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT 'observable';

-- Compatibility surface for merchant-risk-screener: its 17 columns over the
-- event tables. `orders.status` is the processor result ('approved' or
-- 'declined'); `chargebacks` has one row per dispute dated when the platform
-- learned of it; `installments` gives each installment's state at the end of
-- observation (paid, late, failed, written_off or pending), the final-state
-- view the screener reads. `plans` is a table above.
CREATE VIEW users AS
SELECT user_id, created_at AS signup_ts FROM accounts;

CREATE VIEW orders AS
SELECT order_id, user_id, merchant_id, occurred_at AS ts,
       CAST(amount_cents / 100 AS DECIMAL(12, 2)) AS amount,
       processor_result AS status
FROM order_attempts;

CREATE VIEW chargebacks AS
SELECT dispute_id AS chargeback_id, order_id, reason, known_at AS opened_ts
FROM dispute_openings;

CREATE VIEW installments AS
SELECT s.plan_id, s.seq, s.due_at AS due_ts,
       CAST(s.amount_cents / 100 AS DECIMAL(12, 2)) AS amount,
       CASE
         WHEN paid.paid_at IS NOT NULL AND paid.paid_at <= s.due_at THEN 'paid'
         WHEN paid.paid_at IS NOT NULL THEN 'late'
         WHEN w.plan_id IS NOT NULL THEN 'written_off'
         WHEN failed.plan_id IS NOT NULL THEN 'failed'
         ELSE 'pending'
       END AS outcome
FROM installment_schedule s
LEFT JOIN (
  -- paid in full: from the event after which the standing cents (payments less
  -- reversals, each at its own time, in event order) never again fall short of
  -- the scheduled amount; never when they end short
  SELECT plan_id, seq, MIN(CASE WHEN n > COALESCE(last_short, 0) THEN moved_at END) AS paid_at
  FROM (
    SELECT plan_id, seq, moved_at, n,
           MAX(CASE WHEN standing < amount_cents THEN n END)
             OVER (PARTITION BY plan_id, seq) AS last_short
    FROM (
      SELECT x.plan_id, x.seq, x.moved_at, s2.amount_cents,
             ROW_NUMBER() OVER by_time AS n, SUM(x.cents) OVER by_time AS standing
      FROM (
        SELECT plan_id, seq, occurred_at AS moved_at, event_id, amount_cents AS cents
        FROM payment_attempts WHERE result = 'success'
        UNION ALL
        SELECT a.plan_id, a.seq, r.occurred_at, r.event_id, -r.amount_cents
        FROM payment_reversals r JOIN payment_attempts a ON a.event_id = r.payment_event_id
      ) x
      JOIN installment_schedule s2 ON s2.plan_id = x.plan_id AND s2.seq = x.seq
      WINDOW by_time AS (PARTITION BY x.plan_id, x.seq ORDER BY x.moved_at, x.event_id)
    ) running
  ) marked
  GROUP BY plan_id, seq
) paid ON paid.plan_id = s.plan_id AND paid.seq = s.seq
LEFT JOIN (SELECT DISTINCT plan_id FROM plan_writeoffs) w ON w.plan_id = s.plan_id
LEFT JOIN (
  SELECT DISTINCT plan_id, seq FROM payment_attempts WHERE result = 'failed'
) failed ON failed.plan_id = s.plan_id AND failed.seq = s.seq
WHERE s.seq >= 1;
