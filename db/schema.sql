-- bnpl-fraud-workbench MySQL 8.4 schema. InnoDB, utf8mb4.
-- Loaded by db/load.py (drops + recreates: idempotent by design).

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS labels, chargebacks, promo_redemptions, promos, account_events,
  payments, installments, plans, orders, merchants, addresses, cards, user_devices,
  devices, users, alerts;
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE users (
  user_id      INT PRIMARY KEY,
  signup_ts    DATETIME NOT NULL,
  email        VARCHAR(120) NOT NULL,
  email_domain VARCHAR(60) NOT NULL,
  kyc_country  CHAR(2) NOT NULL,
  dob_year     SMALLINT NOT NULL,
  KEY idx_users_domain (email_domain),
  KEY idx_users_signup (signup_ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE devices (
  device_id   INT PRIMARY KEY,
  fingerprint CHAR(16) NOT NULL,
  ua_family   VARCHAR(20) NOT NULL,
  KEY idx_devices_fp (fingerprint)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE user_devices (
  user_id    INT NOT NULL,
  device_id  INT NOT NULL,
  first_seen DATETIME NOT NULL,
  last_seen  DATETIME NOT NULL,
  PRIMARY KEY (user_id, device_id, first_seen),
  KEY idx_ud_device (device_id),
  CONSTRAINT fk_ud_user FOREIGN KEY (user_id) REFERENCES users (user_id),
  CONSTRAINT fk_ud_device FOREIGN KEY (device_id) REFERENCES devices (device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE cards (
  card_id     INT PRIMARY KEY,
  user_id     INT NOT NULL,
  bin_country CHAR(2) NOT NULL,
  network     VARCHAR(10) NOT NULL,
  last4       CHAR(4) NOT NULL,
  KEY idx_cards_user (user_id),
  CONSTRAINT fk_cards_user FOREIGN KEY (user_id) REFERENCES users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE addresses (
  address_id INT PRIMARY KEY,
  user_id    INT NOT NULL,
  line_hash  VARCHAR(20) NOT NULL,
  city       VARCHAR(40) NOT NULL,
  region     VARCHAR(10) NOT NULL,
  country    CHAR(2) NOT NULL,
  added_ts   DATETIME NOT NULL,
  KEY idx_addr_user (user_id),
  KEY idx_addr_line (line_hash),
  CONSTRAINT fk_addr_user FOREIGN KEY (user_id) REFERENCES users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE merchants (
  merchant_id  INT PRIMARY KEY,
  name         VARCHAR(60) NOT NULL,
  category     VARCHAR(30) NOT NULL,
  risk_tier    TINYINT NOT NULL,
  onboarded_ts DATETIME NOT NULL,
  KEY idx_merch_cat (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE orders (
  order_id        INT PRIMARY KEY,
  user_id         INT NOT NULL,
  merchant_id     INT NOT NULL,
  ts              DATETIME NOT NULL,
  amount          DECIMAL(10,2) NOT NULL,
  ip              VARCHAR(40) NOT NULL,
  ip_country      CHAR(2) NOT NULL,
  device_id       INT NOT NULL,
  card_id         INT NOT NULL,
  ship_address_id INT NOT NULL,
  avs_result      CHAR(1) NOT NULL,
  cvv_result      CHAR(1) NOT NULL,
  status          ENUM('approved','declined','cancelled') NOT NULL,
  KEY idx_orders_user_ts (user_id, ts),
  KEY idx_orders_device (device_id),
  KEY idx_orders_merchant_ts (merchant_id, ts),
  KEY idx_orders_card (card_id),
  KEY idx_orders_ship (ship_address_id),
  CONSTRAINT fk_orders_user FOREIGN KEY (user_id) REFERENCES users (user_id),
  CONSTRAINT fk_orders_merch FOREIGN KEY (merchant_id) REFERENCES merchants (merchant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE plans (
  plan_id        INT PRIMARY KEY,
  order_id       INT NOT NULL,
  principal      DECIMAL(10,2) NOT NULL,
  down_amount    DECIMAL(10,2) NOT NULL,
  n_installments TINYINT NOT NULL,
  KEY idx_plans_order (order_id),
  CONSTRAINT fk_plans_order FOREIGN KEY (order_id) REFERENCES orders (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE installments (
  installment_id INT PRIMARY KEY,
  plan_id        INT NOT NULL,
  seq            TINYINT NOT NULL,
  due_ts         DATETIME NOT NULL,
  paid_ts        DATETIME NULL,
  amount         DECIMAL(10,2) NOT NULL,
  outcome        ENUM('pending','paid','late','failed','written_off') NOT NULL,
  KEY idx_inst_plan_seq (plan_id, seq),
  KEY idx_inst_due (due_ts),
  CONSTRAINT fk_inst_plan FOREIGN KEY (plan_id) REFERENCES plans (plan_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE payments (
  payment_id     INT PRIMARY KEY,
  plan_id        INT NOT NULL,
  installment_id INT NULL,
  ts             DATETIME NOT NULL,
  amount         DECIMAL(10,2) NOT NULL,
  method         VARCHAR(10) NOT NULL,
  result         ENUM('success','fail') NOT NULL,
  KEY idx_pay_plan (plan_id),
  KEY idx_pay_ts (ts),
  CONSTRAINT fk_pay_plan FOREIGN KEY (plan_id) REFERENCES plans (plan_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE account_events (
  event_id  INT PRIMARY KEY,
  user_id   INT NOT NULL,
  ts        DATETIME NOT NULL,
  kind      ENUM('login','password_change','email_change','address_add','device_add') NOT NULL,
  ip        VARCHAR(40) NOT NULL,
  device_id INT NOT NULL,
  KEY idx_ev_user_ts (user_id, ts),
  KEY idx_ev_kind (kind),
  CONSTRAINT fk_ev_user FOREIGN KEY (user_id) REFERENCES users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE promos (
  promo_id     INT PRIMARY KEY,
  code         VARCHAR(20) NOT NULL,
  discount_pct TINYINT NOT NULL,
  valid_from   DATETIME NOT NULL,
  valid_to     DATETIME NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE promo_redemptions (
  redemption_id INT PRIMARY KEY,
  promo_id      INT NOT NULL,
  user_id       INT NOT NULL,
  order_id      INT NOT NULL,
  ts            DATETIME NOT NULL,
  KEY idx_pr_user (user_id),
  KEY idx_pr_order (order_id),
  CONSTRAINT fk_pr_promo FOREIGN KEY (promo_id) REFERENCES promos (promo_id),
  CONSTRAINT fk_pr_user FOREIGN KEY (user_id) REFERENCES users (user_id),
  CONSTRAINT fk_pr_order FOREIGN KEY (order_id) REFERENCES orders (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE chargebacks (
  chargeback_id INT PRIMARY KEY,
  order_id      INT NOT NULL,
  reason        ENUM('fraud','inr','not_as_described') NOT NULL,
  opened_ts     DATETIME NOT NULL,
  outcome       ENUM('lost','won','pending') NOT NULL,
  KEY idx_cb_order (order_id),
  KEY idx_cb_opened (opened_ts),
  CONSTRAINT fk_cb_order FOREIGN KEY (order_id) REFERENCES orders (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Ground truth. Analyst-facing layers (queries Q01–Q10, rules, packets, memos)
-- must never join this table; offline evaluation only.
CREATE TABLE labels (
  order_id   INT NOT NULL,
  user_id    INT NOT NULL,
  pattern_id VARCHAR(15) NOT NULL,
  PRIMARY KEY (order_id, pattern_id),
  KEY idx_labels_user (user_id),
  CONSTRAINT fk_labels_order FOREIGN KEY (order_id) REFERENCES orders (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Written by rules/engine.py (not the simulator); created here so FKs and
-- permissions are in one place.
CREATE TABLE alerts (
  alert_id    INT PRIMARY KEY,
  order_id    INT NOT NULL,
  user_id     INT NOT NULL,
  ts          DATETIME NOT NULL,
  score       INT NOT NULL,
  band        ENUM('review','decline') NOT NULL,
  fired_rules JSON NOT NULL,
  KEY idx_alerts_order (order_id),
  KEY idx_alerts_ts (ts),
  CONSTRAINT fk_alerts_order FOREIGN KEY (order_id) REFERENCES orders (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
