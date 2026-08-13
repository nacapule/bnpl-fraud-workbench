# Case prevention follow-ups

Read-only counterfactuals over `data/`, using `config.yaml` holdout_start and the committed operating point in `reports/operating_point.json`.

## CASE-01 — ATO address conjunction

```json
{"added_benign_auto_declines": 0, "baseline_auto_declines": 1, "holdout_ato_orders": 50, "proposed_auto_declines": 13}
```

## CASE-02 — device cooldown

```json
{"benign_linked_devices": 0, "devices_cooled": 46, "fraud_linked_devices": 46}
```

## CASE-03 — R08 threshold

```json
{"added_alerts": 4842, "added_alerts_per_day": 57.0, "added_benign_alerts": 4840, "added_fraud_alerts": 2}
```

## CASE-03 — rejected address-attach idea

```json
{"addresses_with_benign_orders": 3382, "flagged_addresses": 3392}
```

## CASE-04 — never-pay checkout candidate

```json
{"benign_first_order_cap_rate": 0.263, "candidate_added_alerts": 1, "candidate_added_pattern": "P-PROMO", "currently_alerted_never_pay_orders": 1, "holdout_never_pay_orders": 63}
```

## CASE-05 — R03 suppression

```json
{"r03_only_alerts": 1303, "suppressed_alerts": 364, "suppressed_alerts_per_day": 4.3, "suppressed_fraud_alerts": 0}
```
