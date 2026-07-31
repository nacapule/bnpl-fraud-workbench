# Fraud model evaluation

Chronological holdout: 2026-03-28 00:34:16 through 2026-06-25 23:52:59.

Orders: 86,595; fraud orders: 354; base rate: 0.409%.

Review capacity: 40/day × 90 days = 3,600.

## Detection performance

| Model | PR-AUC | Precision@3,600 | Capacity threshold |
| --- | --- | --- | --- |
| Logistic Regression | 0.7226 | 9.19% | 0.335188 |
| HistGradient Boosting | 0.9418 | 9.47% | 0.170706 |

PR-AUC is reported instead of ROC-AUC because ROC-AUC can look strong while obscuring false
positives at this 0.41% fraud base rate. PR-AUC measures performance on the rare
positive class, while precision@capacity reflects the actual review constraint.

![Precision-recall curves](model_pr_curve.svg)

## Calibration deciles

Each bin contains one score-decile of holdout orders; predicted probability is compared with
the observed fraud rate.

### Logistic Regression

| bin | orders | min_score | max_score | mean_predicted | observed_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 8660 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| 2 | 8659 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 8660 | 0.0000 | 0.0001 | 0.0000 | 0.0000 |
| 4 | 8659 | 0.0001 | 0.0004 | 0.0002 | 0.0000 |
| 5 | 8660 | 0.0004 | 0.0016 | 0.0009 | 0.0000 |
| 6 | 8659 | 0.0016 | 0.0051 | 0.0031 | 0.0001 |
| 7 | 8660 | 0.0051 | 0.0144 | 0.0090 | 0.0002 |
| 8 | 8659 | 0.0144 | 0.0398 | 0.0250 | 0.0003 |
| 9 | 8660 | 0.0398 | 0.1281 | 0.0728 | 0.0007 |
| 10 | 8659 | 0.1282 | 1.0000 | 0.3649 | 0.0393 |
### HistGradient Boosting

| bin | orders | min_score | max_score | mean_predicted | observed_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 8660 | 0.0000 | 0.0001 | 0.0001 | 0.0000 |
| 2 | 8659 | 0.0001 | 0.0001 | 0.0001 | 0.0000 |
| 3 | 8660 | 0.0001 | 0.0002 | 0.0002 | 0.0000 |
| 4 | 8659 | 0.0002 | 0.0002 | 0.0002 | 0.0000 |
| 5 | 8660 | 0.0002 | 0.0002 | 0.0002 | 0.0000 |
| 6 | 8659 | 0.0002 | 0.0003 | 0.0002 | 0.0000 |
| 7 | 8660 | 0.0003 | 0.0003 | 0.0003 | 0.0000 |
| 8 | 8659 | 0.0003 | 0.0005 | 0.0003 | 0.0000 |
| 9 | 8660 | 0.0005 | 0.0494 | 0.0175 | 0.0003 |
| 10 | 8659 | 0.0494 | 0.9999 | 0.1985 | 0.0405 |

## Recall by fraud pattern at capacity

| Pattern | Fraud orders | Logistic Regression | HistGradient Boosting |
| --- | --- | --- | --- |
| P-ATO | 68 | 85.29% | 100.00% |
| P-INR-ABUSE | 17 | 23.53% | 23.53% |
| P-NEVERPAY | 72 | 100.00% | 100.00% |
| P-PROMO | 95 | 100.00% | 100.00% |
| P-STOLEN | 102 | 100.00% | 100.00% |

![Recall by fraud pattern](model_recall_by_pattern.svg)

## Cost-optimal thresholds

| Model | Threshold | Alerts | TP | FP | Fraud $ caught | Total cost |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.804048 | 494 | 273 | 221 | $64,197.53 | $20,262.79 |
| HistGradient Boosting | 0.965480 | 359 | 327 | 32 | $75,250.63 | $2,551.73 |

Total cost counts realized principal not collected on missed fraud, $2.50
per alert, and the configured margin-plus-LTV insult cost on false positives. A caught fraud
avoids its realized loss exposure; successful down and installment payments reduce that exposure.

## Equal-capacity fraud-dollar comparison

The higher-PR-AUC model (HistGradient Boosting) supplies the model ranking.

| Strategy | Reviewed | Fraud $ caught |
| --- | --- | --- |
| Rules alerts | 2226 | $44,657.85 |
| Model top-k | 3600 | $75,830.02 |
| Hybrid (half rules / half model) | 3600 | $75,830.04 |

Hybrid takes up to half of capacity from ranked rules alerts and fills the remainder from the model ranking without duplicate reviews.

## Runtime

Evaluation, including CSV load, point-in-time feature construction, scoring, plots, and report:
10.82 seconds.
