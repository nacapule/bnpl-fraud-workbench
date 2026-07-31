# Fraud model evaluation

Chronological holdout: 2026-03-28 00:34:16 through 2026-06-25 23:52:59.

Orders: 86,716; fraud orders: 488; base rate: 0.563%.

Review capacity: 40/day × 90 days = 3,600.

## Detection performance

| Model | PR-AUC | Precision@3,600 | Capacity threshold |
| --- | --- | --- | --- |
| Logistic Regression | 0.7998 | 12.50% | 0.328702 |
| HistGradient Boosting | 0.9381 | 12.78% | 0.183204 |

PR-AUC is reported instead of ROC-AUC because ROC-AUC can look strong while obscuring false
positives at this 0.56% fraud base rate. PR-AUC measures performance on the rare
positive class, while precision@capacity reflects the actual review constraint.

![Precision-recall curves](model_pr_curve.svg)

## Calibration deciles

Each bin contains one score-decile of holdout orders; predicted probability is compared with
the observed fraud rate.

### Logistic Regression

| bin | orders | min_score | max_score | mean_predicted | observed_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 8672 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| 2 | 8672 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| 3 | 8671 | 0.0000 | 0.0001 | 0.0001 | 0.0000 |
| 4 | 8672 | 0.0001 | 0.0005 | 0.0003 | 0.0000 |
| 5 | 8671 | 0.0005 | 0.0019 | 0.0011 | 0.0001 |
| 6 | 8672 | 0.0019 | 0.0057 | 0.0035 | 0.0002 |
| 7 | 8672 | 0.0057 | 0.0152 | 0.0097 | 0.0001 |
| 8 | 8671 | 0.0152 | 0.0403 | 0.0257 | 0.0008 |
| 9 | 8672 | 0.0403 | 0.1242 | 0.0721 | 0.0015 |
| 10 | 8671 | 0.1242 | 1.0000 | 0.3694 | 0.0533 |
### HistGradient Boosting

| bin | orders | min_score | max_score | mean_predicted | observed_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 8672 | 0.0000 | 0.0001 | 0.0001 | 0.0000 |
| 2 | 8672 | 0.0001 | 0.0002 | 0.0001 | 0.0000 |
| 3 | 8671 | 0.0002 | 0.0002 | 0.0002 | 0.0000 |
| 4 | 8672 | 0.0002 | 0.0002 | 0.0002 | 0.0000 |
| 5 | 8671 | 0.0002 | 0.0002 | 0.0002 | 0.0000 |
| 6 | 8672 | 0.0002 | 0.0002 | 0.0002 | 0.0000 |
| 7 | 8672 | 0.0002 | 0.0003 | 0.0002 | 0.0000 |
| 8 | 8671 | 0.0003 | 0.0004 | 0.0003 | 0.0000 |
| 9 | 8672 | 0.0004 | 0.0661 | 0.0221 | 0.0023 |
| 10 | 8671 | 0.0661 | 0.9999 | 0.2277 | 0.0540 |

## Recall by fraud pattern at capacity

| Pattern | Fraud orders | Logistic Regression | HistGradient Boosting |
| --- | --- | --- | --- |
| P-ATO | 68 | 85.29% | 100.00% |
| P-INR-ABUSE | 30 | 6.67% | 6.67% |
| P-NEVERPAY | 86 | 100.00% | 100.00% |
| P-PROMO | 95 | 100.00% | 100.00% |
| P-STOLEN | 88 | 100.00% | 100.00% |
| P-SYNTH | 121 | 100.00% | 100.00% |

![Recall by fraud pattern](model_recall_by_pattern.svg)

## Cost-optimal thresholds

| Model | Threshold | Alerts | TP | FP | Fraud $ caught | Total cost |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.784516 | 688 | 394 | 294 | $151,276.86 | $22,171.76 |
| HistGradient Boosting | 0.975684 | 462 | 450 | 12 | $161,059.01 | $2,296.53 |

Total cost counts realized principal not collected on missed fraud, $2.50
per alert, and the configured margin-plus-LTV insult cost on false positives. A caught fraud
avoids its realized loss exposure; successful down and installment payments reduce that exposure.

## Equal-capacity fraud-dollar comparison

The higher-PR-AUC model (HistGradient Boosting) supplies the model ranking.

| Strategy | Reviewed | Fraud $ caught |
| --- | --- | --- |
| Model top-k | 3600 | $161,554.43 |

`data/alerts.csv` was absent, so rules-only and hybrid were skipped.

## Runtime

Evaluation, including CSV load, point-in-time feature construction, scoring, plots, and report:
11.58 seconds.
