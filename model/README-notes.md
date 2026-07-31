# Model results
- Holdout: 86,716 approved orders; base rate 0.563%.
- PR-AUC: Logistic Regression 0.7998; HistGradient Boosting 0.9381.
- Precision@capacity: Logistic Regression 12.50%; HistGradient Boosting 12.78% (3,600 reviews).
- Recall by pattern: P-ATO Logistic Regression 85.29%/HistGradient Boosting 100.00%; P-INR-ABUSE Logistic Regression 6.67%/HistGradient Boosting 6.67%; P-NEVERPAY Logistic Regression 100.00%/HistGradient Boosting 100.00%; P-PROMO Logistic Regression 100.00%/HistGradient Boosting 100.00%; P-STOLEN Logistic Regression 100.00%/HistGradient Boosting 100.00%; P-SYNTH Logistic Regression 100.00%/HistGradient Boosting 100.00%.
- Full evaluation runtime: 11.58 seconds.
