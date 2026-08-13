# Model results
- Holdout: 82,717 approved orders; base rate 0.382%.
- PR-AUC: Logistic Regression 0.7337; HistGradient Boosting 0.9455.
- Precision@capacity: Logistic Regression 8.74%; HistGradient Boosting 9.03% (3,400 reviews).
- Recall by pattern: P-ATO Logistic Regression 82.00%/HistGradient Boosting 100.00%; P-INR-ABUSE Logistic Regression 28.57%/HistGradient Boosting 35.71%; P-NEVERPAY Logistic Regression 100.00%/HistGradient Boosting 100.00%; P-PROMO Logistic Regression 100.00%/HistGradient Boosting 100.00%; P-STOLEN Logistic Regression 100.00%/HistGradient Boosting 100.00%.
- Runtime is machine-dependent and is not part of the committed metrics.
