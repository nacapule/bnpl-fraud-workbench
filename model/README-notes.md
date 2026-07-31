# Model results
- Holdout: 86,595 approved orders; base rate 0.409%.
- PR-AUC: Logistic Regression 0.7226; HistGradient Boosting 0.9418.
- Precision@capacity: Logistic Regression 9.19%; HistGradient Boosting 9.47% (3,600 reviews).
- Recall by pattern: P-ATO Logistic Regression 85.29%/HistGradient Boosting 100.00%; P-INR-ABUSE Logistic Regression 23.53%/HistGradient Boosting 23.53%; P-NEVERPAY Logistic Regression 100.00%/HistGradient Boosting 100.00%; P-PROMO Logistic Regression 100.00%/HistGradient Boosting 100.00%; P-STOLEN Logistic Regression 100.00%/HistGradient Boosting 100.00%.
- Full evaluation runtime: 10.82 seconds.
