# Rule threshold tuning — selected before 2026-04-01, reported after

Cost model: review $2.50/case; false decline = 5% margin + $15 LTV proxy (config `costs`, assumptions documented in README).

| review_band | decline_band | alerts_per_day | precision | recall_overall | caught_usd | n_insults | net_usd |
|---|---|---|---|---|---|---|---|
| 30 | 90 | 24.9 | 0.089 | 0.598 | 37847.0 | 0 | 32550.0 | **⬅ chosen**
| 30 | 100 | 24.9 | 0.089 | 0.598 | 37847.0 | 0 | 32550.0 |
| 30 | 110 | 24.9 | 0.089 | 0.598 | 37847.0 | 0 | 32550.0 |
| 30 | 80 | 24.9 | 0.089 | 0.598 | 37847.0 | 6 | 32406.0 |
| 30 | 70 | 24.9 | 0.089 | 0.598 | 37847.0 | 8 | 32305.0 |
| 35 | 90 | 24.7 | 0.089 | 0.595 | 36855.0 | 0 | 31598.0 |
| 35 | 100 | 24.7 | 0.089 | 0.595 | 36855.0 | 0 | 31598.0 |
| 35 | 110 | 24.7 | 0.089 | 0.595 | 36855.0 | 0 | 31598.0 |
| 35 | 80 | 24.7 | 0.089 | 0.595 | 36855.0 | 6 | 31454.0 |
| 35 | 70 | 24.7 | 0.089 | 0.595 | 36855.0 | 8 | 31353.0 |
| 40 | 90 | 7.6 | 0.259 | 0.528 | 29222.0 | 0 | 27609.0 |
| 40 | 100 | 7.6 | 0.259 | 0.528 | 29222.0 | 0 | 27609.0 |
| 40 | 110 | 7.6 | 0.259 | 0.528 | 29222.0 | 0 | 27609.0 |
| 40 | 80 | 7.6 | 0.259 | 0.528 | 29222.0 | 6 | 27465.0 |
| 40 | 70 | 7.6 | 0.259 | 0.528 | 29222.0 | 8 | 27364.0 |
| 45 | 90 | 7.4 | 0.244 | 0.487 | 24807.0 | 0 | 23227.0 |
| 45 | 100 | 7.4 | 0.244 | 0.487 | 24807.0 | 0 | 23227.0 |
| 45 | 110 | 7.4 | 0.244 | 0.487 | 24807.0 | 0 | 23227.0 |
| 45 | 80 | 7.4 | 0.244 | 0.487 | 24807.0 | 6 | 23083.0 |
| 45 | 70 | 7.4 | 0.244 | 0.487 | 24807.0 | 8 | 22982.0 |
| 50 | 90 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 50 | 100 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 50 | 110 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 55 | 90 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 55 | 100 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 55 | 110 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 60 | 90 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 60 | 100 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 60 | 110 | 1.4 | 0.93 | 0.339 | 9902.0 | 0 | 9614.0 |
| 50 | 80 | 1.4 | 0.93 | 0.339 | 9902.0 | 6 | 9470.0 |
| 55 | 80 | 1.4 | 0.93 | 0.339 | 9902.0 | 6 | 9470.0 |
| 60 | 80 | 1.4 | 0.93 | 0.339 | 9902.0 | 6 | 9470.0 |
| 50 | 70 | 1.4 | 0.93 | 0.339 | 9902.0 | 8 | 9369.0 |
| 55 | 70 | 1.4 | 0.93 | 0.339 | 9902.0 | 8 | 9369.0 |
| 60 | 70 | 1.4 | 0.93 | 0.339 | 9902.0 | 8 | 9369.0 |

Fit-window decline-band tie: 90, 100, 110 share net $414,321; 90 is the lowest tied value and is selected.

**Chosen operating point: review ≥ 30, auto-decline ≥ 90.** It maximizes net $ (32,550.0) under the ≤40 alerts/day capacity constraint: 24.9/day at precision 9%, catching $37,847 of holdout fraud exposure with 0 auto-declined legitimate orders. Raising the review band further trades linearly less review cost for disproportionate recall loss on ATO and stolen-card patterns; lowering it overruns the review team. The classic fraud triangle — loss caught vs review cost vs insult rate — made explicit.

Recall by pattern at the chosen point: P-ATO 100%, P-INR-ABUSE 71%, P-NEVERPAY 2%, P-PROMO 96%, P-STOLEN 39%
