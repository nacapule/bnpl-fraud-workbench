# Uncertainty

Every rate this repository publishes is a proportion on a finite sample, and several of them sit on denominators small enough that the point estimate is the least interesting part. This report attaches an interval to each one, tests the prompt comparison as the paired experiment it is, puts a bootstrap band on the holdout PR-AUC, and reads the sensitivity of the rules operating point off the committed tuning frontier. Nothing here regenerates an artifact; it re-reads them.

## Wilson 95% intervals on published rates

| rate | k / n | point | 95% interval |
|---|---|---|---|
| rules precision at review ≥ 30 | 189 / 2,119 | 8.9% | [7.8%, 10.2%] |
| rules recall over holdout fraud orders | 189 / 316 | 59.8% | [54.3%, 65.1%] |
| claude-sonnet-5 memo_v1 action accuracy | 122 / 200 | 61.0% | [54.1%, 67.5%] |
| claude-sonnet-5 memo_v2 action accuracy | 147 / 200 | 73.5% | [67.0%, 79.1%] |
| gpt-5.6-terra memo_v2 (quota-cut) action accuracy | 66 / 86 | 76.7% | [66.8%, 84.4%] |
| gpt-5.6-luna memo_v2 action accuracy (valid outputs) | 119 / 199 | 59.8% | [52.9%, 66.4%] |
| gpt-5.6-luna memo_v2 action accuracy (schema failure counted wrong) | 119 / 200 | 59.5% | [52.6%, 66.1%] |
| claude-sonnet-5 memo_v2 decline precision | 30 / 31 | 96.8% | [83.8%, 99.4%] |
| claude-sonnet-5 memo_v2 decline recall | 30 / 57 | 52.6% | [39.9%, 65.0%] |
| claude-sonnet-5 memo_v2 pattern identification | 150 / 200 | 75.0% | [68.6%, 80.5%] |
| claude-sonnet-5 memo_v2 memos with a non-verbatim concrete-token field | 26 / 200 | 13.0% | [9.0%, 18.4%] |
| claude-sonnet-5 memo_v2 memos with an unsupported field | 9 / 200 | 4.5% | [2.4%, 8.3%] |
| claude-sonnet-5 memo_v2 perturbation consistency | 42 / 50 | 84.0% | [71.5%, 91.7%] |
| Logistic Regression precision at review capacity | 297 / 3,400 | 8.7% | [7.8%, 9.7%] |
| HistGradient Boosting precision at review capacity | 307 / 3,400 | 9.0% | [8.1%, 10.0%] |

The interval is the Wilson score interval, which stays inside the unit interval on the small and near-boundary denominators here. The widest entry above is claude-sonnet-5 memo_v2 decline recall at ±12.5 points on n=57. Only the two rules rates carry denominators in the thousands; the cross-model comparison lives entirely on 200 cases, and the sonnet-v2 interval (67.0%–79.1%) clears the luna interval (52.9%–66.4%) by 0.6 points. The headline gap between those two arms rests on that margin and nothing more.

## Per-pattern strata (claude-sonnet-5, memo_v2)

| truth pattern | k / n | action accuracy | 95% interval |
|---|---|---|---|
| account_takeover | 22 / 28 | 78.6% | [60.5%, 89.8%] |
| benign | 38 / 60 | 63.3% | [50.7%, 74.4%] |
| inr_abuse | 21 / 22 | 95.5% | [78.2%, 99.2%] |
| merchant_bustout (indicative only) | 0 / 3 | 0.0% | [0.0%, 56.1%] |
| never_pay (indicative only) | 0 / 7 | 0.0% | [0.0%, 35.4%] |
| promo_abuse | 23 / 23 | 100.0% | [85.7%, 100.0%] |
| stolen_card | 8 / 22 | 36.4% | [19.7%, 57.0%] |
| synthetic_ring | 35 / 35 | 100.0% | [90.1%, 100.0%] |

Strata with fewer than 10 cases are marked indicative only: merchant_bustout, never_pay. The merchant_bustout figure of 0.0% on 3 cases is consistent with anything up to 56.1%, so the per-pattern table describes where the eval set is thin as much as it describes the model.

## Prompt v1 → v2, tested as a paired comparison

Both arms ran the same cases through the same model, so the informative quantity is the discordant pairs — cases the two prompts decided differently — not the difference of two independent-looking rates. McNemar's exact test is used because the discordant counts are small.

| comparison | v1 | v2 | v2 better | v2 worse | discordant | exact p |
|---|---|---|---|---|---|---|
| action correct | 61.0% | 73.5% | 33 | 8 | 41 | 1.12e-04 |
| memo carries an unsupported field | 33.5% | 4.5% | 59 | 1 | 60 | 1.06e-16 |

Both changes hold up on 200 paired cases. That is a statement about this prompt on this eval set and nothing wider: the cases were frozen once, and the prompt was written with knowledge of the v1 failure modes.

## Holdout PR-AUC, bootstrapped

| model | PR-AUC | 95% interval | resamples |
|---|---|---|---|
| Logistic Regression | 0.7337 | [0.6868, 0.7753] | 1,000 |
| HistGradient Boosting | 0.9455 | [0.9238, 0.9652] | 1,000 |

Resampling is stratified by class, so the holdout base rate is held fixed and the band describes noise in the ranking. The HistGradient Boosting interval does not overlap the Logistic Regression interval, so the gap between them survives resampling. The fourth decimal place in the point estimate does not: the interval is two orders of magnitude wider than that, and quoting PR-AUC to four figures overstates what one holdout supports.

One caveat the resample cannot fix: orders are drawn independently, but fraud in this world arrives in rings and episodes that share users, devices, and addresses. The effective sample is therefore smaller than 82,717 orders and the band above is optimistic by an amount this method cannot measure.

## Threshold sensitivity at the rules operating point

Holding the decline band at 90, net dollars only fall as the review band rises: $32,550 at 30, $31,598 at 35, $27,609 at 40, $23,227 at 45, $9,614 at 50, $9,614 at 55, $9,614 at 60 — a spread of 70% across the swept range. The operating point is sensitive to the threshold, and sensitive in one direction.

One grid step up, to 35, costs $952 (2.9%) and 0.3 points of recall, so the choice between the first two bands is immaterial. Beyond that the sweep buys precision with recall — precision 8.9% → 24.4% → 93.0% against recall 59.8% → 48.7% → 33.9% — and the cost model prices that as a losing trade at every step, because the review cost it saves is small next to the fraud exposure it stops catching.

The selected band is the lowest value on the grid, so the sweep bounds the operating point from above and not from below. At 24.9 alerts/day it also sits well inside the 40/day review capacity, so what stops the search going lower is the grid, not the queue. Whether a band below the grid edge would price better is an open question this report does not answer.

```json
{"mcnemar": [{"comparison": "action correct", "improved_by_v2": 33, "p_value": 0.00011222142620681552, "worsened_by_v2": 8}, {"comparison": "memo carries an unsupported field", "improved_by_v2": 59, "p_value": 1.0581813203458523e-16, "worsened_by_v2": 1}], "per_pattern": [{"high": 0.8979, "low": 0.6046, "pattern": "account_takeover", "successes": 22, "trials": 28}, {"high": 0.7438, "low": 0.5068, "pattern": "benign", "successes": 38, "trials": 60}, {"high": 0.9919, "low": 0.782, "pattern": "inr_abuse", "successes": 21, "trials": 22}, {"high": 0.5615, "low": 0.0, "pattern": "merchant_bustout", "successes": 0, "trials": 3}, {"high": 0.3543, "low": 0.0, "pattern": "never_pay", "successes": 0, "trials": 7}, {"high": 1.0, "low": 0.8569, "pattern": "promo_abuse", "successes": 23, "trials": 23}, {"high": 0.5705, "low": 0.1973, "pattern": "stolen_card", "successes": 8, "trials": 22}, {"high": 1.0, "low": 0.9011, "pattern": "synthetic_ring", "successes": 35, "trials": 35}], "pr_auc_bootstrap": {"HistGradient Boosting": {"high": 0.965214, "low": 0.923801, "point": 0.945547, "resamples": 1000}, "Logistic Regression": {"high": 0.775315, "low": 0.686818, "point": 0.733658, "resamples": 1000}}, "seed": 416, "wilson": [{"high": 0.1021, "low": 0.0778, "point": 0.0892, "rate": "rules precision at review \u2265 30", "successes": 189, "trials": 2119}, {"high": 0.6507, "low": 0.5432, "point": 0.5981, "rate": "rules recall over holdout fraud orders", "successes": 189, "trials": 316}, {"high": 0.6749, "low": 0.5409, "point": 0.61, "rate": "claude-sonnet-5 memo_v1 action accuracy", "successes": 122, "trials": 200}, {"high": 0.7913, "low": 0.6698, "point": 0.735, "rate": "claude-sonnet-5 memo_v2 action accuracy", "successes": 147, "trials": 200}, {"high": 0.8441, "low": 0.6679, "point": 0.7674, "rate": "gpt-5.6-terra memo_v2 (quota-cut) action accuracy", "successes": 66, "trials": 86}, {"high": 0.6636, "low": 0.5286, "point": 0.598, "rate": "gpt-5.6-luna memo_v2 action accuracy (valid outputs)", "successes": 119, "trials": 199}, {"high": 0.6606, "low": 0.5258, "point": 0.595, "rate": "gpt-5.6-luna memo_v2 action accuracy (schema failure counted wrong)", "successes": 119, "trials": 200}, {"high": 0.9943, "low": 0.8381, "point": 0.9677, "rate": "claude-sonnet-5 memo_v2 decline precision", "successes": 30, "trials": 31}, {"high": 0.6501, "low": 0.3992, "point": 0.5263, "rate": "claude-sonnet-5 memo_v2 decline recall", "successes": 30, "trials": 57}, {"high": 0.8049, "low": 0.6857, "point": 0.75, "rate": "claude-sonnet-5 memo_v2 pattern identification", "successes": 150, "trials": 200}, {"high": 0.1837, "low": 0.0903, "point": 0.13, "rate": "claude-sonnet-5 memo_v2 memos with a non-verbatim concrete-token field", "successes": 26, "trials": 200}, {"high": 0.0833, "low": 0.0239, "point": 0.045, "rate": "claude-sonnet-5 memo_v2 memos with an unsupported field", "successes": 9, "trials": 200}, {"high": 0.9166, "low": 0.7149, "point": 0.84, "rate": "claude-sonnet-5 memo_v2 perturbation consistency", "successes": 42, "trials": 50}, {"high": 0.0973, "low": 0.0783, "point": 0.0874, "rate": "Logistic Regression precision at review capacity", "successes": 297, "trials": 3400}, {"high": 0.1004, "low": 0.0811, "point": 0.0903, "rate": "HistGradient Boosting precision at review capacity", "successes": 307, "trials": 3400}]}
```
