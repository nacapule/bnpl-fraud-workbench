# Detection appendix

How well each policy's score ranks fraud before any threshold is set, and what each
policy did with its score on the same replay. The worlds are synthetic, so a score's
ranking measures how distinct the generator makes each fraud pattern, not how a model
would do on real traffic ([methods](../docs/methods.md#limits)). The policies, their
features and how they were fitted are in the [methods](../docs/methods.md#policies).

## Ranking and calibration on the validation window

The classifiers were fitted on each seed's fit window ({{ protocol:windows.fit.start | date }} to {{ protocol:windows.fit.end | date }}) with the labels known by the classifier freeze ({{ protocol:freezes.classifier | date }}), and one isotonic calibrator per score was fitted on the calibration month with the labels known by {{ protocol:freezes.calibrator | date }}. Both were then measured on the validation window's processor-approved orders whose label was known by the policy freeze, before any threshold was tuned.

| Score | Average precision | Brier score |
| --- | ---: | ---: |
| Rule score (incumbent rules; the hybrid's decline score) | {{ fit.detection.rules.average_precision.validation | num:3 }} | {{ fit.detection.rules.brier.validation | num:4 }} |
| depth-3 tree | {{ fit.detection.tree.average_precision.validation | num:3 }} | {{ fit.detection.tree.brier.validation | num:4 }} |
| logistic regression | {{ fit.detection.logistic.average_precision.validation | num:3 }} | {{ fit.detection.logistic.brier.validation | num:4 }} |
| gradient boosting (also the hybrid's review score and the expected-loss policy's probability) | {{ fit.detection.boosting.average_precision.validation | num:3 }} | {{ fit.detection.boosting.brier.validation | num:4 }} |

| Window | Orders with a known label | Of which fraud |
| --- | ---: | ---: |
| Fit | {{ fit.detection.fit.orders | num:0 }} | {{ fit.detection.fit.positives | num:0 }} |
| Calibration | {{ fit.detection.calibration.orders | num:0 }} | {{ fit.detection.calibration.positives | num:0 }} |
| Validation | {{ fit.detection.validation.orders | num:0 }} | {{ fit.detection.validation.positives | num:0 }} |

*Average precision* is computed on each raw score: the mean, over the fraud orders, of
the precision among all orders scored at or above that order. A score that ranked at
random would have an average precision close to the window's fraud share (fraud
orders divided by orders in the second table). *Brier score* is the mean squared
difference between the calibrated probability and the label (one for fraud, zero
otherwise); lower is better. Unknown labels are left out, never counted as
legitimate. All figures are means over the final seeds, one world each; the counts are
means per world.

These are supporting rows. A score that ranks well can still do poorly as a policy,
because the replay also decides how many orders reach review before they ship, what
the allotment allows and what turning away good customers costs.

## Every policy on the same replay

The test window in the primary cell (baseline world, base allotment, current shift
layout, each policy with its own history): the same orders, analyst, allotment and
ledger for every policy.

| Policy | Ledger net against the incumbent | Across seeds | Fraud loss (bps of GMV) | Legitimate declined per 10,000 | Legitimate held per 10,000 | Review minutes used |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| approve-all | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.approve_all | usd:signed }} | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.approve_all | signs }} | {{ evaluate.loss_of_gmv.baseline.base.approve_all | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.approve_all | num:1 }} | {{ evaluate.legitimate_held_per_10k.baseline.base.approve_all | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.approve_all }} |
| incumbent rules | reference | reference | {{ evaluate.loss_of_gmv.baseline.base.incumbent_rules | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.incumbent_rules | num:1 }} | {{ evaluate.legitimate_held_per_10k.baseline.base.incumbent_rules | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.incumbent_rules }} |
| depth-3 tree | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.tree_depth3 | usd:signed }} | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.tree_depth3 | signs }} | {{ evaluate.loss_of_gmv.baseline.base.tree_depth3 | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.tree_depth3 | num:1 }} | {{ evaluate.legitimate_held_per_10k.baseline.base.tree_depth3 | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.tree_depth3 }} |
| logistic regression | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.logistic | usd:signed }} | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.logistic | signs }} | {{ evaluate.loss_of_gmv.baseline.base.logistic | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.logistic | num:1 }} | {{ evaluate.legitimate_held_per_10k.baseline.base.logistic | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.logistic }} |
| gradient boosting | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.boosting | usd:signed }} | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.boosting | signs }} | {{ evaluate.loss_of_gmv.baseline.base.boosting | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.boosting | num:1 }} | {{ evaluate.legitimate_held_per_10k.baseline.base.boosting | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.boosting }} |
| hybrid | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.hybrid | usd:signed }} | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.hybrid | signs }} | {{ evaluate.loss_of_gmv.baseline.base.hybrid | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.hybrid | num:1 }} | {{ evaluate.legitimate_held_per_10k.baseline.base.hybrid | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.hybrid }} |
| expected loss | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.expected_loss | usd:signed }} | {{ evaluate.net_contribution.vs_incumbent_rules.baseline.base.expected_loss | signs }} | {{ evaluate.loss_of_gmv.baseline.base.expected_loss | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.expected_loss | num:1 }} | {{ evaluate.legitimate_held_per_10k.baseline.base.expected_loss | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.expected_loss }} |

*Ledger net against the incumbent:* the policy's net cash for the window's orders
minus the incumbent's in the same world, mean over seeds, before the friction cost
and the analyst allotment that the recommendation rule deducts (the
[operating review](operating-review.md) gives the rule's figures). *Across seeds:*
how many worlds the difference was positive or negative in. Rates pool every seed's
orders. The other measures are defined in the [operating review](operating-review.md).

## What each policy prevented, by label

{{ table:replay.prevented_by_pattern where family=baseline sum prevented_cents by basis across policy | basis "Adjudicated label" label, incumbent_rules "Incumbent rules" usd, tree_depth3 "Depth-3 tree" usd, logistic "Logistic regression" usd, boosting "Gradient boosting" usd, hybrid "Hybrid" usd, expected_loss "Expected loss" usd }}

{{ table:replay.prevented_by_pattern where family=baseline policy=approve_all sum orders, approve_all_net_cents by basis | basis "Adjudicated label" label, orders "Orders" count, approve_all_net_cents "Net cash under approve-all" usd }}

*Prevented* is the policy's net cash on the orders with that label minus approve-all's
on the same orders, summed over the final seeds' baseline worlds in the test window.
For fraud labels it is loss avoided; for "no finding" and credit loss a negative
figure is margin given up on good orders that were declined or cancelled. The second
table gives the orders and approve-all's net cash for each label over the same worlds,
so each row's prevented figure can be read against what was at stake.

## The policy's own history (a diagnostic)

Each policy decides with a context built from the history its own decisions produced:
an account it declined has no approved order or repaid installment afterwards. The
diagnostic replays every policy with the outcome-derived columns frozen at their
approve-all values, so the difference shows how much each policy's results depend on
its own history. It is not part of the recommendation rule.

| Policy | Ledger net, own history | Ledger net, approve-all history | Difference | Fraud loss difference | Legitimate declined difference per 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| approve-all | {{ evaluate.net_contribution.baseline.base.approve_all }} | {{ evaluate.net_contribution.baseline.base_frozen_history.approve_all }} | {{ evaluate.net_contribution.baseline.base.approve_all - evaluate.net_contribution.baseline.base_frozen_history.approve_all | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.approve_all - evaluate.loss_of_gmv.baseline.base_frozen_history.approve_all | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.approve_all - evaluate.legitimate_declined_per_10k.baseline.base_frozen_history.approve_all | num:1:signed }} |
| incumbent rules | {{ evaluate.net_contribution.baseline.base.incumbent_rules }} | {{ evaluate.net_contribution.baseline.base_frozen_history.incumbent_rules }} | {{ evaluate.net_contribution.baseline.base.incumbent_rules - evaluate.net_contribution.baseline.base_frozen_history.incumbent_rules | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.incumbent_rules - evaluate.loss_of_gmv.baseline.base_frozen_history.incumbent_rules | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.incumbent_rules - evaluate.legitimate_declined_per_10k.baseline.base_frozen_history.incumbent_rules | num:1:signed }} |
| depth-3 tree | {{ evaluate.net_contribution.baseline.base.tree_depth3 }} | {{ evaluate.net_contribution.baseline.base_frozen_history.tree_depth3 }} | {{ evaluate.net_contribution.baseline.base.tree_depth3 - evaluate.net_contribution.baseline.base_frozen_history.tree_depth3 | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.tree_depth3 - evaluate.loss_of_gmv.baseline.base_frozen_history.tree_depth3 | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.tree_depth3 - evaluate.legitimate_declined_per_10k.baseline.base_frozen_history.tree_depth3 | num:1:signed }} |
| logistic regression | {{ evaluate.net_contribution.baseline.base.logistic }} | {{ evaluate.net_contribution.baseline.base_frozen_history.logistic }} | {{ evaluate.net_contribution.baseline.base.logistic - evaluate.net_contribution.baseline.base_frozen_history.logistic | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.logistic - evaluate.loss_of_gmv.baseline.base_frozen_history.logistic | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.logistic - evaluate.legitimate_declined_per_10k.baseline.base_frozen_history.logistic | num:1:signed }} |
| gradient boosting | {{ evaluate.net_contribution.baseline.base.boosting }} | {{ evaluate.net_contribution.baseline.base_frozen_history.boosting }} | {{ evaluate.net_contribution.baseline.base.boosting - evaluate.net_contribution.baseline.base_frozen_history.boosting | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.boosting - evaluate.loss_of_gmv.baseline.base_frozen_history.boosting | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.boosting - evaluate.legitimate_declined_per_10k.baseline.base_frozen_history.boosting | num:1:signed }} |
| hybrid | {{ evaluate.net_contribution.baseline.base.hybrid }} | {{ evaluate.net_contribution.baseline.base_frozen_history.hybrid }} | {{ evaluate.net_contribution.baseline.base.hybrid - evaluate.net_contribution.baseline.base_frozen_history.hybrid | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.hybrid - evaluate.loss_of_gmv.baseline.base_frozen_history.hybrid | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.hybrid - evaluate.legitimate_declined_per_10k.baseline.base_frozen_history.hybrid | num:1:signed }} |
| expected loss | {{ evaluate.net_contribution.baseline.base.expected_loss }} | {{ evaluate.net_contribution.baseline.base_frozen_history.expected_loss }} | {{ evaluate.net_contribution.baseline.base.expected_loss - evaluate.net_contribution.baseline.base_frozen_history.expected_loss | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.expected_loss - evaluate.loss_of_gmv.baseline.base_frozen_history.expected_loss | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.expected_loss - evaluate.legitimate_declined_per_10k.baseline.base_frozen_history.expected_loss | num:1:signed }} |

*Ledger net* is the mean over seeds of each world's net cash for the test window's
orders; the differences are own history minus approve-all history.

## Tuned thresholds

Each policy's review and decline thresholds were chosen per seed on the validation
window through the replay, from cut-points its scores attain at preset review and
decline rates ([methods](../docs/methods.md#threshold-tuning)). The chosen points,
whether either sits at the edge of the searched grid, and every replayed point are in
`results/tune.json` (`tune.chosen` and `tune.frontier`).

The gradient-boosting policy's chosen points, seed by seed (the canonical world, seed {{ protocol:seeds.canonical }}, is tuned too, for the case files; it is not one of the final seeds):

{{ table:tune.chosen where policy=boosting | seed "Seed" id, review_threshold "Review threshold" num:3, decline_threshold "Decline threshold" num:3, review_on_boundary "Review at the grid's edge", decline_on_boundary "Decline at the grid's edge", feasible_points "Feasible points" count, points "Points replayed" count }}

*Review threshold* and *decline threshold* are cut-points on boosting's own score, which ranks orders but is not a probability (the classifiers weight the two classes equally in fitting; only the expected-loss policy routes on calibrated probabilities); "n/a" means the chosen point has no review route on that seed, so the policy only approves or declines there. *At the grid's edge:* the threshold is the cut-point for the highest rate searched, so a wider grid might choose another point. *Points replayed:* the grid points the frozen screen replayed; *feasible points:* those whose offered review minutes fit the allotment.

Every chosen point that sits at the edge of the grid, for any policy. Decline thresholds at the edge:

{{ table:tune.chosen where decline_on_boundary=true | policy "Policy" label, seed "Seed" id, review_threshold "Review threshold" num:3, decline_threshold "Decline threshold" num:3, review_on_boundary "Review at the edge" }}

Review thresholds at the edge:

{{ table:tune.chosen where review_on_boundary=true | policy "Policy" label, seed "Seed" id, review_threshold "Review threshold" num:3, decline_threshold "Decline threshold" num:3, decline_on_boundary "Decline at the edge" }}

Thresholds are on each policy's own score: the summed rule weights for today's rules and for the hybrid's decline threshold, and the classifier's own score for the depth-3 tree, logistic regression and gradient boosting, and for the hybrid's review threshold, which uses boosting's score. A decline threshold at the edge is the cut-point for the highest decline rate searched, the lowest score the grid would decline at.
