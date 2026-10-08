# Operating review: which review policy to run

**For** the fraud operations lead. **The question:** which review-and-intervention
policy should a small pay-in-4 fraud team run, with a fixed allotment of analyst
time, evidence and labels that arrive late, goods that ship within hours, and a real
cost to holding or declining good customers? And when does the answer change?

**The evidence** is simulated, not a real portfolio: synthetic marketplaces generated from one set of stated assumptions ([methods](../docs/methods.md)). Every policy was tuned on an earlier window and frozen, then replayed on the orders placed in the test window, from {{ protocol:windows.test.start | date }} until {{ protocol:windows.test.end | date }}, with outcomes observed until {{ protocol:windows.follow_up.end | date }}. Each final seed is one simulated world; figures are means over those worlds unless a caption says they are pooled, and a paired difference is said to be "positive on k of n seeds" when it is above zero in k of the n worlds.

Unless a table says otherwise, figures are for the **primary cell**: the baseline
world, the base allotment on the current shift layout, each policy seeing the history
its own decisions produced, and the evidence-based reviewer with standard
verification rates.

## Decision

**Replace today's rules with the gradient-boosting policy, after a pilot.** In the primary cell, it is the eligible challenger with the highest mean gain in rule net contribution against today's rules: {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.boosting | usd:signed }} per 1,000 orders decided, clearing the hurdle of ${{ protocol:reporting.recommendation_rule.hurdle.mean_improvement_usd_per_1000_orders }}. The gain is {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.boosting | signs }}.

The policy acts mainly through score-based declines at checkout ([operations appendix](appendix-operations.md#the-queue)). It loses {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.boosting | num:1 }} legitimate customers per 10,000 legitimate orders on average, inside the guardrail of {{ protocol:reporting.recommendation_rule.eligibility.lost_legitimate_per_10000.mean_at_most }}, against {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.incumbent_rules | num:1 }} under today's rules. The cost of those customers, at the lifetime-value proxy, is already deducted from its gain.

**Today's rules miss the service target.** At P1 and P2, their share of queue entries decided within the target does not meet the required share (table below), so the rule uses today's level as the service criterion for challengers at both priorities. Gradient boosting meets the P2 target and today's level at P1, where it still misses the target.

**The answer holds in {{ evaluate.recommendation.holds | numerator }} of the {{ evaluate.recommendation.holds | denominator }} other operating cells.** In the acquisition surge of new customers, gradient boosting misses the guardrail on lost legitimate customers, so today's rules stay ([When the answer changes](#when-the-answer-changes)). The world is synthetic, so these are results about how the policies behave under stated assumptions, not estimates for a real book ([Limits](#limits-that-bear-on-this-decision)).

## The options and the rule that chooses

Seven policies saw the same orders, the same simulated analyst, the same allotment
of review time and the same cash ledger. Each routes an order once, at checkout:
decline it, send it to review, or approve it.

| Policy | Sends to review | Declines at checkout |
| --- | --- | --- |
| approve-all (reference) | nothing | nothing |
| incumbent rules | orders whose fraud-policy rule weights (FP-2 §6.2) sum to the review threshold | orders whose rule weights sum to the decline threshold |
| depth-3 tree | a decision tree's score on four features named in advance: account age, the device's age on the account, recent accounts on the device, and the amount against its category's median | the same score at the decline threshold |
| logistic regression | its score on the context's model features | the same score at the decline threshold |
| gradient boosting | its score on the same features | the same score at the decline threshold |
| hybrid | the gradient-boosting score | the rule score |
| expected loss | calibrated fraud probability times the cash at risk | orders where the expected loss avoided covers the expected cost of declining a good customer |

Every policy's tunable thresholds (all but the expected-loss policy's decline rule,
which is fixed) were chosen on the validation window through the same replay, before
the test window was seen ([methods](../docs/methods.md#threshold-tuning)).
A rule written down before any final world existed then chooses among them:

- **Eligibility.** Lost legitimate customers (below) within {{ protocol:reporting.recommendation_rule.eligibility.lost_legitimate_per_10000.mean_at_most }} per 10,000 legitimate orders on the mean over seeds and within {{ protocol:reporting.recommendation_rule.eligibility.lost_legitimate_per_10000.any_seed_at_most }} on every seed; legitimate orders held within {{ protocol:reporting.recommendation_rule.eligibility.held_legitimate_per_10000.mean_at_most }} and {{ protocol:reporting.recommendation_rule.eligibility.held_legitimate_per_10000.any_seed_at_most }}; and at each queue priority, a minimum share of {{ protocol:reporting.recommendation_rule.eligibility.service_share_at_least | num:2 }} of the entries decided within the service target (a priority without enough entries is reported, not assessed). Where today's rules miss a criterion, challengers are held only to today's level on it.
- **Hurdle.** A mean gain in rule net contribution over the incumbent of ${{ protocol:reporting.recommendation_rule.hurdle.mean_improvement_usd_per_1000_orders }} per 1,000 orders decided or above, an allowance for the cost of changing a policy, and a positive gain on nearly every seed (the count needed is in the table). This is a consistency bar across simulated worlds, not a significance test.
- **Choice.** Among eligible challengers that clear the hurdle, the one with the highest mean gain. If none clears it, the incumbent stays.

{{ table:evaluate.recommendation where cell=primary | policy "Policy" label, rule_net_vs_incumbent_rules_per_1000_mean_cents "Mean gain" usd:signed, rule_net_vs_incumbent_rules_per_1000_min_cents "Lowest seed" usd:signed, rule_net_vs_incumbent_rules_per_1000_max_cents "Highest seed" usd:signed, positive_seeds_count "Seeds positive" count, positive_seeds_needed_count "Needed" count, eligible "Eligible", fails "Misses" label, recommended "Recommended" }}

*Mean gain:* the policy's rule net contribution minus the incumbent's in the same world, per 1,000 orders decided, mean over seeds. These are processor-approved checkouts in the test window. *Rule net contribution* is the ledger's net cash for the window's orders, minus ${{ config:policy:costs.false_decline_ltv_usd | num:0 }} (the lifetime-value proxy) for each lost legitimate customer and minus the analyst allotment at ${{ config:policy:costs.analyst_loaded_hourly_usd | num:0 }} an hour, unused minutes included; a policy that sends nothing to review is charged no allotment. *Misses:* the criteria a policy fails; for the incumbent, which is the reference and has no gain of its own, the criteria today's rules miss.

## Customers, service and loss

{{ table:evaluate.recommendation where cell=primary | policy "Policy" label, lost_legitimate_per_10k_mean_bps "Lost per 10k (mean)" num:1, lost_legitimate_per_10k_max_bps "Lost per 10k (highest seed)" num:1, held_legitimate_per_10k_mean_bps "Held per 10k (mean)" num:1, service_p1_entries_count "P1 entries" count, service_p1_in_time_mean_share "P1 in time" pct, service_p2_entries_count "P2 entries" count, service_p2_in_time_mean_share "P2 in time" pct }}

*Lost legitimate customers:* legitimate orders declined at checkout, refused because the account had been blocked, declined or escalated after review, or cancelled after a verification request nobody answered, per 10,000 legitimate orders. *Held:* legitimate orders asked to verify, per 10,000 legitimate orders. "Legitimate" is the adjudicated label at the end of observation: no fraud finding, credit losses included. *In time:* the share of the orders entering the review queue at that priority that were decided within its target ({{ config:policy:sla.target_hours.P1 }} service hours for P1, {{ config:policy:sla.target_hours.P2 }} for P2, counted from {{ config:policy:sla.calendar.start }} to {{ config:policy:sla.calendar.end }} every day), mean over seeds; entries are pooled over seeds. P0 entries are too few to assess in this world ([operations appendix](appendix-operations.md)).

| Policy | Ledger net against approve-all | Fraud loss (bps of GMV) | Legitimate declined per 10,000 | Review minutes used | Decided after shipping |
| --- | ---: | ---: | ---: | ---: | ---: |
| approve-all | reference | {{ evaluate.loss_of_gmv.baseline.base.approve_all | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.approve_all | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.approve_all }} | {{ evaluate.decided_after_shipping.baseline.base.approve_all | num:1 }} |
| incumbent rules | {{ evaluate.net_contribution.vs_approve_all.baseline.base.incumbent_rules | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.incumbent_rules | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.incumbent_rules | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.incumbent_rules }} | {{ evaluate.decided_after_shipping.baseline.base.incumbent_rules | num:1 }} |
| depth-3 tree | {{ evaluate.net_contribution.vs_approve_all.baseline.base.tree_depth3 | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.tree_depth3 | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.tree_depth3 | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.tree_depth3 }} | {{ evaluate.decided_after_shipping.baseline.base.tree_depth3 | num:1 }} |
| logistic regression | {{ evaluate.net_contribution.vs_approve_all.baseline.base.logistic | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.logistic | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.logistic | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.logistic }} | {{ evaluate.decided_after_shipping.baseline.base.logistic | num:1 }} |
| gradient boosting | {{ evaluate.net_contribution.vs_approve_all.baseline.base.boosting | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.boosting | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.boosting | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.boosting }} | {{ evaluate.decided_after_shipping.baseline.base.boosting | num:1 }} |
| hybrid | {{ evaluate.net_contribution.vs_approve_all.baseline.base.hybrid | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.hybrid | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.hybrid | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.hybrid }} | {{ evaluate.decided_after_shipping.baseline.base.hybrid | num:1 }} |
| expected loss | {{ evaluate.net_contribution.vs_approve_all.baseline.base.expected_loss | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base.expected_loss | num:1 }} | {{ evaluate.legitimate_declined_per_10k.baseline.base.expected_loss | num:1 }} | {{ evaluate.review_minutes_used_share.baseline.base.expected_loss }} | {{ evaluate.decided_after_shipping.baseline.base.expected_loss | num:1 }} |

*Ledger net against approve-all:* the policy's net cash for the window's orders minus
approve-all's in the same world, mean over seeds, before any friction or staffing
cost. *Fraud loss:* cash lost on orders labelled fraud, in basis points of the
window's gross merchandise value, pooled over seeds. *Legitimate declined:* as lost customers, without
the cancelled holds, pooled over seeds. *Review minutes used:* minutes of review work
begun on the window's orders (including work finished after it ended) over the
minutes allotted to the queue during the window. *Decided after shipping:* reviews
whose first decision came after the goods had shipped, when a hold can no longer
stop the shipment; mean per world.

{{ claim:rules-cash-vs-approve-all }} Two challengers are eligible and clear the hurdle: gradient boosting and the hybrid policy. Gradient boosting has the highest mean gain among them.

The hybrid policy keeps today's rules for declines and uses the boosting score to select orders for review. Its mean gain is {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.hybrid | usd:signed }} per 1,000 orders decided, with {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.hybrid | num:1 }} legitimate customers lost per 10,000 legitimate orders, against {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.incumbent_rules | num:1 }} under today's rules. {{ claim:hybrid-loss-vs-rules }} {{ claim:hybrid-declined-vs-rules }}

Logistic regression has the highest mean gain of all, {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.logistic | usd:signed }} per 1,000 orders decided, but is ineligible: its mean loss of {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.logistic | num:1 }} legitimate customers per 10,000 legitimate orders is over the guardrail. {{ claim:logistic-declined-vs-rules }}

The expected-loss policy uses {{ evaluate.review_minutes_used_share.baseline.base.expected_loss }} of the allotted review minutes and misses today's service level at both priorities.

![Fraud loss against lost legitimate customers, each policy in the primary cell](figures/frontier.svg)

## Review capacity and staffing

Capacity here is analyst time allotted to the fraud queue: one analyst on each covered shift, with {{ config:policy:capacity.levels.base.review_minutes_per_shift.early }} minutes per shift for review and escalation work at the base level. At this world's volume, the review work a day brings is a small part of one analyst's productive hours on a shift, so headcount cannot be the binding limit; the allotment is. The base was sized on today's queue before any policy comparison was read ([methods](../docs/methods.md#capacity)).

The table varies the staffing with each policy's thresholds held as tuned at the base
allotment. Each cell is the policy's mean gain over the incumbent in rule net
contribution per 1,000 orders, in that staffing; the cost of the allotment is inside
the measure, so the high allotment pays for its minutes.

| Policy | Low allotment ({{ config:policy:capacity.levels.low.review_minutes_per_shift.early }} min per shift) | Base ({{ config:policy:capacity.levels.base.review_minutes_per_shift.early }}) | High ({{ config:policy:capacity.levels.high.review_minutes_per_shift.early }}) | Evening layout ({{ config:policy:capacity.redesigned.review_minutes_per_shift.early }}) |
| --- | ---: | ---: | ---: | ---: |
| approve-all | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.low.approve_all | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.approve_all | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.high.approve_all | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.redesigned_layout.approve_all | usd:signed }} |
| depth-3 tree | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.low.tree_depth3 | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.tree_depth3 | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.high.tree_depth3 | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.redesigned_layout.tree_depth3 | usd:signed }} |
| logistic regression | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.low.logistic | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.logistic | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.high.logistic | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.redesigned_layout.logistic | usd:signed }} |
| gradient boosting | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.low.boosting | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.boosting | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.high.boosting | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.redesigned_layout.boosting | usd:signed }} |
| hybrid | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.low.hybrid | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.hybrid | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.high.hybrid | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.redesigned_layout.hybrid | usd:signed }} |
| expected loss | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.low.expected_loss | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.expected_loss | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.high.expected_loss | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.redesigned_layout.expected_loss | usd:signed }} |

The evening layout moves the same two shifts later in the day to cover the evening
peak of orders, with the same minutes. Whether a challenger is eligible in each column
is in the flip table below and in the [operations appendix](appendix-operations.md).

![Gain over the incumbent per 1,000 orders at each staffing level](figures/staffing.svg)

Gradient boosting remains recommended at the low and high allotments and on the
evening layout. Service depends on the allotment and shift coverage: today's rules
meet both service targets only at the high allotment; at the low allotment, every
policy that reviews misses both.

On the evening layout, {{ evaluate.decided_after_shipping.baseline.redesigned_layout.incumbent_rules | num:1 }} reviews per world under today's rules are decided after shipment, against {{ evaluate.decided_after_shipping.baseline.base.incumbent_rules | num:1 }} on the current layout. No analyst is on shift before noon, while the service clock starts at {{ config:policy:sla.calendar.start }}, so morning P1 entries wait for the first shift ([operations appendix](appendix-operations.md#staffing)).

## What checkout review cannot reach

Some loss sits outside what the analyst's checks can catch. Merchant bust-out is the
merchant's fraud: its customers are genuine and claim non-delivery only after the
merchant has closed. Never-pay customers are first-party: they pass verification at
genuine customers' rates, because checks establish who is ordering, not what they
intend. Credit loss appears for scale.

{{ table:replay.prevented_by_pattern where family=baseline policy=approve_all sum orders, approve_all_net_cents by basis | basis "Adjudicated label" label, orders "Orders" count, approve_all_net_cents "Net cash under approve-all" usd }}

*Orders and net cash* are summed over the final seeds' baseline worlds for the test
window's orders, by the label each order earned; a negative figure is a loss.

Review settles few of these orders: the analyst usually clears a reviewed never-pay
order and cleared every reviewed bust-out order
([operations appendix](appendix-operations.md#the-analysts-decisions-by-label)). What
can stop them is a decline on the score at checkout; the
[detection appendix](appendix-detection.md#what-each-policy-prevented-by-label) shows
what each policy prevented, by label. Declining a bust-out merchant's orders also
turns away its genuine customers. Their orders count as fraud in this evaluation, so
declining them does not count as lost legitimate customers.

## When the answer changes

The rule was applied again, unchanged, in each cell that differs from the primary cell in one respect, never two at once, with the thresholds tuned in the primary cell: the two other world families (an acquisition surge of new customers and a shift towards account takeover and aged stolen-card accounts), goods shipping in half or twice the time, the low and high allotments, the evening layout, weak verification, and a lifetime-value proxy of ${{ protocol:sensitivity.ltv_proxy_usd.0 }} or ${{ protocol:sensitivity.ltv_proxy_usd.1 }} in place of ${{ config:policy:costs.false_decline_ltv_usd | num:0 }}.

{{ table:evaluate.flips | cell "Cell" label, outcome "Outcome" label, recommended "Recommended" label, best_challenger "Leading challenger" label, reason "Against the primary cell" label, incumbent_misses "Today's rules miss" label }}

The primary cell's outcome holds in {{ evaluate.recommendation.holds | numerator }} of the {{ evaluate.recommendation.holds | denominator }} other cells. *Leading challenger:* the eligible challenger with the highest mean gain; when none is eligible, the challenger with the highest mean gain among all policies.

![The rule's outcome in each operating cell](figures/operating_cells.svg)

The acquisition surge is the only cell where the answer changes, because gradient boosting misses the lost-customer guardrail. {{ claim:surge-boosting-cash }} {{ claim:surge-boosting-declines }} It loses {{ evaluate.rule_lost_legitimate_per_10k.acquisition_surge.base.boosting | num:1 }} legitimate customers per 10,000 legitimate orders on average, over the guardrail; it also goes over the per-seed cap on some seeds.

The hybrid policy is eligible in the surge, but its mean gain of {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.acquisition_surge.base.hybrid | usd:signed }} per 1,000 orders decided is short of the hurdle, so today's rules stay. Reassess the recommendation before using it during a growth campaign.

## Piloting it

1. **Shadow.** Score live orders with the gradient-boosting policy beside today's rules for a full label horizon ({{ protocol:label_horizon_days }} days), without acting on the candidate's scores. Compare their proposed routes on the same orders: which orders each would decline or review. Estimate the candidate's lost-customer rate from its proposed routes as the labels mature; the traffic pilot then checks that estimate against what happens.
2. **A randomized share of traffic.** Route a fixed share of checkouts through the candidate. Use the rule's guardrails as stop criteria, reviewed each week: lost legitimate customers within {{ protocol:reporting.recommendation_rule.eligibility.lost_legitimate_per_10000.mean_at_most }} per 10,000 legitimate orders, legitimate orders held within {{ protocol:reporting.recommendation_rule.eligibility.held_legitimate_per_10000.mean_at_most }}, and service at each priority held to today's level. Assess the customer guardrails and fraud loss on cohorts with mature labels. Check service from queue records.
3. **Keep the queue's minutes.** Gradient boosting uses {{ evaluate.review_minutes_used_share.baseline.base.boosting }} of the allotted review minutes, against {{ evaluate.review_minutes_used_share.baseline.base.incumbent_rules }} under today's rules, and sends nothing to review on some seeds ([operations appendix](appendix-operations.md#the-queue)). Keep the allotment through the pilot: the candidate's P1 service still misses the target at the base allotment, and only the high allotment meets it ([operations appendix](appendix-operations.md#operating-sensitivities)). Plan staffing from the workload the shadow period projects, then check it against the candidate's queue in the traffic pilot.
4. **Pause it for a growth campaign.** The acquisition surge is the only cell where the candidate misses its guardrail. Use today's rules, or reassess the candidate's lost-customer rate, before a campaign brings in new customers.
5. **Check what the replay cannot.** Use the pilot to check attacker adaptation, customers leaving after a decline or hold, verification pass rates and the review allotment against live data. These are assumptions the replay cannot verify.

## Limits that bear on this decision

- **A synthetic world.** Fraud patterns, signals and rates are the generator's
  choices; the results show how the policies behave under these assumptions, not real
  fraud rates or detection performance.
- **Seeds measure variation between simulated worlds** that share one generator and
  its parameters, not whether those parameters are right.
- **Attackers do not adapt.** A declined fraudster does not return with new details,
  so the replay can overstate what declines and blocks are worth.
- **Customers do not leave.** A lost legitimate customer costs the flat
  lifetime-value proxy; churn after a hold is not simulated.
- **Labels come from the approve-all world.** A prevented order is judged by what it
  would have become; no action re-labels an order.
- **The analyst follows one fixed procedure**, and check pass rates are assumptions;
  the weak-verification cell tests one change to them.
- **Costs and caps are assumptions:** the lifetime-value proxy, the analyst hour, the
  hurdle and the guardrails are stated risk appetite, not benchmarks.
- **The searched grid has an edge.** On some seeds, the model policies' chosen decline
  thresholds sit at the grid's decline-rate ceiling. Extending the grid could bring
  other orders into the decline band. Gradient boosting sends nothing to review on some seeds
  ([detection appendix](appendix-detection.md#tuned-thresholds)).

The full list, and which sensitivity tests which assumption, is in the
[methods](../docs/methods.md#limits).
