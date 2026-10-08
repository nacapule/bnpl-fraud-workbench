# Operating review: which review policy to run

**For** the fraud operations lead. **The question:** which review-and-intervention
policy should a small pay-in-4 fraud team run, with a fixed allotment of analyst
time, evidence and labels that arrive late, goods that ship within hours, and a real
cost to holding or declining good customers? And when does the answer change?

**The evidence** is simulated, not a real portfolio: synthetic marketplaces generated from one set of stated assumptions ([methods](../docs/methods.md)). Every policy was tuned on an earlier window and frozen, then replayed on the orders placed in the test window, from {{ protocol:windows.test.start | date }} until {{ protocol:windows.test.end | date }}, with outcomes observed until {{ protocol:windows.follow_up.end | date }}. Each final seed is one simulated world; figures are means over those worlds, and a paired difference is said to be "positive on k of n seeds" when it is above zero in k of the n worlds.

Unless a table says otherwise, figures are for the **primary cell**: the baseline
world, the base allotment on the current shift layout, each policy seeing the history
its own decisions produced, and the evidence-based reviewer with standard
verification rates.

## Decision

<!-- Phase B, from evaluate.flips (primary row) and evaluate.recommendation:
- If the incumbent misses a criterion in the primary cell (evaluate.flips
  incumbent_misses), open with that: today's rules miss the target on <criteria>;
  the rule then holds challengers to today's level on that criterion.
- The rule's outcome: recommend <policy> or keep the incumbent rules, in one sentence.
- The deciding numbers through placeholders: the recommended (or best eligible)
  challenger's mean gain and range (the seeds format on
  evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.<policy>), never a
  p-value for the selected policy.
- The verdict line: holds in N of M cells (evaluate.recommendation.holds numerator,
  denominator), pointing to "When the answer changes".
- One line on what the memo cannot say (synthetic world; see Limits). -->

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

Every policy's thresholds were chosen on the validation window through the same
replay, before the test window was seen ([methods](../docs/methods.md#threshold-tuning)).
A rule written down before any final world existed then chooses among them:

- **Eligibility.** Lost legitimate customers (below) within {{ protocol:reporting.recommendation_rule.eligibility.lost_legitimate_per_10000.mean_at_most }} per 10,000 legitimate orders on the mean over seeds and within {{ protocol:reporting.recommendation_rule.eligibility.lost_legitimate_per_10000.any_seed_at_most }} on every seed; legitimate orders held within {{ protocol:reporting.recommendation_rule.eligibility.held_legitimate_per_10000.mean_at_most }} and {{ protocol:reporting.recommendation_rule.eligibility.held_legitimate_per_10000.any_seed_at_most }}; and at each queue priority, a share of entries decided within the service target of {{ protocol:reporting.recommendation_rule.eligibility.service_share_at_least | num:2 }} or above (a priority with too few entries is reported, not assessed). Where today's rules miss a criterion, challengers are held only to today's level on it.
- **Hurdle.** A mean gain in rule net contribution over the incumbent of ${{ protocol:reporting.recommendation_rule.hurdle.mean_improvement_usd_per_1000_orders }} per 1,000 orders or above, an allowance for the cost of changing a policy, and a positive gain on nearly every seed (the count needed is in the table). This is a consistency bar across simulated worlds, not a significance test.
- **Choice.** Among eligible challengers that clear the hurdle, the one with the highest mean gain. If none clears it, the incumbent stays.

{{ table:evaluate.recommendation where cell=primary | policy "Policy" label, rule_net_vs_incumbent_rules_per_1000_mean_cents "Mean gain" usd:signed, rule_net_vs_incumbent_rules_per_1000_min_cents "Lowest seed" usd:signed, rule_net_vs_incumbent_rules_per_1000_max_cents "Highest seed" usd:signed, positive_seeds_count "Seeds positive" count, positive_seeds_needed_count "Needed" count, eligible "Eligible", fails "Misses" label, recommended "Recommended" }}

*Mean gain:* per 1,000 orders, the policy's rule net contribution minus the incumbent's in the same world, mean over seeds. The orders are those the policy decides: processor-approved checkouts in the test window. *Rule net contribution* is the ledger's net cash for the window's orders, minus ${{ config:policy:costs.false_decline_ltv_usd | num:0 }} (the lifetime-value proxy) for each lost legitimate customer and minus the analyst allotment at ${{ config:policy:costs.analyst_loaded_hourly_usd | num:0 }} an hour, unused minutes included; a policy that sends nothing to review is charged no allotment. *Misses:* the criteria a policy fails; for the incumbent, which is the reference and has no gain of its own, the criteria today's rules miss.

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
window's gross merchandise value. *Legitimate declined:* as lost customers, without
the cancelled holds, pooled over seeds. *Review minutes used:* minutes of review work
begun on the window's orders (including work finished after it ended) over the
minutes allotted to the queue during the window. *Decided after shipping:* reviews
whose first decision came after the goods had shipped, when a hold can no longer
stop the shipment; mean per world.

![Fraud loss against lost legitimate customers, each policy in the primary cell](figures/frontier.svg)

<!-- Phase B: two or three sentences on what these tables show, directions only through
claims (report/claims.yaml), e.g. a policy's loss against the incumbent's (sign test)
and its lost customers against the incumbent's, never for the selected policy. -->

## Review capacity and staffing

Capacity here is analyst time allotted to the fraud queue: one analyst on each covered shift, with {{ config:policy:capacity.levels.base.review_minutes_per_shift.early }} minutes per shift for review and escalation work at the base level. At this world's volume a single analyst's shift could clear the whole queue many times over, so headcount cannot be the binding limit; the allotment is. The base was sized on today's queue before any policy comparison was read ([methods](../docs/methods.md#capacity)).

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

<!-- Phase B: what the staffing comparison shows (does the answer depend on the
allotment or the layout; is the incumbent's service miss a staffing problem), with
directions only through claims or the rule's own outcome per column. -->

## What checkout review cannot reach

Some loss sits outside what a decision at checkout can stop. Merchant bust-out is the
merchant's fraud: its customers are genuine, pass every check, and claim non-delivery
only after the merchant has closed. Never-pay customers are first-party: they pass
verification at customers' rates, because checks establish who is ordering and not
what they intend, so review clears them unless an earlier outcome already settles the
order; what stops them is a decline on the score at checkout. Credit loss is not
fraud and appears for scale.

{{ table:replay.prevented_by_pattern where family=baseline policy=approve_all sum orders, approve_all_net_cents by basis | basis "Adjudicated label" label, orders "Orders" count, approve_all_net_cents "Net cash under approve-all" usd }}

*Orders and net cash* are summed over the final seeds' baseline worlds for the test
window's orders, by the label each order earned; a negative figure is a loss. What each
policy prevented in each row is in the [detection appendix](appendix-detection.md).

<!-- Phase B: one or two sentences placing the unreachable share in context, from the
table (no computed share is rendered; state the rows, not a percentage). -->

## When the answer changes

The rule was applied again, unchanged, in each cell that differs from the primary cell in one respect, never two at once, with the thresholds tuned in the primary cell: the two other world families (an acquisition surge of new customers and a shift towards account takeover and aged stolen-card accounts), goods shipping in half or twice the time, the low and high allotments, the evening layout, weaker verification, and a lifetime-value proxy of ${{ protocol:sensitivity.ltv_proxy_usd.0 }} or ${{ protocol:sensitivity.ltv_proxy_usd.1 }} in place of ${{ config:policy:costs.false_decline_ltv_usd | num:0 }}.

{{ table:evaluate.flips | cell "Cell" label, outcome "Outcome" label, recommended "Recommended" label, best_challenger "Leading challenger" label, reason "Against the primary cell" label, incumbent_misses "Today's rules miss" label }}

The primary cell's outcome holds in {{ evaluate.recommendation.holds | numerator }} of the {{ evaluate.recommendation.holds | denominator }} other cells. *Leading challenger:* the eligible challenger with the highest mean gain, or the highest of all when none is eligible.

![The rule's outcome in each operating cell](figures/operating_cells.svg)

<!-- Phase B: name the cells where the answer changes and why (the flip table's reason),
and what that means for the decision. -->

## Piloting it

<!-- Phase B outline:
- What to pilot: the rule's outcome (switch to <policy>, or keep the rules and fix
  what they miss, e.g. P1 service if it is still missed).
- Shadow first: score live orders with the candidate beside today's rules, without
  acting, for a fixed period; compare its routing with the rules' on the same orders.
- Then a randomized share of traffic, with the rule's guardrails as stop criteria
  (lost and held legitimate customers per 10,000, service at each priority), checked
  weekly, and loss measured once labels mature (the label horizon).
- Check what the replay assumes and cannot test: attackers adapting to declines,
  customers leaving after friction, real verification pass rates, the real allotment.
- The flip table's cells that change the answer are the conditions to watch. -->

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

The full list, and which sensitivity tests which assumption, is in the
[methods](../docs/methods.md#limits).
