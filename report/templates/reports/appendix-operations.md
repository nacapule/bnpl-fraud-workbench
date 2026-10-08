# Operations appendix

The review queue and its service times, staffing, the simulated analyst's decisions
by label, and the operating sensitivities. The worlds are synthetic; queue and
service figures follow from the stated review times, shift layouts and allotments
([methods](../docs/methods.md#capacity), [reviewer](../docs/methods.md#reviewer)).
Unless a table says otherwise it covers the primary cell (baseline world, base
allotment, current shift layout, each policy with its own history, the evidence-based
reviewer with standard verification) on the test window.

## The queue

{{ table:replay.outcomes where family=baseline capacity_level=base layout=current history=policy reviewer=evidence verification=verification sum reviews, holds, checks_run, escalations, accounts_blocked, decided_after_shipping by policy | policy "Policy" label, reviews "Reviews" count, holds "Holds" count, checks_run "Checks" count, escalations "Escalations" count, accounts_blocked "Accounts blocked" count, decided_after_shipping "Decided after shipping" count, rows_count "Worlds" count }}

*Reviews:* orders that entered the queue. *Holds:* orders paused for verification
(at most {{ config:policy:actions.hold_max_hours }} hours; an order the checks have not
cleared by then is cancelled before it ships). *Checks:* verification checks started.
*Escalations:* declines that also blocked linked accounts and added
{{ config:policy:actions.senior_review_minutes }} minutes of senior review.
*Accounts blocked:* distinct accounts whose later orders were refused. *Decided after
shipping:* reviews first decided after the goods shipped. All are totals over the
final seeds' worlds (the last column counts them).

{{ table:replay.outcomes where family=baseline capacity_level=base layout=current history=policy reviewer=evidence verification=verification sum review_minutes_offered, review_minutes_used, senior_minutes, available_minutes by policy | policy "Policy" label, review_minutes_offered "Minutes of work offered" count, review_minutes_used "Minutes used" count, senior_minutes "Of which senior review" count, available_minutes "Minutes allotted" count, rows_count "Worlds" count }}

*Minutes of work offered:* the review time of every order that reached the queue plus
escalations' senior review. *Minutes used:* the part of that work begun, including
work on the window's orders after the window ended. *Minutes allotted:* the
allotment of the shifts inside the window, used or not. Totals over the worlds.

{{ table:replay.outcomes where family=baseline capacity_level=base layout=current history=policy reviewer=evidence verification=verification mean wait_p50_minutes, wait_p90_minutes, max_backlog by policy | policy "Policy" label, wait_p50_minutes "Median minutes to decision" num:0, wait_p90_minutes "90th percentile" num:0, max_backlog "Largest backlog" num:1, rows_count "Worlds" count }}

*Minutes to decision:* wall-clock minutes from queue entry to the analyst's first
decision, among decided reviews; the median and the 90th percentile are each world's,
then the mean over seeds. *Largest backlog:* the most orders waiting at once in a
world, mean over seeds. Approve-all sends nothing to review, so its row reads zero.

{{ table:evaluate.recommendation where cell=primary | policy "Policy" label, service_p0_entries_count "P0 entries" count, service_p0_in_time_mean_share "P0 in time" pct, service_p1_entries_count "P1 entries" count, service_p1_in_time_mean_share "P1 in time" pct, service_p2_entries_count "P2 entries" count, service_p2_in_time_mean_share "P2 in time" pct, fails "Misses" label }}

*In time:* the share of the orders entering the queue at that priority that were
decided within its target ({{ config:policy:sla.target_hours.P0 }},
{{ config:policy:sla.target_hours.P1 }} and {{ config:policy:sla.target_hours.P2 }}
service hours for P0 to P2, counted from {{ config:policy:sla.calendar.start }} to
{{ config:policy:sla.calendar.end }} every day; an order still undecided at the end
of observation is a miss), mean over the seeds with entries. Entries are pooled over
seeds; a priority with fewer than
{{ protocol:reporting.recommendation_rule.eligibility.service_min_pooled_entries }}
pooled entries is reported but not assessed by the recommendation rule. Priority is
fixed at queue entry (FP-2 §7.1); P3 does not occur because every order enters at
checkout ([methods](../docs/methods.md#capacity)).

## Staffing

Each policy keeps the thresholds tuned at the base allotment. The low and high levels
change the minutes per shift; the evening layout moves the same two shifts later in
the day with the base's minutes.

| Policy | Minutes used, low ({{ config:policy:capacity.levels.low.review_minutes_per_shift.early }} per shift) | Base ({{ config:policy:capacity.levels.base.review_minutes_per_shift.early }}) | High ({{ config:policy:capacity.levels.high.review_minutes_per_shift.early }}) | Evening layout | Decided after shipping, base | Evening layout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| approve-all | {{ evaluate.review_minutes_used_share.baseline.low.approve_all }} | {{ evaluate.review_minutes_used_share.baseline.base.approve_all }} | {{ evaluate.review_minutes_used_share.baseline.high.approve_all }} | {{ evaluate.review_minutes_used_share.baseline.redesigned_layout.approve_all }} | {{ evaluate.decided_after_shipping.baseline.base.approve_all | num:1 }} | {{ evaluate.decided_after_shipping.baseline.redesigned_layout.approve_all | num:1 }} |
| incumbent rules | {{ evaluate.review_minutes_used_share.baseline.low.incumbent_rules }} | {{ evaluate.review_minutes_used_share.baseline.base.incumbent_rules }} | {{ evaluate.review_minutes_used_share.baseline.high.incumbent_rules }} | {{ evaluate.review_minutes_used_share.baseline.redesigned_layout.incumbent_rules }} | {{ evaluate.decided_after_shipping.baseline.base.incumbent_rules | num:1 }} | {{ evaluate.decided_after_shipping.baseline.redesigned_layout.incumbent_rules | num:1 }} |
| depth-3 tree | {{ evaluate.review_minutes_used_share.baseline.low.tree_depth3 }} | {{ evaluate.review_minutes_used_share.baseline.base.tree_depth3 }} | {{ evaluate.review_minutes_used_share.baseline.high.tree_depth3 }} | {{ evaluate.review_minutes_used_share.baseline.redesigned_layout.tree_depth3 }} | {{ evaluate.decided_after_shipping.baseline.base.tree_depth3 | num:1 }} | {{ evaluate.decided_after_shipping.baseline.redesigned_layout.tree_depth3 | num:1 }} |
| logistic regression | {{ evaluate.review_minutes_used_share.baseline.low.logistic }} | {{ evaluate.review_minutes_used_share.baseline.base.logistic }} | {{ evaluate.review_minutes_used_share.baseline.high.logistic }} | {{ evaluate.review_minutes_used_share.baseline.redesigned_layout.logistic }} | {{ evaluate.decided_after_shipping.baseline.base.logistic | num:1 }} | {{ evaluate.decided_after_shipping.baseline.redesigned_layout.logistic | num:1 }} |
| gradient boosting | {{ evaluate.review_minutes_used_share.baseline.low.boosting }} | {{ evaluate.review_minutes_used_share.baseline.base.boosting }} | {{ evaluate.review_minutes_used_share.baseline.high.boosting }} | {{ evaluate.review_minutes_used_share.baseline.redesigned_layout.boosting }} | {{ evaluate.decided_after_shipping.baseline.base.boosting | num:1 }} | {{ evaluate.decided_after_shipping.baseline.redesigned_layout.boosting | num:1 }} |
| hybrid | {{ evaluate.review_minutes_used_share.baseline.low.hybrid }} | {{ evaluate.review_minutes_used_share.baseline.base.hybrid }} | {{ evaluate.review_minutes_used_share.baseline.high.hybrid }} | {{ evaluate.review_minutes_used_share.baseline.redesigned_layout.hybrid }} | {{ evaluate.decided_after_shipping.baseline.base.hybrid | num:1 }} | {{ evaluate.decided_after_shipping.baseline.redesigned_layout.hybrid | num:1 }} |
| expected loss | {{ evaluate.review_minutes_used_share.baseline.low.expected_loss }} | {{ evaluate.review_minutes_used_share.baseline.base.expected_loss }} | {{ evaluate.review_minutes_used_share.baseline.high.expected_loss }} | {{ evaluate.review_minutes_used_share.baseline.redesigned_layout.expected_loss }} | {{ evaluate.decided_after_shipping.baseline.base.expected_loss | num:1 }} | {{ evaluate.decided_after_shipping.baseline.redesigned_layout.expected_loss | num:1 }} |

*Minutes used* is the share of the window's allotted minutes that the work begun
took (it can pass the allotment by work done after the window); *decided after
shipping* is the mean per world. The rule's figures for each staffing level are in
the [operating review](operating-review.md#review-capacity-and-staffing) and below.

{{ table:replay.outcomes where capacity_level=base layout=current history=policy reviewer=evidence verification=verification policy=incumbent_rules sum orders, available_minutes, review_minutes_offered by family | family "World family" label, orders "Orders" count, available_minutes "Minutes allotted" count, review_minutes_offered "Minutes the incumbent's queue offered" count, rows_count "Worlds" count }}

The allotment per shift is the same in every family, so the acquisition surge brings
its extra orders to the same minutes. Orders and minutes are totals over the worlds;
dividing one by the other gives the realized minutes per order.

## The analyst's decisions by label

How the simulated analyst's reviews of the incumbent's queue ended, by the label each
order earned, before and after the order shipped (at the first decision). Counts are
orders summed over the final seeds' baseline worlds.

{{ table:replay.confusion where family=baseline policy=incumbent_rules decision_point=before_shipping sum orders by truth across final | truth "Decided before shipping" label, clear "Cleared" count, decline "Declined" count, escalate "Escalated" count, cancelled "Cancelled" count }}

{{ table:replay.confusion where family=baseline policy=incumbent_rules decision_point=after_shipping sum orders by truth across final | truth "Decided after shipping" label, clear "Cleared" count, decline "Declined" count, escalate "Escalated" count, unchanged "Unchanged" count }}

*Cleared:* cleared by the analyst, at once or once the checks passed; a held order
then ships. *Declined* and *escalated:* declined after a failed check or an earlier
outcome that settles the order; an order not yet shipped is voided, a shipped one's
loss stands, and the account is blocked (an escalation also blocks linked accounts and
adds senior review). *Cancelled:* held before shipment with no answer to the checks
within {{ config:policy:actions.hold_max_hours }} hours, then cancelled and refunded.
*Unchanged:* held after shipment with no answer, which changes nothing. The analyst
reads only the evidence known at the decision, so a fraud order cleared here showed
no adverse evidence or passed its checks, with nothing yet known that settled it
([methods](../docs/methods.md#procedure)).

<!-- Phase B: add the recommended (or leading) challenger's matrix if it reviews, with
the columns its final values need. -->

The same decisions by the order's latent pattern, a diagnostic the analyst never
sees ("legitimate" is no pattern; a bust-out merchant's customers are genuine):

{{ table:replay.confusion_latent where family=baseline policy=incumbent_rules decision_point=before_shipping sum orders by truth across final | truth "Latent pattern (before shipping)" label, clear "Cleared" count, decline "Declined" count, escalate "Escalated" count, cancelled "Cancelled" count }}

{{ table:replay.confusion_latent where family=baseline policy=incumbent_rules decision_point=after_shipping sum orders by truth across final | truth "Latent pattern (after shipping)" label, clear "Cleared" count, decline "Declined" count, escalate "Escalated" count, unchanged "Unchanged" count }}

## Operating sensitivities

The recommendation rule was applied again in each cell that differs from the primary
cell in one respect, with the primary cell's thresholds: the acquisition surge and
fraud-mix shift families, goods shipping in
{{ protocol:sensitivity.fulfilment_lag.families.lag_half | num:1 }} and
{{ protocol:sensitivity.fulfilment_lag.families.lag_double | num:1 }} times the
drawn time from the test window on, the low and high allotments, the evening layout,
and weaker verification (a takeover's contact check passing at
{{ config:policy:reviewer.verification_weak.takeover.contact.passed | num:2 }}
instead of {{ config:policy:reviewer.verification.takeover.contact.passed | num:2 }},
and takeover and third-party identity checks at
{{ config:policy:reviewer.verification_weak.takeover.id_check.passed | num:2 }}
instead of {{ config:policy:reviewer.verification.takeover.id_check.passed | num:2 }}).
The outcome in each cell is the flip table in the
[operating review](operating-review.md#when-the-answer-changes); the rule's figures
for every policy in each cell follow.

![Each challenger's gain over the incumbent in each operating cell](figures/operating_cells.svg)

<details>
<summary>World families: the rule's figures in each</summary>

{{ table:evaluate.recommendation where varies=family | cell "Cell" label, policy "Policy" label, rule_net_vs_incumbent_rules_per_1000_mean_cents "Mean gain" usd:signed, positive_seeds_count "Seeds positive" count, positive_seeds_needed_count "Needed" count, lost_legitimate_per_10k_mean_bps "Lost per 10k" num:1, held_legitimate_per_10k_mean_bps "Held per 10k" num:1, service_p1_entries_count "P1 entries" count, service_p1_in_time_mean_share "P1 in time" pct, service_p2_entries_count "P2 entries" count, service_p2_in_time_mean_share "P2 in time" pct, eligible "Eligible", fails "Misses" label }}

</details>

<details>
<summary>Allotment and shift layout: the rule's figures in each</summary>

{{ table:evaluate.recommendation where varies=allotment | cell "Cell" label, policy "Policy" label, rule_net_vs_incumbent_rules_per_1000_mean_cents "Mean gain" usd:signed, positive_seeds_count "Seeds positive" count, positive_seeds_needed_count "Needed" count, lost_legitimate_per_10k_mean_bps "Lost per 10k" num:1, held_legitimate_per_10k_mean_bps "Held per 10k" num:1, service_p1_entries_count "P1 entries" count, service_p1_in_time_mean_share "P1 in time" pct, service_p2_entries_count "P2 entries" count, service_p2_in_time_mean_share "P2 in time" pct, eligible "Eligible", fails "Misses" label }}

{{ table:evaluate.recommendation where varies=layout | cell "Cell" label, policy "Policy" label, rule_net_vs_incumbent_rules_per_1000_mean_cents "Mean gain" usd:signed, positive_seeds_count "Seeds positive" count, positive_seeds_needed_count "Needed" count, lost_legitimate_per_10k_mean_bps "Lost per 10k" num:1, held_legitimate_per_10k_mean_bps "Held per 10k" num:1, service_p1_entries_count "P1 entries" count, service_p1_in_time_mean_share "P1 in time" pct, service_p2_entries_count "P2 entries" count, service_p2_in_time_mean_share "P2 in time" pct, eligible "Eligible", fails "Misses" label }}

</details>

<details>
<summary>Weak verification: the rule's figures</summary>

{{ table:evaluate.recommendation where varies=verification | cell "Cell" label, policy "Policy" label, rule_net_vs_incumbent_rules_per_1000_mean_cents "Mean gain" usd:signed, positive_seeds_count "Seeds positive" count, positive_seeds_needed_count "Needed" count, lost_legitimate_per_10k_mean_bps "Lost per 10k" num:1, held_legitimate_per_10k_mean_bps "Held per 10k" num:1, service_p1_entries_count "P1 entries" count, service_p1_in_time_mean_share "P1 in time" pct, service_p2_entries_count "P2 entries" count, service_p2_in_time_mean_share "P2 in time" pct, eligible "Eligible", fails "Misses" label }}

</details>

*Mean gain:* rule net contribution against the incumbent per 1,000 orders decided,
as in the operating review; lost and held customers per 10,000 legitimate orders,
mean over seeds; entries pooled over seeds, and a priority with too few entries is not
assessed, whatever its share in time. P0 entries are too few to assess in any cell.

<!-- Phase B: check the P0 sentence against the final evaluate.recommendation
(service_p0_assessed); drop it if any cell assesses P0. -->

## A reviewer who always knows the truth (a diagnostic)

The same queue, timing and allotment with an analyst who declines every order
generated as fraud or abuse and clears the rest at review completion, without checks.
It shows how much of each policy's result is limited by what review can learn, as
opposed to which orders reach review and when; it is not part of the recommendation
rule. Its declines include the genuine customers of a bust-out merchant
([methods](../docs/methods.md#upper-bound)).

| Policy | Ledger net, evidence-based reviewer | Ledger net, all-knowing reviewer | Difference | Fraud loss difference | Legitimate declined difference per 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| approve-all | {{ evaluate.net_contribution.baseline.base.approve_all }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.approve_all }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.approve_all - evaluate.net_contribution.baseline.base.approve_all | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base_perfect_reviewer.approve_all - evaluate.loss_of_gmv.baseline.base.approve_all | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base_perfect_reviewer.approve_all - evaluate.legitimate_declined_per_10k.baseline.base.approve_all | num:1:signed }} |
| incumbent rules | {{ evaluate.net_contribution.baseline.base.incumbent_rules }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.incumbent_rules }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.incumbent_rules - evaluate.net_contribution.baseline.base.incumbent_rules | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base_perfect_reviewer.incumbent_rules - evaluate.loss_of_gmv.baseline.base.incumbent_rules | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base_perfect_reviewer.incumbent_rules - evaluate.legitimate_declined_per_10k.baseline.base.incumbent_rules | num:1:signed }} |
| depth-3 tree | {{ evaluate.net_contribution.baseline.base.tree_depth3 }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.tree_depth3 }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.tree_depth3 - evaluate.net_contribution.baseline.base.tree_depth3 | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base_perfect_reviewer.tree_depth3 - evaluate.loss_of_gmv.baseline.base.tree_depth3 | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base_perfect_reviewer.tree_depth3 - evaluate.legitimate_declined_per_10k.baseline.base.tree_depth3 | num:1:signed }} |
| logistic regression | {{ evaluate.net_contribution.baseline.base.logistic }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.logistic }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.logistic - evaluate.net_contribution.baseline.base.logistic | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base_perfect_reviewer.logistic - evaluate.loss_of_gmv.baseline.base.logistic | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base_perfect_reviewer.logistic - evaluate.legitimate_declined_per_10k.baseline.base.logistic | num:1:signed }} |
| gradient boosting | {{ evaluate.net_contribution.baseline.base.boosting }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.boosting }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.boosting - evaluate.net_contribution.baseline.base.boosting | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base_perfect_reviewer.boosting - evaluate.loss_of_gmv.baseline.base.boosting | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base_perfect_reviewer.boosting - evaluate.legitimate_declined_per_10k.baseline.base.boosting | num:1:signed }} |
| hybrid | {{ evaluate.net_contribution.baseline.base.hybrid }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.hybrid }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.hybrid - evaluate.net_contribution.baseline.base.hybrid | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base_perfect_reviewer.hybrid - evaluate.loss_of_gmv.baseline.base.hybrid | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base_perfect_reviewer.hybrid - evaluate.legitimate_declined_per_10k.baseline.base.hybrid | num:1:signed }} |
| expected loss | {{ evaluate.net_contribution.baseline.base.expected_loss }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.expected_loss }} | {{ evaluate.net_contribution.baseline.base_perfect_reviewer.expected_loss - evaluate.net_contribution.baseline.base.expected_loss | usd:signed }} | {{ evaluate.loss_of_gmv.baseline.base_perfect_reviewer.expected_loss - evaluate.loss_of_gmv.baseline.base.expected_loss | bps:signed }} | {{ evaluate.legitimate_declined_per_10k.baseline.base_perfect_reviewer.expected_loss - evaluate.legitimate_declined_per_10k.baseline.base.expected_loss | num:1:signed }} |

*Ledger net* is the mean over seeds of each world's net cash for the test window's
orders; the differences are the all-knowing reviewer minus the evidence-based one.
