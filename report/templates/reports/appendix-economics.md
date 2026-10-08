# Economics appendix

How the cash ledger values each policy, how its figures reconcile, the economic
assumptions behind them, and how the answer moves with the value of a lost customer.
The worlds are synthetic, and every cost below is a stated modelling assumption, not
a measurement (FP-2 §11). The ledger's rules are set out in the
[methods](../docs/methods.md#ledger).

## The product and the ledger

All money is one ledger of cash events in integer cents, from the platform's point of
view (`core/ledger.py`).

| Term | Value |
| --- | --- |
| Down payment | {{ config:world:product.down_payment_bps | bps_pct }} of the price, collected at approval |
| Installments | {{ config:world:product.n_installments }} more, every {{ config:world:product.installment_interval_days }} days, the first {{ config:world:product.installment_interval_days }} days after approval |
| Merchant fee | {{ config:world:product.merchant_discount_bps | bps_pct }} of the price, netted from the merchant's settlement when it ships; the platform's income on a repaid order |
| Promotions | funded by the platform, paid to the merchant as a separate event |
| Disputes | the disputed payments debited and a {{ config:world:product.dispute_fee_cents | usd }} fee charged when the platform learns of a dispute; the disputed amount is credited back if it is won, and the fee stays |
| Who bears a lost dispute | the platform for unauthorized use; the merchant for item-not-received and not-as-described claims, unless it has closed |
| Write-off | {{ config:world:product.writeoff_after_days }} days after the last installment is due, if a balance remains; a status, not cash |
| Recovery | {{ config:world:product.recovery_rate_bps | bps_pct }} of the written-off balance, {{ config:world:product.recovery_lag_days }} days after the write-off |

A policy's cash comes from the same ledger: an order it leaves alone keeps the
world's cash; an order voided before shipment keeps the cash that had moved, with
refunds of what was collected; a released hold's cash moves with its shipment
([methods](../docs/methods.md#the-replay)).

## The costs the recommendation rule adds

| Cost | Value | Where it applies |
| --- | --- | --- |
| Lifetime-value proxy | ${{ config:policy:costs.false_decline_ltv_usd | num:0 }} per lost legitimate customer | the rule's net contribution, the tuning objective, and the expected-loss policy's decline rule |
| Analyst time | ${{ config:policy:costs.analyst_loaded_hourly_usd | num:0 }} per allotted hour, unused minutes included | the rule's net contribution; nothing for a policy whose chosen point sends no order to review |
| Hurdle | ${{ protocol:reporting.recommendation_rule.hurdle.mean_improvement_usd_per_1000_orders }} per 1,000 orders | the mean gain over the incumbent a challenger must reach, an allowance for the cost of changing a policy |

A lost legitimate customer is one declined at checkout, refused because the account
was blocked, declined or escalated after review, or cancelled after an unanswered
verification request. Churn is not simulated, so the proxy stands for the future
value such a customer takes away.

## Reconciliation in the primary cell

Totals over the final seeds' baseline worlds at the base allotment, test window.

{{ table:replay.outcomes where family=baseline capacity_level=base layout=current history=policy reviewer=evidence verification=verification sum orders, net_cents, approve_all_net_cents, net_vs_approve_all_cents, fraud_net_cents, approve_all_fraud_net_cents, prevented_loss_cents, credit_loss_net_cents by policy | policy "Policy" label, orders "Orders" count, net_cents "Ledger net" usd, approve_all_net_cents "Approve-all net" usd, net_vs_approve_all_cents "Difference" usd:signed, fraud_net_cents "On fraud orders" usd, approve_all_fraud_net_cents "Approve-all on fraud orders" usd, prevented_loss_cents "Prevented loss" usd:signed, credit_loss_net_cents "On credit-loss orders" usd }}

Each row reconciles: the difference is the ledger net minus approve-all's, and the
prevented loss is the net on fraud orders minus approve-all's on the same orders (the
fraud loss is the negative of the net on fraud orders). The rest of the difference
falls on orders without a fraud label: margin kept or given up on good customers and
credit-loss orders.

{{ table:replay.prevented_by_pattern where family=baseline sum policy_net_cents, approve_all_net_cents by policy | policy "Policy" label, policy_net_cents "Ledger net by label" usd, approve_all_net_cents "Approve-all net by label" usd }}

The same cash summed one adjudicated label at a time (the table in the
[detection appendix](appendix-detection.md#what-each-policy-prevented-by-label))
gives the same totals as the first table's ledger and approve-all columns.

{{ table:replay.outcomes where family=baseline capacity_level=base layout=current history=policy reviewer=evidence verification=verification sum legitimate_declined, legitimate_cancelled, friction_cost_cents, available_minutes, review_band by policy | policy "Policy" label, legitimate_declined "Legitimate declined" count, legitimate_cancelled "Legitimate cancelled" count, friction_cost_cents "Friction cost" usd, available_minutes "Minutes allotted" count, review_band "Worlds with a review route" count, rows_count "Worlds" count }}

In each world, the rule's net contribution is the ledger net minus the friction cost (${{ config:policy:costs.false_decline_ltv_usd | num:0 }} for each legitimate order declined or cancelled) minus the allotment's cost (the allotted minutes at ${{ config:policy:costs.analyst_loaded_hourly_usd | num:0 }} an hour, charged only in worlds where the policy has a review route). The rule compares it with the incumbent's in the same world, per 1,000 orders, and averages over worlds, so these totals reproduce its figures only up to that averaging.

## The value of a lost customer

The rule recomputed with a lifetime-value proxy of ${{ protocol:sensitivity.ltv_proxy_usd.0 }} and ${{ protocol:sensitivity.ltv_proxy_usd.1 }} in place of ${{ config:policy:costs.false_decline_ltv_usd | num:0 }}, from the same replays: only the valuation changes, not the decisions (the expected-loss policy keeps ${{ config:policy:costs.false_decline_ltv_usd | num:0 }} inside its decline rule).

| Policy | Mean gain, ${{ protocol:sensitivity.ltv_proxy_usd.0 }} proxy | ${{ config:policy:costs.false_decline_ltv_usd | num:0 }} proxy | ${{ protocol:sensitivity.ltv_proxy_usd.1 }} proxy | Lost legitimate customers per 10,000 |
| --- | ---: | ---: | ---: | ---: |
| approve-all | {{ evaluate.rule_net_per_1000_orders_ltv_5_usd.vs_incumbent_rules.baseline.base.approve_all | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.approve_all | usd:signed }} | {{ evaluate.rule_net_per_1000_orders_ltv_45_usd.vs_incumbent_rules.baseline.base.approve_all | usd:signed }} | {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.approve_all | num:1 }} |
| incumbent rules | reference | reference | reference | {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.incumbent_rules | num:1 }} |
| depth-3 tree | {{ evaluate.rule_net_per_1000_orders_ltv_5_usd.vs_incumbent_rules.baseline.base.tree_depth3 | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.tree_depth3 | usd:signed }} | {{ evaluate.rule_net_per_1000_orders_ltv_45_usd.vs_incumbent_rules.baseline.base.tree_depth3 | usd:signed }} | {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.tree_depth3 | num:1 }} |
| logistic regression | {{ evaluate.rule_net_per_1000_orders_ltv_5_usd.vs_incumbent_rules.baseline.base.logistic | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.logistic | usd:signed }} | {{ evaluate.rule_net_per_1000_orders_ltv_45_usd.vs_incumbent_rules.baseline.base.logistic | usd:signed }} | {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.logistic | num:1 }} |
| gradient boosting | {{ evaluate.rule_net_per_1000_orders_ltv_5_usd.vs_incumbent_rules.baseline.base.boosting | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.boosting | usd:signed }} | {{ evaluate.rule_net_per_1000_orders_ltv_45_usd.vs_incumbent_rules.baseline.base.boosting | usd:signed }} | {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.boosting | num:1 }} |
| hybrid | {{ evaluate.rule_net_per_1000_orders_ltv_5_usd.vs_incumbent_rules.baseline.base.hybrid | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.hybrid | usd:signed }} | {{ evaluate.rule_net_per_1000_orders_ltv_45_usd.vs_incumbent_rules.baseline.base.hybrid | usd:signed }} | {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.hybrid | num:1 }} |
| expected loss | {{ evaluate.rule_net_per_1000_orders_ltv_5_usd.vs_incumbent_rules.baseline.base.expected_loss | usd:signed }} | {{ evaluate.rule_net_per_1000_orders.vs_incumbent_rules.baseline.base.expected_loss | usd:signed }} | {{ evaluate.rule_net_per_1000_orders_ltv_45_usd.vs_incumbent_rules.baseline.base.expected_loss | usd:signed }} | {{ evaluate.rule_lost_legitimate_per_10k.baseline.base.expected_loss | num:1 }} |

*Mean gain:* rule net contribution against the incumbent per 1,000 orders decided,
mean over seeds. The lost customers, the same at every proxy, are what the proxy
prices. The rule's outcome at each proxy is in the flip table of the
[operating review](operating-review.md#when-the-answer-changes).

<details>
<summary>The rule's figures at each proxy</summary>

{{ table:evaluate.recommendation where varies=ltv | cell "Cell" label, policy "Policy" label, rule_net_vs_incumbent_rules_per_1000_mean_cents "Mean gain" usd:signed, positive_seeds_count "Seeds positive" count, positive_seeds_needed_count "Needed" count, eligible "Eligible", fails "Misses" label, recommended "Recommended" }}

</details>
