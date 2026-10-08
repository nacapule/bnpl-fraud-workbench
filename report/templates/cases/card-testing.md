# Card testing: a new account's order declined at checkout on R07 and R05

A new account made repeated order attempts on a device with recent processor declines. R07 and
R05 took the score over the checkout decline threshold, and the incumbent declined the order
without review, as it had declined the account's earlier order that morning. Nothing shipped.

The world is synthetic, and the canonical world the cases come from (seed
{{ fact:card_testing:source.world.seed }}, {{ fact:card_testing:source.world.family }} family)
is a development seed, so this short case illustrates how the policy behaves rather than
measuring it. It is the first in hash order of
{{ fact:card_testing:alerts.card_testing.publication.candidates.eligible }} test-window alerts
eligible for its slot ([selection](../docs/methods.md#case-selection); the slot is under the
latent truth). The full record is in
[the facts file](facts/card_testing.json).

## Decision and evidence

Alert `{{ fact:card_testing:alerts.card_testing.publication.alert_id }}`, decided at checkout
on {{ fact:card_testing:alerts.card_testing.decision.checkout_at }} by the score band, from the
saved checkout row:

{{ facts:card_testing:alerts.card_testing.evidence.row "Fact at checkout" "Value" | amount_cents "Order amount" usd:2, attempts_user_1h "Order attempts by the account in the past hour (this one included)" count, processor_declines_device_24h "Processor declines on this device in the past day" count, processor_declines_card_24h "Processor declines on this card in the past day" count, device_link_age_hours "Hours since the device was first used on this account" num:2, bin_ip_country_mismatch "Card country differs from the IP country" yesno, installments_due_user "Installments due on the account" count, approved_orders_user_24h "Orders approved on the account in the past day (see below)" count, open_balance_user_cents "Open balance on earlier plans (see below)" usd:2 }}

The last two rows are approve-all values. The replay refreshes outcome columns at the start of
each day ([methods](../docs/methods.md#the-replay)), so they still count the account's earlier
order that day as approved, but the incumbent had declined it at checkout:

{{ table:fact:card_testing:alerts.card_testing.decision.same_day_orders | checkout_at "Checked out", route "Incumbent's decision" }}

No order of this session was approved.

R07 (three or more processor declines on the device in a day) and R05 (more than three order
attempts by the account in a day) gave a score of
{{ fact:card_testing:alerts.card_testing.routing.rule_score | num:0 }}, over the decline
threshold of {{ fact:card_testing:source.replay.decline_threshold | num:0 }}. The recorded
action was **`{{ fact:card_testing:alerts.card_testing.decision.recorded_action }}`**.

## What FP-2 supports

The checkout band may decline automatically (§4.1); that records no fraud finding and blocks no
account. Reviewed, the same row would show two adverse families
({{ fact:card_testing:alerts.card_testing.policy_view.families.0 }} from R07,
{{ fact:card_testing:alerts.card_testing.policy_view.families.1 }} from R05) under row
{{ fact:card_testing:alerts.card_testing.policy_view.row }}: standard
`{{ fact:card_testing:alerts.card_testing.policy_view.standard.0 }}` with
`{{ fact:card_testing:alerts.card_testing.policy_view.required_checks.0 }}`; `decline` and
`needs_check` also permitted, since two families are present and no check has passed; `clear`
and `escalate` prohibited. So the automatic decline matches an action an analyst could have
taken, except that an analyst's decline would also block the account (§4.2). No installment
was due, so nothing could cite non-payment (§8.1). The competing readings are someone testing
stolen card details, or a new customer retrying after declines on a device new to the account
(§6.5(b) covers the new device), which the identity check would settle.

## What happened later

The order was never created: no shipment, no plan and no cash under the incumbent
({{ fact:card_testing:alerts.card_testing.later.incumbent.net_cents | usd:2 }}), and the account
placed no later order. Had it been approved (the approve-all world), it would have shipped, the
cardholder would have disputed it as unauthorized and the plan would have been written off: a
net of {{ fact:card_testing:alerts.card_testing.later.approve_all.net_cents | usd:2 }}, so the
decline prevented {{ fact:card_testing:alerts.card_testing.later.prevented_cents | usd:2 }}. The
order's label is `{{ fact:card_testing:alerts.card_testing.later.label.basis }}`, known when the
dispute was resolved at {{ fact:card_testing:alerts.card_testing.later.label.label_known_at }}
(§9(a)).

## Tested change: R07 at one device decline for new accounts

Declared before it ran and replayed once against the unchanged incumbent. The motivation is
printed as declared: its "order already approved" read the approve-all value in the row, which
the note added after the run corrects.

> **Change.** {{ fact:card_testing:tested_change.change }}
>
> **Motivation.** {{ fact:card_testing:tested_change.motivation }}
>
> **Noted after the run.** {{ fact:card_testing:tested_change.noted_after_run }}

Both of the session's orders are declined at checkout either way. Across the test window,
legitimate orders declined at checkout changed by
{{ fact:card_testing:tested_change.result.world.difference.adjudicated.legitimate_declined_checkout | count:signed }},
fraud orders declined at checkout by
{{ fact:card_testing:tested_change.result.world.difference.adjudicated.fraud_declined_checkout | count:signed }}
and fraud loss prevented by
{{ fact:card_testing:tested_change.result.world.difference.cash.prevented_loss_cents | usd:2:signed }};
the rule net contribution ([definition](../docs/methods.md#recommendation-rule)) changed by
{{ fact:card_testing:tested_change.result.world.difference.rule_net_vs_incumbent_per_1000_orders_cents | usd:2:signed }}
per thousand decided orders. A negative result: the earlier trigger caught no fraud the
incumbent missed and turned away legitimate new customers.

## Latent truth (diagnostic)

Unavailable to rules and analysts. The slot is stolen-card orders with a processor decline on
the device before checkout. This order belongs to a stolen-card episode
(`{{ fact:card_testing:alerts.card_testing.latent.episode.pattern_id }}`) by a
{{ fact:card_testing:alerts.card_testing.latent.account.actor }} on one account,
{{ fact:card_testing:alerts.card_testing.latent.episode.orders }} order attempts in all, and
the incumbent declined every one of them the processor approved.

## The memo drafter's memo

Pending: the LLM drafter's memo for this alert's packet will appear here with the memo
benchmark results.
