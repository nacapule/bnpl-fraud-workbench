# Ring: declined at checkout on a shared device, without an account block

The order came from a device used by
{{ fact:ring:alerts.ring.evidence.row.accounts_on_device_30d }} accounts within thirty days.
R02 alone reached the checkout decline threshold. The incumbent declined the order
automatically, recording no fraud finding and blocking no account.

The world is synthetic, and the canonical world the cases come from (seed
{{ fact:ring:source.world.seed }}, {{ fact:ring:source.world.family }} family) is a development
seed, so the case illustrates how the policy behaves rather than measuring it. The selection rule was fixed before
evaluation ([methods](../docs/methods.md#case-selection)). This is the first in hash order
among {{ fact:ring:alerts.ring.publication.candidates.eligible }} test-window alerts on
synthetic-identity ring orders.
Values come from [the facts file](facts/ring.json).

| | |
|---|---|
| Alert | `{{ fact:ring:alerts.ring.publication.alert_id }}` (order id and policy version) |
| Decision | at checkout, {{ fact:ring:alerts.ring.decision.checkout_at }}: the routing reads the checkout row; no analyst is involved |

## Evidence at decision time

The checkout router used this saved row. It contains only information available at checkout.

{{ facts:ring:alerts.ring.evidence.row "Fact at checkout" "Value" | amount_cents "Order amount" usd:2, order_exposure_cents "Cash at risk if approved" usd:2, account_age_days "Account age (days)" num:0, accounts_on_device_30d "Accounts on this device in the past thirty days" count, device_link_age_hours "Hours since the device was first used on this account" num:0, accounts_on_address_30d "Accounts shipping to this address in the past thirty days" count, ship_to_home "Ships to the account's home address" yesno, email_root_other_accounts "Other accounts with the same normalized email" count, bin_ip_country_mismatch "Card country differs from the IP country" yesno, avs_mismatch "AVS check failed" yesno, cvv_mismatch "CVV check failed" yesno, approved_orders_user_ever "Earlier orders approved on the account" count, installments_due_user "Installments due on the account" count, installments_paid_user "Installments paid" count, open_balance_user_cents "Open balance on earlier plans" usd:2 }}

Only {{ fact:ring:alerts.ring.evidence.rules_held.0 }} held: at least three accounts on the
device within thirty days (§6.2). Its weight alone gave a score of
{{ fact:ring:alerts.ring.routing.rule_score | num:0 }}, equal to the decline threshold
({{ fact:ring:source.replay.decline_threshold | num:0 }}). The household exception to R02
(§6.4(a)) does not apply: the device has not been on the account for the required ninety days.

## What FP-2 supports on this evidence

The [fraud policy](../policy/fraud-policy.md) permits automatic decline by the checkout score
band (§4.1), without a fraud finding or account block. At review, the same evidence would
supply only {{ fact:ring:alerts.ring.policy_view.families.0 }}. Row
{{ fact:ring:alerts.ring.policy_view.row }} would apply:

- **Standard:** `{{ fact:ring:alerts.ring.policy_view.standard.0 }}` with
  `{{ fact:ring:alerts.ring.policy_view.required_checks.0 }}` (§5.2).
- **Also permitted:** `needs_check`, the memo recommendation for a hold with that check
  (§4.3, §6.6(b)).
- **Prohibited:** `clear`, `decline` and `escalate` (§6.6(b)).

An analyst could not decline on this evidence alone. Escalation would require a failed check
or an earlier outcome that settles the order (§5.3(b), §6.6(a)); with Linkage present, it
would also block the linked accounts (§4.2). The checkout band therefore went further than an
analyst could, declining on one linkage condition, and also did less: it recorded no finding and
blocked none of the device's accounts.

The shared device could reflect a ring of synthetic identities or, benignly, a household or
shared computer used by genuine customers (§6.4). The repayment record does not establish
who ordered (§6.5(f)). The required identity check would test whether the customer holds
both the card and the account (§5.1(b)).

## Recorded action in the replay

The router recorded **`{{ fact:ring:alerts.ring.decision.recorded_action }}`** at checkout.
No analyst time was used.

## What happened later

The checkout decline meant no order was created and no cash moved. The incumbent net was
{{ fact:ring:alerts.ring.later.incumbent.net_cents | usd:2 }}. The account placed
{{ fact:ring:alerts.ring.later.account.later_orders }} later order in the window, also declined
at checkout. The approve-all outcomes below were unavailable at that decision.

**Had the order been approved (the approve-all world).** It ships and no installment is ever
paid:

{{ table:fact:ring:alerts.ring.later.approve_all.events | known_at "Known", event "Event", detail "Detail", amount_cents "Amount" usd:2 }}

{{ table:fact:ring:alerts.ring.later.approve_all.cash | known_at "Known", kind "Cash event", amount_cents "Amount" usd:2 }}

The approve-all net would have been
{{ fact:ring:alerts.ring.later.approve_all.net_cents | usd:2 }}, so the decline prevented a loss of
{{ fact:ring:alerts.ring.later.prevented_cents | usd:2 }}. The label, from the approve-all world,
is `{{ fact:ring:alerts.ring.later.label.basis }}`, known at
{{ fact:ring:alerts.ring.later.label.label_known_at }} (§8.3).

## Tested change: Linkage-only orders to review instead of automatic decline

The change, motivation and expected mechanism were declared before testing. The change was
replayed once against the unchanged incumbent.

> **Change.** {{ fact:ring:tested_change.change }}
>
> **Motivation.** {{ fact:ring:tested_change.motivation }}
>
> **Expected mechanism.** {{ fact:ring:tested_change.mechanism }}

**This case's order.** With the change, the account was already blocked at checkout
(`{{ fact:ring:tested_change.result.case_order.variant.route }}`): an analyst had escalated an
order from a linked account, also blocking this account. Its later order was refused for the
same reason.

**The whole test window.**

| Measure | Incumbent | With the change | Difference |
|---|---:|---:|---:|
| Fraud orders declined at checkout | {{ fact:ring:tested_change.result.world.incumbent.adjudicated.fraud_declined_checkout }} | {{ fact:ring:tested_change.result.world.variant.adjudicated.fraud_declined_checkout }} | {{ fact:ring:tested_change.result.world.difference.adjudicated.fraud_declined_checkout | count:signed }} |
| Fraud orders stopped after review, before shipping | {{ fact:ring:tested_change.result.world.incumbent.adjudicated.fraud_stopped_before_shipping }} | {{ fact:ring:tested_change.result.world.variant.adjudicated.fraud_stopped_before_shipping }} | {{ fact:ring:tested_change.result.world.difference.adjudicated.fraud_stopped_before_shipping | count:signed }} |
| Fraud loss prevented | {{ fact:ring:tested_change.result.world.incumbent.cash.prevented_loss_cents | usd:2 }} | {{ fact:ring:tested_change.result.world.variant.cash.prevented_loss_cents | usd:2 }} | {{ fact:ring:tested_change.result.world.difference.cash.prevented_loss_cents | usd:2:signed }} |
| Escalations | {{ fact:ring:tested_change.result.world.incumbent.review.escalations }} | {{ fact:ring:tested_change.result.world.variant.review.escalations }} | {{ fact:ring:tested_change.result.world.difference.review.escalations | count:signed }} |
| Accounts blocked | {{ fact:ring:tested_change.result.world.incumbent.review.accounts_blocked }} | {{ fact:ring:tested_change.result.world.variant.review.accounts_blocked }} | {{ fact:ring:tested_change.result.world.difference.review.accounts_blocked | count:signed }} |
| Reviews decided after the order shipped | {{ fact:ring:tested_change.result.world.incumbent.review.decided_after_shipping }} | {{ fact:ring:tested_change.result.world.variant.review.decided_after_shipping }} | {{ fact:ring:tested_change.result.world.difference.review.decided_after_shipping | count:signed }} |
| Review minutes used | {{ fact:ring:tested_change.result.world.incumbent.review.review_minutes_used }} | {{ fact:ring:tested_change.result.world.variant.review.review_minutes_used }} | {{ fact:ring:tested_change.result.world.difference.review.review_minutes_used | count:signed }} |
| Legitimate orders declined | {{ fact:ring:tested_change.result.world.incumbent.adjudicated.legitimate_declined }} | {{ fact:ring:tested_change.result.world.variant.adjudicated.legitimate_declined }} | {{ fact:ring:tested_change.result.world.difference.adjudicated.legitimate_declined | count:signed }} |
| Rule net contribution | {{ fact:ring:tested_change.result.world.incumbent.cash.rule_net_cents | usd:2 }} | {{ fact:ring:tested_change.result.world.variant.cash.rule_net_cents | usd:2 }} | {{ fact:ring:tested_change.result.world.difference.cash.rule_net_cents | usd:2:signed }} |

Across the {{ fact:ring:tested_change.result.world.incumbent.orders }} orders decided by the
replay, the difference in rule net contribution per thousand orders was
{{ fact:ring:tested_change.result.world.difference.rule_net_vs_incumbent_per_1000_orders_cents | usd:2:signed }}
([recommendation rule](../docs/methods.md#recommendation-rule)). The result was negative.
Escalations and account blocks increased, but orders previously declined at checkout now
waited for review. More reviews finished after shipment. The changed policy prevented less
fraud loss and had lower rule net contribution.

## Latent truth (diagnostic)

These simulation records are unavailable to rules, analysts and the memo drafter. The
synthetic-identity pattern defines the selection slot. The account's holder is
`{{ fact:ring:alerts.ring.latent.account.actor }}`, one of
{{ fact:ring:alerts.ring.latent.episode.accounts }} accounts in a synthetic-identity ring
(`{{ fact:ring:alerts.ring.latent.episode.pattern_id }}`) with
{{ fact:ring:alerts.ring.latent.episode.orders }} orders, from
{{ fact:ring:alerts.ring.latent.episode.started_at }} to
{{ fact:ring:alerts.ring.latent.episode.ended_at }}. The platform's net from the ring's
test-window orders was
{{ fact:ring:tested_change.result.latent_episode.incumbent.net_cents | usd:2 }} under the
incumbent and {{ fact:ring:tested_change.result.latent_episode.variant.net_cents | usd:2 }} with
the change. The changed policy blocked accounts in this ring but left its previously approved
orders unchanged, so the platform's net from those orders did not improve.

## The memo drafter's memo

Pending: the LLM drafter's advisory memo for this alert's packet, built from the evidence row
above, will appear here with the memo benchmark results.
