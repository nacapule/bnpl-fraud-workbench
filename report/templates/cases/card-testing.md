# Card testing: a new account's order declined at checkout on R07 and R05

A new account made repeated order attempts on a device with recent processor declines. R07
and R05 took the score above the checkout decline threshold. The incumbent declined this
order automatically, without analyst review, as it had declined the account's earlier order that
morning. Nothing shipped.

The world is synthetic, and the canonical world the cases come from (seed
{{ fact:card_testing:source.world.seed }}, {{ fact:card_testing:source.world.family }} family)
is a development seed, so the case illustrates how the policy behaves rather than measuring it. The selection rule was fixed before evaluation
([methods](../docs/methods.md#case-selection)). This is the first in hash order among
{{ fact:card_testing:alerts.card_testing.publication.candidates.eligible }} test-window
alerts on stolen-card orders with a processor decline on the device beforehand. Values come from [the facts file](facts/card_testing.json).

| | |
|---|---|
| Alert | `{{ fact:card_testing:alerts.card_testing.publication.alert_id }}` (order id and policy version) |
| Decision | at checkout, {{ fact:card_testing:alerts.card_testing.decision.checkout_at }}: the routing reads the checkout row; no analyst is involved |

## Evidence at decision time

The checkout router used this saved row.

{{ facts:card_testing:alerts.card_testing.evidence.row "Fact at checkout" "Value" | amount_cents "Order amount" usd:2, order_exposure_cents "Cash at risk if approved" usd:2, device_link_age_hours "Hours since the device was first used on this account" num:2, card_link_age_hours "Hours since this card was added to the account" num:2, attempts_user_1h "Order attempts by the account in the past hour (this one included)" count, attempts_device_24h "Order attempts on this device in the past day" count, processor_declines_device_24h "Processor declines on this device in the past day" count, processor_declines_card_24h "Processor declines on this card in the past day" count, bin_ip_country_mismatch "Card country differs from the IP country" yesno, avs_mismatch "AVS check failed" yesno, cvv_mismatch "CVV check failed" yesno, installments_due_user "Installments due on the account" count, approved_orders_user_24h "Orders approved on the account in the past day (see below)" count, open_balance_user_cents "Open balance on earlier plans (see below)" usd:2 }}

The approved-orders and open-balance rows show approve-all values for the account's earlier
order that day. The incumbent had declined that order at checkout. Outcome-derived columns
are refreshed at the start of each replay day; they do not yet reflect decisions made that
day ([methods](../docs/methods.md#the-replay)). The earlier order's recorded decision was:

{{ table:fact:card_testing:alerts.card_testing.decision.same_day_orders | checkout_at "Checked out", route "Incumbent's decision" }}

No order in this session was approved by the incumbent. The approval count and balance in
the saved row do not describe its decisions.

R07 held because the device had at least three processor declines in the past day. R05 held
because the account had more than three order attempts in the past day (§6.2). Their score,
{{ fact:card_testing:alerts.card_testing.routing.rule_score | num:0 }}, reached the decline
threshold ({{ fact:card_testing:source.replay.decline_threshold | num:0 }}).

## What FP-2 supports on this evidence

The [fraud policy](../policy/fraud-policy.md) permits automatic decline by the checkout score
band (§4.1), without a fraud finding or account block. At review, the same evidence would
supply {{ fact:card_testing:alerts.card_testing.policy_view.families.0 }} from R07 and
{{ fact:card_testing:alerts.card_testing.policy_view.families.1 }} from R05. Row
{{ fact:card_testing:alerts.card_testing.policy_view.row }} would apply:

- **Standard:** `{{ fact:card_testing:alerts.card_testing.policy_view.standard.0 }}` with
  `{{ fact:card_testing:alerts.card_testing.policy_view.required_checks.0 }}` (§5.2).
- **Also permitted:** `decline`, because two adverse families are present and no check has
  passed; `needs_check`, the memo recommendation for a hold with the check (§4.3, §6.6(b)).
- **Prohibited:** `clear` and `escalate` (§6.6(b)).

An analyst could also have declined the order, but that action would have blocked the account
(§4.2). No installment had fallen due, so non-payment or never-pay could not justify the
decision (§8.1).

The attempts could reflect someone testing stolen card details until a payment goes through.
The benign explanation is a new customer retrying after payment declines (§6.5(b)). The required
identity check would test whether the customer holds both the card and the account (§5.1(b)).

## Recorded action in the replay

The router recorded **`{{ fact:card_testing:alerts.card_testing.decision.recorded_action }}`**
at checkout. No analyst time was used.

## What happened later

The checkout decline meant no order or plan was created, no shipment occurred and no cash
moved. The incumbent net was
{{ fact:card_testing:alerts.card_testing.later.incumbent.net_cents | usd:2 }}. The account
placed no later order in the window. The approve-all outcomes below were unavailable at
checkout.

**Had the order been approved (the approve-all world).** It ships, the cardholder disputes the
charge as unauthorized, the plan is written off and the dispute is lost:

{{ table:fact:card_testing:alerts.card_testing.later.approve_all.events | known_at "Known", event "Event", detail "Detail", amount_cents "Amount" usd:2 }}

{{ table:fact:card_testing:alerts.card_testing.later.approve_all.cash | known_at "Known", kind "Cash event", amount_cents "Amount" usd:2 }}

The approve-all net would have been
{{ fact:card_testing:alerts.card_testing.later.approve_all.net_cents | usd:2 }}, so the decline
prevented a loss of {{ fact:card_testing:alerts.card_testing.later.prevented_cents | usd:2 }}. The label,
from the approve-all world, is `{{ fact:card_testing:alerts.card_testing.later.label.basis }}`,
known at {{ fact:card_testing:alerts.card_testing.later.label.label_known_at }} when the dispute
was resolved (§9(a)).

## Tested change: R07 at one device decline for new accounts

The change, motivation and expected mechanism were declared before testing. The change was
replayed once against the unchanged incumbent. The motivation below is verbatim; the note
after it corrects its reading of the outcome-derived columns.

> **Change.** {{ fact:card_testing:tested_change.change }}
>
> **Motivation.** {{ fact:card_testing:tested_change.motivation }}
>
> **Noted after the run.** {{ fact:card_testing:tested_change.noted_after_run }}
>
> **Expected mechanism.** {{ fact:card_testing:tested_change.mechanism }}

The case's order and the session's earlier order are declined at checkout under both policies.
Across the test window:

| Measure | Incumbent | With the change | Difference |
|---|---:|---:|---:|
| Fraud orders declined at checkout | {{ fact:card_testing:tested_change.result.world.incumbent.adjudicated.fraud_declined_checkout }} | {{ fact:card_testing:tested_change.result.world.variant.adjudicated.fraud_declined_checkout }} | {{ fact:card_testing:tested_change.result.world.difference.adjudicated.fraud_declined_checkout | count:signed }} |
| Fraud loss prevented | {{ fact:card_testing:tested_change.result.world.incumbent.cash.prevented_loss_cents | usd:2 }} | {{ fact:card_testing:tested_change.result.world.variant.cash.prevented_loss_cents | usd:2 }} | {{ fact:card_testing:tested_change.result.world.difference.cash.prevented_loss_cents | usd:2:signed }} |
| Legitimate orders declined at checkout | {{ fact:card_testing:tested_change.result.world.incumbent.adjudicated.legitimate_declined_checkout }} | {{ fact:card_testing:tested_change.result.world.variant.adjudicated.legitimate_declined_checkout }} | {{ fact:card_testing:tested_change.result.world.difference.adjudicated.legitimate_declined_checkout | count:signed }} |
| Rule net contribution | {{ fact:card_testing:tested_change.result.world.incumbent.cash.rule_net_cents | usd:2 }} | {{ fact:card_testing:tested_change.result.world.variant.cash.rule_net_cents | usd:2 }} | {{ fact:card_testing:tested_change.result.world.difference.cash.rule_net_cents | usd:2:signed }} |

Across the {{ fact:card_testing:tested_change.result.world.incumbent.orders }} orders decided
by the replay, the difference in rule net contribution per thousand orders was
{{ fact:card_testing:tested_change.result.world.difference.rule_net_vs_incumbent_per_1000_orders_cents | usd:2:signed }}
([recommendation rule](../docs/methods.md#recommendation-rule)). The result was negative. The
earlier trigger declined no additional fraud orders and prevented no additional fraud loss.
It increased legitimate checkout declines and reduced rule net contribution.

## Latent truth (diagnostic)

These simulation records are unavailable to rules, analysts and the memo drafter. The
stolen-card pattern and a prior processor decline on the device define the selection slot.
The alert belongs to a stolen-card episode
(`{{ fact:card_testing:alerts.card_testing.latent.episode.pattern_id }}`) by a
{{ fact:card_testing:alerts.card_testing.latent.account.actor }}, with
{{ fact:card_testing:alerts.card_testing.latent.episode.orders }} order attempts from
{{ fact:card_testing:alerts.card_testing.latent.episode.started_at }} to
{{ fact:card_testing:alerts.card_testing.latent.episode.ended_at }}. The incumbent declined
every processor-approved order in the episode at checkout.

## The memo drafter's memo

Pending: the LLM drafter's advisory memo for this alert's packet, built from the evidence row
above, will appear here with the memo benchmark results.
