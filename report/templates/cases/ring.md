# Ring: declined at checkout on a shared device, without an account block

The order came from a device that
{{ fact:ring:alerts.ring.evidence.row.accounts_on_device_30d }} accounts had used within thirty
days. R02 alone reached the checkout decline threshold, so the incumbent declined the order
automatically, recording no fraud finding and blocking no account.

The world is synthetic, and the canonical world the cases come from (seed
{{ fact:ring:source.world.seed }}, {{ fact:ring:source.world.family }} family) is a development
seed, so this short case illustrates how the policy behaves rather than measuring it. It is the
first in hash order of {{ fact:ring:alerts.ring.publication.candidates.eligible }} test-window
alerts on synthetic-identity ring orders ([selection](../docs/methods.md#case-selection)). The
full record is in [the facts file](facts/ring.json).

## Decision and evidence

Alert `{{ fact:ring:alerts.ring.publication.alert_id }}`, decided at checkout on
{{ fact:ring:alerts.ring.decision.checkout_at }} by the score band, from the saved checkout row:

{{ facts:ring:alerts.ring.evidence.row "Fact at checkout" "Value" | amount_cents "Order amount" usd:2, accounts_on_device_30d "Accounts on this device in the past thirty days" count, device_link_age_hours "Hours since the device was first used on this account" num:0, accounts_on_address_30d "Accounts shipping to this address in the past thirty days" count, bin_ip_country_mismatch "Card country differs from the IP country" yesno, approved_orders_user_ever "Earlier orders approved on the account" count, installments_due_user "Installments due on the account" count, installments_paid_user "Installments paid" count }}

Only {{ fact:ring:alerts.ring.evidence.rules_held.0 }} held (three or more accounts on the
device within thirty days). Its weight alone gave a score of
{{ fact:ring:alerts.ring.routing.rule_score | num:0 }}, equal to the decline threshold. The
household exception to R02 (§6.4(a)) does not apply, because the device had been on the
account for less than ninety days. The recorded action was
**`{{ fact:ring:alerts.ring.decision.recorded_action }}`**.

## What FP-2 supports

The checkout band may decline automatically (§4.1); that records no fraud finding and blocks no
account. Reviewed, the same row would show one adverse family
({{ fact:ring:alerts.ring.policy_view.families.0 }}) under row
{{ fact:ring:alerts.ring.policy_view.row }}: standard
`{{ fact:ring:alerts.ring.policy_view.standard.0 }}` with
`{{ fact:ring:alerts.ring.policy_view.required_checks.0 }}`; `needs_check` also permitted;
`clear`, `decline` and `escalate` prohibited. An analyst could not decline on one family, and
escalation, which would also block the linked accounts, needs a failed check or an earlier
outcome that settles the order (§5.3(b), §6.6(a)). The checkout band therefore went further than
an analyst could, declining on one linkage condition, and also did less: it recorded no finding
and blocked none of the device's accounts. The competing readings are a ring of synthetic
identities sharing a device, or a household or shared computer (§6.4); the account's clean
repayment record counts for nothing either way (§6.5(f)).

## What happened later

The order was never created and no cash moved under the incumbent
({{ fact:ring:alerts.ring.later.incumbent.net_cents | usd:2 }}). The account placed
{{ fact:ring:alerts.ring.later.account.later_orders }} later order in the window, also declined
at checkout. Had the order been approved (the approve-all world), it would have shipped and no
installment would ever have been paid: a net of
{{ fact:ring:alerts.ring.later.approve_all.net_cents | usd:2 }}, so the decline prevented
{{ fact:ring:alerts.ring.later.prevented_cents | usd:2 }}. The order's label is
`{{ fact:ring:alerts.ring.later.label.basis }}` (§8.3), known at
{{ fact:ring:alerts.ring.later.label.label_known_at }}.

## Tested change: Linkage-only orders to review instead of automatic decline

Declared before it ran and replayed once against the unchanged incumbent.

> **Change.** {{ fact:ring:tested_change.change }}
>
> **Motivation.** {{ fact:ring:tested_change.motivation }}

Under the change this account was already blocked when it checked out
(`{{ fact:ring:tested_change.result.case_order.variant.route }}`): an analyst had escalated an
order from a linked account, which blocked it too, and its later order was refused the same way.
Across the test window, escalations changed by
{{ fact:ring:tested_change.result.world.difference.review.escalations | count:signed }} and
blocked accounts by
{{ fact:ring:tested_change.result.world.difference.review.accounts_blocked | count:signed }}, as
intended; but fraud orders declined at checkout changed by
{{ fact:ring:tested_change.result.world.difference.adjudicated.fraud_declined_checkout | count:signed }},
reviews decided only after the order had shipped by
{{ fact:ring:tested_change.result.world.difference.review.decided_after_shipping | count:signed }},
review minutes by
{{ fact:ring:tested_change.result.world.difference.review.review_minutes_used | count:signed }}
and fraud loss prevented by
{{ fact:ring:tested_change.result.world.difference.cash.prevented_loss_cents | usd:2:signed }}.
The rule net contribution ([definition](../docs/methods.md#recommendation-rule)) changed by
{{ fact:ring:tested_change.result.world.difference.rule_net_vs_incumbent_per_1000_orders_cents | usd:2:signed }}
per thousand decided orders. A negative result: orders the band used to decline now waited for
review, and some shipped before an analyst reached them.

## Latent truth (diagnostic)

Unavailable to rules and analysts: the account's holder is
`{{ fact:ring:alerts.ring.latent.account.actor }}`, one of
{{ fact:ring:alerts.ring.latent.episode.accounts }} accounts in a ring
(`{{ fact:ring:alerts.ring.latent.episode.pattern_id }}`) with
{{ fact:ring:alerts.ring.latent.episode.orders }} orders. The platform's net from the ring's
test-window orders was {{ fact:ring:tested_change.result.latent_episode.incumbent.net_cents | usd:2 }}
under the incumbent and
{{ fact:ring:tested_change.result.latent_episode.variant.net_cents | usd:2 }} with the change,
which blocked ring accounts but did not reach the ring orders the rules had approved.

## The memo drafter's memo

Pending: the LLM drafter's memo for this alert's packet will appear here with the memo
benchmark results.
