# Never-pay versus hardship: both cleared after a passed identity check; one plan was never paid, the other defaulted after two installments

The never-pay order reached review because accounts shared its shipping address. The
hardship order reached review because the card and IP countries differed and the CVV check
failed. Both came from new accounts. The analyst held each order before shipment and started
an identity check. Neither account had an installment due, so repayment could not support
the decision, and the check could not tell the two apart.

The world is synthetic, and the canonical world the cases come from (seed
{{ fact:never_pay_vs_hardship:source.world.seed }},
{{ fact:never_pay_vs_hardship:source.world.family }} family) is a development seed, so the case
illustrates how the policy behaves rather than measuring it. The selection rule was
fixed before evaluation ([methods](../docs/methods.md#case-selection)). Each slot takes its
first analyst-reviewed alert in hash order:

| | Never-pay order | Hardship order |
|---|---|---|
| Alert (order id and policy version) | `{{ fact:never_pay_vs_hardship:alerts.never_pay.publication.alert_id }}` | `{{ fact:never_pay_vs_hardship:alerts.hardship.publication.alert_id }}` |
| Test-window alerts in the slot | {{ fact:never_pay_vs_hardship:alerts.never_pay.publication.candidates.eligible }} | {{ fact:never_pay_vs_hardship:alerts.hardship.publication.candidates.eligible }} |
| Of those, reviewed by an analyst | {{ fact:never_pay_vs_hardship:alerts.never_pay.publication.candidates.reviewed }} | {{ fact:never_pay_vs_hardship:alerts.hardship.publication.candidates.reviewed }} |

Both orders come from the incumbent replay in the test window, with tuned thresholds, the
base review allotment, the evidence-based reviewer and standard verification rates. Values
come from [the facts file](facts/never_pay_vs_hardship.json).

## Decision times

| Moment | Never-pay order | Hardship order |
|---|---|---|
| Checkout | {{ fact:never_pay_vs_hardship:alerts.never_pay.decision.checkout_at }} | {{ fact:never_pay_vs_hardship:alerts.hardship.decision.checkout_at }} |
| Evidence assembled | {{ fact:never_pay_vs_hardship:alerts.never_pay.decision.evidence_at }} | {{ fact:never_pay_vs_hardship:alerts.hardship.decision.evidence_at }} |
| Analyst started the review | {{ fact:never_pay_vs_hardship:alerts.never_pay.decision.analyst_started_at }} | {{ fact:never_pay_vs_hardship:alerts.hardship.decision.analyst_started_at }} |
| Analyst acted | {{ fact:never_pay_vs_hardship:alerts.never_pay.decision.action_at }} | {{ fact:never_pay_vs_hardship:alerts.hardship.decision.action_at }} |

The hardship order reached checkout in the evening and review the next morning. Its evidence
row was rebuilt at the start of the review day, under the replay's daily refresh
([methods](../docs/methods.md#what-it-reads)).

## Evidence at decision time

These are the saved context rows used at review. They contain only information available at
the evidence times above.

| Fact at the decision | Never-pay order | Hardship order |
|---|---:|---:|
| Order amount | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.amount_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.amount_cents | usd:2 }} |
| Cash at risk if approved | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.order_exposure_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.order_exposure_cents | usd:2 }} |
| Account age at checkout (days) | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.account_age_days | num:2 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.account_age_days | num:2 }} |
| The account's first order attempt | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.is_first_attempt_user | yesno }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.is_first_attempt_user | yesno }} |
| Earlier orders approved on the account | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.approved_orders_user_ever }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.approved_orders_user_ever }} |
| Open balance on earlier plans | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.open_balance_user_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.open_balance_user_cents | usd:2 }} |
| Installments due on the account | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.installments_due_user }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.installments_due_user }} |
| Accounts shipping to this address in the past thirty days | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.accounts_on_address_30d }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.accounts_on_address_30d }} |
| Ships to the account's home address | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.ship_to_home | yesno }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.ship_to_home | yesno }} |
| Age of the home address (days) | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.home_address_age_days | num:2 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.home_address_age_days | num:2 }} |
| Hours since the device was first used on this account | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.device_link_age_hours | num:2 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.device_link_age_hours | num:2 }} |
| Card country differs from the IP country | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.bin_ip_country_mismatch | yesno }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.bin_ip_country_mismatch | yesno }} |
| CVV check failed | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.cvv_mismatch | yesno }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.cvv_mismatch | yesno }} |
| AVS check failed | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.avs_mismatch | yesno }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.row.avs_mismatch | yesno }} |
| Rule condition that held | {{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.rules_held.0 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.evidence.rules_held.0 }} |
| Its family | {{ fact:never_pay_vs_hardship:alerts.never_pay.policy_view.families.0 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.policy_view.families.0 }} |
| Rule score (review threshold {{ fact:never_pay_vs_hardship:source.replay.review_threshold | num:0 }}, decline threshold {{ fact:never_pay_vs_hardship:source.replay.decline_threshold | num:0 }}) | {{ fact:never_pay_vs_hardship:alerts.never_pay.routing.rule_score | num:0 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.routing.rule_score | num:0 }} |

The never-pay account placed another order while its earlier plan was open, before any
installment had fallen due. R08 held:
{{ fact:never_pay_vs_hardship:alerts.never_pay.evidence.row.accounts_on_address_30d }}
accounts, including this account, had attempted orders to the address within thirty days.
It is the account's home address, but it is too recent for the R08 household exception,
which requires at least ninety days (§6.4(b)). The hardship order was a first attempt from
a newly opened account. R03 held because the card's issuing country differed from the IP
country and the CVV check failed.

## What FP-2 supports on this evidence

The [fraud policy](../policy/fraud-policy.md) prescribes the same action for both rows.
Each has one adverse family, so row
{{ fact:never_pay_vs_hardship:alerts.never_pay.policy_view.row }} applies to both:

- **Standard:** `{{ fact:never_pay_vs_hardship:alerts.never_pay.policy_view.standard.0 }}` with
  `{{ fact:never_pay_vs_hardship:alerts.never_pay.policy_view.required_checks.0 }}`, the check
  §5.2 requires for both Linkage and Card.
- **Also permitted:** `needs_check`, the memo recommendation for a hold with that check
  (§4.3, §6.6(b)).
- **Prohibited:** `clear`, `decline` and `escalate` (§6.6(b)).

No installment had fallen due on either account. Under §8.1, non-payment or never-pay cannot
justify an action, determination or memo recommendation; a memo may name never-pay as a
hypothesis. The identity check establishes who is ordering, not whether they intend to repay
(§5.1).

The competing explanations at the decisions:

- **Never-pay order.** The new account's open plan and shared address could fit a customer
  stacking purchases without intending to repay. They could also fit a genuine repeat buyer
  in a household or shared building (§6.4, §6.5(b)). The address triggers Linkage, but no
  missed payment is known.
- **Hardship order.** The country mismatch and failed CVV could reflect stolen card details.
  A genuine new customer could also be using a card issued in another country and mistype
  the code; the first attempt does not establish fraud (§6.5(b), §6.5(c)). Never-pay remains
  a possible hypothesis, but this row gives no evidence of repayment intent.

## Recorded action in the replay

The simulated analyst recorded the standard action on each order:
**`{{ fact:never_pay_vs_hardship:alerts.never_pay.decision.recorded_action }}`** on the
never-pay order and **`{{ fact:never_pay_vs_hardship:alerts.hardship.decision.recorded_action }}`**
on the hardship order. Both actions preceded shipment and started an identity check.

## What happened later

The following outcomes were unavailable at the decisions. The required checks passed, so
FP-2 required the analyst to clear the orders (§5.3(a)). Both shipped. Labels come from the
approve-all world and are used to score every policy.

| | Never-pay order | Hardship order |
|---|---|---|
| Identity check | {{ fact:never_pay_vs_hardship:alerts.never_pay.later.review.checks.0.outcome }} at {{ fact:never_pay_vs_hardship:alerts.never_pay.later.review.checks.0.completed_at }} | {{ fact:never_pay_vs_hardship:alerts.hardship.later.review.checks.0.outcome }} at {{ fact:never_pay_vs_hardship:alerts.hardship.later.review.checks.0.completed_at }} |
| Final decision (§5.3(a)) | {{ fact:never_pay_vs_hardship:alerts.never_pay.later.review.final }} | {{ fact:never_pay_vs_hardship:alerts.hardship.later.review.final }} |
| Released from the hold | {{ fact:never_pay_vs_hardship:alerts.never_pay.later.fate.released_at }} | {{ fact:never_pay_vs_hardship:alerts.hardship.later.fate.released_at }} |
| Net for the platform | {{ fact:never_pay_vs_hardship:alerts.never_pay.later.incumbent.net_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:alerts.hardship.later.incumbent.net_cents | usd:2 }} |
| Label | `{{ fact:never_pay_vs_hardship:alerts.never_pay.later.label.basis }}`, known {{ fact:never_pay_vs_hardship:alerts.never_pay.later.label.label_known_at }} | `{{ fact:never_pay_vs_hardship:alerts.hardship.later.label.basis }}`, known {{ fact:never_pay_vs_hardship:alerts.hardship.later.label.label_known_at }} |

**The never-pay order** under the incumbent policy:

{{ table:fact:never_pay_vs_hardship:alerts.never_pay.later.incumbent.events | known_at "Known", event "Event", detail "Detail", amount_cents "Amount" usd:2 }}

{{ table:fact:never_pay_vs_hardship:alerts.never_pay.later.incumbent.cash | known_at "Known", kind "Cash event", amount_cents "Amount" usd:2 }}

No installment payment succeeded. The plan met the zero-effort default definition (§8.2)
and had the intent marker required for a never-pay determination (§8.3), confirming
first-party fraud (§9(c)). The later recovery is recorded separately from repayment.

**The hardship order** under the incumbent policy:

{{ table:fact:never_pay_vs_hardship:alerts.hardship.later.incumbent.events | known_at "Known", event "Event", detail "Detail", amount_cents "Amount" usd:2 }}

{{ table:fact:never_pay_vs_hardship:alerts.hardship.later.incumbent.cash | known_at "Known", kind "Cash event", amount_cents "Amount" usd:2 }}

Installments were paid before the final installment failed, so the plan is not a zero-effort
default. With no fraud confirmed, the unpaid balance is a credit loss and does not receive
a fraud label (§8.4).

**Had the orders been approved (the approve-all world).** Without the holds, each order ships
earlier and its installment schedule starts from checkout rather than release from the hold.
The same payments succeed and fail, and the nets are unchanged:
{{ fact:never_pay_vs_hardship:alerts.never_pay.later.approve_all.net_cents | usd:2 }} for the
never-pay order and {{ fact:never_pay_vs_hardship:alerts.hardship.later.approve_all.net_cents | usd:2 }}
for the hardship order. The holds prevented no loss. They delayed shipment and used analyst
time without establishing repayment intent.

## Tested change: no second plan before the first installment is due

The change, motivation and expected mechanism were declared before testing. The change was
replayed once against the unchanged incumbent.

> **Change.** {{ fact:never_pay_vs_hardship:tested_change.change }}
>
> **Motivation.** {{ fact:never_pay_vs_hardship:tested_change.motivation }}
>
> **Expected mechanism.** {{ fact:never_pay_vs_hardship:tested_change.mechanism }}

**This case's orders.** The never-pay order is declined at checkout
(`{{ fact:never_pay_vs_hardship:tested_change.result.case_order.variant.route }}`), so its net
becomes {{ fact:never_pay_vs_hardship:tested_change.result.case_order.variant.net_cents | usd:2 }}
instead of {{ fact:never_pay_vs_hardship:tested_change.result.case_order.incumbent.net_cents | usd:2 }}.
The account's earlier order is unchanged: it was held and cleared under both policies. The
hardship order had no earlier open plan, so the new condition does not apply to it.

**The whole test window.**

| Measure | Incumbent | With the change | Difference |
|---|---:|---:|---:|
| Fraud loss prevented | {{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.cash.prevented_loss_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.variant.cash.prevented_loss_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.difference.cash.prevented_loss_cents | usd:2:signed }} |
| Legitimate orders declined | {{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.adjudicated.legitimate_declined }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.variant.adjudicated.legitimate_declined }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.difference.adjudicated.legitimate_declined | count:signed }} |
| Friction cost of declined and cancelled legitimate orders | {{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.cash.friction_cost_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.variant.cash.friction_cost_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.difference.cash.friction_cost_cents | usd:2:signed }} |
| Legitimate orders held | {{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.adjudicated.legitimate_held }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.variant.adjudicated.legitimate_held }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.difference.adjudicated.legitimate_held | count:signed }} |
| Reviews | {{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.review.reviews }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.variant.review.reviews }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.difference.review.reviews | count:signed }} |
| Review minutes used | {{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.review.review_minutes_used }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.variant.review.review_minutes_used }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.difference.review.review_minutes_used | count:signed }} |
| Rule net contribution | {{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.cash.rule_net_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.variant.cash.rule_net_cents | usd:2 }} | {{ fact:never_pay_vs_hardship:tested_change.result.world.difference.cash.rule_net_cents | usd:2:signed }} |

Rule net contribution is the ledger net after subtracting
${{ config:policy:costs.false_decline_ltv_usd | num:0 }} for each legitimate order declined or
cancelled and the cost of
the analyst allotment, as in the
[recommendation rule](../docs/methods.md#recommendation-rule). Both policies have the same
allotment, so changes in minutes used do not change that cost. Holds are counted but not
priced. Per thousand of the
{{ fact:never_pay_vs_hardship:tested_change.result.world.incumbent.orders }} orders the replay
decided, the change's difference is
{{ fact:never_pay_vs_hardship:tested_change.result.world.difference.rule_net_vs_incumbent_per_1000_orders_cents | usd:2:signed }}.

The change prevented more fraud loss and raised rule net contribution, while declining
{{ fact:never_pay_vs_hardship:tested_change.result.world.difference.adjudicated.legitimate_declined }}
more legitimate orders. The net prices those declines at the assumed lifetime-value proxy;
whether that is the right price is a business judgement the replay cannot make.
The condition limits exposure on additional plans before an installment is due; it does not
establish repayment intent or distinguish hardship from never-pay on a first order.

## Latent truth (diagnostic)

These simulation records are unavailable to rules, analysts and the memo drafter. The
never-pay pattern defines one selection slot; the generator's natural-default draw for a
legitimate customer defines the hardship slot. These traits do not rank candidates, and
later outcomes do not select either order.

| Truth | Never-pay order | Hardship order |
|---|---|---|
| Intent of the order | {{ fact:never_pay_vs_hardship:alerts.never_pay.latent.order.intent }} | {{ fact:never_pay_vs_hardship:alerts.hardship.latent.order.intent }} |
| Account holder | {{ fact:never_pay_vs_hardship:alerts.never_pay.latent.account.actor }} | {{ fact:never_pay_vs_hardship:alerts.hardship.latent.account.actor }} |
| Fraud pattern or benign trait | {{ fact:never_pay_vs_hardship:alerts.never_pay.latent.order.pattern_id }} | `{{ fact:never_pay_vs_hardship:alerts.hardship.latent.order.mimic }}` |

The never-pay order belongs to an episode of
{{ fact:never_pay_vs_hardship:alerts.never_pay.latent.episode.orders }} orders across
{{ fact:never_pay_vs_hardship:alerts.never_pay.latent.episode.accounts }} accounts, from
{{ fact:never_pay_vs_hardship:alerts.never_pay.latent.episode.started_at }} to
{{ fact:never_pay_vs_hardship:alerts.never_pay.latent.episode.ended_at }}. The platform's net
from the episode's orders was
{{ fact:never_pay_vs_hardship:tested_change.result.latent_episode.incumbent.net_cents | usd:2 }}
under the incumbent and
{{ fact:never_pay_vs_hardship:tested_change.result.latent_episode.variant.net_cents | usd:2 }}
with the tested change.
The hardship customer is legitimate and was drawn to default by the generator.

## The memo drafter's memo

Pending: the LLM drafter's advisory memos for these alerts' packets, built from the evidence
rows above, will appear here with the memo benchmark results.
