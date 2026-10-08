# Account takeover: held for an identity check, which failed, and declined before shipping

An established account placed an order from a newly seen device, for delivery to an address
it had not used before. The incumbent rules sent the order to review. The analyst held it
before shipment for an identity check; the check failed the next day and the order was declined
before the merchant shipped it.

The world is synthetic, and the canonical world the cases come from (seed
{{ fact:account_takeover:source.world.seed }}, {{ fact:account_takeover:source.world.family }}
family) is a development seed, so the case illustrates how the policy behaves rather than
measuring it. The selection rule was fixed before evaluation
([methods](../docs/methods.md#case-selection)): of the
{{ fact:account_takeover:alerts.account_takeover.publication.candidates.eligible }}
test-window alerts eligible for this case's slot (see the latent truth below),
{{ fact:account_takeover:alerts.account_takeover.publication.candidates.reviewed }} were
reviewed by an analyst. The case takes the first reviewed alert in hash order. Values come
from [the facts file](facts/account_takeover.json).

| | |
|---|---|
| Alert | `{{ fact:account_takeover:alerts.account_takeover.publication.alert_id }}` (order id and policy version) |
| Replay | the incumbent rules with their tuned thresholds, in the test window, at the base review allotment, with the evidence-based reviewer and standard verification rates |

## Decision times

{{ facts:account_takeover:alerts.account_takeover.decision "Moment" "Time" | checkout_at "Checkout", evidence_at "Evidence assembled", analyst_started_at "Analyst started the review", action_at "Analyst acted" }}

## Evidence at decision time

The analyst used this saved context row, assembled at the evidence time above. It contains
only information available then.

{{ facts:account_takeover:alerts.account_takeover.evidence.row "Fact at the decision" "Value" | amount_cents "Order amount" usd:2, order_exposure_cents "Cash at risk if approved" usd:2, account_age_days "Account age (days)" num:0, device_link_age_hours "Hours since the device was first used on this account" num:2, ship_address_first_use_age_hours "Hours since the account's first order attempt to this address" num:2, ship_to_home "Ships to the account's home address" yesno, home_address_age_days "Age of the current home address (days)" num:0, ip_country_not_home "IP country differs from the home country" yesno, bin_ip_country_mismatch "Card country differs from the IP country" yesno, cvv_mismatch "Card security code (CVV) check failed" yesno, avs_mismatch "Address verification (AVS) failed: billing address not the issuer's record" yesno, card_first_use_age_hours "Hours since the account first used this card" num:0, hours_since_credential_change "Hours since a password change or reset or an email change (capped)" num:0, accounts_on_device_30d "Accounts on this device in the past thirty days" count, accounts_on_address_30d "Accounts with an order attempt to this address in the past thirty days" count, approved_orders_user_ever "Earlier orders approved on the account" count, installments_due_user "Installments due on the account" count, installments_paid_user "Installments paid" count, merchant_fulfilment_median_hours "Merchant's median hours to shipment" num:1 }}

Only {{ fact:account_takeover:alerts.account_takeover.evidence.rules_held.0 }} held: the card's
issuing country differed from the IP country and the CVV check failed. The score,
{{ fact:account_takeover:alerts.account_takeover.routing.rule_score | num:0 }}, met the review
threshold ({{ fact:account_takeover:source.replay.review_threshold | num:0 }}) and stayed under
the decline threshold ({{ fact:account_takeover:source.replay.decline_threshold | num:0 }}), so
the order went to the review queue. The credential-change columns are capped. No password
change, reset or email change was recent enough for R01, which requires one within the two
days before the order.

## What FP-2 supports on this evidence

Under the [fraud policy](../policy/fraud-policy.md), R03 supplies the only adverse family,
{{ fact:account_takeover:alerts.account_takeover.policy_view.families.0 }}. Row
{{ fact:account_takeover:alerts.account_takeover.policy_view.row }} applies:

- **Standard:** `{{ fact:account_takeover:alerts.account_takeover.policy_view.standard.0 }}`
  with the check §5.2 requires for the Card family,
  `{{ fact:account_takeover:alerts.account_takeover.policy_view.required_checks.0 }}`: the
  customer authenticates the payment with the card's issuer and verifies the identity on the
  account.
- **Also permitted:** `needs_check`, the memo recommendation for a hold with that check
  (§4.3, §6.6(b)).
- **Prohibited:** `clear`, `decline` and `escalate` (§6.6(b)).

No Account access condition holds, so `contact` is not required (§5.2). A newly seen device
and shipping address do not supply an adverse family on their own (§6.5(b)).

The competing explanations at the decision:

1. **Account takeover without a credential change.** An attacker could use the account from
   the newly seen device and send goods to the unused address. The foreign IP and failed CVV
   fit that explanation, but the row does not establish who ordered or how they gained access.
2. **The holder using a new phone while away (benign).** Travel, a replacement phone, a gift
   or a move could explain the device, IP and shipping address (§6.5(b)). The current home
   address is also recent. The failed CVV still requires verification under R03.
3. **Stolen card details.** The failed CVV could reflect misuse of the card. The card's long
   history on this account does not establish who used it for this order.

Account age and repayment history do not establish who ordered and cannot support an adverse
action here (§6.4, §6.5(f)).

## The memo drafter's memo

The project's memo drafter wrote this memo in the benchmark's case phase
([protocol](../llm/eval/PROTOCOL.md)). Its packet came from the evidence row above, with no check
completed. The pinned model was `{{ bench:2026-10-cases:/pins/sol/model }}` at
{{ bench:2026-10-cases:/pins/sol/effort }} effort ({{ bench:2026-10-cases:/pins/sol/cli_version }}).

| Recommendation and score | |
|---|---|
| Recommended disposition | `{{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/memo/disposition }}` |
| Next check | `{{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/memo/next_check }}` |
| Explanations named (likelihood) | `{{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/memo/hypotheses/[explanation=stolen_card]/explanation }}` ({{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/memo/hypotheses/[explanation=stolen_card]/likelihood }}); `{{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/memo/hypotheses/[explanation=traveller]/explanation }}` ({{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/memo/hypotheses/[explanation=traveller]/likelihood }}) |
| Complete benchmark pass | {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/score/complete_pass | yesno }} |
| Acceptable under FP-2 | {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/score/acceptable | yesno }} |
| Matches FP-2's standard action | {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/score/standard_action | yesno }} |
| Names the required check | {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/score/next_check_ok | yesno }} |
| Citations valid | {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/score/citation_ok | yesno }} |
| Structured claims checked | {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/score/verification/n_claims }} |
| Structured claim errors found | {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/score/verification/n_claim_errors }} |

For this packet, FP-2's standard action is `{{ fact:account_takeover:alerts.account_takeover.policy_view.standard.0 }}` with
`{{ fact:account_takeover:alerts.account_takeover.policy_view.required_checks.0 }}` under {{ fact:account_takeover:alerts.account_takeover.policy_view.row }}; `needs_check` is also
permitted. The memo agrees with that action and check: the analyst would carry out its
`needs_check` recommendation as a hold with the required check (§4.3).

It names stolen card use and travel, but omits takeover without a credential change, which
the case lists first. The required check is the same. In its own words:

> {{ bench:2026-10-cases:/case_memos/[file=account_takeover,slot=account_takeover]/arms/sol/memo/memo }}

The memo is advisory; the case's recorded action does not depend on it.

## Recorded action in the replay

The simulated analyst recorded the standard action,
**`{{ fact:account_takeover:alerts.account_takeover.decision.recorded_action }}`** at
{{ fact:account_takeover:alerts.account_takeover.decision.action_at }}, before shipment, and
started the identity check.

## What happened later

The following outcomes were unavailable at the decision.

The identity check returned
**{{ fact:account_takeover:alerts.account_takeover.later.review.checks.0.outcome }}** at
{{ fact:account_takeover:alerts.account_takeover.later.review.checks.0.completed_at }}. A failed
check decides the order immediately (§5.3(b)). With no Linkage present, the standard is
`decline`, so the analyst's
final decision was **{{ fact:account_takeover:alerts.account_takeover.later.review.final }}**.
The hold had paused shipment. The decline voided the order, refunded the checkout payment and
blocked the account (§4.2). The account placed no later order in the window.

Under the incumbent policy:

{{ table:fact:account_takeover:alerts.account_takeover.later.incumbent.cash | known_at "Known", kind "Cash event", amount_cents "Amount" usd:2 }}

Net for the platform: {{ fact:account_takeover:alerts.account_takeover.later.incumbent.net_cents | usd:2 }}.

**Had the order been approved (the approve-all world).** The merchant ships, the account holder
reports orders they did not place, and the plan is written off:

{{ table:fact:account_takeover:alerts.account_takeover.later.approve_all.events | known_at "Known", event "Event", detail "Detail", amount_cents "Amount" usd:2 }}

{{ table:fact:account_takeover:alerts.account_takeover.later.approve_all.cash | known_at "Known", kind "Cash event", amount_cents "Amount" usd:2 }}

The approve-all net would have been
{{ fact:account_takeover:alerts.account_takeover.later.approve_all.net_cents | usd:2 }}, so the
hold and decline prevented a loss of
{{ fact:account_takeover:alerts.account_takeover.later.prevented_cents | usd:2 }}. The order's
label, which every policy is scored with, comes from the approve-all world:
`{{ fact:account_takeover:alerts.account_takeover.later.label.basis }}`, known at
{{ fact:account_takeover:alerts.account_takeover.later.label.label_known_at }}, when the victim's
report arrived (§9(b)).

## Tested change: an account-access condition without a credential change

The change, motivation and expected mechanism were declared before testing. The change was
replayed once against the unchanged incumbent.

> **Change.** {{ fact:account_takeover:tested_change.change }}
>
> **Motivation.** {{ fact:account_takeover:tested_change.motivation }}
>
> **Expected mechanism.** {{ fact:account_takeover:tested_change.mechanism }}

**This case's order.** With the change, the order scores
{{ fact:account_takeover:tested_change.result.case_order.variant.rule_score | num:0 }} and is
declined at checkout (`{{ fact:account_takeover:tested_change.result.case_order.variant.route }}`)
instead of held and declined after the check. Its net is the same either way
({{ fact:account_takeover:tested_change.result.case_order.variant.net_cents | usd:2 }}); the
automatic decline uses no analyst time, records no fraud finding and blocks no account (§4.1).

**The whole test window.**

| Measure | Incumbent | With the change | Difference |
|---|---:|---:|---:|
| Fraud loss prevented | {{ fact:account_takeover:tested_change.result.world.incumbent.cash.prevented_loss_cents | usd:2 }} | {{ fact:account_takeover:tested_change.result.world.variant.cash.prevented_loss_cents | usd:2 }} | {{ fact:account_takeover:tested_change.result.world.difference.cash.prevented_loss_cents | usd:2:signed }} |
| Legitimate orders declined | {{ fact:account_takeover:tested_change.result.world.incumbent.adjudicated.legitimate_declined }} | {{ fact:account_takeover:tested_change.result.world.variant.adjudicated.legitimate_declined }} | {{ fact:account_takeover:tested_change.result.world.difference.adjudicated.legitimate_declined | count:signed }} |
| Legitimate orders cancelled after a hold | {{ fact:account_takeover:tested_change.result.world.incumbent.adjudicated.legitimate_cancelled }} | {{ fact:account_takeover:tested_change.result.world.variant.adjudicated.legitimate_cancelled }} | {{ fact:account_takeover:tested_change.result.world.difference.adjudicated.legitimate_cancelled | count:signed }} |
| Legitimate orders held | {{ fact:account_takeover:tested_change.result.world.incumbent.adjudicated.legitimate_held }} | {{ fact:account_takeover:tested_change.result.world.variant.adjudicated.legitimate_held }} | {{ fact:account_takeover:tested_change.result.world.difference.adjudicated.legitimate_held | count:signed }} |
| Reviews | {{ fact:account_takeover:tested_change.result.world.incumbent.review.reviews }} | {{ fact:account_takeover:tested_change.result.world.variant.review.reviews }} | {{ fact:account_takeover:tested_change.result.world.difference.review.reviews | count:signed }} |
| Review minutes used | {{ fact:account_takeover:tested_change.result.world.incumbent.review.review_minutes_used }} | {{ fact:account_takeover:tested_change.result.world.variant.review.review_minutes_used }} | {{ fact:account_takeover:tested_change.result.world.difference.review.review_minutes_used | count:signed }} |
| Rule net contribution | {{ fact:account_takeover:tested_change.result.world.incumbent.cash.rule_net_cents | usd:2 }} | {{ fact:account_takeover:tested_change.result.world.variant.cash.rule_net_cents | usd:2 }} | {{ fact:account_takeover:tested_change.result.world.difference.cash.rule_net_cents | usd:2:signed }} |

Rule net contribution is the ledger net after subtracting
${{ config:policy:costs.false_decline_ltv_usd | num:0 }} for each legitimate order declined or
cancelled and the cost of
the analyst allotment at
${{ config:policy:costs.analyst_loaded_hourly_usd | num:0 }} an hour, as in the
[recommendation rule](../docs/methods.md#recommendation-rule). Both policies have the same
allotment, so changes in minutes used do not change that cost. Holds are counted but not
priced. Per thousand of the
{{ fact:account_takeover:tested_change.result.world.incumbent.orders }} orders the replay
decided, the change's difference is
{{ fact:account_takeover:tested_change.result.world.difference.rule_net_vs_incumbent_per_1000_orders_cents | usd:2:signed }}.
The change prevented more fraud loss and raised rule net contribution. It also increased
legitimate declines, holds and review time.

## Latent truth (diagnostic)

These simulation records are unavailable to rules, analysts and the memo drafter. The slot is
defined by the simulated takeover pattern (`P-ATO`); that defines which alerts are eligible but
does not rank them.

{{ facts:account_takeover:alerts.account_takeover.latent.order "The order" "Truth" | pattern_id "Fraud pattern", intent "Intent" }}

{{ facts:account_takeover:alerts.account_takeover.latent.account "The account" "Truth" | actor "Account holder" }}

The account belongs to a legitimate customer. The takeover episode began at
{{ fact:account_takeover:alerts.account_takeover.latent.episode.started_at }} and contains only
this order.
