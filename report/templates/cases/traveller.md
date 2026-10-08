# Traveller: held for an identity check, passed the next day, cleared and shipped

An established account placed an order from a long-used device, for delivery to its home
address, with an IP in another country. The card's issuing country differed from the IP
country and the address verification check (AVS, which compares the billing address with the
card issuer's record) failed, triggering review. The analyst held the order before
shipment for an identity check, as the policy requires; the customer passed it the next day and
the order shipped.

The world is synthetic, and the canonical world the cases come from (seed
{{ fact:traveller:source.world.seed }}, {{ fact:traveller:source.world.family }} family) is a
development seed, so the case illustrates how the policy behaves rather than measuring it.
The selection rule was fixed before evaluation
([methods](../docs/methods.md#case-selection)): of the
{{ fact:traveller:alerts.traveller.publication.candidates.eligible }} test-window alerts
eligible for this case's slot (see the latent truth below),
{{ fact:traveller:alerts.traveller.publication.candidates.reviewed }} were reviewed by an
analyst. The case takes the first reviewed alert in hash order. Values come from
[the facts file](facts/traveller.json).

| | |
|---|---|
| Alert | `{{ fact:traveller:alerts.traveller.publication.alert_id }}` (order id and policy version) |
| Replay | the incumbent rules with their tuned thresholds, in the test window, at the base review allotment, with the evidence-based reviewer and standard verification rates |

## Decision times

{{ facts:traveller:alerts.traveller.decision "Moment" "Time" | checkout_at "Checkout", evidence_at "Evidence assembled", analyst_started_at "Analyst started the review", action_at "Analyst acted" }}

## Evidence at decision time

The analyst used this saved context row, assembled at the evidence time above. It contains
only information available then.

{{ facts:traveller:alerts.traveller.evidence.row "Fact at the decision" "Value" | amount_cents "Order amount" usd:2, order_exposure_cents "Cash at risk if approved" usd:2, account_age_days "Account age (days)" num:0, device_link_age_hours "Hours since the device was first used on this account" num:0, ship_to_home "Ships to the account's home address" yesno, home_address_age_days "Age of the home address (days)" num:0, ip_country_not_home "IP country differs from the home country" yesno, bin_ip_country_mismatch "Card country differs from the IP country" yesno, avs_mismatch "Address verification (AVS) failed" yesno, cvv_mismatch "Card security code (CVV) check failed" yesno, card_first_use_age_hours "Hours since the account first used this card" num:0, hours_since_credential_change "Hours since a password change or reset or an email change (capped)" num:0, accounts_on_device_30d "Accounts on this device in the past thirty days" count, accounts_on_address_30d "Accounts with an order attempt to this address in the past thirty days" count, approved_orders_user_ever "Earlier orders approved on the account" count, installments_due_user "Installments due on the account" count, installments_paid_user "Installments paid" count, merchant_fulfilment_median_hours "Merchant's median hours to shipment" num:1 }}

Only {{ fact:traveller:alerts.traveller.evidence.rules_held.0 }} held: the card's issuing
country differed from the IP country and the AVS check failed. The score,
{{ fact:traveller:alerts.traveller.routing.rule_score | num:0 }}, met the review threshold
({{ fact:traveller:source.replay.review_threshold | num:0 }}) and stayed under the decline
threshold ({{ fact:traveller:source.replay.decline_threshold | num:0 }}), routing the order to
review. The device, card and home address have been associated with the account since it
opened. No recent credential change is recorded, and the
{{ fact:traveller:alerts.traveller.evidence.row.accounts_on_address_30d }} accounts with order
attempts to the address within thirty days are below R08's threshold of three.

## What FP-2 supports on this evidence

Under the [fraud policy](../policy/fraud-policy.md), R03 supplies the only adverse family,
{{ fact:traveller:alerts.traveller.policy_view.families.0 }}. Row
{{ fact:traveller:alerts.traveller.policy_view.row }} applies:

- **Standard:** `{{ fact:traveller:alerts.traveller.policy_view.standard.0 }}` with
  `{{ fact:traveller:alerts.traveller.policy_view.required_checks.0 }}`, the check §5.2
  requires for the Card family.
- **Also permitted:** `needs_check`, the memo recommendation for a hold with that check
  (§4.3, §6.6(b)).
- **Prohibited:** `clear`, `decline` and `escalate` (§6.6(b)).

However benign the rest of the row looks, the analyst cannot clear it at first sight. Travel
could explain the foreign IP (§6.5(b)), but it does not exempt R03 from verification.
The household exceptions cover R02 and R08 (§6.4). Account age and repayment history do not
establish who ordered and cannot support an adverse action here (§6.4, §6.5(f)).

The competing explanations at the decision:

1. **The customer travelling (benign).** The long-used device and card, home delivery and
   foreign IP fit a customer ordering while away. The AVS failure alone would not support an
   adverse action (§6.5(a)); the mismatch between card and IP countries makes R03 apply.
2. **Stolen card details.** The country mismatch and failed AVS fit this explanation, though
   it must also account for the known device and delivery to the account's home. A known
   device does not prove who used it.
3. **Account takeover.** No new device, shipping address or recent credential change is
   recorded. The row offers no account-access condition, but it does not establish the
   orderer's identity.

## The memo drafter's memo

The project's memo drafter wrote this memo in the benchmark's case phase
([protocol](../llm/eval/PROTOCOL.md)). Its packet came from the evidence row above, with no check
completed. The pinned model was `{{ bench:2026-10-cases:/pins/sol/model }}` at
{{ bench:2026-10-cases:/pins/sol/effort }} effort ({{ bench:2026-10-cases:/pins/sol/cli_version }}).

| Recommendation and score | |
|---|---|
| Recommended disposition | `{{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/memo/disposition }}` |
| Next check | `{{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/memo/next_check }}` |
| Explanations named (likelihood) | `{{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/memo/hypotheses/[explanation=traveller]/explanation }}` ({{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/memo/hypotheses/[explanation=traveller]/likelihood }}); `{{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/memo/hypotheses/[explanation=stolen_card]/explanation }}` ({{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/memo/hypotheses/[explanation=stolen_card]/likelihood }}) |
| Complete benchmark pass | {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/score/complete_pass | yesno }} |
| Acceptable under FP-2 | {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/score/acceptable | yesno }} |
| Matches FP-2's standard action | {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/score/standard_action | yesno }} |
| Names the required check | {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/score/next_check_ok | yesno }} |
| Citations valid | {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/score/citation_ok | yesno }} |
| Structured claims checked | {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/score/verification/n_claims }} |
| Structured claim errors found | {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/score/verification/n_claim_errors }} |

For this packet, FP-2's standard action is `{{ fact:traveller:alerts.traveller.policy_view.standard.0 }}` with
`{{ fact:traveller:alerts.traveller.policy_view.required_checks.0 }}` under {{ fact:traveller:alerts.traveller.policy_view.row }}; `needs_check` is also
permitted. The memo agrees with that action and check: the analyst would carry out its
`needs_check` recommendation as a hold with the required check (§4.3).

It gives travel, the benign explanation, and stolen card use the same likelihood. In its own
words:

> {{ bench:2026-10-cases:/case_memos/[file=traveller,slot=traveller]/arms/sol/memo/memo }}

The memo is advisory; the case's recorded action does not depend on it.

## Recorded action in the replay

The simulated analyst recorded the standard action,
**`{{ fact:traveller:alerts.traveller.decision.recorded_action }}`** at
{{ fact:traveller:alerts.traveller.decision.action_at }}, before shipment, and started the
identity check.

## What happened later

The following outcomes were unavailable at the decision.

The identity check returned **{{ fact:traveller:alerts.traveller.later.review.checks.0.outcome }}**
at {{ fact:traveller:alerts.traveller.later.review.checks.0.completed_at }}. Every required check
had passed, so FP-2 required `clear` (§5.3(a)). The analyst's final decision was
**{{ fact:traveller:alerts.traveller.later.review.final }}**. The order was released at
{{ fact:traveller:alerts.traveller.later.fate.released_at }}. Under the incumbent policy:

{{ table:fact:traveller:alerts.traveller.later.incumbent.events | known_at "Known", event "Event", detail "Detail", amount_cents "Amount" usd:2 }}

{{ table:fact:traveller:alerts.traveller.later.incumbent.cash | known_at "Known", kind "Cash event", amount_cents "Amount" usd:2 }}

Net for the platform: {{ fact:traveller:alerts.traveller.later.incumbent.net_cents | usd:2 }}.
The order is labelled `{{ fact:traveller:alerts.traveller.later.label.basis }}` (known
{{ fact:traveller:alerts.traveller.later.label.label_known_at }}) in the approve-all world,
the label source used to score every policy. It counts as legitimate in that evaluation.

**Had the order been approved (the approve-all world).** It ships without the wait, and the
installments are paid on a schedule starting from checkout. The net is the same,
{{ fact:traveller:alerts.traveller.later.approve_all.net_cents | usd:2 }}: the hold prevented
no loss. It delayed shipment and used analyst time.

## Tested change: an R03 exception for a long-used device shipping home

The change, motivation and expected mechanism were declared before testing. The change was
replayed once against the unchanged incumbent.

> **Change.** {{ fact:traveller:tested_change.change }}
>
> **Motivation.** {{ fact:traveller:tested_change.motivation }}
>
> **Expected mechanism.** {{ fact:traveller:tested_change.mechanism }}

**This case's order.** Under the change no rule holds, its score is
{{ fact:traveller:tested_change.result.case_order.variant.rule_score | num:0 }}, and it is
approved at checkout (`{{ fact:traveller:tested_change.result.case_order.variant.route }}`)
with no hold and no review. Its net is unchanged,
{{ fact:traveller:tested_change.result.case_order.variant.net_cents | usd:2 }}.

**The whole test window.**

| Measure | Incumbent | With the change | Difference |
|---|---:|---:|---:|
| Fraud loss prevented | {{ fact:traveller:tested_change.result.world.incumbent.cash.prevented_loss_cents | usd:2 }} | {{ fact:traveller:tested_change.result.world.variant.cash.prevented_loss_cents | usd:2 }} | {{ fact:traveller:tested_change.result.world.difference.cash.prevented_loss_cents | usd:2:signed }} |
| Legitimate orders held | {{ fact:traveller:tested_change.result.world.incumbent.adjudicated.legitimate_held }} | {{ fact:traveller:tested_change.result.world.variant.adjudicated.legitimate_held }} | {{ fact:traveller:tested_change.result.world.difference.adjudicated.legitimate_held | count:signed }} |
| Legitimate orders declined | {{ fact:traveller:tested_change.result.world.incumbent.adjudicated.legitimate_declined }} | {{ fact:traveller:tested_change.result.world.variant.adjudicated.legitimate_declined }} | {{ fact:traveller:tested_change.result.world.difference.adjudicated.legitimate_declined | count:signed }} |
| Legitimate orders cancelled after a hold | {{ fact:traveller:tested_change.result.world.incumbent.adjudicated.legitimate_cancelled }} | {{ fact:traveller:tested_change.result.world.variant.adjudicated.legitimate_cancelled }} | {{ fact:traveller:tested_change.result.world.difference.adjudicated.legitimate_cancelled | count:signed }} |
| Reviews | {{ fact:traveller:tested_change.result.world.incumbent.review.reviews }} | {{ fact:traveller:tested_change.result.world.variant.review.reviews }} | {{ fact:traveller:tested_change.result.world.difference.review.reviews | count:signed }} |
| Review minutes used | {{ fact:traveller:tested_change.result.world.incumbent.review.review_minutes_used }} | {{ fact:traveller:tested_change.result.world.variant.review.review_minutes_used }} | {{ fact:traveller:tested_change.result.world.difference.review.review_minutes_used | count:signed }} |
| Rule net contribution | {{ fact:traveller:tested_change.result.world.incumbent.cash.rule_net_cents | usd:2 }} | {{ fact:traveller:tested_change.result.world.variant.cash.rule_net_cents | usd:2 }} | {{ fact:traveller:tested_change.result.world.difference.cash.rule_net_cents | usd:2:signed }} |

Rule net contribution is the ledger net after subtracting
${{ config:policy:costs.false_decline_ltv_usd | num:0 }} for each legitimate order declined or
cancelled and the cost of
the analyst allotment, as in the
[recommendation rule](../docs/methods.md#recommendation-rule). Both policies have the same
allotment, so changes in minutes used do not change that cost. Holds are counted but not
priced. Per thousand of the {{ fact:traveller:tested_change.result.world.incumbent.orders }}
orders the replay decided, the change's difference is
{{ fact:traveller:tested_change.result.world.difference.rule_net_vs_incumbent_per_1000_orders_cents | usd:2:signed }}.

The result was negative. The exception reduced legitimate holds, declines and cancellations,
and used less review time. It also prevented less fraud loss and lowered rule net contribution.
Avoided holds benefit customers, but that benefit is not priced in the net, so the trade turns
on what a hold costs a good customer.

## Latent truth (diagnostic)

These simulation records are unavailable to rules, analysts and the memo drafter. The slot is
legitimate orders the generator drew with the travel trait; that defines which alerts are
eligible but does not rank them.

{{ facts:traveller:alerts.traveller.latent.order "The order" "Truth" | intent "Intent", mimic "Benign trait the generator drew" }}

{{ facts:traveller:alerts.traveller.latent.account "The account" "Truth" | actor "Account holder" }}
