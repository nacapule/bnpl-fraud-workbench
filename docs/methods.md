# Methods

How the synthetic world, the evaluation protocol, the cash ledger and the
simulated reviewer work, and what the results can and cannot say. Values in
this document are modelling assumptions unless a source is given; each is set
in one configuration file and stated once here.

## World assumptions

The world is a synthetic pay-in-4 marketplace generated from
`config/world.yaml`. Its tables, their layers (observable, latent, adjudicated)
and the chronology rules every world must pass are specified in
`core/world.py`.

### Time and knowledge

Every operational event has the time it happened (`occurred_at`) and the time
the platform could know it (`known_at`). Decisions see only what is known
before them, in the fixed order `(known_at, event kind, event id)`. Event and
entity ids are assigned after sorting by time, so they carry no information
about how an event was generated.

### Population and behaviour

*To be written with the generator: the parameter table with the reason for
each value, fixed before any model is fitted (legitimate new customers,
households, movers, device changes, travel, hardship, home geography, time of
day, fraud patterns and episodes, family shifts).*

### Truth and labels

Three layers are stored separately: latent truth (who the actor is, which
episode, what intent), observable outcomes (payments, failures, reversals,
disputes and their resolutions, victim reports, deliveries), and an adjudicated
label computed from observable outcomes only (`core.world.adjudicate`). The
label follows the fraud policy's determinations: an unauthorized-use dispute
lost by the platform, a victim report, a never-pay determination, two
item-not-received claims resolved against the customer, three or more linked
accounts on one first-purchase promotion, and orders left undelivered when a
merchant closed. Each determination becomes known at a stated time; an order
with no determination becomes a known negative 60 days after it, later if a
dispute is still open. Until then its label is unknown, and unknown is never
treated as negative. Latent truth is reported only as a separate diagnostic.

## Evaluation protocol

The windows, freeze dates, families, seeds, policies and metrics are
pre-registered in `experiments/protocol.yaml` and checked by
`core/protocol.py`.

| Window | Dates (end exclusive) | Use |
| --- | --- | --- |
| Warm-up | 2025-01-01 to 2025-04-01 | History only |
| Fit | 2025-04-01 to 2025-08-01 | Classifiers |
| Gap | 2025-08-01 to 2025-10-01 | Fit labels mature |
| Calibration | 2025-10-01 to 2025-11-01 | Calibrator; never used to fit classifiers |
| Gap | 2025-11-01 to 2026-01-01 | Calibration labels mature |
| Validation | 2026-01-01 to 2026-04-01 | Bands, thresholds, policy choice |
| Embargo | 2026-04-01 to 2026-06-01 | Validation labels mature |
| Test | 2026-06-01 to 2026-09-01 | Frozen policies evaluated |
| Follow-up | 2026-09-01 to 2026-12-30 | No new orders; outcomes mature |

The classifier freezes on 2025-10-01, the calibrator on 2026-01-01 and the
policy on 2026-06-01; each uses only labels known before its freeze. The ten
final seeds were drawn before any final world existed, and no final-seed world
can be generated before the freeze commit.

*To be completed at the freeze: tuning rule, capacity levels, recommendation
rule, friction guardrail, sensitivity values.*

## Ledger

All money is computed from one cash ledger in integer cents from the platform's
point of view (`core/ledger.py`). The product it models:

- **Down payment.** 25% of the customer's principal is collected at approval,
  rounded down to the cent.
- **Installments.** The rest is paid in three fortnightly installments, the
  first 14 days after approval. Leftover cents go to the earliest installments,
  one cent each, so a $100.01 order is paid as $25.00, $25.01, $25.00 and $25.00.
- **Merchant settlement.** The merchant is paid when it ships, net of a 5%
  merchant fee on its price (rounded half up). The fee is income only through
  this netting, never a separate cash event.
- **Promotions.** Discounts are funded by the platform: the customer owes the
  discounted price, and the platform pays the merchant the discount on top of the
  settlement.
- **Disputes.** When the platform learns of a dispute, the disputed payments are
  debited and a $15 fee is charged. If the dispute is resolved for the platform,
  the amount is credited back (the fee is not). If it is lost, the platform
  bears unauthorized-use disputes; the merchant bears item-not-received and
  not-as-described disputes and the platform recovers the amount from it, unless
  the merchant has closed.
- **Defaults.** A plan is written off 14 days after its last installment is due
  if a balance remains. Write-off is a status, not cash. One recovery of 20% of
  the written-off balance arrives 30 days later.
- **Reversals and refunds.** A collected payment can bounce (a reversal); a
  cancelled order's collected payments are refunded. An order cancelled after its
  down payment but before shipping therefore nets zero.

Loss over a horizon is the negative of the sum of cash events in it. A fully
repaid $100 order nets +$5.00: the merchant fee, not the $10 a fee-plus-margin
count would give.

## Reviewer

*To be written with the reviewer procedure: the evidence standard it applies
(`core/evidence.py`), the verification checks and their rates, and its
confusion matrix by pattern, decision point and evidence strength.*

## Limits

*To be written with the results: what a synthetic world can and cannot show,
the replay's stated assumptions (fixed attempted traffic, no attacker
adaptation, no churn), and the sensitivity of the conclusions.*
