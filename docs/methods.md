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

### How a world is generated

`python -m simulator.generate --seed S --family F` builds one world. Customers,
fraudsters and merchants act through one set of primitives
(`simulator/builder.py`): open an account, link a device or an address, add a
card, log in or change a credential, place an order, pay or miss an
installment, have a payment reversed, ship, deliver, dispute, report a
takeover. Each primitive checks the account's state at that moment (the
account exists, the device and address are linked to it, the card is on it,
the merchant is trading), so every event is generated from what existed when
it happened; the contract validator then checks the assembled world again.
Fraud patterns are parameter sets over the same primitives
(`simulator/fraud.py`), not separate generators appended to a benign world.

The world holds every attempted order with the outcomes it would have if the
platform approved everything. Only processor declines happen inside it;
policies act later, in the replay. Outcomes come from behaviour: a cardholder
disputes a charge, a takeover victim reports orders, a customer in hardship
stops paying. The generator never sets a label. The adjudicated label is
computed afterwards from the observable outcomes (below), so a fraud order can
end up unlabelled (a never-pay customer whose single default looks like credit
loss) and a legitimate order can end up labelled (a customer who disputes their
own order as unauthorized and wins).

Construction details carry no information about the pattern. Every actor takes
the hour of each action from one time-of-day profile (an evening peak), drawn
inside the interval the action must fall in (after the signup, before the
burst), never drawn freely and moved to the interval's edge. An IP address is
drawn from blocks that depend only on its country, and a customer's home
country decides the country of their home and mobile IPs; only trips abroad
(or a fraudster's own location) produce another country. Fraud accounts price,
connect and use promotions the way customers do: order amounts have the same
dispersion around the category median, 35% of their orders come over mobile
data, and their first orders take FIRST10 and later ones seasonal codes at the
customers' rates. The departures from that are behaviour and are stated in the
fraud table (resale merchants, larger tickets, card failures, desktops,
promotion farms' FIRST10, one session's IP for a takeover or a card-testing
run). Entity and event ids are assigned after the whole world is sorted by
time, with exact ties broken by random draws, so an id says when something
happened and nothing else.

Randomness comes from independent streams, one per component and actor
(`numpy.random.SeedSequence(seed, spawn_key=(component, index))`). A world is
therefore byte-identical when regenerated, and adding or removing one actor
never changes another's draws.

### Customers and their behaviour

Volumes are for one world at scale 1; `--scale` shrinks a world for tests and
CI. Every value is in `config/world.yaml`.

| Parameter | Value | Why |
| --- | --- | --- |
| Order attempts | about 160,000 over 20 months | The protocol's support check needs at least 30 test-window orders of every checkout pattern on every development seed with fraud held near 1%; 120,000 left some patterns under 40 per window, so the volume was raised rather than the fraud share. |
| Households | 255 per 1,000 orders, 45% opened between July 2021 and the start of the horizon | An established book with steady growth; the repeat-order rate is calibrated from the target volume less the 1% fraud share. |
| Household size | 1 (86%), 2 (10%), 3 (4%) accounts | Households share a home address, so rules counting accounts per address meet them. |
| Shared tablet | 30% of multi-account households | Households also share devices (R02's benign case). |
| First order of a new account | 65% within minutes of signup, 25% within days (mean 10), 10% never | Most pay-in-4 accounts are opened at a merchant's checkout, so legitimate new accounts order immediately, exactly as new fraud accounts do. |
| Large first order | 15% of new customers, 1.8 to 3 times the category median | Legitimate front-loaded first orders exist (R04's benign case). |
| Repeat orders | Poisson, gamma-distributed rates (shape 0.8), holiday peak 1.6 times, weekends 1.15 times | Most customers order rarely and a few often. |
| Order amounts | lognormal around the merchant category's median (sigma 0.65), $12 to $4,000, for every actor; patterns scale the median (fraud table) | One price process, so a residual price spread never marks an actor. |
| Customer lifetime | exponential, mean 900 days; 10% of older accounts dormant | Accounts stop shopping; dormant accounts exist. |
| New phone | 0.35 a year; the old phone stays linked up to 3 days | Phones are replaced about every three years. |
| Password reset after a new phone | 35% of new phones, within a day | A new phone often means a forgotten password (R01's benign case). |
| Laptop added | 30% of accounts, once | A second device. |
| Cards | added 0.12 and replaced 0.20 a year; 3% issued abroad | Cards expire and are added; a few customers hold foreign cards. |
| Credential and contact changes | password 0.04, email 0.01, phone 0.05 a year | Rare but present. |
| Moves | 0.07 a year per household; the old home link ends | Movers ship to a new address. |
| Trips abroad | 0.40 a year, 4 to 16 days; 35% to the neighbouring country (US and Canada) | Travellers order from a foreign IP (R03 and R11's benign cases). |
| Gifts | 5% of orders, half to another customer's home | Shipping away from home is ordinary. |
| Mobile data | 35% of actions | Mobile IPs vary but stay in the home country. |
| Logins | 35% of orders preceded by a login within 10 minutes (every actor), plus 4 a year | Account activity around orders. |
| Promotions | FIRST10 on 30% of first orders; seasonal codes on 12% of orders while they run | Platform-funded promotions are common. |
| AVS and CVV failures | 4.0% and 2.0% of legitimate orders | Stated in the fraud policy (FP-2 §6.5(a)). |
| Processor declines | 1.2% of legitimate attempts, half retried minutes later | Insufficient funds and card errors. |
| Late payers | 15% of customers; their installments fail first 30% of the time (others 3%), retried after 3 and 7 days | Late but recovered payments. |
| Defaults | 3.5% of a new customer's first plan, 1.2% of other plans, 4 times for 5% fragile customers; 70% (first plans) and 35% (others) zero-effort | Credit loss, including new customers who never pay; most first-plan defaults look exactly like never-pay. |
| After a default | 70% stop ordering | Defaulters rarely keep shopping. |
| Reversals | 0.3% of collected installments bounce 2 to 5 days later; 75% are repaid 2 days later | Bank returns. |
| Fulfilment | merchant median hours lognormal around 12 h (sigma 0.5, 2 to 72 h); each order lognormal around its merchant's median (sigma 0.6) | Goods ship within hours, which the replay's race between review and shipment depends on. |
| Delivery | lognormal, median 2.5 days | |
| Lost parcels | 0.5% of shipments never delivered, 70% claimed (upheld) | Genuine item-not-received claims. |
| Unconfirmed deliveries | 1.5% delivered without a carrier confirmation, no claim | Not every carrier confirms. |
| Other disputes | delivered-but-claimed 0.2% (80% rejected), not as described 0.4% (half upheld), customer disputing their own order as unauthorized 0.05% (half upheld) | Disputes on legitimate orders, some of which become fraud labels. |
| Disputes' timing | notified 1 to 4 days after filing, resolved 20 to 60 days later | |

### Fraud patterns

Each pattern's episodes start evenly over the order horizon: the horizon is cut
into as many equal slices as there are episodes, and each episode starts at a
time drawn from the time-of-day profile within its slice. Every evaluation window
therefore receives fresh episodes of every pattern, and episodes are many and
small rather than a few large ones. Episode counts are per 100,000 target
orders.

| Pattern | Episodes | What the actors do | Typical outcome |
| --- | --- | --- | --- |
| Account takeover (`P-ATO`) | 100 | A fraudster logs into an established customer's account (75% at least 90 days old, the rest 14 to 89 days, always with an earlier order) from a new device and one IP for the session, in the victim's country (60%) or abroad, usually resets or changes the password or the email (75%), enters a drop address at checkout (80%) and places 1 to 3 orders, mostly at resale merchants, with the stored card (a stolen card 15% of the time). | The owner reports the orders 1 to 21 days after the last one (65%), disputes each with the card issuer 7 to 40 days after it (20%, lost 85% of the time) or never notices (15%). Installments on the card are collected until the owner reports or disputes, and fail from then on. |
| Stolen card (`P-STOLEN`) | 95 | A new account (60% from a phone) with a stolen card, from an IP abroad 40% of the time; 35% first test 3 to 6 stolen cards in minutes from one IP (processor declines); 1 to 3 orders. A quarter are sleeper accounts opened 60 to 300 days earlier, kept warm with logins and sometimes a small repaid order, then used with a newly added stolen card. | The cardholder disputes the charge 5 to 40 days after it (75%, lost 90% of the time) or only has the card blocked 2 to 30 days after it; installments are collected until then and fail afterwards. |
| Synthetic ring (`P-SYNTH`) | 18 | 3 to 5 synthetic identities (born 1986 to 2002, thin files) opened over weeks; 45% share 1 or 2 devices, 30% share drop addresses and 25% share neither; 40% use spellings of one email address. Each repays one small warm-up order, then all place 1 or 2 large orders at resale merchants within 72 hours and never pay. | Never-pay through the shared device, address or email, or unlabelled credit loss when nothing links them. |
| Never-pay (`P-NEVERPAY`) | 75 | First-party: single orders (45%), bursts of 2 to 4 plans within 6 days (30%) and linked groups of 2 to 4 accounts sharing a device, an address or an email (25%). Nothing is paid after checkout. | Bursts and groups meet the never-pay marker; single orders read as credit loss. |
| Promotion farm (`P-PROMO`) | 40 | 3 to 6 new accounts over 3 weeks, each using FIRST10 once; 50% share devices, 25% email spellings, 25% only an address. 70% repay. | Promotion abuse when linked by device or email; address-only farms stay unlabelled, like households. |
| Item-not-received abuse (`P-INR-ABUSE`) | 16 | An account opened weeks earlier orders 3 to 6 times, pays, and claims 1 to 3 delivered orders never arrived; 85% of claims are rejected. | Abuse once two rejected claims on delivered orders are known. |
| Merchant bust-out (`P-MERCH`) | 5 | A resale merchant onboards, ramps up sales for 40 to 75 days with new customers acquired at its checkout (0.2 a day rising to 0.8) and rising tickets (to 1.8 times), stops delivering 8 to 14 days before it disappears, and still reports shipments. Its customers claim non-delivery after the closure (75%). | Bust-out on the undelivered orders, with the claims dated after the closure; the merchants' ids are in the manifest for evaluation only. |

The generator reports how each pattern's orders end up labelled
(`simulator.support.labelled_share`). Patterns a checkout decision can act on
are listed in `config/world.yaml` (`support.checkout_patterns`); item-not-received
abuse shows only when claims arrive, and merchant bust-out is the merchant's
fraud, so neither is a checkout target.

### Families

The three families share every event known before the test window
(2025-06-01) for a seed: families add actors from their own random streams and
never change the base world's. The acquisition surge adds new households from
the test window at the base arrival rate again (twice the inflow), and these
campaign customers use FIRST10 on 80% of first orders. The fraud-mix shift adds
as many account takeovers again from the test window and activates 25 aged
sleeper accounts with stolen cards; those accounts are opened before the test
window in every family, log in now and then to the end of the horizon, and are
used only in this family.

### Truth and labels

Three layers are stored separately: latent truth (who the actor is, which
episode, what intent), observable outcomes (payments, failures, reversals,
disputes and their resolutions, victim reports, deliveries), and an adjudicated
label computed from observable outcomes only (`core.world.adjudicate`). The
label follows the fraud policy's determinations: an unauthorized-use dispute
lost by the platform, a victim report, never-pay, two item-not-received claims
resolved against the customer on orders the carrier confirmed delivering,
promotion abuse and merchant bust-out. Never-pay needs a zero-effort default
(nothing paid after the checkout payment and the first installment still unpaid
30 days after its due date) plus a mark of intent: a second such plan on the
account within 7 days, or two or more other accounts sharing its device,
shipping address or email that defaulted the same way within 30 days. A
zero-effort default without that mark is a credit loss, not fraud. Promotion
abuse is a use of a first-purchase promotion when two or more other accounts
sharing a device or email with its account (not an address, which households
share) used the same promotion within 90 days of it, and none of them ordered
without a promotion within 90 days of its own use. Bust-out is an
item-not-received claim upheld on an order the merchant reported shipped and
the carrier never confirmed delivering, after the merchant stopped trading.
Each determination becomes known at a stated time; an order with no
determination becomes a known negative 60 days after it, later if a dispute is
still open. Until then its label is unknown, and unknown is never treated as
negative. Latent truth is reported only as a separate diagnostic.

## Evaluation protocol

The windows, freeze dates, families, seeds, policies and metrics are
pre-registered in `experiments/protocol.yaml` and checked by
`core/protocol.py`.

| Window | Dates (end exclusive) | Use |
| --- | --- | --- |
| Warm-up | 2024-01-01 to 2024-04-01 | History only |
| Fit | 2024-04-01 to 2024-08-01 | Classifiers |
| Gap | 2024-08-01 to 2024-10-01 | Fit labels mature |
| Calibration | 2024-10-01 to 2024-11-01 | Calibrator; never used to fit classifiers |
| Gap | 2024-11-01 to 2025-01-01 | Calibration labels mature |
| Validation | 2025-01-01 to 2025-04-01 | Bands, thresholds, policy choice |
| Embargo | 2025-04-01 to 2025-06-01 | Validation labels mature |
| Test | 2025-06-01 to 2025-09-01 | Frozen policies evaluated |
| Follow-up | 2025-09-01 to 2025-12-30 | No new orders; outcomes mature |

The classifier freezes on 2024-10-01, the calibrator on 2025-01-01 and the
policy on 2025-06-01; each uses only labels known before its freeze. The ten
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
