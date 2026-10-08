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
run). A device or address several accounts share is first seen when the first
of them uses it. Entity and event ids are assigned after the whole world is sorted by
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
| Account holders' age | born 19 to 77 calendar years before signup (every actor) | Pay-in-4 accounts are for adults. |
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
| Defaults | 3.5% of a new customer's first plan (placed within 30 days of signup), 1.2% of other plans, 4 times for 5% fragile customers; 70% (first plans) and 35% (others) zero-effort | Credit loss, including new customers who never pay; most first-plan defaults look exactly like never-pay. |
| After a default | 70% stop ordering | Defaulters rarely keep shopping. |
| Reversals | 0.3% of collected installments bounce 2 to 5 days later; 75% are repaid 2 days later | Bank returns. |
| Fulfilment | merchant median hours lognormal around 12 h (sigma 0.5, 2 to 72 h); each order lognormal around its merchant's median (sigma 0.6) | Goods ship within hours, which the replay's race between review and shipment depends on. |
| Merchant risk tier | at onboarding, tiers 1/2/3 for 65/28/7% of merchants, 30/45/25% in the resale categories (electronics, jewelry, gaming); 10% of merchants join during the horizon | One onboarding rule for every merchant, including those that later bust out. |
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
| Account takeover (`P-ATO`) | 100 | A fraudster logs into an established customer's account (75% at least 90 days old, the rest 14 to 89 days, always with an earlier order) from a new device and one IP for the session, in the victim's country (60%) or abroad, usually resets or changes the password or the email (75%), enters a drop address at checkout (80%) and places 1 to 3 orders, mostly at resale merchants, with the stored card (a stolen card 15% of the time). | The owner reports the orders 1 to 21 days after the last one (65%), disputes each with the card issuer 7 to 40 days after it (20%, lost 85% of the time) or never notices (15%). Installments on the card are collected until the owner reports or first disputes, and fail from then on. |
| Stolen card (`P-STOLEN`) | 95 | A new account (60% from a phone) with a stolen card, from an IP abroad 40% of the time; 35% first test 3 to 6 stolen cards in minutes from one IP (processor declines); 1 to 3 orders. A quarter are sleeper accounts opened 60 to 300 days earlier, kept warm with logins and sometimes a small repaid order, then used with a newly added stolen card. | The cardholder has the card blocked 2 to 30 days after its first misuse, or at the first dispute if sooner; installments on it are collected until then and fail afterwards, and a later checkout with it is declined. Each charge is disputed 5 to 40 days after it (75%, lost 90% of the time). |
| Synthetic ring (`P-SYNTH`) | 18 | 3 to 5 synthetic identities (young adults born 22 to 38 calendar years before signup, thin files) opened over weeks; 45% share 1 or 2 devices, 30% share drop addresses and 25% share neither; 40% use spellings of one email address. Each repays one small warm-up order, then all place 1 or 2 large orders at resale merchants within 72 hours and never pay. | Never-pay through the shared device, address or email, or unlabelled credit loss when nothing links them. |
| Never-pay (`P-NEVERPAY`) | 75 | First-party: single orders (45%), bursts of 2 to 4 plans within 6 days (30%) and linked groups of 2 to 4 accounts sharing a device, an address or an email (25%). Nothing is paid after checkout. | Bursts and groups meet the never-pay marker; single orders read as credit loss. |
| Promotion farm (`P-PROMO`) | 40 | 3 to 6 new accounts over 3 weeks, each using FIRST10 once; 50% share devices, 25% email spellings, 25% only an address. 70% repay. | Promotion abuse when linked by device or email; address-only farms earn no promotion-abuse label, like households, though members who never pay can meet the never-pay marker through the shared address. |
| Item-not-received abuse (`P-INR-ABUSE`) | 16 | An account opened weeks earlier orders 3 to 6 times, pays, and claims 1 to 3 delivered orders never arrived; 85% of claims are rejected. | Abuse once two rejected claims on delivered orders are known. |
| Merchant bust-out (`P-MERCH`) | 5 | A resale merchant onboards, ramps up sales for 40 to 75 days with new customers acquired at its checkout (0.2 a day rising to 0.8) and rising tickets (to 1.8 times), stops delivering 8 to 14 days before it disappears, and still reports shipments. Its customers claim non-delivery after the closure (75%). | Bust-out on the undelivered orders, with the claims dated after the closure; the merchants' ids are in the manifest for evaluation only. |

The generator reports how each pattern's orders end up labelled
(`simulator.support.labelled_share`). Patterns a checkout decision can act on
are listed in `config/world.yaml` (`support.checkout_patterns`); item-not-received
abuse shows only when claims arrive, and merchant bust-out is the merchant's
fraud, so neither is a checkout target.

### Families

The families share every event known before the test window
(2025-06-01) for a seed: they act only from the test window on, adding actors
from their own random streams or rescaling later orders' shipments (below), and
never change the base world's earlier history. The acquisition surge adds new households from
the test window at the base arrival rate again (twice the inflow), and these
campaign customers use FIRST10 on 80% of first orders. The fraud-mix shift adds
as many account takeovers again from the test window and activates 25 aged
sleeper accounts with stolen cards; those accounts are opened before the test
window in every family, log in now and then to the end of the horizon, and are
used only in this family.

Two further families are a sensitivity check on the race between review and
shipment, not a claim about how merchants behave: `lag_half` and `lag_double`
multiply each order's drawn time to shipment by 0.5 or 2 for orders placed from
the test window on (before the 15-minute floor), with the same random draws as
the baseline. Everything anchored on the shipment or the delivery moves with it
(delivery, settlement and promotion funding, claims filed after delivery, a
closing merchant's shipments, which are all reported before it closes, and the
labels that follow). What is anchored on the checkout stays as in the baseline: the
installment schedule and collections, card blocks, takeover reports and
unauthorized disputes, claims for parcels that never arrive (10 to 25 days after
the order) and bust-out claims (after the closure).

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

The windows, freeze dates, families, seeds, policies, tuning rule, capacity
levels, recommendation rule and sensitivities are pre-registered in
`experiments/protocol.yaml` and checked by `core/protocol.py`.

| Window | Dates (end exclusive) | Use |
| --- | --- | --- |
| Warm-up | 2024-01-01 to 2024-04-01 | History only |
| Fit | 2024-04-01 to 2024-08-01 | Classifiers; the capacity base |
| Gap | 2024-08-01 to 2024-10-01 | Fit labels mature |
| Calibration | 2024-10-01 to 2024-11-01 | Calibrator; never used to fit classifiers |
| Gap | 2024-11-01 to 2025-01-01 | Calibration labels mature |
| Validation | 2025-01-01 to 2025-04-01 | Each policy's thresholds |
| Embargo | 2025-04-01 to 2025-06-01 | Validation labels mature |
| Test | 2025-06-01 to 2025-09-01 | Frozen policies evaluated |
| Follow-up | 2025-09-01 to 2025-12-30 | No new orders; outcomes mature |

The classifier freezes on 2024-10-01, the calibrator on 2025-01-01 and the
policy on 2025-06-01. Each uses only labels known before its freeze. The ten
final seeds were drawn before any final world existed; final-seed worlds
cannot be generated before the freeze commit.

### Policies

Seven policies are compared (`queue_sim/policies.py`). Each routes an order
once at checkout, using that order's as-of context row and thresholds fixed
before replay: decline if its decline score reaches the decline threshold;
otherwise review if its review score reaches the review threshold; otherwise
approve. Routing does not rank orders within a day or compare them with other
orders.

| Policy | Review score | Decline score |
| --- | --- | --- |
| `approve_all` (reference) | none | none |
| `incumbent_rules` | rule score: the sum of the weights of the FP-2 §6.2 rules that hold (`rules/definitions.py`) | the same |
| `tree_depth3` | depth-3 decision tree on four features named before fitting: account age, the device's age on the account, accounts on the device in 30 days, amount against the category median | the same |
| `logistic` | logistic regression on the context's model features | the same |
| `boosting` | histogram gradient boosting on the same features | the same |
| `hybrid` | boosting score | rule score |
| `expected_loss` | calibrated probability (boosting) times the order's cash at risk if approved | the expected loss avoided minus the expected cost of declining a legitimate order (merchant fee plus the $15 LTV proxy); declines at zero or above |

Each seed's classifiers are fitted on its baseline world's fit-window orders
with balanced class weights. Only labels known before the classifier freeze
are used; unknown labels are omitted, never counted as negatives. One isotonic
calibrator per score, including the rule score, is fitted on the calibration
month using labels known before the calibrator freeze. Only the expected-loss
policy uses calibrated probabilities for decisions.

### The replay

The replay (`queue_sim/replay.py`) processes the window's processor-approved
checkouts in time order. Each policy starts acting at the window's start;
earlier orders retain their approve-all history. Reviewed orders enter one
queue, ordered by FP-2 §7.1 priority, then by review score, highest first,
then by earlier checkout and lower order id. The simulated reviewer decides at
review completion and whenever a verification check answers. Actions follow
FP-2 §4 and `core/actions.py`:

- An order proceeds while waiting for review. The merchant ships at the
  world's time unless a hold or decline comes first.
- A checkout decline means the order is never created: no plan, shipment,
  payments or later outcomes, and no cash. The attempt stays logged, and no
  account is blocked.
- A pre-shipment hold pauses shipment and merchant settlement for up to 48
  hours. When the checks clear the order, it is released: shipment and
  everything after it move by the pause, and installments not yet due start
  from the release. An order the checks have not decided within 48 hours is
  cancelled and its checkout payment refunded, without blocking the account.
  After shipment a hold pauses nothing, and one that runs out changes nothing.
- A pre-shipment decline voids the order, refunds the checkout payment and
  prevents later events. After shipment the loss stands. Both block the
  account, whose later orders are declined at checkout.
- An escalation is a decline that also blocks accounts linked at the decision
  and adds 20 minutes of senior review to the queue. Under FP-2 §2.5, linked
  accounts are other accounts with an order attempt or account event on the
  device, or an order attempt to the shipping address, in the preceding 30 days.

The policy's cash comes from the same ledger as the world's (below). An order
the policy left alone, or declined after shipment, keeps the world's cash
events; an order voided before shipment (declined, escalated or cancelled)
keeps the cash that had moved and gets refunds of the payments collected; a
released order's cash moves with its events. Outcomes are counted for the
window's orders with the cash and labels known before 2025-12-30, the end of
observation. Reviews still waiting then remain undecided.

The labels are the approve-all world's, the same for every policy: an order a
policy declined is counted as fraud or legitimate by the label its outcomes
would have earned had it been approved, and no action re-adjudicates any
order. The policy's actions change the realized history and the cash, not the
labels.

**History under the policy.** Attempt-derived columns (velocity, linkage,
tenure, credential changes) are shared across policies because declined
attempts remain logged. Outcome-derived columns cover approvals, installments
due and paid, disputes, promotion redemptions and blocks. At each replay day's
start, they are rebuilt from the policy's realized events
(`core.asof.policy_rows`) for every account it has declined, held, voided or
blocked, and every account that has shared a device, address or email with one.
Other accounts retain approve-all rows. So a decision sees the policy's own
history as of the start of its day, never the outcomes of an order the policy
prevented on an earlier day. Decisions made during the day enter these columns
only at the next day's rebuild: until then, a later order of an account the
policy first acted on that day sees the earlier order as approve-all would
(for example approved, with an open balance, after the policy declined it at
checkout that morning). A block takes effect at once: the account's later
orders that day are declined at checkout. The frozen-history diagnostic keeps
all outcome-derived columns at approve-all values to measure the effect of
policy-specific history.

Review times, check outcomes and answer delays are keyed to the world's seed,
a named stream and the order id (`queue_sim/draws.py`). Policies reviewing the
same order therefore receive the same draws.

### Threshold tuning

Each seed's thresholds are tuned once on its baseline world and held fixed
across families, staffing levels and replay variants (`tuning.rule` in the
protocol; `rules/tuning.py`). Tuning replays the validation window at the base
allotment on the current shift layout, using the world as known at the policy
freeze.
Replay stops at the end of 2025-05-31; only labels and cash known before
2025-06-01 count.

**Grid.** Thresholds are scores attained on validation orders at checkout. For
each review rate (0, 0.5, 1, 2, 3, 5, 8 and 12%) and decline rate (0, 0.1,
0.25, 0.5, 1 and 2%), the cut-point is the lowest attained score whose share
of orders at or above it does not exceed the rate. If the share at the highest
score still exceeds the rate, that score is the cut-point. Ties, common for
rule scores, can leave the reached share well below its rate or give several
rates the same cut-point. Rate 0 disables the route, so "no review" and "no
decline" are always in the grid. The expected-loss policy tunes only its review
threshold; its decline rule stays fixed.

**Objective and feasibility.** The objective is ledger net contribution minus
friction cost: the $15 LTV proxy for each legitimate order declined at checkout,
refused because its account was blocked, declined or escalated after review,
or cancelled after an unanswered hold. "Legitimate" here means an adjudicated
label known by the cut that records no fraud (no finding, or credit loss).
Unknown labels count as neither fraud nor legitimate. A point is feasible
when the review minutes it offers (the review time of every order that reached
the queue, plus each escalation's senior minutes) fit in the allotment over the
window, both rounded to whole minutes. Tuning has no separate cap on
legitimate friction; the recommendation rule supplies it.

**Procedure.**

1. Replay every grid point with frozen approve-all history.
2. Shortlist the screen's winner, points within one step of it on each
   threshold (including diagonal neighbours), and the screen's five best
   feasible points.
3. Replay the shortlist with policy-specific history. Choose the feasible
   point with the highest objective; ties go to fewer reviews, then fewer
   checkout declines among labelled orders (for fraud orders, refusals of
   blocked accounts count too).

The chosen point is the shortlist's best feasible point in the searched grid.
It is flagged if either threshold is the cut-point for the highest rate
searched, since a wider grid might do better. Every replayed point is kept
with its history, and the frontier is published even when flat. A point with
review switched off offers no minutes and is feasible in the screen, but it
need not enter the shortlist; if no shortlisted point is feasible under
policy-specific history, the policy is reported as not evaluated on that seed,
with no fallback. The shortlist stands in for replaying the whole grid with
policy-specific history. The protocol adopted it because, on the three
development worlds, it chose the same threshold pair as that full replay for
every tuned policy on every world (18 of 18).

### Capacity

Capacity is analyst time allotted to the fraud queue, with two parameters
(`config/policy.yaml` `capacity` and `roster`): coverage hours, set by the
shift layout, decide whether a review can finish before shipment; review
minutes per shift limit the work. Each shift has one analyst who takes the next
queued order whenever on shift with allotment left. Reviews and escalations'
senior work use the same allotment; senior work goes first, in escalation
order. Work that exhausts the allotment or reaches shift end resumes on that
analyst's next shift. Analysts keep working the window's queue after the
window ends, through the end of observation, with each later shift's
allotment. Feasibility compares the work offered with the allotment inside the
window, and the recommendation rule charges only that allotment; review
minutes used include the work done after the window.

| Layout | Early shift | Late shift |
| --- | --- | --- |
| Current | Monday to Friday from 08:00 | Wednesday to Sunday from 11:00 |
| Evening | Monday to Friday from 12:00 | Wednesday to Sunday from 15:00 |

Each shift has 6.5 productive hours. The evening layout moves the same shifts
later to cover the evening arrival peak.

The base allotment was set from today's queue before any policy comparison.
Today's rules at today's bands (review at a rule score of 30, decline at 90)
were replayed over the fit windows of development worlds 416, 1041 and 2718 with
policy-specific history and staffing far above demand. They offered 4,481, 4,452
and 4,846 review minutes including senior work (mean 4,593.1). Staffing so that
this queue uses 80% of the allotment gives 4,593.1 / 0.8 / 174 fit-window shift
instances = 33.0, or 33 minutes per shift. A productive shift is almost 12 times
that, so capacity levels change the allotment rather than whole analysts.

| Level | Review minutes per shift | Times the base |
| --- | --- | --- |
| Low | 17 | 0.515 |
| Base | 33 | 1 |
| High | 50 | 1.515 |

Low and high are the fewest whole minutes per shift that give at least half and
1.5 times the base's minutes. Low binds on every development world: 17 minutes
on 174 shifts give 2,958 fit-window minutes, below each world's offered 4,452 to
4,846. Families share the allotment, so acquisition surge brings more orders to
the same capacity. The evening layout has 33 minutes on each of the same shift
instances, giving the same total. Base-tuned thresholds stay fixed at low, high
and evening. Realized review minutes per 1,000 orders are reported by family.

**Priority and service targets.** Priority is fixed at queue entry (FP-2 §7.1):
P0 when R05 or R07 holds, or the merchant's stated median time to shipment is
under 2 hours (excluded by the generator's 2-hour floor); P1 when the amount is
at least $500 or R02, R08 or R10 holds; P2 otherwise. All entries occur at
checkout, so P3 (already shipped or cancelled at entry) does not occur. Targets
are 1, 4 and 8 service hours for P0 to P2 (24 for P3), from entry to the
analyst's first decision (a hold counts as one). The service calendar is fixed
at 08:00 to 20:00 every day, regardless of roster. An order undecided at
observation end misses its target.

### Seeds, pooling and intervals

Each final seed supplies one world per family. Shared pre-test events allow
one fit and tuning per seed. Seed 416 is also fitted and tuned for the case
files and SQL library, but is outside the evaluated worlds. Policies use the
same worlds and draws, so comparisons are paired by seed.

For each family, replay cell and policy, rates pool total numerators over
total denominators (for example legitimate orders held per 10,000 legitimate
orders); money is the mean over seeds. Per-seed values are retained as the
spread. Paired differences against approve-all and the incumbent are reported
with their mean, minimum, maximum and sign count ("positive on 9 of 10 seeds").
Rate differences are recorded in basis points and printed as percentage
points. When a policy has no feasible point on some seed, its pooled metrics
and its paired comparisons in that family carry no value and name the seeds
(pooled counts are still shown). The recommendation rule instead judges each
policy on the seeds where it was evaluated, and its hurdle on the seeds it
shares with the incumbent.

Replay outcomes have no within-world confidence intervals. Seed variation
shows differences between simulated worlds sharing one generator and its
parameters, not parameter uncertainty. Directional sentences in the README
and operating review are generated from their supporting results. Rendering
requires an exact two-sided sign test over seeds, excluding zeros, at or below
the claim's level (0.05 unless it states another) in the stated direction; at
0.05, ten seeds without zeros need at least 9 of 10. No test is published for
a comparison with a policy the recommendation rule selected in that operating
cell, as either side, since it was chosen among six challengers. The LLM
evaluations use within-world intervals instead (Wilson intervals, and cluster
bootstrap intervals that resample linked cases together); the archived study
the pipeline replays reports exact McNemar tests on paired cases, and the new
benchmark an exact sign test over clusters of linked cases
([LLM appendix](../reports/appendix-llm.md)).

### Recommendation rule

The rule (`reporting.recommendation_rule`; `core/recommendation.py`) was set
before any final world existed. It uses each policy's final-seed outcome rows
in an operating cell: a family plus a replay cell. The primary cell is the
baseline family at the base allotment on the current layout, with
policy-specific history, the evidence-based reviewer and the standard
verification rates.

The measures, per seed, are:

- **Rule net contribution:** ledger net for the window's orders, less friction
  cost (the LTV proxy for each lost legitimate customer) and the analyst
  allotment at $35 per allotted hour, including unused minutes. A policy with
  no review route at its chosen point releases the allotment and is charged
  nothing, so approve-all can show whether screening pays at all.
- **Lost legitimate customers:** legitimate orders declined at checkout,
  refused because the account was blocked, declined or escalated after review,
  or cancelled after an unanswered hold, per 10,000 legitimate orders.
- **Legitimate orders held:** legitimate orders asked to verify, per 10,000
  legitimate orders.

"Legitimate" uses the adjudicated label at observation end (no finding, or
credit loss). Each order counts once per measure. Simulation-truth counts
(no fraud pattern, or a customer of a fraudulent merchant) appear alongside
as a diagnostic.

**Eligibility.** In the cell, a challenger must meet these limits:

- lost legitimate customers: at most 100 per 10,000 on the mean over seeds
  and 200 on every seed;
- legitimate orders held: at most 300 per 10,000 on the mean and 600 on every
  seed;
- service: at every priority, the mean of the share decided within target,
  over the seeds with entries at that priority, must be at least 90%.
  Priorities with fewer than 50 entries pooled over seeds are reported without
  assessment; approve-all has no queue.

The friction caps form the guardrail: risk-appetite assumptions, not
benchmarks. If the incumbent fails a criterion in a cell, that criterion
becomes "no worse than the incumbent" for challengers there. If it fails a
criterion in the primary cell, the operating review opens by stating that
today's rules miss the target.

**Hurdle.** A challenger's rule net contribution minus the incumbent's, per
1,000 orders and paired by seed, must average at least $100 and be strictly
positive (zero is not) on at least 9 of 10 seeds (8 of 9, 8 of 8, every seed
with fewer). The $100 allows for the cost of changing a policy. "Orders" are
the orders a policy decides: processor-approved checkouts in the test window.
Processor-declined attempts are left out; the generator declines 1.2% of
legitimate attempts, plus card-testing attempts, a little over 1% of all
attempts. The sign bar is a consistency rule across simulated worlds, not a
significance test.

**Choice.** Among eligible challengers that clear the hurdle, the one with the
highest mean improvement is recommended. Exact ties go to fewer lost
legitimate customers, then fewer review minutes used, then the simpler policy,
in the order approve-all, incumbent rules, tree, logistic, boosting, hybrid,
expected loss. Otherwise the incumbent stays, and the best eligible
challenger's mean, range and sign count are printed (the best of all when none
is eligible). Approve-all is a challenger too. A cell where the incumbent has no
evaluated row is not assessed.

**Flip table.** The whole rule (eligibility, hurdle, choice) is applied again
in each cell that differs from the primary cell in one respect:

- acquisition surge, fraud-mix shift, fulfilment lag ×0.5 and ×2, each at base
  allotment on the current layout;
- low and high allotments;
- evening layout at base allotment;
- weak verification;
- LTV proxies of $5 and $45.

For the recommendation rule and flip table, sensitivities are never combined.
For each cell the flip table states whether the primary outcome holds and, if
not, why: a different policy cleared the bar, the primary winner fails an
eligibility criterion there, or no challenger cleared the bar. The verdict reads
"holds in N of M cells", counting only the assessed cells besides the primary
one.

### Sensitivities and diagnostics

No sensitivity re-tunes. Thresholds, models and queue rules remain those
chosen for each seed in the primary cell. The broader replay tables include
combinations of traffic or fraud-mix changes with staffing and replay variants.

| Sensitivity | Values | How it is run |
| --- | --- | --- |
| Allotment | 17 and 50 minutes per shift | replays at those levels |
| Shift layout | evening layout, 33 minutes per shift | replay on that layout |
| Verification | `reviewer.verification_weak` (Reviewer, below) | replay variant at the base allotment |
| LTV proxy | $5 and $45 | friction cost recomputed from the same base replays, for all seven policies; the expected-loss policy keeps $15 inside its decline rule |
| Fulfilment lag | ×0.5 and ×2 (`lag_half`, `lag_double`) | world families generated for the final seeds: from the test window, each order's drawn time to shipment is multiplied before the 15-minute floor; replayed at the base allotment, current layout, standard variant only |
| Traffic and fraud mix | `acquisition_surge`, `fraud_mix_shift` | world families from the test window, replayed in every cell |

Frozen approve-all history and the perfect reviewer (below) are diagnostics
replayed at base allotment on the current layout. They appear beside the tables
and never enter the recommendation rule or flip table.

### Case selection

The five case files illustrate the replay on development world 416 (baseline
family). The
protocol's `cases` rule was fixed before any final world existed. Candidates
are the incumbent's test-window alerts in the primary-cell replay, ordered by
a SHA-256 hash of order id. Each subject (takeover, never-pay, hardship,
traveller, card testing, ring) is a slot defined by latent pattern or benign
trait. Never-pay and hardship share a file. Each slot takes its first
candidate, preferring analyst-reviewed orders for the three full cases.
Labels, cash, later outcomes and reviewer decisions never rank or filter
candidates. Empty slots are reported, never filled from another seed, window,
family or policy.

### The freeze

The freeze commit adds `experiments/FREEZE.json` with the SHA-256 of every file
that determines the numbers (`freeze.files` in the protocol): the protocol,
`config/world.yaml`, `config/policy.yaml`, the fraud policy, the code for
world generation, context building, fitting, routing, review, replay,
accounting, evaluation and choice, the LLM referee, the archived LLM study the
pipeline replays, the freeze check, every module these import except
`report/` (which writes sentences and figures from the results), and the
dependency lock.

The final run refuses to generate a final-seed world without a complete
protocol, a clean working tree, every listed file committed and matching its
hash, no file added under a frozen directory (ignored files and caches aside),
and the installed packages and Python version matching the lock. The marker is
written once. A bug fixed after the freeze gets a commit of its own that runs
`core.protocol.log_fix`: it records the changed files' new hashes and appends to
the marker's `fixes` the files, what was wrong and the before/after effect on
the results. This document and the report templates are not frozen; changes to
them after the freeze commit are in their history.

## Ledger

All money is computed from one cash ledger in integer cents from the platform's
point of view (`core/ledger.py`). The product it models:

- **Down payment.** 25% of the customer's principal is collected at approval,
  rounded down to the cent.
- **Installments.** The rest is paid in three fortnightly installments, the
  first 14 days after approval. Leftover cents go to the earliest installments,
  one cent each, so a $100.01 order is paid as $25.00, $25.01, $25.00 and $25.00.
- **Merchant settlement.** The merchant is paid when it ships: its price net of
  a 5% merchant fee on the price (rounded half up) and of any promotion
  discount. The fee is income only through this netting, never a separate cash
  event.
- **Promotions.** Discounts are funded by the platform: the customer owes the
  discounted price, and the platform pays the merchant the discount as a
  separate promotion-funding event, so the merchant receives its price less the
  fee.
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

The simulated analyst (`queue_sim/reviewer.py`) follows a fixed procedure
through `core/evidence.py`, shared with the memo referee. Both use the same
written fraud policy. The analyst reads evidence and takes the standard
disposition; simulation truth supplies verification answers, never the action.

### What it reads

Each evaluation reads the order's as-of context row and completed checks.
Checkout columns (the attempt, velocity, tenure, link ages, credential
changes) stay as at checkout. Decision-time columns (linkage, current email,
earlier outcomes that settle an order, repayment and shipment status, blocks)
are as known at the evaluation, under the replayed policy's decisions. The
reviewer never reads labels, latent tables or policy scores (FP-2 §6.5(d)).

| Moment | Evidence |
| --- | --- |
| Checkout | the policy routes the order; the reviewer is not involved |
| End of the review (first decision) | on the checkout day, the checkout row; on a later day, the row as known at 00:00 that day |
| A check answers | the row as for a first decision on that day, with every check completed so far |
| The hold's 48 hours end | no evaluation: an order not yet shipped is cancelled, a shipped one is unchanged |

Evidence can be up to one replay day old, never from the future; the
policy's own decisions earlier the same day are not in it yet (The replay).

### Procedure

Each evaluation has two steps:

1. **Classify** (`evidence.classify`): identify the FP-2 §6.2 rule conditions
   that hold, §6.4 household exceptions to R02 or R08, §6.3 earlier outcomes
   that settle the order, and completed checks. A family is present if a
   condition holds without an exception. Evidence strength counts the adverse
   families present: Account access, Card, Velocity and Linkage. Context never
   counts.
2. **Decide** (`evidence.permitted_actions`, the §6.6 table): take the first
   applicable action:
   - an earlier outcome settles the order (§6.3): decline, or escalate if
     Linkage is present, regardless of check results;
   - a check failed: decline, or escalate if Linkage is present; after shipment
     the loss stands, but blocks apply;
   - no adverse family is present: clear;
   - every required check passed: clear;
   - otherwise: hold and start required checks that have not run.

Under §5.2, Account access requires `contact`; Card, Velocity or Linkage
requires `id_check`. Both run when required, each at most once per order.
The procedure always takes the standard disposition. It does not take the
decline also permitted by §6.6(b) with two or more families: only a failed
check or settling outcome leads to decline.

### Checks and their rates

The latent pattern determines actor class (`reviewer.actor_class` in
`config/policy.yaml`); orders without a pattern are legitimate. Checks
establish who is ordering, never intent (FP-2 §5.1), so first-party fraud
passes at customer rates.

| Class | Patterns | `contact` passed / failed | `id_check` passed / failed |
| --- | --- | --- | --- |
| Legitimate | none; merchant bust-out (the customer is genuine) | 0.85 / 0 | 0.85 / 0.02 |
| First party | never-pay, item-not-received abuse, promotion farm | 0.85 / 0 | 0.85 / 0.02 |
| Takeover | account takeover: the holder answers and disowns the order | 0.25 / 0.60 | 0.05 / 0.60 |
| Third party | stolen card (holds the account, not the card); synthetic identity | 0.85 / 0 | 0.05 / 0.60 |

The remaining probability is no response. Weak verification
(`reviewer.verification_weak`) assumes better-prepared attackers: takeover
`contact` passes at 0.50 and fails at 0.35; takeover and third-party
`id_check` passes at 0.25 and fails at 0.45. Other classes are unchanged.

Outcomes are drawn per order and check, keyed by order id, independently of
evidence, time and other orders, so two orders by one fraudster are drawn
separately. Answer
delays use the same keying and are lognormal, with median 6 hours and sigma
1.0. Only answers arriving before the hold's 48 hours end are delivered.
These rates are simulation assumptions (FP-2 §11).

### Timing and capacity

Review time is drawn per order from a lognormal distribution with arithmetic
mean 7 minutes and sigma 0.6. Analysts use the shifts, allotments and senior
work priority described under Capacity. Re-evaluation when a check answers
takes no analyst time.

### Upper bound

The perfect reviewer uses simulation truth at review completion, without
checks: it declines every order generated as fraud or abuse and clears the
rest. It
uses the same queue, timing and allotment and is reported as a labelled upper
bound diagnostic. Its declines include undelivered merchant bust-out orders
whose customers are genuine. Both friction definitions are the same in every
replay: the simulation-truth counts treat these customers as legitimate, and
the adjudicated counts the rule uses follow each order's label. Only the
perfect reviewer declines them on simulation truth, so only its truth-layer
friction includes those declines.

### Its confusion matrix

The replay records each reviewed order's final outcome (clear, decline,
escalate, cancelled after an unanswered hold, unchanged after shipment,
undecided) by adjudicated label basis, shipment status at the first decision,
and evidence strength then (0, 1, 2 or more adverse families). An order never
decided has strength "undecided" and is filed under "before shipping", which
for it records no decision. A diagnostic version substitutes the latent pattern
for the label. Both appear as `replay.confusion` and
`replay.confusion_latent` in `results/replay.json` and are rendered in the
[operations appendix](../reports/appendix-operations.md).

## Limits

### What a synthetic world can show

The synthetic world supplies the same traffic to every policy and records
truth for evaluation. Policies use the permitted evidence, and every dollar
is traced through one ledger. The world represents the race between review
and shipment, a fixed review allotment, labels arriving weeks later, and the
cost of holding or declining legitimate customers. Comparisons show what
happens under these parameters and which changes of assumption change the
answer.

They do not estimate real fraud rates, losses, detection performance,
verification pass rates or customer and attacker behaviour. Patterns, signals
and rates are generator choices (the fraud table above); detection performance
measures how distinct the design makes each pattern. At about 8,000 orders a
month, the world is small for a pay-in-4 platform, which is why capacity is an
allotment of minutes rather than headcount. Costs (LTV proxy, analyst hour,
hurdle) and friction caps are assumptions and risk appetite, not benchmarks.
Results describe mechanisms under these assumptions, not a real portfolio.

The ten final worlds share the generator and its parameters. Their spread
measures variation between worlds, not whether the parameters are right.
World size, capacity base and tuning procedure were set with the development
worlds open; final worlds were generated after the freeze. Adjudicated labels
miss fraud without a determinable trace: a never-pay customer's single default
reads as credit loss, and an address-only promotion farm earns no
promotion-abuse determination (its members who never pay can still meet the
never-pay marker through the shared address). Orders whose label ends negative
count as legitimate in the friction measures; simulation-truth counts appear
alongside. Case files illustrate a development-world replay, not final
evidence.

### Assumptions of the replay

- **Fixed attempted traffic.** Policies see the same attempts; they neither
  bring customers nor drive them away.
- **No attacker adaptation.** Declined or blocked fraudsters do not retry with
  new details, move accounts or change tactics; their later attempts arrive as
  generated. The replay can overstate the benefit of declines and blocks if
  attackers would evade them.
- **No customer churn.** Legitimate customers keep placing generated orders
  after holds, cancellations or declines; account blocks refuse them at
  checkout. The flat LTV proxy prices lost future value.
- **Training labels for every order.** Classifiers use approve-all labels,
  including those for orders today's rules would decline.
- **Evaluation labels from the approve-all world.** A prevented order is judged
  by what it would have become; actions never re-adjudicate an order.
- **Linkage at decision time.** Escalation blocks accounts linked then; later
  links cause no blocks.
- **Action starts at the evaluated window.** Every policy uses approve-all
  history before it.
- **Evidence up to one replay day old** (The replay, Reviewer): a decision
  does not yet see the policy's own decisions earlier the same day.
- **A fixed reviewer procedure.** It always takes the standard disposition
  and makes no other errors. All orders use the same review-time distribution,
  independent of evidence. Check-triggered re-evaluation takes no analyst
  time, and checks cost nothing.
- **Check outcomes depend only on who ordered.** They are independent across
  orders and of evidence and time; late answers are lost.
- **Holds affect only their order.** A verified order ships later by the
  pause, with no other change to the customer.
- **Merchants act as generated.** A busting merchant keeps selling regardless
  of policy; merchant-level actions are outside scope (FP-2 §1.2).
- **One queue, fixed allotment.** Every family has the same allotment, with no
  overtime; excess work waits.
- **Ledger product terms and liability rules** (Ledger, above).

### Which sensitivity tests which assumption

| Assumption | Sensitivity | Pre-registered values | Where reported |
| --- | --- | --- | --- |
| Verification pass rates | weak verification | takeover `contact` 0.50 passed, 0.35 failed; takeover and third-party `id_check` 0.25 passed, 0.45 failed | flip table; [operations appendix](../reports/appendix-operations.md) |
| The value of a lost customer | LTV proxy | $5 and $45 (base $15) | flip table; [economics appendix](../reports/appendix-economics.md) |
| How fast goods ship | fulfilment lag | ×0.5 and ×2 from the test window | flip table; operations appendix |
| The review allotment | allotment level | 17 and 50 minutes per shift (base 33) | flip table; operations appendix |
| When analysts work | shift layout | evening layout at 33 minutes per shift | flip table; operations appendix |
| Stable traffic and fraud mix | world family | acquisition surge; fraud-mix shift | flip table; appendices |
| Policy-specific history | frozen approve-all history (diagnostic) | none | appendices |
| How much better review could be | perfect reviewer (diagnostic) | none | appendices |

The [operating review](../reports/operating-review.md) reports the flip table
and its verdict ("holds in N of M cells"); appendices give the figures for
each cell. Fixed traffic, attacker adaptation, churn and training-label
availability have no sensitivity, and neither do the review-time distribution,
the check answer delays, the 48-hour hold or the 20 senior minutes. Without
attacker adaptation, the replay can overstate what declines and blocks are
worth. Customer churn is not simulated; the flat LTV proxy is an assumed cost
of lost future value.
