# Fraud Operations Policy

**Document:** FP-2 (replaces FP-1) · **Applies to:** checkout decisions and analyst review
of consumer orders on the simulated pay-in-4 platform · **Changes:** §12

This policy governs order decisions. The simulated review procedure and memo checks
implement each clause; every adverse action cites its clause and facts, for example
"FP-2 §6.6(b), R01".

## 1. Scope

**1.1** Customers pay 25% at checkout, the rest in three fortnightly installments.
The platform pays merchants when they report shipment, net of their fee, and carries
fraud and credit risk on every approved order. Merchants ship within hours; decisions
after shipment cannot prevent the order's loss.

**1.2** This policy covers checkout routing, analyst review, verification, queue priority
and later determinations. Merchant-level actions, such as payout freezes, belong to
merchant risk.

## 2. Definitions

**2.1 Fraud** is loss caused by someone other than the legitimate account holder or
cardholder (third-party: stolen card, account takeover, synthetic identity), an account
holder who never intended to repay or keep the terms (first-party: never-pay,
item-not-received abuse, promotion abuse), or a complicit merchant (merchant bust-out).

**2.2 Credit loss** is unpaid money without evidence of intent not to repay (§8.4).

**2.3 Friction** is any hold, cancellation or decline affecting a legitimate customer,
including orders refused for account blocks.

**2.4 Decision time** is checkout for routing; the analyst's decision for review.

**2.5 Linked accounts** are the other accounts counted by R02 or R08 for this order: any
account with an order attempt or account event on its device, or an order attempt to its
shipping address, in the 30 days before decision time.

## 3. What is known when

**3.1** Make and judge each decision using only evidence known at its decision time.

**3.2** Evidence arrives at different times:

| Evidence | Known |
|---|---|
| Account opening; password, email, device and address changes | when they happen |
| Order attempts, processor declines, AVS and CVV results, IP, device, card, address, promotion use | at the attempt |
| Shipment reports by the merchant; carrier-confirmed deliveries; merchant closures | when they happen |
| Verification outcomes (§5) | at check completion |
| Installment payments and failures | each due or retry date, from two weeks after checkout |
| Default (§8.2) | 30 days after a missed due date |
| Disputes | reason (`unauthorized`, `item_not_received`, `not_as_described`) at opening; outcome (`won`: decided against the customer; `lost`) at resolution, weeks later |
| Victim reports | when filed |

**3.3** Simulation truth and outcome labels are unavailable to analysts, rules, the review
procedure and memo drafter. Any workflow reading them is invalid.

**3.4** An order whose outcomes are not yet known is unknown, not legitimate.

## 4. Actions

**4.1** Automatic checkout routing follows score bands fixed before use in the active
detection policy (`config/policy.yaml`). An `auto_decline` records no fraud finding and
blocks no account.

**4.2** Analysts resolve reviewed orders with `clear`, `hold`, `decline` or `escalate`.
Only `decline` and `escalate` block accounts, only under §6.

**4.3** `needs_check` is a memo recommendation, not an action: the evidence does not
decide the order, and the memo names the §5.1 check that would. The analyst carries it
out as a `hold` with that check. It is permitted only where §6.6(b) permits it, while a
required check has not yet run.

| Action | Order | Customer | Analyst time |
|---|---|---|---|
| `approve` | ships | none | none |
| `review` | ships unless held or declined first | none unless held or declined | one review at §7 priority |
| `auto_decline` | not created; no money moves | declined, not blocked | none |
| `clear` | ships | none | review |
| `hold` | before shipment: shipment and merchant payment paused up to 48 h for the checks; after shipment: the checks run and nothing is paused | asked to verify; before shipment, released if verified, otherwise cancelled with the checkout payment refunded; after shipment, unchanged unless a check fails (§5.3) | review and checks |
| `decline` | before shipment: voided and refunded; after shipment (decline after fulfilment): loss stands | blocked; later orders declined | review |
| `escalate` | as `decline` | as `decline`; all linked accounts also blocked | review and senior review |

## 5. Verification

**5.1** Two checks, each available at most once per order, establish who is ordering,
never repayment intent, promotion entitlement or the truth of delivery claims.

- (a) `contact`: ask the account holder to confirm they placed the order, using contact
  details on the account before any change in the 30 days before the order.
- (b) `id_check`: the customer authenticates the payment with the card's issuer and
  verifies the identity on the account, which establishes that they hold both.

Each check returns `passed`, `failed` (account holder disowns the order, or
authentication or identity verification fails) or `no_response` (including technical
errors).

**5.2** Families present (§6.2) require `contact` for Account access and `id_check` for
Card, Velocity and Linkage. Run both when both are required.

**5.3** Check outcomes under §6.6(b), applied as each check completes:

- (a) Every required check `passed`: standard `clear`; `hold`, `decline`, `escalate` and
  `needs_check` prohibited.
- (b) A check `failed`, which decides at once, also after shipment (the loss stands):
  standard `escalate` if Linkage is present, otherwise `decline`; `decline` also
  permitted if Linkage is present; `clear`, `hold` and `needs_check` prohibited, and
  `escalate` if Linkage is absent. Cite the failure.
- (c) No response within 48 hours: before shipment, cancel and refund without blocking
  accounts; after shipment, nothing changes. Until then the sets of §6.6(b) apply.

## 6. Evidence standards

**6.1** Every `hold`, `decline`, `escalate` or memo recommending one cites the clause
permitting it and supporting facts: values, times and counts known at decision time.

**6.2** Each rule is an evidence condition in one family, evaluated at decision time.
The four adverse families can support `hold` or `decline`; Context conditions cannot,
alone or together, because no check resolves what they suggest. R12 (a vendor score) is
retired; never reuse its id.

| Rule | Condition | Family | Intent |
|---|---|---|---|
| R01 | password changed or reset, or email changed, in the 48 h before the order; account at least 90 days old; and the device first used on this account in the 72 h before the order | Account access | takeover leaves a trail |
| R11 | the account's previous order attempt came from another country under 12 h earlier, implying over 900 km/h between the countries' centroids | Account access | two places at once |
| R03 | card issuing and IP countries differ, and AVS or CVV failed | Card | stolen card details |
| R07 | 3 or more processor declines on this card, or 3 or more on this device, in the 24 h before the order | Card | testing stolen cards |
| R05 | more than 3 order attempts on the account, or more than 5 on the device, in the 24 h up to and including this one | Velocity | faster than ordinary shopping |
| R02 | 3 or more accounts, this one included, with an order attempt or account event on this device in the past 30 days | Linkage | shared devices tie rings |
| R08 | 3 or more accounts, this one included, with an order attempt to this shipping address in the past 30 days | Linkage | goods land somewhere |
| R06(b) | the normalized email (plus-tags removed; dots removed for Gmail only) matches another account's | Linkage | duplicated identities |
| R10 | the order uses a first-purchase promotion that 3 or more accounts, this one included, linked by a shared device or normalized email have used | Linkage | multi-account promotion use |
| R06(a) | disposable email domain | Context | cheap identity |
| R04 | the account's first order attempt; amount above the 95th percentile of earlier processor-approved amounts in its merchant category; account under 7 days old | Context | front-loaded exposure |
| R09 | 2 or more `item_not_received` disputes opened on earlier account orders | Context | repeated claims |

**6.3** Earlier outcomes known at decision time settle the order: (a) an earlier order
on the account had an `unauthorized` dispute resolved `lost` (§9(a)) and the account
holder has not reported a takeover (§9(b)); (b) an earlier plan on the account met the
never-pay determination (§8.3); (c) item-not-received abuse was determined on the
account (§9(d)).

**6.4** Households share devices and addresses. These exceptions apply to the family
count of §6.6 only; priority (§7.1) uses the rules as they hold. (a) R02 does not count
when the device was first used on this account at least 90 days before the order and R02
counts at most 4 accounts. (b) R08 does not count when the shipping address is the
account's current home address, registered on this account at least 90 days before the
order, and R08 counts at most 4 accounts. Accounts sharing a device or home address for 90 days before
acting are a known gap. Account age and repayment history alone explain nothing:
they describe the account holder, not who ordered.

**6.5** These never support `hold`, `decline` or `escalate`; memos weigh them as benign
explanations or context:

- (a) AVS or CVV failure without R03's geography. Legitimate simulated orders have 4.0%
  AVS and 2.0% CVV failure rates.
- (b) New devices or addresses, travel, password resets after new phones, or shipping
  away from home. Travellers, movers, new phones and gift buyers explain these; when
  they trigger a rule, the check settles it.
- (c) First or large orders outside R04, time of day, or IP range.
- (d) Model scores, rule scores or resemblance to fraud patterns.
- (e) Merchant evidence belongs to merchant risk.
- (f) Other plans' repayment history, good or bad, except §6.3(b).

**6.6** A family is present when one of its conditions holds without a §6.4 exception.
Evaluate at review and again when each check completes, with the evidence known then,
and use the first applicable row. Each row, and each outcome in §5.3, places every
disposition in exactly one set; **Standard** is the review procedure's action. Row (a)
overrides any check result. In row (b), once a check has failed or every required check
has passed, §5.3(b) or §5.3(a) gives the sets instead.

| | Evidence | Standard | Also permitted | Prohibited |
|---|---|---|---|---|
| (a) | an earlier outcome settles it (§6.3) | `escalate` if Linkage is present, otherwise `decline` | `decline` if Linkage is present | `clear`, `hold`, `needs_check`; `escalate` if Linkage is absent |
| (b) | one or more adverse families | `hold` with the §5.2 checks | `needs_check` while a required check has not run; `decline` when two or more families are present and no check has passed | `clear`; `escalate`; `needs_check` once every required check has run; `decline` with one family or after a check has passed |
| (c) | no adverse family | `clear` | none | `hold`, `decline`, `escalate`, `needs_check` |

`escalate` requires Linkage because it blocks linked accounts. R06(b) supports Linkage
but extends no block beyond §2.5.

## 7. Queue priority and service targets

**7.1** Assign priority automatically at queue entry from facts known then; it never
changes. Orders already shipped or cancelled at entry are P3; otherwise, use the first
applicable row. The detection policy orders each priority's queue.

| Priority | Assigned when | Target |
|---|---|---|
| P0 | the merchant's median fulfilment time leaves under 2 h before shipment, or R05 or R07 holds | 1 service hour |
| P1 | the amount is at least $500, or R02, R08 or R10 holds | 4 service hours |
| P2 | any other order | 8 service hours |
| P3 | the order had shipped or been cancelled when it entered the queue | 24 service hours |

**7.2** Service hours follow the fixed `config/policy.yaml` calendar, independent of
the staffing roster. Clock time to decision, and whether the decision preceded shipment,
are reported alongside the target.

## 8. Credit versus fraud

**8.1** First plans lack repayment history, so repayment cannot be judged at checkout.
If no account installment was due before decision time, actions, determinations and memo
recommendations cannot cite non-payment or never-pay; apply §6 alone. Memos may still
name never-pay as a hypothesis.

**8.2** A plan is a zero-effort default when its first installment after the checkout
payment remains unpaid 30 days past due and no full or partial payment has been made
on the plan after the checkout payment.

**8.3** A zero-effort default is never-pay (first-party fraud) only with an intent marker:
(a) another plan on the account, created within 7 days of this plan's creation, is also
a zero-effort default; or (b) two or more other accounts that share a device, a shipping
address or a normalized email with this account have zero-effort defaults whose default
dates (§3.2) fall within 30 days of this plan's. The determination becomes known when the last of its qualifying
facts is known, the shared device, address or email included, and is final; later
payments are recorded as recoveries.

**8.4** Any other default, including a zero-effort default without a marker, is not
never-pay: it is a credit loss unless fraud is confirmed on other grounds (§9), and the
default alone gives no fraud label and no fraud-operations action. Non-payment alone
cannot establish intent. Some of these defaults are never-pay fraud that the evidence
cannot establish; the simulation counts them against hidden truth, for diagnosis only.

## 9. Confirmed outcomes

These outcomes confirm fraud. Each is known when the last fact it rests on is known
(§3.2); the Known column names that fact. Once known, they label orders to measure loss
by pattern and to train models. A decision is still judged on the evidence known when it was
made (§3.1), never by its outcome.

| | Outcome | Confirms | Known |
|---|---|---|---|
| (a) | an `unauthorized` dispute resolved `lost` | third-party fraud on that order | at resolution |
| (b) | account holder reports orders they did not place | account takeover on those orders | at the report |
| (c) | never-pay determination (§8.3) | first-party fraud on that plan | as in §8.3 |
| (d) | two `item_not_received` disputes on the account resolved `won` on orders with a carrier-confirmed delivery | item-not-received abuse on those orders | at the later of the second resolution and the last delivery confirmation |
| (e) | a use of a first-purchase promotion, when two or more other accounts sharing a device or normalized email with its account used the same promotion within 90 days of it, and none of these accounts placed an order without a promotion within 90 days after its own use | promotion abuse on that use | 90 days after the latest of these uses, or when the shared device or email becomes known, if later |
| (f) | an `item_not_received` dispute resolved `lost` on an order its merchant reported shipped, with no carrier-confirmed delivery, where the merchant closed before the resolution | merchant bust-out on that order | at the resolution or the closure, whichever is later |

## 10. Memos

Memos are advisory. They trigger no order action alone and set no queue priority. Each states
facts tied individually to packet fields; competing hypotheses, including the most plausible
benign one (§6.5); applicable clauses; a disposition (`clear`, `hold`, `decline`,
`escalate` or `needs_check`) that is standard or permitted under §6.6, or under §5.3 once
a check has decided; and the cheapest next check that would most change the decision, or
none when the evidence decides.

## 11. Assumptions

Review minutes, check and response times, senior review time, friction cost (merchant
fee and customer-lifetime-value proxy lost on declined or cancelled legitimate orders)
and analysts' hourly cost are assumptions in `config/policy.yaml`, not measurements.
Verification pass rates are simulation assumptions. The AVS and CVV rates
in §6.5(a) come from the simulated world's generator.

## 12. Changes

**12.1** Rule, threshold or clause changes state their supporting case and replay effects
on validation windows (alerts, prevented loss by pattern, friction, review minutes,
net contribution), measured before seeing any test-window result.

**12.2** Rule thresholds (`rules/definitions.py`) and score bands (`config/policy.yaml`)
must match this document. Each change creates a new version, recorded by the review
procedure, memo checks and stored memos. Clause ids are permanent; removed ids are
retired, never reused.
