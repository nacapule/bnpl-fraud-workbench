# CASE-04 — Never-pay vs hardship: two written-off accounts, opposite verdicts

**Alert:** #736 (order 352579, user 41005, score 35: R03) · **Comparison account:** user
4969 (order 303062 — never alerted) · Both ended in written-off installments. One is
fraud; one is a customer in trouble. Getting this distinction wrong in either direction
is a reportable QA error under FP-1 §1/§6.3. · **Patterns (ground truth, post-hoc):**
P-NEVERPAY vs unlabeled (benign hardship)

## The two accounts, side by side

| | user 41005 | user 4969 |
|---|---|---|
| signup | 2025-08-15 | 2024-04-06 (2+ years tenure) |
| first order | **22 hours after signup** | 19 months after signup |
| order history | 1 order, $579.54 (1.38× category median) | 8 orders, $62–$698, habitual range |
| down payment | succeeded | succeeded |
| installments | **0/3 paid — every attempt AND retry failed** | 7 paid across the year, then failures accumulate |
| failure shape | cliff: nothing ever collects | spiral: pays installment 1s, misses 2s and 3s, keeps trying |
| device/IP | single new device, consistent | own device for 2 years |
| account events | none | none |

Repayment detail for user 4969 (Q04's partial-payer shape): November–March orders show
`paid` on first installments (2025-11-26, -12-31 ×2, 2026-01-18, 2026-03-20) with
`written_off` seconds and thirds — payment *effort* continuing while capacity shrinks.
User 41005's ledger shows six `fail` rows (three due dates × retry each) and zero
successes after the down payment.

## Alert context

Ironically, the alert on 41005 wasn't a repayment rule at all — R03 fired at checkout
(BIN US / IP CA with AVS mismatch, score 35, review band). The reviewing analyst cleared
the *order* at checkout time (single mild geo signal on an otherwise plausible first
purchase — correct call on the evidence then available). The account returned to view
two weeks later when the first installment failed both attempts; this case file
documents the *account-level* determination FP-1 §6.3 requires.

## Competing hypotheses (the §6.3 determination)

For user 41005:
1. **First-party never-pay (high).** The §6.3 checklist: tenure zero (first order 22h
   after signup); payment effort zero (no successful payment after the mandatory down);
   order size 1.38× category median on a first-ever order; failure is a cliff, not a
   spiral. The account was built to place this order.
2. **Hardship (rejected).** Hardship shows *some* effort — a paid first installment, a
   successful retry, tenure, an order history in normal range. All absent.
3. **Stolen card (rejected).** No chargeback in 90+ days (a real cardholder disputes),
   and the R03 geo signal alone, with AVS mismatch at the ~4% benign noise rate, is
   corroboration-grade, not determination-grade (FP-1 §6.5).

For user 4969: tenure 2 years, 7 paid installments with continuing partial effort, no
new devices/addresses/credential events, order sizes inside habit. **Credit loss.**

## Resolutions

- **41005:** never-pay determination → `decline_block` future orders; existing plan to
  collections; written-off $434.66 (principal minus down) booked as fraud loss.
- **4969:** hardship → standard dunning path, **no fraud action, no account block**
  (FP-1 §1: misclassifying hardship as fraud is the same severity of error as missing
  fraud). Account remains open with reduced exposure via normal credit controls.

## Prevention follow-up (measured — an honest dead end)

Only **1 of 63** holdout never-pay orders crossed the review band at all: after the
simulator's realism pass, never-pay first orders sit *inside* the benign amount
distribution (1.05–1.6× median), on clean devices, with plausible identities. We tested
a candidate rule — first order >1.5× category median on a <48h account, +15 — and it
added **one** alert on the whole holdout (a stolen-card order, not never-pay). We also
sized a first-order amount cap and found it binds on 26.8% of *benign* first orders
while barely touching this pattern. Both rejected. The honest conclusion, recorded per
FP-1 §8: **first-party never-pay is not a transaction-time detection problem.** The
lever is credit strategy — start small, grow limits with repayment history — plus the
repayment-outcome models the ML layer's installment-history features feed on repeat
orders. Rules that pretend otherwise just tax legitimate first-time customers.

## Claude-drafted memo (advisory) + analyst verdict

Triage memo for alert #736 (claude-sonnet-5, memo_v1) at *checkout time*:
`recommended_action: hold_contact, priority P1`, hypotheses stolen_card (med) / benign
(med), citing R03 and FP-1 §6, with evidence gaps "confirm whether ship address matches
the cardholder's billing address" and "check card for use across other accounts."

**Analyst verdict:** the memo's caution was reasonable at checkout (geo + AVS on a fresh
account), and its med/med split correctly refused to call fraud on one weak signal. The
checkout-time analyst cleared with monitoring; the account-level never-pay determination
two weeks later used evidence (repayment behavior) that did not exist at memo time.
Advisory drafts age; ledgers decide.
