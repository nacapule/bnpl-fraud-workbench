# CASE-05 — False positive: the traveler with an expensive taste, cleared

**Alert:** #7592 (order 276276, user 1284, score 35, band review: R03) · **Window:**
2026-04-11 · **Amount:** $1,258.71 — the account's largest-ever order, from a Vietnam
IP · **Resolution:** **cleared**, order fulfilled normally · **Pattern (ground truth,
post-hoc):** none — benign traveler

The system exists to catch CASE-01 through CASE-04. This case is the other half of the
job: FP-1 §1 prices a false decline (margin + lifetime value + the insult), and §6
exists so that a 2.4-year customer buying a laptop in Hanoi doesn't get treated like a
criminal.

## Alert context

R03 fired: card BIN US vs IP VN, with a CVV mismatch (`AVS=Y/CVV=N`). Score 35 —
review band, no auto-action. On its face: foreign IP + verification failure + 13.7× the
account's average ticket. That resembles the opening of a stolen-card case.

## Evidence — why this is §6.1, not P-STOLEN

Account 1284, signed up 2023-12-28, 36 approved orders over the observed year:

| signal | value | reading |
|---|---|---|
| repayment | **105/105 installments paid** | 2.4 years of perfect standing |
| device | d_1284, first seen 2024-05-05, used for **every** order incl. this one | no device change — the ATO/stolen tells are absent |
| credential events | **none, ever** (no password/email/address changes) | nothing was manipulated before the order |
| the VN window | orders 276276 + 276481 on 2026-04-11; every order before and after is US | contiguous foreign window, then reversion — the §6.1 traveler shape |
| the second VN order | $96.93 shipped to a *different* address (a_2696) | souvenir to family — gift-buyer §6.3, consistent with travel, not with theft |
| AVS | **Y** — the address on file matches | a thief shipping to a drop fails this; CVV=N alone sits inside the ~4% typo noise (§6.5) |
| amount | $1,258.71 vs prior max $723.92, in electronics while abroad | large, but the account has bought $338–$723 before; 1 order, not a burst (R05 silent) |
| aftermath | May 20 order $723.92 from US home IP, home address; installments on the VN order: paid on time | the customer came home and kept paying |

The stolen-card counter-hypothesis fails on structure: no card testing (zero declines on
the card, R07 silent), no linkage (0 other accounts on the device/address, R02/R08
silent), no credential changes (R01 silent), AVS pass, and repayment that a fraudster
has no reason to continue.

## Competing hypotheses

1. **Benign traveler (high).** Contiguous foreign-IP window with normal life on both
   sides; AVS pass; perfect 2.4-year repayment continuing through and after the trip.
2. **Account takeover (rejected).** FP-1 §6.6: takeover leaves a manipulation trail
   (CASE-01's password→email→address chain). Here there are zero credential events and
   the order came from the customer's own 2-year-old device.
3. **Stolen card in Vietnam (rejected).** A thief with the card number lacks the
   customer's device and address; this order has both, and no testing prelude.

## Resolution

**`clear`** at first touch; order fulfilled on schedule; no customer contact needed (the
evidence answered the memo's verification question without a step-up — contact has its
own friction cost and FP-1 §3 reserves `hold_contact` for cases where the packet can't
decide). Logged as a benign-explanation clear under §6.1 for QA sampling.

## Prevention follow-up (measured)

R03-only alerts are the queue's biggest benign contributor. Proposal: **suppress R03
when the account is >180 days old, AVS passes, and there has been no credential change
within 72h** — exactly this case's profile. Measured on holdout months 10–12: removes
**364 of 1,303** R03-only alerts (−4.0 alerts/day, a 28% cut of that segment) with
**zero** fraud among the suppressed set (every true-fraud R03 alert in the holdout also
carried a mismatched AVS, a young account, or a credential change). Adopted per FP-1 §8.
The insult that never happens is invisible in a loss ledger; this is where it gets
counted.

## Claude-drafted memo (advisory) + analyst verdict

Triage memo for alert #7592 (claude-sonnet-5, memo_v1): top hypothesis **benign (high)**
with account_takeover (med) behind it; `recommended_action: hold_contact, priority P1`;
citations R03, FP-1 §6; evidence gaps: *"contact cardholder to verify travel status"*,
*"check whether the IP resolves to VPN/proxy vs genuine VN geolocation"*, *"confirm
fulfillment status to assess reversibility window"*.

**Analyst verdict:** diagnosis shared, action overridden — `clear` instead of
`hold_contact`. The memo hedged toward step-up verification; the packet already
contained the disambiguating facts (AVS pass, device continuity, zero credential
events, repayment through the trip), and §6 instructs analysts to weigh the mimic
checklist *before* spending customer friction. This is the intended division of labor:
the model drafts and surfaces gaps, the human owns the judgment — and the QA record of
overrides like this one is exactly what the next prompt iteration learns from.
