# CASE-02 — Card testing burst converting to stolen-card checkout

**Alert:** #3946 (order 350401, score 75: R03 + R07) and #3957 (order 350400, score 40:
R07) · **Account:** user 40044 (created 2025-12-03) · **Window:** 2025-12-04 01:09 →
13:30 UTC · **Exposure:** $1,939.78 across 2 approved orders · **Resolution:** confirmed
stolen card → `decline_block`, device + card token blocked · **Pattern (ground truth,
post-hoc):** P-STOLEN

## Alert context

R07 (card testing) fired: five declined attempts on the same card and device inside 16
minutes, followed by approvals hours later. On the first approval R03 co-fired (BIN US,
IP BR, CVV mismatch), lifting it to score 75. Account was one day old at the time —
Q03's "new account, starts big, verifies badly" shape.

## Evidence

Full attempt sequence on the card/device (Q08 output for this device):

| order | ts | amount | status | AVS | CVV |
|---|---|---|---|---|---|
| 350395 | 01:09:03 | $38.31 | declined | Y | N |
| 350396 | 01:13:03 | $25.94 | declined | N | N |
| 350397 | 01:17:03 | $26.79 | declined | Y | M |
| 350398 | 01:21:03 | $38.42 | declined | N | N |
| 350399 | 01:25:03 | $28.19 | declined | N | M |
| **350401** | 05:25:10 | **$733.09** | approved | Y | N |
| **350400** | 13:30:05 | **$1,206.69** | approved | Y | M |

The tell is the *shape*, not any single attempt: five sub-$40 probes at exact 4-minute
intervals (scripted cadence), mixed AVS/CVV results as the attacker permutes address and
code guesses, then a 20–30× jump in ticket size once a combination sticks. Benign
retries after a declined purchase re-attempt the *same* amount within a minute or two —
they do not probe with five different small baskets.

Corroboration: signup→first-attempt gap 17h; IP geolocates BR against a US-BIN card for
every attempt; no order or device history before the burst. A `fraud`-reason chargeback
arrived on order 350400 on 2026-01-14 (outcome: lost) — the cardholder found the charge,
six weeks after the goods shipped.

## Competing hypotheses

1. **Stolen card after testing (high).** Decline burst → success → high-value burst, on
   a day-old account, foreign IP vs BIN, chargeback later. Every element agrees.
2. **Legitimate customer with card trouble (rejected).** FP-1 §6.5 covers verification
   typos — but a typo produces 1–2 mismatches on the *same* basket, not five distinct
   small baskets at fixed intervals; and the benign AVS/CVV mismatch base rate (~4%)
   cannot explain 5-of-7 attempts mismatching.
3. **Traveler on a new account (rejected).** A Brazilian traveler with a US card would
   fail R03 occasionally, but has no reason to probe $25–$38 baskets at 01:00 and then
   buy $1,940 of high-value goods the same day on a brand-new account.

## Resolution

`decline_block` per FP-1 §5-Stolen: account blocked, device fingerprint and card token
denylisted, both orders reported to loss accounting without waiting for the second
dispute (FP-1 §5: chargeback-certain). Loss on fulfilled orders: principal minus the
down payments (~$1,455); the down payments themselves will be clawed back by the
issuing bank's dispute.

## Prevention follow-up (measured)

Reproduced by [`analysis/followups.py`](../analysis/followups.py) and the generated
[`reports/followups.md`](../reports/followups.md) result.

The alerts fired on the *approvals* — after exposure existed. Proposal: **pre-approval
device cooldown — ≥5 declines on one device within 24h places the device on a 24h
order-hold** (holds, not blocks: FP-1 §3 reversibility). Measured across the full year
of data: **46 devices** would have been cooled down, **all 46** belonging to
labeled-fraud users; **zero** benign-user devices reach 5 declines/24h (benign declines
are isolated events at a 1.5% rate). Applied to this case, the cooldown triggers at
01:25 — four hours before the first approval — and the $1,939.78 exposure never opens.

## Claude-drafted memo (advisory) + analyst verdict

Triage memo for alert #3946 (claude-sonnet-5, memo_v1): `recommended_action:
decline_block, priority P0`; top hypothesis stolen_card (high) with benign weighed low;
citations R03, R07; evidence gaps led with checking the same card/device across *other*
accounts (Q02/Q08 linkage — correct instinct; none found beyond this account).

**Analyst verdict:** agree, executed as recommended. P0 was right at review time: the
second order was still inside the 12h fulfillment window when the alert queue surfaced
it.
