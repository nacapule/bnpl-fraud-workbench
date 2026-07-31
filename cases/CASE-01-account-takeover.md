# CASE-01 — Account takeover of a 395-day tenured account

**Alert:** #3215 (order 350035, score 80, band review) + three sibling alerts on the same
account (#3202, #3234, #3213) · **Account:** user 21069 · **Window:** 2025-11-12 04:56 →
2025-11-13 00:44 UTC · **Exposure:** $3,642.95 across 4 orders · **Resolution:**
confirmed third-party ATO → `decline_block` + account recovery · **Pattern (ground truth,
post-hoc):** P-ATO

## Alert context

R01 fired on four orders from account 21069 inside 20 hours: credential/contact changes
2.1–19.6h before each order, all placed from a device the account had never used. R03
(BIN US vs IP IN, AVS mismatch) co-fired on the $577.72 order, pushing alert #3215 to
score 80. The account is 395 days old with clean history — exactly the profile FP-1 §4
weights R01 for.

## Evidence

Account events (Q05 chain — the manipulation trail, all from new device `d_5181268`,
IP `77.110.184.81`, geolocating to IN):

| ts | kind | ip | device |
|---|---|---|---|
| 2025-11-12 04:56:23 | password_change | 77.110.184.81 | d_5181268 |
| 2025-11-12 05:06:23 | email_change | 77.110.184.81 | d_5181268 |
| 2025-11-12 05:56:23 | address_add | 77.110.184.81 | d_5181268 |

Orders that followed, against the account's own baseline:

| order | ts | amount | category | ip_country | device | ship addr |
|---|---|---|---|---|---|---|
| 350032 | 11-12 07:12 | **$1,365.75** | jewelry | IN | d_5181268 | a_2898594 (new) |
| 350034 | 11-12 07:48 | **$951.72** | electronics | IN | d_5181268 | a_2898594 |
| 350035 | 11-12 14:29 | **$577.72** | jewelry | IN | d_5181268 | a_2898594 |
| 350033 | 11-13 00:43 | **$747.76** | electronics | IN | d_5181268 | a_2898594 |
| *prior:* 82215 | 10-06 12:31 | $29.84 | apparel | US | d_21069 | home |

Pre-takeover repayment: 6/6 installments paid on time. Post-takeover: all 12
installments on the four orders failed. On 2025-12-17 the victim placed a $205.30 order
from their **original** device and home address — the legitimate customer never left.

```mermaid
sequenceDiagram
    participant A as Attacker (IN ip, new device)
    participant Acct as Account u_21069
    participant P as Platform
    A->>Acct: 04:56 password_change
    A->>Acct: 05:06 email_change (locks out victim notifications)
    A->>Acct: 05:56 address_add (drop address a_2898594)
    A->>P: 07:12–14:29 three orders $2,895 → new address
    A->>P: 00:43 (+20h) fourth order $747
    Note over P: R01 fires on each order; R03 on #350035 (score 80)
```

## Competing hypotheses

1. **Account takeover (high).** Credential change → contact change → address add →
   high-value orders to the new address within hours, from a first-seen device on a
   foreign IP. This is the R01 sequence, and the *order* of operations matters: a real
   user changes a password *after* a new phone, not before shipping jewelry to an
   address added 76 minutes ago.
2. **Traveler with a new phone (rejected).** FP-1 §6.1/§6.6: travelers keep their
   shipping address and amount habits. Here shipping moved to a brand-new address and
   spend jumped from a $29.84 apparel baseline to $1,365.75 jewelry.
3. **Legitimate relocation (rejected).** Movers (FP-1 §6.2) add an address without
   credential churn, and their old-address life stops. The victim's December order from
   the original device/address shows the real customer's life continuing unchanged.

## Resolution

`decline_block` per FP-1 §5-ATO: the two unfulfilled orders voided; device d_5181268 and
address a_2898594 blocked; forced credential reset through the pre-takeover email;
account restored to the victim. Fraud loss on fulfilled orders $2,183 (principal minus
down payments), chargebacks anticipated (60% of ATO orders in this dataset produce a
`fraud`-reason chargeback within 2–6 weeks). Priority was P0 at first review: the third
order was inside the 12-hour fulfillment window.

## Prevention follow-up (measured)

Alert #3202 (first order, score 45) sat in the review band — at 26 alerts/day a queue
delay could have cost the whole burst. Proposal: **conjunction bonus — R01 + ship-to
address added <48h before the order → +15**. Measured on holdout months 10–12: ATO
orders crossing the auto-decline band rise **1 → 13 (of 50)**, with **zero** additional
false declines (no benign order in the holdout fires R01 while shipping to a <48h-old
address). Adopted into the proposed-changes queue per FP-1 §8.

## Claude-drafted memo (advisory) + analyst verdict

The triage layer (claude-sonnet-5, prompt memo_v1) drafted for alert #3215:
`recommended_action: escalate, priority P0` — hypotheses ATO high / stolen-card low /
benign low; citations R01, R03, FP-1 §1, §6; evidence gaps: *"contact customer via a
channel predating the 2025-11-12 04:56 credential changes"*, *"IP/proxy reputation check
on 77.110.184.81"*, *"check deliverability/fraud history on ship_address_id 2898594"*.

**Analyst verdict:** agree with the diagnosis; action executed as `decline_block` +
recovery rather than escalate — aggregate exposure ($3.6k) is above the FP-1 §3 escalate
line, but single-account ATO with a complete evidence chain is within analyst authority;
escalation reserved for linkage to other accounts (none found via Q02 on the device and
address). The memo's first evidence gap (out-of-band contact) is exactly the FP-1 §5
recovery path taken.
