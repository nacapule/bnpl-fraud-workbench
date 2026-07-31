# Fraud Operations Policy

**Document:** FP-1 · **Applies to:** consumer and merchant transaction review on the
simulated BNPL platform · **Review cadence:** quarterly, or after any rule/threshold change

This is the internal policy that governs how alerts are investigated and actioned in this
workbench. Analyst memos, the Claude triage layer, and the written case files all cite it
by section and rule id (e.g. FP-1 §4, R03). It is written for a pay-in-4
product: the platform pays the merchant up front and collects 25% at checkout plus three
biweekly installments, so the platform holds both the fraud risk and the credit risk of
every approved order.

---

## 1. Definitions and loss taxonomy

**Fraud loss** — loss caused by a party who never intended to be the accountholder or
never intended to repay *at the moment of the transaction*, or a merchant complicit in
extracting funds. Third-party fraud (stolen card, account takeover, synthetic identity)
and first-party fraud (never-pay, friendly fraud / INR abuse, promo abuse, merchant
bust-out) are both fraud.

**Credit loss** — loss from a genuine customer who intended to repay and could not.
Distinguishing marks: partial repayment effort (some installments paid or retried
successfully before failure), stable device/IP/address history, no post-order account
manipulation, order size consistent with history. Credit losses are **not** actioned under
this policy beyond standard dunning; misclassifying hardship as fraud (or vice versa) is a
reportable QA error. See §6.3 and CASE-04.

**Abuse** — policy exploitation that may not be chargeable fraud: repeated
item-not-received (INR) disputes on delivered goods, multi-account promo redemption.
Actioned under this policy with account-level remedies (§5), not order declines alone.

**Insult** — a false decline of a legitimate customer. Insults carry a measured cost
(config `costs.false_decline_*`) and a reputational cost; §6 exists to keep them rare.

## 2. Evidence standards

1. Every action stronger than *clear* must cite at least one fired rule (R01–R12) **and**
   the underlying observable facts (rows, timestamps, linkage counts). "Score was high" is
   not a finding.
2. Facts must come from the case packet or from queries an analyst ran; memos quote them
   verbatim. A claim that cannot be traced to a packet field or query result is treated as
   unsupported and voids the memo (this is enforced mechanically for LLM-drafted memos by
   the eval verifier).
3. Alternative benign explanations (§6) must be considered and explicitly rejected in any
   memo recommending `decline_block`. A memo that never weighs the benign hypothesis is
   incomplete.
4. Labels/ground truth (simulator artifacts) are **never** available to analysts, the rules
   engine, or the triage layer; they exist only in offline evaluation. Any workflow found
   reading them is invalid.

## 3. Action ladder

| Action | Meaning | Reversibility |
|---|---|---|
| `clear` | Alert closed as benign; order proceeds; no customer contact | — |
| `hold_contact` | Order/installment plan paused pending customer verification (step-up: email/SMS confirm, card re-auth). Auto-releases in 48h if verified | Fully reversible |
| `decline_block` | Order declined (or plan frozen if post-fulfillment); account blocked from new orders pending manual review | Reversible pre-fulfillment; loss-bearing after |
| `escalate` | Suspected ring, merchant complicity, or ≥ $2,000 aggregate exposure: hand to senior review with linkage evidence | — |

Priorities: **P0** — fulfillment imminent (< 2h) or active burst in progress; **P1** —
high-exposure single account (> $500) or growing linkage cluster; **P2** — standard queue;
**P3** — post-loss documentation / no live exposure. SLA targets: P0 ≤ 1 business hour,
P1 ≤ 4, P2 ≤ 8, P3 ≤ 24 (see queue simulation).

## 4. Rules and their intent

Thresholds live in `config.yaml` and `rules/definitions.py`; this section records intent
and rationale so threshold changes don't silently change meaning. Score = Σ weights of
fired rules; bands (config `rules.bands`): below review band → auto-approve, review band →
analyst queue, decline band → auto-decline pending review.

| Rule | Intent | Primary pattern |
|---|---|---|
| **R01** credential/contact change ≤ 48h before order on a ≥ 90-day account, from a new device | Account takeover leaves a manipulation trail before the money moves | ATO |
| **R02** ≥ 3 accounts sharing one device in 30d | Device reuse is the cheapest ring signal | Synthetic, promo |
| **R03** card BIN country ≠ IP country **and** AVS or CVV mismatch | Geography plus verification failure compounds; either alone is noise (§6.1) | Stolen card |
| **R04** first order > category P95 with account age < 7d | New accounts that start at the top of the amount distribution front-load risk | Stolen, never-pay |
| **R05** order burst: > 3 orders/24h per user or > 5 per device | Velocity beyond organic shopping rhythm | Stolen, ATO |
| **R06** disposable email domain, or plus-addressed duplicate of an existing account | Identity cost reduction is a precondition of scaled abuse | Synthetic, promo |
| **R07** decline burst then success on the same card/device | The card-testing signature: validate stolen credentials cheaply, then spend | Stolen |
| **R08** ship-to address shared by ≥ 3 unlinked accounts | Goods have to land somewhere; drops collapse rings | Synthetic |
| **R09** ≥ 2 prior INR chargebacks on delivered orders | Repeat INR with healthy repayment is the friendly-fraud shape | INR abuse |
| **R10** promo redemption inside a device/address cluster ≥ 3 | Promo economics attract multi-accounting | Promo |
| **R11** impossible geo-velocity between consecutive events | Two locations faster than travel allows means two actors or proxying | ATO, stolen |
| **R12** vendor email/IP risk score ≥ threshold | Independent external signal on identity infrastructure (leased IPs, abuse-listed ranges, throwaway mailboxes) | Stolen, synthetic |

Rule weights encode *specificity*, not severity: R01 and R09 are heavily weighted because
their trigger conditions are rare among legitimate customers; R03's components are
individually common, so it earns weight only in conjunction.

## 5. Standard resolutions by confirmed pattern

- **ATO:** decline_block open orders; force credential reset; restore account to victim;
  refund/void manipulated orders; P0/P1.
- **Stolen card:** decline_block; block device fingerprint and card token; report
  chargeback-certain orders to loss accounting immediately (don't wait for the dispute).
- **Synthetic ring:** escalate with linkage map; block all member accounts and shared
  devices/addresses; sweep for not-yet-transacted members via R02/R08 linkage.
- **Never-pay:** decline_block future orders; existing plan to collections path; document
  the credit-vs-fraud determination per §6.3.
- **INR abuse:** hold_contact; require signature confirmation on future deliveries;
  ≥ 3 confirmed abusive disputes → account closure per terms of service.
- **Promo abuse:** void unredeemed promos across the cluster; decline_block only the
  coordinating accounts; small-value organic members may be cleared with promo forfeiture.
- **Merchant bust-out:** freeze settlement, hold pending payouts, escalate to merchant
  risk (see companion screener project); consumer-side orders from the merchant enter
  enhanced review.

## 6. False-positive guidance — the benign explanations that mimic fraud

Analysts must check these before any decline:

1. **Travelers** mimic ATO/stolen geography: foreign IP, sometimes a new device — but no
   credential changes, shipping unchanged, amounts in habitual range, and the window is
   contiguous then reverts (CASE-05 is the canonical exoneration).
2. **Movers** mimic ring/drop signals: new address plus shipping shift — but single
   account per address, old address activity stops, no device sharing.
3. **Gift buyers** mimic drop shipping: ship-to ≠ home — but the recipient address has no
   account cluster and the buyer's history is otherwise stable.
4. **Benign hardship defaulters** mimic never-pay: installments fail — but there is
   partial payment effort, tenure, and no first-order concentration (§6.3).
5. **Typos** mimic verification failure: AVS/CVV mismatch base rate among legitimate
   orders is ~3%; a mismatch is corroborating, never sufficient (pairs with R03's design).
6. **New phones** mimic ATO device changes: an organic device_add without credential
   changes or address changes, followed by normal-pattern orders, is not takeover.

**6.3 Credit-vs-fraud determination (required for every never-pay recommendation):** state
tenure, payment effort (count and amounts of successful/retried payments), order-size
ratio to history/category, and post-order account activity. Zero-effort + first-order-heavy
+ oversized → fraud path; any genuine repayment effort or tenure → credit path with
dunning, **never** a fraud label.

## 7. Memo format

Every reviewed alert produces a memo with exactly these fields: `signals_observed`
(verbatim facts), `hypotheses` (each candidate pattern **including the benign one**, with
likelihood low/med/high and reasoning), `policy_citations` (rule ids + sections),
`recommended_action` (§3 ladder), `priority` (P0–P3), `evidence_gaps` (the cheapest query
or check that would most change the decision), `memo_markdown` (prose for the case file).
Machine-drafted memos follow the same schema and are advisory: **the analyst owns the
decision**; auto-actions occur only at the score bands of §4, never from a memo alone.

## 8. Change control

Rule/threshold changes require: the motivating case (what was missed or falsely flagged),
the proposed change, and its measured effect on the holdout months (alert volume, recall
by pattern, insult rate, net $ under the cost model). Case files end with exactly this
analysis in their "Prevention follow-up" section.
