# Prompt iteration log — triage memo task

The eval harness (`harness.py`) is the referee: no prompt change ships without a
before/after metric table on the frozen eval set (`cases.json`, committed packets,
committed response cache — anyone can re-run `python -m llm.eval.harness --offline` and
get these exact numbers).

## Metric definitions (fixed across versions)

| metric | definition |
|---|---|
| action accuracy | `recommended_action` == ground-truth action (mapping in `select_cases.py`: ATO/stolen/synthetic/never-pay → decline_block, INR/promo abuse → hold_contact, merchant bust-out → escalate, benign → clear) |
| decline precision / recall | positive class = `decline_block` only; escalate does NOT count as a decline. Precision protects legitimate customers (insults), recall protects loss. |
| pattern id rate | top hypothesis (highest likelihood; ties → first listed) matches true pattern |
| hallucination rate (memo) | share of memos with ≥1 unsupported claim: a `signals_observed` entry containing a concrete token (timestamp, amount, id, number) absent from the packet (`verifier.py`, strict by design) |
| hallucination rate (claim) | unsupported claims / all claims |
| citation validity | cited rule ids exist in FP-1; `citation_fired_rate` = share of cited rules that actually fired on the alert |
| consistency | 3 runs on 50 cases with a neutral numbered no-op line injected into the packet; all-3-agree rate on `recommended_action`. (Byte-identical prompts at temp 0 would be answered from the response cache, so consistency is measured under neutral perturbation — that is the honest version of the metric.) |
| cost / latency | API-equivalent cost from CLI-reported token counts at 2026-07 list prices; wall-clock latency per call |

A 30-case manual spot-check protocol accompanies the mechanical verifier each version:
read the raw memos for the flagged-unsupported claims and record whether the verifier's
strictness produced false alarms (e.g. legitimate derived arithmetic like "9 + 1 orders").

## Harness corrections logged during v1 (applied to all versions/arms equally)

Reading the v1 misses raw exposed two defects in the *referee*, not the model:

1. **Ground-truth actions contradicted the policy.** The original single-action mapping
   scored `synthetic_ring → decline_block` as the only correct answer, while FP-1 §5
   explicitly says rings **escalate** with a linkage map — the model was following the
   policy and being marked wrong for it. Corrected to policy-derived acceptable-action
   *sets* (`select_cases.py TRUTH_ACTIONS`): rings {escalate, decline_block}, ATO
   {decline_block, escalate} (§3's $2k aggregate line), promo clusters {hold_contact,
   escalate} (§3: clusters are suspected rings). Strict single answers remain where
   policy is unambiguous: stolen/never-pay → decline_block, INR → hold_contact,
   bust-out → escalate, benign → clear.
2. **The verifier punished legitimate rounding.** Packets carried unrounded floats
   (`avg_amount_prior: 101.11857…`); memos citing "$101.12" were flagged as
   hallucinations. The verifier now accepts a numeric token when some packet number
   rounds to it at the token's own precision. Genuinely *computed* values (sums,
   ratios not present in the packet) remain unsupported — e.g. a memo totaling four
   orders into "$2,421.00" is still flagged, by design.

Both corrections are visible in git history; v1 numbers below are post-correction.

## memo_v1 — baseline

Design: full FP-1 context in prompt; packet as JSON; JSON-only output contract with
field-by-field schema; benign-mimic checklist (travelers/movers/gift/hardship/typos/new
phones) included verbatim; instruction that every signal must quote a packet fact.

Hypothesis: a policy-grounded prompt with an explicit benign-hypothesis requirement
should already give high action accuracy on clear-cut patterns; expected weak spots were
(a) the hold_contact/decline_block boundary and (b) hallucinated derived numbers.

Results (post-correction):

| metric | claude-sonnet-5 (N=200) |
|---|---|
| action accuracy (in-policy set) | **61.0%** |
| decline precision / recall | 93.1% / 47.4% |
| pattern identification | 80.5% |
| hallucination rate (memo / claim) | 10.0% / 1.05% |
| invalid citations | 0 memos |
| consistency (3 perturbed runs, 50 cases) | 87.2% |
| schema failures | 0 |
| latency p50 | 56.5 s |


Failure taxonomy from reading the misses raw (n=140 preliminary read, confirmed on the
full run):

- **Hedging is the dominant failure, not misdiagnosis.** Pattern-identification ran
  ~0.81 while action accuracy lagged far behind it; the confusion matrix concentrates
  in `clear → hold_contact` (32 of 42 benign misses) and `decline_block →
  hold_contact`. The model treats hold_contact as a safe default; FP-1 prices customer
  friction, so it isn't one.
- **Decline recall is the weak dangerous-direction number** (~0.45 preliminary): the
  model correctly names stolen_card/ATO then still recommends holding.
- **Residual hallucinations are self-computed aggregates** (order-total sums the packet
  never stated). The instruction "quote a packet fact" did not stop arithmetic.

## memo_v2 — action-calibration + numbers discipline

Change hypothesis: v1's failures are calibration failures, so v2 adds exactly two
blocks and changes nothing else (isolating the lever): (1) an explicit ACTION SELECTION
procedure — escalate criteria (linkage ≥3 / merchant complicity / ≥$2k aggregate),
decline_block requires a §5 pattern at high likelihood with ≥2 corroborating signal
*families*, clear when every fired rule is explained by a §6 mimic within noise rates
("do NOT hold what you can clear"), hold_contact only when a step-up would actually
decide; (2) NUMBERS DISCIPLINE — copy values verbatim, never compute, describe derived
quantities qualitatively.

Predicted effects: action accuracy and decline recall rise materially; benign accuracy
(clear-rate on hard negatives) rises most; hallucination rate drops toward the
claim-level floor; pattern-id rate unchanged (diagnosis was never the problem).

Results:

| metric | sonnet-5 (N=200) | gpt-5.6-luna (N=199¹) | gpt-5.6-terra (N=86²) |
|---|---|---|---|
| action accuracy | **73.5%** | 59.8% | **76.7%** |
| decline precision / recall | 96.8% / 52.6% | 100% / 37.5% | 93.3% / 70.0% |
| pattern identification | 75.0% | 65.3% | 72.1% |
| hallucination (memo) | **0.0%** | 0.0% | 0.0% |
| consistency (perturbed) | 84.0% | — | — |
| schema failure rate | 0.0% | 0.5% | 0.0% |
| latency p50 | 58.3 s | 39.1 s | 27.7 s |

¹ one Luna response was truncated JSON — scored as a schema failure, not dropped
silently. ² the Terra run was cut short by a provider quota; reported on its
contiguous 86-case prefix and labeled as such. Cost columns are omitted: CLI
backends don't expose reliable token accounting; latency is measured directly.


Delta and verdict:

sonnet v1 → v2 on identical cases: action accuracy **+12.5pp**, decline
precision **+3.7pp**, decline recall **+5.2pp**, hallucinations **10% → 0%**.
Two honest regressions: pattern-id −5.5pp and consistency −3.2pp — the action
procedure spends attention the free-form hypothesis step previously used, and a
stricter decision boundary flips more actions under perturbation. Verdict: v2
adopted (the decision metrics are the product; diagnosis remains strong), with
the regression logged as the target for a future v3. Cross-model read: terra
leads accuracy and recall at the lowest latency (partial n), sonnet leads
precision and is the only arm with full consistency data, luna is maximally
conservative (perfect decline precision, weakest recall) — arm choice is a
config switch (`llm.tasks.triage_memo.model`), and this table is how it should
be made.

## Manual spot-check protocol

Alongside the mechanical verifier, each version gets a 30-case human read: sample 15
flagged-unsupported and 15 clean memos, record verifier false alarms (legitimate
derived arithmetic, rounding) and false passes (fluent claims built from real tokens
that misstate the packet). The v1 read is what produced verifier correction #2 above.
