# Prompt iteration log — triage memo task

The eval harness (`harness.py`) is the referee: no prompt change ships without a
before/after metric table on the frozen eval set (`cases.json`, committed packets,
committed response cache — anyone can re-run `python -m llm.eval.harness --offline` and
get these exact numbers).

## Metric definitions (fixed across versions)

| metric | definition |
|---|---|
| action accuracy | `recommended_action` ∈ the policy-derived acceptable-action set for the true pattern (`select_cases.py TRUTH_ACTIONS`; single-action wherever FP-1 is unambiguous — see the corrections section) |
| decline precision / recall | positive class = `decline_block` only; escalate does NOT count as a decline. Precision protects legitimate customers (insults), recall protects loss. |
| pattern id rate | top hypothesis (highest likelihood; ties → first listed) matches true pattern |
| non-verbatim concrete-token rate | share of memos/fields where a numeric, id, timestamp, or money token in `signals_observed` is not present at a token boundary in the packet; includes derived and unsupported fields |
| unsupported concrete-token rate | share of memos/fields still unsupported after visible packet-list counts and sums are classified as derived; this is a token check, not semantic-truth verification |
| citation validity | cited rule ids exist in FP-1; `citation_fired_rate` = share of cited rules that actually fired on the alert |
| consistency | target: 3 runs on 50 cases with a neutral numbered no-op line injected into the packet; report `consistency_n` as the number with all three probes cached and valid, plus all-3-agree rate on `recommended_action` |
| latency | wall-clock per call; cost columns omitted (CLI backends lack reliable token accounting) |

A future manual spot-check can sample flagged and clean memos to measure false alarms
and false passes. No annotation set is committed, so no human-verified grounding metric
is reported here.

## Harness corrections logged during v1 (applied to all versions/arms equally)

The replay exposed three defects in the evaluator:

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
3. **Substring matching passed digits inside longer values.** A short count such as
   `7` could match `1370.55`. The current verifier uses numeric token boundaries and
   classifies checked signal fields as verbatim, derived from a visible packet-list
   count/sum, or unsupported. The scope is concrete tokens in `signals_observed`; it
   does not verify the semantic relationship asserted by the prose.

The current tables use all three corrections. The superseded substring series is kept
as a labeled row for auditability.

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
| non-verbatim concrete tokens (memo / checked field) | **45.0% / 5.21%** |
| unsupported after derived-list classification (memo / checked field) | **33.5% / 3.35%** |
| legacy substring “hallucination” metric — superseded (memo / claim) | 10.0% / 1.05% |
| invalid citations | 0 memos |
| consistency (3 perturbed runs) | 87.2% (41/47 agree; 47 cases had all probes cached) |
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
- **Residual non-verbatim fields are often self-computed aggregates.** The derived
  category separates valid visible-list arithmetic from unsupported numbers.

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
(clear-rate on hard negatives) rises most; unsupported-token rate drops toward the
claim-level floor; pattern-id rate unchanged (diagnosis was never the problem).

Results:

| metric | sonnet-5 (N=200) | gpt-5.6-luna (N=199¹) | gpt-5.6-terra (N=86²) |
|---|---|---|---|
| action accuracy | **73.5%** | 59.8% | **76.7%** |
| decline precision / recall | 96.8% / 52.6% | 100% / 37.5% | 93.3% / 70.0% |
| pattern identification | 75.0% | 65.3% | 72.1% |
| non-verbatim concrete tokens (memo) | **13.0%** | 1.5% | 1.2% |
| unsupported after derived-list classification (memo) | **4.5%** | 1.5% | 1.2% |
| legacy substring “hallucination” metric — superseded (memo) | 0.0% | 0.0% | 0.0% |
| consistency (perturbed) | 84.0% (42/50) | — | — |
| schema failure rate | 0.0% | 0.5% | 0.0% |
| latency p50 | 58.3 s | 39.1 s | 27.7 s |

¹ one Luna response was truncated JSON. Valid-output action accuracy is 119/199
(59.8%); counting that schema failure as wrong gives 119/200 (59.5%). ² the Terra run
was cut short by a provider quota; its contiguous 86-case prefix reproduces offline with
`python -m llm.eval.harness --offline --no-consistency --arms gpt-5.6-terra --n 86`.
Cost columns are omitted: CLI
backends don't expose reliable token accounting; latency is measured directly.


Delta and verdict:

sonnet v1 → v2 on identical cases: action accuracy **+12.5pp**, decline
precision **+3.7pp**, decline recall **+5.2pp**.
The superseded substring metric moved 10% → 0%; under token-boundary matching,
non-verbatim memo fields moved **45% → 13%**, and unsupported memo fields after
derived-list classification moved **33.5% → 4.5%**.
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

## Frozen-packet temporal note

Future packet builds use alert-time predicates for linkage and installment state and an
as-of category median. The 200 frozen v1 packets predate that fix and remain unchanged
so the cached evaluation stays valid: 59/200 contain all-time linkage values, with a
worst case of 39 versus 13 on alert 2655, and 3 packets include an installment paid
after the alert. These limitations apply to every arm in the frozen comparison.
