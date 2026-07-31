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

## memo_v1 — baseline

Design: full FP-1 context in prompt; packet as JSON; JSON-only output contract with
field-by-field schema; benign-mimic checklist (travelers/movers/gift/hardship/typos/new
phones) included verbatim; instruction that every signal must quote a packet fact.

Hypothesis: a policy-grounded prompt with an explicit benign-hypothesis requirement
should already give high action accuracy on clear-cut patterns; expected weak spots are
(a) hold_contact vs decline_block boundary (INR/promo), (b) escalate under-use on
merchant bust-out (order-level packets make merchant-level evidence indirect), and
(c) hallucinated derived numbers.

Results: (filled by harness after the v1 run — see results/memo_v1__*.json)

## memo_v2 — (planned after v1 failure analysis)

Change hypothesis: TBD from reading v1 misses raw. Candidate levers, to be selected by
evidence: hypothesis-first ordering (list hypotheses before signals to reduce
anchor-on-amount), explicit escalate criteria block, tightened quoting rule ("copy
values verbatim; do not compute"), merchant-context line in packets for bust-out cases.

Results: TBD
