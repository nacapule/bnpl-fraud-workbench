# LLM triage eval

prompt version: **memo_v2** · requested cases: 200

Grounding scope: concrete numeric, entity-id, timestamp, and money tokens in `signals_observed`; token presence is not semantic-truth verification.

| metric | claude-sonnet-5 | gpt-5.6-luna |
|---|---|---|
| n_cases | 200 | 199 |
| cache_misses | 0 | 50 |
| schema_failures | 0 | 1 |
| action_accuracy | 0.735 | 0.598 |
| decline_precision | 0.968 | 1.0 |
| decline_recall | 0.526 | 0.375 |
| pattern_id_rate | 0.75 | 0.653 |
| non_verbatim_claim_rate_memo | 0.13 | 0.015 |
| non_verbatim_claim_rate_checked_fields | 0.0142 | 0.0014 |
| unsupported_claim_rate_memo | 0.045 | 0.015 |
| unsupported_claim_rate_checked_fields | 0.0036 | 0.0014 |
| derived_claim_rate_checked_fields | 0.0107 | 0.0 |
| consistency_action_agreement | 0.84 | None |
| consistency_n | 50 | 0 |
| latency_p50_ms | 58275 | 39105 |

Decision accuracy denominators:
- claude-sonnet-5: 0.735 over 200 valid outputs; 0.735 over 200 outputs with schema failures counted as wrong.
- gpt-5.6-luna: 0.598 over 199 valid outputs; 0.595 over 200 outputs with schema failures counted as wrong.

## Action confusion

Rows are the labelled truth action, columns the recommended action. Accuracy above is scored against the policy-derived acceptable set in `cases.json`, which is wider than the single truth label, so off-diagonal cells are not all errors.

### claude-sonnet-5

| truth \ recommended | clear | hold_contact | escalate | decline_block |
|---|---|---|---|---|
| clear | 38 | 19 | 2 | 1 |
| hold_contact | 1 | 21 | 23 | 0 |
| escalate | 0 | 3 | 35 | 0 |
| decline_block | 0 | 27 | 0 | 30 |

124 on the diagonal against 147 scored correct: the remaining 23 recommendations differ from the labelled truth action but sit inside its acceptable set.

### gpt-5.6-luna

| truth \ recommended | clear | hold_contact | escalate | decline_block |
|---|---|---|---|---|
| clear | 19 | 38 | 3 | 0 |
| hold_contact | 1 | 21 | 23 | 0 |
| escalate | 0 | 3 | 35 | 0 |
| decline_block | 0 | 35 | 0 | 21 |

96 on the diagonal against 119 scored correct: the remaining 23 recommendations differ from the labelled truth action but sit inside its acceptable set.

## Accuracy by truth pattern

Pattern identification is the share of memos whose highest-likelihood hypothesis names the injected pattern. Strata are small — see `reports/uncertainty.md` for the intervals.

### claude-sonnet-5

| truth pattern | n | action accuracy | pattern identification |
|---|---|---|---|
| account_takeover | 28 | 0.786 | 1.0 |
| benign | 60 | 0.633 | 0.8 |
| inr_abuse | 22 | 0.955 | 0.864 |
| merchant_bustout | 3 | 0.0 | 0.0 |
| never_pay | 7 | 0.0 | 0.0 |
| promo_abuse | 23 | 1.0 | 0.0 |
| stolen_card | 22 | 0.364 | 0.909 |
| synthetic_ring | 35 | 1.0 | 1.0 |

### gpt-5.6-luna

| truth pattern | n | action accuracy | pattern identification |
|---|---|---|---|
| account_takeover | 27 | 0.519 | 1.0 |
| benign | 60 | 0.317 | 0.433 |
| inr_abuse | 22 | 0.955 | 0.909 |
| merchant_bustout | 3 | 0.0 | 0.0 |
| never_pay | 7 | 0.0 | 0.0 |
| promo_abuse | 23 | 1.0 | 0.0 |
| stolen_card | 22 | 0.318 | 1.0 |
| synthetic_ring | 35 | 1.0 | 1.0 |
