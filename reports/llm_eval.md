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
