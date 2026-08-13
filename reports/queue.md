# Alert queue / SLA simulation (from 2026-04-01)

Operating point: review ≥ 30 → 2119 alerts (24.9/day). Two analysts on offset 5-day shifts, 6.5 productive h, lognormal service (arithmetic mean 7.0 min), orders ship 12h after checkout; SLA target 4 business hours.

| policy | SLA ≤ 4bh | P50 ttd (bh) | P90 ttd (bh) | max backlog | fraud-$ blocked pre-ship | fraud alerts blocked |
|---|---|---|---|---|---|---|
| fifo | 99.9% | 0.53 | 1.82 | 25 | $30,246 | 147/189 |
| score | 99.9% | 0.49 | 1.9 | 25 | $31,350 | 151/189 |
| llm | 99.9% | 0.49 | 1.9 | 25 | $31,350 | 151/189 |

_LLM priorities cover 2% of alerts (the frozen eval set); uncovered alerts fall back to score order, so at this coverage the llm policy tracks score-priority — the comparison becomes meaningful when triage runs on the full stream._

The configured workload is far below saturation (36.7% utilization of union shift coverage), so high SLA attainment is expected.

Reading: the 12-hour fulfillment race means weekend/coverage gaps, not average throughput, decide how much fraud ships. Score-priority beats FIFO by resolving high-score (fraud-dense) alerts inside the ship window even when the backlog spans a coverage hole; the offset-shift pairing leaves parts of the week single-covered, visible as the weekly backlog sawtooth.

```json
{"policy": "fifo", "n_alerts": 2119, "sla_attainment": 0.999, "ttd_p50_h": 0.53, "ttd_p90_h": 1.82, "max_backlog": 25, "fraud_blocked_usd": 30246.0, "fraud_blocked_n": 147}
{"policy": "score", "n_alerts": 2119, "sla_attainment": 0.999, "ttd_p50_h": 0.49, "ttd_p90_h": 1.9, "max_backlog": 25, "fraud_blocked_usd": 31350.0, "fraud_blocked_n": 151}
{"policy": "llm", "n_alerts": 2119, "sla_attainment": 0.999, "ttd_p50_h": 0.49, "ttd_p90_h": 1.9, "max_backlog": 25, "fraud_blocked_usd": 31350.0, "fraud_blocked_n": 151}
```
