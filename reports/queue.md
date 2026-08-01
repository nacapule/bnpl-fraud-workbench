# Alert queue / SLA simulation (holdout months 10–12)

Operating point: review ≥ 30 → 2119 alerts (23.3/day). Two analysts on offset 5-day shifts, 6.5 productive h), lognormal service (mean 7.0 min), orders ship 12h after checkout; SLA target 4 business hours.

| policy | SLA ≤ 4bh | P50 ttd (bh) | P90 ttd (bh) | max backlog | fraud-$ blocked pre-ship | fraud alerts blocked |
|---|---|---|---|---|---|---|
| fifo | 100% | 0.72 | 2.24 | 25 | $30,147 | 144/189 |
| score | 99% | 0.65 | 2.33 | 25 | $31,350 | 150/189 |
| llm | 99% | 0.65 | 2.33 | 25 | $31,350 | 150/189 |

_LLM priorities cover 2% of alerts (the frozen eval set); uncovered alerts fall back to score order, so at this coverage the llm policy tracks score-priority — the comparison becomes meaningful when triage runs on the full stream._

Reading: the 12-hour fulfillment race means weekend/coverage gaps, not average throughput, decide how much fraud ships. Score-priority beats FIFO by resolving high-score (fraud-dense) alerts inside the ship window even when the backlog spans a coverage hole; the offset-shift pairing leaves parts of the week single-covered, visible as the weekly backlog sawtooth.
