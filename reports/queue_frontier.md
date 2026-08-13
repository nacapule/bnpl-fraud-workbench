# Queue staffing frontier

The single-point run in `reports/queue.md` puts the configured roster against the observed holdout stream (24.9 alerts/day) at 36.7% utilization of union shift coverage. Its 99.9% SLA attainment could not have come out otherwise, so it is not a finding. The frontier below is.

Volume is scaled by replicating the holdout alert stream: at ×k every alert appears k times at its own timestamp, so the weekly and diurnal arrival shape is preserved and only intensity changes. Rosters reuse the two configured analysts verbatim and extend the same pattern — each further analyst covers five consecutive days starting two weekdays later, alternating between the two configured start hours. Dispatch is score-priority in every cell, service times are drawn from the same seeded stream (seed 416), and the ×1 / 2-analyst cell reproduces the score-priority row of `reports/queue.md` exactly.

## SLA attainment (decision within 4 business hours)

| arrivals | 1 analyst | 2 analysts | 3 analysts | 4 analysts |
|---|---|---|---|---|
| ×1 (2,119 alerts) | 74.7% | 99.9% ✓ | 99.8% ✓ | 99.8% ✓ |
| ×2 (4,238 alerts) | 28.8% | 91.2% ✓ | 98.7% ✓ | 99.9% ✓ |
| ×4 (8,476 alerts) | 18.6% | 30.4% | 72.5% | 87.2% |
| ×8 (16,952 alerts) | 3.8% | 18.8% | 28.1% | 29.4% |

✓ marks cells at or above the configured 90% target: ×1 needs 2 analysts, ×2 needs 2 analysts, ×4 needs more than 4 analysts, ×8 needs more than 4 analysts.

Attainment runs on the union-of-shifts clock, so a larger roster widens the clock as well as the capacity. Where the queue is already empty that can cost a tenth of a point: the ×1 row slips from 99.9% at two analysts to 99.8% at three while median time-to-decision falls from 0.49 to 0.28 business hours.

## Fraud dollars blocked before fulfillment

| arrivals | 1 analyst | 2 analysts | 3 analysts | 4 analysts |
|---|---|---|---|---|
| ×1 (2,119 alerts) | $19,546 (52%) | $31,350 (83%) | $33,019 (87%) | $33,654 (89%) |
| ×2 (4,238 alerts) | $32,384 (43%) | $58,203 (77%) | $62,370 (82%) | $64,215 (85%) |
| ×4 (8,476 alerts) | $55,898 (37%) | $90,116 (60%) | $114,639 (76%) | $127,994 (84%) |
| ×8 (16,952 alerts) | $49,890 (16%) | $135,693 (45%) | $181,923 (60%) | $195,461 (65%) |

Percentages are the share of fraud dollars at stake in that stream. Replication multiplies the exposure as well as the workload, so dollar figures compare down a column, not across a row; the share compares in both directions.

## Marginal fraud dollars per added analyst

| arrivals | 2nd analyst | 3rd analyst | 4th analyst |
|---|---|---|---|
| ×1 | +$11,804 | +$1,669 | +$635 |
| ×2 | +$25,819 | +$4,167 | +$1,845 |
| ×4 | +$34,218 | +$24,523 | +$13,355 |
| ×8 | +$85,803 | +$46,230 | +$13,538 |

Fraud dollars flatten only where the queue is not capacity-bound: at ×1 the third and fourth add $1,669 and $635 against the second analyst's $11,804; at ×2 the third and fourth add $4,167 and $1,845 against the second analyst's $25,819. Where the SLA target cannot be held they have not flattened at all — ×4 is still at 1.21 utilization with a fourth analyst worth $13,355, ×8 is still at 2.42 utilization with a fourth analyst worth $13,538 — so for those volumes the sweep's answer is that this roster is too small to price, not that another analyst is poor value.

Headcount also runs into a ceiling that has nothing to do with capacity. At ×1 with 4 analysts the queue is 30% utilized and 99.8% of alerts are decided inside the SLA, yet $4,193 of fraud exposure (11%) still ships. Of those, 17 alerts carrying $1,289 arrive after the last shift start of the day: no analyst comes on duty inside their 12-hour fulfillment window, so no roster of this shape resolves them in time whatever its size. That residue is a shift-design question, not a staffing one.

The two surfaces therefore answer different questions. SLA attainment is an average over every alert and degrades smoothly with load, so it prices the analyst experience; blocked fraud dollars depend on where a small number of high-exposure alerts fall relative to coverage boundaries, so they price the loss. Staffing to one is not staffing to the other.

```json
{"analysts": 1, "arrival_multiplier": 1, "fraud_alerts": 189, "fraud_at_risk_usd": 37847.0, "fraud_blocked_n": 94, "fraud_blocked_share": 0.516, "fraud_blocked_usd": 19546.0, "max_backlog": 79, "n_alerts": 2119, "sla_attainment": 0.747, "sla_target_met": false, "ttd_p50_h": 1.56, "ttd_p90_h": 6.17, "utilization": 0.613}
{"analysts": 2, "arrival_multiplier": 1, "fraud_alerts": 189, "fraud_at_risk_usd": 37847.0, "fraud_blocked_n": 151, "fraud_blocked_share": 0.828, "fraud_blocked_usd": 31350.0, "max_backlog": 25, "n_alerts": 2119, "sla_attainment": 0.999, "sla_target_met": true, "ttd_p50_h": 0.49, "ttd_p90_h": 1.9, "utilization": 0.367}
{"analysts": 3, "arrival_multiplier": 1, "fraud_alerts": 189, "fraud_at_risk_usd": 37847.0, "fraud_blocked_n": 160, "fraud_blocked_share": 0.872, "fraud_blocked_usd": 33019.0, "max_backlog": 25, "n_alerts": 2119, "sla_attainment": 0.998, "sla_target_met": true, "ttd_p50_h": 0.28, "ttd_p90_h": 1.35, "utilization": 0.332}
{"analysts": 4, "arrival_multiplier": 1, "fraud_alerts": 189, "fraud_at_risk_usd": 37847.0, "fraud_blocked_n": 164, "fraud_blocked_share": 0.889, "fraud_blocked_usd": 33654.0, "max_backlog": 22, "n_alerts": 2119, "sla_attainment": 0.998, "sla_target_met": true, "ttd_p50_h": 0.23, "ttd_p90_h": 1.24, "utilization": 0.303}
{"analysts": 1, "arrival_multiplier": 2, "fraud_alerts": 378, "fraud_at_risk_usd": 75695.0, "fraud_blocked_n": 171, "fraud_blocked_share": 0.428, "fraud_blocked_usd": 32384.0, "max_backlog": 815, "n_alerts": 4238, "sla_attainment": 0.288, "sla_target_met": false, "ttd_p50_h": 48.07, "ttd_p90_h": 101.08, "utilization": 1.227}
{"analysts": 2, "arrival_multiplier": 2, "fraud_alerts": 378, "fraud_at_risk_usd": 75695.0, "fraud_blocked_n": 294, "fraud_blocked_share": 0.769, "fraud_blocked_usd": 58203.0, "max_backlog": 59, "n_alerts": 4238, "sla_attainment": 0.912, "sla_target_met": true, "ttd_p50_h": 1.34, "ttd_p90_h": 3.87, "utilization": 0.735}
{"analysts": 3, "arrival_multiplier": 2, "fraud_alerts": 378, "fraud_at_risk_usd": 75695.0, "fraud_blocked_n": 313, "fraud_blocked_share": 0.824, "fraud_blocked_usd": 62370.0, "max_backlog": 51, "n_alerts": 4238, "sla_attainment": 0.987, "sla_target_met": true, "ttd_p50_h": 0.63, "ttd_p90_h": 2.79, "utilization": 0.664}
{"analysts": 4, "arrival_multiplier": 2, "fraud_alerts": 378, "fraud_at_risk_usd": 75695.0, "fraud_blocked_n": 324, "fraud_blocked_share": 0.848, "fraud_blocked_usd": 64215.0, "max_backlog": 45, "n_alerts": 4238, "sla_attainment": 0.999, "sla_target_met": true, "ttd_p50_h": 0.45, "ttd_p90_h": 2.47, "utilization": 0.605}
{"analysts": 1, "arrival_multiplier": 4, "fraud_alerts": 756, "fraud_at_risk_usd": 151389.0, "fraud_blocked_n": 308, "fraud_blocked_share": 0.369, "fraud_blocked_usd": 55898.0, "max_backlog": 4989, "n_alerts": 8476, "sla_attainment": 0.186, "sla_target_met": false, "ttd_p50_h": 374.4, "ttd_p90_h": 543.73, "utilization": 2.454}
{"analysts": 2, "arrival_multiplier": 4, "fraud_alerts": 756, "fraud_at_risk_usd": 151389.0, "fraud_blocked_n": 506, "fraud_blocked_share": 0.595, "fraud_blocked_usd": 90116.0, "max_backlog": 1688, "n_alerts": 8476, "sla_attainment": 0.304, "sla_target_met": false, "ttd_p50_h": 72.28, "ttd_p90_h": 175.78, "utilization": 1.469}
{"analysts": 3, "arrival_multiplier": 4, "fraud_alerts": 756, "fraud_at_risk_usd": 151389.0, "fraud_blocked_n": 592, "fraud_blocked_share": 0.757, "fraud_blocked_usd": 114639.0, "max_backlog": 106, "n_alerts": 8476, "sla_attainment": 0.725, "sla_target_met": false, "ttd_p50_h": 2.31, "ttd_p90_h": 5.46, "utilization": 1.327}
{"analysts": 4, "arrival_multiplier": 4, "fraud_alerts": 756, "fraud_at_risk_usd": 151389.0, "fraud_blocked_n": 632, "fraud_blocked_share": 0.845, "fraud_blocked_usd": 127994.0, "max_backlog": 92, "n_alerts": 8476, "sla_attainment": 0.872, "sla_target_met": false, "ttd_p50_h": 1.43, "ttd_p90_h": 4.24, "utilization": 1.21}
{"analysts": 1, "arrival_multiplier": 8, "fraud_alerts": 1512, "fraud_at_risk_usd": 302778.0, "fraud_blocked_n": 295, "fraud_blocked_share": 0.165, "fraud_blocked_usd": 49890.0, "max_backlog": 13442, "n_alerts": 16952, "sla_attainment": 0.038, "sla_target_met": false, "ttd_p50_h": 871.87, "ttd_p90_h": 1441.14, "utilization": 4.908}
{"analysts": 2, "arrival_multiplier": 8, "fraud_alerts": 1512, "fraud_at_risk_usd": 302778.0, "fraud_blocked_n": 860, "fraud_blocked_share": 0.448, "fraud_blocked_usd": 135693.0, "max_backlog": 10070, "n_alerts": 16952, "sla_attainment": 0.188, "sla_target_met": false, "ttd_p50_h": 636.77, "ttd_p90_h": 910.69, "utilization": 2.939}
{"analysts": 3, "arrival_multiplier": 8, "fraud_alerts": 1512, "fraud_at_risk_usd": 302778.0, "fraud_blocked_n": 991, "fraud_blocked_share": 0.601, "fraud_blocked_usd": 181923.0, "max_backlog": 6747, "n_alerts": 16952, "sla_attainment": 0.281, "sla_target_met": false, "ttd_p50_h": 335.98, "ttd_p90_h": 462.48, "utilization": 2.655}
{"analysts": 4, "arrival_multiplier": 8, "fraud_alerts": 1512, "fraud_at_risk_usd": 302778.0, "fraud_blocked_n": 1058, "fraud_blocked_share": 0.646, "fraud_blocked_usd": 195461.0, "max_backlog": 3361, "n_alerts": 16952, "sla_attainment": 0.294, "sla_target_met": false, "ttd_p50_h": 94.64, "ttd_p90_h": 216.9, "utilization": 2.421}
{"analysts": 4, "fraud_at_risk_usd": 37847.0, "unreachable_alerts": 17, "unreachable_usd": 1289.0}
```
