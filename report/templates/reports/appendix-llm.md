# LLM appendix

The memo drafter writes an advisory memo for an analyst from one case packet: claims
tied to packet fields, competing explanations (one of them benign), a suggested
disposition, the fraud-policy clauses it rests on, the next check and a short memo. It
decides nothing. Its benchmark asks whether those memos follow fraud policy FP-2 on
analyst decisions drawn from the synthetic worlds. The protocol,
[`llm/eval/PROTOCOL.md`](../llm/eval/PROTOCOL.md), was fixed before any final case
existed; this appendix reports its endpoints as it registers them. Every figure below is
read from the benchmarks' results files, the final cohort's in
[`llm/eval/benchmarks/2026-10-final/results.json`](../llm/eval/benchmarks/2026-10-final/results.json).

## The benchmark

The final cohort holds {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/review/unweighted | denominator }} review decisions, taken when an analyst first picks up an order, and {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/check_completed/unweighted | denominator }} decisions after a check has answered. They come from the test windows of the baseline world of every final seed, under today's rules. The cases fall into {{ bench:2026-10-final:/statistics/clusters | count }} clusters, each a linked group of accounts and episodes, with at most {{ config:llm:benchmark.max_cases_per_cluster }} cases from any one group. Each case's weight is the inverse of its selection probability. Of the cases, {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/action | denominator }} are asked again, once with their facts shuffled and once with fresh placeholder names.

The benchmark's arms, `opus` and `sol`, answer the same packets with the same prompt, `memo_fp2_v2`, and the same isolation. Each call is one model turn, and each arm's command-line tool is pinned at the benchmark's first live call:

| Arm | Model | Effort | Through | Pinned version |
| --- | --- | --- | --- | --- |
| `opus` | {{ bench:2026-10-final:/pins/opus/model }} | {{ bench:2026-10-final:/pins/opus/effort }} | the Claude Code CLI | {{ bench:2026-10-final:/pins/opus/cli_version }} |
| `sol` | {{ bench:2026-10-final:/pins/sol/model }} | {{ bench:2026-10-final:/pins/sol/effort }} | the Codex CLI | {{ bench:2026-10-final:/pins/sol/cli_version }} |

The referee ([`llm/referee.py`](../llm/referee.py)) scores each memo against FP-2 with the
same evidence code the simulated analyst uses. The verifier
([`llm/eval/verifier.py`](../llm/eval/verifier.py)) checks every structured claim, with
its field, value and any declared calculation, against the packet.

## The primary endpoint: complete-memo pass rate

A memo passes only if all of these hold:
- its format is valid;
- its disposition is acceptable, either the policy's standard or one it also permits;
- its next check is the one the policy requires;
- its citations are valid: no unknown clause, no rule that does not hold, the
  disposition's own clause for anything but a clear, and no non-payment clause when no
  installment is due;
- the verifier finds no claim error.

A memo that fails to arrive counts as failing. The rate is reported as the natural-mix rate. Within each axis it is the selection-weighted rate, and the axes are weighted by their shares of the eligible decisions: {{ bench:2026-10-final:/axes/review | pct }} review decisions and {{ bench:2026-10-final:/axes/check_completed | pct }} decisions after a check. The intervals are percentile cluster-bootstrap intervals, which resample whole clusters. The unweighted rate, its Wilson interval and each axis's rates go beside it.

| Complete-memo pass | `opus` | `sol` |
| --- | --- | --- |
| **Natural-mix rate (primary)** | **{{ bench:2026-10-final:/endpoints/primary/arms/opus/natural_mix | pct }}** | **{{ bench:2026-10-final:/endpoints/primary/arms/sol/natural_mix | pct }}** |
| its cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/primary/arms/opus/natural_mix_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/natural_mix_cluster_bootstrap | bounds }} |
| Unweighted rate | {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted | of }} ({{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted | pct }}) | {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted | of }} ({{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted | pct }}) |
| its cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted_cluster_bootstrap | bounds }} |
| its Wilson interval | {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted_wilson | bounds }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted_wilson | bounds }} |
| Review decisions, unweighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/review/unweighted | of }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/review/unweighted | of }} |
| Review decisions, weighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/review/weighted | pct }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/review/weighted | pct }} |
| After a check, unweighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/check_completed/unweighted | of }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/check_completed/unweighted | of }} |
| After a check, weighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/check_completed/weighted | pct }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/check_completed/weighted | pct }} |

## What failed

`sol` passed {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted | of }} memos, and every memo it failed, it failed on citations alone. Its dispositions were acceptable in {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/acceptable | of }} cases and its next checks right in {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/next_check_ok | of }}, and the verifier found no error in its {{ bench:2026-10-final:/arms/sol/summary/claim_errors | denominator }} claims. Apart from one memo that cited §4.4, a clause FP-2 does not define, the failing memos listed among their citations a rule that does not hold on the packet: R03 on orders `sol` cleared as the policy requires, R01 beside a `needs_check`, and R02, R08 and R10 together on another clear.

`opus` passed {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted | of }}. One of its failing answers held a complete memo, then a note correcting it and a second JSON object, so it was not a single JSON object and could not be scored. In the other, a derived claim declared its value as a difference of `context.accounts_on_address_30d` and `decision.checks`. The second is the list of completed checks, not a number, so the verifier could not recompute the claim and counted it as an error.

## The paired comparison

The protocol registers a paired comparison of the two arms on the same cases, as a secondary endpoint. It reports the counts, the unweighted and natural-mix differences (`opus` minus `sol`), each with a cluster-bootstrap interval that resamples the same clusters for both arms, and a sign test over clusters.

| On the same cases | Complete-memo pass | Acceptable disposition |
| --- | --- | --- |
| Both arms passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/both | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/both | count }} |
| Only `opus` passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/only_first | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/only_first | count }} |
| Only `sol` passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/only_second | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/only_second | count }} |
| Neither passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/neither | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/neither | count }} |
| Unweighted difference | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/difference | pp }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/difference | pp }} |
| its cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/difference_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/difference_cluster_bootstrap | bounds }} |
| Natural-mix difference | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/natural_difference | pp }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/natural_difference | pp:2 }} |
| its cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/natural_difference_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/natural_difference_cluster_bootstrap | bounds:2 }} |
| Clusters favouring `opus`, favouring `sol`, tied | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/clusters_favouring_first | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/clusters_favouring_second | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/clusters_tied | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/clusters_favouring_first | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/clusters_favouring_second | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/clusters_tied | count }} |
| Sign test over clusters | p = {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/cluster_sign_test_p | num:2 }} | p = {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/cluster_sign_test_p | num:2 }} |

The comparison of complete-memo pass rates is inconclusive. The interval of the unweighted difference includes zero, and the sign test over clusters gives p = {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/cluster_sign_test_p | num:2 }}. Had the arms tied, that would not have shown them to be equivalent either.

## Secondary endpoints

**Components.** Each part of the complete-memo pass, and the claims with an error among all the claims each arm made:

| Component | `opus` | `sol` |
| --- | --- | --- |
| Format valid | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/format_valid | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/format_valid | of }} |
| Disposition acceptable | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/acceptable | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/acceptable | of }} |
| Next check right | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/next_check_ok | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/next_check_ok | of }} |
| Citations valid | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/citations_valid | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/citations_valid | of }} |
| No claim error | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/no_claim_error | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/no_claim_error | of }} |
| Claims with an error | {{ bench:2026-10-final:/arms/opus/summary/claim_errors | of }} | {{ bench:2026-10-final:/arms/sol/summary/claim_errors | of }} |

**Acceptable rate.** The share of memos whose disposition the policy accepts, with the same weighting and intervals as the primary endpoint:

| Acceptable disposition | `opus` | `sol` |
| --- | --- | --- |
| Natural-mix rate | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/natural_mix | pct:2 }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/natural_mix | pct:2 }} |
| its cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/natural_mix_cluster_bootstrap | bounds:2 }} | none: {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/degenerate }} |
| Unweighted rate | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/unweighted | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/unweighted | of }} |
| its cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/unweighted_cluster_bootstrap | bounds }} | none: {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/degenerate }} |
| its Wilson interval | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/unweighted_wilson | bounds }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/unweighted_wilson | bounds }} |

When every case passes, every bootstrap resample gives the same rate, so the protocol reports no bootstrap interval.

**Standard-action agreement.** The memo's action matched the referee's standard action in {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/rate | of }} cases for `opus` and {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/rate | of }} for `sol`. A `needs_check` counts as the hold when it names a required check that has not run yet (FP-2 §4.3). The table pairs the referee's standard disposition with the memo's:

| Standard disposition, memo's disposition | `opus` | `sol` |
| --- | --- | --- |
| clear, clear | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/clear -> clear | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/clear -> clear | count }} |
| hold, `needs_check` | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/hold -> needs_check | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/hold -> needs_check | count }} |
| hold, hold | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/hold -> hold | count }} | none |
| decline, decline | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/decline -> decline | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/decline -> decline | count }} |
| decline, no scorable memo | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/decline -> failure | count }} | none |
| escalate, escalate | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/escalate -> escalate | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/escalate -> escalate | count }} |

**By policy row.** The acceptable and complete-memo pass rates for each row of FP-2's decision table:

| Policy row | `opus` acceptable | `opus` pass | `sol` acceptable | `sol` pass |
| --- | --- | --- | --- | --- |
| §6.6(c), no adverse rule family | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(c)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(c)/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(c)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(c)/complete_pass | n }} |
| §6.6(b), one adverse family | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(b), one family/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(b), one family/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(b), one family/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(b), one family/complete_pass | n }} |
| §6.6(b), two or more families | not evaluated: {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(b), two or more families/not_evaluated }} | | | |
| §6.6(a), an earlier outcome settles the order | not evaluated: {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(a)/not_evaluated }} | | | |
| §5.3(a), every required check passed | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(a)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(a)/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(a)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(a)/complete_pass | n }} |
| §5.3(b), a check failed, without Linkage | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), without Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), without Linkage/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), without Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), without Linkage/complete_pass | n }} |
| §5.3(b), a check failed, with Linkage | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), with Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), with Linkage/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), with Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), with Linkage/complete_pass | n }} |
| §5.3(c), no answer to a check | not evaluated: {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(c)/not_evaluated }} | | | |

**Functional invariance.** Each probe's answer set against the same case's original answer, over the {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/action | denominator }} probed cases. Functional agreement means the canonical action, the next check and whether the memo passes all agree. Raw disposition agreement is shown beside it, since a `needs_check` that names a required check not yet run is the hold:

| Agreement with the original answer | `opus`, shuffled | `opus`, renamed | `sol`, shuffled | `sol`, renamed |
| --- | --- | --- | --- | --- |
| Action | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/action | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/action | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/action | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/action | n }} |
| Next check | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/next_check | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/next_check | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/next_check | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/next_check | n }} |
| Pass or fail | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/complete_pass | n }} |
| **All three (functional)** | **{{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/functional | n }}** | **{{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/functional | n }}** | **{{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/functional | n }}** | **{{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/functional | n }}** |
| Raw disposition | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/disposition | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/disposition | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/disposition | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/disposition | n }} |

**Spend.** The calls behind the scored answers, a retry counting as a call, and their tokens as each command-line tool reported them. Charged tokens add a call's whole bound when its usage was not reported.

| Spend | `opus` | `sol` |
| --- | --- | --- |
| Calls | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/calls | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/calls | count }} |
| Input tokens | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/input_tokens | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/input_tokens | count }} |
| Output tokens | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/output_tokens | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/output_tokens | count }} |
| Calls with no reported usage | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/unknown_usage_calls | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/unknown_usage_calls | count }} |
| Charged tokens | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/charged_tokens | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/charged_tokens | count }} |

## Against simulation truth

These diagnostics set each memo against the hidden truth of the simulation, which neither
the policy nor the drafter sees. They are reported apart from the policy score: a
disposition FP-2 requires on the evidence in the packet can still be adverse for a
legitimate order.

| Diagnostic | `opus` | `sol` |
| --- | --- | --- |
| Legitimate orders given an adverse suggestion (hold, decline, escalate or `needs_check`) | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/legitimate_adverse | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/legitimate_adverse | of }} |
| Fraud cleared | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/fraud_cleared | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/fraud_cleared | of }} |
| The top hypothesis names the hidden pattern (any benign explanation for a legitimate order) | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_names_latent | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_names_latent | of }} |
| on fraud | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_names_latent_by_class/fraud | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_names_latent_by_class/fraud | of }} |
| on legitimate orders | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_names_latent_by_class/legitimate | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_names_latent_by_class/legitimate | of }} |
| Memos whose top hypotheses tie (they name nothing) | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_tied | count }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_tied | count }} |
| Clusters holding a failing memo | {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/clusters_with_failure | count }} of {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/clusters | count }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/failing_cluster_bound/clusters_with_failure | count }} of {{ bench:2026-10-final:/endpoints/diagnostics/sol/failing_cluster_bound/clusters | count }} |
| Exact one-sided upper bound on that share, at {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/level | pct:0 }} | {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/failure_share_upper_one_sided | pct }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/failing_cluster_bound/failure_share_upper_one_sided | pct }} |

The cluster bound (Clopper–Pearson, with clusters as the units) treats the clusters as
independent and alike and ignores the selection weights. It bounds the unweighted share
of clusters with a failing memo, not the natural-mix rate.

## Development

Development ran on the first prompt, `memo_fp2_v1`, with small sets from the development
seeds. Each set's results are scored with the final cohort's scoring code
and name the scoring changes made since the set ran.

| Set | Arm | Cases | Complete-memo pass |
| --- | --- | --- | --- |
| `2026-10-dev` | `sol` | review decisions | {{ bench:2026-10-dev:/arms/sol/summary/complete_pass | of }} |
| `2026-10-dev-checks` | `sol` | decisions after a check | {{ bench:2026-10-dev-checks:/arms/sol/summary/complete_pass | of }} |
| `2026-10-dev-opus` | `opus` | drawn from the two sets above | {{ bench:2026-10-dev-opus:/arms/opus/summary/complete_pass | of }} |

Every development failure listed R04 among its citations although R04 did not hold.
`opus` read `amount_over_category_p95`, the order's amount as a ratio to the percentile
R04 names, as a yes-or-no flag, and wrote that the amount was above that percentile when
it was a small fraction of it. `sol`'s failure read the ratio correctly and listed R04
anyway. The first prompt had described these ratio fields next to the sentence on flags,
without saying they were ratios or what their values mean. That is a fault in the
instrument, not in how a model follows the policy. `memo_fp2_v2` gives every field of that kind a plain description and changes
nothing else; both arms use it in the final cohort, and no development call ran on it.

The case memos shown in the [case files](../README.md#one-investigation) are `sol`'s memos on
`memo_fp2_v2` for each alert the files present, scored the same way and kept out of the
rates above: {{ bench:2026-10-cases:/arms/sol/summary/complete_pass | of }} passed.

An earlier memo study, `2026-08-dev`, ran on a world, packet builder and action
referee that have since been replaced. Its cases, prompts and stored responses are
archived byte for byte, `make llm-history` replays them offline, and its
[history](../llm/eval/benchmarks/2026-08-dev/HISTORY.md) sets out what it shows, what
it cannot show, and the corrections to its published results.

## Limits

- **One reading of the policy.** The referee and the simulated analyst both apply FP-2
  through `core/evidence.py`, so the score measures agreement with that reading, not with
  an independent one. The scoring rules have not yet been checked by a person.
- **The verifier reads structure, not meaning.** It checks each claim's field, value and
  declared calculation; a sentence that misstates a correct field passes. Memos with a
  number, amount, time or id that matches no packet value are counted but do not fail:
  {{ bench:2026-10-final:/arms/opus/summary/memos_with_unmatched_token | of }} scored memos for `opus`, {{ bench:2026-10-final:/arms/sol/summary/memos_with_unmatched_token | of }} for `sol`.
- **What the cohort leaves out.** A hold that no check answers expires without an
  analyst decision, so §5.3(c) is outside the benchmark; FP-2's other rows without a case
  are marked in the table.
- **Simulated decisions.** Every case is a decision in a synthetic world, and a decision
  after a check rests on facts as of the start of its day.

Each benchmark's `results.json` is rebuilt from its cached answers, with no calls, by
`python -m llm.eval.results` ([reproducing](../llm/eval/PROTOCOL.md#reproducing)).
