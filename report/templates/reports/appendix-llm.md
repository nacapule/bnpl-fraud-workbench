# LLM appendix

The memo drafter produces an advisory investigation memo from a case packet. It includes claims tied to packet fields, competing explanations including a benign explanation, a suggested disposition, supporting fraud-policy clauses, a next check and a short narrative. The analyst makes the decision. The benchmark measures whether these memos follow fraud policy FP-2 on analyst decisions in synthetic worlds. Its endpoints were specified in [`llm/eval/PROTOCOL.md`](../llm/eval/PROTOCOL.md) before any final case existed. All figures come from the benchmark results files; the final cohort's results are in [`llm/eval/benchmarks/2026-10-final/results.json`](../llm/eval/benchmarks/2026-10-final/results.json).

## The benchmark

The final cohort contains {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/review/unweighted | denominator }} review decisions, made when an analyst first takes up an order, and {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/check_completed/unweighted | denominator }} decisions after checks have answered. The cases come from the test windows of the baseline world for every final seed, under today's rules. They form {{ bench:2026-10-final:/statistics/clusters | count }} clusters of linked accounts and episodes, with at most {{ config:llm:benchmark.max_cases_per_cluster }} cases per cluster. Each case's weight is the inverse of its selection probability. Each of {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/action | denominator }} cases is also presented once with its facts shuffled and once with new placeholder names.

The `opus` and `sol` arms receive the same packets and prompt, `memo_fp2_v2`, under the same isolation settings. Each call consists of a single model turn. Each arm's command-line tool version is fixed at the benchmark's first live call:

| Arm | Model | Effort | Command-line tool | Fixed version |
| --- | --- | --- | --- | --- |
| `opus` | {{ bench:2026-10-final:/pins/opus/model }} | {{ bench:2026-10-final:/pins/opus/effort }} | Claude Code CLI | {{ bench:2026-10-final:/pins/opus/cli_version }} |
| `sol` | {{ bench:2026-10-final:/pins/sol/model }} | {{ bench:2026-10-final:/pins/sol/effort }} | Codex CLI | {{ bench:2026-10-final:/pins/sol/cli_version }} |

The referee ([`llm/referee.py`](../llm/referee.py)) scores each memo against FP-2 using the same evidence code as the simulated analyst. The verifier ([`llm/eval/verifier.py`](../llm/eval/verifier.py)) checks the fields, values and declared calculations in every structured claim against the packet.

## The primary endpoint: complete-memo pass rate

A memo passes only if all of these conditions hold:

- its format is valid;
- its disposition is the policy's standard or another permitted disposition;
- its next check is the one the policy requires;
- its citations use defined clauses, cite only rules that hold, include the disposition's supporting clause unless it recommends `clear`, and contain no non-payment clause when no installment is due;
- the verifier finds no error in its structured claims.

A missing model response counts as a failed memo. The primary endpoint is the natural-mix pass rate for each arm. It uses the case weights within each decision axis, then combines the axis rates using their shares of eligible decisions: {{ bench:2026-10-final:/axes/review | pct }} review decisions and {{ bench:2026-10-final:/axes/check_completed | pct }} decisions after a check. Percentile cluster-bootstrap intervals resample whole clusters. The table also reports the unweighted rate with its cluster-bootstrap and Wilson intervals, and each axis's unweighted and weighted rates.

| Complete-memo pass rate | `opus` | `sol` |
| --- | --- | --- |
| **Natural-mix rate (primary)** | **{{ bench:2026-10-final:/endpoints/primary/arms/opus/natural_mix | pct }}** | **{{ bench:2026-10-final:/endpoints/primary/arms/sol/natural_mix | pct }}** |
| Cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/primary/arms/opus/natural_mix_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/natural_mix_cluster_bootstrap | bounds }} |
| Unweighted rate | {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted | of }} ({{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted | pct }}) | {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted | of }} ({{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted | pct }}) |
| Cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted_cluster_bootstrap | bounds }} |
| Wilson interval | {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted_wilson | bounds }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted_wilson | bounds }} |
| Review decisions, unweighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/review/unweighted | of }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/review/unweighted | of }} |
| Review decisions, weighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/review/weighted | pct }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/review/weighted | pct }} |
| After a check, unweighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/check_completed/unweighted | of }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/check_completed/unweighted | of }} |
| After a check, weighted | {{ bench:2026-10-final:/endpoints/primary/arms/opus/per_axis/check_completed/weighted | pct }} | {{ bench:2026-10-final:/endpoints/primary/arms/sol/per_axis/check_completed/weighted | pct }} |

## What failed

`sol` passed {{ bench:2026-10-final:/endpoints/primary/arms/sol/unweighted | of }} memos. Every failure was a citation error. Its dispositions were acceptable in {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/acceptable | of }} cases, and its next checks were correct in {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/next_check_ok | of }}. The verifier found no error in its {{ bench:2026-10-final:/arms/sol/summary/claim_errors | denominator }} structured claims. One failed memo cited §4.4, which FP-2 does not define. The others cited rules that did not hold for the packet: R03 on orders for which the policy required `clear`, R01 with `needs_check`, and R02, R08 and R10 together on another `clear` recommendation.

`opus` passed {{ bench:2026-10-final:/endpoints/primary/arms/opus/unweighted | of }} memos. One failed response contained a complete memo, a correction note and a second JSON object. The parser requires a single JSON object, so the response could not be scored. The other failure was a derived claim that subtracted `decision.checks` from `context.accounts_on_address_30d`. `decision.checks` is a list of completed checks, not a number; the verifier could not recompute the calculation and counted the claim as an error.

## The paired comparison

The paired comparison is a secondary endpoint specified in the protocol. Both arms are scored on the same cases. The table reports the paired counts and the unweighted and natural-mix differences, expressed as `opus` minus `sol`. Each difference has a cluster-bootstrap interval that resamples the same clusters for both arms. The sign test uses clusters.

| Paired result | Complete-memo pass rate | Acceptable disposition |
| --- | --- | --- |
| Both arms passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/both | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/both | count }} |
| Only `opus` passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/only_first | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/only_first | count }} |
| Only `sol` passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/only_second | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/only_second | count }} |
| Neither passed | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/neither | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/neither | count }} |
| Unweighted difference | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/difference | pp }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/difference | pp }} |
| Cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/difference_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/difference_cluster_bootstrap | bounds }} |
| Natural-mix difference | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/natural_difference | pp }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/natural_difference | pp:2 }} |
| Cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/natural_difference_cluster_bootstrap | bounds }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/natural_difference_cluster_bootstrap | bounds:2 }} |
| Clusters favouring `opus`, favouring `sol`, tied | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/clusters_favouring_first | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/clusters_favouring_second | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/clusters_tied | count }} | {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/clusters_favouring_first | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/clusters_favouring_second | count }}, {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/clusters_tied | count }} |
| Sign test over clusters | p = {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/cluster_sign_test_p | num:2 }} | p = {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/acceptable/cluster_sign_test_p | num:2 }} |

The complete-memo comparison is inconclusive: the unweighted difference's interval includes zero, and the cluster sign test gives p = {{ bench:2026-10-final:/endpoints/secondary/paired_comparison/value/opus vs sol/complete_pass/cluster_sign_test_p | num:2 }}. A tie would not have shown equivalence either.

## Secondary endpoints

**Components.** The table shows how many memos passed each component, followed by the number of claims with an error among all structured claims each arm made:

| Component | `opus` | `sol` |
| --- | --- | --- |
| Format valid | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/format_valid | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/format_valid | of }} |
| Disposition acceptable | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/acceptable | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/acceptable | of }} |
| Next check correct | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/next_check_ok | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/next_check_ok | of }} |
| Citations valid | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/citations_valid | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/citations_valid | of }} |
| No claim error | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/components/value/no_claim_error | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/components/value/no_claim_error | of }} |
| Claims with an error | {{ bench:2026-10-final:/arms/opus/summary/claim_errors | of }} | {{ bench:2026-10-final:/arms/sol/summary/claim_errors | of }} |

**Acceptable rate.** The share of memos with a disposition permitted by FP-2, using the same weighting and interval methods as the primary endpoint:

| Acceptable disposition | `opus` | `sol` |
| --- | --- | --- |
| Natural-mix rate | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/natural_mix | pct:2 }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/natural_mix | pct:2 }} |
| Cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/natural_mix_cluster_bootstrap | bounds:2 }} | none: {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/degenerate }} |
| Unweighted rate | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/unweighted | of }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/unweighted | of }} |
| Cluster-bootstrap interval | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/unweighted_cluster_bootstrap | bounds }} | none: {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/degenerate }} |
| Wilson interval | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/acceptable_rate/unweighted_wilson | bounds }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/acceptable_rate/unweighted_wilson | bounds }} |

When every case passes, all bootstrap resamples yield the same rate, so the protocol reports no bootstrap interval.

**Standard-action agreement.** The memo's action agreed with the referee's standard action in {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/rate | of }} cases for `opus` and {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/rate | of }} for `sol`. Under FP-2 §4.3, `needs_check` counts as `hold` when it names a required check that has not yet run. The table shows the referee's standard disposition followed by the memo's disposition:

| Referee's standard, memo's disposition | `opus` | `sol` |
| --- | --- | --- |
| clear, clear | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/clear -> clear | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/clear -> clear | count }} |
| hold, `needs_check` | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/hold -> needs_check | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/hold -> needs_check | count }} |
| hold, hold | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/hold -> hold | count }} | none |
| decline, decline | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/decline -> decline | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/decline -> decline | count }} |
| decline, no scorable memo | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/decline -> failure | count }} | none |
| escalate, escalate | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/standard_action_agreement/table/value/escalate -> escalate | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/standard_action_agreement/table/value/escalate -> escalate | count }} |

**By policy row.** Acceptable-disposition and complete-memo pass rates for each row of FP-2's decision table:

| Policy row | `opus` acceptable | `opus` complete pass | `sol` acceptable | `sol` complete pass |
| --- | --- | --- | --- | --- |
| §6.6(c), no adverse rule family | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(c)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(c)/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(c)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(c)/complete_pass | n }} |
| §6.6(b), one adverse family | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(b), one family/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(b), one family/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(b), one family/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§6.6(b), one family/complete_pass | n }} |
| §6.6(b), two or more families | not evaluated: {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(b), two or more families/not_evaluated }} | | | |
| §6.6(a), an earlier outcome settles the order | not evaluated: {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§6.6(a)/not_evaluated }} | | | |
| §5.3(a), every required check passed | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(a)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(a)/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(a)/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(a)/complete_pass | n }} |
| §5.3(b), a check failed, without Linkage | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), without Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), without Linkage/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), without Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), without Linkage/complete_pass | n }} |
| §5.3(b), a check failed, with Linkage | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), with Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(b), with Linkage/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), with Linkage/acceptable | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/per_policy_row/value/§5.3(b), with Linkage/complete_pass | n }} |
| §5.3(c), no answer to a check | not evaluated: {{ bench:2026-10-final:/endpoints/secondary/arms/opus/per_policy_row/value/§5.3(c)/not_evaluated }} | | | |

**Functional invariance.** The table compares each modified packet's answer with the original answer for the {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/action | denominator }} probed cases. Functional agreement means the canonical action, next check and pass-or-fail result all match the original. Raw disposition agreement is also reported; `needs_check` counts as `hold` when it names a required check that has not yet run.

| Agreement with the original answer | `opus`, shuffled | `opus`, renamed | `sol`, shuffled | `sol`, renamed |
| --- | --- | --- | --- | --- |
| Canonical action | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/action | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/action | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/action | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/action | n }} |
| Next check | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/next_check | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/next_check | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/next_check | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/next_check | n }} |
| Pass or fail | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/complete_pass | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/complete_pass | n }} |
| **All functional measures** | **{{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/functional | n }}** | **{{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/functional | n }}** | **{{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/functional | n }}** | **{{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/functional | n }}** |
| Raw disposition | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/shuffled/disposition | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/functional_invariance/value/renamed/disposition | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/shuffled/disposition | n }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/functional_invariance/value/renamed/disposition | n }} |

**Spend.** The table reports calls used for the benchmark answers, including retries, and token usage reported by each command-line tool. The benchmark budget counts reported usage when available. When usage is missing, it counts the maximum token allowance reserved for that call.

| Spend | `opus` | `sol` |
| --- | --- | --- |
| Calls | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/calls | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/calls | count }} |
| Input tokens | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/input_tokens | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/input_tokens | count }} |
| Output tokens | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/output_tokens | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/output_tokens | count }} |
| Calls with no reported usage | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/unknown_usage_calls | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/unknown_usage_calls | count }} |
| Tokens counted against budget | {{ bench:2026-10-final:/endpoints/secondary/arms/opus/spend/value/charged_tokens | count }} | {{ bench:2026-10-final:/endpoints/secondary/arms/sol/spend/value/charged_tokens | count }} |

## Against simulation truth

These diagnostics compare memos with the simulation's hidden truth, which neither the policy nor the drafter sees. They are separate from the policy score: FP-2 can require an adverse disposition for a legitimate order based on the packet's evidence. A top hypothesis matches when the single most likely explanation names the hidden fraud pattern, or when it names any benign explanation for a legitimate order. Tied top hypotheses count as no match.

| Diagnostic | `opus` | `sol` |
| --- | --- | --- |
| Legitimate orders with an adverse suggestion (hold, decline, escalate or `needs_check`) | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/legitimate_adverse | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/legitimate_adverse | of }} |
| Fraud cleared | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/fraud_cleared | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/fraud_cleared | of }} |
| Top hypothesis matches simulation truth | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_names_latent | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_names_latent | of }} |
| Top hypothesis matches on fraud | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_names_latent_by_class/fraud | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_names_latent_by_class/fraud | of }} |
| Top hypothesis matches on legitimate orders | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_names_latent_by_class/legitimate | of }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_names_latent_by_class/legitimate | of }} |
| Memos with tied top hypotheses | {{ bench:2026-10-final:/endpoints/diagnostics/opus/against_simulation_truth/value/top_hypothesis_tied | count }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/against_simulation_truth/value/top_hypothesis_tied | count }} |
| Clusters containing a failing memo | {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/clusters_with_failure | count }} of {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/clusters | count }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/failing_cluster_bound/clusters_with_failure | count }} of {{ bench:2026-10-final:/endpoints/diagnostics/sol/failing_cluster_bound/clusters | count }} |
| Exact one-sided upper bound on the failing-cluster share, at {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/level | pct:0 }} | {{ bench:2026-10-final:/endpoints/diagnostics/opus/failing_cluster_bound/failure_share_upper_one_sided | pct }} | {{ bench:2026-10-final:/endpoints/diagnostics/sol/failing_cluster_bound/failure_share_upper_one_sided | pct }} |

The Clopper–Pearson bound uses clusters as its units. It bounds the unweighted share of clusters containing a failing memo, rather than the natural-mix pass rate.

## Development

Development used `memo_fp2_v1` and small case sets from the development seeds. Each set's results use the final cohort's scoring code and record the subsequent scoring changes.

| Set | Arm | Cases | Complete-memo pass rate |
| --- | --- | --- | --- |
| `2026-10-dev` | `sol` | review decisions | {{ bench:2026-10-dev:/arms/sol/summary/complete_pass | of }} |
| `2026-10-dev-checks` | `sol` | decisions after a check | {{ bench:2026-10-dev-checks:/arms/sol/summary/complete_pass | of }} |
| `2026-10-dev-opus` | `opus` | drawn from the two sets above | {{ bench:2026-10-dev-opus:/arms/opus/summary/complete_pass | of }} |

Every development failure cited R04 when it did not hold. `opus` treated `amount_over_category_p95` as a yes-or-no flag, although the field is the order amount divided by the category percentile used by R04. It described amounts that were a small fraction of that threshold as exceeding it. `sol` read the ratio correctly but still cited R04. The first prompt described ratio fields beside the explanation of flags without identifying them as ratios or explaining their values. The second prompt, `memo_fp2_v2`, clarifies the field descriptions and changes nothing else. Both arms use it in the final cohort; no development call used it.

The [case files](../README.md#one-investigation) show `sol`'s memos using `memo_fp2_v2` for each alert. They use the same scoring and are excluded from the benchmark rates above: {{ bench:2026-10-cases:/arms/sol/summary/complete_pass | of }} passed.

The earlier memo study, `2026-08-dev`, used a simulation world, packet builder and action referee that have since been replaced. Its cases, prompts and stored responses are archived byte for byte, and `make llm-history` replays them offline. Its [history](../llm/eval/benchmarks/2026-08-dev/HISTORY.md) explains what the study shows, its limits and the corrections to its published results.

## Limits

- **One reading of the policy.** The referee and the simulated analyst both apply FP-2 through `core/evidence.py`, so the score reflects their shared reading rather than independent interpretations of the policy. The scoring rules have not yet been checked by a person.
- **Claim verification.** The verifier checks structured fields, values and declared calculations. It does not check whether a claim's sentence states what its field means, so a sentence that misstates a correct field can pass. Memos containing a number, amount, time or entity id that matches no packet value are counted but do not fail for that reason: {{ bench:2026-10-final:/arms/opus/summary/memos_with_unmatched_token | of }} scored memos for `opus` and {{ bench:2026-10-final:/arms/sol/summary/memos_with_unmatched_token | of }} for `sol`.
- **Coverage.** A hold that receives no answer to its checks expires without an analyst decision, so §5.3(c) is outside the benchmark. Other FP-2 rows without a case are marked in the policy-row table.
- **Cluster bound.** The bound treats clusters as independent, with a common probability of containing a failing memo. It ignores the selection weights.
- **Simulated decisions.** Every case is a decision in a synthetic world. Decisions after a check use facts from the start of the day, or from checkout for orders placed that day. Those facts can precede check completion.

To rebuild each benchmark's `results.json` from its cached answers without model calls, use `python -m llm.eval.results` ([reproducing](../llm/eval/PROTOCOL.md#reproducing)).
