# The memo benchmark's protocol

This is the memo drafter's evaluation, fixed before any final case exists. The code
that carries it out is [`select_cases.py`](select_cases.py) (the cohorts),
[`check_points.py`](check_points.py) (decisions after a check),
[`harness.py`](harness.py) (calls, caps and scores), [`../referee.py`](../referee.py)
(what the policy permits) and [`verifier.py`](verifier.py) (claims against the packet).
The sizes are in [`config/llm.yaml`](../../config/llm.yaml).

## What is measured

The drafter reads one packet: the facts an analyst sees about an order the incumbent
policy sent to review, at the moment of a decision. It returns claims tied to packet
fields, competing hypotheses (one benign), a suggested disposition, the clauses it rests
on, the next check and a short memo. It decides nothing; the referee scores the memo
against fraud policy FP-2 and the verifier checks every claim against the packet.

Two arms answer the same packets with the same prompt (`memo_fp2_v1`) and the same
isolation: `sol` (GPT-6.1 Sol at high effort, through the Codex CLI) and `opus` (Claude
Opus 5.5 at high effort, through the Claude Code CLI). The call caps are 400 for `sol` and
330 for `opus` (with 10M tokens), counting retries and isolation checks. Each call is one
model turn, with at most 16,000 output tokens a request, and at most one retry, for a
transport failure. Development ran on codex-cli 0.159.2 and Claude Code 2.1.293. Each
benchmark pins its arms' CLI versions and isolation at its first live call
(`pins.json`), and a later run of that benchmark under other versions is refused; the
final cohort's versions are the ones it pins, reported beside these.

## Decision points

A case is one analyst decision, on one of two axes.

- **Review** (`review`): the first decision, when an analyst takes the order up. No check
  has run yet.
- **After a check** (`check_completed`): the decision when the checks a hold started have
  answered. The case sits at the last completion and carries every completed check. These
  cases hold the policy's §5.3 rows: once every required check has passed, every hold is
  prohibited; after a failed one the order is declined, or escalated when Linkage is
  present. A case whose other required check has not answered yet stays under §6.6(b),
  where the hold continues.

The context row is the one the replay's simulated reviewer read at that moment. The
replay assembles each order's row at the start of the day (at the checkout, for an order
placed that day), under the incumbent's own decisions before that day. A check that
answers on the day of the first decision therefore reads the first decision's row; a
later answer reads the row of the start of its own day. The packet's decision time is the
completion, and its facts are those of that row, at most one replay day old. The rows
come from replaying the incumbent's main run again with the frozen code. The replay is
refused unless it reproduces, column by column, every review decision the run kept.

## Cohorts

All cases come from the test windows, under the incumbent rules at their tuned thresholds.

| Benchmark | Worlds | Cases | Arms |
| --- | --- | --- | --- |
| `2026-10-dev` | development seeds 416, 1041, 2718, baseline | 40 review decisions | `sol` |
| `2026-10-dev-checks` | the same | 25 decisions after a check | `sol` |
| `2026-10-dev-opus` | drawn from the two above | 20 + 20 | `opus` |
| the final cohort | every final seed's baseline world | 120 review decisions + 80 after a check | both |
| the case memos | the case world (416, baseline) of the final run | the case files' alerts (at most 6) | `sol` |

- **Strata.** On the review axis: each fraud pattern, and the legitimate orders by
  their rarest benign trait. After a check: the completed checks with their outcomes,
  the referee's standard disposition and the latent class.
- **Selection.** Selection is two-phase. Phase one draws at most two cases from each
  linked group of accounts and episodes, over both axes together. Phase two gives every
  stratum of an axis an equal share of the axis's count. A stratum whose every case lost
  its group's draw in phase one gets none, and the definition names it.
- **Weights and minimums.** Each case's weight is the inverse of its two-phase
  probability. On the final cohort, each check outcome present in the eligible pool gets
  at least 20 cases, or the selection is refused.
- **Exclusions.** The final cohort shares no account and no episode with any development
  benchmark.
- **Probes and seed.** 40 final cases are also asked twice more: with their facts
  shuffled, and with fresh placeholder names. Selection uses seed `20261006`.
- **Why 80 decisions after a check.** It is about their share of analyst decisions on
  the development seeds: 283 decisions after a check (the eligible count
  `2026-10-dev-checks` records) against 460 first decisions, counted in the development
  run's kept review decisions (pipeline output, not committed). That is 283 of 743.
- **Why baseline worlds only.** A lag-sensitivity world repeats its baseline world's
  reviewed orders, and the shifted futures would blend other populations into one
  natural-mix rate.

`2026-10-dev-opus` takes 20 cases from each development set:
- within a set, strata go in order of the SHA-256 of their names, and the cases in a
  stratum in order of the SHA-256 of their ids;
- one case is taken per stratum in turn;
- a case that would put more than two cases in one linked group is passed over.

The subset is not a probability sample, so its reweighted rates estimate nothing. Its
axes' shares are the sources' eligible decisions when every source recorded them;
`2026-10-dev` predates that record, so this subset's shares are its own case shares (one
half each).

The case memos (phase `cases`) are one memo for each alert the case files present. The
packet is built from the row the replay decided on, with no check completed. They are
scored like any other case and reported with the case files, not in the benchmark's
rates.

## Endpoints

**Primary: the complete-memo pass rate, per arm.** A memo passes only if all of these
hold:
- its format is valid;
- its disposition is acceptable (standard or also permitted);
- its next check is the one the policy requires (`none` when none is);
- its citations are valid: no unknown clause and no rule that does not hold; for an
  adverse disposition or `needs_check`, the disposition's own clause cited (a clear may
  cite none); and no non-payment clause cited when no installment is due (the clause ids
  are checked, not the prose);
- the verifier finds no claim error in the structured claims (their fields, values and
  declared calculations).

A failure to answer counts as a failing memo. The rate is reported as the natural-mix
rate:
- within each axis, the two-phase Hájek rate with the selection weights;
- the axes weighted by their shares of the eligible decisions.

Beside it are the unweighted rate, and each axis's unweighted and weighted (Hájek) rates.

**Secondary endpoints:**
- each of the five components;
- the acceptable rate (natural mix and unweighted);
- standard-action agreement, beside the raw table of the referee's standard disposition
  against the memo's disposition. `needs_check` counts as the hold only when it names a
  required check that has not run yet (§4.3);
- the acceptable and complete-memo pass rates per policy row: §6.6(c), §6.6(b) with one
  family, §6.6(b) with two or more, §6.6(a), §5.3(a), and §5.3(b) without and with
  Linkage. A row with no case is reported as not evaluated;
- functional invariance under each probe: agreement on the canonical action, on the next
  check and on complete-memo validity, and all three together, beside raw disposition
  agreement;
- the paired comparison of the arms on the same cases;
- spend.

**Diagnostics against simulation truth,** reported apart from the policy score:
- adverse suggestions for legitimate orders;
- cleared fraud;
- whether the single most likely hypothesis names the latent pattern (for a legitimate
  order, any benign explanation). A tie at the top names nothing.

## Intervals, bounds and ties

- **Clusters.** The cluster is the linked group of accounts and episodes.
- **Bootstrap.** The complete-memo pass rate and the acceptable rate, natural-mix and
  unweighted, get percentile cluster-bootstrap intervals: 2,000 resamples of whole
  clusters, seed 0, 95%. The Wilson interval of the unweighted rate is shown beside them.
  The other endpoints are reported as rates without intervals.
- **When nothing fails.** If every case passes (or none does), every resample gives the
  same value and a bootstrap interval collapses to a point, so none is reported; the
  primary rate then has no interval. What is reported instead, and in every case, is a
  separate diagnostic: the exact Clopper–Pearson bound on the share of clusters that hold
  a failing memo, with clusters as the units. With no failing cluster among k, its
  one-sided 95% upper bound is 1 − 0.05^(1/k), about 3/k. It treats the clusters as
  independent and alike, ignores the selection weights, and bounds that unweighted
  cluster share, not the natural-mix rate.
- **Paired comparison.** The two arms are compared on the same cases:
  - counts of cases both arms passed, only one passed, or neither;
  - the difference in rates, unweighted and natural-mix, each with a cluster-bootstrap
    interval that resamples the same clusters for both arms;
  - a sign test over clusters.

  It is made only once both arms are scored.
- **Ties.** Equal unweighted rates (as many cases passed by only one arm as by only the
  other) are reported as a tie, with the natural-mix difference beside it, which can
  differ. A tie is not evidence that the arms are equivalent, and the comparison may well
  be inconclusive.

## Development so far

- **`2026-10-dev`.** Every development case passed: all 40 memos from `sol` passed every
  component, with no claim error in 889 claims. All 40 were first decisions with no
  check run (§6.6(c) and §6.6(b) with one family), which is why the cases after a check
  were added.
- **`2026-10-dev-checks`.** Every disposition was acceptable and standard, and the next
  check was right in all 25 cases (§5.3(a) 10, §5.3(b) without Linkage 9, with Linkage
  6). There was no claim error in 529 claims. 24 of 25 memos passed. The one that did not
  cleared a §5.3(a) case correctly but listed R04 among its citations while its text said
  no Context rule held; the referee counts a cited rule that does not hold as an invalid
  citation. That case sits in the largest stratum (a passed id_check on a legitimate
  order). Its weight therefore takes the natural-mix pass rate on this set to 0.65, with
  a cluster-bootstrap interval of 0.23 to 1. Development sets are this small by design;
  the final cohort's 80 decisions after a check are what the estimate rests on.
- **`2026-10-dev-opus`.** All 40 `opus` memos were valid in format (each in a JSON code
  fence, which the parser accepts), acceptable, standard in action and right in their next
  check, with no claim error in 735 claims. 37 of 40 passed. The three that did not list
  R04 among their citations though R04 does not hold, as `sol`'s one failure does; one of
  the three is that same case. Two memos stated a correct duration from placement to the
  decision as a derived difference of two packet times, which the verifier could not yet
  recompute. The verifier now reads such a difference in the unit the claim names
  (seconds, minutes, hours or days), for both arms; no `sol` memo had one. 41 calls (with
  the isolation check), 516,616 tokens, median 23 s a call, no retry.
- **The two arms on the same 40 cases.** Complete-memo pass: `opus` 37, `sol` 39. Both
  passed 37, only `sol` 2, neither 1. The difference is −0.05, with a cluster-bootstrap
  interval of −0.13 to 0. Two clusters favour `sol` and none `opus` (sign test p = 0.5).
  Acceptable dispositions are 40 of 40 for both, a tie. `opus` held two first-review
  orders outright where `sol` asked for the check (`needs_check`); both are the standard
  action. This subset is not a probability sample, so these counts describe development
  only.
- **The prompt** stays at its first version, as decided before these sets ran. A prompt
  changed to fix one citation would be tuned on the development cases. No format problem
  appeared in either arm.

## Limits

- **§5.3(c) is not evaluated (N = 0).** A hold that no check answers expires: cancelled
  before shipment, unchanged after it. No analyst decision follows. The policy's rule
  after `no_response` is therefore outside the benchmark.
- **The referee and the simulated reviewer share one reading of FP-2.** Both apply
  `core.evidence`. On the development run's three baseline worlds, the referee's
  standard on the packets built at the 283 completions contained the reviewer's decision
  at each of them. That was a check run on the development run's output before selection
  and is not committed; the committed set holds 25 of those packets. That agreement checks the packet's round trip (fields, values, checks); it is
  not two independent readings of the policy. The scoring rules have not been checked by
  a person until the author's referee check (below) is complete.
- **The verifier reads structure, not meaning.** It checks each claim's field, value and
  declared calculation against the packet. It does not check that a claim's sentence says
  what the field shows: a sentence that misstates a correct field passes. It also looks
  for concrete tokens in the memo (numbers, amounts, times, entity ids) that match no
  packet value; memos with one are counted and reported, but do not fail. Ordinary words
  are not checked, so an assertion without such a token goes unchecked.
  Whether the sentences mean what they cite, whether decision-critical evidence is left
  out and whether a memo is defensible are left to the author's referee check.
- **The decisions are simulated.** Every decision after a check rests on facts as of the
  start of its day, as the simulated reviewer saw them.
- **No headroom is claimed.** A rate of one on development cases is a result on those
  cases, not a ceiling.

## The author's referee check

This check is designed now and prepared once the final cohort exists. It is kept outside
the repository and takes about an hour. It is reported descriptively, and nothing waits
for it.

- **Cases.** 24 final cases, aliased: 14 review decisions and 10 decisions after a check
  (120 : 80, rounded).
  - Within an axis, strata go in order of the SHA-256 of `referee-check:<stratum>`.
  - Within a stratum, cases go in order of the SHA-256 of `referee-check:<case id>`.
  - One case is taken per stratum in turn, so no stratum gets a second case before every
    stratum of its axis has one.
- **Material.** Each case's packet, the policy and a short form.
- **Pass 1, before any memo.** For each case, mark every action he would accept.
- **Pass 2.** Show one memo per case, blind to the arm and to simulation truth:
  - Within each axis, cases go in order of the SHA-256 of `referee-arm:<case id>`, and the
    arms alternate starting with `sol`. That gives 12 memos from each arm.
  - Cases are shown in order of the SHA-256 of `referee-order:<case id>`.
  - A case whose assigned arm gave no valid memo shows that.
  - Rubric, for each memo:
    - do the claim sentences mean what their cited fields show?
    - is any decision-critical evidence omitted?
    - does the benign explanation fit?
    - is the memo defensible overall?

## Reproducing

```sh
python -m llm.eval.select_cases --id 2026-10-dev-checks --phase development \
    --axis check_completed --world <run>/worlds/416-baseline --world ...
python -m llm.eval.select_cases --id 2026-10-dev-opus --combine 2026-10-dev \
    --combine 2026-10-dev-checks --arm opus
python -m llm.eval.select_cases --id <final id> --phase final --world <run>/worlds/<seed>-baseline ... \
    --development 2026-10-dev --development 2026-10-dev-checks --development 2026-10-dev-opus
python -m llm.eval.select_cases --id <case memos id> --case-memos <run> --arm sol
python -m llm.eval.harness --benchmark <id>             # every arm, from the cache
python -m llm.eval.harness --benchmark 2026-10-dev --arm sol \
    --amend-scoring "the scoring changes made after this set ran"
python -m llm.eval.harness --benchmark 2026-10-dev-checks --arm sol --amend-scoring "..."
python -m llm.eval.harness --benchmark 2026-10-dev-opus --arm opus --amend-scoring "..."
```

The development sets each hold one arm's records (`sol` for `2026-10-dev` and
`2026-10-dev-checks`, `opus` for `2026-10-dev-opus`), so they are scored with `--arm`.
All three were fixed before the last scoring change (the time differences), so scoring
them names the change with `--amend-scoring`, which the results record. The final
cohort is fixed with the scoring code as it stands at this protocol's commit.
