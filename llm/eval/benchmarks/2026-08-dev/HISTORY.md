# The August 2026 memo study

In August 2026, this project tested a language-model memo drafter on 200 alerts from the
simulated world then in use (seed 416). Each model received a JSON case packet and returned
observed signals, competing hypotheses, policy citations, a recommended action and a
priority. Four arms used the same cases: Claude Sonnet 5 with prompt v1 and with prompt v2,
and GPT-5.6 Luna and GPT-5.6 Terra with prompt v2. A provider quota cut Terra's run short.
Its published results cover the first 86 cases; 8 stored replies for later cases are not
scored.

`original/` preserves the cases, packets, prompts, the 997 stored responses, result files
and FP-1 policy the memos cite, byte for byte. `MANIFEST.json` holds their
hashes. The world, packet builder and action-scoring method have since been replaced. This
archive records the earlier study and the corrections to its published results.

The [iteration log](ITERATION.md) is a narrative record of the study's prompt changes
and results.

`python -m llm.eval.history` replays the stored responses offline using the study's own
protocol. It writes `corrected/`: `statistics.json` (every number with its numerator and
denominator), `attempt_chains.json` (every attempt behind every memo) and `summary.md`
(tables). `python -m llm.eval.history --check` fails if those files are out of date.

## What it shows

- **Deterministic replay.** The stored responses rebuild the four archived result files
  exactly. 9 of the 997 stored responses match no prompt produced by the study's protocol
  and are not used.
- **The numbers rule in prompt v2 cut unmatched-token memos.** A token here is a number,
  money amount, identifier or timestamp. The check calls it unmatched if it cannot match it
  to the packet or classify it as a count or sum over a packet list named in the text.
  Prompt v2 told the model to copy numbers from the packet and never compute them. On the
  same 200 cases, under the study's own check, Sonnet memos with an unmatched token in their
  observed signals fell from 67/200 to 9/200: 59 memos improved and 1 got worse (exact
  McNemar p = 1.06e-16). Under the corrected check below, they fell from 68/200 to 9/200
  (60 improved, 1 worse). This token check is independent of action scoring. These figures
  cover only the signals list, and the check does not verify the relationship stated in a
  sentence. Across all memo text, which the original study did not check, the corrected
  check flags 86/200 v2 memos (143/200 with v1). That count includes thresholds quoted from
  the prompt, such as $2,000.
- **More agreement with the old action referee.** Sonnet's recommended actions fell within
  the referee's acceptable set in 122/200 cases with v1 and 147/200 with v2, an increase of
  12.5 percentage points (33 cases improved, 8 got worse, p = 0.000112). The referee mapped
  each case's hidden simulated fraud pattern to a set of actions. It did not judge the
  decision against the evidence in the packet. All seven never-pay cases, for example,
  required a decline even though their packets held no repayment history. No arm matched
  the required action on any of them.

## What it cannot show

- **Decision quality.** Agreement with a referee that graded hidden simulator intent does
  not measure whether a decision was justified by the available evidence.
- **Held-out performance.** Prompt v2 was written after reading v1's misses on these same
  200 cases. No untouched set was kept.
- **A model ranking.** The arms used different command-line tools, and some settings were
  not recorded. The paired comparisons below describe these stored outputs.

## Why it is not rescored

The study is not rescored with a new referee. Changing the scoring after seeing the outputs
would make the result depend on choices made with the answers in view. The corrections
below retain the original action referee.

## Corrections

Corrected results are from `corrected/statistics.json`.

- **Luna's stored retry.** The protocol allowed a single retry after a reply failed
  validation. For Luna's alert 6188, the first reply was cut off. A valid retry was stored,
  but the archived replay read only first replies. First attempts: 199/200 valid, 119/199
  correct (119/200 with the failure counted wrong). Final outputs: 200/200 valid,
  120/200 correct, decline recall 22/57. Here, "correct" means agreement with the original
  referee.
- **Consistency.** Each of the first 50 cases was sent three more times with a neutral line
  added to the packet. The README printed 84% (N=200) for the revised prompt. The result is
  42/50 probe sets whose three actions agree; all three also match the original memo's
  action in 40/50. Prompt v1 has 47 complete probe sets (41/47 agree). On the cases complete
  for both prompts, the probes agree in 41 and 39 of 47 for v1 and v2, respectively. The
  log's "−3.2 points" compared 41/47 with 42/50, using different case sets.
- **Units.** The prompt change is 12.5 percentage points; the README printed "+12.5%".
- **Paired comparisons.** Arms are compared case by case on the cases both completed,
  rather than by comparing separate intervals. For Sonnet v2 against Luna, using final
  outputs for all 200 cases, only Sonnet matched the referee on 32 cases and only Luna on
  5 (p = 7.43e-06). Sonnet v2 against Terra on Terra's 86 cases: 65/86 against 66/86, with
  3 cases where only Sonnet matched and 4 where only Terra matched (p = 1). Terra's lead is
  one case.
- **Intervals.** Wilson intervals treat cases as independent. They are shown alongside
  bootstraps that resample users or injected fraud stories (137 groups for 200 cases; the
  35 synthetic-ring cases come from 3 rings).
- **"Holdout".** These are development cases drawn from all dates: alerts from 2025-07-18
  to 2026-06-21. Of these, 163/200 predate 2026-04-01, the configured holdout start. They
  come from 190 users; 140/200 are labelled fraud, against 1889/9450 of all alerts. The
  benign cases' shortest tenure is 78 days, so no new legitimate customer was tested.
- **Strict validation.** Every accepted output was checked again for correct types, then
  allowed values, a benign hypothesis and non-empty prose. This rejected 0 of the 979
  outputs the study accepted.
- **The token check.** Now called the "unmatched-token" check, it matches an id only as a
  whole identifier, a negative amount only to a negative value, and a number written with
  trailing zeros at the precision written (`$99.00` no longer matches 99.49). In the scored
  memos this changes one v1 signal, so v1 has 68/200 unmatched-token memos under the
  corrected check and 67/200 under the archived one.

## Limitations of the stored packets

The packets are preserved because changing them would change the inputs the models saw.

- 59 packets count linked accounts using orders or sign-ups after the alert, and 3 count
  installment payments made after it (alerts 3682, 5279 and 2587).
- In 200/200 packets, `amount_over_category_median` is the amount divided by the category's
  mean across the whole period. It uses neither a median nor a cutoff at the alert.
- In 32 of the 58 packets where rule R06 fired, its rationale counts email-root accounts
  that signed up after the alert. In all 58, the rationale's count is the packet's
  `other_accounts_same_email_root` plus one because it includes the account itself. The
  model therefore saw two numbers for one fact.
- 5 packets have a negative account tenure (alerts 686, 698, 842, 848 and 2655).
- The simulator appended injected fraud orders in one id block per pattern. For six
  patterns, every alert inside the block carries that pattern's label, so the order id alone
  identifies 118 of the 140 labelled cases. The packets exposed this shortcut; the stored
  replies do not show that a model used it.

## How the arms ran

The arms ran through the Claude and Codex command-line tools from the author's account,
in the caller's working directory. The stored records contain each final reply and its
duration, with incomplete token counts for the Claude arms. They do not contain
transcripts, so they cannot rule out influence from global instructions or other files.
Reasoning effort was not recorded. The latency medians come from different tools and do
not establish a speed comparison.
