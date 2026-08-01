# bnpl-fraud-workbench

An end-to-end buy-now-pay-later fraud operation in miniature: a deterministic BNPL
transaction simulator with labeled fraud patterns injected into realistic benign
traffic; a MySQL investigation layer with a twelve-query analyst library; an
interpretable rules engine tuned against explicit loss/review/insult costs; an alert
queue with SLA simulation; a chronologically validated ML detection layer with
per-pattern error analysis; and a Claude-powered investigation-memo drafter with a
measured evaluation harness (action accuracy, hallucination rate, policy-citation
validity, consistency, cost). Five written case investigations show the analyst
workflow end to end.

**All data is synthetic and labeled.** Fraud patterns are injected by
[`simulator/patterns.py`](simulator/patterns.py) — every performance number below is a
statement about method on simulated data, not a real-world rate claim. See
[Limitations](#limitations--honesty).

```mermaid
graph LR
    S[simulator<br/>353k orders, 7 patterns] --> DB[(MySQL 8.4)]
    DB --> Q[Q01–Q12<br/>investigation queries]
    DB --> R[rules engine R01–R12<br/>cost-tuned bands]
    R --> AL[alerts]
    AL --> QS[queue / SLA sim]
    AL --> PK[case packets]
    PK --> LLM[Claude triage memos<br/>+ eval harness]
    S --> M[ML layer<br/>LR + HGB, chrono split]
    AL --> C[case files 01–05]
    LLM --> C
```

## Headline results (regenerate with `make demo`)

| layer | result (holdout months 10–12 unless noted) |
|---|---|
| simulator | 353,276 orders / 41,410 users / 12 months; fraud base rate 0.94% of approved orders; generation 78s, deterministic (seed 416) |
| rules @ tuned bands (30/90) | 24.9 alerts/day · precision 8.9% · recall 59.8% · $37,847 fraud caught · 0 false auto-declines · net +$32,550 under the cost model |
| ML (HistGradientBoosting) | PR-AUC 0.94 vs logistic 0.72 · precision@capacity 9.5% · recall-by-pattern table incl. the honest failures (INR abuse 24%, never-pay: see CASE-04) |
| queue sim | 2 analysts, offset 7-day-coverage shifts · SLA(≤4 business h) 99.6% · score-priority blocks $27,496 of fraud pre-fulfillment vs $25,884 FIFO |
| Claude triage (claude-sonnet-5, prompt v2) | action accuracy 73.5% · decline precision 96.8% / recall 52.6% · **hallucination rate 0.0%** (mechanically verified) · consistency 84% · N=200 |
| prompt iteration v1→v2 (same model, same cases) | action accuracy **+12.5pp** · hallucinations **10%→0%** · full log in [ITERATION.md](llm/eval/ITERATION.md) |
| cross-model arms (same harness) | gpt-5.6-terra 76.7% acc / 70% decline recall (n=86, quota-cut) · gpt-5.6-luna 59.8% acc / 100% decline precision (n=199) · model per task is a config switch |

## Quickstart

```bash
docker compose up -d        # MySQL 8.4 (or: docker-compose up -d)
make venv                   # python 3.11+
make demo                   # generate → load → rules → model → queue → llm-eval (offline)
```

`make demo` runs end-to-end from a fresh clone with **no API keys and no network** —
LLM eval-set responses are committed to [`llm/eval/cache/`](llm/eval/cache) and the
harness replays them (`--offline`). Live modes: any Claude Code-compatible CLI
(`CLAUDE_CLI_BIN`) or the `anthropic` SDK (`ANTHROPIC_API_KEY`, `LLM_BACKEND=api`).

## The layers

**Simulator** ([`simulator/`](simulator)) — pay-in-4 mechanics (25% down + 3 biweekly
installments; the platform pays the merchant up front, so loss = principal − collected).
Benign traffic is deliberately rich: repeat customers, seasonality, travelers, movers,
gift buyers, device churn, and ~2.5% *benign hardship defaulters* — the mimics that make
detection non-trivial (FP-1 §6). Seven injected patterns: account takeover, stolen-card
(with card-testing preludes), synthetic rings (warm-up → bust-out), first-party
never-pay, INR/friendly-fraud abuse, promo farming, merchant bust-out. Ground truth
lives in a separate `labels` table that analyst-facing layers never read — enforced by
tests and a packet-builder assertion.

**Investigation library** ([`db/queries/`](db/queries)) — twelve MySQL-8 queries, each
headed by the investigative question it answers: velocity, shared-attribute linkage,
new-account risk, repayment cohorts, the ATO event chain, merchant health, promo
clusters, card testing, dispute abuse, geo-velocity (with an embedded centroid CTE),
queue ops, and loss accounting.

**Policy + rules** ([`policy/fraud-policy.md`](policy/fraud-policy.md),
[`rules/`](rules)) — FP-1 is a written internal policy: definitions (fraud vs credit vs
abuse), evidence standards, an action ladder, per-rule intent, and false-positive
guidance. R01–R12 implement it with per-order rationale strings, so every alert is
explainable. [`rules/tuning.py`](rules/tuning.py) sweeps the review/decline bands —
**selected on months 1–9, reported on months 10–12** — under an explicit cost model
(review $2.50/case; false decline = 5% margin + $15 LTV proxy; both are stated
assumptions). Output: [`reports/tradeoffs.md`](reports/tradeoffs.md) + frontier SVG.

**Queue / SLA simulation** ([`queue_sim/`](queue_sim)) — discrete-event sim of two
analysts on offset shifts with a 12-hour fulfillment race: an alert resolved after the
order ships doesn't save the money. Compares FIFO vs rule-score vs LLM-priority
dispatch on SLA attainment, time-to-decision, and **fraud-$ blocked before
fulfillment** ([`reports/queue.md`](reports/queue.md)).

**ML layer** ([`model/`](model)) — point-in-time features with a **leakage-guard test**
(25 random orders recomputed on time-truncated data must reproduce identical rows),
chronological train/holdout split (no shuffling: fraud drifts, and random splits leak
ring structure), logistic regression + HistGradientBoosting, PR-AUC (not ROC at a ~1%
base rate), precision@review-capacity, calibration, recall-by-pattern, and a
rules-vs-model-vs-hybrid comparison at equal review capacity
([`reports/model.md`](reports/model.md)).

**Claude triage** ([`llm/`](llm)) — the packet builder assembles everything an analyst
would pull (profile, history, fired rules with rationales, linkage counts, repayment)
and **provably excludes ground truth** (label-key walk in tests). The memo drafter
returns structured JSON: signals (verbatim facts), competing hypotheses *including the
benign one*, policy citations, action, priority, evidence gaps. Human-in-the-loop by
design: the model drafts and prioritizes; the analyst owns decisions (FP-1 §7) — CASE-05
shows an analyst override in action.

**The eval harness** ([`llm/eval/`](llm/eval)) is the part I'd show a hiring manager
first: 200 frozen stratified cases (all 7 patterns + 60 hard negatives), mechanical
hallucination verification (every concrete token a memo cites must exist in its packet),
citation-validity checks, perturbation-consistency, per-arm cost/latency, and a
versioned prompt-iteration log ([`llm/eval/ITERATION.md`](llm/eval/ITERATION.md)) where
every prompt change carries its before/after metric table. Model routing is
config-driven per task ([`config.yaml`](config.yaml) `llm.tasks`) with env/CLI
overrides.

**Vendor enrichment** ([`vendor/`](vendor)) — optional IPQualityScore email/IP scoring
feeding rule R12: `make vendor` (synthetic stand-in scores, clearly labeled) or
`make vendor-live` with `IPQS_API_KEY`. Kept out of `make demo` so the headline numbers
above stay exactly reproducible from the seed; the run reported here executed without
vendor signals.

## Case files — the analyst workflow, end to end

| case | one line |
|---|---|
| [CASE-01](cases/CASE-01-account-takeover.md) | Tenured account, credential chain → $3.6k jewelry burst; the R01 signature vs the new-phone mimic; measured follow-up: +12 ATO auto-declines at zero insult cost |
| [CASE-02](cases/CASE-02-card-testing-stolen-card.md) | Five $25–$38 probes at 4-minute cadence → $1,940 of approvals; proposed device cooldown hits 46 fraud devices, 0 benign |
| [CASE-03](cases/CASE-03-synthetic-ring-bustout.md) | 80 mailinator accounts, 26 devices, 2 drop addresses; $9.8k of perfectly repaid warm-up buys $68.7k of burst; includes a measured *rejected* proposal |
| [CASE-04](cases/CASE-04-neverpay-vs-hardship.md) | Two written-off accounts, opposite verdicts — the credit-vs-fraud line, and why never-pay isn't a transaction-time detection problem |
| [CASE-05](cases/CASE-05-traveler-cleared.md) | The false positive, cleared: 105/105 installments paid, a laptop in Hanoi, and a measured R03 suppression that cuts 4 benign alerts/day at zero fraud cost |

## Repository practices

Deterministic everywhere (single seeded RNG; two runs are byte-identical — tested).
Ruff-clean, 30+ tests, CI runs the full pipeline against a MySQL service container at
5% scale plus the offline LLM harness. No keys, no network in CI.

## Limitations & honesty

- **Synthetic data.** Injected patterns are scripted adversaries; real fraud adapts,
  drifts, and correlates with things no simulator captures. Numbers here demonstrate
  that the *measurement machinery* works, not that these rates transfer.
- The ML layer's high recall on several patterns reflects simulation separability and
  generous review capacity (10× the holdout fraud count) — the interesting artifacts
  are the *misses* (INR abuse, never-pay) and the capacity/threshold analysis, not the
  wins.
- Cost parameters ($2.50 review, 5%+$15 insult) are assumptions, stated and centralized
  in config; the tuning frontier is conditional on them.
- Chargeback timing, issuer behavior, and label latency are simplified; production
  systems fight feedback loops and delayed ground truth this repo doesn't model.
- The LLM eval measures grounding and decision quality *against this policy on these
  packets*; it is a methodology demonstration, not a claim that memo drafting is
  solved.

## License

MIT · Author: Alejandro Guerrero Padrés
