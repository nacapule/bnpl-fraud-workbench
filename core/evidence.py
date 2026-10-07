"""Evidence classification shared by the simulated reviewer and the LLM referee (interface).

One written standard drives both: the fraud policy's clause-cited evidence
conditions, evaluated on the as-of context (core.asof) at the decision time plus
the outcomes of at most two verification checks (core.actions.Check). Latent
truth may generate verification outcomes, at stated per-pattern rates keyed to
the order's stable id, but never chooses an action; sparse evidence stays
sparse. Neither the reviewer nor the referee may read labels or latent tables.

To be implemented here:

* ``classify(context_row, checks) -> Evidence``: the evidence families present
  (identity, device, geography, verification, velocity, linkage, repayment,
  disputes, merchant), each with the policy clause it satisfies and its
  strength;
* ``permitted_actions(evidence) -> dict``: permitted, required and prohibited
  dispositions, each citing a policy clause, used by the reviewer procedure
  and by the referee;
* the reviewer's procedure (which check to run next, which disposition to take),
  frozen with the generator parameters; if it outgrows about 250 lines or needs
  more than two checks, a stated confusion table replaces it (never-pay limited
  to clear or hold) and the switch is recorded.
"""
