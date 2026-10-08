"""The case files' alerts and their facts.

The alerts are chosen from the canonical world by the rule pre-registered in
``experiments/protocol.yaml`` (``cases``): :mod:`cases.rule` reads it and applies it to
what the pipeline's replay kept from the incumbent's run. :mod:`cases.facts` writes each
case file's facts (``cases/facts/<file>.json``): the evidence at decision time from the
row the replay decided on, the recorded action, what happened later under the incumbent
and under approve-all, and the simulation's latent truth as a separate diagnostic.
:mod:`cases.changes` runs one change to the operating policy per case file through the
replay. ``python -m cases.select --run <run directory>`` rebuilds all of it.
"""
