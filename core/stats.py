"""Statistics for published results (interface; moved here from analysis/uncertainty.py).

To be implemented here, each tested against hand-computed values:

* ``wilson_interval(successes, trials, level=0.95)``;
* ``mcnemar_exact(b, c)`` for paired binary outcomes (discordant counts);
* ``sign_test(differences)`` and ``seed_summary(per_seed)``: per-seed paired
  differences with mean, min-max and sign count, the main measure of variation
  across final seeds and families (core.results.SeedSpread);
* ``cluster_bootstrap(values, clusters, statistic, resamples, seed)``: intervals
  that resample users or episodes rather than orders.
"""
