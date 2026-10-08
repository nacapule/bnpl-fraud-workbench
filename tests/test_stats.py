"""Statistics checked against hand-computed values and an independent library."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binomtest

from core.results import Interval, SeedSpread
from core.stats import (
    binomial_two_sided,
    cluster_bootstrap,
    mcnemar_exact,
    paired_outcomes,
    paired_seed_differences,
    seed_summary,
    sign_test,
    wilson_interval,
    z_score,
)


def test_z_score_is_the_normal_quantile() -> None:
    assert z_score(0.95) == pytest.approx(1.959963984540054, abs=1e-12)
    assert z_score(0.90) == pytest.approx(1.6448536269514722, abs=1e-12)


@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 1), (1, 1), (0, 10), (3, 10), (10, 10), (189, 2119), (147, 200), (42, 50), (1, 1000)],
)
def test_wilson_matches_an_independent_implementation(
    successes: int, trials: int, level: float
) -> None:
    ours = wilson_interval(successes, trials, level)
    theirs = binomtest(successes, trials).proportion_ci(confidence_level=level, method="wilson")
    assert isinstance(ours, Interval)
    assert (ours.method, ours.level) == ("wilson", level)
    assert ours.low == pytest.approx(float(theirs.low), abs=1e-12)
    assert ours.high == pytest.approx(float(theirs.high), abs=1e-12)


def test_wilson_hand_computed_case() -> None:
    # 0 of 10: upper bound z^2 / (n + z^2) = 3.8415 / 13.8415
    interval = wilson_interval(0, 10)
    # the centre and the half-width are equal in exact arithmetic; their difference can
    # round to a few times 1e-17 on some platforms
    assert interval.low == pytest.approx(0.0, abs=1e-12)
    assert interval.high == pytest.approx(z_score() ** 2 / (10 + z_score() ** 2), abs=1e-12)
    assert interval.high == pytest.approx(0.27753, abs=1e-5)


def test_wilson_stays_inside_the_unit_interval() -> None:
    for successes, trials in [(0, 3), (3, 3), (1, 1), (0, 1), (999, 1000)]:
        interval = wilson_interval(successes, trials)
        assert 0.0 <= interval.low <= interval.high <= 1.0


@pytest.mark.parametrize(
    ("successes", "trials"), [(5, 3), (0, 0), (-1, 4), (1.5, 4), (True, 4)]
)
def test_wilson_rejects_impossible_counts(successes, trials) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


def test_binomial_two_sided_matches_an_independent_implementation() -> None:
    for n in range(0, 41):
        for k in range(0, n + 1):
            expected = 1.0 if n == 0 else binomtest(k, n, 0.5).pvalue
            assert binomial_two_sided(k, n) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_mcnemar_uses_only_the_discordant_pairs() -> None:
    # 32 cases right under one arm only and 5 under the other only.
    assert mcnemar_exact(32, 5) == pytest.approx(7.4275e-6, rel=1e-4)
    assert mcnemar_exact(5, 32) == mcnemar_exact(32, 5)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 2**10)


def test_sign_test_gives_the_seed_counts_the_protocol_relies_on() -> None:
    nine_of_ten = sign_test([1.0] * 9 + [-1.0])
    assert (nine_of_ten.positive, nine_of_ten.negative, nine_of_ten.zero) == (9, 1, 0)
    assert nine_of_ten.p_value == pytest.approx(22 / 1024)  # about 0.02
    assert nine_of_ten.direction == "positive"
    five_of_five = sign_test([-3, -1, -2, -5, -4])
    assert five_of_five.p_value == pytest.approx(2 / 32)  # about 0.06
    assert five_of_five.direction == "negative"


def test_sign_test_excludes_zeros_from_the_test_but_counts_them() -> None:
    result = sign_test([2, 0, 0, 1, -1])
    assert (result.positive, result.negative, result.zero, result.n) == (2, 1, 2, 5)
    assert result.p_value == pytest.approx(binomtest(2, 3, 0.5).pvalue)
    assert sign_test([0, 0]).p_value == 1.0
    assert sign_test([0, 0]).direction == "none"


@pytest.mark.parametrize("bad", [[], [float("nan")], [math.inf], ["1"]])
def test_sign_test_rejects_bad_input(bad) -> None:
    with pytest.raises(ValueError):
        sign_test(bad)


def test_paired_seed_differences_pair_by_seed() -> None:
    hybrid = {3: 120, 1: 100, 2: 90}
    incumbent = {1: 80, 2: 95, 3: 100}
    spread = paired_seed_differences(hybrid, incumbent)
    assert isinstance(spread, SeedSpread)
    assert spread.per_seed == {1: 20, 2: -5, 3: 20}
    assert all(isinstance(value, int) for value in spread.per_seed.values())
    summary = seed_summary(spread)
    assert (summary.n, summary.min, summary.max) == (3, -5, 20)
    assert summary.mean == pytest.approx(35 / 3)
    assert (summary.test.positive, summary.test.negative) == (2, 1)
    assert seed_summary({1: 20, 2: -5, 3: 20}) == summary


def test_unpaired_seeds_are_an_error() -> None:
    with pytest.raises(ValueError, match="not paired"):
        paired_seed_differences({1: 1.0, 2: 2.0}, {1: 0.5, 3: 1.0})


def test_paired_outcomes_cross_tabulate_matched_cases() -> None:
    first = {"a": True, "b": True, "c": False, "d": False, "e": True}
    second = {"e": False, "d": False, "c": True, "b": True, "a": True}
    result = paired_outcomes(first, second)
    assert (result.both, result.only_first, result.only_second, result.neither) == (2, 1, 1, 1)
    assert result.n == 5
    assert result.first_rate == pytest.approx(3 / 5)
    assert result.second_rate == pytest.approx(3 / 5)
    assert result.p_value == mcnemar_exact(1, 1)


def test_paired_outcomes_refuse_unmatched_cases_and_non_booleans() -> None:
    with pytest.raises(ValueError, match="not matched"):
        paired_outcomes({"a": True, "b": False}, {"a": True})
    with pytest.raises(ValueError, match="booleans"):
        paired_outcomes({"a": 1}, {"a": True})
    with pytest.raises(ValueError, match="no cases"):
        paired_outcomes({}, {})


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    # Two clusters of identical rows: a resample holds 0, 1 or 2 copies of
    # cluster "a", so the mean is 0, 1/2 or 1 with probabilities 1/4, 1/2, 1/4.
    # At level 0.8 (10% in each tail) the interval is therefore [0, 1].
    values = np.array([1.0, 1.0, 0.0, 0.0])
    interval = cluster_bootstrap(
        ["a", "a", "b", "b"], lambda rows: values[rows].mean(), resamples=4000, seed=1, level=0.8
    )
    assert (interval.low, interval.high, interval.method) == (0.0, 1.0, "cluster_bootstrap")
    assert interval.level == 0.8
    # Each row its own cluster: the mean is Binomial(4, 1/2) / 4, with
    # P(0) = 1/16 and P(<= 1/4) = 5/16, so the 10% and 90% points are 1/4 and 3/4.
    rows_only = cluster_bootstrap(
        [0, 1, 2, 3], lambda rows: values[rows].mean(), resamples=4000, seed=1, level=0.8
    )
    assert (rows_only.low, rows_only.high) == (0.25, 0.75)


def test_cluster_bootstrap_is_seeded_and_order_free() -> None:
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 25, size=300)
    values = rng.normal(size=300)

    def mean(rows: np.ndarray) -> float:
        return float(values[rows].mean())

    first = cluster_bootstrap(clusters, mean, resamples=300, seed=7)
    assert cluster_bootstrap(clusters, mean, resamples=300, seed=7) == first
    assert cluster_bootstrap(clusters, mean, resamples=300, seed=8) != first
    assert first.low < values.mean() < first.high


def test_cluster_bootstrap_covers_the_truth_where_a_row_bootstrap_does_not() -> None:
    """Rows of one cluster share an effect, so they are not independent trials."""
    rng = np.random.default_rng(416)
    clusters = np.repeat(np.arange(30), 10)
    covered = {"clusters": 0, "rows": 0}
    replications = 150
    for _ in range(replications):
        values = rng.normal(size=30)[clusters] + rng.normal(scale=0.2, size=300)

        def mean(rows: np.ndarray, values: np.ndarray = values) -> float:
            return float(values[rows].mean())

        by_cluster = cluster_bootstrap(clusters, mean, resamples=400, seed=1)
        by_row = cluster_bootstrap(np.arange(300), mean, resamples=400, seed=1)
        covered["clusters"] += by_cluster.low <= 0.0 <= by_cluster.high
        covered["rows"] += by_row.low <= 0.0 <= by_row.high
    assert covered["clusters"] / replications >= 0.85
    assert covered["rows"] / replications <= 0.6


def test_cluster_labels_are_compared_as_they_are() -> None:
    """1 and "1" are two clusters; tuples are labels too."""
    values = np.array([0.0, 1.0])
    interval = cluster_bootstrap([1, "1"], lambda rows: values[rows].mean(), resamples=4000,
                                 seed=3, level=0.8)
    assert (interval.low, interval.high) == (0.0, 1.0)
    pairs = cluster_bootstrap([(1, "a"), (1, "a"), (2, "b"), (2, "b")],
                              lambda rows: np.array([1.0, 1.0, 0.0, 0.0])[rows].mean(),
                              resamples=4000, seed=3, level=0.8)
    assert (pairs.low, pairs.high) == (0.0, 1.0)


def test_cluster_bootstrap_does_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(5)
    clusters = rng.integers(0, 12, size=120)
    values = rng.normal(size=120)
    order = rng.permutation(120)

    def interval(labels, data):
        return cluster_bootstrap(labels, lambda rows: float(data[rows].mean()), resamples=500,
                                 seed=2)

    first = interval(clusters, values)
    second = interval(clusters[order], values[order])
    assert first.low == pytest.approx(second.low) and first.high == pytest.approx(second.high)


def test_cluster_bootstrap_rejects_undefined_statistics() -> None:
    with pytest.raises(ValueError, match="finite"):
        cluster_bootstrap([1, 2, 3], lambda rows: float("nan"), resamples=10)
    with pytest.raises(ValueError):
        cluster_bootstrap([], lambda rows: 0.0)
    with pytest.raises(ValueError):
        cluster_bootstrap([1, 2], lambda rows: 0.0, resamples=1)
