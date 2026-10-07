"""Statistics behind published results.

The main measure of variation is replication across worlds: every policy is
run on the same final seeds, so a comparison is a set of per-seed paired
differences, reported with their mean, min-max and sign count (for example
"positive on 9/10 seeds") and an exact two-sided sign test
(:func:`paired_seed_differences`, :func:`seed_summary`, :func:`sign_test`).
Within one world:

* :func:`wilson_interval` for a proportion of independent trials;
* :func:`mcnemar_exact` and :func:`paired_outcomes` for two methods judged on
  the same cases (both right, one right, neither), never two marginal rates;
* :func:`cluster_bootstrap` for a statistic over rows that are not
  independent (orders of one user, cases of one ring): it resamples whole
  clusters, so the interval reflects how many independent units there are.

Exact tests use the binomial distribution with p = 1/2, two-sided by doubling
the smaller tail (capped at 1). Inputs are validated; nothing here reads files.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from core.results import Interval, SeedSpread

Number = int | float


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{what} must be a non-negative integer, got {value!r}")
    return int(value)


def _level(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level!r}")
    return float(level)


def z_score(level: float = 0.95) -> float:
    """The two-sided standard normal quantile for a confidence level (1.96 at 0.95)."""
    return NormalDist().inv_cdf(0.5 + _level(level) / 2)


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> Interval:
    """Wilson score interval for ``successes`` out of ``trials`` independent trials.

    Preferred to the normal approximation because published rates sit near 0
    or 1 on small denominators, where the normal interval leaves [0, 1].
    """
    successes = _count(successes, "successes")
    trials = _count(trials, "trials")
    if trials == 0:
        raise ValueError("trials must be positive")
    if successes > trials:
        raise ValueError(f"successes ({successes}) exceed trials ({trials})")
    z = z_score(level)
    share = successes / trials
    scale = 1 + z * z / trials
    center = (share + z * z / (2 * trials)) / scale
    half = z * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials)) / scale
    return Interval(max(0.0, center - half), min(1.0, center + half), "wilson", level)


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value of ``k`` successes in ``n`` fair-coin trials."""
    k, n = _count(k, "k"), _count(n, "n")
    if k > n:
        raise ValueError(f"k ({k}) exceeds n ({n})")
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2**n)


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Exact McNemar p-value from the two discordant counts of a paired comparison.

    ``only_first`` cases went one way and ``only_second`` the other (for
    example, correct under the first prompt only, and under the second only).
    Under no difference the discordant cases split like a fair coin.
    """
    only_first = _count(only_first, "only_first")
    only_second = _count(only_second, "only_second")
    return binomial_two_sided(only_first, only_first + only_second)


@dataclass(frozen=True)
class SignTest:
    """Signs of paired differences and the exact two-sided sign test.

    Zeros are counted but excluded from the test, as usual for the sign test.
    """

    positive: int
    negative: int
    zero: int
    p_value: float

    @property
    def n(self) -> int:
        """All differences, zeros included."""
        return self.positive + self.negative + self.zero

    @property
    def direction(self) -> str:
        """``positive``, ``negative`` or ``none`` (a tie in signs)."""
        if self.positive > self.negative:
            return "positive"
        if self.negative > self.positive:
            return "negative"
        return "none"


def sign_test(differences: Iterable[Number]) -> SignTest:
    """Count the signs of paired differences and test them against a fair coin."""
    values = [_finite(value, "difference") for value in differences]
    if not values:
        raise ValueError("a sign test needs at least one difference")
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    return SignTest(positive, negative, zero, binomial_two_sided(positive, positive + negative))


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return number


def paired_seed_differences(
    treatment: Mapping[int, Number], reference: Mapping[int, Number]
) -> SeedSpread:
    """Per-seed ``treatment - reference``, paired by seed.

    Both sides must cover the same seeds: an unpaired seed is an error, not a
    seed to drop silently.
    """
    if set(treatment) != set(reference):
        only_treatment = sorted(set(treatment) - set(reference))
        only_reference = sorted(set(reference) - set(treatment))
        raise ValueError(
            f"seeds are not paired: only in treatment {only_treatment}, "
            f"only in reference {only_reference}"
        )
    return SeedSpread(
        {
            seed: _difference(treatment[seed], reference[seed], seed)
            for seed in treatment
        }
    )


def _difference(first: Number, second: Number, seed: int) -> Number:
    _finite(first, f"treatment value for seed {seed}")
    _finite(second, f"reference value for seed {seed}")
    both_int = isinstance(first, numbers.Integral) and isinstance(second, numbers.Integral)
    return int(first) - int(second) if both_int else float(first) - float(second)


@dataclass(frozen=True)
class SeedSummary:
    """What the documents report for a value replicated over seeds."""

    n: int
    mean: float
    min: Number
    max: Number
    test: SignTest


def seed_summary(per_seed: SeedSpread | Mapping[int, Number]) -> SeedSummary:
    """Mean, min-max, sign count and sign test of one value per seed."""
    spread = per_seed if isinstance(per_seed, SeedSpread) else SeedSpread(dict(per_seed))
    return SeedSummary(
        n=spread.n,
        mean=spread.mean,
        min=spread.min,
        max=spread.max,
        test=sign_test(spread.per_seed.values()),
    )


@dataclass(frozen=True)
class PairedOutcomes:
    """Two methods judged on the same cases, with the exact McNemar test."""

    both: int
    only_first: int
    only_second: int
    neither: int
    p_value: float

    @property
    def n(self) -> int:
        return self.both + self.only_first + self.only_second + self.neither

    @property
    def first_rate(self) -> float:
        return (self.both + self.only_first) / self.n

    @property
    def second_rate(self) -> float:
        return (self.both + self.only_second) / self.n


def paired_outcomes(
    first: Mapping[Hashable, bool], second: Mapping[Hashable, bool]
) -> PairedOutcomes:
    """Cross-tabulate two methods' binary outcomes on matched cases.

    Both mappings must hold the same case ids; to compare on a subset (for
    example the cases an arm completed), pass that subset explicitly.
    """
    if set(first) != set(second):
        raise ValueError(
            f"cases are not matched: {len(set(first) - set(second))} only in the first, "
            f"{len(set(second) - set(first))} only in the second"
        )
    if not first:
        raise ValueError("no cases to compare")
    for name, outcomes in (("first", first), ("second", second)):
        bad = [case for case, value in outcomes.items() if not isinstance(value, bool | np.bool_)]
        if bad:
            raise ValueError(f"{name} outcomes must be booleans; case {bad[0]!r} is not")
    both = sum(bool(first[case]) and bool(second[case]) for case in first)
    only_first = sum(bool(first[case]) and not bool(second[case]) for case in first)
    only_second = sum(not bool(first[case]) and bool(second[case]) for case in first)
    neither = len(first) - both - only_first - only_second
    return PairedOutcomes(
        both, only_first, only_second, neither, mcnemar_exact(only_first, only_second)
    )


def cluster_bootstrap(
    clusters: Sequence[Hashable] | np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Percentile bootstrap interval that resamples whole clusters.

    ``clusters`` gives each row's cluster (a user, an episode, a ring).
    Each resample draws as many clusters as there are, with replacement, and
    calls ``statistic`` with the row positions of the drawn clusters (a row
    repeats when its cluster is drawn more than once). The statistic must
    return a finite number for every resample; a statistic that can be
    undefined on a resample (for example with no positive rows) must handle
    that itself. The result depends only on the inputs and ``seed``.
    """
    labels = np.asarray(clusters)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("clusters must be a non-empty one-dimensional sequence")
    resamples = _count(resamples, "resamples")
    if resamples < 2:
        raise ValueError("resamples must be at least 2")
    level = _level(level)
    _, codes = np.unique(labels, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    count = len(sizes)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, count, size=count)
        lengths = sizes[chosen]
        starts = np.cumsum(lengths) - lengths
        shift = np.repeat(offsets[chosen] - starts, lengths)
        positions = np.arange(lengths.sum()) + shift
        value = statistic(order[positions])
        draws[index] = _finite(value, "statistic")
    tail = (1 - level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return Interval(float(low), float(high), "cluster_bootstrap", level)
