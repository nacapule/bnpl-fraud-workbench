"""Merchants and how shoppers pick one.

Every actor picks merchants through :class:`Market`: by popularity among the
merchants trading at that moment, optionally restricted to some categories.
A bust-out merchant draws more traffic as its promotion ramps up.
"""

from __future__ import annotations

import numpy as np

from simulator.builder import DAY, Actor, Builder
from simulator.population import CATEGORIES

ADJECTIVES = ["urban", "nova", "prime", "lux", "peak", "true", "bright", "swift", "pure",
              "north", "blue", "gold", "iron", "cedar", "atlas", "amber", "coast", "maple"]
NOUNS = ["threads", "tech", "goods", "supply", "market", "collective", "haus", "works",
         "trading", "outfitters", "depot", "labs", "gallery", "province", "row", "studio"]


class Market:
    def __init__(self, b: Builder) -> None:
        self.b = b
        self.pks: list[int] = []
        self.weight: list[float] = []
        self.by_category: dict[str, list[int]] = {}
        self.ramp: dict[int, tuple[int, int]] = {}  # bust-out: (ramp start, full traffic)
        self._names: set[str] = set()
        self._arrays: tuple[np.ndarray, np.ndarray] | None = None

    def name(self, rng: np.random.Generator) -> str:
        while True:
            name = (f"{ADJECTIVES[int(rng.integers(0, len(ADJECTIVES)))]}-"
                    f"{NOUNS[int(rng.integers(0, len(NOUNS)))]}-{int(rng.integers(10, 99))}")
            if name not in self._names:
                self._names.add(name)
                return name

    def add(self, pk: int, weight: float, ramp: tuple[int, int] | None = None) -> None:
        self.pks.append(pk)
        self.weight.append(weight)
        self.by_category.setdefault(self.b.merchants[pk]["category"], []).append(len(self.pks) - 1)
        if ramp is not None:
            self.ramp[pk] = ramp
        self._arrays = None

    def _cdf(self) -> tuple[np.ndarray, np.ndarray]:
        if self._arrays is None:
            weight = np.array(self.weight)
            self._arrays = (np.array(self.pks), np.cumsum(weight) / weight.sum())
        return self._arrays

    def accepts(self, rng: np.random.Generator, pk: int, t: int) -> bool:
        if not self.b.merchant_open(pk, t):
            return False
        if pk in self.ramp:
            start, full = self.ramp[pk]
            share = min(1.0, max(0.0, (t - start) / max(1, full - start)))
            return rng.random() < share
        return True

    def choose(self, rng: np.random.Generator, t: int,
               categories: tuple[str, ...] | None = None) -> int:
        """A merchant trading at ``t``, by popularity (within ``categories`` if given)."""
        pks, cdf = self._cdf()
        if categories is not None:
            index = [i for c in categories for i in self.by_category.get(c, [])]
            weight = np.array([self.weight[i] for i in index])
            for _ in range(200):
                i = index[int(rng.choice(len(index), p=weight / weight.sum()))]
                if self.accepts(rng, self.pks[i], t):
                    return self.pks[i]
        for _ in range(500):
            pk = int(pks[min(int(np.searchsorted(cdf, rng.random(), side="right")),
                             len(pks) - 1)])
            if self.accepts(rng, pk, t):
                return pk
        raise RuntimeError("no merchant trading at this time")


def legit_merchants(b: Builder, market: Market, a: Actor, n: int, order_start: int,
                    order_end: int, since: int, onboarding_share: float,
                    median_hours: float, median_sigma: float) -> None:
    """The ordinary merchants: most trade from before the horizon, some join during it."""
    rng = a.rng
    names = [c[0] for c in CATEGORIES]
    weight = np.array([c[2] for c in CATEGORIES], dtype=float)
    popularity = rng.dirichlet(np.full(n, 0.6))
    for k in range(n):
        category = names[int(rng.choice(len(names), p=weight / weight.sum()))]
        if rng.random() < onboarding_share:
            created = order_start + int(rng.uniform(0, order_end - order_start - 60 * DAY))
        else:
            created = since + int(rng.uniform(0, order_start - since))
        hours = float(np.clip(np.exp(rng.normal(np.log(median_hours), median_sigma)), 2, 72))
        tier = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        pk = b.merchant(a, created, market.name(rng), category, tier, round(hours, 1))
        market.add(pk, float(popularity[k]))
