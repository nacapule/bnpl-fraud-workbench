"""Random draws keyed by stable ids, so every policy sees the same draws.

A draw is a pure function of the world's seed, a named stream and an entity's stable
id (an order id, for its review time or a check's outcome). Two policies that review
the same order therefore get the same service time and the same check outcomes,
whatever else they did, and a draw never depends on the order in which a replay
happens to ask for it.
"""

from __future__ import annotations

import hashlib
from statistics import NormalDist

import numpy as np

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_STANDARD = NormalDist()


def _stream_key(stream: str) -> np.uint64:
    digest = hashlib.blake2b(stream.encode(), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64's finaliser: a bijection that scrambles every input bit."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def keyed_uniform(seed: int, stream: str, keys: np.ndarray | list[int] | int) -> np.ndarray:
    """One uniform draw in [0, 1) per key, fixed by (seed, stream, key)."""
    ids = np.atleast_1d(np.asarray(keys)).astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        base = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) ^ _stream_key(stream)) & _MASK
        x = _mix(base + ids * np.uint64(0x9E3779B97F4A7C15))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def lognormal(uniforms: np.ndarray, *, median: float | None = None,
              mean: float | None = None, sigma: float) -> np.ndarray:
    """Lognormal values from uniforms, given the median or the arithmetic mean."""
    if (median is None) == (mean is None):
        raise ValueError("give exactly one of median and mean")
    mu = np.log(median) if median is not None else np.log(mean) - sigma**2 / 2
    clipped = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1 - 1e-12)
    z = np.array([_STANDARD.inv_cdf(u) for u in clipped.ravel()]).reshape(clipped.shape)
    return np.exp(mu + sigma * z)
