"""Small, fast statistics for rates over task samples.

Outcomes are binary per task, so a bootstrap of the mean reduces to
binomial resampling. That keeps a thousand resamples under a millisecond
and makes every interval reproducible from a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

DEFAULT_RESAMPLES = 1000


@dataclass(slots=True, frozen=True)
class Rate:
    value: float
    low: float
    high: float
    n: int

    @property
    def per_ten(self) -> int:
        return int(round(self.value * 10))

    def as_dict(self) -> dict:
        return {"value": round(self.value, 4), "low": round(self.low, 4), "high": round(self.high, 4), "n": self.n}


def _binomial(rng: random.Random, n: int, p: float) -> int:
    bv = getattr(rng, "binomialvariate", None)
    if bv is not None:
        return bv(n, p)
    return sum(1 for _ in range(n) if rng.random() < p)


def rate(outcomes: Sequence[bool], seed: int = 7, resamples: int = DEFAULT_RESAMPLES) -> Rate:
    n = len(outcomes)
    if n == 0:
        return Rate(0.0, 0.0, 0.0, 0)
    k = sum(1 for o in outcomes if o)
    p = k / n
    rng = random.Random(seed)
    means = sorted(_binomial(rng, n, p) / n for _ in range(resamples))
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return Rate(p, lo, hi, n)


@dataclass(slots=True, frozen=True)
class PairedDelta:
    """Change in a rate between two runs over the same tasks."""

    delta: float
    low: float
    high: float
    n: int

    @property
    def significant(self) -> bool:
        return self.n > 0 and (self.low > 0 or self.high < 0)

    def as_dict(self) -> dict:
        return {"delta": round(self.delta, 4), "low": round(self.low, 4), "high": round(self.high, 4), "n": self.n, "significant": self.significant}


def paired_delta(before: Sequence[bool], after: Sequence[bool], seed: int = 7, resamples: int = DEFAULT_RESAMPLES) -> PairedDelta:
    n = min(len(before), len(after))
    if n == 0:
        return PairedDelta(0.0, 0.0, 0.0, 0)
    diffs = [int(after[i]) - int(before[i]) for i in range(n)]
    mean = sum(diffs) / n
    rng = random.Random(seed)
    # Resample the multiset of {-1, 0, +1} differences by counts.
    neg = diffs.count(-1)
    pos = diffs.count(1)
    p_neg, p_pos = neg / n, pos / n
    means = []
    for _ in range(resamples):
        rn = _binomial(rng, n, p_neg)
        rp = _binomial(rng, n, p_pos)
        means.append((rp - rn) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return PairedDelta(mean, lo, hi, n)


def confidence_word(n: int) -> str:
    if n >= 150:
        return "high"
    if n >= 60:
        return "moderate"
    if n >= 20:
        return "low"
    return "very low"


def per_ten_phrase(value: float) -> str:
    k = int(round(value * 10))
    if k <= 0:
        return "almost never"
    if k >= 10:
        return "nearly always"
    return f"about {k} in 10"
