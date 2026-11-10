"""Statistics for paired pass/fail comparisons. Standard library only.

Two runs of the same cases give paired outcomes: for each case, did run A pass, did run B pass.
Cases that both pass or both fail carry no information about which run is better. Only the
discordant cases (the flips) do, so the tests below look only at them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

ALPHA = 0.05


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (same rule as project 1's evals): p in 0..100, empty list gives 0."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return float(s[idx])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (95% by default). Returns (low, high) in 0..1."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class McNemar:
    """Exact McNemar test on paired outcomes.

    only_a: cases A passed and B failed (B got worse).
    only_b: cases A failed and B passed (B got better).
    """

    only_a: int
    only_b: int
    p_value: float

    @property
    def flips(self) -> int:
        return self.only_a + self.only_b

    @property
    def net(self) -> int:
        """B minus A, in cases."""
        return self.only_b - self.only_a

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA


def _binom_two_sided(n: int, k_small: int) -> float:
    """Two-sided exact p for X ~ Binomial(n, 1/2): min(1, 2 * P(X <= k_small)). Exact integer arithmetic."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k_small + 1))
    return min(1.0, 2 * tail / 2**n)


def mcnemar_exact(only_a: int, only_b: int) -> McNemar:
    """Exact (binomial) McNemar test. With n = only_a + only_b flips, under 'the runs are equally good'
    each flip is equally likely to go either way, so only_b ~ Binomial(n, 1/2). Two-sided p-value.
    Exact rather than chi-square because flip counts are small (often under 10)."""
    if only_a < 0 or only_b < 0:
        raise ValueError("flip counts must be non-negative")
    n = only_a + only_b
    return McNemar(only_a, only_b, _binom_two_sided(n, min(only_a, only_b)))


def min_significant_net(n_flips: int, alpha: float = ALPHA) -> int | None:
    """Smallest net imbalance |only_b - only_a| that would be significant with n_flips flips.

    None when even a 100% one-sided split cannot reach significance (n_flips of 5 or fewer at 5%).
    """
    for d in range(n_flips % 2, n_flips + 1, 2):  # imbalance has the same parity as n_flips
        small = (n_flips - d) // 2
        if _binom_two_sided(n_flips, small) < alpha:
            return d
    return None


@dataclass(frozen=True)
class Bootstrap:
    diff: float  # mean(B) - mean(A) as a proportion of cases (0.05 = 5 points)
    low: float
    high: float
    n_boot: int  # Monte Carlo resamples; 0 when the distribution was computed exactly
    seed: int
    exact: bool = False

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


EXACT_MAX_N = 1500  # beyond this the exact computation is slow; fall back to Monte Carlo
_EPS = 1e-12


def bootstrap_pmf(n: int, fixed: int, regressed: int) -> list[float]:
    """Exact distribution of S = (cases fixed) - (cases regressed) in one resample of n paired cases.

    Each of the n draws picks a case at random: it is a fixed case with probability fixed/n, a regressed
    case with probability regressed/n, and otherwise changes nothing. So S is a sum of n i.i.d. draws from
    {+1, -1, 0} and its distribution is computed by convolution. Returns pmf[i] = P(S = i - n).
    """
    up, down = fixed / n, regressed / n
    stay = 1.0 - up - down
    pmf = [1.0]  # S = 0 before any draw; index i means S = i - draws
    for step in range(1, n + 1):
        new = [0.0] * (2 * step + 1)
        for i, p in enumerate(pmf):
            if p == 0.0:
                continue
            new[i] += p * down  # S - 1
            new[i + 1] += p * stay  # S unchanged
            new[i + 2] += p * up  # S + 1
        pmf = new
    return pmf


def bootstrap_diff_ci(
    a: list[bool], b: list[bool], n_boot: int = 100_000, seed: int = 0, level: float = 0.95
) -> Bootstrap:
    """Paired percentile bootstrap CI for accuracy(B) - accuracy(A).

    Cases are resampled with replacement as pairs, so the case-by-case pairing is kept. a and b must be
    aligned lists (same case order). With up to EXACT_MAX_N cases the bootstrap distribution is computed
    exactly (every possible resample, no random draws, no seed). That matters because the difference is a
    whole number of cases, so the distribution is lumpy and a Monte Carlo percentile at a lump boundary
    would wobble from run to run. Beyond that size, a seeded Monte Carlo with n_boot resamples is used.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")
    n = len(a)
    if n == 0:
        return Bootstrap(0.0, 0.0, 0.0, 0, seed, exact=True)
    d = [int(y) - int(x) for x, y in zip(a, b, strict=True)]  # per case: +1 fixed, -1 regressed, 0 same
    point = sum(d) / n
    tail = (1 - level) / 2
    if n <= EXACT_MAX_N:
        pmf = bootstrap_pmf(n, d.count(1), d.count(-1))
        lo_i = hi_i = None
        cdf = 0.0
        for i, p in enumerate(pmf):
            cdf += p
            if lo_i is None and cdf >= tail - _EPS:
                lo_i = i
            if hi_i is None and cdf >= 1 - tail - _EPS:
                hi_i = i
                break
        lo_i = 0 if lo_i is None else lo_i
        hi_i = len(pmf) - 1 if hi_i is None else hi_i
        return Bootstrap(point, (lo_i - n) / n, (hi_i - n) / n, 0, seed, exact=True)
    rng = random.Random(seed)
    sums = sorted(sum(rng.choices(d, k=n)) / n for _ in range(n_boot))
    lo = sums[max(0, math.floor(tail * n_boot))]
    hi = sums[min(n_boot - 1, math.ceil((1 - tail) * n_boot) - 1)]
    return Bootstrap(point, lo, hi, n_boot, seed)
