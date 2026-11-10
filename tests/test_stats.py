import math
from fractions import Fraction

import pytest

from harness.stats import (
    bootstrap_diff_ci,
    mcnemar_exact,
    mean,
    median,
    min_significant_net,
    percentile,
    wilson_interval,
)


def brute_mcnemar_p(b: int, c: int) -> float:
    """Independent check: two-sided exact binomial p with Fractions, summing every outcome as unlikely as observed."""
    n = b + c
    if n == 0:
        return 1.0
    pmf = [Fraction(math.comb(n, k), 2**n) for k in range(n + 1)]
    return float(min(Fraction(1), sum(p for p in pmf if p <= pmf[min(b, c)])))


@pytest.mark.parametrize("b,c", [(0, 0), (1, 0), (5, 0), (10, 0), (7, 1), (3, 3), (2, 9), (20, 11), (1, 1)])
def test_mcnemar_matches_independent_computation(b, c):
    assert mcnemar_exact(b, c).p_value == pytest.approx(brute_mcnemar_p(b, c), abs=1e-12)


def test_mcnemar_known_values():
    assert mcnemar_exact(7, 1).p_value == pytest.approx(18 / 256)  # 2 * (1 + 8) / 2^8
    assert mcnemar_exact(5, 0).p_value == pytest.approx(2 / 32)
    assert mcnemar_exact(10, 0).p_value == pytest.approx(2 / 1024)
    assert mcnemar_exact(0, 0).p_value == 1.0
    assert mcnemar_exact(4, 4).p_value == 1.0


def test_mcnemar_is_symmetric_and_reports_net():
    a, b = mcnemar_exact(9, 2), mcnemar_exact(2, 9)
    assert a.p_value == b.p_value
    assert a.net == -7 and b.net == 7
    assert a.flips == 11 and a.significant is (a.p_value < 0.05)


def test_mcnemar_rejects_negative():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_mcnemar_large_counts_do_not_overflow():
    assert 0 < mcnemar_exact(400, 300).p_value < 1


@pytest.mark.parametrize(
    "n,expected", [(0, None), (1, None), (5, None), (6, 6), (7, 7), (8, 8), (9, 7), (10, 8), (20, 10)]
)
def test_min_significant_net(n, expected):
    assert min_significant_net(n) == expected


def test_min_significant_net_is_the_boundary():
    for n in range(6, 40):
        d = min_significant_net(n)
        assert mcnemar_exact((n - d) // 2, (n + d) // 2).p_value < 0.05
        if d >= 2:  # one step smaller must not be significant
            assert mcnemar_exact((n - d + 2) // 2, (n + d - 2) // 2).p_value >= 0.05


def test_wilson_known_values():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(60, 60)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.9398, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_bootstrap_is_exact_so_it_has_no_seed_and_never_wobbles():
    a = [True] * 40 + [False] * 20
    b = [True] * 45 + [False] * 15
    one = bootstrap_diff_ci(a, b, n_boot=10, seed=3)
    two = bootstrap_diff_ci(a, b, n_boot=99_999, seed=4)
    assert one.exact and one.n_boot == 0
    assert (one.diff, one.low, one.high) == (two.diff, two.low, two.high)  # settings do not matter
    assert one.diff == pytest.approx(5 / 60)


def test_bootstrap_interval_matches_the_normal_approximation():
    # 10 cases fixed, 2 broken, out of 60: SE of the paired difference is sqrt(b + c - (b - c)^2 / n) / n
    a = [False] * 10 + [True] * 2 + [True] * 24 + [False] * 24
    b = [True] * 10 + [False] * 2 + [True] * 24 + [False] * 24
    n = len(a)
    se = math.sqrt(12 - 64 / n) / n
    bs = bootstrap_diff_ci(a, b, n_boot=10_000, seed=0)
    assert bs.diff == pytest.approx(8 / 60)
    assert bs.low == pytest.approx(bs.diff - 1.96 * se, abs=0.02)
    assert bs.high == pytest.approx(bs.diff + 1.96 * se, abs=0.02)
    assert bs.excludes_zero


def test_bootstrap_identical_runs_have_zero_width_interval():
    a = [True, False, True, True]
    bs = bootstrap_diff_ci(a, a, n_boot=500)
    assert (bs.diff, bs.low, bs.high) == (0.0, 0.0, 0.0)
    assert not bs.excludes_zero


def test_bootstrap_even_flips_include_zero():
    a = [True] * 5 + [False] * 5 + [True] * 50
    b = [False] * 5 + [True] * 5 + [True] * 50
    bs = bootstrap_diff_ci(a, b, n_boot=4000)
    assert bs.low < 0 < bs.high


def test_bootstrap_input_checks():
    with pytest.raises(ValueError):
        bootstrap_diff_ci([True], [True, False])
    assert bootstrap_diff_ci([], []).n_boot == 0


def test_percentile_matches_project_1_nearest_rank():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(vals, 50) == 5 and percentile(vals, 95) == 10 and percentile(vals, 0) == 1
    assert percentile([], 50) == 0.0
    assert median([1, 2, 3, 4]) == 2.5 and median([3, 1, 2]) == 2 and median([]) == 0.0
    assert mean([1, 2, 3]) == 2 and mean([]) == 0.0


def brute_force_bootstrap(a, b, level=0.95):
    """Independent check: enumerate every sequence of n draws (3^n of them) with its probability."""
    import itertools

    n = len(a)
    d = [int(y) - int(x) for x, y in zip(a, b, strict=True)]
    probs = {1: d.count(1) / n, -1: d.count(-1) / n, 0: d.count(0) / n}
    pmf: dict[int, float] = {}
    for draw in itertools.product((1, -1, 0), repeat=n):
        p = 1.0
        for x in draw:
            p *= probs[x]
        pmf[sum(draw)] = pmf.get(sum(draw), 0.0) + p
    tail = (1 - level) / 2
    cdf, lo, hi = 0.0, None, None
    for s in sorted(pmf):
        cdf += pmf[s]
        if lo is None and cdf >= tail - 1e-12:
            lo = s
        if hi is None and cdf >= 1 - tail - 1e-12:
            hi = s
    return lo / n, hi / n, pmf


@pytest.mark.parametrize(
    "fixed,regressed,same", [(3, 2, 3), (0, 4, 4), (5, 0, 3), (1, 1, 6), (0, 0, 8), (4, 4, 0)]
)
def test_exact_bootstrap_matches_brute_force_enumeration(fixed, regressed, same):
    a = [False] * fixed + [True] * regressed + [True] * same
    b = [True] * fixed + [False] * regressed + [True] * same
    lo, hi, pmf = brute_force_bootstrap(a, b)
    got = bootstrap_diff_ci(a, b)
    assert (got.low, got.high) == pytest.approx((lo, hi))
    from harness.stats import bootstrap_pmf

    dp = bootstrap_pmf(len(a), fixed, regressed)
    assert all(dp[s + len(a)] == pytest.approx(p, abs=1e-12) for s, p in pmf.items())
    assert sum(dp) == pytest.approx(1.0)


def test_exact_bootstrap_at_a_discrete_boundary_is_stable_where_monte_carlo_can_wobble():
    """4 fixed, 0 regressed of 60: P(no fixed case drawn) = (56/60)^60, about 1.6%, so the lower end sits exactly
    on a lump at 0 and a Monte Carlo percentile near 2.5% would flip between 0 and 1 case from seed to seed."""
    a = [False] * 4 + [True] * 56
    b = [True] * 60
    exact = bootstrap_diff_ci(a, b)
    assert exact.low == pytest.approx(1 / 60) and exact.excludes_zero  # 1.6% < 2.5%: the interval excludes 0
    # the same run twice, with a different seed and count, gives the identical interval
    other = bootstrap_diff_ci(a, b, n_boot=3, seed=99)
    assert (other.low, other.high) == (exact.low, exact.high)


def test_monte_carlo_fallback_agrees_with_the_exact_answer(monkeypatch):
    import harness.stats as stats

    a = [False] * 6 + [True] * 4 + [True] * 40
    b = [True] * 6 + [False] * 4 + [True] * 40
    exact = bootstrap_diff_ci(a, b)
    monkeypatch.setattr(stats, "EXACT_MAX_N", 10)  # force the fallback for a 50-case run
    mc = bootstrap_diff_ci(a, b, n_boot=200_000, seed=11)
    assert not mc.exact and mc.n_boot == 200_000
    assert mc.low == pytest.approx(exact.low, abs=0.021) and mc.high == pytest.approx(exact.high, abs=0.021)
