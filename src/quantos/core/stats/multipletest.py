"""Multiple-testing corrections and tests of superior predictive ability.

The problem this module exists to solve
---------------------------------------
Search 1,000 strategy configurations on the same data and the best in-sample
Sharpe ratio will be impressive even when every configuration is worthless. At
the 5% level you expect 50 spurious "discoveries". This is not a subtlety --
it is the dominant failure mode of quantitative research, and the reason most
published anomalies do not survive out of sample.

Two families of tool, answering different questions:

**Family-wise / FDR corrections** (:func:`bonferroni`, :func:`holm`,
:func:`benjamini_hochberg`, :func:`benjamini_yekutieli`) adjust a set of
p-values you already have.

**Data-snooping tests** (:func:`whites_reality_check`, :func:`hansen_spa`,
:func:`stepm`) ask the sharper question: *is the best of these N strategies
better than the benchmark, accounting for the fact that I picked it by looking?*
They bootstrap the whole cross-section jointly, so they capture the correlation
between strategies -- which Bonferroni ignores, making it hopelessly
conservative when your 1,000 strategies are really 5 ideas with 200 parameter
settings each.

Hansen's SPA improves on White's Reality Check by recentring the studentised
statistics and excluding strategies that are so poor they cannot plausibly be
best. White's test is sensitive to padding the universe with rubbish; SPA is
not, and is the one to prefer.

References
----------
White, H. (2000), "A reality check for data snooping", *Econometrica* 68(5).
Hansen, P. R. (2005), "A test for superior predictive ability", *JBES* 23(4).
Romano, J. P. & Wolf, M. (2005), "Stepwise multiple testing as formalized data
    snooping", *Econometrica* 73(4), 1237-1282.
Benjamini, Y. & Hochberg, Y. (1995), *JRSS-B* 57(1), 289-300.
Benjamini, Y. & Yekutieli, D. (2001), *Ann. Statist.* 29(4), 1165-1188.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MultipleTestResult",
    "SnoopingResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni",
    "hansen_spa",
    "holm",
    "stepm",
    "whites_reality_check",
]


@dataclass(frozen=True)
class MultipleTestResult:
    """Adjusted p-values and rejection flags from a correction procedure."""

    method: str
    p_values: NDArray[np.float64]
    adjusted: NDArray[np.float64]
    rejected: NDArray[np.bool_]
    alpha: float

    @property
    def n_rejected(self) -> int:
        return int(np.sum(self.rejected))

    @property
    def n_tests(self) -> int:
        return int(self.p_values.size)


@dataclass(frozen=True)
class SnoopingResult:
    """Outcome of a data-snooping test over a universe of strategies."""

    method: str
    #: Studentised performance of the best strategy.
    statistic: float
    p_value: float
    best_index: int
    n_strategies: int
    n_bootstrap: int
    detail: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display
        return (
            f"{self.method}: best strategy #{self.best_index} of "
            f"{self.n_strategies}, stat={self.statistic:.4f}, p={self.p_value:.4f}"
        )


def _as_p(p: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(p, dtype=float).ravel()
    if np.any((a < 0.0) | (a > 1.0)) or not np.all(np.isfinite(a)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    return a


def bonferroni(p_values: ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    r"""Bonferroni correction: :math:`p_i^{adj} = \min(1, m p_i)`.

    Controls the family-wise error rate under *any* dependence structure, which
    is its virtue and its problem: that generality makes it very conservative
    when tests are correlated. With 1,000 highly-correlated strategy variants it
    will find nothing. Use it as a sanity floor, not as the primary tool.
    """
    p = _as_p(p_values)
    adjusted = np.minimum(1.0, p * p.size)
    return MultipleTestResult("Bonferroni", p, adjusted, adjusted <= alpha, alpha)


def holm(p_values: ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    r"""Holm-Bonferroni step-down: uniformly more powerful than Bonferroni.

    Sorts p-values ascending and compares :math:`p_{(i)}` to
    :math:`\alpha/(m-i+1)`, stopping at the first failure. Controls FWER under
    arbitrary dependence, strictly dominating Bonferroni -- there is no reason
    to prefer Bonferroni on power grounds.
    """
    p = _as_p(p_values)
    m = p.size
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return MultipleTestResult("Holm", p, adjusted, adjusted <= alpha, alpha)


def benjamini_hochberg(p_values: ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    r"""Benjamini-Hochberg FDR control.

    Controls the *expected proportion of false discoveries* rather than the
    probability of any. That is usually the right target in strategy screening:
    you are not trying to guarantee zero false positives, you are trying to
    keep the shortlist mostly real. Valid under independence and positive
    regression dependence.
    """
    p = _as_p(p_values)
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    # Step-up: enforce monotonicity from the largest p-value downward.
    adjusted_sorted = np.minimum.accumulate((m / ranks * p[order])[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return MultipleTestResult("Benjamini-Hochberg", p, adjusted, adjusted <= alpha, alpha)


def benjamini_yekutieli(p_values: ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    r"""Benjamini-Yekutieli FDR control, valid under *arbitrary* dependence.

    Inflates BH by the harmonic factor :math:`c(m) = \sum_{i=1}^{m} 1/i`. Use it
    when strategies may be negatively correlated, which breaks BH's positive-
    dependence assumption -- long/short variants of the same signal are the
    obvious case.
    """
    p = _as_p(p_values)
    m = p.size
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate((c_m * m / ranks * p[order])[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return MultipleTestResult("Benjamini-Yekutieli", p, adjusted, adjusted <= alpha, alpha)


# --------------------------------------------------------------------------- #
# Data-snooping tests                                                         #
# --------------------------------------------------------------------------- #
def _bootstrap_index_matrix(
    n: int, n_bootstrap: int, block_length: float, rng: np.random.Generator
) -> NDArray[np.intp]:
    from quantos.core.stats.bootstrap import block_indices

    return np.stack(
        [block_indices(n, block_length, rng, stationary=True) for _ in range(n_bootstrap)]
    )


def whites_reality_check(
    performance: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    rng: np.random.Generator | None = None,
) -> SnoopingResult:
    r"""White's (2000) Reality Check for data snooping.

    Purpose
        Test :math:`H_0: \max_k \mathbb{E}[f_k] \le 0` -- that *no* strategy in
        the universe beats the benchmark -- against the alternative that at
        least one does, correctly accounting for the selection of the best.
    Inputs
        ``performance`` -- ``(T, K)`` array of per-period performance
        *differentials* against the benchmark (e.g. strategy return minus
        benchmark return). Column ``k`` is strategy ``k``.
    Outputs
        :class:`SnoopingResult`; ``p_value`` is the bootstrap probability that
        the maximum studentised mean exceeds the observed one under the null.
    Method
        Stationary-bootstrap the *joint* cross-section so the strategies'
        contemporaneous correlation is preserved, recentre each column, and
        compare the observed :math:`\max_k \sqrt{T}\bar f_k` to its bootstrap
        distribution.
    Known limitation
        Sensitive to the composition of the universe: adding obviously terrible
        strategies raises the critical value and can hide a genuinely good one.
        :func:`hansen_spa` fixes precisely this and should be preferred.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> junk = rng.standard_normal((500, 20)) * 0.01      # no real edge
        >>> res = whites_reality_check(junk, n_bootstrap=200, rng=rng)
        >>> bool(res.p_value > 0.05)      # correctly finds nothing
        True
    """
    f = np.asarray(performance, dtype=float)
    if f.ndim != 2:
        raise ValueError("performance must be a 2-D (T, K) array")
    t, k = f.shape
    if t < 10 or k < 1:
        raise ValueError("need at least 10 periods and 1 strategy")
    if rng is None:
        from quantos.core.rng import SeedBank

        rng = SeedBank().child("reality_check").generator()
    if block_length is None:
        from quantos.core.stats.bootstrap import politis_white_block_length

        block_length = politis_white_block_length(f[:, 0])

    means = f.mean(axis=0)
    observed = float(np.max(np.sqrt(t) * means))
    best = int(np.argmax(means))

    idx = _bootstrap_index_matrix(t, n_bootstrap, block_length, rng)
    # Recentre by the sample mean: this imposes the null on the bootstrap
    # distribution, which is the whole point of the construction.
    boot_max = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        resampled = f[idx[b]]
        boot_max[b] = np.max(np.sqrt(t) * (resampled.mean(axis=0) - means))

    p = float(np.mean(boot_max >= observed))
    return SnoopingResult(
        "White's Reality Check",
        observed,
        p,
        best,
        k,
        n_bootstrap,
        detail={"block_length": float(block_length), "best_mean": float(means[best])},
    )


def hansen_spa(
    performance: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    rng: np.random.Generator | None = None,
) -> SnoopingResult:
    r"""Hansen's (2005) test for Superior Predictive Ability.

    Two improvements over :func:`whites_reality_check`, both material:

    1. **Studentisation.** Statistics are scaled by each strategy's own
       bootstrap standard deviation, so a high-variance strategy cannot
       dominate the maximum purely through noise.
    2. **Recentring only the plausible.** Strategies whose mean is worse than
       :math:`-\sqrt{\hat\omega_k^2 \, 2\log\log T / T}` are treated as
       having no chance of being best and are *not* recentred. This removes
       White's sensitivity to padding the universe with poor strategies.

    Returns the consistent p-value in ``p_value``; the lower and upper bounds
    (which bracket it and are cheap by-products) are in ``detail``.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> f = rng.standard_normal((600, 15)) * 0.01
        >>> f[:, 3] += 0.004                     # one genuinely good strategy
        >>> res = hansen_spa(f, n_bootstrap=300, rng=rng)
        >>> res.best_index, bool(res.p_value < 0.05)
        (3, True)
    """
    f = np.asarray(performance, dtype=float)
    if f.ndim != 2:
        raise ValueError("performance must be a 2-D (T, K) array")
    t, k = f.shape
    if t < 20:
        raise ValueError("SPA needs at least 20 periods")
    if rng is None:
        from quantos.core.rng import SeedBank

        rng = SeedBank().child("spa").generator()
    if block_length is None:
        from quantos.core.stats.bootstrap import politis_white_block_length

        block_length = politis_white_block_length(f[:, 0])

    means = f.mean(axis=0)
    idx = _bootstrap_index_matrix(t, n_bootstrap, block_length, rng)
    boot_means = np.stack([f[idx[b]].mean(axis=0) for b in range(n_bootstrap)])

    # omega_k^2: bootstrap variance of sqrt(T) * mean, the HAC-consistent
    # scale Hansen prescribes.
    omega = np.sqrt(np.maximum(np.var(np.sqrt(t) * boot_means, axis=0, ddof=1), 1e-300))

    observed = float(np.max(np.sqrt(t) * means / omega))
    best = int(np.argmax(means / omega))

    # Hansen's threshold rate A_T = sqrt(2 log log T).
    threshold = -np.sqrt(omega**2 * 2.0 * np.log(np.log(t)) / t)
    plausible = means >= threshold

    def bootstrap_p(centre: NDArray[np.float64]) -> float:
        z = np.sqrt(t) * (boot_means - centre[None, :]) / omega[None, :]
        return float(np.mean(np.max(np.maximum(z, 0.0), axis=1) >= max(observed, 0.0)))

    p_consistent = bootstrap_p(np.where(plausible, means, 0.0))
    p_lower = bootstrap_p(np.where(means >= 0.0, means, 0.0))
    p_upper = bootstrap_p(means)

    return SnoopingResult(
        "Hansen SPA",
        observed,
        p_consistent,
        best,
        k,
        n_bootstrap,
        detail={
            "p_lower": p_lower,
            "p_upper": p_upper,
            "block_length": float(block_length),
            "n_plausible": float(np.sum(plausible)),
            "best_mean": float(means[best]),
        },
    )


def stepm(
    performance: ArrayLike,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.bool_]:
    r"""Romano-Wolf StepM: identify *which* strategies beat the benchmark.

    :func:`hansen_spa` answers "is any strategy good?" -- a single yes/no. StepM
    answers the question you actually have, "which ones?", while still
    controlling the family-wise error rate. It proceeds in rounds: reject
    everything exceeding the bootstrap max-statistic critical value, remove the
    rejected set, recompute the critical value on the survivors, repeat until no
    further rejections. Removing confirmed winners lowers the bar for the
    remainder, which is where the power gain over a one-shot correction comes
    from.

    Returns a boolean mask of rejected (i.e. genuinely superior) strategies.
    """
    f = np.asarray(performance, dtype=float)
    if f.ndim != 2:
        raise ValueError("performance must be a 2-D (T, K) array")
    t, k = f.shape
    if rng is None:
        from quantos.core.rng import SeedBank

        rng = SeedBank().child("stepm").generator()
    if block_length is None:
        from quantos.core.stats.bootstrap import politis_white_block_length

        block_length = politis_white_block_length(f[:, 0])

    means = f.mean(axis=0)
    idx = _bootstrap_index_matrix(t, n_bootstrap, block_length, rng)
    boot_centred = np.stack(
        [np.sqrt(t) * (f[idx[b]].mean(axis=0) - means) for b in range(n_bootstrap)]
    )

    rejected = np.zeros(k, dtype=bool)
    active = np.ones(k, dtype=bool)
    statistics = np.sqrt(t) * means

    while np.any(active):
        critical = float(np.quantile(np.max(boot_centred[:, active], axis=1), 1.0 - alpha))
        newly = active & (statistics > critical)
        if not np.any(newly):
            break
        rejected |= newly
        active &= ~newly
    return rejected
