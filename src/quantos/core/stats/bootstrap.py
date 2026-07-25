"""Resampling for dependent data.

The i.i.d. bootstrap is wrong for return series
-----------------------------------------------
Financial returns are not independent: volatility clusters, and many series are
serially correlated. Resampling individual observations destroys that
dependence, so the resulting confidence intervals are too narrow -- often by a
large factor. Every interval in QuantOS that involves a time series therefore
uses a *block* bootstrap.

Which block scheme?
-------------------
=========================  ==================================================
:func:`iid_bootstrap`      Only for genuinely independent samples.
:func:`circular_block`     Fixed block length, wrapped so every observation
                           has equal resampling probability. Simple, and the
                           right default when you know the dependence length.
:func:`stationary_bootstrap` Politis-Romano: **geometric** random block
                           lengths. The resampled series is strictly
                           stationary, which the fixed-block schemes are not,
                           and results are far less sensitive to getting the
                           block-length parameter wrong. The default here.
=========================  ==================================================

:func:`politis_white_block_length` implements the automatic block-length
selector, so the tuning parameter has a defensible default rather than a
magic number.

References
----------
Politis, D. N. & Romano, J. P. (1994), "The stationary bootstrap",
    *JASA* 89(428), 1303-1313.
Politis, D. N. & White, H. (2004), "Automatic block-length selection for the
    dependent bootstrap", *Econometric Reviews* 23(1), 53-70.
Efron, B. & Tibshirani, R. (1993), *An Introduction to the Bootstrap*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BootstrapResult",
    "block_indices",
    "bootstrap_statistic",
    "circular_block_bootstrap",
    "iid_bootstrap",
    "politis_white_block_length",
    "stationary_bootstrap",
]


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution of a statistic, with intervals."""

    point_estimate: float
    replicates: NDArray[np.float64]
    method: str

    @property
    def standard_error(self) -> float:
        return float(np.std(self.replicates, ddof=1))

    @property
    def bias(self) -> float:
        """Bootstrap estimate of the statistic's bias."""
        return float(np.mean(self.replicates) - self.point_estimate)

    def percentile_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Percentile interval. Simple, but biased when the statistic is skewed."""
        alpha = 0.5 * (1.0 - level)
        lo, hi = np.quantile(self.replicates, [alpha, 1.0 - alpha])
        return float(lo), float(hi)

    def basic_interval(self, level: float = 0.95) -> tuple[float, float]:
        r"""Basic, or reverse-percentile, confidence interval.

        :math:`[2\hat\theta - \theta^*_{1-\alpha},\,
        2\hat\theta - \theta^*_{\alpha}]`.

        Corrects for first-order bias, which the percentile interval does not.
        """
        alpha = 0.5 * (1.0 - level)
        lo, hi = np.quantile(self.replicates, [alpha, 1.0 - alpha])
        return float(2.0 * self.point_estimate - hi), float(2.0 * self.point_estimate - lo)

    def bca_interval(
        self, level: float = 0.95, *, jackknife: NDArray[np.float64] | None = None
    ) -> tuple[float, float]:
        r"""Bias-corrected and accelerated (BCa) interval.

        Adjusts the percentile endpoints for both median bias
        (:math:`\hat z_0`) and skewness (the acceleration :math:`\hat a`,
        estimated from jackknife values when supplied). Second-order accurate,
        against the percentile interval's first order -- worth the extra cost
        for a Sharpe ratio, whose sampling distribution is visibly skewed.

        Without ``jackknife`` the acceleration is taken as zero, which reduces
        this to the bias-corrected (BC) interval.
        """
        from quantos.core.special import ndtr, ndtri

        theta = self.point_estimate
        reps = self.replicates
        prop = float(np.mean(reps < theta))
        prop = min(max(prop, 1e-6), 1.0 - 1e-6)
        z0 = float(ndtri(np.array(prop)))

        if jackknife is not None and jackknife.size > 2:
            diff = float(np.mean(jackknife)) - jackknife
            denom = 6.0 * float(np.sum(diff**2)) ** 1.5
            accel = float(np.sum(diff**3) / denom) if denom != 0 else 0.0
        else:
            accel = 0.0

        alpha = 0.5 * (1.0 - level)
        out: list[float] = []
        for a in (alpha, 1.0 - alpha):
            za = float(ndtri(np.array(a)))
            adjusted = z0 + (z0 + za) / (1.0 - accel * (z0 + za))
            out.append(float(np.quantile(reps, np.clip(ndtr(np.array(adjusted)), 0.0, 1.0))))
        return out[0], out[1]


def block_indices(
    n: int, block_length: float, rng: np.random.Generator, *, stationary: bool = True
) -> NDArray[np.intp]:
    r"""Resampled index array of length ``n``.

    ``stationary=True`` draws each block's length from
    :math:`\text{Geometric}(1/L)`, giving the Politis-Romano stationary
    bootstrap; ``False`` uses fixed-length circular blocks. Both wrap modulo
    ``n``, which equalises each observation's inclusion probability -- without
    wrapping, points near the ends are systematically under-sampled.
    """
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    idx = np.empty(n, dtype=np.intp)
    filled = 0
    p = 1.0 / block_length
    while filled < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p)) if stationary else round(block_length)
        length = max(1, min(length, n - filled))
        idx[filled : filled + length] = (start + np.arange(length)) % n
        filled += length
    return idx


def iid_bootstrap(data: ArrayLike, n_replicates: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Index matrix ``(n_replicates, n)`` of i.i.d. resamples.

    Correct only for independent data. Using it on a return series produces
    confidence intervals that are too narrow; prefer
    :func:`stationary_bootstrap`.
    """
    n = np.asarray(data).shape[0]
    return rng.integers(0, n, size=(n_replicates, n), dtype=np.intp)


def circular_block_bootstrap(
    data: ArrayLike, n_replicates: int, block_length: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Index matrix from fixed-length circular blocks."""
    n = np.asarray(data).shape[0]
    return np.stack(
        [block_indices(n, block_length, rng, stationary=False) for _ in range(n_replicates)]
    )


def stationary_bootstrap(
    data: ArrayLike, n_replicates: int, block_length: float, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Index matrix from geometric-length blocks (Politis-Romano)."""
    n = np.asarray(data).shape[0]
    return np.stack(
        [block_indices(n, block_length, rng, stationary=True) for _ in range(n_replicates)]
    )


def politis_white_block_length(x: ArrayLike, *, max_lag: int | None = None) -> float:
    r"""Automatic block length for the stationary bootstrap.

    Implements the Politis-White (2004) selector: the optimal length is

    .. math:: L^* = \left(\frac{2 \hat G^2}{\hat D}\right)^{1/3} n^{1/3}

    where :math:`\hat G = \sum_k |k| \hat\gamma_k` and
    :math:`\hat D = 2 \big(\sum_k \hat\gamma_k\big)^2` are formed from the
    autocovariances up to a data-dependent lag, chosen as the point beyond
    which the ACF is statistically indistinguishable from zero (the
    :math:`2\sqrt{\log_{10} n / n}` threshold).

    Returns a length clipped to :math:`[1, n/3]`: blocks longer than a third of
    the sample leave too few distinct resamples to be useful.
    """
    from quantos.core.stats.descriptive import autocorrelation

    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    if n < 20:
        return 1.0
    if max_lag is None:
        max_lag = min(n - 2, int(np.ceil(2.0 * np.sqrt(n))))

    rho = autocorrelation(a, max_lag)
    variance = float(np.var(a, ddof=1))
    gamma = rho * variance

    # Largest lag whose ACF exceeds the significance band; beyond it, treat the
    # dependence as exhausted.
    threshold = 2.0 * np.sqrt(np.log10(n) / n)
    significant = np.nonzero(np.abs(rho[1:]) > threshold)[0]
    m = int(significant[-1]) + 1 if significant.size else 1
    m = min(m, max_lag)

    lags = np.arange(-m, m + 1)
    gam = np.array([gamma[abs(int(k))] for k in lags])
    g_hat = float(np.sum(np.abs(lags) * gam))
    d_hat = 2.0 * float(np.sum(gam)) ** 2
    if d_hat <= 0:
        return 1.0
    length = (2.0 * g_hat**2 / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(length if np.isfinite(length) else 1.0, 1.0, max(1.0, n / 3.0)))


def bootstrap_statistic(
    data: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float],
    *,
    n_replicates: int = 2000,
    rng: np.random.Generator | None = None,
    method: str = "stationary",
    block_length: float | None = None,
) -> BootstrapResult:
    """Bootstrap any scalar statistic of a (possibly dependent) series.

    Purpose
        Confidence intervals for quantities with no tractable sampling
        distribution -- Sharpe ratio, maximum drawdown, tail index, a
        strategy's hit rate.
    Inputs
        ``statistic`` -- maps a resampled array to a scalar.
        ``method`` -- ``"stationary"`` (default), ``"circular"``, or ``"iid"``.
        ``block_length`` -- defaults to
        :func:`politis_white_block_length` for the block methods.
    Outputs
        :class:`BootstrapResult`.
    Complexity
        ``n_replicates`` evaluations of ``statistic``.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(500) + 0.1
        >>> res = bootstrap_statistic(x, np.mean, n_replicates=400, rng=rng)
        >>> lo, hi = res.percentile_interval(0.95)
        >>> bool(lo < 0.1 < hi)
        True
    """
    a = np.asarray(data, dtype=float)
    if rng is None:
        from quantos.core.rng import SeedBank

        rng = SeedBank().child("bootstrap").generator()
    if a.shape[0] < 3:
        raise ValueError("need at least 3 observations to bootstrap")

    if method == "iid":
        idx = iid_bootstrap(a, n_replicates, rng)
    else:
        if block_length is None:
            block_length = politis_white_block_length(a.ravel() if a.ndim == 1 else a[:, 0])
        if method == "stationary":
            idx = stationary_bootstrap(a, n_replicates, block_length, rng)
        elif method == "circular":
            idx = circular_block_bootstrap(a, n_replicates, round(block_length), rng)
        else:
            raise ValueError(f"unknown method {method!r}")

    replicates = np.array([float(statistic(a[row])) for row in idx])
    return BootstrapResult(
        point_estimate=float(statistic(a)),
        replicates=replicates,
        method=f"{method}(L={block_length:.2f})" if method != "iid" else "iid",
    )
