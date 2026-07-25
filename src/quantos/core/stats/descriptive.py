"""Descriptive statistics, including streaming estimators.

The streaming estimators matter more than they look. A simulation produces
events one at a time and may run for tens of millions of steps; storing every
observation to compute a variance at the end is both wasteful and, for a live
system, impossible. :class:`RunningMoments` maintains mean through kurtosis in
constant memory using Welford's update, which is numerically stable where the
textbook ``E[X^2] - E[X]^2`` catastrophically is not.

How bad is the naive formula? For ``x`` near 1e9 with unit variance, the two
terms agree to 18 digits and their difference is pure rounding noise -- the
naive estimator routinely returns a *negative* variance. Welford's never does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "RunningMoments",
    "autocorrelation",
    "ewm_std",
    "ewma",
    "hill_estimator",
    "kurtosis",
    "quantile",
    "rolling_apply",
    "skewness",
    "winsorise",
]


@dataclass
class RunningMoments:
    """Streaming mean/variance/skew/kurtosis via Welford's online algorithm.

    Supports :meth:`merge`, so partial results from parallel workers combine
    exactly (Chan-Golub-LeVeque parallel update). That is what makes the
    estimator usable across a multiprocess simulation.

    Example
        >>> import numpy as np
        >>> rm = RunningMoments()
        >>> for v in [1.0, 2.0, 3.0, 4.0]:
        ...     rm.update(v)
        >>> rm.mean, round(rm.variance, 10)
        (2.5, 1.6666666667)
    """

    count: int = 0
    _m1: float = 0.0
    _m2: float = 0.0
    _m3: float = 0.0
    _m4: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, x: float) -> None:
        """Absorb one observation in O(1) time and memory."""
        n1 = self.count
        self.count += 1
        n = self.count
        delta = x - self._m1
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term = delta * delta_n * n1

        self._m1 += delta_n
        self._m4 += (
            term * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * self._m2 - 4 * delta_n * self._m3
        )
        self._m3 += term * delta_n * (n - 2) - 3 * delta_n * self._m2
        self._m2 += term

        self.minimum = min(self.minimum, x)
        self.maximum = max(self.maximum, x)

    def update_many(self, xs: ArrayLike) -> None:
        """Absorb an array. Equivalent to repeated :meth:`update`."""
        for x in np.asarray(xs, dtype=float).ravel():
            self.update(float(x))

    def merge(self, other: RunningMoments) -> RunningMoments:
        """Combine two independent accumulators exactly (not approximately)."""
        if other.count == 0:
            return self
        if self.count == 0:
            return other

        na, nb = self.count, other.count
        n = na + nb
        delta = other._m1 - self._m1
        d2, d3, d4 = delta**2, delta**3, delta**4

        m1 = (na * self._m1 + nb * other._m1) / n
        m2 = self._m2 + other._m2 + d2 * na * nb / n
        m3 = (
            self._m3
            + other._m3
            + d3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * other._m2 - nb * self._m2) / n
        )
        m4 = (
            self._m4
            + other._m4
            + d4 * na * nb * (na * na - na * nb + nb * nb) / (n**3)
            + 6.0 * d2 * (na * na * other._m2 + nb * nb * self._m2) / (n * n)
            + 4.0 * delta * (na * other._m3 - nb * self._m3) / n
        )
        return RunningMoments(
            count=n,
            _m1=m1,
            _m2=m2,
            _m3=m3,
            _m4=m4,
            minimum=min(self.minimum, other.minimum),
            maximum=max(self.maximum, other.maximum),
        )

    @property
    def mean(self) -> float:
        return self._m1 if self.count else float("nan")

    @property
    def variance(self) -> float:
        """Unbiased (``ddof=1``) sample variance."""
        return self._m2 / (self.count - 1) if self.count > 1 else float("nan")

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def skewness(self) -> float:
        """Population (biased) skewness, matching ``scipy.stats.skew`` default."""
        if self.count < 3 or self._m2 == 0.0:
            return float("nan")
        return float(np.sqrt(self.count) * self._m3 / self._m2**1.5)

    @property
    def excess_kurtosis(self) -> float:
        """Excess kurtosis (0 for a Gaussian)."""
        if self.count < 4 or self._m2 == 0.0:
            return float("nan")
        return float(self.count * self._m4 / (self._m2 * self._m2) - 3.0)


def ewma(
    x: ArrayLike, halflife: float | None = None, *, alpha: float | None = None, adjust: bool = True
) -> NDArray[np.float64]:
    r"""Exponentially weighted moving average.

    Specify exactly one of ``halflife`` (intuitive: the lag at which weight
    halves, :math:`\alpha = 1 - 2^{-1/h}`) or ``alpha`` directly.

    ``adjust=True`` divides by the finite sum of weights, correcting the
    downward bias at the start of the series. Set it ``False`` only when
    reproducing a recursive filter that a live system will run.
    """
    x = np.asarray(x, dtype=float)
    if (halflife is None) == (alpha is None):
        raise ValueError("specify exactly one of halflife or alpha")
    if halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must be positive")
        alpha = 1.0 - 2.0 ** (-1.0 / halflife)
    assert alpha is not None
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")

    out = np.empty_like(x)
    if adjust:
        num = 0.0
        den = 0.0
        w = 1.0
        for i, v in enumerate(x):
            num = v + (1.0 - alpha) * num
            den = 1.0 + (1.0 - alpha) * den
            out[i] = num / den
        del w
    else:
        out[0] = x[0]
        for i in range(1, x.size):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def ewm_std(x: ArrayLike, halflife: float) -> NDArray[np.float64]:
    """Exponentially weighted standard deviation (the RiskMetrics estimator)."""
    x = np.asarray(x, dtype=float)
    mean = ewma(x, halflife=halflife)
    var = ewma((x - mean) ** 2, halflife=halflife)
    return np.sqrt(np.maximum(var, 0.0))


def skewness(x: ArrayLike, *, bias: bool = True) -> float:
    """Sample skewness; ``bias=False`` applies the G1 small-sample correction."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        return float("nan")
    m = x.mean()
    m2 = np.mean((x - m) ** 2)
    m3 = np.mean((x - m) ** 3)
    if m2 == 0.0:
        return float("nan")
    g1 = m3 / m2**1.5
    if bias:
        return float(g1)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def kurtosis(x: ArrayLike, *, excess: bool = True, bias: bool = True) -> float:
    """Sample kurtosis, excess by default."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 4:
        return float("nan")
    m = x.mean()
    m2 = np.mean((x - m) ** 2)
    m4 = np.mean((x - m) ** 4)
    if m2 == 0.0:
        return float("nan")
    g2 = m4 / (m2 * m2)
    if not bias:
        g2 = ((n + 1) * (g2 - 3.0) + 6.0) * (n - 1) / ((n - 2) * (n - 3)) + 3.0
    return float(g2 - 3.0) if excess else float(g2)


def autocorrelation(x: ArrayLike, max_lag: int, *, demean: bool = True) -> NDArray[np.float64]:
    r"""Sample autocorrelation function for lags ``0..max_lag``.

    Uses the *biased* (divide by :math:`n`, not :math:`n-k`) estimator. That is
    the standard choice in time-series work despite the name: it guarantees the
    resulting autocorrelation sequence is positive semi-definite, which the
    unbiased version does not, and a non-PSD ACF makes downstream spectral and
    Yule-Walker computations produce negative variances.

    Computed by FFT in :math:`O(n \log n)` -- necessary because the stylised-
    facts analysis in :mod:`quantos.sim.stylized_facts` needs 1000+ lags on
    millions of observations.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if max_lag >= n:
        raise ValueError(f"max_lag {max_lag} must be < series length {n}")
    z = x - x.mean() if demean else x

    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(z, size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: max_lag + 1] / n
    if acov[0] == 0.0:
        return np.zeros(max_lag + 1)
    return np.asarray(acov / acov[0], dtype=np.float64)


def quantile(x: ArrayLike, q: ArrayLike, *, method: str = "linear") -> NDArray[np.float64]:
    """Empirical quantiles of a sample.

    A thin wrapper over NumPy, present so that callers depend on the QuantOS
    surface rather than on NumPy's default interpolation method changing.
    """
    return np.asarray(
        np.quantile(np.asarray(x, dtype=float), np.asarray(q, dtype=float), method=method),  # type: ignore[call-overload]
        dtype=np.float64,
    )


def winsorise(x: ArrayLike, lower: float = 0.01, upper: float = 0.99) -> NDArray[np.float64]:
    """Clip to empirical quantiles rather than dropping outliers.

    Preferred to trimming in return series: dropping observations breaks the
    time index and silently changes the sample size of every downstream test,
    whereas clipping preserves both.
    """
    x = np.asarray(x, dtype=float)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("require 0 <= lower < upper <= 1")
    lo, hi = np.quantile(x, [lower, upper])
    return np.asarray(np.clip(x, lo, hi), dtype=np.float64)


def rolling_apply(
    x: ArrayLike, window: int, func: Callable[[NDArray[np.float64]], float]
) -> NDArray[np.float64]:
    """Apply ``func`` to each rolling window, via a zero-copy strided view.

    ``sliding_window_view`` creates no copies, so a 10-million-point series
    with a 250-day window costs no extra memory.
    """
    x = np.asarray(x, dtype=float).ravel()
    if window < 1 or window > x.size:
        raise ValueError(f"window must lie in [1, {x.size}], got {window}")
    views = np.lib.stride_tricks.sliding_window_view(x, window)
    out = np.full(x.size, np.nan)
    out[window - 1 :] = np.asarray([func(v) for v in views], dtype=float)
    return out


def hill_estimator(x: ArrayLike, k: int | None = None) -> float:
    r"""Hill estimator of the tail index :math:`\alpha` of a power-law tail.

    For order statistics :math:`X_{(1)} \ge \cdots \ge X_{(n)}`,

    .. math::
        \hat\alpha_k^{-1} = \frac{1}{k}\sum_{i=1}^{k}
            \log X_{(i)} - \log X_{(k+1)}

    Applied to :math:`|r|`, this is *the* measurement behind the claim that
    financial returns have a tail index near 3 -- finite variance, infinite
    kurtosis. :mod:`quantos.sim.stylized_facts` uses it to check that the
    simulated market reproduces the inverse cubic law rather than the Gaussian
    tails its component agents were built from.

    ``k`` defaults to :math:`\lfloor \sqrt{n} \rfloor`, a common compromise:
    the estimator's bias grows with ``k`` while its variance falls, and there
    is no universally optimal choice. Report a Hill plot over a range of ``k``
    before trusting any single value.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x) & (x > 0.0)]
    n = x.size
    if n < 10:
        return float("nan")
    if k is None:
        k = max(2, int(np.sqrt(n)))
    k = min(k, n - 1)
    order = np.sort(x)[::-1]
    logs = np.log(order[:k]) - np.log(order[k])
    mean_excess = float(np.mean(logs))
    return float("inf") if mean_excess == 0.0 else 1.0 / mean_excess
