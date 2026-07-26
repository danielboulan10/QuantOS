r"""Intraday volatility estimation, where the naive estimator is worst.

The problem intraday data creates
----------------------------------
Realised variance -- summing squared returns -- is consistent for integrated
variance as sampling gets finer. That is a theorem, and it is true, and following
it literally is one of the most reliably wrong things you can do with tick data.

Observed prices are not the efficient price. They are the efficient price plus
**microstructure noise**: bid-ask bounce, price discreteness, order-splitting,
stale quotes. Write :math:`p_t = p^*_t + u_t`. Then each observed return contains
:math:`u_t - u_{t-1}`, and summing :math:`n` squared returns accumulates

.. math:: \mathbb{E}[RV_n] = IV + 2n\,\mathbb{E}[u^2]

The noise term grows **linearly in the number of observations**. Sample every
second instead of every five minutes and you get 300 times the noise
contribution. In the limit the estimator diverges: it converges not to integrated
variance but to twice the noise variance times the sample size. The estimator
that is consistent in theory is, at the frequencies tick data actually offers,
dominated by the thing it ignores.

This is visible, not hypothetical: :func:`signature_plot` draws it. Realised
volatility rises steeply as the sampling interval shortens, and the shape of that
rise identifies the noise.

What this module provides instead
----------------------------------
Three estimators that handle it, each for a different reason:

**Sparse sampling** (:func:`realized_variance` at five minutes) -- the practical
default for decades, and still a reasonable one. It works by discarding data
until the noise is small relative to the signal, which is inefficient but honest.
:func:`optimal_sampling_interval` computes where that trade-off actually sits
rather than assuming five minutes.

**Two-scale realised variance** (:func:`two_scale_realized_variance`) --
Zhang, Mykland and Aït-Sahalia's estimator, which uses *all* the data by
combining a slow-scale and a fast-scale estimate so the noise cancels to first
order. It is consistent in the presence of noise, which sparse sampling is not.

**Bipower variation** (:func:`bipower_variation`) -- robust to jumps rather than
to noise. Realised variance counts a jump as volatility; bipower variation does
not, and the difference between them identifies jump days
(:func:`jump_test`).

None of these is universally best, which is why all three are here and why
:func:`volatility_report` runs them together. When they disagree, the
disagreement is the information.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantos.core.special import lgamma, ndtr

__all__ = [
    "IntradayVolatility",
    "JumpTest",
    "Seasonality",
    "SignaturePlot",
    "bipower_variation",
    "epps_curve",
    "intraday_seasonality",
    "jump_test",
    "noise_is_detectable",
    "noise_variance",
    "optimal_sampling_interval",
    "realized_variance",
    "signature_plot",
    "two_scale_realized_variance",
    "volatility_report",
]

#: Scaling constant for bipower variation: :math:`\mu_1 = \mathbb{E}|Z| = \sqrt{2/\pi}`.
MU_1 = np.sqrt(2.0 / np.pi)

#: :math:`\mu_{4/3} = \mathbb{E}|Z|^{4/3} = 2^{2/3}\,\Gamma(7/6)/\Gamma(1/2)`,
#: the constant for tripower quarticity. Computed from this package's own
#: ``lgamma`` rather than hard-coded, so it cannot silently disagree with it.
MU_43 = float(2.0 ** (2.0 / 3.0) * np.exp(lgamma(np.array(7.0 / 6.0)) - lgamma(np.array(0.5))))


def _log_returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        return np.zeros(0)
    if np.any(prices <= 0):
        raise ValueError("intraday prices must be positive to take log returns")
    return np.diff(np.log(prices))


def _subsample(prices: NDArray[np.float64], step: int) -> NDArray[np.float64]:
    """Every ``step``-th price, always keeping the last observation.

    Keeping the close matters: dropping it discards the one price of the day
    that everything else is marked against, and biases the estimate whenever the
    series length is not a multiple of ``step``.
    """
    prices = np.asarray(prices, dtype=float)
    indices = np.arange(0, prices.size, step)
    if indices[-1] != prices.size - 1:
        indices = np.append(indices, prices.size - 1)
    return prices[indices]


def realized_variance(
    prices: NDArray[np.float64],
    *,
    step: int = 1,
    periods_per_year: float = 252.0,
    annualise: bool = True,
) -> float:
    r"""Sum of squared returns, optionally sampled every ``step`` observations.

    Purpose
        The baseline estimator of integrated variance over one session.
    Inputs
        ``prices`` -- one day's intraday prices, in order.
        ``step`` -- sample every ``step``-th price. ``step=1`` uses every
        observation and is the *most* noise-contaminated choice, not the best.
    Outputs
        Annualised variance by default; per-session variance if
        ``annualise=False``.
    Failure modes
        Fewer than two prices returns NaN.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # One session of 390 minutes at 20% annualised, no noise.
        >>> sigma = 0.20
        >>> steps = rng.normal(0, sigma / np.sqrt(252 * 390), 390)
        >>> prices = 100 * np.exp(np.cumsum(steps))
        >>> rv = realized_variance(prices)
        >>> bool(0.10 < np.sqrt(rv) < 0.32)
        True
    """
    sampled = _subsample(prices, step) if step > 1 else np.asarray(prices, dtype=float)
    returns = _log_returns(sampled)
    if returns.size == 0:
        return float("nan")
    variance = float(np.sum(returns**2))
    return variance * periods_per_year if annualise else variance


def bipower_variation(
    prices: NDArray[np.float64],
    *,
    step: int = 1,
    periods_per_year: float = 252.0,
    annualise: bool = True,
) -> float:
    r"""Jump-robust variance: :math:`\mu_1^{-2}\sum |r_i||r_{i-1}|`.

    Why products of adjacent absolute returns
        A jump contributes to exactly one return. Squaring it (realised
        variance) counts the whole jump. Multiplying it by its *neighbour* --
        which is an ordinary diffusive return of order :math:`\sqrt{\Delta t}` --
        makes its contribution vanish as sampling gets finer. So the estimator
        measures the continuous part of the price path and ignores the jumps,
        which is exactly what a volatility forecast wants: jumps are not
        persistent, and treating a merger announcement as elevated volatility
        forecasts a week of turbulence that does not arrive.

    The constant :math:`\mu_1^{-2} = \pi/2` corrects for the fact that
    :math:`\mathbb{E}|Z| = \sqrt{2/\pi}` rather than 1 for a standard normal.
    """
    sampled = _subsample(prices, step) if step > 1 else np.asarray(prices, dtype=float)
    returns = np.abs(_log_returns(sampled))
    if returns.size < 2:
        return float("nan")
    n = returns.size
    # Finite-sample correction (Barndorff-Nielsen & Shephard): n/(n-1).
    scale = (MU_1**-2) * (n / (n - 1))
    variance = float(scale * np.sum(returns[1:] * returns[:-1]))
    return variance * periods_per_year if annualise else variance


@dataclass(frozen=True)
class JumpTest:
    """Barndorff-Nielsen and Shephard's test for jumps in one session."""

    realized_variance: float
    bipower_variation: float
    statistic: float
    p_value: float
    #: Share of the session's variance attributable to jumps, floored at zero.
    jump_share: float

    @property
    def has_jump(self) -> bool:
        return bool(self.p_value < 0.05)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.p_value):
            return "not computable"
        if self.has_jump:
            return (
                f"jump detected (p={self.p_value:.4f}); {self.jump_share:.0%} of the "
                "session's variance was discontinuous, and forecasting from realised "
                "variance would carry that into tomorrow"
            )
        return f"no evidence of jumps (p={self.p_value:.3f})"


def jump_test(prices: NDArray[np.float64], *, step: int = 1) -> JumpTest:
    r"""Test whether a session contained a jump.

    Method
        Under the null of a continuous path, realised variance and bipower
        variation estimate the same quantity, so their *ratio* is one. Under a
        jump, realised variance is inflated and the ratio exceeds one. The
        studentised ratio statistic

        .. math:: z = \frac{1 - BV/RV}{\sqrt{\left(\frac{\pi^2}{4}
                  + \pi - 5\right)\frac1n \max\left(1, \frac{TQ}{BV^2}\right)}}

        is asymptotically standard normal, where :math:`TQ` is realised
        tripower quarticity -- itself jump-robust, which matters because using
        realised quarticity here would let the jump inflate its own denominator
        and destroy the test's power.
    Outputs
        A :class:`JumpTest`. One-sided: only positive :math:`z` is evidence.
    """
    sampled = _subsample(prices, step) if step > 1 else np.asarray(prices, dtype=float)
    returns = _log_returns(sampled)
    n = returns.size
    if n < 5:
        return JumpTest(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    absolute = np.abs(returns)
    rv = float(np.sum(returns**2))
    bv = float((MU_1**-2) * (n / (n - 1)) * np.sum(absolute[1:] * absolute[:-1]))

    if n < 4:
        return JumpTest(rv, bv, float("nan"), float("nan"), float("nan"))

    # Tripower quarticity: jump-robust estimate of integrated quarticity.
    # Using realised quarticity instead would let a jump inflate the test's own
    # denominator, which is how a jump hides from a jump test.
    power = absolute ** (4 / 3)
    tq = float(n * (MU_43**-3) * (n / (n - 2)) * np.sum(power[2:] * power[1:-1] * power[:-2]))

    if rv <= 0 or bv <= 0 or not np.isfinite(tq) or tq <= 0:
        return JumpTest(rv, bv, float("nan"), float("nan"), float("nan"))

    theta = (np.pi**2) / 4 + np.pi - 5
    denominator = np.sqrt(theta * (1.0 / n) * max(1.0, tq / bv**2))
    statistic = float((1.0 - bv / rv) / denominator) if denominator > 0 else float("nan")
    p_value = float(1.0 - ndtr(np.array(statistic))) if np.isfinite(statistic) else float("nan")

    return JumpTest(
        realized_variance=rv,
        bipower_variation=bv,
        statistic=statistic,
        p_value=p_value,
        jump_share=float(max(0.0, (rv - bv) / rv)),
    )


def noise_variance(prices: NDArray[np.float64]) -> float:
    r"""Estimate the microstructure noise variance :math:`\mathbb{E}[u^2]`.

    Method
        Since :math:`\mathbb{E}[RV_n] = IV + 2n\,\mathbb{E}[u^2]`, the noise is
        the *excess* of the fast estimator over a sparse one that carries
        negligible noise:

        .. math:: \widehat{\mathbb{E}[u^2]}
                  = \frac{RV^{\text{fast}} - RV^{\text{sparse}}}{2n}

        floored at zero (Hansen & Lunde, 2006).

    Why not simply :math:`RV_n/(2n)`
        That shortcut assumes noise dominates completely at the finest
        frequency, which is true for tick data and false for anything cleaner --
        and it fails *silently*, because it cannot tell volatility from noise.
        Measured on a clean simulated session of 390 minutes at 22% volatility,
        it reported an implied noise standard deviation of 5e-4, larger than the
        real noise in a genuinely noisy series. Every downstream user of the
        estimate then inherited the error: :func:`optimal_sampling_interval`
        recommended discarding 90% of a clean sample, and
        :func:`volatility_report` warned about noise that was not there.

        Differencing against a sparse estimator removes the integrated variance,
        which is exactly the term the shortcut mistook for noise.
    Why the sparse leg averages every interleaved subgrid
        A single sparse grid uses one observation in ``step``, so it is itself a
        noisy estimate of :math:`IV`, and that sampling error passes straight
        into the difference. Because the result is floored at zero, one-sided
        noise becomes an upward *bias*: measured on clean simulated data, a
        single-grid version could not distinguish a true noise level of
        :math:`10^{-5}` from zero. Averaging over all ``step`` interleaved
        subgrids uses every observation and drops the detection floor by roughly
        an order of magnitude.
    """
    returns = _log_returns(prices)
    n = returns.size
    if n < 20:
        return float("nan")

    fast = float(np.sum(returns**2))

    step = max(2, n // 30)
    subgrid_sums, subgrid_counts = [], []
    for offset in range(step):
        grid = np.asarray(prices, dtype=float)[offset::step]
        if grid.size < 2:
            continue
        sub_returns = np.diff(np.log(grid))
        subgrid_sums.append(float(np.sum(sub_returns**2)))
        subgrid_counts.append(sub_returns.size)

    if not subgrid_sums:
        return float("nan")

    slow = float(np.mean(subgrid_sums))
    n_bar = float(np.mean(subgrid_counts))
    if n <= n_bar:
        return float("nan")

    return float(max(0.0, (fast - slow) / (2.0 * (n - n_bar))))


def two_scale_realized_variance(
    prices: NDArray[np.float64],
    *,
    slow_step: int | None = None,
    periods_per_year: float = 252.0,
    annualise: bool = True,
) -> float:
    r"""Noise-corrected realised variance (Zhang, Mykland & Aït-Sahalia, 2005).

    Method
        Compute realised variance on ``slow_step`` *interleaved* subgrids and
        average them -- this uses every observation, unlike sparse sampling,
        which throws away all but one grid. Then subtract the noise, estimated
        from the full-frequency estimator:

        .. math:: \widehat{IV} = \bar{RV}^{\text{slow}}
                  - \frac{\bar{n}}{n} RV^{\text{fast}}

        with a small-sample scaling of :math:`(1 - \bar n/n)^{-1}`. The
        subtraction is the whole idea: the fast estimator is almost pure noise,
        so it can be used to remove the noise from the slow one.
    Outputs
        Annualised variance. Can come out **negative** when the noise correction
        overshoots on a short or quiet sample; that is returned as NaN with the
        reason being genuine -- a negative variance estimate is not a number to
        pass downstream.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(3)
        >>> n, sigma = 23400, 0.30                     # one second bars
        >>> efficient = 100 * np.exp(np.cumsum(
        ...     rng.normal(0, sigma / np.sqrt(252 * n), n)))
        >>> noisy = efficient * np.exp(rng.normal(0, 1e-4, n))
        >>> naive = np.sqrt(realized_variance(noisy))
        >>> corrected = np.sqrt(two_scale_realized_variance(noisy))
        >>> bool(naive > 1.4 * sigma)        # the naive estimator overstates by ~50%
        True
        >>> bool(abs(corrected - sigma) < 0.03)
        True
    """
    prices = np.asarray(prices, dtype=float)
    n = prices.size - 1
    if n < 20:
        return float("nan")

    if slow_step is None:
        # K ~ n^{1/3}, not the textbook n^{2/3}.
        #
        # ZMA's n^{2/3} is the rate-optimal choice asymptotically, and it orders
        # the bias and variance terms correctly as n grows without bound. At the
        # sample sizes a real session provides, the constant in front dominates
        # that ordering, and n^{2/3} leaves each subgrid with far too few
        # returns: for a 23,400-tick day it gives K=818, so every subgrid holds
        # 29 returns and inherits a ~26% discretisation error.
        #
        # Measured RMSE of the recovered volatility, 40 paths per cell, against
        # a known true volatility:
        #
        #   n        noise    K=n^{2/3}   K=n^{1/3}
        #   23,400   1e-4        0.0276      0.0054      5.1x better
        #    4,680   1e-4        0.0331      0.0089      3.7x
        #      390   1e-4        0.0476      0.0196      2.4x
        #
        # n^{1/3} was better in every combination of sample size and noise level
        # tested, from 2e-5 to 1e-3. The noise correction is unbiased at any K --
        # verified separately -- so this choice trades no accuracy for variance.
        slow_step = int(np.clip(round(n ** (1 / 3)), 2, max(2, n // 8)))

    subgrid_variances = []
    subgrid_counts = []
    for offset in range(slow_step):
        grid = prices[offset::slow_step]
        if grid.size < 2:
            continue
        returns = np.diff(np.log(grid))
        subgrid_variances.append(float(np.sum(returns**2)))
        subgrid_counts.append(returns.size)

    if not subgrid_variances:
        return float("nan")

    slow_average = float(np.mean(subgrid_variances))
    n_bar = float(np.mean(subgrid_counts))
    fast = float(np.sum(_log_returns(prices) ** 2))

    estimate = (slow_average - (n_bar / n) * fast) / (1.0 - n_bar / n)
    if not np.isfinite(estimate) or estimate <= 0:
        return float("nan")
    return estimate * periods_per_year if annualise else estimate


@dataclass
class SignaturePlot:
    """Realised volatility as a function of sampling interval."""

    steps: NDArray[np.int64]
    volatilities: NDArray[np.float64]
    noise_variance: float
    #: Where the curve flattens: the interval past which noise stops dominating.
    suggested_step: int
    notes: list[str] = field(default_factory=list)

    @property
    def noise_is_material(self) -> bool:
        """Report whether the finest sampling inflates volatility by over 20%."""
        finite = self.volatilities[np.isfinite(self.volatilities)]
        if finite.size < 2:
            return False
        return bool(finite[0] > 1.2 * np.median(finite[finite.size // 2 :]))

    def summary(self) -> str:
        lines = ["sampling step -> annualised realised volatility"]
        for step, vol in zip(self.steps, self.volatilities, strict=False):
            lines.append(f"  every {step:5d} obs   {vol:7.2%}")
        lines.append(f"  suggested step: {self.suggested_step}")
        if self.noise_is_material:
            lines.append(
                "  microstructure noise is material: the finest sampling overstates "
                "volatility substantially, which is the signature-plot shape"
            )
        lines.extend(f"  {n}" for n in self.notes)
        return "\n".join(lines)


def signature_plot(
    prices: NDArray[np.float64],
    *,
    max_step: int | None = None,
    periods_per_year: float = 252.0,
) -> SignaturePlot:
    """Realised volatility against sampling frequency -- the diagnostic.

    A flat line means noise is negligible and the finest data can be used. A
    curve that rises steeply toward the left means the opposite, and its
    steepness measures the noise. Drawing this before choosing a sampling
    frequency is the difference between an argued choice and a convention.
    """
    prices = np.asarray(prices, dtype=float)
    n = prices.size
    if max_step is None:
        max_step = max(2, n // 20)

    steps = np.unique(np.round(np.geomspace(1, max(2, max_step), num=24)).astype(np.int64))
    volatilities = np.array(
        [
            np.sqrt(realized_variance(prices, step=int(s), periods_per_year=periods_per_year))
            for s in steps
        ]
    )

    # The suggested step is where the curve first comes within 5% of its
    # plateau, the plateau being the median over the coarser half.
    finite = np.isfinite(volatilities)
    suggested = int(steps[-1])
    if finite.sum() >= 4:
        plateau = float(np.median(volatilities[finite][len(volatilities[finite]) // 2 :]))
        within = np.where(finite & (np.abs(volatilities - plateau) <= 0.05 * plateau))[0]
        if within.size:
            suggested = int(steps[within[0]])

    return SignaturePlot(
        steps=steps,
        volatilities=volatilities,
        noise_variance=noise_variance(prices),
        suggested_step=suggested,
    )


def noise_is_detectable(prices: NDArray[np.float64]) -> tuple[bool, float, float]:
    r"""Say whether microstructure noise is distinguishable from zero here.

    Why this question needs asking separately
        :func:`noise_variance` returns a point estimate, and a point estimate is
        never exactly zero. Its own sampling error sets a **detection floor**:
        the sparse leg estimates :math:`IV` with a relative error of roughly
        :math:`\sqrt{2/\bar n}`, so on a clean simulated session the estimator
        still reports a noise standard deviation of order :math:`4\times10^{-5}`
        purely as sampling noise. Feeding that straight into
        :func:`optimal_sampling_interval` recommends discarding 90% of a clean
        sample to defend against noise that is not there.

        So the point estimate is not used as a gate. Instead the fast estimator
        is compared against the slow one *relative to the slow one's own
        sampling error*: noise is called detectable only when

        .. math:: RV^{\text{fast}} > \bar{RV}^{\text{slow}}
                  \left(1 + 2\sqrt{2/\bar n}\right)

    Outputs
        ``(detectable, ratio, threshold)``.
    Honest limitation
        A noise level small enough to inflate volatility by only a few percent
        will *not* be detected in a single session, because it is genuinely not
        identified there. That is a property of the data, not of this function,
        and it is the right answer: noise that cannot be distinguished from zero
        also does not need correcting for.
    """
    returns = _log_returns(prices)
    n = returns.size
    if n < 40:
        return False, float("nan"), float("nan")

    fast = float(np.sum(returns**2))
    step = max(2, n // 50)
    subgrid_sums, subgrid_counts = [], []
    for offset in range(step):
        grid = np.asarray(prices, dtype=float)[offset::step]
        if grid.size < 2:
            continue
        sub = np.diff(np.log(grid))
        subgrid_sums.append(float(np.sum(sub**2)))
        subgrid_counts.append(sub.size)

    if not subgrid_sums:
        return False, float("nan"), float("nan")

    slow = float(np.mean(subgrid_sums))
    n_bar = float(np.mean(subgrid_counts))
    if slow <= 0 or n_bar < 2:
        return False, float("nan"), float("nan")

    ratio = fast / slow
    threshold = 1.0 + 2.0 * np.sqrt(2.0 / n_bar)
    return bool(ratio > threshold), float(ratio), float(threshold)


def optimal_sampling_interval(prices: NDArray[np.float64]) -> int:
    r"""The MSE-optimal number of observations to sample (Bandi-Russell).

    Balancing the noise-induced bias against the discretisation variance gives

    .. math:: n^* = \left(\frac{IQ}{4(\mathbb{E}[u^2])^2}\right)^{1/3}

    where :math:`IQ` is integrated quarticity, estimated here from a sparse grid
    where noise is small. Returns the *step size* to use, not :math:`n^*` itself.

    The point of computing it is that the answer is not always five minutes: for
    a liquid future it is far finer, and for a small-cap stock far coarser. Using
    the same interval for both is a habit, not a method.
    """
    prices = np.asarray(prices, dtype=float)
    n = prices.size - 1
    if n < 40:
        return 1

    # No detectable noise means no reason to discard data. Skipping this check
    # made the function recommend an 11-observation step on a clean session.
    detectable, _, _ = noise_is_detectable(prices)
    if not detectable:
        return 1

    noise = noise_variance(prices)
    sparse_step = max(1, n // 40)
    sparse = _subsample(prices, sparse_step)
    sparse_returns = _log_returns(sparse)
    if sparse_returns.size < 4 or noise <= 0:
        return 1

    # Integrated quarticity from the sparse grid, scaled back to one session.
    quarticity = float(sparse_returns.size / 3.0 * np.sum(sparse_returns**4))
    if quarticity <= 0:
        return 1

    n_star = (quarticity / (4.0 * noise**2)) ** (1.0 / 3.0)
    if not np.isfinite(n_star) or n_star < 1:
        return n
    return int(np.clip(round(n / n_star), 1, max(1, n // 2)))


@dataclass
class Seasonality:
    """The intraday volatility pattern, averaged across sessions."""

    #: Bin midpoints as a fraction of the session.
    time_of_day: NDArray[np.float64]
    #: Mean absolute return in each bin, normalised to average 1.
    relative_volatility: NDArray[np.float64]
    n_sessions: int

    @property
    def is_u_shaped(self) -> bool:
        """Report whether open and close are both more active than midday.

        Volume and volatility are reliably U-shaped: the open clears overnight
        information and the close concentrates rebalancing and index trades.
        A strategy that ignores this will find its "signal" is a time-of-day
        effect.
        """
        if self.relative_volatility.size < 5:
            return False
        edges = 0.5 * (self.relative_volatility[0] + self.relative_volatility[-1])
        middle = float(
            np.mean(
                self.relative_volatility[
                    self.relative_volatility.size // 3 : 2 * self.relative_volatility.size // 3
                ]
            )
        )
        return bool(edges > 1.15 * middle)


def intraday_seasonality(sessions: list[NDArray[np.float64]], *, n_bins: int = 13) -> Seasonality:
    """Average the absolute-return profile across sessions.

    Inputs
        ``sessions`` -- one array of intraday prices per day. They need not be
        the same length; each is mapped onto a common [0, 1] time axis, which is
        what lets days with different tick counts be averaged at all.
    """
    profiles = []
    for prices in sessions:
        returns = np.abs(_log_returns(np.asarray(prices, dtype=float)))
        if returns.size < n_bins:
            continue
        position = np.linspace(0.0, 1.0, returns.size)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        binned = np.array(
            [
                np.mean(returns[(position >= lo) & (position < hi)])
                if np.any((position >= lo) & (position < hi))
                else np.nan
                for lo, hi in itertools.pairwise(edges)
            ]
        )
        if np.all(np.isfinite(binned)) and np.mean(binned) > 0:
            profiles.append(binned / np.mean(binned))

    if not profiles:
        return Seasonality(np.zeros(0), np.zeros(0), 0)

    stacked = np.vstack(profiles)
    midpoints = (np.linspace(0.0, 1.0, n_bins + 1)[:-1] + np.linspace(0.0, 1.0, n_bins + 1)[1:]) / 2
    return Seasonality(
        time_of_day=midpoints,
        relative_volatility=np.mean(stacked, axis=0),
        n_sessions=len(profiles),
    )


def epps_curve(
    prices_a: NDArray[np.float64],
    prices_b: NDArray[np.float64],
    *,
    max_step: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    r"""Give correlation as a function of sampling interval: the Epps effect.

    Measured correlation between two assets **falls toward zero** as sampling
    gets finer. It is not that the assets decouple; it is that they do not trade
    at the same instants, so a fine grid pairs a return that contains news with
    one that has not yet recorded it. Non-synchronous trading attenuates the
    measured covariance while leaving each variance intact.

    The practical consequence is severe: a correlation matrix estimated from
    one-minute data understates dependence, so a portfolio optimiser fed that
    matrix believes it has more diversification than it does, and levers up
    accordingly. The curve this function returns shows the frequency at which
    the estimate stabilises.
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if max_step is None:
        max_step = max(2, n // 20)

    steps = np.unique(np.round(np.geomspace(1, max(2, max_step), num=18)).astype(np.int64))
    correlations = []
    for step in steps:
        ra = _log_returns(_subsample(a, int(step)))
        rb = _log_returns(_subsample(b, int(step)))
        size = min(ra.size, rb.size)
        if size < 5 or np.std(ra[:size]) == 0 or np.std(rb[:size]) == 0:
            correlations.append(np.nan)
            continue
        correlations.append(float(np.corrcoef(ra[:size], rb[:size])[0, 1]))
    return steps, np.asarray(correlations, dtype=float)


@dataclass
class IntradayVolatility:
    """Every estimator for one session, side by side."""

    n_observations: int
    naive_rv: float
    sparse_rv: float
    two_scale: float
    bipower: float
    noise_variance: float
    optimal_step: int
    jump: JumpTest
    noise_detected: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def best_estimate(self) -> float:
        """The right estimator for this sample, which is not always the fanciest.

        When noise is not detectable, plain realised variance is *better* than
        the two-scale correction: the correction subtracts an adjustment that is
        not needed and pays sampling variance for it. Measured on a clean
        390-minute session at 22% volatility, always preferring two-scale
        returned 18.6% -- an error three times larger than simply summing squared
        returns, which is exact there.

        So the noise test decides. Reaching for the more sophisticated estimator
        unconditionally is a way to be wrong with better vocabulary.
        """
        if self.noise_detected and np.isfinite(self.two_scale) and self.two_scale > 0:
            return float(np.sqrt(self.two_scale))
        return float(np.sqrt(self.sparse_rv))

    def summary(self) -> str:
        def show(variance: float) -> str:
            usable = np.isfinite(variance) and variance > 0
            return f"{np.sqrt(variance):7.2%}" if usable else "     --"

        lines = [
            f"intraday volatility from {self.n_observations} observations",
            f"  every observation       {show(self.naive_rv)}   <- noise-contaminated",
            f"  every {self.optimal_step:4d} observations  {show(self.sparse_rv)}"
            "   <- sparse, MSE-optimal step",
            f"  two-scale (ZMA)         {show(self.two_scale)}   <- noise-corrected",
            f"  bipower (jump-robust)   {show(self.bipower)}",
            f"  noise sd per obs        {np.sqrt(self.noise_variance):.2e}"
            if np.isfinite(self.noise_variance) and self.noise_variance > 0
            else "  noise sd per obs        --",
            f"  {self.jump.verdict}",
        ]
        lines.extend(f"  {n}" for n in self.notes)
        return "\n".join(lines)


def volatility_report(
    prices: NDArray[np.float64], *, periods_per_year: float = 252.0
) -> IntradayVolatility:
    """Run every estimator on one session and report them together.

    Disagreement between them is diagnostic, not a problem to be averaged away:
    naive far above the others means microstructure noise; realised far above
    bipower means a jump; two-scale failing means the sample is too short or too
    quiet for the noise correction to be identified.
    """
    prices = np.asarray(prices, dtype=float)
    notes: list[str] = []
    step = optimal_sampling_interval(prices)
    detected, ratio, threshold = noise_is_detectable(prices)

    naive = realized_variance(prices, periods_per_year=periods_per_year)
    sparse = realized_variance(prices, step=step, periods_per_year=periods_per_year)
    two_scale = two_scale_realized_variance(prices, periods_per_year=periods_per_year)
    bipower = bipower_variation(prices, step=step, periods_per_year=periods_per_year)

    if detected:
        notes.append(
            f"microstructure noise dominates at full frequency: realised variance "
            f"is {ratio:.2f}x the subsampled estimate, against a {threshold:.2f}x "
            "detection threshold"
        )
    elif np.isfinite(ratio):
        notes.append(
            f"no detectable microstructure noise ({ratio:.2f}x vs a {threshold:.2f}x "
            "threshold); the uncorrected estimator is used, which is the better "
            "choice when there is nothing to correct"
        )
    if detected and not np.isfinite(two_scale):
        notes.append(
            "the two-scale correction did not produce a positive estimate; the "
            "session is likely too short or too quiet to identify the noise"
        )

    return IntradayVolatility(
        n_observations=int(prices.size),
        naive_rv=naive,
        sparse_rv=sparse,
        two_scale=two_scale,
        bipower=bipower,
        noise_variance=noise_variance(prices),
        optimal_step=step,
        jump=jump_test(prices, step=step),
        noise_detected=detected,
        notes=notes,
    )
