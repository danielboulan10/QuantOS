r"""Performance and risk metrics.

Opinions embedded in this module
--------------------------------
**Drawdown is computed on the equity curve, not on returns.** Reconstructing an
equity curve by compounding returns and then measuring drawdown gives a
different answer from measuring it on the actual curve whenever there are
cashflows. Pass the curve when you have it.

**VaR is reported as a positive loss.** A "VaR of -0.03" invites sign errors in
every downstream aggregation. Here a 3% loss is ``0.03``.

**CVaR is preferred to VaR, and the module says so.** VaR is not subadditive:
the VaR of a portfolio can exceed the sum of its parts' VaRs, which makes it
unusable for risk *allocation* and is why Basel moved to expected shortfall.
CVaR (= expected shortfall) is coherent.

**Annualisation is explicit.** Multiplying a Sharpe ratio by :math:`\sqrt{252}`
assumes i.i.d. returns. With autocorrelation :math:`\rho`, the correct scaling
factor is smaller, and :func:`annualisation_factor` computes it -- momentum
strategies with positive autocorrelation have their annualised Sharpe
*overstated* by the naive formula.

References
----------
Artzner, P. et al. (1999), "Coherent measures of risk", *Mathematical Finance*
    9(3), 203-228.
Lo, A. W. (2002), "The statistics of Sharpe ratios", *Financial Analysts
    Journal* 58(4), 36-52.
Cornish, E. A. & Fisher, R. A. (1938), *Revue de l'Institut International de
    Statistique* 5, 307-320.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.special import ndtri

__all__ = [
    "PerformanceReport",
    "annualisation_factor",
    "calmar_ratio",
    "conditional_value_at_risk",
    "cornish_fisher_var",
    "drawdown_series",
    "max_drawdown",
    "omega_ratio",
    "performance_report",
    "sharpe_ratio",
    "sortino_ratio",
    "tail_ratio",
    "ulcer_index",
    "value_at_risk",
]


def _clean(x: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def annualisation_factor(returns: ArrayLike, periods_per_year: float = 252.0) -> float:
    r"""Autocorrelation-adjusted annualisation factor for the Sharpe ratio.

    The naive factor is :math:`\sqrt{q}` with :math:`q` periods per year, which is
    correct only for i.i.d. returns. With first-order autocorrelation
    :math:`\rho`, Lo (2002) gives

    .. math:: \eta(q) = \frac{q}{\sqrt{q + 2\sum_{k=1}^{q-1}(q-k)\rho_k}}

    Direction of the error matters: positive autocorrelation (momentum, illiquid
    assets, anything with smoothed marks) makes the naive factor **overstate** the
    annualised Sharpe ratio, sometimes by 30% or more. Negative autocorrelation
    understates it.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> iid = rng.standard_normal(5000)
        >>> bool(abs(annualisation_factor(iid) - np.sqrt(252)) < 1.0)
        True
    """
    from quantos.core.stats.descriptive import autocorrelation

    r = _clean(returns)
    q = int(periods_per_year)
    max_lag = min(q - 1, r.size // 4)
    if r.size < 20 or max_lag < 1:
        return float(np.sqrt(periods_per_year))
    rho = autocorrelation(r, max_lag)[1:]
    k = np.arange(1, max_lag + 1)
    denominator = periods_per_year + 2.0 * float(np.sum((periods_per_year - k) * rho))
    if denominator <= 0:
        return float(np.sqrt(periods_per_year))
    return float(periods_per_year / np.sqrt(denominator))


def sharpe_ratio(
    returns: ArrayLike,
    *,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
    adjust_autocorrelation: bool = False,
) -> float:
    r"""Annualised Sharpe ratio.

    Set ``adjust_autocorrelation=True`` to use :func:`annualisation_factor`
    instead of :math:`\sqrt{q}`. For any strategy holding positions for more than
    one period, that is the more defensible number.
    """
    r = _clean(returns) - risk_free / periods_per_year
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd == 0:
        return float("nan")
    factor = (
        annualisation_factor(r, periods_per_year)
        if adjust_autocorrelation
        else float(np.sqrt(periods_per_year))
    )
    return float(np.mean(r) / sd * factor)


def sortino_ratio(
    returns: ArrayLike, *, target: float = 0.0, periods_per_year: float = 252.0
) -> float:
    r"""Sortino ratio: excess return over *downside* deviation.

    .. math:: \text{Sortino} = \frac{\mathbb{E}[r] - \tau}
              {\sqrt{\mathbb{E}[\min(r - \tau, 0)^2]}}

    The denominator divides by the **full** sample size, not by the number of
    downside observations. That is Sortino's original definition and it matters:
    dividing by the downside count makes a strategy with few, large losses look
    better than one with many small ones, which inverts the intended ranking.
    """
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    downside = np.minimum(r - target, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0:
        return float("inf") if np.mean(r) > target else float("nan")
    return float((np.mean(r) - target) / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: ArrayLike) -> NDArray[np.float64]:
    r"""Fractional drawdown at each point: :math:`E_t / \max_{s\le t} E_s - 1`.

    Values are negative or zero. Requires a strictly positive equity curve; a
    curve that crosses zero has no meaningful *fractional* drawdown, and this
    raises rather than returning nonsense.
    """
    e = np.asarray(equity, dtype=float).ravel()
    if e.size == 0:
        return np.zeros(0)
    if np.any(e <= 0):
        raise ValueError(
            "equity curve must be strictly positive; fractional drawdown is "
            "undefined once equity reaches zero"
        )
    peak = np.maximum.accumulate(e)
    return e / peak - 1.0


def max_drawdown(equity: ArrayLike) -> tuple[float, int, int]:
    """Maximum drawdown with its peak and trough indices.

    Returns ``(depth, peak_index, trough_index)`` where ``depth`` is negative.

    Example
        >>> depth, peak, trough = max_drawdown([100., 120., 90., 110.])
        >>> round(depth, 4), peak, trough
        (-0.25, 1, 2)
    """
    e = np.asarray(equity, dtype=float).ravel()
    if e.size < 2:
        return 0.0, 0, 0
    dd = drawdown_series(e)
    trough = int(np.argmin(dd))
    peak = int(np.argmax(e[: trough + 1]))
    return float(dd[trough]), peak, trough


def calmar_ratio(equity: ArrayLike, *, periods_per_year: float = 252.0) -> float:
    """Annualised return divided by the absolute maximum drawdown."""
    e = np.asarray(equity, dtype=float).ravel()
    if e.size < 2 or e[0] <= 0:
        return float("nan")
    years = (e.size - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    total = e[-1] / e[0]
    if total <= 0:
        return float("nan")
    annualised = total ** (1.0 / years) - 1.0
    depth = abs(max_drawdown(e)[0])
    return float(annualised / depth) if depth > 0 else float("inf")


def omega_ratio(returns: ArrayLike, *, threshold: float = 0.0) -> float:
    r"""Omega ratio: :math:`\mathbb{E}[(r-\tau)^+] / \mathbb{E}[(r-\tau)^-]`.

    Uses the *entire* return distribution rather than its first two moments,
    which is its appeal: two strategies with identical mean and variance but
    different skew get different Omega. Returns ``inf`` when there are no
    observations below the threshold.
    """
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    gains = float(np.sum(np.maximum(r - threshold, 0.0)))
    losses = float(np.sum(np.maximum(threshold - r, 0.0)))
    return float(gains / losses) if losses > 0 else float("inf")


def value_at_risk(
    returns: ArrayLike, *, confidence: float = 0.95, method: str = "historical"
) -> float:
    r"""Value at Risk, returned as a **positive** loss magnitude.

    ``method``:

    ``"historical"``
        Empirical quantile. Makes no distributional assumption, but cannot see
        beyond the worst observation in the sample -- so a 99% VaR from 250 days
        of data rests on roughly two observations.
    ``"parametric"``
        Gaussian. Understates tail risk for every real return series; included
        for comparison rather than for use.
    ``"cornish_fisher"``
        Gaussian quantile corrected for skew and kurtosis. See
        :func:`cornish_fisher_var`.

    A caution worth repeating: VaR is **not subadditive**, so portfolio VaR can
    exceed the sum of component VaRs. Use :func:`conditional_value_at_risk` for
    anything involving risk allocation.
    """
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")

    if method == "historical":
        return float(-np.quantile(r, 1.0 - confidence))
    if method == "parametric":
        z = float(ndtri(np.array(1.0 - confidence)))
        return float(-(np.mean(r) + z * np.std(r, ddof=1)))
    if method == "cornish_fisher":
        return cornish_fisher_var(r, confidence=confidence)
    raise ValueError(f"unknown method {method!r}")


def cornish_fisher_var(returns: ArrayLike, *, confidence: float = 0.95) -> float:
    r"""VaR from the Cornish-Fisher expansion of the quantile.

    .. math::
        z_{CF} = z + \frac{S}{6}(z^2-1) + \frac{K}{24}(z^3-3z)
                 - \frac{S^2}{36}(2z^3-5z)

    Purpose
        Capture skew and excess kurtosis without assuming a full parametric
        family. It is a genuine middle ground between the Gaussian assumption
        (wrong) and the historical quantile (sample-limited).
    Known limitation
        The expansion is not monotone in the confidence level for large skew or
        kurtosis, and can produce a 99% VaR *smaller* than the 95% one. That is a
        property of the truncated series, not a bug. Check monotonicity before
        relying on it at extreme confidence levels with very fat-tailed data.
    """
    from quantos.core.stats.descriptive import kurtosis, skewness

    r = _clean(returns)
    if r.size < 4:
        return float("nan")
    z = float(ndtri(np.array(1.0 - confidence)))
    s = skewness(r)
    k = kurtosis(r, excess=True)
    z_cf = (
        z
        + (s / 6.0) * (z * z - 1.0)
        + (k / 24.0) * (z**3 - 3.0 * z)
        - (s * s / 36.0) * (2.0 * z**3 - 5.0 * z)
    )
    return float(-(np.mean(r) + z_cf * np.std(r, ddof=1)))


def conditional_value_at_risk(returns: ArrayLike, *, confidence: float = 0.95) -> float:
    r"""Conditional VaR (expected shortfall), as a positive loss.

    .. math:: \text{CVaR}_\alpha = -\mathbb{E}[r \mid r \le q_{1-\alpha}]

    Purpose
        The average loss *given* that the VaR threshold was breached. Coherent in
        the Artzner et al. sense -- in particular subadditive -- so it can be
        allocated across a portfolio, which VaR cannot. It is also sensitive to
        the shape of the tail rather than to a single quantile, so two portfolios
        with equal VaR and very different tails are correctly distinguished.

    Example
        >>> import numpy as np
        >>> r = np.concatenate([np.full(95, 0.01), np.full(5, -0.10)])
        >>> round(conditional_value_at_risk(r, confidence=0.95), 4)
        0.1
    """
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    threshold = np.quantile(r, 1.0 - confidence)
    tail = r[r <= threshold]
    if tail.size == 0:
        return float(-threshold)
    return float(-np.mean(tail))


def ulcer_index(equity: ArrayLike) -> float:
    r"""Ulcer index: RMS drawdown, :math:`\sqrt{\mathbb{E}[D_t^2]}`.

    Penalises deep *and* prolonged drawdowns, where :func:`max_drawdown` sees
    only the single worst point. Two strategies with identical maximum drawdown
    but very different recovery times get very different Ulcer indices, which is
    usually the distinction an investor cares about.
    """
    e = np.asarray(equity, dtype=float).ravel()
    if e.size < 2:
        return 0.0
    return float(np.sqrt(np.mean(drawdown_series(e) ** 2)))


def tail_ratio(returns: ArrayLike, *, quantile: float = 0.05) -> float:
    """Ratio of the right-tail to left-tail magnitude at ``quantile``.

    Above 1 means the upside tail is larger than the downside. A blunt but
    assumption-free summary of asymmetry.
    """
    r = _clean(returns)
    if r.size < 20:
        return float("nan")
    right = abs(float(np.quantile(r, 1.0 - quantile)))
    left = abs(float(np.quantile(r, quantile)))
    return float(right / left) if left > 0 else float("inf")


@dataclass(frozen=True)
class PerformanceReport:
    """A full performance and risk summary."""

    n_periods: int
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    sharpe_autocorr_adjusted: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float
    max_drawdown_duration: int
    ulcer: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    skewness: float
    excess_kurtosis: float
    tail_ratio: float
    hit_rate: float
    #: Mean win divided by mean loss magnitude.
    win_loss_ratio: float

    def __str__(self) -> str:  # pragma: no cover - display
        rows = [
            ("periods", f"{self.n_periods}"),
            ("total return", f"{self.total_return:>10.2%}"),
            ("annualised return", f"{self.annualised_return:>10.2%}"),
            ("annualised vol", f"{self.annualised_volatility:>10.2%}"),
            ("Sharpe", f"{self.sharpe:>10.3f}"),
            ("Sharpe (autocorr-adj)", f"{self.sharpe_autocorr_adjusted:>10.3f}"),
            ("Sortino", f"{self.sortino:>10.3f}"),
            ("Calmar", f"{self.calmar:>10.3f}"),
            ("Omega", f"{self.omega:>10.3f}"),
            ("max drawdown", f"{self.max_drawdown:>10.2%}"),
            ("max dd duration", f"{self.max_drawdown_duration:>10d}"),
            ("Ulcer index", f"{self.ulcer:>10.4f}"),
            ("VaR 95% / CVaR 95%", f"{self.var_95:>10.2%} / {self.cvar_95:.2%}"),
            ("VaR 99% / CVaR 99%", f"{self.var_99:>10.2%} / {self.cvar_99:.2%}"),
            ("skew / excess kurt", f"{self.skewness:>10.3f} / {self.excess_kurtosis:.3f}"),
            ("tail ratio", f"{self.tail_ratio:>10.3f}"),
            ("hit rate", f"{self.hit_rate:>10.2%}"),
            ("win/loss ratio", f"{self.win_loss_ratio:>10.3f}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)


def performance_report(
    returns: ArrayLike,
    *,
    equity: ArrayLike | None = None,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
) -> PerformanceReport:
    """Compute every metric in this module in one pass.

    ``equity`` is derived by compounding ``returns`` if not supplied. Pass it
    explicitly whenever the real curve is available -- see the module docstring.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> rep = performance_report(rng.standard_normal(1000) * 0.01 + 0.0005)
        >>> bool(rep.n_periods == 1000 and rep.max_drawdown < 0)
        True
    """
    from quantos.core.stats.descriptive import kurtosis, skewness

    r = _clean(returns)
    if r.size < 3:
        raise ValueError("need at least 3 returns")
    e = np.asarray(equity, dtype=float).ravel() if equity is not None else np.cumprod(1.0 + r)

    years = r.size / periods_per_year
    total = float(e[-1] / e[0] - 1.0) if e[0] > 0 else float("nan")
    annualised = (
        float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else float("nan")
    )

    dd = drawdown_series(e)
    depth, _, _trough = max_drawdown(e)
    # Longest run of consecutive periods spent below a prior peak.
    underwater = dd < 0
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    wins = r[r > 0]
    losses = r[r < 0]

    return PerformanceReport(
        n_periods=int(r.size),
        total_return=total,
        annualised_return=annualised,
        annualised_volatility=float(np.std(r, ddof=1) * np.sqrt(periods_per_year)),
        sharpe=sharpe_ratio(r, risk_free=risk_free, periods_per_year=periods_per_year),
        sharpe_autocorr_adjusted=sharpe_ratio(
            r,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            adjust_autocorrelation=True,
        ),
        sortino=sortino_ratio(r, periods_per_year=periods_per_year),
        calmar=calmar_ratio(e, periods_per_year=periods_per_year),
        omega=omega_ratio(r),
        max_drawdown=depth,
        max_drawdown_duration=int(longest),
        ulcer=ulcer_index(e),
        var_95=value_at_risk(r, confidence=0.95),
        cvar_95=conditional_value_at_risk(r, confidence=0.95),
        var_99=value_at_risk(r, confidence=0.99),
        cvar_99=conditional_value_at_risk(r, confidence=0.99),
        skewness=skewness(r),
        excess_kurtosis=kurtosis(r, excess=True),
        tail_ratio=tail_ratio(r),
        hit_rate=float(wins.size / r.size),
        win_loss_ratio=(
            float(np.mean(wins) / abs(np.mean(losses)))
            if wins.size and losses.size
            else float("nan")
        ),
    )
