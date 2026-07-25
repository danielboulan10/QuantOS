"""Hypothesis tests, with an emphasis on the ones time series actually need.

Every test returns a :class:`TestResult` carrying the statistic, its p-value,
the null being tested, and -- where the asymptotic distribution is a poor
approximation -- the critical values that should be used instead. Returning a
bare p-value invites the reader to forget that ADF and KPSS p-values come from
*interpolated response surfaces*, not from a closed form.

Contents
--------
====================================  =====================================
:func:`t_test` / :func:`welch_t_test` Location, equal and unequal variance
:func:`jarque_bera`                   Normality via skew and kurtosis
:func:`ljung_box`                     Serial correlation (portmanteau)
:func:`engle_arch`                    Conditional heteroskedasticity
:func:`augmented_dickey_fuller`       Unit root (null: non-stationary)
:func:`kpss`                          Stationarity (null: stationary)
:func:`variance_ratio`                Random walk (Lo-MacKinlay)
:func:`ks_test`                       Distributional fit
:func:`durbin_watson`                 First-order residual correlation
====================================  =====================================

ADF and KPSS test *complementary* nulls, and the standard practice is to run
both: agreement is evidence, disagreement means the series is neither cleanly
stationary nor cleanly integrated -- typically fractional integration, which
is exactly what :func:`quantos.research.features.labeling.frac_diff` is for.

References
----------
Dickey, D. A. & Fuller, W. A. (1979), JASA 74, 427-431.
Kwiatkowski, D. et al. (1992), *J. Econometrics* 54, 159-178.
Ljung, G. M. & Box, G. E. P. (1978), *Biometrika* 65(2), 297-303.
Lo, A. W. & MacKinlay, A. C. (1988), *Rev. Financ. Stud.* 1(1), 41-66.
MacKinnon, J. G. (2010), "Critical values for cointegration tests",
    Queen's Economics Department Working Paper 1227.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.distributions import ChiSquare, Distribution, Normal, StudentT

__all__ = [
    "TestResult",
    "augmented_dickey_fuller",
    "durbin_watson",
    "engle_arch",
    "jarque_bera",
    "kpss",
    "ks_test",
    "ljung_box",
    "t_test",
    "variance_ratio",
    "welch_t_test",
]


@dataclass(frozen=True)
class TestResult:
    """Outcome of a hypothesis test.

    ``critical_values`` is populated for tests whose asymptotic p-values are
    unreliable; when present, **prefer it to** ``p_value``.
    """

    name: str
    statistic: float
    p_value: float
    null_hypothesis: str
    #: Significance level -> critical value, where tabulated.
    critical_values: dict[str, float] = field(default_factory=dict)
    #: Anything test-specific worth carrying (lags used, df, etc.).
    detail: dict[str, float] = field(default_factory=dict)

    def rejects_at(self, alpha: float = 0.05) -> bool:
        """Whether the null is rejected at level ``alpha``."""
        return self.p_value < alpha

    def __str__(self) -> str:  # pragma: no cover - display only
        verdict = "reject" if self.rejects_at() else "fail to reject"
        return (
            f"{self.name}: stat={self.statistic:.4f}, p={self.p_value:.4g} "
            f"-> {verdict} H0 ({self.null_hypothesis}) at 5%"
        )


def _clean(x: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def t_test(x: ArrayLike, mu0: float = 0.0) -> TestResult:
    """One-sample two-sided t-test for the mean.

    Example
        >>> import numpy as np
        >>> r = t_test(np.zeros(50) + 1.0)
        >>> r.p_value < 1e-6
        True
    """
    a = _clean(x)
    n = a.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    se = a.std(ddof=1) / np.sqrt(n)
    stat = float("inf") if se == 0 else float((a.mean() - mu0) / se)
    p = float(2.0 * (1.0 - StudentT(n - 1).cdf(abs(stat))))
    return TestResult(
        "one-sample t", stat, p, f"mean == {mu0}", detail={"df": float(n - 1), "n": float(n)}
    )


def welch_t_test(x: ArrayLike, y: ArrayLike) -> TestResult:
    """Two-sample t-test **without** assuming equal variances.

    Welch is the default here rather than Student's pooled test because equal
    variance is almost never true of two return series, and the pooled test's
    size distortion under heteroskedasticity is severe. Degrees of freedom come
    from the Welch-Satterthwaite approximation.
    """
    a, b = _clean(x), _clean(y)
    if a.size < 2 or b.size < 2:
        raise ValueError("each sample needs at least 2 observations")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    denom = va + vb
    if denom == 0:
        raise ValueError("both samples are constant; the test is undefined")
    stat = float((a.mean() - b.mean()) / np.sqrt(denom))
    df = float(denom**2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1)))
    p = float(2.0 * (1.0 - StudentT(df).cdf(abs(stat))))
    return TestResult("Welch t", stat, p, "equal means", detail={"df": df})


def jarque_bera(x: ArrayLike) -> TestResult:
    r"""Jarque-Bera normality test.

    .. math:: JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \sim \chi^2_2

    A caution the test's ubiquity obscures: on financial returns JB rejects
    essentially always, at every sample size, because returns are genuinely
    non-normal. A rejection is therefore not news. The informative output is
    the *decomposition* -- how much of the statistic comes from skew versus
    kurtosis -- which is returned in ``detail``.
    """
    from quantos.core.stats.descriptive import kurtosis, skewness

    a = _clean(x)
    n = a.size
    if n < 4:
        raise ValueError("need at least 4 observations")
    s = skewness(a)
    k = kurtosis(a, excess=True)
    skew_part = n / 6.0 * s**2
    kurt_part = n / 24.0 * k**2
    stat = float(skew_part + kurt_part)
    p = float(ChiSquare(2).sf(stat))
    return TestResult(
        "Jarque-Bera",
        stat,
        p,
        "residuals are normally distributed",
        detail={
            "skewness": float(s),
            "excess_kurtosis": float(k),
            "skew_contribution": float(skew_part),
            "kurtosis_contribution": float(kurt_part),
        },
    )


def ljung_box(x: ArrayLike, lags: int = 10, *, fitted_params: int = 0) -> TestResult:
    r"""Ljung-Box portmanteau test for autocorrelation up to ``lags``.

    .. math::
        Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2}{n-k} \sim \chi^2_{h-p}

    ``fitted_params`` reduces the degrees of freedom when the series is a
    *residual* from an estimated model -- omitting that correction makes the
    test anti-conservative, which is the most common way it is misapplied.
    """
    from quantos.core.stats.descriptive import autocorrelation

    a = _clean(x)
    n = a.size
    if lags >= n:
        raise ValueError(f"lags {lags} must be < n {n}")
    rho = autocorrelation(a, lags)[1:]
    k = np.arange(1, lags + 1)
    stat = float(n * (n + 2) * np.sum(rho**2 / (n - k)))
    df = max(1, lags - fitted_params)
    p = float(ChiSquare(df).sf(stat))
    return TestResult(
        "Ljung-Box",
        stat,
        p,
        f"no autocorrelation up to lag {lags}",
        detail={"lags": float(lags), "df": float(df)},
    )


def engle_arch(x: ArrayLike, lags: int = 5) -> TestResult:
    r"""Engle's LM test for ARCH effects (conditional heteroskedasticity).

    Regresses :math:`r_t^2` on its own ``lags`` lags; under the null of no ARCH,
    :math:`nR^2 \sim \chi^2_{\text{lags}}`.

    On real returns this rejects overwhelmingly, and that rejection *is* the
    justification for the GARCH machinery in
    :mod:`quantos.core.timeseries.garch`. It is also the test that distinguishes
    a simulated market that merely has fat tails from one that has genuine
    volatility clustering -- the two are different stylised facts and a
    simulator can easily produce one without the other.
    """
    from quantos.core.timeseries.ols import ols

    a = _clean(x)
    e2 = (a - a.mean()) ** 2
    n = e2.size
    if lags >= n - 1:
        raise ValueError("too many lags for the sample size")

    y = e2[lags:]
    design = np.column_stack([np.ones(y.size)] + [e2[lags - i : -i] for i in range(1, lags + 1)])
    fit = ols(y, design)
    stat = float(y.size * fit.r_squared)
    p = float(ChiSquare(lags).sf(stat))
    return TestResult(
        "Engle ARCH-LM",
        stat,
        p,
        f"no ARCH effects up to lag {lags}",
        detail={"lags": float(lags), "r_squared": float(fit.r_squared)},
    )


# MacKinnon (2010) response-surface coefficients for the ADF tau distribution.
# Rows: (constant, beta1/T, beta2/T^2) per significance level.
_ADF_CRIT: dict[str, dict[str, tuple[float, float, float]]] = {
    "nc": {  # no constant
        "1%": (-2.5658, -1.960, -10.04),
        "5%": (-1.9393, -0.398, 0.0),
        "10%": (-1.6156, -0.181, 0.0),
    },
    "c": {  # constant only
        "1%": (-3.4336, -5.999, -29.25),
        "5%": (-2.8621, -2.738, -8.36),
        "10%": (-2.5671, -1.438, -4.48),
    },
    "ct": {  # constant and trend
        "1%": (-3.9638, -8.353, -47.44),
        "5%": (-3.4126, -4.039, -17.83),
        "10%": (-3.1279, -2.418, -7.58),
    },
}


def _adf_critical_values(trend: str, n: int) -> dict[str, float]:
    """MacKinnon response surface: crit = b0 + b1/T + b2/T^2."""
    table = _ADF_CRIT[trend]
    return {level: b0 + b1 / n + b2 / (n * n) for level, (b0, b1, b2) in table.items()}


def augmented_dickey_fuller(
    x: ArrayLike, *, lags: int | None = None, trend: str = "c"
) -> TestResult:
    r"""Augmented Dickey-Fuller unit-root test.

    Estimates

    .. math::
        \Delta y_t = \alpha + \beta t + \gamma y_{t-1}
                     + \sum_{i=1}^{p}\delta_i \Delta y_{t-i} + \varepsilon_t

    and tests :math:`H_0: \gamma = 0` (a unit root, hence non-stationarity).

    **The t-statistic does not have a t-distribution under the null.** It
    follows the Dickey-Fuller tau distribution, whose critical values are more
    negative. Using normal or t critical values here -- a genuinely common
    error -- causes spurious rejections and, downstream, "cointegrated" pairs
    that are nothing of the kind. This implementation returns MacKinnon (2010)
    response-surface critical values in ``critical_values``, and the reported
    ``p_value`` is interpolated from them, so it is approximate by
    construction. **Compare the statistic to the critical values.**

    ``lags`` defaults to the Schwert rule :math:`\lfloor 12(n/100)^{1/4}\rfloor`.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> rw = np.cumsum(rng.standard_normal(600))        # unit root
        >>> augmented_dickey_fuller(rw).rejects_at(0.05)
        False
        >>> wn = rng.standard_normal(600)                   # stationary
        >>> augmented_dickey_fuller(wn).rejects_at(0.05)
        True
    """
    from quantos.core.timeseries.ols import ols

    a = _clean(x)
    n = a.size
    if trend not in _ADF_CRIT:
        raise ValueError(f"trend must be one of {sorted(_ADF_CRIT)}, got {trend!r}")
    if lags is None:
        lags = int(np.floor(12.0 * (n / 100.0) ** 0.25))
        lags = min(lags, max(1, n // 4))
    if n < lags + 5:
        raise ValueError(f"series too short ({n}) for {lags} lags")

    dy = np.diff(a)
    y = dy[lags:]
    columns = [a[lags:-1]]  # the level term, whose coefficient is gamma
    for i in range(1, lags + 1):
        columns.append(dy[lags - i : -i])
    if trend in ("c", "ct"):
        columns.append(np.ones(y.size))
    if trend == "ct":
        columns.append(np.arange(y.size, dtype=float))

    design = np.column_stack(columns)
    fit = ols(y, design)
    stat = float(fit.t_statistics[0])

    crit = _adf_critical_values(trend, n)
    p = _interpolate_p_from_criticals(stat, crit)
    return TestResult(
        "Augmented Dickey-Fuller",
        stat,
        p,
        "series has a unit root (is non-stationary)",
        critical_values=crit,
        detail={"lags": float(lags), "n": float(n), "gamma": float(fit.coefficients[0])},
    )


def _interpolate_p_from_criticals(stat: float, crit: dict[str, float]) -> float:
    """Approximate a p-value by log-linear interpolation across tabulated levels.

    Deliberately crude, and deliberately visible: no closed-form p-value exists
    for these statistics. Callers are directed to the critical values.
    """
    levels = sorted(
        ((float(k.rstrip("%")) / 100.0, v) for k, v in crit.items()), key=lambda t: t[1]
    )
    values = [v for _, v in levels]
    alphas = [a for a, _ in levels]
    if stat <= values[0]:
        return float(alphas[0] * 0.5)
    if stat >= values[-1]:
        return float(min(0.99, alphas[-1] + 0.4))
    return float(np.interp(stat, values, alphas))


def kpss(x: ArrayLike, *, trend: str = "c", lags: int | None = None) -> TestResult:
    r"""KPSS stationarity test -- the null is *stationarity*, the reverse of ADF.

    The statistic is the scaled sum of squared partial sums of the residuals,
    normalised by a Newey-West long-run variance estimate:

    .. math:: \eta = \frac{1}{n^2 \hat\sigma^2_{LR}} \sum_{t=1}^{n} S_t^2 ,
              \qquad S_t = \sum_{i=1}^{t} \hat e_i

    Critical values are the asymptotic ones from Kwiatkowski et al. (1992)
    Table 1; the returned p-value is interpolated between them.
    """
    a = _clean(x)
    n = a.size
    if trend not in ("c", "ct"):
        raise ValueError("trend must be 'c' or 'ct'")
    if lags is None:
        # Newey-West's bandwidth, 4*(n/100)^(1/4), NOT Schwert's 12*(n/100)^(1/4).
        #
        # The choice materially affects power, because the bandwidth enters the
        # long-run variance in the denominator of the statistic. Measured over 60
        # replications at n = 800 (see tests/core/test_statistics.py):
        #
        #   bandwidth   size (nominal 5%)   power vs a unit root
        #   7  (NW)          0.07                  0.98
        #   21 (Schwert)     0.05                  0.92
        #
        # Schwert's larger bandwidth over-estimates the long-run variance of an
        # integrated series, deflating the statistic toward its acceptance region.
        # The Newey-West rule dominates on power at effectively identical size.
        lags = int(np.ceil(4.0 * (n / 100.0) ** 0.25))
        lags = min(lags, n - 1)

    if trend == "c":
        resid = a - a.mean()
        crit = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    else:
        from quantos.core.timeseries.ols import ols

        design = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        resid = ols(a, design).residuals
        crit = {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}

    partial = np.cumsum(resid)
    # Newey-West long-run variance with Bartlett weights.
    s2 = float(np.dot(resid, resid) / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        s2 += 2.0 * weight * float(np.dot(resid[lag:], resid[:-lag])) / n
    if s2 <= 0:
        raise ValueError("non-positive long-run variance estimate")

    stat = float(np.sum(partial**2) / (n * n * s2))
    ordered = sorted(
        ((float(k.rstrip("%")) / 100.0, v) for k, v in crit.items()), key=lambda t: t[1]
    )
    p = float(np.interp(stat, [v for _, v in ordered], [a_ for a_, _ in ordered]))
    p = float(np.clip(p, 0.01, 0.10))
    return TestResult(
        "KPSS",
        stat,
        p,
        "series is stationary",
        critical_values=crit,
        detail={"lags": float(lags), "long_run_variance": s2},
    )


def variance_ratio(x: ArrayLike, q: int = 2, *, heteroskedastic: bool = True) -> TestResult:
    r"""Lo-MacKinlay variance ratio test of the random walk hypothesis.

    Under a random walk, variance scales linearly with the horizon, so

    .. math:: VR(q) = \frac{\operatorname{Var}(r_t^{(q)})/q}
                           {\operatorname{Var}(r_t)} = 1 .

    ``VR > 1`` indicates positive serial correlation (trending/momentum);
    ``VR < 1`` indicates mean reversion. This is more informative than
    Ljung-Box for trading purposes because its *direction* maps directly onto a
    strategy family, and its magnitude onto that strategy's edge.

    ``heteroskedastic=True`` uses the robust standard error, which is the one
    to use on returns -- the homoskedastic version rejects the random walk
    almost automatically in the presence of volatility clustering, confusing
    conditional heteroskedasticity for predictability.
    """
    a = _clean(x)
    n = a.size
    if q < 2:
        raise ValueError("q must be >= 2")
    if n < 2 * q:
        raise ValueError(f"need at least {2 * q} observations for q={q}")

    mu = a.mean()
    var_1 = float(np.sum((a - mu) ** 2) / (n - 1))
    aggregated = np.convolve(a, np.ones(q), mode="valid")
    m = q * (n - q + 1) * (1.0 - q / n)
    var_q = float(np.sum((aggregated - q * mu) ** 2) / m)
    if var_1 == 0:
        raise ValueError("zero variance series")
    vr = var_q / var_1

    if not heteroskedastic:
        variance = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n)
    else:
        # Lo-MacKinlay (1988) eq. 14: heteroskedasticity-consistent variance
        #
        #   delta_j = sum_t e_t^2 e_{t-j}^2 / (sum_t e_t^2)^2 ,
        #   theta   = sum_{j=1}^{q-1} [2(q-j)/q]^2 delta_j ,
        #
        # and (VR - 1)/sqrt(theta) is asymptotically standard normal. Note
        # delta_j is already O(1/n) -- its numerator carries one factor of n
        # and its denominator two -- so theta must NOT be rescaled by n again.
        # Doing so understates the statistic by a factor of sqrt(n), which
        # turns an overwhelming rejection into an apparent non-result.
        e = a - mu
        e2 = e * e
        denom = float(np.sum(e2)) ** 2
        variance = 0.0
        for j in range(1, q):
            delta = float(np.sum(e2[j:] * e2[:-j])) / denom
            variance += (2.0 * (q - j) / q) ** 2 * delta
    stat = float((vr - 1.0) / np.sqrt(variance)) if variance > 0 else float("nan")
    p = float(2.0 * Normal().sf(abs(stat)))
    return TestResult(
        "Lo-MacKinlay variance ratio",
        stat,
        p,
        f"series follows a random walk (VR({q}) == 1)",
        detail={
            "variance_ratio": vr,
            "q": float(q),
            "interpretation_sign": 1.0 if vr > 1 else -1.0,
        },
    )


def ks_test(x: ArrayLike, distribution: Distribution) -> TestResult:
    r"""One-sample Kolmogorov-Smirnov goodness-of-fit test.

    ``distribution`` is any :class:`~quantos.core.distributions.Distribution`.
    The asymptotic p-value uses the Kolmogorov series
    :math:`Q(\lambda)=2\sum_{k\ge1}(-1)^{k-1}e^{-2k^2\lambda^2}`.

    Caveat: the null distribution assumes the parameters were **not** estimated
    from the same sample. If they were, this p-value is badly conservative and
    a Lilliefors correction (or a bootstrap) is required.
    """
    a = np.sort(_clean(x))
    n = a.size
    if n < 5:
        raise ValueError("need at least 5 observations")
    cdf = np.asarray(distribution.cdf(a), dtype=float)
    upper = np.arange(1, n + 1) / n - cdf
    lower = cdf - np.arange(0, n) / n
    stat = float(max(upper.max(), lower.max()))

    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * stat
    k = np.arange(1, 101)
    p = float(np.clip(2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2)), 0.0, 1.0))
    return TestResult(
        "Kolmogorov-Smirnov",
        stat,
        p,
        f"sample is drawn from {type(distribution).__name__}",
        detail={"n": float(n)},
    )


def durbin_watson(residuals: ArrayLike) -> TestResult:
    r"""Durbin-Watson statistic for first-order residual autocorrelation.

    .. math:: d = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
              \approx 2(1 - \hat\rho_1)

    so ``d`` near 2 means no correlation, near 0 strong positive, near 4 strong
    negative. The exact null distribution depends on the regressor matrix and
    is not computed here; the reported p-value uses the :math:`\hat\rho_1`
    normal approximation and should be read as indicative.
    """
    e = _clean(residuals)
    if e.size < 3:
        raise ValueError("need at least 3 residuals")
    denom = float(np.dot(e, e))
    if denom == 0:
        raise ValueError("residuals are identically zero")
    stat = float(np.sum(np.diff(e) ** 2) / denom)
    rho = 1.0 - stat / 2.0
    z = rho * np.sqrt(e.size)
    p = float(2.0 * Normal().sf(abs(z)))
    return TestResult(
        "Durbin-Watson",
        stat,
        p,
        "no first-order residual autocorrelation",
        detail={"implied_rho1": float(rho)},
    )
