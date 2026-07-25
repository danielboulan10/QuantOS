r"""Ordinary least squares with heteroskedasticity- and autocorrelation-robust errors.

Why not just ``np.linalg.lstsq`` and be done?
---------------------------------------------
Because the coefficient estimate is the easy part and almost never the part
that is wrong. What breaks financial regressions is the *standard error*:

* Returns are heteroskedastic, so classical OLS standard errors are biased.
* Overlapping observations (a 20-day forward return sampled daily) induce
  autocorrelation in the residuals by construction, inflating t-statistics by a
  factor of roughly :math:`\\sqrt{h}` for horizon :math:`h`. A 20-day
  overlapping predictive regression can show a t-stat of 4.5 that is really
  about 1.0. Whole literatures have been built on this error.

So this module makes the covariance estimator an explicit, named choice, and
:func:`newey_west` is one keyword away. The design matrix is factorised by QR
rather than by forming the normal equations, because :math:`X^{\\top}X` squares
the condition number, and factor portfolios are routinely near-collinear.

References
----------
White, H. (1980), *Econometrica* 48(4), 817-838.
Newey, W. K. & West, K. D. (1987), *Econometrica* 55(3), 703-708.
Hansen, L. P. & Hodrick, R. J. (1980), *J. Polit. Econ.* 88(5), 829-853.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.distributions import FisherF, StudentT

__all__ = ["OLSResult", "newey_west", "ols", "rolling_beta", "white_cov"]


@dataclass(frozen=True)
class OLSResult:
    """Fitted linear model with inference.

    ``t_statistics`` and ``p_values`` are derived from whichever covariance
    estimator was requested, so switching to Newey-West changes them
    automatically -- there is no way to report a HAC covariance alongside
    classical t-stats by accident.
    """

    coefficients: NDArray[np.float64]
    residuals: NDArray[np.float64]
    covariance: NDArray[np.float64]
    n_obs: int
    n_params: int
    r_squared: float
    cov_type: str
    fitted: NDArray[np.float64]

    @property
    def standard_errors(self) -> NDArray[np.float64]:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    @property
    def t_statistics(self) -> NDArray[np.float64]:
        se = self.standard_errors
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se > 0, self.coefficients / se, np.nan)

    @property
    def p_values(self) -> NDArray[np.float64]:
        df = max(1, self.n_obs - self.n_params)
        return np.asarray(2.0 * (1.0 - StudentT(df).cdf(np.abs(self.t_statistics))))

    @property
    def degrees_of_freedom(self) -> int:
        return self.n_obs - self.n_params

    @property
    def adjusted_r_squared(self) -> float:
        if self.degrees_of_freedom <= 0:
            return float("nan")
        return 1.0 - (1.0 - self.r_squared) * (self.n_obs - 1) / self.degrees_of_freedom

    @property
    def sigma2(self) -> float:
        """Residual variance estimate."""
        if self.degrees_of_freedom <= 0:
            return float("nan")
        return float(np.dot(self.residuals, self.residuals) / self.degrees_of_freedom)

    def f_test(self) -> tuple[float, float]:
        """Joint F-test that all slopes (excluding an intercept) are zero.

        Returns ``(statistic, p_value)``. Assumes a column of ones is present;
        with no intercept the test is not meaningful and returns ``nan``.
        """
        k = self.n_params - 1
        if k < 1 or self.degrees_of_freedom <= 0 or not np.isfinite(self.r_squared):
            return float("nan"), float("nan")
        stat = (self.r_squared / k) / ((1.0 - self.r_squared) / self.degrees_of_freedom)
        p = float(1.0 - FisherF(k, self.degrees_of_freedom).cdf(stat))
        return float(stat), p

    def summary(self) -> str:
        """A compact regression table, for the research journal."""
        lines = [
            f"OLS  n={self.n_obs}  k={self.n_params}  R2={self.r_squared:.4f}  "
            f"adjR2={self.adjusted_r_squared:.4f}  cov={self.cov_type}",
            f"{'param':>8} {'coef':>12} {'std err':>12} {'t':>9} {'p':>9}",
        ]
        for i, (c, s, t, p) in enumerate(
            zip(
                self.coefficients,
                self.standard_errors,
                self.t_statistics,
                self.p_values,
                strict=False,
            )
        ):
            lines.append(f"{'b' + str(i):>8} {c:12.6f} {s:12.6f} {t:9.3f} {p:9.4f}")
        return "\n".join(lines)


def ols(
    y: ArrayLike,
    x: ArrayLike,
    *,
    cov_type: str = "classical",
    hac_lags: int | None = None,
) -> OLSResult:
    r"""Fit :math:`y = X\beta + \varepsilon` by least squares.

    Purpose
        The regression primitive underlying cointegration, factor attribution,
        ADF, ARCH-LM and beta estimation throughout QuantOS.
    Inputs
        ``y`` -- ``(n,)`` response. ``x`` -- ``(n, k)`` design matrix,
        **intercept not added automatically** (an implicit intercept is a
        frequent source of silent misspecification in factor work, so it is
        always the caller's explicit choice).
        ``cov_type`` -- ``"classical"``, ``"white"`` (HC0) or ``"hac"``
        (Newey-West). ``hac_lags`` -- bandwidth; defaults to
        :math:`\lfloor 4(n/100)^{2/9} \rfloor` (Newey-West's rule).
    Outputs
        :class:`OLSResult`.
    Complexity
        :math:`O(nk^2)` for the QR factorisation.
    Failure modes
        Raises :class:`numpy.linalg.LinAlgError` on a rank-deficient design.
        Perfect collinearity is a modelling error, not something to paper over
        with a pseudo-inverse that silently picks an arbitrary solution.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = np.column_stack([np.ones(500), rng.standard_normal(500)])
        >>> y = X @ np.array([1.0, 2.0]) + 0.1 * rng.standard_normal(500)
        >>> np.round(ols(y, X).coefficients, 1).tolist()
        [1.0, 2.0]
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, k = x.shape
    if y.size != n:
        raise ValueError(f"y has {y.size} rows but X has {n}")
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")

    # QR rather than the normal equations: cond(X'X) = cond(X)^2.
    q, r = np.linalg.qr(x)
    if np.linalg.matrix_rank(r) < k:
        raise np.linalg.LinAlgError(
            f"design matrix is rank deficient (rank {np.linalg.matrix_rank(r)} < {k}); "
            "drop the collinear column rather than regularising silently"
        )
    beta = np.linalg.solve(r, q.T @ y)
    fitted = x @ beta
    resid = y - fitted

    xtx_inv = np.linalg.solve(r, np.linalg.solve(r.T, np.eye(k)))

    if cov_type == "classical":
        sigma2 = float(np.dot(resid, resid) / (n - k))
        cov = sigma2 * xtx_inv
    elif cov_type == "white":
        cov = white_cov(x, resid, xtx_inv)
    elif cov_type == "hac":
        if hac_lags is None:
            hac_lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        cov = newey_west(x, resid, xtx_inv, lags=hac_lags)
    else:
        raise ValueError(f"unknown cov_type {cov_type!r}")

    tss = float(np.sum((y - y.mean()) ** 2))
    rss = float(np.dot(resid, resid))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    return OLSResult(
        coefficients=beta,
        residuals=resid,
        covariance=cov,
        n_obs=n,
        n_params=k,
        r_squared=r2,
        cov_type=cov_type,
        fitted=fitted,
    )


def white_cov(
    x: NDArray[np.float64], resid: NDArray[np.float64], xtx_inv: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""White (HC0) heteroskedasticity-consistent covariance.

    .. math:: (X'X)^{-1}\Big(\sum_i e_i^2 x_i x_i'\Big)(X'X)^{-1}
    """
    meat = (x * resid[:, None]).T @ (x * resid[:, None])
    return xtx_inv @ meat @ xtx_inv


def newey_west(
    x: NDArray[np.float64],
    resid: NDArray[np.float64],
    xtx_inv: NDArray[np.float64],
    *,
    lags: int,
) -> NDArray[np.float64]:
    r"""Newey-West HAC covariance with Bartlett kernel weights.

    .. math::
        S = \Gamma_0 + \sum_{j=1}^{L} w_j (\Gamma_j + \Gamma_j'),
        \qquad w_j = 1 - \frac{j}{L+1}

    The Bartlett weights are not cosmetic: they are what guarantees :math:`S`
    is positive semi-definite. Truncating the sum without them can produce a
    "covariance" matrix with negative diagonal entries, and hence imaginary
    standard errors.

    Choosing ``lags`` for overlapping returns: use **at least** the overlap
    horizon, since residuals are correlated by construction out to exactly that
    lag. Hansen-Hodrick uses ``h-1``; Newey-West's automatic rule is often too
    small for that case.
    """
    if lags < 0:
        raise ValueError("lags must be non-negative")
    u = x * resid[:, None]
    s = u.T @ u
    n = x.shape[0]
    for j in range(1, min(lags, n - 1) + 1):
        gamma = u[j:].T @ u[:-j]
        weight = 1.0 - j / (lags + 1.0)
        s = s + weight * (gamma + gamma.T)
    return xtx_inv @ s @ xtx_inv


def rolling_beta(
    y: ArrayLike, x: ArrayLike, window: int, *, min_periods: int | None = None
) -> NDArray[np.float64]:
    r"""Rolling univariate beta of ``y`` on ``x``.

    Computed from rolling sums in :math:`O(n)` rather than by refitting the
    regression at every step (:math:`O(nw)`), using the identity
    :math:`\beta = \operatorname{Cov}(x,y)/\operatorname{Var}(x)` with
    cumulative sums.

    Numerical note: cumulative-sum differencing loses precision when the
    running totals grow much larger than the window's contribution. That is
    acceptable for returns, which are centred near zero; it would not be for
    price *levels*, so pass returns.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.size != x.size:
        raise ValueError("y and x must have the same length")
    if min_periods is None:
        min_periods = window

    def csum(a: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate([[0.0], np.cumsum(a)])

    n = y.size
    sx, sy, sxx, sxy = csum(x), csum(y), csum(x * x), csum(x * y)
    out = np.full(n, np.nan)
    for i in range(min_periods - 1, n):
        lo = max(0, i - window + 1)
        hi = i + 1
        m = hi - lo
        mx = sx[hi] - sx[lo]
        my = sy[hi] - sy[lo]
        cov = (sxy[hi] - sxy[lo]) - mx * my / m
        var = (sxx[hi] - sxx[lo]) - mx * mx / m
        out[i] = cov / var if var > 0 else np.nan
    return out
