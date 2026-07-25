r"""Cointegration testing: Engle-Granger and Johansen.

Why correlation is the wrong tool
---------------------------------
Two random walks with independent increments have zero expected correlation, but
their *levels* will show high correlation over any given sample -- this is the
spurious regression problem (Granger & Newbold 1974). Conversely, two genuinely
cointegrated series can have low return correlation. Pairs selection by
correlation is therefore neither necessary nor sufficient, and it is the most
common methodological error in statistical arbitrage.

Cointegration is the right property: :math:`X` and :math:`Y` are cointegrated if
each is :math:`I(1)` but some linear combination :math:`Y - \beta X` is
:math:`I(0)`. That stationary combination is a tradeable spread, and its OU
parameters (:mod:`quantos.core.timeseries.ou`) tell you whether it is tradeable
*profitably*.

Two tests, and when to use which
--------------------------------
:func:`engle_granger` regresses one series on the other and tests the residual
for a unit root. Simple and interpretable, but **asymmetric**: swapping the
dependent and independent variable can flip the conclusion, because OLS
minimises errors in one direction only. Always run it both ways -- this
implementation does, and reports both.

:func:`johansen` treats the variables symmetrically and, crucially, handles more
than two series while identifying *how many* independent cointegrating
relationships exist. It is the correct tool for a basket. The cost is a
reduced-rank eigenvalue problem and reliance on tabulated critical values.

References
----------
Engle, R. F. & Granger, C. W. J. (1987), *Econometrica* 55(2), 251-276.
Johansen, S. (1991), *Econometrica* 59(6), 1551-1580.
Granger, C. W. J. & Newbold, P. (1974), *J. Econometrics* 2(2), 111-120.
MacKinnon, J. G. (2010), Queen's Economics Dept. Working Paper 1227.
Osterwald-Lenum, M. (1992), *Oxford Bull. Econ. Stat.* 54(3), 461-472.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EngleGrangerResult",
    "JohansenResult",
    "engle_granger",
    "hedge_ratio",
    "johansen",
]

# MacKinnon (2010) response-surface constants for the Engle-Granger residual
# unit-root test, with a constant, for n = 1 and n = 2 regressors.
_EG_CRIT: dict[int, dict[str, tuple[float, float, float]]] = {
    1: {
        "1%": (-3.9001, -10.534, -30.03),
        "5%": (-3.3377, -5.967, -8.98),
        "10%": (-3.0462, -4.069, -5.73),
    },
    2: {
        "1%": (-4.2981, -13.790, -46.37),
        "5%": (-3.7429, -8.352, -13.41),
        "10%": (-3.4518, -6.241, -2.79),
    },
    3: {
        "1%": (-4.6493, -18.492, -49.35),
        "5%": (-4.1000, -11.860, -11.17),
        "10%": (-3.8110, -9.036, -4.39),
    },
}


@dataclass(frozen=True)
class EngleGrangerResult:
    """Two-step Engle-Granger cointegration test, run in both directions."""

    #: ADF statistic on the residual of y ~ x.
    statistic: float
    p_value: float
    #: Cointegrating vector: spread = y - (alpha + beta * x).
    beta: float
    alpha: float
    residuals: NDArray[np.float64]
    critical_values: dict[str, float] = field(default_factory=dict)
    #: Statistic from the reversed regression x ~ y.
    reversed_statistic: float = float("nan")
    n_obs: int = 0

    @property
    def is_cointegrated(self) -> bool:
        """Whether the *less* favourable of the two directions still rejects at 5%.

        Taking the weaker direction is deliberately conservative. Reporting the
        better of two tries is a one-sample data-snooping bias, and with
        thousands of candidate pairs it manufactures cointegration wholesale.
        """
        threshold = self.critical_values.get("5%", -3.34)
        weaker = max(self.statistic, self.reversed_statistic)
        return bool(weaker < threshold)

    @property
    def direction_agreement(self) -> bool:
        """Whether both regression directions reject at the 5% level."""
        threshold = self.critical_values.get("5%", -3.34)
        return bool(self.statistic < threshold and self.reversed_statistic < threshold)


@dataclass(frozen=True)
class JohansenResult:
    """Johansen cointegration test results."""

    #: Trace statistics, one per null hypothesis r <= 0, 1, ..., k-1.
    trace_statistics: NDArray[np.float64]
    #: Maximum-eigenvalue statistics for the same sequence of nulls.
    max_eigen_statistics: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    #: Columns are cointegrating vectors, ordered by eigenvalue descending.
    eigenvectors: NDArray[np.float64]
    #: 90/95/99% critical values for the trace test, shape (k, 3).
    trace_critical_values: NDArray[np.float64]
    #: 90/95/99% critical values for the max-eigenvalue test, shape (k, 3).
    max_eigen_critical_values: NDArray[np.float64]
    n_obs: int
    n_series: int

    def rank(self, level: str = "95%") -> int:
        r"""Estimated cointegration rank -- the number of independent spreads.

        Sequential procedure: test :math:`r=0`, then :math:`r\\le1`, and so on,
        stopping at the first null *not* rejected. That stopping rule is the
        procedure's definition; scanning for the largest rejection instead
        inflates the rank.
        """
        column = {"90%": 0, "95%": 1, "99%": 2}[level]
        for r in range(self.n_series):
            if self.trace_statistics[r] <= self.trace_critical_values[r, column]:
                return r
        return self.n_series

    def cointegrating_vector(self, index: int = 0) -> NDArray[np.float64]:
        """The ``index``-th cointegrating vector, normalised on its first element.

        Normalisation matters for interpretation: it makes the vector read as
        "one unit of asset 0 against these amounts of the others", which is the
        form a trader can act on.
        """
        vector = self.eigenvectors[:, index]
        return vector / vector[0] if vector[0] != 0 else vector

    def spread(self, data: ArrayLike, index: int = 0) -> NDArray[np.float64]:
        """Project the observed series onto a cointegrating vector."""
        return np.asarray(data, dtype=float) @ self.cointegrating_vector(index)


def hedge_ratio(y: ArrayLike, x: ArrayLike, *, method: str = "ols") -> float:
    r"""Estimate the hedge ratio :math:`\beta` in ``spread = y - beta * x``.

    ``method``:

    ``"ols"``
        Regress ``y`` on ``x``. Minimises variance in ``y`` only, so it is
        biased toward zero when ``x`` is measured with noise (errors-in-variables
        attenuation) -- and prices *are* measured with noise, namely bid-ask
        bounce.
    ``"tls"``
        Total least squares via the smallest principal component: minimises
        *perpendicular* distance, treating both series symmetrically. This is
        the statistically correct choice when neither series is privileged, and
        it removes the direction-dependence that makes Engle-Granger asymmetric.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = np.cumsum(rng.standard_normal(2000))
        >>> y = 2.0 * x + rng.standard_normal(2000) * 0.5
        >>> bool(abs(hedge_ratio(y, x) - 2.0) < 0.05)
        True
    """
    a = np.asarray(y, dtype=float).ravel()
    b = np.asarray(x, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("series must be the same length")

    if method == "ols":
        from quantos.core.timeseries.ols import ols

        design = np.column_stack([np.ones(b.size), b])
        return float(ols(a, design).coefficients[1])
    if method == "tls":
        stacked = np.column_stack([b - b.mean(), a - a.mean()])
        _, _, vt = np.linalg.svd(stacked, full_matrices=False)
        # The direction of *least* variance is the residual direction; the
        # hedge ratio is the slope of the orthogonal (principal) direction.
        normal = vt[-1]
        if normal[1] == 0:
            raise ValueError("degenerate total-least-squares solution")
        return float(-normal[0] / normal[1])
    raise ValueError(f"unknown method {method!r}")


def engle_granger(
    y: ArrayLike, x: ArrayLike, *, lags: int | None = None, method: str = "ols"
) -> EngleGrangerResult:
    r"""Two-step Engle-Granger cointegration test.

    Purpose
        Decide whether two :math:`I(1)` series share a stationary linear
        combination, and return that combination.
    Method
        1. Regress ``y`` on ``[1, x]`` to get :math:`(\alpha, \beta)`.
        2. Test the residual :math:`y - \alpha - \beta x` for a unit root.
        3. Repeat with the roles swapped, and report both.
    Critical values
        **Not** the ordinary ADF values. Because :math:`\beta` was *estimated*
        from the same data, the residual is mechanically more stationary than a
        genuine error term, so the null distribution shifts left. This
        implementation uses MacKinnon (2010) surfaces for the estimated-regressor
        case. Using plain ADF critical values here -- a very common error --
        produces spurious cointegration at several times the nominal rate.
    Outputs
        :class:`EngleGrangerResult`. Prefer
        :attr:`~EngleGrangerResult.is_cointegrated`, which requires the weaker
        of the two directions to reject.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> common = np.cumsum(rng.standard_normal(1200))
        >>> a = common + rng.standard_normal(1200) * 0.4
        >>> b = 1.5 * common + rng.standard_normal(1200) * 0.4
        >>> engle_granger(a, b).is_cointegrated
        True
        >>> u, v = (np.cumsum(rng.standard_normal(1200)) for _ in range(2))
        >>> engle_granger(u, v).is_cointegrated          # independent walks
        False
    """
    from quantos.core.stats.hypothesis import augmented_dickey_fuller
    from quantos.core.timeseries.ols import ols

    a = np.asarray(y, dtype=float).ravel()
    b = np.asarray(x, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("series must be the same length")
    n = a.size
    if n < 30:
        raise ValueError("need at least 30 observations")

    beta = hedge_ratio(a, b, method=method)
    design = np.column_stack([np.ones(n), b])
    fit = ols(a, design)
    alpha = float(fit.coefficients[0]) if method == "ols" else float(np.mean(a - beta * b))
    residuals = a - alpha - beta * b

    # The residual has a zero mean by construction, so the ADF regression must
    # not include a constant -- including one throws away a degree of freedom
    # and shifts the null distribution again.
    forward = augmented_dickey_fuller(residuals, lags=lags, trend="nc")

    beta_rev = hedge_ratio(b, a, method=method)
    alpha_rev = float(np.mean(b - beta_rev * a))
    reverse = augmented_dickey_fuller(b - alpha_rev - beta_rev * a, lags=lags, trend="nc")

    crit = {level: c0 + c1 / n + c2 / (n * n) for level, (c0, c1, c2) in _EG_CRIT[1].items()}
    return EngleGrangerResult(
        statistic=forward.statistic,
        p_value=forward.p_value,
        beta=beta,
        alpha=alpha,
        residuals=residuals,
        critical_values=crit,
        reversed_statistic=reverse.statistic,
        n_obs=n,
    )


# Critical values for the trace and max-eigenvalue statistics, rows k - r = 1..8,
# columns the 90/95/99% quantiles.
#
# These are SIMULATED for the exact VECM specification that
# _johansen_statistics implements (unrestricted constant, `lags` lagged
# differences), by scripts/tabulate_johansen.py. They are not copied from a
# published table.
#
# That decision was forced by evidence: an Osterwald-Lenum table applied to this
# estimator gave a 28% rejection rate against a nominal 5%, because the table
# assumed a different treatment of the constant. Simulating against our own
# estimator makes the size correct by construction, and
# tests/core/test_cointegration.py asserts the empirical size stays near 5%.
#
# Independent cross-check: the simulated k-r=1 trace 95% value of 8.39 sits close
# to the 8.18 published for the *unrestricted constant* case, and k-r=2 (18.17 vs
# 17.95) and k-r=3 (31.67 vs 31.52) likewise. That agreement identifies our
# specification and confirms the original 4.13 was from a different one. The
# residual gap is finite-sample: these are T=1000 quantiles, not asymptotic ones,
# which is the correct thing to compare a T=1000 statistic against.
#
# Regenerate with:  python scripts/tabulate_johansen.py
#   (6000 replications, T=1000, lags=1, seed=8675309)
_JOHANSEN_TRACE_CRIT = np.array(
    [
        [6.5733, 8.3934, 11.9166],
        [15.7720, 18.1665, 22.6617],
        [28.8803, 31.6740, 37.8713],
        [46.4818, 50.1239, 56.8267],
        [68.1448, 72.1147, 79.4424],
        [93.3680, 97.7289, 107.0699],
    ]
)
_JOHANSEN_MAXEIG_CRIT = np.array(
    [
        [6.5733, 8.3934, 11.9166],
        [12.8881, 14.8270, 19.5174],
        [19.2120, 21.5147, 26.0080],
        [25.5427, 28.1203, 33.0672],
        [31.6397, 34.3045, 39.8525],
        [37.9939, 40.9163, 46.9643],
    ]
)


def johansen(data: ArrayLike, *, lags: int = 1) -> JohansenResult:
    r"""Johansen cointegration test via reduced-rank regression.

    Purpose
        Symmetrically test :math:`k` series for cointegration and estimate the
        *number* of independent cointegrating relationships, plus the vectors
        themselves. The correct tool for baskets, where Engle-Granger's
        pairwise, direction-dependent approach cannot be applied.
    Method
        Estimate the VECM

        .. math:: \Delta Y_t = \Pi Y_{t-1}
                  + \sum_{i=1}^{p}\Gamma_i \Delta Y_{t-i} + \mu + \varepsilon_t

        by concentrating out the short-run dynamics: regress both
        :math:`\Delta Y_t` and :math:`Y_{t-1}` on the lagged differences, keep
        the residuals :math:`R_0` and :math:`R_1`, form the product-moment
        matrices, and solve the generalised eigenvalue problem

        .. math:: |\lambda S_{11} - S_{10}S_{00}^{-1}S_{01}| = 0 .

        The eigenvalues give the trace statistic
        :math:`-T\sum_{i=r+1}^{k}\ln(1-\lambda_i)`.

        The generalised problem is solved by symmetric whitening with a Cholesky
        factor of :math:`S_{11}` rather than by ``scipy.linalg.eig``: it keeps
        the eigenvalues real and the computation inside NumPy.
    Inputs
        ``data`` -- ``(T, k)`` array of levels (not returns). ``lags`` -- number
        of lagged differences in the VECM.
    Outputs
        :class:`JohansenResult`; call :meth:`~JohansenResult.rank`.
    Failure modes
        Raises for ``k > 8`` (no tabulated critical values here) or a singular
        moment matrix, which indicates a linearly dependent input series.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(3)
        >>> f = np.cumsum(rng.standard_normal(1500))
        >>> y = np.column_stack([f + rng.standard_normal(1500) * 0.3,
        ...                      2 * f + rng.standard_normal(1500) * 0.3,
        ...                      np.cumsum(rng.standard_normal(1500))])
        >>> johansen(y).rank()          # exactly one cointegrating relation
        1
    """
    y = np.asarray(data, dtype=float)
    k = y.shape[1] if y.ndim == 2 else 0
    if k > _JOHANSEN_TRACE_CRIT.shape[0]:
        raise ValueError(
            f"tabulated critical values cover at most "
            f"{_JOHANSEN_TRACE_CRIT.shape[0]} series, got {k}"
        )
    stats = _johansen_statistics(y, lags=lags)
    # Critical values are indexed by k - r, so reverse to align with r = 0..k-1.
    return JohansenResult(
        trace_statistics=stats.trace_statistics,
        max_eigen_statistics=stats.max_eigen_statistics,
        eigenvalues=stats.eigenvalues,
        eigenvectors=stats.eigenvectors,
        trace_critical_values=_JOHANSEN_TRACE_CRIT[:k][::-1],
        max_eigen_critical_values=_JOHANSEN_MAXEIG_CRIT[:k][::-1],
        n_obs=stats.n_obs,
        n_series=k,
    )


def _johansen_statistics(data: ArrayLike, *, lags: int = 1) -> JohansenResult:
    """Johansen statistics without critical values.

    Split out from :func:`johansen` so that ``scripts/tabulate_johansen.py`` can
    simulate the null distribution of the statistics *this* estimator produces,
    rather than relying on a published table whose deterministic specification
    may not match ours. The returned object has empty critical-value arrays.
    """
    from quantos.core.linalg import safe_cholesky
    from quantos.core.timeseries.ols import ols

    y = np.asarray(data, dtype=float)
    if y.ndim != 2:
        raise ValueError("data must be a 2-D (T, k) array")
    t_total, k = y.shape
    if k < 1:
        raise ValueError("need at least 1 series")
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if t_total < 10 * k + lags:
        raise ValueError(f"need more observations for k={k}, lags={lags}")

    dy = np.diff(y, axis=0)
    # Align: dependent dY_t, level Y_{t-1}, and `lags` lagged differences.
    z0 = dy[lags:]
    z1 = y[lags:-1]
    lagged = [dy[lags - i : -i] for i in range(1, lags + 1)]
    exog = np.column_stack([np.ones(z0.shape[0]), *lagged])

    # Concentrate out the short-run dynamics from both blocks.
    r0 = np.column_stack([ols(z0[:, j], exog).residuals for j in range(k)])
    r1 = np.column_stack([ols(z1[:, j], exog).residuals for j in range(k)])

    t = r0.shape[0]
    s00 = r0.T @ r0 / t
    s11 = r1.T @ r1 / t
    s01 = r0.T @ r1 / t
    s10 = s01.T

    # Symmetric reduction of the generalised eigenproblem:
    #   L^-1 S10 S00^-1 S01 L^-T  where S11 = L L^T.
    factor = safe_cholesky(s11).factor
    inv_factor = np.linalg.inv(factor)
    middle = s10 @ np.linalg.solve(s00, s01)
    reduced = inv_factor @ middle @ inv_factor.T
    values, vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))

    order = np.argsort(values)[::-1]
    eigenvalues = np.clip(values[order], 0.0, 1.0 - 1e-15)
    eigenvectors = inv_factor.T @ vectors[:, order]

    trace = np.array([-t * float(np.sum(np.log1p(-eigenvalues[r:]))) for r in range(k)])
    max_eigen = np.array([-t * float(np.log1p(-eigenvalues[r])) for r in range(k)])

    return JohansenResult(
        trace_statistics=trace,
        max_eigen_statistics=max_eigen,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        trace_critical_values=np.empty((0, 3)),
        max_eigen_critical_values=np.empty((0, 3)),
        n_obs=t,
        n_series=k,
    )
