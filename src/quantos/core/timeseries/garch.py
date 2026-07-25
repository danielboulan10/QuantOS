r"""GARCH(1,1) volatility models estimated by maximum likelihood.

Why GARCH earns its place here
------------------------------
:func:`quantos.core.stats.hypothesis.engle_arch` rejects "no ARCH effects" on
essentially every financial return series ever recorded. GARCH is the minimal
model that responds to that fact, and its one-step-ahead variance forecast is
the input to every risk number that is not simply a rolling standard deviation.

Implementation notes that matter
--------------------------------
**Parameterisation.** Optimising :math:`(\\omega, \\alpha, \\beta)` directly means
fighting the constraints :math:`\\omega>0`, :math:`\\alpha,\\beta\\ge0`,
:math:`\\alpha+\\beta<1` at every step. Instead we optimise unconstrained
transformed parameters and map them into the feasible region, so the optimiser
never sees a boundary. Stationarity holds by construction rather than by luck.

**Variance targeting.** :math:`\\omega` is not a free parameter by default:
given the sample variance :math:`\\hat\\sigma^2`, stationarity forces
:math:`\\omega = \\hat\\sigma^2 (1-\\alpha-\\beta)`. This removes one dimension
from the search, and the remaining two are far better identified. It is what
makes the estimator converge reliably from a fixed starting point instead of
needing a multi-start.

**Student-t innovations.** Optional and worth using: GARCH with Gaussian
innovations captures volatility clustering but still under-predicts extreme
moves, because standardised residuals remain fat-tailed after conditioning on
volatility. The estimated ``df`` is itself informative -- typically 5-8 on daily
equity data.

References
----------
Bollerslev, T. (1986), *J. Econometrics* 31(3), 307-327.
Engle, R. F. (1982), *Econometrica* 50(4), 987-1008.
Bollerslev, T. (1987), *Rev. Econ. Stat.* 69(3), 542-547.  [t-GARCH]
Glosten, L., Jagannathan, R. & Runkle, D. (1993), *J. Finance* 48(5).  [GJR]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.special import lgamma

__all__ = ["GarchResult", "GjrGarchResult", "fit_garch", "fit_gjr_garch", "garch_filter"]

_SQRT_2PI_LOG = 0.5 * float(np.log(2.0 * np.pi))


def garch_filter(
    returns: NDArray[np.float64], omega: float, alpha: float, beta: float
) -> NDArray[np.float64]:
    r"""Run the GARCH(1,1) recursion, returning the conditional variance path.

    .. math:: \sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2

    Initialised at the unconditional variance :math:`\omega/(1-\alpha-\beta)`,
    the stationary choice. Initialising at the sample variance instead (a common
    shortcut) biases the first several dozen fitted variances and, with them,
    the likelihood.

    Complexity
        :math:`O(n)`, inherently sequential -- the recursion cannot be
        vectorised, which is the one genuinely slow part of GARCH estimation.
    """
    n = returns.size
    variance = np.empty(n)
    persistence = alpha + beta
    variance[0] = omega / (1.0 - persistence) if persistence < 1.0 else float(np.var(returns))
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
    return variance


def _gjr_filter(
    returns: NDArray[np.float64], omega: float, alpha: float, gamma: float, beta: float
) -> NDArray[np.float64]:
    r"""GJR-GARCH recursion with an asymmetry term.

    .. math::
        \sigma_t^2 = \omega + (\alpha + \gamma \mathbb{1}[\varepsilon_{t-1}<0])
                     \varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2
    """
    n = returns.size
    variance = np.empty(n)
    persistence = alpha + 0.5 * gamma + beta
    variance[0] = omega / (1.0 - persistence) if persistence < 1.0 else float(np.var(returns))
    for t in range(1, n):
        shock = returns[t - 1] ** 2
        leverage = gamma if returns[t - 1] < 0.0 else 0.0
        variance[t] = omega + (alpha + leverage) * shock + beta * variance[t - 1]
    return variance


def _student_t_loglik(z2: NDArray[np.float64], variance: NDArray[np.float64], df: float) -> float:
    """Log-likelihood of standardised residuals under a scaled Student-t."""
    const = float(
        lgamma(np.array(0.5 * (df + 1.0)))
        - lgamma(np.array(0.5 * df))
        - 0.5 * np.log(np.pi * (df - 2.0))
    )
    return float(
        np.sum(
            const
            - 0.5 * np.log(variance)
            - 0.5 * (df + 1.0) * np.log1p(z2 / (variance * (df - 2.0)))
        )
    )


def _gaussian_loglik(z2: NDArray[np.float64], variance: NDArray[np.float64]) -> float:
    return float(np.sum(-_SQRT_2PI_LOG - 0.5 * np.log(variance) - 0.5 * z2 / variance))


@dataclass(frozen=True)
class GarchResult:
    """Fitted GARCH(1,1) model."""

    omega: float
    alpha: float
    beta: float
    conditional_variance: NDArray[np.float64]
    log_likelihood: float
    n_obs: int
    distribution: str
    df: float | None = None
    converged: bool = True
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def persistence(self) -> float:
        r"""Shock persistence, :math:`\alpha + \beta`.

        Values above ~0.99 mean shocks decay so slowly the process is nearly
        integrated -- typical for daily equity data, and a warning that
        long-horizon variance forecasts are fragile.
        """
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        if self.persistence >= 1.0:
            return float("inf")
        return self.omega / (1.0 - self.persistence)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay by half."""
        if not 0.0 < self.persistence < 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(self.persistence))

    @property
    def n_params(self) -> int:
        return 3 + (1 if self.df is not None else 0)

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_params * float(np.log(self.n_obs)) - 2.0 * self.log_likelihood

    def standardised_residuals(self, returns: ArrayLike) -> NDArray[np.float64]:
        """Residuals divided by fitted conditional volatility.

        If the model is adequate these should show no remaining ARCH effects --
        run :func:`~quantos.core.stats.hypothesis.engle_arch` on their squares
        to check. That diagnostic is the whole point of fitting the model.
        """
        r = np.asarray(returns, dtype=float).ravel()
        return r / np.sqrt(self.conditional_variance)

    def forecast(self, horizon: int = 1) -> NDArray[np.float64]:
        r"""Multi-step variance forecasts.

        Iterating :math:`\mathbb{E}[\sigma_{t+h}^2] = \omega +
        (\alpha+\beta)\mathbb{E}[\sigma_{t+h-1}^2]` shows forecasts decay
        geometrically toward the unconditional variance at rate
        :attr:`persistence`.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        out = np.empty(horizon)
        current = float(self.conditional_variance[-1])
        for h in range(horizon):
            current = self.omega + self.persistence * current
            out[h] = current
        return out


def _transform(theta: NDArray[np.float64]) -> tuple[float, float]:
    r"""Map unconstrained :math:`\mathbb{R}^2` to the feasible ARCH/GARCH region.

    A logistic maps the first coordinate to total persistence
    :math:`\pi \in (0,1)`, the second splits it between :math:`\alpha` and
    :math:`\beta`. Stationarity :math:`\alpha+\beta<1` therefore holds
    identically, and the optimiser works on an unbounded domain where a plain
    Nelder-Mead is well behaved.
    """
    persistence = 1.0 / (1.0 + np.exp(-theta[0]))
    share = 1.0 / (1.0 + np.exp(-theta[1]))
    alpha = persistence * share
    beta = persistence * (1.0 - share)
    return float(alpha), float(beta)


def _inverse_transform(alpha: float, beta: float) -> NDArray[np.float64]:
    persistence = min(max(alpha + beta, 1e-6), 1.0 - 1e-6)
    share = min(max(alpha / persistence if persistence > 0 else 0.5, 1e-6), 1.0 - 1e-6)
    return np.array([np.log(persistence / (1.0 - persistence)), np.log(share / (1.0 - share))])


def fit_garch(
    returns: ArrayLike,
    *,
    distribution: str = "normal",
    variance_targeting: bool = True,
    max_iter: int = 2000,
) -> GarchResult:
    r"""Estimate GARCH(1,1) by maximum likelihood.

    Purpose
        Produce conditional volatility estimates and forecasts, plus the
        standardised residuals used to check model adequacy.
    Inputs
        ``returns`` -- ``(n,)`` array of **mean-zero** returns (the mean is
        removed internally). ``distribution`` -- ``"normal"`` or ``"t"``.
        ``variance_targeting`` -- pin :math:`\omega` to the sample variance.
    Outputs
        :class:`GarchResult`.
    Method
        Nelder-Mead on the transformed parameters (see :func:`_transform`).
        Derivative-free is a deliberate choice: the GARCH likelihood's gradient
        involves a recursive derivative of the variance path, which is easy to
        get subtly wrong, and the parameter space is only two- or
        three-dimensional so the robustness is worth more than the speed.
    Complexity
        :math:`O(n)` per likelihood evaluation, a few hundred evaluations.
    Failure modes
        Sets ``converged=False`` rather than raising if the optimiser stalls --
        the returned parameters are still the best found, and volatility
        modelling on short samples legitimately sometimes fails to identify
        :math:`\alpha` and :math:`\beta` separately.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 4000
        >>> r = np.zeros(n); v = np.zeros(n); v[0] = 1e-4
        >>> for t in range(1, n):
        ...     v[t] = 2e-6 + 0.08 * r[t-1]**2 + 0.90 * v[t-1]
        ...     r[t] = np.sqrt(v[t]) * rng.standard_normal()
        >>> fit = fit_garch(r)
        >>> bool(0.80 < fit.persistence < 1.0)
        True
    """
    from quantos.core.optimize.minimize import nelder_mead

    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 100:
        raise ValueError(f"GARCH needs at least 100 observations, got {r.size}")
    if distribution not in ("normal", "t"):
        raise ValueError("distribution must be 'normal' or 't'")
    r = r - r.mean()
    sample_var = float(np.var(r))
    if sample_var <= 0:
        raise ValueError("returns have zero variance")

    def unpack(theta: NDArray[np.float64]) -> tuple[float, float, float, float | None]:
        alpha, beta = _transform(theta[:2])
        if variance_targeting:
            omega = sample_var * max(1.0 - alpha - beta, 1e-8)
        else:
            omega = float(np.exp(theta[2]))
        df = None
        if distribution == "t":
            # df mapped into (2.05, 200) by a logistic: the lower bound keeps
            # the variance finite, and the upper bound stops the optimiser
            # wandering to 1e7 when the innovations really are Gaussian (the
            # likelihood is flat out there, since t(df) -> normal). A fitted df
            # at the 200 cap is the model telling you Gaussian is adequate --
            # compare AIC against distribution="normal" to confirm.
            df = 2.05 + 197.95 / (1.0 + float(np.exp(-theta[-1])))
        return omega, alpha, beta, df

    def negative_loglik(theta: NDArray[np.float64]) -> float:
        omega, alpha, beta, df = unpack(theta)
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
            return 1e12
        variance = garch_filter(r, omega, alpha, beta)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            return 1e12
        z2 = r * r
        ll = _student_t_loglik(z2, variance, df) if df else _gaussian_loglik(z2, variance)
        return -ll if np.isfinite(ll) else 1e12

    start = list(_inverse_transform(0.08, 0.90))
    if not variance_targeting:
        start.append(float(np.log(sample_var * 0.02)))
    if distribution == "t":
        # Start at df ~ 6, the middle of the range seen on daily equity data.
        start.append(float(np.log((6.0 - 2.05) / (200.0 - 6.0))))

    result = nelder_mead(negative_loglik, np.array(start), max_iter=max_iter)
    omega, alpha, beta, df = unpack(result.x)
    variance = garch_filter(r, omega, alpha, beta)

    return GarchResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_variance=variance,
        log_likelihood=-result.fun,
        n_obs=r.size,
        distribution=distribution,
        df=df,
        converged=result.converged,
        detail={
            "iterations": float(result.iterations),
            "sample_variance": sample_var,
            "variance_targeting": float(variance_targeting),
        },
    )


@dataclass(frozen=True)
class GjrGarchResult(GarchResult):
    """Fitted GJR-GARCH with a leverage term."""

    gamma: float = 0.0

    @property
    def persistence(self) -> float:
        r"""Shock persistence, :math:`\alpha + \gamma/2 + \beta`.

        The asymmetry term contributes half its weight because negative shocks
        occur half the time.
        """
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def leverage_ratio(self) -> float:
        r"""How much more a negative shock raises variance than a positive one.

        :math:`(\alpha+\gamma)/\alpha`. Values around 3-5 are typical for equity
        indices, quantifying the leverage effect: volatility responds far more
        to selloffs than to rallies. A symmetric GARCH forces this to 1 and so
        systematically under-forecasts volatility after a drawdown.
        """
        return float((self.alpha + self.gamma) / self.alpha) if self.alpha > 0 else float("nan")


def fit_gjr_garch(returns: ArrayLike, *, max_iter: int = 3000) -> GjrGarchResult:
    r"""Estimate GJR-GARCH(1,1,1), capturing the leverage effect.

    Parameterised so that :math:`\alpha, \gamma, \beta \ge 0` and
    :math:`\alpha + \gamma/2 + \beta < 1` hold by construction, as in
    :func:`fit_garch`.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(2)
        >>> n = 5000
        >>> r = np.zeros(n); v = np.zeros(n); v[0] = 1e-4
        >>> for t in range(1, n):
        ...     lev = 0.12 if r[t-1] < 0 else 0.0
        ...     v[t] = 2e-6 + (0.03 + lev) * r[t-1]**2 + 0.88 * v[t-1]
        ...     r[t] = np.sqrt(v[t]) * rng.standard_normal()
        >>> fit = fit_gjr_garch(r)
        >>> bool(fit.gamma > 0.0)
        True
    """
    from quantos.core.optimize.minimize import nelder_mead

    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 200:
        raise ValueError("GJR-GARCH needs at least 200 observations")
    r = r - r.mean()
    sample_var = float(np.var(r))

    def unpack(theta: NDArray[np.float64]) -> tuple[float, float, float, float]:
        persistence = 1.0 / (1.0 + np.exp(-theta[0]))
        # Two-stage stick-breaking over (alpha, gamma/2, beta).
        w1 = 1.0 / (1.0 + np.exp(-theta[1]))
        w2 = 1.0 / (1.0 + np.exp(-theta[2]))
        alpha = persistence * w1 * w2
        gamma = 2.0 * persistence * w1 * (1.0 - w2)
        beta = persistence * (1.0 - w1)
        omega = sample_var * max(1.0 - persistence, 1e-8)
        return float(omega), float(alpha), float(gamma), float(beta)

    def negative_loglik(theta: NDArray[np.float64]) -> float:
        omega, alpha, gamma, beta = unpack(theta)
        variance = _gjr_filter(r, omega, alpha, gamma, beta)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            return 1e12
        ll = _gaussian_loglik(r * r, variance)
        return -ll if np.isfinite(ll) else 1e12

    start = np.array([np.log(0.95 / 0.05), np.log(0.15 / 0.85), 0.0])
    result = nelder_mead(negative_loglik, start, max_iter=max_iter)
    omega, alpha, gamma, beta = unpack(result.x)
    variance = _gjr_filter(r, omega, alpha, gamma, beta)

    return GjrGarchResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        conditional_variance=variance,
        log_likelihood=-result.fun,
        n_obs=r.size,
        distribution="normal",
        converged=result.converged,
        detail={"iterations": float(result.iterations)},
    )
