r"""Ornstein-Uhlenbeck process: the mathematical core of statistical arbitrage.

The model
---------
.. math:: dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t

A mean-reverting diffusion with three parameters: the reversion speed
:math:`\theta`, the long-run level :math:`\mu`, and the diffusion
:math:`\sigma`. It is *the* model for a cointegration residual, and every
quantity a pairs trader needs is a closed-form function of these three numbers:

* **Half-life** :math:`\ln 2/\theta` -- how long a divergence takes to close.
  This, not the correlation, is what determines whether a pair is tradeable at
  a given holding cost.
* **Stationary standard deviation** :math:`\sigma/\sqrt{2\theta}` -- the natural
  unit for entry and exit thresholds. Setting thresholds in raw price units, or
  in units of a rolling standard deviation, both ignore that the spread's own
  equilibrium dispersion is a *derived* quantity.
* **Expected first-passage time** -- the honest estimate of holding period.

Estimation
----------
The OU process observed at discrete intervals :math:`\Delta t` is *exactly* an
AR(1):

.. math:: X_{t+\Delta t} = X_t e^{-\theta \Delta t}
          + \mu(1 - e^{-\theta \Delta t}) + \epsilon,
          \quad \operatorname{Var}(\epsilon)
          = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta\Delta t}\right)

so the exact MLE is available in closed form from an OLS regression -- no
numerical optimisation, no discretisation bias. Estimating instead via an Euler
discretisation, as is common, biases :math:`\theta` downward and the bias does
not vanish as the sample grows; it vanishes only as :math:`\Delta t \to 0`.

References
----------
Uhlenbeck, G. E. & Ornstein, L. S. (1930), *Phys. Rev.* 36, 823-841.
Tang, C. Y. & Chen, S. X. (2009), "Parameter estimation and bias correction for
    diffusion processes", *J. Econometrics* 149, 65-81.
Avellaneda, M. & Lee, J.-H. (2010), "Statistical arbitrage in the US equities
    market", *Quantitative Finance* 10(7), 761-782.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["OUParameters", "expected_time_to_mean", "fit_ou", "simulate_ou"]


@dataclass(frozen=True)
class OUParameters:
    """Estimated OU parameters with the trading quantities they imply."""

    theta: float
    mu: float
    sigma: float
    #: Observation interval used in estimation, in the same time unit as theta.
    dt: float = 1.0
    n_obs: int = 0
    #: R-squared of the underlying AR(1) regression.
    r_squared: float = float("nan")

    @property
    def half_life(self) -> float:
        r""":math:`\ln 2 / \theta`, in the same units as ``dt``.

        The single most useful number for judging tradeability: a pair with a
        200-day half-life is not a mean-reversion opportunity regardless of how
        significant its cointegration test is, because the carry and financing
        cost of holding for that long exceeds the expected convergence.
        """
        return float(np.log(2.0) / self.theta) if self.theta > 0 else float("inf")

    @property
    def stationary_std(self) -> float:
        r""":math:`\sigma/\sqrt{2\theta}`, the equilibrium dispersion.

        Entry thresholds should be quoted in multiples of *this*, not of a
        rolling standard deviation -- the rolling estimate conflates the
        stationary dispersion with the sampling noise of the mean.
        """
        return float(self.sigma / np.sqrt(2.0 * self.theta)) if self.theta > 0 else float("inf")

    @property
    def is_mean_reverting(self) -> bool:
        """Whether ``theta`` is positive, i.e. the process reverts at all."""
        return self.theta > 0.0

    def zscore(self, x: ArrayLike) -> NDArray[np.float64]:
        """Standardise observations by the *stationary* distribution."""
        s = self.stationary_std
        if not np.isfinite(s) or s <= 0:
            raise ValueError("cannot standardise a non-reverting process")
        return (np.asarray(x, dtype=float) - self.mu) / s

    def optimal_entry_threshold(self, transaction_cost: float) -> float:
        r"""Approximate optimal entry z-score for a given round-trip cost.

        Trades the expected convergence profit against the cost of round-tripping,
        using the standard approximation that the expected profit from entering
        at :math:`z` standard deviations and exiting at the mean is
        :math:`z\sigma_{\text{stat}}`, so entry is worthwhile once
        :math:`z \sigma_{\text{stat}} > c`, with a factor of two margin for the
        variance of the outcome.

        This is deliberately a rule of thumb, not the solution of the
        Ornstein-Uhlenbeck optimal-stopping problem: the exact answer requires
        solving a free-boundary ODE and depends on the utility function. Treat
        it as a starting point for a threshold sweep, not as an optimum.
        """
        if transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")
        s = self.stationary_std
        return float(max(1.0, 2.0 * transaction_cost / s)) if s > 0 else float("inf")


def fit_ou(x: ArrayLike, dt: float = 1.0) -> OUParameters:
    r"""Exact maximum-likelihood estimation of an OU process from discrete data.

    Purpose
        Recover :math:`(\theta, \mu, \sigma)` from an observed spread, giving
        its half-life and stationary dispersion.
    Method
        Regress :math:`X_{t+1}` on :math:`X_t` to get the AR(1) coefficient
        :math:`\phi = e^{-\theta\Delta t}` and intercept
        :math:`c = \mu(1-\phi)`, then invert:

        .. math::
            \theta = -\frac{\ln\phi}{\Delta t}, \quad
            \mu = \frac{c}{1-\phi}, \quad
            \sigma = \sigma_\epsilon\sqrt{\frac{2\theta}{1-\phi^2}}

        These are the *exact* transition-density MLEs, not an approximation.
    Inputs
        ``x`` -- equally spaced observations. ``dt`` -- spacing.
    Outputs
        :class:`OUParameters`. When :math:`\hat\phi \ge 1` the series shows no
        reversion and ``theta`` is returned as ``0.0`` with infinite half-life
        rather than raising -- a non-reverting spread is a valid finding, and
        the caller should check :attr:`~OUParameters.is_mean_reverting`.
    Complexity
        :math:`O(n)`.

    Example
        >>> import numpy as np
        >>> x = simulate_ou(4.0, 0.0, 0.5, n=20000, dt=1/252,
        ...                 rng=np.random.default_rng(0))
        >>> fit = fit_ou(x, dt=1/252)
        >>> bool(3.0 < fit.theta < 5.0)
        True
    """
    from quantos.core.timeseries.ols import ols

    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}")
    if dt <= 0:
        raise ValueError("dt must be positive")

    y = a[1:]
    design = np.column_stack([np.ones(n - 1), a[:-1]])
    fit = ols(y, design)
    intercept, phi = float(fit.coefficients[0]), float(fit.coefficients[1])
    residual_std = float(np.std(fit.residuals, ddof=2))

    if phi <= 0.0 or phi >= 1.0:
        # phi >= 1: no reversion (or explosive). phi <= 0: reversion so fast
        # that it overshoots every step, which the continuous-time OU model
        # cannot represent -- the data are over-sampled relative to theta.
        return OUParameters(
            theta=0.0,
            mu=float(a.mean()),
            sigma=residual_std / np.sqrt(dt),
            dt=dt,
            n_obs=n,
            r_squared=fit.r_squared,
        )

    theta = -np.log(phi) / dt
    mu = intercept / (1.0 - phi)
    sigma = residual_std * np.sqrt(2.0 * theta / (1.0 - phi * phi))
    return OUParameters(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        dt=dt,
        n_obs=n,
        r_squared=fit.r_squared,
    )


def simulate_ou(
    theta: float,
    mu: float,
    sigma: float,
    *,
    n: int,
    dt: float = 1.0,
    x0: float | None = None,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    r"""Simulate an OU path using the **exact** transition density.

    Because the OU transition law is Gaussian in closed form, there is no reason
    to use an Euler-Maruyama scheme here: the exact update

    .. math:: X_{t+\Delta} = \mu + (X_t - \mu)e^{-\theta\Delta}
              + \sigma\sqrt{\frac{1-e^{-2\theta\Delta}}{2\theta}}\, Z

    is unbiased at *any* step size, whereas Euler is biased at every step size.
    The simulation therefore validates the estimator without confounding
    discretisation error with estimation error -- which is what makes
    ``tests/core/test_ou.py`` a meaningful test.

    ``x0`` defaults to a draw from the stationary distribution, so the path is
    stationary from its first observation rather than needing a burn-in.
    """
    if theta <= 0 or sigma <= 0 or n < 1 or dt <= 0:
        raise ValueError("require theta > 0, sigma > 0, n >= 1, dt > 0")
    stationary_std = sigma / np.sqrt(2.0 * theta)
    start = mu + stationary_std * rng.standard_normal() if x0 is None else x0

    decay = float(np.exp(-theta * dt))
    shock_std = float(sigma * np.sqrt((1.0 - decay * decay) / (2.0 * theta)))
    shocks = rng.standard_normal(n) * shock_std

    out = np.empty(n)
    current = start
    for i in range(n):
        current = mu + (current - mu) * decay + shocks[i]
        out[i] = current
    return out


def expected_time_to_mean(params: OUParameters, start: float) -> float:
    r"""Expected first-passage time from ``start`` to the long-run mean.

    Purpose
        Turn a pair's estimated parameters into an *expected holding period*.
        That number determines capital turnover, and hence whether the
        strategy's gross Sharpe ratio survives financing and borrow costs. A
        z-score entry threshold chosen without reference to it is chosen blind.

    Derivation
        Standardise to :math:`U = (X-\mu)/\sigma_{\text{stat}}` and rescale time
        by :math:`\theta`, giving :math:`dU = -U\,dt + \sqrt{2}\,dW`. The
        expected hitting time of the origin, :math:`T(u)`, solves the ODE
        :math:`T'' - u T' = -1` with :math:`T(0) = 0`. Writing
        :math:`v = T'` gives a first-order linear equation whose
        integrating factor is :math:`e^{-u^2/2}`:

        .. math:: T'(u) = e^{u^2/2}\!\int_u^{\infty}\! e^{-s^2/2}\,ds
                        = \sqrt{\tfrac{\pi}{2}}\, e^{u^2/2}
                          \operatorname{erfc}(u/\sqrt2)

        where the constant of integration is fixed by requiring :math:`T'` to
        stay bounded as :math:`u \to \infty`. Integrating once more,

        .. math:: T(u) = \sqrt{\tfrac{\pi}{2}} \int_0^{|u|}
                         e^{v^2/2}\operatorname{erfc}(v/\sqrt2)\,dv ,

        and the answer in the original time unit is :math:`T(u)/\theta`. The
        modulus reflects the symmetry :math:`T(-u) = T(u)`.

    Sanity check
        Starting two stationary standard deviations out, the expected time to
        revert is close to two half-lives -- which is the intuition, and a
        useful check that the quadrature is not off by a factor.

    Inputs
        ``start`` -- level in the process's own units.
    Outputs
        Expected time in the same units as ``params.dt``.
    Failure modes
        Returns ``inf`` for a non-reverting process (``theta <= 0``), and
        ``0.0`` when ``start`` already equals the mean.

    Note
        Restricted to the mean as the target on purpose. For a general barrier
        the two-sided problem needs a second boundary condition, and which one
        is appropriate depends on the trading rule (a stop-loss is an absorbing
        barrier, a wider band is not). Solving it in closed form for an
        unspecified rule would be a formula with no defined meaning.

    Example
        >>> import numpy as np
        >>> params = OUParameters(theta=4.0, mu=0.0, sigma=0.5, dt=1/252)
        >>> t = expected_time_to_mean(params, 2 * params.stationary_std)
        >>> bool(1.0 < t / params.half_life < 3.0)      # ~2 half-lives
        True
    """
    from quantos.core.numerics import adaptive_quad
    from quantos.core.special import erfc

    if not params.is_mean_reverting:
        return float("inf")
    scale = params.stationary_std
    u = abs(start - params.mu) / scale
    if u == 0.0:
        return 0.0

    def integrand(v: float) -> float:
        return float(np.exp(0.5 * v * v) * erfc(v / np.sqrt(2.0)))

    integral = adaptive_quad(integrand, 0.0, u, tol=1e-10)
    return float(np.sqrt(np.pi / 2.0) * integral / params.theta)
