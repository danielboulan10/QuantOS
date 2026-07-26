r"""Fit a volatility surface, and measure what the options market charges for risk.

Two things live here, and the second is the point
--------------------------------------------------
**Fitting a smile** (SVI) turns a scatter of quoted implied volatilities into a
curve you can interpolate, differentiate, and check for arbitrage.

**Measuring the variance risk premium** answers a question a fitted curve cannot:
*is the options market overcharging for volatility?* Implied variance has
exceeded subsequent realised variance in equity indices for as long as options
have been quoted -- typically by two to four points of volatility. That gap is
one of the most robust risk premia in finance, and unlike most published
anomalies it does not require a trading rule to observe. It is simply the
difference between a price and an outcome.

SVI, and why this parameterisation
-----------------------------------
Gatheral's stochastic volatility inspired form models *total* implied variance
:math:`w(k) = \sigma^2_{\text{imp}} T` against log-moneyness
:math:`k = \log(K/F)`:

.. math::
   w(k) = a + b\left[\rho(k - m) + \sqrt{(k-m)^2 + s^2}\right]

Five parameters, each of which does one thing: :math:`a` sets the level,
:math:`b` the wing slope, :math:`\rho` the skew, :math:`m` the horizontal shift
and :math:`s` the curvature at the money.

The reason to use it rather than a polynomial is asymptotic. Lee's moment formula
says total implied variance must grow **linearly** in :math:`|k|` in the wings --
no faster, or the underlying would have no finite moments; no slower, or the
density would be negative. SVI is linear in the wings by construction. A quadratic
fit is not: extrapolate a parabola a little too far and it curves back down,
producing implied volatilities that fall to zero and then go imaginary. That is
not a rare edge case; it happens on any day the wings are sparse.

No-arbitrage is checked, not assumed
-------------------------------------
A fitted curve can be smooth, accurate, and still admit arbitrage. Two conditions
are checked explicitly:

**Butterfly** (no negative densities). The risk-neutral density implied by the
curve must be non-negative everywhere. This is Durrleman's condition; a violation
means the fitted surface prices a butterfly spread at a negative value, which is
free money and therefore wrong.

**Calendar** (no negative forward variance). Total variance must be
non-decreasing in maturity at every strike, or the surface implies negative
variance over some future interval.

A fit that violates either is reported as violating it. Silently returning it
would be worse than not fitting at all.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantos.core.optimize.minimize import nelder_mead
from quantos.data.options import OptionChain

__all__ = [
    "SVIParameters",
    "SmileFit",
    "SurfaceFit",
    "VarianceRiskPremium",
    "fit_surface",
    "fit_svi",
    "model_free_implied_variance",
    "variance_risk_premium",
]


@dataclass(frozen=True)
class SVIParameters:
    r"""The five raw SVI parameters for one expiry."""

    a: float
    b: float
    rho: float
    m: float
    s: float
    time_to_expiry: float

    def total_variance(self, k: NDArray[np.float64] | float) -> NDArray[np.float64]:
        r"""Total implied variance :math:`w(k) = \sigma^2 T`."""
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s**2))

    def implied_volatility(self, k: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Implied volatility at log-moneyness ``k``."""
        w = np.maximum(self.total_variance(k), 0.0)
        if self.time_to_expiry <= 0:
            return np.full_like(np.asarray(k, dtype=float), np.nan)
        return np.sqrt(w / self.time_to_expiry)

    @property
    def atm_volatility(self) -> float:
        """Implied volatility at the forward -- the surface's level."""
        return float(self.implied_volatility(0.0))

    @property
    def skew(self) -> float:
        r""":math:`\partial \sigma / \partial k` at the money.

        Negative in equity indices, essentially always: crash protection costs
        more than upside. The magnitude is what changes, and it widens before
        stress rather than after.
        """
        h = 1e-4
        up = float(self.implied_volatility(h))
        down = float(self.implied_volatility(-h))
        return (up - down) / (2 * h)

    def density_is_nonnegative(
        self, k_grid: NDArray[np.float64] | None = None
    ) -> tuple[bool, float]:
        r"""Durrleman's butterfly condition.

        Returns ``(passes, worst_value)``. The function

        .. math::
           g(k) = \left(1 - \frac{k w'}{2w}\right)^2
                  - \frac{(w')^2}{4}\left(\frac1w + \frac14\right) + \frac{w''}{2}

        is proportional to the risk-neutral density. Where it goes negative, the
        fitted surface prices a butterfly at a negative value.
        """
        if k_grid is None:
            k_grid = np.linspace(-1.5, 1.5, 601)
        h = 1e-5
        w = self.total_variance(k_grid)
        w_prime = (self.total_variance(k_grid + h) - self.total_variance(k_grid - h)) / (2 * h)
        w_second = (
            self.total_variance(k_grid + h) - 2 * w + self.total_variance(k_grid - h)
        ) / h**2

        with np.errstate(divide="ignore", invalid="ignore"):
            g = (
                (1 - k_grid * w_prime / (2 * w)) ** 2
                - (w_prime**2 / 4) * (1 / w + 0.25)
                + w_second / 2
            )
        finite = g[np.isfinite(g)]
        if finite.size == 0:
            return False, float("nan")
        worst = float(np.min(finite))
        return bool(worst >= -1e-8), worst


@dataclass
class SmileFit:
    """One expiry's fitted smile, with residuals and arbitrage checks."""

    expiry: np.datetime64
    time_to_expiry: float
    forward: float
    parameters: SVIParameters
    n_points: int
    rmse_vol_points: float
    max_error_vol_points: float
    butterfly_free: bool
    worst_density: float
    converged: bool
    notes: list[str] = field(default_factory=list)

    @property
    def quality(self) -> str:
        if not self.converged:
            return "did not converge"
        if not self.butterfly_free:
            return f"fits ({self.rmse_vol_points:.2f} vol pts) but admits butterfly arbitrage"
        if self.rmse_vol_points < 0.5:
            return f"good ({self.rmse_vol_points:.2f} vol pts RMSE)"
        if self.rmse_vol_points < 1.5:
            return f"acceptable ({self.rmse_vol_points:.2f} vol pts RMSE)"
        return f"poor ({self.rmse_vol_points:.2f} vol pts RMSE)"


@dataclass
class SurfaceFit:
    """Every expiry, plus the cross-expiry checks."""

    symbol: str
    smiles: list[SmileFit]
    calendar_arbitrage_free: bool = True
    calendar_violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def term_structure(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """At-the-money volatility against maturity."""
        times = np.array([s.time_to_expiry for s in self.smiles])
        vols = np.array([s.parameters.atm_volatility for s in self.smiles])
        return times, vols

    def skew_structure(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""At-the-money skew against maturity. Flattens as :math:`1/\sqrt{T}`."""
        times = np.array([s.time_to_expiry for s in self.smiles])
        skews = np.array([s.parameters.skew for s in self.smiles])
        return times, skews

    def summary(self) -> str:
        lines = [f"SVI surface for {self.symbol}: {len(self.smiles)} expiries"]
        for smile in self.smiles:
            lines.append(
                f"  {smile.expiry!s}  T={smile.time_to_expiry:.3f}  "
                f"n={smile.n_points:3d}  ATM={smile.parameters.atm_volatility:6.2%}  "
                f"skew={smile.parameters.skew:+7.3f}  {smile.quality}"
            )
        if not self.calendar_arbitrage_free:
            lines.append("  CALENDAR ARBITRAGE:")
            lines.extend(f"    {v}" for v in self.calendar_violations)
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def fit_svi(
    log_moneyness: NDArray[np.float64],
    implied_vols: NDArray[np.float64],
    time_to_expiry: float,
    *,
    weights: NDArray[np.float64] | None = None,
) -> tuple[SVIParameters, bool, float]:
    r"""Fit SVI to one expiry's smile.

    Method
        Least squares on **total variance**, not on volatility. Fitting variance
        makes the objective far closer to quadratic in the parameters, which
        matters because the fit is unconstrained and the wings would otherwise
        dominate the gradient.
    Parameterisation of the constraints
        Rather than constrained optimisation, the raw parameters are mapped from
        an unconstrained space: :math:`b = e^{\beta} > 0`, :math:`\rho =
        \tanh(\tilde\rho) \in (-1, 1)` and :math:`s = e^{\varsigma} > 0`. This
        keeps the iterate feasible at every step, so the simplex never evaluates
        a nonsensical curve and there are no penalty terms to tune.
    Inputs
        ``log_moneyness`` -- :math:`\log(K/F)`. Must be the forward, not spot.
    Outputs
        ``(parameters, converged, rmse_in_vol_points)``.
    Failure modes
        Fewer than five points cannot determine five parameters; the fit is
        returned with ``converged=False`` rather than silently overfitting.

    Example
        >>> import numpy as np
        >>> k = np.linspace(-0.3, 0.3, 15)
        >>> true = SVIParameters(0.02, 0.15, -0.5, 0.01, 0.10, 1.0)
        >>> vols = true.implied_volatility(k)
        >>> fitted, ok, rmse = fit_svi(k, vols, 1.0)
        >>> ok and rmse < 0.05
        True
    """
    k = np.asarray(log_moneyness, dtype=float)
    vols = np.asarray(implied_vols, dtype=float)
    good = np.isfinite(k) & np.isfinite(vols) & (vols > 0)
    k, vols = k[good], vols[good]

    if k.size < 5:
        fallback = SVIParameters(
            a=float(np.mean(vols**2) * time_to_expiry) if vols.size else 0.04,
            b=0.1,
            rho=0.0,
            m=0.0,
            s=0.1,
            time_to_expiry=time_to_expiry,
        )
        return fallback, False, float("nan")

    target = vols**2 * time_to_expiry
    weight = np.ones_like(target) if weights is None else np.asarray(weights, dtype=float)[good]

    def unpack(theta: NDArray[np.float64]) -> SVIParameters:
        return SVIParameters(
            a=float(theta[0]),
            b=float(np.exp(np.clip(theta[1], -20, 5))),
            rho=float(np.tanh(theta[2])),
            m=float(theta[3]),
            s=float(np.exp(np.clip(theta[4], -20, 5))),
            time_to_expiry=time_to_expiry,
        )

    def objective(theta: NDArray[np.float64]) -> float:
        model = unpack(theta).total_variance(k)
        if not np.all(np.isfinite(model)):
            return 1e12
        # Total variance must stay positive; penalise rather than reject so the
        # simplex is guided back rather than walking a flat infeasible plateau.
        penalty = float(np.sum(np.minimum(model, 0.0) ** 2)) * 1e6
        return float(np.sum(weight * (model - target) ** 2)) + penalty

    atm_variance = float(np.interp(0.0, k, target))
    start = np.array(
        [
            0.5 * atm_variance,  # a: half the level, wings supply the rest
            np.log(max(atm_variance, 1e-4)),  # b
            np.arctanh(-0.3),  # rho: equities skew negative
            0.0,  # m
            np.log(0.1),  # s
        ]
    )

    # Restart the simplex until the fitted CURVE stops moving.
    #
    # Two separate facts force this shape. First, SVI is over-parameterised:
    # (a, b, m) trade off along a nearly flat valley, so the simplex creeps along
    # it indefinitely and ``result.converged`` stays False even for a fit that
    # reproduces the inputs to a thousandth of a volatility point -- measured
    # directly, 20,000 iterations on a clean synthetic smile still reported
    # failure while recovering the true skew to four decimals. So the optimiser's
    # own flag cannot be used, and loosening its tolerance until it claims
    # success would be worse: it would claim success for bad fits too.
    #
    # Second, a Nelder-Mead simplex can collapse into a degenerate shape and stall
    # short of the minimum, which a restart fixes. But a restart that *improves*
    # the fit is a success, not an instability -- an earlier version of this
    # function compared the curve before and after one restart and reported the
    # improvement as non-convergence, failing a fit whose RMSE was 0.06 volatility
    # points.
    #
    # Iterating to a fixed point resolves both: restart until a restart no longer
    # changes the curve. What is then reported is what a caller actually needs to
    # know -- the curve matches the quotes, and it is stable.
    STABILITY_VOL_POINTS = 0.02  # far inside any real bid-ask implied uncertainty
    MAX_RESTARTS = 6

    result = nelder_mead(objective, start, max_iter=4000, xtol=1e-10, ftol=1e-12)
    curve = unpack(np.asarray(result.x, dtype=float)).implied_volatility(k)
    stable = False

    for _ in range(MAX_RESTARTS):
        restarted = nelder_mead(
            objective, np.asarray(result.x, dtype=float), max_iter=4000, xtol=1e-10, ftol=1e-12
        )
        if restarted.fun > result.fun:  # never accept a worse point
            stable = True
            break
        new_curve = unpack(np.asarray(restarted.x, dtype=float)).implied_volatility(k)
        moved = float(np.max(np.abs(new_curve - curve)) * 100)
        result, curve = restarted, new_curve
        if moved < STABILITY_VOL_POINTS:
            stable = True
            break

    parameters = unpack(np.asarray(result.x, dtype=float))
    residual = parameters.implied_volatility(k) - vols
    rmse = float(np.sqrt(np.mean(residual**2)) * 100)  # volatility points

    converged = bool(np.isfinite(rmse) and rmse < 0.5 and stable)
    return parameters, converged, rmse


def fit_surface(chain: OptionChain, *, min_points: int = 5) -> SurfaceFit:
    """Fit every expiry in a chain and run the cross-expiry arbitrage check.

    Weighting
        Points are weighted by ``1/(1 + |k|)``, which favours the at-the-money
        region. That is where the options actually trade; a wing quote is a real
        price but it is one contract of open interest, and letting it pull the
        level around is how a surface ends up fitting noise precisely.
    """
    smiles: list[SmileFit] = []
    notes: list[str] = []

    for expiry in chain.unique_expiries():
        data = chain.slice_at(expiry)
        k = data.log_moneyness
        vols = data.implied_vols
        if k.size < min_points:
            notes.append(f"{expiry}: only {k.size} points, skipped")
            continue

        weights = 1.0 / (1.0 + np.abs(k))
        parameters, converged, rmse = fit_svi(k, vols, data.time_to_expiry, weights=weights)
        butterfly_free, worst = parameters.density_is_nonnegative()
        errors = np.abs(parameters.implied_volatility(k) - vols) * 100

        smiles.append(
            SmileFit(
                expiry=expiry,
                time_to_expiry=data.time_to_expiry,
                forward=data.forward,
                parameters=parameters,
                n_points=int(k.size),
                rmse_vol_points=rmse,
                max_error_vol_points=float(np.max(errors)) if errors.size else float("nan"),
                butterfly_free=butterfly_free,
                worst_density=worst,
                converged=converged,
            )
        )

    # Calendar check: total variance must not decrease with maturity.
    violations: list[str] = []
    k_grid = np.linspace(-0.5, 0.5, 101)
    ordered = sorted(smiles, key=lambda s: s.time_to_expiry)
    for earlier, later in itertools.pairwise(ordered):
        gap = later.parameters.total_variance(k_grid) - earlier.parameters.total_variance(k_grid)
        if np.any(gap < -1e-8):
            worst_k = float(k_grid[int(np.argmin(gap))])
            violations.append(
                f"{earlier.expiry} -> {later.expiry}: total variance falls by "
                f"{-float(np.min(gap)):.5f} at k={worst_k:+.2f}"
            )

    return SurfaceFit(
        symbol=chain.symbol,
        smiles=ordered,
        calendar_arbitrage_free=not violations,
        calendar_violations=violations,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Model-free implied variance and the variance risk premium
# --------------------------------------------------------------------------- #
def model_free_implied_variance(
    strikes: NDArray[np.float64],
    prices: NDArray[np.float64],
    is_call: NDArray[np.bool_],
    forward: float,
    time_to_expiry: float,
    rate: float = 0.0,
) -> float:
    r"""Implied variance without assuming any model -- the VIX construction.

    Purpose
        Estimate the risk-neutral expected variance over ``time_to_expiry``
        directly from a strip of option prices.
    Why model-free
        A Black-Scholes implied volatility is the volatility that *would* justify
        the price under an assumption everyone knows to be false. The
        Demeterfi-Derman-Kamal-Zou result avoids that entirely: a variance swap's
        fair strike is replicated exactly by a static portfolio of OTM options
        weighted :math:`1/K^2`, with no volatility model at all.

        .. math::
           \sigma^2 = \frac{2e^{rT}}{T}\sum_i \frac{\Delta K_i}{K_i^2} Q(K_i)
                      - \frac1T\left(\frac{F}{K_0} - 1\right)^2

        This is the CBOE's own VIX formula, and it is the correct input to a
        variance risk premium: comparing a model-free implied variance against a
        realised variance compares like with like.
    Inputs
        OTM options only: puts below the forward, calls above. Passing ITM
        options double-counts intrinsic value and inflates the result.
    Outputs
        Annualised variance. Take the square root for a VIX-style volatility.
    Failure modes
        Fewer than three strikes returns NaN -- the integral is not estimable.
        A strike grid that does not straddle the forward returns NaN, because the
        correction term is then extrapolation rather than interpolation.

    Example
        >>> import numpy as np
        >>> # A flat 20% smile must reproduce 20% variance to within discretisation.
        >>> from quantos.derivatives.black_scholes import black_scholes_price, OptionType
        >>> F, T, sigma = 100.0, 1.0, 0.20
        >>> K = np.arange(40.0, 181.0, 1.0)
        >>> calls = K >= F
        >>> px = np.array([
        ...     float(black_scholes_price(F, k, T, sigma,
        ...           option_type=OptionType.CALL if c else OptionType.PUT))
        ...     for k, c in zip(K, calls)])
        >>> var = model_free_implied_variance(K, px, calls, F, T)
        >>> bool(abs(np.sqrt(var) - 0.20) < 0.005)
        True
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    good = np.isfinite(strikes) & np.isfinite(prices) & (prices > 0) & (strikes > 0)
    strikes, prices, is_call = strikes[good], prices[good], is_call[good]
    if strikes.size < 3 or time_to_expiry <= 0:
        return float("nan")

    order = np.argsort(strikes)
    strikes, prices, is_call = strikes[order], prices[order], is_call[order]

    if not (strikes[0] < forward < strikes[-1]):
        return float("nan")

    # K0: the largest strike at or below the forward, per the CBOE definition.
    below = strikes[strikes <= forward]
    if below.size == 0:
        return float("nan")
    k0 = float(below[-1])

    # Trapezoidal strike spacing: half the distance to each neighbour, with the
    # endpoints taking their single interval. Using a constant spacing here is a
    # common error and biases the tails, where strikes are widest.
    delta_k = np.empty_like(strikes)
    if strikes.size > 2:
        delta_k[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
    delta_k[0] = strikes[1] - strikes[0]
    delta_k[-1] = strikes[-1] - strikes[-2]

    contribution = float(np.sum(delta_k / strikes**2 * prices))
    discount = float(np.exp(rate * time_to_expiry))

    variance = (2.0 / time_to_expiry) * discount * contribution - (1.0 / time_to_expiry) * (
        forward / k0 - 1.0
    ) ** 2
    return float(variance)


@dataclass
class VarianceRiskPremium:
    """What the options market charged for variance, against what happened."""

    symbol: str
    time_to_expiry: float
    implied_volatility: float
    realised_volatility: float
    #: Implied minus realised, in variance units.
    premium_variance: float
    #: The same gap expressed in volatility points, which is how desks quote it.
    premium_vol_points: float
    n_realised_observations: int
    notes: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """Implied over realised. Around 1.1-1.3 in equity indices."""
        if self.realised_volatility <= 0:
            return float("nan")
        return self.implied_volatility / self.realised_volatility

    @property
    def interpretation(self) -> str:
        if not np.isfinite(self.premium_vol_points):
            return "not computable"
        if self.premium_vol_points > 1.0:
            return (
                f"options were expensive: implied exceeded realised by "
                f"{self.premium_vol_points:.2f} volatility points "
                f"(ratio {self.ratio:.2f}). Selling variance would have paid, "
                "as it usually does -- and as it catastrophically does not in the "
                "one month per decade that matters."
            )
        if self.premium_vol_points < -1.0:
            return (
                f"options were cheap: realised exceeded implied by "
                f"{-self.premium_vol_points:.2f} volatility points. This is the "
                "minority outcome, and it is where variance sellers lose years of "
                "premium in days."
            )
        return (
            f"implied and realised were within {abs(self.premium_vol_points):.2f} volatility points"
        )

    def summary(self) -> str:
        return (
            f"{self.symbol} variance risk premium over {self.time_to_expiry * 365.25:.0f} days\n"
            f"  implied  {self.implied_volatility:6.2%}\n"
            f"  realised {self.realised_volatility:6.2%}  "
            f"({self.n_realised_observations} observations)\n"
            f"  premium  {self.premium_vol_points:+6.2f} volatility points\n"
            f"  {self.interpretation}"
        )


def variance_risk_premium(
    chain: OptionChain,
    realised_returns: NDArray[np.float64],
    *,
    expiry: np.datetime64 | None = None,
    periods_per_year: float = 252.0,
) -> VarianceRiskPremium:
    r"""Compare model-free implied variance against subsequent realised variance.

    Purpose
        Measure the single most persistent premium in derivatives markets.
    Inputs
        ``chain`` -- the option chain as of the decision date.
        ``realised_returns`` -- log returns **over the period the option
        covered**, i.e. from the chain date forward to expiry. Passing trailing
        returns instead measures something different and much less interesting
        (whether the market extrapolates), so the direction matters.
    Outputs
        A :class:`VarianceRiskPremium`.
    Failure modes
        Too few realised observations makes the realised leg noisier than the
        premium being measured; a note records this rather than the result
        pretending to precision it does not have.

    Realised variance convention
        Zero-mean sum of squares, not the sample variance about the mean.
        Subtracting a drift estimated from thirty observations removes a quantity
        known far less precisely than the variance itself, and the variance-swap
        payoff it is being compared against does not subtract one either.
    """
    expiries = chain.unique_expiries()
    if not expiries:
        raise ValueError(f"{chain.symbol}: chain has no usable expiries")
    chosen = expiry if expiry is not None else expiries[0]

    data = chain.slice_at(chosen)
    implied_variance = model_free_implied_variance(
        data.strikes,
        data.mids,
        data.is_call,
        data.forward,
        data.time_to_expiry,
        chain.rate,
    )

    returns = np.asarray(realised_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    notes: list[str] = []

    if returns.size < 5:
        realised_variance = float("nan")
        notes.append("fewer than five realised observations; the realised leg is not meaningful")
    else:
        realised_variance = float(np.sum(returns**2) / returns.size * periods_per_year)
        if returns.size < 15:
            notes.append(
                f"only {returns.size} realised observations: the realised leg has a "
                "standard error of roughly "
                f"{100 * np.sqrt(realised_variance) / np.sqrt(2 * returns.size):.1f} "
                "volatility points, which may exceed the premium being measured"
            )

    implied_vol = float(np.sqrt(implied_variance)) if implied_variance > 0 else float("nan")
    realised_vol = float(np.sqrt(realised_variance)) if realised_variance > 0 else float("nan")

    if not np.isfinite(implied_variance):
        notes.append(
            "model-free implied variance is not computable for this expiry: the "
            "strike grid must straddle the forward and hold at least three strikes"
        )

    return VarianceRiskPremium(
        symbol=chain.symbol,
        time_to_expiry=data.time_to_expiry,
        implied_volatility=implied_vol,
        realised_volatility=realised_vol,
        premium_variance=float(implied_variance - realised_variance),
        premium_vol_points=float((implied_vol - realised_vol) * 100),
        n_realised_observations=int(returns.size),
        notes=notes,
    )
