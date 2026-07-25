r"""Black-Scholes-Merton pricing, the full Greek set, and implied volatility.

Scope
-----
European options on an asset with continuous dividend yield :math:`q`:

.. math::
    C = S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2), \qquad
    d_{1,2} = \frac{\ln(S/K) + (r - q \pm \sigma^2/2)T}{\sigma\sqrt{T}}

The Black-76 forward form is obtained with :math:`q = r`, so futures options need
no separate implementation.

Numerical care, which is most of the work
-----------------------------------------
The formula is trivial; the edge cases are where implementations fail. Handled
explicitly here:

* **:math:`T \to 0`.** :math:`d_1, d_2 \to \pm\infty` and vega, gamma and theta
  all degenerate. We return the intrinsic value and exact limiting Greeks
  rather than dividing by :math:`\sqrt{T} = 0`.
* **:math:`\sigma \to 0`.** The option becomes a deterministic forward payoff;
  again a limit, not a division.
* **Deep out of the money.** :math:`\Phi(d_2)` underflows. Because
  :func:`quantos.core.special.ndtr` routes through the ``erfc`` continued
  fraction, prices stay relatively accurate to ~1e-300 instead of collapsing to
  zero at ~8 standard deviations -- which matters for far-wing implied vols.
* **Implied volatility.** See :func:`implied_volatility`; naive Newton on vega
  diverges precisely where vega underflows.

Greeks are analytic throughout. Finite-difference Greeks are a legitimate
fallback for exotics but are the wrong choice here: they cost 2-3 revaluations
each, and their accuracy is limited to :math:`\sqrt{\varepsilon}` at best.
``tests/derivatives/test_black_scholes.py`` checks every analytic Greek against a
central difference of the price and against put-call parity relations.

References
----------
Black, F. & Scholes, M. (1973), *J. Political Economy* 81(3), 637-654.
Merton, R. C. (1973), *Bell J. Econ. Manag. Sci.* 4(1), 141-183.
Jaeckel, P. (2015), "Let's be rational", *Wilmott* 2015(75), 40-53.
Haug, E. G. (2007), *The Complete Guide to Option Pricing Formulas* (2nd ed.).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.special import ndtr, norm_pdf

__all__ = [
    "BlackScholesInputs",
    "Greeks",
    "OptionType",
    "black_scholes_greeks",
    "black_scholes_price",
    "forward_price",
    "implied_volatility",
    "put_call_parity_check",
]


class OptionType(enum.Enum):
    """Call or put. Values are the payoff sign, usable as a multiplier."""

    CALL = 1
    PUT = -1

    @property
    def sign(self) -> int:
        return int(self.value)

    @property
    def opposite(self) -> OptionType:
        return OptionType.PUT if self is OptionType.CALL else OptionType.CALL


@dataclass(frozen=True)
class BlackScholesInputs:
    """Validated Black-Scholes inputs.

    Validation is centralised here rather than repeated in every pricer. Silently
    accepting a negative time to expiry or a negative volatility produces NaNs
    that surface much later, usually inside a calibration, where the cause is far
    harder to find.
    """

    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    rate: float = 0.0
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.time_to_expiry < 0:
            raise ValueError(f"time_to_expiry must be non-negative, got {self.time_to_expiry}")
        if self.volatility < 0:
            raise ValueError(f"volatility must be non-negative, got {self.volatility}")

    @property
    def forward(self) -> float:
        r""":math:`F = S e^{(r-q)T}`."""
        return float(self.spot * np.exp((self.rate - self.dividend_yield) * self.time_to_expiry))

    @property
    def total_variance(self) -> float:
        r""":math:`\sigma^2 T` -- the only combination that matters for the shape."""
        return float(self.volatility**2 * self.time_to_expiry)

    @property
    def log_moneyness(self) -> float:
        r""":math:`\ln(F/K)`, the natural coordinate for a volatility surface."""
        return float(np.log(self.forward / self.strike))


def forward_price(
    spot: ArrayLike, rate: float, dividend_yield: float, time: ArrayLike
) -> NDArray[np.float64]:
    r"""Forward price :math:`F = S e^{(r-q)T}`."""
    return np.asarray(spot, dtype=float) * np.exp(
        (rate - dividend_yield) * np.asarray(time, dtype=float)
    )


def _d1_d2(
    spot: NDArray[np.float64],
    strike: NDArray[np.float64],
    time: NDArray[np.float64],
    vol: NDArray[np.float64],
    rate: float,
    dividend: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Return ``(d1, d2, sqrt_variance)``, guarding the degenerate branch.

    ``sqrt_variance`` is :math:`\sigma\sqrt{T}`. Where it is zero the option has
    no remaining uncertainty, and ``d1``/``d2`` are set to :math:`\pm\infty`
    according to whether the forward is above or below the strike -- which makes
    every downstream formula produce the correct deterministic limit without a
    special case of its own.
    """
    sqrt_var = vol * np.sqrt(time)
    degenerate = sqrt_var <= 0.0
    safe = np.where(degenerate, 1.0, sqrt_var)

    log_ratio = np.log(spot / strike) + (rate - dividend) * time
    d1 = (log_ratio + 0.5 * vol * vol * time) / safe
    d2 = d1 - safe

    # Deterministic limit: in-the-money forward -> +inf, out -> -inf.
    limit = np.where(log_ratio > 0.0, np.inf, np.where(log_ratio < 0.0, -np.inf, 0.0))
    d1 = np.where(degenerate, limit, d1)
    d2 = np.where(degenerate, limit, d2)
    return d1, d2, sqrt_var


def black_scholes_price(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> NDArray[np.float64]:
    r"""Black-Scholes-Merton price, vectorised over every argument.

    Purpose
        Price European options; the reference against which the tree, PDE, Monte
        Carlo and stochastic-volatility engines in this package are validated.
    Inputs
        Broadcastable arrays. ``volatility`` and ``time_to_expiry`` may be zero.
    Outputs
        Prices, shaped by broadcasting.
    Complexity
        :math:`O(n)`, two normal CDF evaluations per element.
    Failure modes
        Negative spot, strike, time or volatility produce NaN rather than raising
        (the array API makes per-element exceptions impractical); use
        :class:`BlackScholesInputs` when you want validation.

    Example
        >>> import numpy as np
        >>> float(np.round(black_scholes_price(100, 100, 1.0, 0.2, rate=0.05), 6))
        10.450584
        >>> float(np.round(black_scholes_price(100, 100, 1.0, 0.2, rate=0.05,
        ...                option_type=OptionType.PUT), 6))
        5.573526
        >>> float(black_scholes_price(100, 90, 0.0, 0.2))    # expiry: intrinsic
        10.0
    """
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(time_to_expiry, dtype=float)
    v = np.asarray(volatility, dtype=float)
    s, k, t, v = np.broadcast_arrays(s, k, t, v)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2, _ = _d1_d2(s, k, t, v, rate, dividend_yield)
        discount = np.exp(-rate * t)
        carry = np.exp(-dividend_yield * t)
        sign = option_type.sign
        price = sign * (s * carry * ndtr(sign * d1) - k * discount * ndtr(sign * d2))

    invalid = (s <= 0) | (k <= 0) | (t < 0) | (v < 0)
    return np.where(invalid, np.nan, price)


@dataclass(frozen=True)
class Greeks:
    r"""The full first- and second-order Greek set.

    Conventions, since they vary between references and getting them wrong is
    the commonest source of risk-report disagreement:

    * ``delta`` -- per unit change in spot.
    * ``gamma`` -- per unit change in spot, of delta.
    * ``vega`` -- per **1.00** change in volatility (not per vol point). Divide
      by 100 for the per-percentage-point convention.
    * ``theta`` -- per **year**. Divide by 365 for per-calendar-day.
    * ``rho`` -- per 1.00 change in the rate.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    #: dVega/dVol -- the convexity of value in volatility.
    volga: float = float("nan")
    #: dVega/dSpot = dDelta/dVol. The key skew-risk sensitivity.
    vanna: float = float("nan")
    #: dDelta/dTime.
    charm: float = float("nan")
    #: Sensitivity to the dividend yield / carry.
    epsilon: float = float("nan")
    #: Discounted probability of finishing in the money (= -dPrice/dStrike).
    dual_delta: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in vars(self).items()}


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> Greeks:
    r"""Analytic Greeks for a European option.

    Purpose
        Supply exact sensitivities for hedging and risk aggregation.
    Method
        Closed-form differentiation of the Black-Scholes formula. The scalar
        signature is deliberate: Greeks are usually wanted for one contract at a
        time, and a scalar API lets the degenerate cases be handled by explicit
        branches rather than by masked array arithmetic.
    Degenerate cases
        At :math:`T = 0` or :math:`\sigma = 0` the price is deterministic. Delta
        becomes the indicator of finishing in the money, gamma and vega vanish,
        and theta is zero. These are returned exactly rather than as NaN.
    Complexity
        :math:`O(1)`.

    Example
        >>> g = black_scholes_greeks(100, 100, 1.0, 0.2, rate=0.05)
        >>> round(g.delta, 6), round(g.gamma, 6), round(g.vega, 6)
        (0.636831, 0.018762, 37.524035)
        >>> # Put-call parity for delta: delta_call - delta_put = exp(-qT)
        >>> p = black_scholes_greeks(100, 100, 1.0, 0.2, rate=0.05,
        ...                         option_type=OptionType.PUT)
        >>> round(g.delta - p.delta, 10)
        1.0
    """
    inputs = BlackScholesInputs(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    s, k, t, v = spot, strike, time_to_expiry, volatility
    sign = option_type.sign
    discount = float(np.exp(-rate * t))
    carry = float(np.exp(-dividend_yield * t))
    price = float(
        black_scholes_price(
            s, k, t, v, rate=rate, dividend_yield=dividend_yield, option_type=option_type
        )
    )

    if t <= 0.0 or v <= 0.0:
        # Deterministic payoff on the forward.
        in_the_money = (inputs.forward - k) * sign > 0.0
        return Greeks(
            price=price,
            delta=float(sign * carry) if in_the_money else 0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=float(sign * k * t * discount) if in_the_money else 0.0,
            volga=0.0,
            vanna=0.0,
            charm=0.0,
            epsilon=float(-sign * s * t * carry) if in_the_money else 0.0,
            dual_delta=float(-sign * discount) if in_the_money else 0.0,
        )

    sqrt_t = float(np.sqrt(t))
    d1 = float((np.log(s / k) + (rate - dividend_yield + 0.5 * v * v) * t) / (v * sqrt_t))
    d2 = d1 - v * sqrt_t
    pdf_d1 = float(norm_pdf(np.array(d1)))
    cdf_sd1 = float(ndtr(np.array(sign * d1)))
    cdf_sd2 = float(ndtr(np.array(sign * d2)))

    delta = sign * carry * cdf_sd1
    gamma = carry * pdf_d1 / (s * v * sqrt_t)
    vega = s * carry * pdf_d1 * sqrt_t
    theta = (
        -s * carry * pdf_d1 * v / (2.0 * sqrt_t)
        + sign * dividend_yield * s * carry * cdf_sd1
        - sign * rate * k * discount * cdf_sd2
    )
    rho = sign * k * t * discount * cdf_sd2

    # Second order.
    volga = vega * d1 * d2 / v
    vanna = -carry * pdf_d1 * d2 / v
    charm = sign * dividend_yield * carry * cdf_sd1 - carry * pdf_d1 * (
        2.0 * (rate - dividend_yield) * t - d2 * v * sqrt_t
    ) / (2.0 * t * v * sqrt_t)
    epsilon = -sign * s * t * carry * cdf_sd1
    # dual_delta = -dPrice/dStrike. For a call dC/dK = -e^{-rT}Phi(d2), so the
    # negated derivative is +e^{-rT}Phi(d2) -- the discounted in-the-money
    # probability, which must be positive. Hence sign*, not -sign*.
    dual_delta = sign * discount * cdf_sd2

    return Greeks(
        price=price,
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        volga=float(volga),
        vanna=float(vanna),
        charm=float(charm),
        epsilon=float(epsilon),
        dual_delta=float(dual_delta),
    )


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
    tolerance: float = 1e-12,
    max_volatility: float = 10.0,
) -> float:
    r"""Invert Black-Scholes for volatility, robustly.

    Purpose
        Recover :math:`\sigma` from a market price. This is the single most
        frequently executed numerical routine in an options business, and the one
        most often implemented badly.
    Why naive Newton fails
        The obvious iteration :math:`\sigma \leftarrow \sigma - (C(\sigma) -
        C^{*})/\mathcal{V}(\sigma)` divides by vega, and vega **underflows** for
        deep out-of-the-money or very short-dated options. There the step becomes
        enormous, the iterate leaves :math:`(0, \infty)`, and the routine either
        returns nonsense or raises. That region is not exotic -- it is where the
        wings of every volatility surface live.
    Method
        1. **Check arbitrage bounds first.** A price outside
           :math:`[\max(0, \text{intrinsic}), \text{upper bound}]` has *no*
           implied volatility, and no amount of iteration will find one. Raise
           immediately with the violated bound rather than failing to converge.
        2. **Bracket** on :math:`[10^{-9}, \text{max\_volatility}]`. Price is
           strictly increasing in :math:`\sigma`, so a bracket always exists
           within the arbitrage bounds.
        3. **Safeguarded Newton** (:func:`~quantos.core.optimize.roots.newton_safeguarded`):
           take the Newton step when it stays inside the bracket and is
           contracting, otherwise bisect. Quadratic convergence in the normal
           case, guaranteed convergence in the pathological one.
    Inputs
        ``price`` -- the observed option price.
    Outputs
        Implied volatility as a decimal (0.2 = 20%).
    Failure modes
        :class:`ValueError` if the price violates an arbitrage bound or if
        ``time_to_expiry <= 0`` (an expired option has no implied volatility).
        :class:`~quantos.core.optimize.roots.ConvergenceError` never occurs in
        practice given the bracket, but is not suppressed if it does.

    Example
        >>> p = float(black_scholes_price(100, 100, 1.0, 0.23, rate=0.05))
        >>> round(implied_volatility(p, 100, 100, 1.0, rate=0.05), 10)
        0.23
        >>> # A far out-of-the-money option where vega is ~1e-9: naive Newton
        >>> # diverges here, this does not.
        >>> deep = float(black_scholes_price(100, 250, 0.05, 0.6))
        >>> round(implied_volatility(deep, 100, 250, 0.05), 6)
        0.6
    """
    from quantos.core.optimize.roots import newton_safeguarded

    if time_to_expiry <= 0:
        raise ValueError("an expired option has no implied volatility")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")

    discount = float(np.exp(-rate * time_to_expiry))
    carry = float(np.exp(-dividend_yield * time_to_expiry))
    forward = spot * carry / discount

    # Arbitrage bounds. Checking these first turns an opaque convergence failure
    # into an actionable error message naming the violated bound.
    if option_type is OptionType.CALL:
        lower = max(0.0, discount * (forward - strike))
        upper = spot * carry
    else:
        lower = max(0.0, discount * (strike - forward))
        upper = strike * discount

    if price < lower - 1e-12:
        raise ValueError(
            f"price {price:.10g} is below the no-arbitrage lower bound "
            f"{lower:.10g} (discounted intrinsic value); no implied volatility exists"
        )
    if price > upper + 1e-12:
        raise ValueError(
            f"price {price:.10g} exceeds the no-arbitrage upper bound {upper:.10g}; "
            "no implied volatility exists"
        )
    # Identifiability. Implied volatility is recoverable only from the option's
    # *time value*; when that falls below what float64 can represent next to the
    # intrinsic value, no algorithm can retrieve sigma and any returned number is
    # fabricated. For S=100, K=20, T=2 at 15% vol the true price and the
    # intrinsic value are bit-identical, and an unguarded solver happily returns
    # 0.0 -- a plausible-looking answer that is off by the entire volatility.
    # Raising here is the only honest option.
    time_value = price - lower
    resolution = 8.0 * float(np.finfo(float).eps) * max(spot, strike)
    if time_value <= resolution:
        raise ValueError(
            f"option time value ({time_value:.3g}) is at or below the float64 "
            f"resolution next to its intrinsic value ({resolution:.3g}); implied "
            f"volatility is not identifiable from this price. This is a precision "
            f"limit, not a solver failure -- the option is too deep in or out of "
            f"the money at this maturity to carry volatility information. Note that "
            f"sigma=0 does reprice the input exactly, but so does every sigma below "
            f"the identifiability floor, which is why returning a number here would "
            f"be misleading."
        )

    def objective(sigma: float) -> float:
        return (
            float(
                black_scholes_price(
                    spot,
                    strike,
                    time_to_expiry,
                    sigma,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=option_type,
                )
            )
            - price
        )

    def derivative(sigma: float) -> float:
        """Return vega, computed directly rather than via the Greeks object.

        Avoids constructing and validating a dataclass on every iteration.
        """
        sqrt_t = float(np.sqrt(time_to_expiry))
        d1 = float(
            (np.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma**2) * time_to_expiry)
            / (sigma * sqrt_t)
        )
        return float(spot * carry * norm_pdf(np.array(d1)) * sqrt_t)

    # Brenner-Subrahmanyam initial guess: sigma ~ sqrt(2*pi/T) * price / spot.
    # Exact for an at-the-money forward option, and a good starting point
    # elsewhere; the safeguarding handles the cases where it is poor.
    guess = float(np.sqrt(2.0 * np.pi / time_to_expiry) * price / spot)
    guess = min(max(guess, 1e-6), max_volatility * 0.5)

    # Converge on *volatility* (xtol), not on price (ftol). In the deep wings the
    # price is O(1e-11) and any ftol large enough to be reachable is satisfied
    # while sigma is still wrong in the fourth decimal -- measured at 0.60046
    # against a true 0.60 for a 5%-maturity, 2.5x-strike call. ftol=0 disables
    # early exit on the residual and forces the bracket to close on sigma.
    result = newton_safeguarded(
        objective, derivative, guess, 1e-9, max_volatility, xtol=tolerance, ftol=0.0
    )
    return float(result.root)


def put_call_parity_check(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    r"""Residual of put-call parity; zero means no static arbitrage.

    .. math:: C - P = S e^{-qT} - K e^{-rT}

    Purpose
        The cheapest and strongest sanity check on any option pricer or market
        data snapshot. It is *model-free*: it follows from replication alone, so
        a non-zero residual means either an arbitrage or bad data -- never a
        different volatility view. Every pricing engine in this package is checked
        against it.

    Example
        >>> c = float(black_scholes_price(100, 95, 0.75, 0.25, rate=0.04))
        >>> p = float(black_scholes_price(100, 95, 0.75, 0.25, rate=0.04,
        ...                              option_type=OptionType.PUT))
        >>> abs(put_call_parity_check(c, p, 100, 95, 0.75, rate=0.04)) < 1e-12
        True
    """
    return float(
        call_price
        - put_price
        - spot * np.exp(-dividend_yield * time_to_expiry)
        + strike * np.exp(-rate * time_to_expiry)
    )
