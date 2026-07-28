"""Validation for Heston pricing, and a demonstration of the branch cut.

The branch-cut tests are the reason this file is longer than the module deserves.
That failure is *silent*: the integral converges, the price looks plausible, and
it is wrong by tens of percent. A test suite that only checked "does it run" would
pass on the broken formulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.derivatives.black_scholes import OptionType, black_scholes_price
from quantos.derivatives.heston import (
    HestonParameters,
    characteristic_function,
    heston_implied_volatility,
    heston_price,
)


def equity_like(**kwargs) -> HestonParameters:
    base = {"kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7, "v0": 0.04}
    return HestonParameters(**{**base, **kwargs})


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        {"kappa": 0.0},
        {"theta": -0.01},
        {"xi": 0.0},
        {"v0": -1.0},
    ],
)
def test_nonpositive_parameters_are_refused(bad):
    with pytest.raises(ValueError, match="must be positive"):
        equity_like(**bad)


@pytest.mark.parametrize("rho", [-1.5, 1.5])
def test_a_correlation_outside_the_unit_interval_is_refused(rho):
    with pytest.raises(ValueError, match="rho must lie"):
        equity_like(rho=rho)


def test_the_feller_condition_is_reported():
    r"""2*kappa*theta > xi^2 decides whether variance can reach zero."""
    safe = HestonParameters(kappa=3.0, theta=0.04, xi=0.2, rho=-0.5, v0=0.04)
    risky = HestonParameters(kappa=0.5, theta=0.04, xi=1.0, rho=-0.5, v0=0.04)

    assert safe.feller_satisfied
    assert not risky.feller_satisfied
    assert "variance can reach zero" in risky.summary()


# --------------------------------------------------------------------------- #
# The characteristic function
# --------------------------------------------------------------------------- #
def test_the_characteristic_function_is_one_at_zero():
    """phi(0) = E[1] = 1 for any parameters. The cheapest possible sanity check."""
    phi = characteristic_function(np.array([0.0 + 0j]), equity_like(), 1.0)
    assert phi[0].real == pytest.approx(1.0)
    assert abs(phi[0].imag) < 1e-12


def test_the_characteristic_function_is_conjugate_symmetric():
    r"""phi(-u) = conj(phi(u)), because the underlying distribution is real."""
    u = np.array([0.5, 1.0, 3.0, 10.0], dtype=complex)
    positive = characteristic_function(u, equity_like(), 1.0)
    negative = characteristic_function(-u, equity_like(), 1.0)
    np.testing.assert_allclose(negative, np.conj(positive), rtol=1e-10)


def test_an_unknown_formulation_is_refused():
    with pytest.raises(ValueError, match="formulation must be"):
        characteristic_function(np.array([1.0 + 0j]), equity_like(), 1.0, formulation="magic")


# --------------------------------------------------------------------------- #
# Pricing against a known answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_zero_vol_of_vol_reproduces_black_scholes(strike):
    """With xi -> 0 the variance is deterministic and Heston must collapse to BS.

    This is the strongest available check: an independent closed form, exact.
    """
    volatility = 0.2
    parameters = HestonParameters(
        kappa=2.0, theta=volatility**2, xi=1e-4, rho=0.0, v0=volatility**2
    )
    heston = heston_price(100.0, strike, 1.0, parameters, rate=0.03)
    black = float(black_scholes_price(100.0, strike, 1.0, volatility, rate=0.03))
    assert heston == pytest.approx(black, abs=1e-6)


def test_put_call_parity_holds_exactly():
    """Puts come from parity rather than a second integral, so this is exact."""
    spot, strike, expiry, rate = 100.0, 95.0, 1.5, 0.03
    parameters = equity_like()
    call = heston_price(spot, strike, expiry, parameters, rate=rate, option_type=OptionType.CALL)
    put = heston_price(spot, strike, expiry, parameters, rate=rate, option_type=OptionType.PUT)
    assert call - put == pytest.approx(spot - strike * np.exp(-rate * expiry), abs=1e-10)


def test_prices_are_monotone_in_strike():
    parameters = equity_like()
    prices = [heston_price(100.0, k, 1.0, parameters) for k in (80, 90, 100, 110, 120)]
    assert prices == sorted(prices, reverse=True)


def test_prices_respect_the_arbitrage_bounds():
    parameters = equity_like()
    spot, strike, expiry, rate = 100.0, 100.0, 1.0, 0.03
    call = heston_price(spot, strike, expiry, parameters, rate=rate)
    lower = max(spot - strike * np.exp(-rate * expiry), 0.0)
    assert lower <= call <= spot


def test_zero_maturity_returns_intrinsic():
    parameters = equity_like()
    assert heston_price(110.0, 100.0, 0.0, parameters) == pytest.approx(10.0)
    assert heston_price(90.0, 100.0, 0.0, parameters, option_type=OptionType.PUT) == pytest.approx(
        10.0
    )


def test_the_integration_has_converged_at_the_default_node_count():
    """Doubling the nodes must not move the price."""
    parameters = equity_like()
    coarse = heston_price(100.0, 100.0, 2.0, parameters, n_nodes=256)
    fine = heston_price(100.0, 100.0, 2.0, parameters, n_nodes=1024)
    assert coarse == pytest.approx(fine, abs=1e-8)


# --------------------------------------------------------------------------- #
# The model's economics
# --------------------------------------------------------------------------- #
def test_correlation_generates_skew_with_the_right_sign():
    r"""Negative rho must make downside strikes dearer.

    This is what makes Heston a *model* rather than a fit: the skew is produced
    by a correlation parameter, not described by a shape parameter.
    """

    def skew(rho: float) -> float:
        parameters = equity_like(rho=rho)
        low = heston_implied_volatility(100.0, 85.0, 1.0, parameters)
        high = heston_implied_volatility(100.0, 115.0, 1.0, parameters)
        return high - low

    assert skew(-0.8) < -0.03, "equity-like correlation must give a downward skew"
    assert abs(skew(0.0)) < 0.01, "zero correlation gives a near-symmetric smile"
    assert skew(0.8) > 0.03


def test_vol_of_vol_generates_smile_curvature():
    """xi controls curvature: with xi -> 0 the smile is flat."""

    def curvature(xi: float) -> float:
        parameters = equity_like(xi=xi, rho=0.0)
        wing = heston_implied_volatility(100.0, 80.0, 1.0, parameters)
        centre = heston_implied_volatility(100.0, 100.0, 1.0, parameters)
        return wing - centre

    assert curvature(1e-4) == pytest.approx(0.0, abs=0.002)
    assert curvature(0.8) > 0.01


def test_implied_volatility_starts_near_the_spot_variance():
    """A short-dated at-the-money option prices at roughly sqrt(v0)."""
    parameters = equity_like(v0=0.09, theta=0.09)
    assert heston_implied_volatility(100.0, 100.0, 1 / 52, parameters) == pytest.approx(
        0.30, abs=0.02
    )


# --------------------------------------------------------------------------- #
# THE BRANCH CUT
# --------------------------------------------------------------------------- #
def test_the_original_formulation_diverges_at_longer_maturities():
    r"""Heston's 1993 form is algebraically identical and numerically wrong.

    The two differ only in whether :math:`g` or :math:`1/g` appears. The stable
    form keeps :math:`|g| \le 1` so the principal-branch logarithm never
    approaches its cut; the original does not, and the integrand jumps.

    Measured on kappa=8, xi=1.0, rho=-0.8, an at-the-money call:

    =====  ==========  ============  =========
    T      stable      original      error
    =====  ==========  ============  =========
    1.0    12.401720   14.291873     +15%
    2.0    18.068739   26.969893     +49%
    5.0    29.581290   57.232172     +93%
    10.0   42.414594   NaN           overflow
    =====  ==========  ============  =========

    Nothing raises for T <= 5. The integral converges to a wrong number, which is
    why this is worth a test rather than a comment.
    """
    parameters = HestonParameters(kappa=8.0, theta=0.09, xi=1.0, rho=-0.8, v0=0.09)

    with np.errstate(over="ignore", invalid="ignore"):
        short_stable = heston_price(100.0, 100.0, 0.25, parameters, formulation="stable")
        short_original = heston_price(100.0, 100.0, 0.25, parameters, formulation="original")
        long_stable = heston_price(100.0, 100.0, 5.0, parameters, formulation="stable")
        long_original = heston_price(100.0, 100.0, 5.0, parameters, formulation="original")

    # Short dated: the integrand never nears the cut, so both agree.
    assert short_stable == pytest.approx(short_original, rel=0.02)

    # Long dated: the original is wrong by a factor, silently.
    assert long_original > 1.5 * long_stable
    assert np.isfinite(long_stable), "the stable form stays finite"


def test_the_original_integrand_is_discontinuous_and_the_stable_one_is_not():
    """The mechanism itself, measured rather than asserted.

    Sampling the integrand densely in u, the largest step between neighbouring
    points is ~0.004 for the stable form and ~5.2 for the original at T=20 --
    three orders of magnitude, which is a jump, not curvature.
    """
    parameters = HestonParameters(kappa=8.0, theta=0.09, xi=1.0, rho=-0.8, v0=0.09)
    u = np.linspace(0.01, 30.0, 40_000)

    def largest_step(formulation: str) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            phi = characteristic_function(
                u.astype(complex) - 0.5j, parameters, 20.0, formulation=formulation
            )
        integrand = np.real(phi) / (u**2 + 0.25)
        finite = integrand[np.isfinite(integrand)]
        return float(np.max(np.abs(np.diff(finite))))

    stable = largest_step("stable")
    original = largest_step("original")

    assert stable < 0.05, "the stable integrand is smooth"
    assert original > 20 * stable, "the original jumps across the branch cut"


def test_the_stable_form_survives_long_maturities_that_overflow_the_original():
    """exp(+d*tau) overflows; exp(-d*tau) decays. Same sign flip, second benefit."""
    parameters = HestonParameters(kappa=8.0, theta=0.09, xi=1.0, rho=-0.8, v0=0.09)

    for expiry in (10.0, 20.0, 30.0):
        price = heston_price(100.0, 100.0, expiry, parameters, formulation="stable")
        assert np.isfinite(price)
        assert 0.0 < price < 100.0

        with np.errstate(over="ignore", invalid="ignore"):
            broken = heston_price(100.0, 100.0, expiry, parameters, formulation="original")
        assert not np.isfinite(broken) or broken > 1.5 * price


def test_the_stable_form_is_the_default():
    """Because the other one is a trap, and defaults are what people use."""
    parameters = HestonParameters(kappa=8.0, theta=0.09, xi=1.0, rho=-0.8, v0=0.09)
    assert heston_price(100.0, 100.0, 5.0, parameters) == pytest.approx(
        heston_price(100.0, 100.0, 5.0, parameters, formulation="stable")
    )
