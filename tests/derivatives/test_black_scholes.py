"""Black-Scholes: analytic Greeks against finite differences, and identities."""

from __future__ import annotations

import numpy as np
import pytest

from quantos.derivatives.black_scholes import (
    OptionType,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    put_call_parity_check,
)

BASE = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_expiry": 1.0,
    "volatility": 0.2,
    "rate": 0.05,
    "dividend_yield": 0.02,
}


def price(**overrides: float) -> float:
    kwargs = {**BASE, **overrides}
    option_type = kwargs.pop("option_type", OptionType.CALL)
    return float(
        black_scholes_price(
            kwargs["spot"],
            kwargs["strike"],
            kwargs["time_to_expiry"],
            kwargs["volatility"],
            rate=kwargs["rate"],
            dividend_yield=kwargs["dividend_yield"],
            option_type=option_type,
        )
    )


def test_price_matches_published_reference() -> None:
    """Hull's textbook example: S=K=100, T=1, sigma=0.2, r=0.05, q=0."""
    assert float(black_scholes_price(100, 100, 1.0, 0.2, rate=0.05)) == pytest.approx(
        10.450583572185565, rel=1e-12
    )
    assert float(
        black_scholes_price(100, 100, 1.0, 0.2, rate=0.05, option_type=OptionType.PUT)
    ) == pytest.approx(5.573526022256971, rel=1e-12)


@pytest.mark.parametrize(
    "greek,perturb,step",
    [
        ("delta", "spot", 1e-5),
        ("gamma", "spot", 1e-2),  # second difference needs a larger step
        ("vega", "volatility", 1e-5),
        ("rho", "rate", 1e-5),
        ("epsilon", "dividend_yield", 1e-5),
    ],
)
def test_greek_matches_central_difference(greek: str, perturb: str, step: float) -> None:
    analytic = black_scholes_greeks(**BASE).as_dict()[greek]
    base = BASE[perturb]
    if greek == "gamma":
        numeric = (
            price(**{perturb: base + step}) - 2 * price() + price(**{perturb: base - step})
        ) / step**2
    else:
        numeric = (price(**{perturb: base + step}) - price(**{perturb: base - step})) / (2 * step)
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_theta_and_dual_delta_signs_and_magnitudes() -> None:
    greeks = black_scholes_greeks(**BASE)
    step = 1e-5
    # Theta is -dPrice/dT.
    numeric_theta = -(price(time_to_expiry=1 + step) - price(time_to_expiry=1 - step)) / (2 * step)
    assert greeks.theta == pytest.approx(numeric_theta, rel=1e-6)
    # dual_delta = -dPrice/dStrike, and must be POSITIVE for a call: it is the
    # discounted in-the-money probability. A sign error here was a real bug.
    numeric_dual = -(price(strike=100 + step) - price(strike=100 - step)) / (2 * step)
    assert greeks.dual_delta == pytest.approx(numeric_dual, rel=1e-6)
    assert greeks.dual_delta > 0.0


@pytest.mark.parametrize("greek,perturb", [("vanna", "spot"), ("volga", "volatility")])
def test_second_order_greeks_match_a_vega_difference(greek: str, perturb: str) -> None:
    step = 1e-5
    base = BASE[perturb]
    up = black_scholes_greeks(**{**BASE, perturb: base + step}).vega
    down = black_scholes_greeks(**{**BASE, perturb: base - step}).vega
    assert black_scholes_greeks(**BASE).as_dict()[greek] == pytest.approx(
        (up - down) / (2 * step), rel=1e-6
    )


def test_charm_matches_a_delta_difference() -> None:
    step = 1e-5
    up = black_scholes_greeks(**{**BASE, "time_to_expiry": 1 + step}).delta
    down = black_scholes_greeks(**{**BASE, "time_to_expiry": 1 - step}).delta
    assert black_scholes_greeks(**BASE).charm == pytest.approx(-(up - down) / (2 * step), rel=1e-6)


def test_put_call_parity_holds_across_the_surface() -> None:
    """Model-free identity: a non-zero residual means an arbitrage or bad data."""
    for strike in (60.0, 90.0, 100.0, 130.0, 200.0):
        for maturity in (0.05, 0.5, 2.0, 10.0):
            call = price(strike=strike, time_to_expiry=maturity)
            put = price(strike=strike, time_to_expiry=maturity, option_type=OptionType.PUT)
            residual = put_call_parity_check(
                call,
                put,
                BASE["spot"],
                strike,
                maturity,
                rate=BASE["rate"],
                dividend_yield=BASE["dividend_yield"],
            )
            assert abs(residual) < 1e-11


def test_delta_parity() -> None:
    r"""``delta_call - delta_put == exp(-qT)``."""
    call = black_scholes_greeks(**BASE)
    put = black_scholes_greeks(**{**BASE, "option_type": OptionType.PUT})
    assert call.delta - put.delta == pytest.approx(
        np.exp(-BASE["dividend_yield"] * BASE["time_to_expiry"]), rel=1e-12
    )
    # Gamma and vega are identical for calls and puts.
    assert call.gamma == pytest.approx(put.gamma, rel=1e-14)
    assert call.vega == pytest.approx(put.vega, rel=1e-14)


def test_degenerate_cases_return_limits_not_nan() -> None:
    # At expiry the price is intrinsic and the Greeks take their limits.
    assert float(black_scholes_price(100, 90, 0.0, 0.2)) == 10.0
    assert float(black_scholes_price(100, 110, 0.0, 0.2)) == 0.0
    at_expiry = black_scholes_greeks(100, 90, 0.0, 0.2)
    assert at_expiry.gamma == 0.0
    assert at_expiry.vega == 0.0
    assert at_expiry.delta == pytest.approx(1.0)
    # Zero volatility: deterministic forward payoff.
    zero_vol = black_scholes_greeks(100, 90, 1.0, 0.0, rate=0.05)
    assert zero_vol.vega == 0.0
    assert np.isfinite(zero_vol.price)


def test_price_is_monotone_in_volatility_and_bounded() -> None:
    vols = np.linspace(0.01, 3.0, 200)
    prices = black_scholes_price(100, 100, 1.0, vols, rate=0.05)
    assert np.all(np.diff(prices) > 0)
    assert np.all(prices < 100.0)  # cannot exceed the spot
    assert np.all(prices > 100 - 100 * np.exp(-0.05))  # above discounted intrinsic


@pytest.mark.parametrize(
    "spot,strike,maturity,vol",
    [
        (100, 100, 1.0, 0.23),
        (100, 250, 0.05, 0.60),  # deep OTM: vega ~ 1e-9, naive Newton diverges
        (100, 60, 3.0, 0.12),
        (100, 100, 1e-4, 0.35),  # nearly expired
        (100, 140, 0.25, 0.90),
        (100, 100, 10.0, 0.05),  # long-dated, low vol
        (100, 180, 1.0, 0.45),
    ],
)
def test_implied_volatility_round_trip(
    spot: float, strike: float, maturity: float, vol: float
) -> None:
    observed = float(black_scholes_price(spot, strike, maturity, vol, rate=0.05))
    assert implied_volatility(observed, spot, strike, maturity, rate=0.05) == pytest.approx(
        vol, abs=1e-10
    )


def test_implied_volatility_works_for_puts() -> None:
    observed = float(
        black_scholes_price(100, 120, 0.5, 0.31, rate=0.03, option_type=OptionType.PUT)
    )
    assert implied_volatility(
        observed, 100, 120, 0.5, rate=0.03, option_type=OptionType.PUT
    ) == pytest.approx(0.31, abs=1e-10)


def test_implied_volatility_rejects_arbitrage_violations() -> None:
    with pytest.raises(ValueError, match="upper bound"):
        implied_volatility(200.0, 100, 100, 1.0)
    with pytest.raises(ValueError, match="lower bound"):
        implied_volatility(-1.0, 100, 100, 1.0)
    with pytest.raises(ValueError, match="expired"):
        implied_volatility(5.0, 100, 100, 0.0)


def test_implied_volatility_refuses_when_time_value_is_unidentifiable() -> None:
    """S=100, K=20, T=2 at 15% vol: price equals intrinsic to the last bit.

    An unguarded solver returns 0.0 here -- a plausible number that is wrong by
    the entire volatility. Refusing is the only honest behaviour.
    """
    observed = float(black_scholes_price(100, 20, 2.0, 0.15, rate=0.05))
    with pytest.raises(ValueError, match="not identifiable"):
        implied_volatility(observed, 100, 20, 2.0, rate=0.05)


def test_black76_via_zero_carry() -> None:
    """Setting q = r prices a future, so no separate implementation is needed."""
    forward_style = float(black_scholes_price(100, 100, 1.0, 0.2, rate=0.05, dividend_yield=0.05))
    undiscounted = float(black_scholes_price(100, 100, 1.0, 0.2, rate=0.0))
    assert forward_style == pytest.approx(undiscounted * np.exp(-0.05), rel=1e-12)


def test_vectorises_over_every_argument() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    maturities = np.array([[0.25], [1.0]])
    out = black_scholes_price(100.0, strikes, maturities, 0.2, rate=0.05)
    assert out.shape == (2, 3)
    assert np.all(np.diff(out, axis=1) < 0)  # calls fall as strike rises
    assert np.all(np.diff(out, axis=0) > 0)  # and rise with maturity
