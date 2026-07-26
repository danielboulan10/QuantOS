"""Validation for the option-chain pipeline and the volatility surface.

Every test here checks a *recovery*: construct data with a known answer, run it
through the pipeline, and require the answer back. Tests that merely check the
code runs would pass just as happily on a surface fitted to the wrong forward.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from quantos.data.options import (
    ChainFilter,
    OptionQuote,
    build_chain,
    implied_forward,
)
from quantos.derivatives.black_scholes import OptionType, black_scholes_price
from quantos.research.vol_surface import (
    SVIParameters,
    fit_surface,
    fit_svi,
    model_free_implied_variance,
    variance_risk_premium,
)

AS_OF = date(2025, 1, 2)


def make_quotes(
    *,
    expiry: date,
    spot: float = 100.0,
    vol: float | dict[float, float] = 0.20,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    strikes: np.ndarray | None = None,
    spread_fraction: float = 0.01,
) -> list[OptionQuote]:
    """Build a chain from Black-Scholes, optionally with a strike-dependent smile."""
    time_to_expiry = (expiry - AS_OF).days / 365.25
    if strikes is None:
        strikes = np.arange(70.0, 131.0, 2.5)

    quotes: list[OptionQuote] = []
    for strike in strikes:
        sigma = vol if isinstance(vol, float) else vol[float(strike)]
        for option_type in (OptionType.CALL, OptionType.PUT):
            fair = float(
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
            if fair <= 0.01:
                continue
            quotes.append(
                OptionQuote(
                    expiry=expiry,
                    strike=float(strike),
                    option_type=option_type,
                    bid=fair * (1 - spread_fraction),
                    ask=fair * (1 + spread_fraction),
                )
            )
    return quotes


# --------------------------------------------------------------------------- #
# The chain: does it recover what was put in?
# --------------------------------------------------------------------------- #
def test_flat_volatility_chain_recovers_flat_implied_volatility():
    """The most basic recovery. If this fails nothing downstream means anything."""
    expiry = AS_OF + timedelta(days=180)
    chain = build_chain(
        make_quotes(expiry=expiry, vol=0.25),
        symbol="FLAT",
        as_of=AS_OF,
        spot=100.0,
    )
    assert len(chain) > 15
    np.testing.assert_allclose(chain.implied_vols, 0.25, atol=2e-3)


def test_forward_is_recovered_from_put_call_parity_not_assumed():
    """A 3% dividend yield must be discovered from prices, not supplied.

    This is the test that catches the most damaging silent error in the module:
    using spot as the forward. With q=3% over a year the forward sits ~3 below
    spot, and using spot instead would tilt the whole smile and read as skew.
    """
    expiry = AS_OF + timedelta(days=365)
    time_to_expiry = 365 / 365.25
    quotes = make_quotes(expiry=expiry, vol=0.20, rate=0.05, dividend_yield=0.03)

    recovered, how = implied_forward(quotes, spot=100.0, time_to_expiry=time_to_expiry, rate=0.05)
    expected = 100.0 * np.exp((0.05 - 0.03) * time_to_expiry)

    assert "parity" in how
    assert recovered == pytest.approx(expected, rel=1e-6)

    # And the wrong answer -- spot, or the no-dividend forward -- is far enough
    # away that the test would catch it.
    assert abs(recovered - 100.0) > 1.5
    assert abs(recovered - 100.0 * np.exp(0.05 * time_to_expiry)) > 2.5


def test_forward_falls_back_and_says_so_when_no_parity_pair_exists():
    expiry = AS_OF + timedelta(days=90)
    calls_only = [q for q in make_quotes(expiry=expiry) if q.option_type is OptionType.CALL]
    _, how = implied_forward(calls_only, spot=100.0, time_to_expiry=0.25, rate=0.02)
    assert "no parity pair" in how


def test_in_the_money_options_are_excluded():
    expiry = AS_OF + timedelta(days=180)
    chain = build_chain(make_quotes(expiry=expiry), symbol="X", as_of=AS_OF, spot=100.0)
    forward = float(chain.forwards[0])
    for strike, is_call in zip(chain.strikes, chain.is_call, strict=True):
        assert (is_call and strike >= forward) or (not is_call and strike <= forward)


def test_zero_bid_contracts_are_dropped():
    expiry = AS_OF + timedelta(days=180)
    quotes = make_quotes(expiry=expiry)
    quotes.append(OptionQuote(expiry, 200.0, OptionType.CALL, bid=0.0, ask=0.05))
    chain = build_chain(quotes, symbol="X", as_of=AS_OF, spot=100.0)
    assert 200.0 not in set(chain.strikes.tolist())
    assert chain.rejections.get("zero or missing bid", 0) >= 1


def test_wide_spreads_are_dropped():
    expiry = AS_OF + timedelta(days=180)
    quotes = make_quotes(expiry=expiry)
    quotes.append(OptionQuote(expiry, 125.0, OptionType.CALL, bid=0.20, ask=3.00))
    chain = build_chain(quotes, symbol="X", as_of=AS_OF, spot=100.0)
    assert any("spread wider" in reason for reason in chain.rejections)


def test_every_rejection_is_counted():
    """The filters are strict, so the accounting must be complete."""
    expiry = AS_OF + timedelta(days=180)
    quotes = make_quotes(expiry=expiry)
    chain = build_chain(quotes, symbol="X", as_of=AS_OF, spot=100.0)
    assert len(chain) + sum(chain.rejections.values()) == len(quotes)


def test_near_expiry_contracts_are_dropped():
    soon = AS_OF + timedelta(days=3)
    chain = build_chain(
        make_quotes(expiry=soon),
        symbol="X",
        as_of=AS_OF,
        spot=100.0,
        chain_filter=ChainFilter(min_days_to_expiry=7),
    )
    assert len(chain) == 0
    assert any("within 7 days" in reason for reason in chain.rejections)


# --------------------------------------------------------------------------- #
# SVI
# --------------------------------------------------------------------------- #
def test_svi_recovers_its_own_parameters():
    """Round-trip: generate from known SVI, fit, and compare the curves.

    The *curves* are compared rather than the parameters, because SVI is not
    identified pointwise -- different (a, b, m) combinations produce nearly the
    same smile. Comparing parameters would fail for a fit that is perfect.
    """
    truth = SVIParameters(a=0.015, b=0.12, rho=-0.45, m=0.02, s=0.09, time_to_expiry=1.0)
    k = np.linspace(-0.4, 0.4, 25)
    vols = truth.implied_volatility(k)

    fitted, converged, rmse = fit_svi(k, vols, 1.0)
    assert converged
    assert rmse < 0.05  # volatility points
    np.testing.assert_allclose(fitted.implied_volatility(k), vols, atol=1e-3)


def test_svi_recovers_a_known_skew_from_a_realistic_equity_smile():
    """A smile with a known at-the-money slope must return that slope.

    The smile is built smooth and linear in log-moneyness,
    :math:`\\sigma(k) = 0.22 - 0.15k + 0.10k^2`, for a reason: a smile defined
    with ``max(0, K_0 - K)`` has a *kink* at the money, and the derivative there
    is genuinely undefined -- a finite difference across it returns whichever
    branch it lands on rather than a skew. Real smiles are smooth, and a test
    should not demand that the code reproduce an artefact of the test data.
    """
    expiry = AS_OF + timedelta(days=180)
    strikes = np.arange(70.0, 131.0, 2.5)
    forward = 100.0  # rate and dividend are zero, so the forward is spot
    true_skew, curvature = -0.15, 0.10
    smile = {
        float(k): 0.22 + true_skew * np.log(k / forward) + curvature * np.log(k / forward) ** 2
        for k in strikes
    }

    chain = build_chain(
        make_quotes(expiry=expiry, vol=smile, strikes=strikes),
        symbol="SKEW",
        as_of=AS_OF,
        spot=100.0,
    )
    surface = fit_surface(chain)
    assert len(surface.smiles) == 1
    smile_fit = surface.smiles[0]

    assert smile_fit.converged
    assert smile_fit.rmse_vol_points < 0.5
    assert smile_fit.parameters.atm_volatility == pytest.approx(0.22, abs=0.01)
    assert smile_fit.parameters.skew == pytest.approx(true_skew, abs=0.03)


def test_svi_refuses_to_pretend_with_too_few_points():
    k = np.array([-0.1, 0.0, 0.1])
    vols = np.array([0.22, 0.20, 0.21])
    _, converged, _ = fit_svi(k, vols, 0.5)
    assert converged is False


def test_butterfly_violation_is_detected():
    """A deliberately arbitrageable parameter set must be flagged.

    Large b with rho near -1 produces a curve steep enough that the implied
    density goes negative. If the check cannot detect this, it cannot detect
    anything.
    """
    bad = SVIParameters(a=0.001, b=0.9, rho=-0.99, m=0.0, s=0.01, time_to_expiry=1.0)
    passes, worst = bad.density_is_nonnegative()
    assert passes is False
    assert worst < 0

    good = SVIParameters(a=0.04, b=0.10, rho=-0.3, m=0.0, s=0.20, time_to_expiry=1.0)
    passes_good, worst_good = good.density_is_nonnegative()
    assert passes_good is True
    assert worst_good >= -1e-8


def test_calendar_arbitrage_is_detected():
    """Total variance that falls with maturity must be reported."""
    expiries = [AS_OF + timedelta(days=90), AS_OF + timedelta(days=270)]
    # The far expiry is quoted at a much LOWER volatility -- so much lower that
    # its total variance falls below the near expiry's. That is arbitrageable.
    quotes = make_quotes(expiry=expiries[0], vol=0.40) + make_quotes(expiry=expiries[1], vol=0.15)
    chain = build_chain(quotes, symbol="CAL", as_of=AS_OF, spot=100.0)
    surface = fit_surface(chain)

    assert len(surface.smiles) == 2
    assert surface.calendar_arbitrage_free is False
    assert surface.calendar_violations


def test_a_sane_term_structure_is_not_flagged():
    """Guards the test above: the check must not fire on normal surfaces."""
    quotes = []
    for days, vol in ((60, 0.18), (150, 0.20), (300, 0.22)):
        quotes += make_quotes(expiry=AS_OF + timedelta(days=days), vol=vol)
    surface = fit_surface(build_chain(quotes, symbol="OK", as_of=AS_OF, spot=100.0))

    assert len(surface.smiles) == 3
    assert surface.calendar_arbitrage_free, surface.calendar_violations
    times, vols = surface.term_structure()
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(vols, [0.18, 0.20, 0.22], atol=0.01)


# --------------------------------------------------------------------------- #
# Model-free implied variance and the variance risk premium
# --------------------------------------------------------------------------- #
def test_model_free_variance_recovers_a_flat_black_scholes_volatility():
    """The VIX integral must return sigma for a flat smile."""
    forward, time_to_expiry, sigma = 100.0, 1.0, 0.25
    strikes = np.arange(30.0, 221.0, 1.0)
    is_call = strikes >= forward
    prices = np.array(
        [
            float(
                black_scholes_price(
                    forward,
                    k,
                    time_to_expiry,
                    sigma,
                    option_type=OptionType.CALL if c else OptionType.PUT,
                )
            )
            for k, c in zip(strikes, is_call, strict=True)
        ]
    )

    variance = model_free_implied_variance(strikes, prices, is_call, forward, time_to_expiry)
    assert np.sqrt(variance) == pytest.approx(sigma, abs=0.003)


def test_model_free_variance_is_higher_for_a_skewed_smile():
    """Model-free variance exceeds at-the-money implied volatility under skew.

    This is the whole reason the VIX is not simply an at-the-money quote: the
    fair variance strike integrates the entire smile, and a negatively skewed
    smile puts extra weight on expensive downside strikes. A model-free estimate
    that came back equal to the ATM volatility would mean the integral is
    ignoring the wings.
    """
    forward, time_to_expiry, atm = 100.0, 1.0, 0.20
    strikes = np.arange(40.0, 181.0, 1.0)
    is_call = strikes >= forward
    vols = atm + 0.30 * np.maximum(0.0, (forward - strikes) / forward)
    prices = np.array(
        [
            float(
                black_scholes_price(
                    forward,
                    k,
                    time_to_expiry,
                    v,
                    option_type=OptionType.CALL if c else OptionType.PUT,
                )
            )
            for k, v, c in zip(strikes, vols, is_call, strict=True)
        ]
    )

    variance = model_free_implied_variance(strikes, prices, is_call, forward, time_to_expiry)
    assert np.sqrt(variance) > atm + 0.01


def test_model_free_variance_refuses_a_grid_that_does_not_straddle_the_forward():
    strikes = np.arange(120.0, 161.0, 5.0)
    prices = np.full(strikes.size, 1.0)
    result = model_free_implied_variance(strikes, prices, strikes > 0, 100.0, 1.0)
    assert np.isnan(result)


def test_variance_risk_premium_recovers_an_injected_premium():
    """The central measurement, validated against a known answer.

    Options are priced at 25% volatility; the underlying then realises 15%. The
    measured premium must come back at about ten volatility points. Getting the
    sign backwards here -- realised minus implied -- would be invisible in casual
    use and would invert the conclusion of every study built on it.
    """
    expiry = AS_OF + timedelta(days=90)
    chain = build_chain(
        make_quotes(expiry=expiry, vol=0.25, strikes=np.arange(60.0, 141.0, 2.5)),
        symbol="VRP",
        as_of=AS_OF,
        spot=100.0,
    )

    rng = np.random.default_rng(4)
    realised_sigma = 0.15
    daily = rng.normal(0.0, realised_sigma / np.sqrt(252), 63)
    # Rescale so the sample realises exactly 15%, isolating the estimator from
    # sampling noise -- the property under test is the arithmetic, not the draw.
    daily *= realised_sigma / np.sqrt(np.sum(daily**2) / daily.size * 252)

    premium = variance_risk_premium(chain, daily)

    assert premium.implied_volatility == pytest.approx(0.25, abs=0.01)
    assert premium.realised_volatility == pytest.approx(0.15, abs=0.005)
    assert premium.premium_vol_points == pytest.approx(10.0, abs=1.2)
    assert premium.ratio == pytest.approx(0.25 / 0.15, rel=0.06)
    assert "expensive" in premium.interpretation


def test_variance_risk_premium_reports_a_negative_premium_when_realised_exceeds_implied():
    """The minority outcome must also be reported correctly."""
    expiry = AS_OF + timedelta(days=90)
    chain = build_chain(
        make_quotes(expiry=expiry, vol=0.15, strikes=np.arange(60.0, 141.0, 2.5)),
        symbol="VRP",
        as_of=AS_OF,
        spot=100.0,
    )
    rng = np.random.default_rng(5)
    daily = rng.normal(0.0, 0.35 / np.sqrt(252), 63)
    daily *= 0.35 / np.sqrt(np.sum(daily**2) / daily.size * 252)

    premium = variance_risk_premium(chain, daily)
    assert premium.premium_vol_points < -10
    assert "cheap" in premium.interpretation


def test_variance_risk_premium_warns_on_a_thin_realised_sample():
    expiry = AS_OF + timedelta(days=90)
    chain = build_chain(
        make_quotes(expiry=expiry, vol=0.25, strikes=np.arange(60.0, 141.0, 2.5)),
        symbol="VRP",
        as_of=AS_OF,
        spot=100.0,
    )
    premium = variance_risk_premium(chain, np.full(8, 0.01))
    assert any("standard error" in note for note in premium.notes)
