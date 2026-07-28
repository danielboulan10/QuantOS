"""Validation for the options market maker.

The tests that matter are the sign conventions. A maker that skews the wrong way
accumulates risk faster the more it holds, and a P&L attribution with a flipped
sign tells a desk to widen when it should hedge. Neither failure raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.derivatives.black_scholes import black_scholes_greeks
from quantos.derivatives.market_making import (
    GreekInventory,
    PnLAttribution,
    SurfaceMarketMaker,
)
from quantos.research.vol_surface import SVIParameters


def flat_surface(volatility: float = 0.20, time_to_expiry: float = 0.25) -> SVIParameters:
    """A surface with no smile, so tests isolate the maker's own behaviour."""
    return SVIParameters(
        a=volatility**2 * time_to_expiry,
        b=0.0,
        rho=0.0,
        m=0.0,
        s=0.1,
        time_to_expiry=time_to_expiry,
    )


def maker(**kwargs) -> SurfaceMarketMaker:
    return SurfaceMarketMaker(surface=flat_surface(), **kwargs)


# --------------------------------------------------------------------------- #
# Quoting
# --------------------------------------------------------------------------- #
def test_quotes_straddle_fair_value():
    market = maker().quote(100.0, (95.0, 100.0, 105.0), 0.25)
    assert len(market.quotes) == 3
    for quote in market.quotes:
        assert quote.bid < quote.fair_value < quote.ask
        assert quote.bid_volatility < quote.fair_volatility < quote.ask_volatility


def test_a_flat_book_quotes_symmetrically_in_volatility():
    """With no inventory there is no reason to prefer a side."""
    market = maker().quote(100.0, (100.0,), 0.25)
    assert market.inventory_skew_volatility == pytest.approx(0.0, abs=1e-12)
    assert market.quotes[0].volatility_skew == pytest.approx(0.0, abs=1e-12)


def test_a_volatility_symmetric_quote_is_not_price_symmetric():
    """Option price is convex in volatility, so the price midpoint is not fair.

    An earlier version of this test asserted price symmetry and failed by 1.2e-05
    with a completely flat book. The maker was right and the test was wrong: a
    desk quotes in volatility, and volga makes the price midpoint drift from fair
    by a second-order term in the width. It is small, it never vanishes, and it
    is why `volatility_skew` rather than `price_skew` measures inventory lean.
    """
    quote = maker().quote(100.0, (100.0,), 0.25).quotes[0]
    assert quote.volatility_skew == pytest.approx(0.0, abs=1e-12)
    assert quote.price_skew != pytest.approx(0.0, abs=1e-9)
    assert abs(quote.price_skew) < 1e-3, "but the effect must stay second order"

    # Halving the width must shrink the residual by roughly four times.
    narrow = SurfaceMarketMaker(surface=flat_surface(), base_half_spread=0.005)
    ratio = abs(quote.price_skew) / abs(narrow.quote(100.0, (100.0,), 0.25).quotes[0].price_skew)
    assert 2.0 < ratio < 8.0


def test_the_surface_smile_reaches_the_quotes():
    """A skewed surface must produce skewed quotes, or it is not being used."""
    skewed = SVIParameters(a=0.01, b=0.10, rho=-0.7, m=0.0, s=0.10, time_to_expiry=0.25)
    market = SurfaceMarketMaker(surface=skewed).quote(100.0, (90.0, 100.0, 110.0), 0.25)
    vols = [q.fair_volatility for q in market.quotes]
    assert vols[0] > vols[-1], "a negatively skewed surface must price downside higher"


# --------------------------------------------------------------------------- #
# Inventory skew -- the sign that matters
# --------------------------------------------------------------------------- #
def test_long_vega_marks_the_book_down():
    """Long vega must LOWER quoted volatility, so the next trade is a sale.

    The opposite sign produces a maker that bids more aggressively the more vega
    it already owns, which accumulates without bound instead of mean-reverting.
    """
    long_vega = maker()
    long_vega.inventory.vega = 1000.0
    short_vega = maker()
    short_vega.inventory.vega = -1000.0

    long_market = long_vega.quote(100.0, (100.0,), 0.25)
    short_market = short_vega.quote(100.0, (100.0,), 0.25)

    assert long_market.inventory_skew_volatility < 0
    assert short_market.inventory_skew_volatility > 0
    assert long_market.quotes[0].volatility_skew < short_market.quotes[0].volatility_skew


def test_skew_scales_with_inventory():
    small, large = maker(), maker()
    small.inventory.vega = 200.0
    large.inventory.vega = 1600.0
    assert abs(large.quote(100.0, (100.0,), 0.25).inventory_skew_volatility) > abs(
        small.quote(100.0, (100.0,), 0.25).inventory_skew_volatility
    )


def test_risk_aversion_controls_how_hard_the_book_leans():
    timid = SurfaceMarketMaker(surface=flat_surface(), risk_aversion=2.0)
    bold = SurfaceMarketMaker(surface=flat_surface(), risk_aversion=0.1)
    timid.inventory.vega = bold.inventory.vega = 1000.0
    assert abs(timid.quote(100.0, (100.0,), 0.25).inventory_skew_volatility) > abs(
        bold.quote(100.0, (100.0,), 0.25).inventory_skew_volatility
    )


def test_spread_widens_as_the_book_fills():
    empty, full = maker(), maker()
    full.inventory.vega = 1500.0
    assert (
        full.quote(100.0, (100.0,), 0.25).half_spread_volatility
        > empty.quote(100.0, (100.0,), 0.25).half_spread_volatility
    )


def test_hitting_a_limit_is_flagged_rather_than_ignored():
    at_limit = maker()
    at_limit.inventory.vega = 2500.0  # over the 2000 default
    market = at_limit.quote(100.0, (100.0,), 0.25)
    assert market.at_limit
    assert any("utilisation" in note for note in market.notes)


def test_utilisation_takes_the_worst_greek_not_the_average():
    """A book at its vega limit is constrained even with flat delta."""
    inventory = GreekInventory(delta=0.0, gamma=0.0, vega=2000.0)
    limits = {"delta": 500.0, "gamma": 50.0, "vega": 2000.0}
    assert inventory.utilisation(limits) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Inventory accounting
# --------------------------------------------------------------------------- #
def test_a_trade_moves_every_greek_together():
    m = maker()
    market = m.quote(100.0, (100.0,), 0.25)
    m.on_trade(market.quotes[0], 10.0, 100.0, side="buy")

    expected = black_scholes_greeks(100.0, 100.0, 0.25, market.quotes[0].fair_volatility)
    assert m.inventory.delta == pytest.approx(10 * expected.delta)
    assert m.inventory.vega == pytest.approx(10 * expected.vega)
    assert m.inventory.gamma == pytest.approx(10 * expected.gamma)


def test_buying_and_selling_the_same_size_returns_to_flat():
    m = maker()
    market = m.quote(100.0, (100.0,), 0.25)
    m.on_trade(market.quotes[0], 5.0, 100.0, side="buy")
    m.on_trade(market.quotes[0], 5.0, 100.0, side="sell")
    assert m.inventory.is_flat
    # And the round trip earned the spread twice.
    assert m.inventory.cash > 0


def test_delta_hedging_flattens_delta_and_records_its_cost():
    m = maker()
    market = m.quote(100.0, (100.0,), 0.25)
    m.on_trade(market.quotes[0], 20.0, 100.0, side="buy")
    assert abs(m.inventory.delta) > 1

    shares = m.hedge_delta(100.0, cost_per_share=0.01)
    assert m.inventory.delta == pytest.approx(0.0)
    assert shares < 0  # long calls means short shares to hedge
    assert m.attribution.hedge_cost < 0, "hedging is not free"


# --------------------------------------------------------------------------- #
# P&L attribution -- the point of the module
# --------------------------------------------------------------------------- #
def test_spread_capture_is_positive_on_both_sides():
    """Trading at your own quote beats trading at mid, whichever way you go."""
    for side in ("buy", "sell"):
        m = maker()
        market = m.quote(100.0, (100.0,), 0.25)
        m.on_trade(market.quotes[0], 10.0, 100.0, side=side)
        assert m.attribution.spread_capture > 0


def test_informed_flow_shows_up_as_negative_adverse_selection():
    """The maker sells a call and spot promptly rises. That must be recorded."""
    m = maker()
    market = m.quote(100.0, (100.0,), 0.25)
    # Someone lifts the offer, then spot moves up: classic adverse selection.
    m.on_trade(market.quotes[0], 10.0, 100.0, side="sell", subsequent_spot=104.0)
    assert m.attribution.adverse_selection < 0


def test_uninformed_flow_does_not_systematically_cost_the_maker():
    """Guards the test above: random flow must not read as adverse selection."""
    rng = np.random.default_rng(0)
    m = maker()
    for _ in range(200):
        market = m.quote(100.0, (100.0,), 0.25)
        side = "buy" if rng.random() < 0.5 else "sell"
        m.on_trade(
            market.quotes[0],
            1.0,
            100.0,
            side=side,
            subsequent_spot=100.0 * np.exp(rng.normal(0, 0.01)),
        )
        m.inventory = GreekInventory()  # flatten between trades to isolate the effect

    # With symmetric flow and symmetric moves the term should be small relative
    # to what the spread earned.
    assert abs(m.attribution.adverse_selection) < 0.5 * m.attribution.spread_capture


def test_gamma_and_theta_have_opposite_signs_for_a_short_book():
    """Short options earn theta and pay gamma. Reporting one nets out the story."""
    m = maker()
    market = m.quote(100.0, (100.0,), 0.25)
    m.on_trade(market.quotes[0], 50.0, 100.0, side="sell")  # short calls

    assert m.inventory.gamma < 0
    assert m.inventory.theta > 0

    rng = np.random.default_rng(1)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    m.accrue_carry(path, time_step=1 / 252)

    assert m.attribution.gamma_pnl < 0, "a short gamma book pays for movement"
    assert m.attribution.theta_pnl > 0, "and is paid for the passage of time"


def test_a_still_market_leaves_theta_but_no_gamma():
    m = maker()
    market = m.quote(100.0, (100.0,), 0.25)
    m.on_trade(market.quotes[0], 50.0, 100.0, side="sell")
    m.accrue_carry(np.full(60, 100.0), time_step=1 / 252)

    assert m.attribution.gamma_pnl == pytest.approx(0.0)
    assert m.attribution.theta_pnl > 0


def test_the_components_sum_to_the_total():
    attribution = PnLAttribution(
        spread_capture=100.0,
        gamma_pnl=-30.0,
        theta_pnl=40.0,
        adverse_selection=-50.0,
        hedge_cost=-5.0,
        n_trades=10,
        volume=100.0,
    )
    assert attribution.total == pytest.approx(55.0)


def test_edge_per_contract_nets_spread_against_adverse_selection():
    """The number that decides whether the business works."""
    winning = PnLAttribution(spread_capture=100.0, adverse_selection=-40.0, volume=100.0)
    losing = PnLAttribution(spread_capture=100.0, adverse_selection=-160.0, volume=100.0)

    assert winning.edge_per_contract == pytest.approx(0.6)
    assert losing.edge_per_contract == pytest.approx(-0.6)
    assert "covers adverse selection" in winning.verdict
    assert "too tight" in losing.verdict


def test_attribution_reports_no_trades_rather_than_dividing_by_zero():
    assert PnLAttribution().verdict == "no trades"
    assert np.isnan(PnLAttribution().edge_per_contract)


# --------------------------------------------------------------------------- #
# Mark to market
# --------------------------------------------------------------------------- #
def test_marking_to_market_reconciles_with_cash_and_position():
    m = maker()
    market = m.quote(100.0, (100.0,), 0.25)
    m.on_trade(market.quotes[0], 10.0, 100.0, side="buy")

    # Bought below fair, so marking at fair shows an immediate gain.
    assert m.mark_to_market(100.0) > 0


def test_an_empty_book_marks_to_its_cash():
    m = maker()
    m.inventory.cash = 123.45
    assert m.mark_to_market(100.0) == pytest.approx(123.45)
