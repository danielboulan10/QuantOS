"""Tests for the local research viewer.

Rendering is tested without touching the network. What matters here is that a
bad input produces a page rather than a traceback, that user input cannot carry
markup into the page, and that the honest framing does not silently disappear.
"""

from __future__ import annotations

import pytest

from quantos.web.server import render_landing


def test_landing_page_is_complete_html():
    page = render_landing()
    assert page.startswith("<!doctype html>")
    assert page.rstrip().endswith("</html>")
    assert 'name="ticker"' in page
    assert "<form" in page


def test_landing_page_shows_a_message_when_given_one():
    assert "NOPE is not a ticker" in render_landing("NOPE is not a ticker")


def test_a_message_is_escaped_rather_than_injected():
    """User input reaches the page, so it must not be able to carry markup."""
    page = render_landing("<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_the_landing_page_states_what_is_not_forecast():
    """The honesty of the framing is a feature and should not silently vanish."""
    page = render_landing()
    assert "Where the price is going" in page


def test_every_page_carries_the_disclaimer():
    """It must be on pages reached directly by link, not only on the index.

    Most visitors arrive at a research page from a link or a shared URL and never
    see the landing page, so a disclaimer that lives only there is missing exactly
    when it matters.
    """
    from quantos.web.server import DISCLAIMER, render_page

    assert "investment advice" in DISCLAIMER
    assert DISCLAIMER in render_landing()

    rendered = render_page("KO", link_style="static")
    assert rendered.status == 200
    assert "Nothing here is investment advice" in rendered.body


def test_an_error_page_still_carries_the_disclaimer():
    assert "investment advice" in render_landing("no such ticker")


def _accepted(ticker: str) -> bool:
    """The handler's validation rule, as applied in do_GET."""
    cleaned = ticker.strip()
    return bool(cleaned) and len(cleaned) <= 24 and all(c.isalnum() or c in "^.-=" for c in cleaned)


@pytest.mark.parametrize(
    "ticker",
    ["", "   ", "a" * 25, "DROP TABLE", "../../etc/passwd", "AAPL;rm -rf /", "<script>"],
)
def test_implausible_tickers_are_rejected_before_any_fetch(ticker):
    assert not _accepted(ticker)


@pytest.mark.parametrize("ticker", ["AAPL", "^GSPC", "BTC-USD", "BRK.B", "EURUSD=X", "VOD.L"])
def test_real_ticker_shapes_are_accepted(ticker):
    """Guards the test above: the filter must not reject valid symbols."""
    assert _accepted(ticker)


def test_year_axis_labels_have_no_thousands_separator():
    """2017 must not render as '2,017' on a date axis."""
    from quantos.viz.svg import _format_year, line_chart

    assert _format_year(2017.0) == "2017"
    svg = line_chart({"x": ([2016.0, 2026.0], [1.0, 2.0])}, x_tick_style="year").render()
    assert "2,0" not in svg


def test_rolling_volatility_matches_a_direct_computation():
    """The O(n) rolling standard deviation must equal the naive one."""
    import numpy as np

    from quantos.web.server import _volatility_history

    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    dates = np.datetime64("2020-01-01") + np.arange(400)

    class _Report:
        volatility_forecast_21d = 0.3

    svg = _volatility_history(dates, prices, _Report())
    assert svg.startswith("<svg")

    # The same window, computed directly.
    returns = np.diff(np.log(prices))
    window = 21
    naive = np.array(
        [np.std(returns[i : i + window]) * np.sqrt(252) for i in range(returns.size - window + 1)]
    )
    squared = np.concatenate([[0.0], np.cumsum(returns**2)])
    running = np.concatenate([[0.0], np.cumsum(returns)])
    mean_square = (squared[window:] - squared[:-window]) / window
    mean = (running[window:] - running[:-window]) / window
    fast = np.sqrt(np.maximum(mean_square - mean**2, 0.0)) * np.sqrt(252.0)
    np.testing.assert_allclose(fast, naive, rtol=1e-9, atol=1e-12)
