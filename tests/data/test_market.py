"""Tests for the keyless ticker price client.

**No test here touches the network.** The client is exercised against recorded
response payloads and a temporary cache directory, because a test suite that
depends on an external undocumented service fails for reasons that have nothing
to do with the code -- and would fail in CI on the day the vendor rate-limits a
build machine.

What is tested is the part that can be wrong in a way nobody notices: the
adjusted-versus-raw close, dropped bars, asset-class detection, and the cache and
offline behaviour.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from quantos.data.market import (
    MarketDataClient,
    MarketDataError,
    TickerInfo,
    fetch_prices,
)

DAY = 86_400
BASE = 1_700_000_000  # a Tuesday, arbitrary


def payload(
    *,
    symbol: str = "TEST",
    instrument: str = "EQUITY",
    closes: list[float | None] | None = None,
    adjusted: list[float | None] | None = None,
    volumes: list[float | None] | None = None,
    name: str = "Test Corp",
    currency: str = "USD",
) -> str:
    """A minimal response in the shape the chart endpoint returns."""
    closes = [100.0, 101.0, 102.0, 103.0] if closes is None else closes
    n = len(closes)
    document = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": symbol,
                        "longName": name,
                        "instrumentType": instrument,
                        "currency": currency,
                        "fullExchangeName": "NasdaqGS",
                    },
                    "timestamp": [BASE + i * DAY for i in range(n)],
                    "indicators": {
                        "quote": [
                            {
                                "close": closes,
                                "volume": volumes if volumes is not None else [1e6] * n,
                            }
                        ],
                        **({"adjclose": [{"adjclose": adjusted}]} if adjusted else {}),
                    },
                }
            ],
        }
    }
    return json.dumps(document)


# --------------------------------------------------------------------------- #
# Adjusted prices: the failure that produces plausible output rather than an error
# --------------------------------------------------------------------------- #
def test_adjusted_close_is_used_by_default():
    """A split in the raw series must not appear in the returned prices.

    The raw column here contains a 4:1 split -- a -75% day that never happened to
    a holder. Using it would put a fabricated crash into every downstream
    statistic, and nothing would raise.
    """
    raw = [400.0, 404.0, 101.0, 102.0]  # split between bar 2 and 3
    adj = [100.0, 101.0, 101.0, 102.0]  # what a holder actually experienced

    series, _ = MarketDataClient._parse(payload(closes=raw, adjusted=adj), "TEST")

    np.testing.assert_allclose(series.prices, adj)
    assert series.price_column == "adjusted close"

    returns = np.diff(np.log(series.prices))
    assert np.min(returns) > -0.05, "no fabricated crash should survive adjustment"


def test_raw_close_is_available_but_must_be_asked_for():
    raw = [400.0, 404.0, 101.0, 102.0]
    adj = [100.0, 101.0, 101.0, 102.0]

    series, _ = MarketDataClient._parse(payload(closes=raw, adjusted=adj), "TEST", adjusted=False)

    np.testing.assert_allclose(series.prices, raw)
    assert "close" in series.price_column
    # And it contains exactly the artefact the default avoids.
    assert np.min(np.diff(np.log(series.prices))) < -1.0


def test_missing_adjusted_block_falls_back_to_close():
    series, _ = MarketDataClient._parse(payload(closes=[10.0, 11.0, 12.0, 13.0]), "TEST")
    np.testing.assert_allclose(series.prices, [10.0, 11.0, 12.0, 13.0])


# --------------------------------------------------------------------------- #
# Incomplete bars
# --------------------------------------------------------------------------- #
def test_null_bars_are_dropped_not_carried_forward():
    """Halted sessions come back as null and must not become zero returns.

    Carrying the previous price forward manufactures a zero-return day, which
    understates volatility and fabricates autocorrelation -- which the momentum
    and mean-reversion signals would then happily trade on.
    """
    series, _ = MarketDataClient._parse(payload(closes=[100.0, None, 102.0, None, 104.0]), "TEST")

    assert len(series) == 3
    np.testing.assert_allclose(series.prices, [100.0, 102.0, 104.0])
    assert np.all(np.diff(np.log(series.prices)) > 0), "no zero-return day was invented"


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_non_positive_prices_are_dropped(bad):
    series, _ = MarketDataClient._parse(payload(closes=[100.0, bad, 102.0, 103.0]), "TEST")
    assert len(series) == 3
    assert np.all(series.prices > 0)


def test_dates_are_calendar_days_in_order():
    series, _ = MarketDataClient._parse(payload(), "TEST")
    assert series.dates.dtype == np.dtype("datetime64[D]")
    assert np.all(np.diff(series.dates.astype("datetime64[D]").astype(int)) > 0)


def test_a_response_with_no_usable_bars_raises():
    with pytest.raises(MarketDataError, match="every returned bar was empty"):
        MarketDataClient._parse(payload(closes=[None, None]), "TEST")


# --------------------------------------------------------------------------- #
# Asset class detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("instrument", "expected"),
    [
        ("EQUITY", "equity"),
        ("ETF", "etf"),
        ("MUTUALFUND", "etf"),
        ("INDEX", "index"),
        ("FUTURE", "future"),
        ("CRYPTOCURRENCY", "fx"),
        ("CURRENCY", "fx"),
        ("SOMETHING_NEW", "equity"),  # unknown types must not crash
    ],
)
def test_asset_class_comes_from_the_venue(instrument, expected):
    """The report gates its own analyses on asset class, so this must be right."""
    _, info = MarketDataClient._parse(payload(instrument=instrument), "TEST")
    assert info.asset_class == expected


def test_ticker_info_records_what_the_venue_said():
    _, info = MarketDataClient._parse(
        payload(symbol="VOD.L", name="Vodafone Group", currency="GBp"), "vod.l"
    )
    assert info.ticker == "VOD.L"
    assert info.name == "Vodafone Group"
    assert info.currency == "GBp"
    assert "VOD.L" in info.describe()
    assert "GBp" in info.describe()


def test_describe_does_not_repeat_the_ticker_as_a_name():
    assert TickerInfo(ticker="AAA", name="AAA").describe().count("AAA") == 1


# --------------------------------------------------------------------------- #
# Malformed responses
# --------------------------------------------------------------------------- #
def test_non_json_response_is_reported_as_a_shape_change():
    with pytest.raises(MarketDataError, match="not JSON"):
        MarketDataClient._parse("<html>bot check</html>", "TEST")


def test_error_field_in_the_response_is_surfaced():
    body = json.dumps({"chart": {"error": {"code": "Not Found"}, "result": None}})
    with pytest.raises(MarketDataError, match="Not Found"):
        MarketDataClient._parse(body, "TEST")


def test_empty_result_list_is_reported():
    with pytest.raises(MarketDataError, match="no data"):
        MarketDataClient._parse(json.dumps({"chart": {"result": []}}), "TEST")


def test_missing_history_names_the_likely_cause():
    body = json.dumps(
        {"chart": {"result": [{"meta": {"symbol": "X"}, "timestamp": [], "indicators": {}}]}}
    )
    with pytest.raises(MarketDataError, match="no price history"):
        MarketDataClient._parse(body, "X")


# --------------------------------------------------------------------------- #
# Cache and offline behaviour -- still no network
# --------------------------------------------------------------------------- #
def test_a_cached_ticker_works_offline(tmp_path):
    client = MarketDataClient(cache_dir=tmp_path, offline=True)
    client._cache_path("TEST", "10y").write_text(payload(symbol="TEST"))

    series, info = client.fetch("TEST")
    assert len(series) == 4
    assert info.ticker == "TEST"
    assert "cache" in series.source


def test_offline_without_a_cache_says_what_to_do(tmp_path):
    client = MarketDataClient(cache_dir=tmp_path, offline=True)
    with pytest.raises(MarketDataError, match=r"offline and .* is not cached"):
        client.fetch("NEVERFETCHED")


def test_the_cache_is_keyed_by_ticker_and_range(tmp_path):
    client = MarketDataClient(cache_dir=tmp_path, offline=True)
    assert client._cache_path("AAPL", "10y") != client._cache_path("AAPL", "1y")
    assert client._cache_path("aapl", "10y") == client._cache_path("AAPL", "10y")


def test_cache_filenames_survive_awkward_tickers(tmp_path):
    """Index and FX tickers contain characters that are not path-safe."""
    client = MarketDataClient(cache_dir=tmp_path, offline=True)
    for ticker in ("^GSPC", "BTC-USD", "BRK.B", "EURUSD=X"):
        path = client._cache_path(ticker, "10y")
        path.write_text(payload(symbol=ticker))
        assert path.exists()
        assert "/" not in path.name


def test_a_stale_cache_is_preferred_to_an_error(tmp_path, monkeypatch):
    """Losing the network must degrade to old data, not to a failure."""
    client = MarketDataClient(cache_dir=tmp_path, max_age_seconds=0.0)
    cache = client._cache_path("TEST", "10y")
    cache.write_text(payload(symbol="TEST"))
    # Make the cache clearly stale.
    old = time.time() - 30 * DAY
    import os

    os.utime(cache, (old, old))

    def refuse(self, ticker, range_key):
        raise MarketDataError("network down")

    monkeypatch.setattr(MarketDataClient, "_download", refuse)

    series, _ = client.fetch("TEST")
    assert len(series) == 4
    assert "STALE" in series.source
    assert "stale" in series.detail.get("warning", "")


def test_a_fresh_cache_is_not_refetched(tmp_path, monkeypatch):
    client = MarketDataClient(cache_dir=tmp_path, max_age_seconds=1e9)
    client._cache_path("TEST", "10y").write_text(payload(symbol="TEST"))

    def explode(self, ticker, range_key):
        raise AssertionError("should not have hit the network")

    monkeypatch.setattr(MarketDataClient, "_download", explode)
    assert len(client.fetch("TEST")[0]) == 4


def test_refresh_bypasses_a_fresh_cache(tmp_path, monkeypatch):
    client = MarketDataClient(cache_dir=tmp_path, max_age_seconds=1e9)
    client._cache_path("TEST", "10y").write_text(payload(symbol="TEST"))

    calls: list[str] = []

    def record(self, ticker, range_key):
        calls.append(ticker)
        return payload(symbol="TEST", closes=[1.0, 2.0, 3.0, 4.0, 5.0])

    monkeypatch.setattr(MarketDataClient, "_download", record)
    series, _ = client.fetch("TEST", refresh=True)
    assert calls == ["TEST"]
    assert len(series) == 5


def test_the_detail_records_whether_prices_were_adjusted(tmp_path):
    client = MarketDataClient(cache_dir=tmp_path, offline=True)
    client._cache_path("TEST", "10y").write_text(payload(symbol="TEST"))

    adjusted, _ = client.fetch("TEST")
    raw, _ = client.fetch("TEST", adjusted=False)
    assert adjusted.detail["adjusted"] == "yes"
    assert "NOT removed" in raw.detail["adjusted"]


def test_fetch_prices_applies_the_start_date(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTOS_CACHE_DIR", str(tmp_path))
    client = MarketDataClient(cache_dir=tmp_path / "market", offline=True)
    client.cache_dir.mkdir(parents=True, exist_ok=True)
    client._cache_path("TEST", "10y").write_text(payload(symbol="TEST"))

    everything, _ = fetch_prices("TEST", offline=True)
    assert len(everything) == 4

    cutoff = str(everything.dates[2])
    trimmed, _ = fetch_prices("TEST", start=cutoff, offline=True)
    assert len(trimmed) == 2
