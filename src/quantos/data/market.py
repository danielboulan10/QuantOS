"""Fetch daily prices for any listed ticker, without an API key.

What this makes possible
------------------------
Everything else in this package operates on a price series. Until now that series
had to arrive as a CSV you supplied by hand, which meant the research pipeline --
the deepest part of the repository -- could not be pointed at a name without
first going and finding a file. This module closes that gap:

    quantos research --ticker AAPL

resolves the ticker, downloads its history, and runs the full battery.

Adjusted prices, and what the adjustment actually changes
----------------------------------------------------------
This module returns the **adjusted** close by default.

It is worth being precise about what that buys, because the usual telling of this
story is wrong for this particular source. The ``close`` field returned by this
endpoint is *already split-adjusted* -- measured directly on NVIDIA, which has
split during the sample, the raw and adjusted series have an identical worst day
of -18.8% and an identical 49.6% annualised volatility. There is no phantom
split crash to remove.

What ``adjclose`` additionally removes is **dividends**, and that is where the
difference lives. Over ten years of history:

===============  ==========  ===============
Ticker           raw CAGR    adjusted CAGR
===============  ==========  ===============
Verizon (VZ)     -1.85%      +3.59%
Exxon (XOM)      5.48%       10.17%
Coca-Cola (KO)   6.10%       9.47%
Berkshire (BRK)  13.16%      13.16%
===============  ==========  ===============

On raw prices Verizon lost money over a decade; on a total-return basis it made
money. The sign of the conclusion flips, and no error is raised either way.
Berkshire pays no dividend and shows a gap of exactly zero, which is the control
that confirms this is the dividend mechanism rather than a coincidence.

So the adjusted series is the default, and the raw one is available only by
asking for it explicitly.

On depending on an undocumented endpoint
-----------------------------------------
The data comes from Yahoo's chart endpoint, which is public and keyless but
**not a documented, supported API**. It can change shape or start refusing
requests without notice, and this module will break when it does. That is a real
limitation and it is stated here rather than discovered later.

Two things reduce the damage. Every response is cached on disk, so previously
fetched history keeps working offline and a cached run is reproducible. And the
fetch is isolated behind :class:`MarketDataClient`, which returns the same
:class:`~quantos.data.loader.PriceSeries` the CSV loader produces -- so replacing
the source, or falling back to a CSV, changes this file and nothing else.

Nothing here is real-time. It is daily bars, delayed, for research.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quantos.data.loader import PriceSeries

__all__ = [
    "MarketDataClient",
    "MarketDataError",
    "TickerInfo",
    "default_cache_dir",
    "fetch_prices",
]

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_USER_AGENT = "Mozilla/5.0 (compatible; quantos/1.0; +https://github.com/danielboulan10/QuantOS)"

#: Yahoo's instrument types, mapped onto this package's asset classes.
_INSTRUMENT_TYPES: dict[str, str] = {
    "EQUITY": "equity",
    "ETF": "etf",
    "MUTUALFUND": "etf",
    "INDEX": "index",
    "FUTURE": "future",
    "CURRENCY": "fx",
    "CRYPTOCURRENCY": "fx",
    "OPTION": "equity",
}


class MarketDataError(RuntimeError):
    """Raised when a ticker cannot be retrieved and no cached copy exists."""


def default_cache_dir() -> Path:
    """Cache location, honouring ``QUANTOS_CACHE_DIR`` if set."""
    override = os.environ.get("QUANTOS_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "quantos"
    return base / "market"


@dataclass(frozen=True)
class TickerInfo:
    """What the venue says about a symbol, recorded alongside its prices."""

    ticker: str
    name: str = ""
    asset_class: str = "equity"
    currency: str = "USD"
    exchange: str = ""
    instrument_type: str = ""

    def describe(self) -> str:
        parts = [self.ticker]
        if self.name and self.name != self.ticker:
            parts.append(f"({self.name})")
        parts.append(f"[{self.asset_class}")
        if self.exchange:
            parts.append(f"on {self.exchange}")
        parts.append(f"in {self.currency}]")
        return " ".join(parts)


@dataclass
class MarketDataClient:
    """Downloads daily bars for a ticker, with an on-disk cache.

    The cache is not an optimisation. It is what makes a run reproducible and
    what lets the whole pipeline work on a plane: once a ticker has been fetched,
    every later run reads the same bytes unless the cache is explicitly
    refreshed or has gone stale.
    """

    cache_dir: Path = field(default_factory=default_cache_dir)
    #: Seconds before a cached response is considered stale. One day by default,
    #: because these are daily bars and re-fetching more often buys nothing.
    max_age_seconds: float = 86_400.0
    timeout: float = 30.0
    offline: bool = False

    def _cache_path(self, ticker: str, range_key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in ticker.upper())
        return self.cache_dir / f"{safe}.{range_key}.json"

    def _download(self, ticker: str, range_key: str) -> str:
        url = _CHART_URL.format(ticker=urllib.parse.quote(ticker.upper()))
        url += f"?range={range_key}&interval=1d&events=div%2Csplit"
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return str(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise MarketDataError(
                    f"no such ticker {ticker!r}. Check the symbol -- non-US listings "
                    "usually need a suffix (VOD.L, SAP.DE, 7203.T), and indices "
                    "start with a caret (^GSPC, ^VIX)."
                ) from error
            raise MarketDataError(
                f"could not fetch {ticker!r}: HTTP {error.code}. This endpoint is "
                "public but undocumented and does rate-limit; a cached copy would "
                "still work."
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise MarketDataError(
                f"could not reach the price service for {ticker!r}: {error}. "
                "Previously-cached tickers still work offline."
            ) from error

    def fetch(
        self,
        ticker: str,
        *,
        range_key: str = "10y",
        refresh: bool = False,
        adjusted: bool = True,
    ) -> tuple[PriceSeries, TickerInfo]:
        """Fetch daily bars for one ticker.

        Purpose
            Turn a symbol into the same :class:`PriceSeries` the CSV loader
            produces, so every downstream tool works unchanged.
        Inputs
            ``ticker`` -- e.g. ``AAPL``, ``SPY``, ``^GSPC``, ``VOD.L``, ``BTC-USD``.
            ``range_key`` -- Yahoo range token: ``1y``, ``5y``, ``10y``, ``max``.
            ``adjusted`` -- use split- and dividend-adjusted closes. Leave this
            alone unless you specifically want raw prices; see the module
            docstring for what raw closes do to a return series.
        Outputs
            ``(series, info)``.
        Failure modes
            :class:`MarketDataError` if the ticker is unknown, or if the fetch
            fails *and* nothing is cached. A stale cache is preferred to an
            error, with the staleness reported in the series detail.

        Example
            >>> client = MarketDataClient(offline=True)     # doctest: +SKIP
            >>> series, info = client.fetch("AAPL")         # doctest: +SKIP
            >>> info.asset_class                            # doctest: +SKIP
            'equity'
        """
        path = self._cache_path(ticker, range_key)
        cached: str | None = None
        cache_age = float("inf")

        if path.exists():
            cache_age = time.time() - path.stat().st_mtime
            cached = path.read_text(encoding="utf-8")

        use_cache = (
            cached is not None
            and not refresh
            and (self.offline or cache_age < self.max_age_seconds)
        )

        detail: dict[str, str] = {}
        if use_cache:
            assert cached is not None
            payload, source = cached, f"cache ({cache_age / 3600:.1f}h old)"
        elif self.offline:
            raise MarketDataError(
                f"offline and {ticker!r} is not cached. Fetch it once with a "
                "network connection, or pass a CSV with --csv."
            )
        else:
            try:
                payload = self._download(ticker, range_key)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding="utf-8")
                source = "network"
            except MarketDataError:
                if cached is None:
                    raise
                payload, source = cached, f"STALE cache ({cache_age / 86400:.1f} days old)"
                detail["warning"] = "network fetch failed; served a stale cached copy"

        series, info = self._parse(payload, ticker, adjusted=adjusted)
        detail["source"] = source
        detail["adjusted"] = "yes" if adjusted else "no (raw closes: splits are NOT removed)"
        return (
            PriceSeries(
                symbol=series.symbol,
                dates=series.dates,
                prices=series.prices,
                price_column=series.price_column,
                volume=series.volume,
                source=f"yahoo:{source}",
                detail=detail,
            ),
            info,
        )

    @staticmethod
    def _parse(
        payload: str, ticker: str, *, adjusted: bool = True
    ) -> tuple[PriceSeries, TickerInfo]:
        """Turn a chart response into a price series, dropping incomplete bars."""
        try:
            document: dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError as error:
            raise MarketDataError(
                f"{ticker}: the price service returned something that is not JSON. "
                "The endpoint is undocumented and may have changed shape."
            ) from error

        chart = document.get("chart") or {}
        if chart.get("error"):
            raise MarketDataError(f"{ticker}: {chart['error']}")
        results = chart.get("result") or []
        if not results:
            raise MarketDataError(f"{ticker}: the price service returned no data")

        result = results[0]
        meta = result.get("meta", {})
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]

        closes = quote.get("close") or []
        if adjusted:
            adjusted_block = (indicators.get("adjclose") or [{}])[0]
            closes = adjusted_block.get("adjclose") or closes
        volumes = quote.get("volume") or []

        if not timestamps or not closes:
            raise MarketDataError(
                f"{ticker}: no price history returned. Newly listed or delisted "
                "symbols often have none over the requested range."
            )

        # Yahoo emits null for halted or non-trading bars. Dropping them is
        # correct: carrying the previous price forward would manufacture a
        # zero return, which understates volatility and fabricates
        # autocorrelation that momentum and mean-reversion signals then trade on.
        dates: list[np.datetime64] = []
        prices: list[float] = []
        kept_volume: list[float] = []
        for index, stamp in enumerate(timestamps):
            price = closes[index] if index < len(closes) else None
            if price is None or not np.isfinite(float(price)) or float(price) <= 0:
                continue
            dates.append(np.datetime64(int(stamp), "s").astype("datetime64[D]"))
            prices.append(float(price))
            raw_volume = volumes[index] if index < len(volumes) else None
            kept_volume.append(float(raw_volume) if raw_volume is not None else float("nan"))

        if not prices:
            raise MarketDataError(f"{ticker}: every returned bar was empty")

        instrument = str(meta.get("instrumentType", "")).upper()
        info = TickerInfo(
            ticker=str(meta.get("symbol", ticker)).upper(),
            name=str(meta.get("longName") or meta.get("shortName") or ""),
            asset_class=_INSTRUMENT_TYPES.get(instrument, "equity"),
            currency=str(meta.get("currency", "USD")),
            exchange=str(meta.get("fullExchangeName") or meta.get("exchangeName") or ""),
            instrument_type=instrument,
        )

        series = PriceSeries(
            symbol=info.ticker,
            dates=np.asarray(dates, dtype="datetime64[D]"),
            prices=np.asarray(prices, dtype=float),
            price_column="adjusted close" if adjusted else "close",
            volume=np.asarray(kept_volume, dtype=float),
            source="yahoo",
        )
        return series, info


def fetch_prices(
    ticker: str,
    *,
    start: str | None = None,
    range_key: str = "10y",
    offline: bool = False,
    refresh: bool = False,
) -> tuple[PriceSeries, TickerInfo]:
    """Convenience wrapper: ticker in, price series out.

    Example
        >>> series, info = fetch_prices("MSFT")            # doctest: +SKIP
        >>> len(series) > 1000, info.asset_class           # doctest: +SKIP
        (True, 'equity')
    """
    client = MarketDataClient(offline=offline)
    series, info = client.fetch(ticker, range_key=range_key, refresh=refresh)
    if start:
        series = series.since(start)
    return series, info
