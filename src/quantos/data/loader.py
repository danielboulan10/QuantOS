"""Load price data from CSV files, and align multiple series onto common dates.

Why a CSV loader exists
-----------------------
FRED covers indices, rates, credit and macro, but not individual equities or
ETFs. Rather than add a dependency to reach them (which ``docs/ddr/DDR-002``
forbids), QuantOS reads whatever CSV you already have: a Yahoo Finance download,
a broker export, a vendor extract, a spreadsheet you saved.

:func:`load_ohlcv_csv` sniffs the column names, so the common layouts work
without configuration:

* Yahoo Finance: ``Date,Open,High,Low,Close,Adj Close,Volume``
* Stooq: ``Date,Open,High,Low,Close,Volume``
* Most brokers: some subset of the above, in any case, in any order

**It defaults to the adjusted close where one is present.** Using the raw close
for a dividend-paying stock or ETF injects a fake negative return on every
ex-dividend date, which quietly biases every volatility and drawdown statistic
downstream. That is the single most common data error in retail backtests, and
the loader tells you which column it chose so the decision is visible.

Alignment
---------
:func:`align` joins several series on their common dates. This matters more than
it sounds: two series with different holiday calendars (a US index and a UK one,
or a daily series and a monthly one) will silently mis-pair if you just zip them,
and a correlation computed from mis-paired dates is meaningless.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["PriceSeries", "align", "load_ohlcv_csv", "to_returns"]

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%Y%m%d")
_CLOSE_PREFERENCE = ("adj close", "adjusted close", "adj_close", "close", "price", "value")


@dataclass(frozen=True)
class PriceSeries:
    """A dated price series loaded from a file."""

    symbol: str
    dates: NDArray[np.datetime64]
    prices: NDArray[np.float64]
    #: Which CSV column the prices came from -- recorded because the choice matters.
    price_column: str = "close"
    volume: NDArray[np.float64] | None = None
    source: str = "csv"
    detail: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.prices.size)

    @property
    def start(self) -> str:
        return str(self.dates[0])[:10] if len(self) else ""

    @property
    def end(self) -> str:
        return str(self.dates[-1])[:10] if len(self) else ""

    def log_returns(self) -> NDArray[np.float64]:
        r"""Log returns :math:`\log(P_t/P_{t-1})`."""
        if len(self) < 2:
            return np.zeros(0)
        if np.any(self.prices <= 0):
            raise ValueError(f"{self.symbol} contains non-positive prices")
        return np.diff(np.log(self.prices))

    def since(self, start: str) -> PriceSeries:
        """Restrict to dates on or after ``start``."""
        mask = self.dates >= np.datetime64(start)
        return PriceSeries(
            symbol=self.symbol,
            dates=self.dates[mask],
            prices=self.prices[mask],
            price_column=self.price_column,
            volume=self.volume[mask] if self.volume is not None else None,
            source=self.source,
            detail=dict(self.detail),
        )

    def __repr__(self) -> str:  # pragma: no cover - display
        return f"PriceSeries({self.symbol!r}, n={len(self)}, {self.start}..{self.end})"


def _parse_date(text: str) -> np.datetime64 | None:
    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return np.datetime64(datetime.strptime(text, fmt).date())
        except ValueError:
            continue
    return None


def load_ohlcv_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
    price_column: str | None = None,
    date_column: str | None = None,
) -> PriceSeries:
    r"""Load a price series from a CSV, sniffing the common layouts.

    Purpose
        Get any stock, ETF or futures series into QuantOS without a data
        dependency. Download it however you like; this reads it.
    Inputs
        ``path`` -- the CSV file. ``price_column`` -- override the automatic
        choice. ``date_column`` -- override the date column.
    Outputs
        :class:`PriceSeries`. Inspect ``price_column`` to see which column was
        used; if it is ``"close"`` rather than an adjusted variant, dividends are
        not accounted for.
    Failure modes
        :class:`ValueError` if no date column or no numeric price column can be
        identified, listing the headers found so the problem is obvious.

    Example
        >>> import tempfile, pathlib
        >>> csv_text = (
        ...     "Date,Open,Close,Adj Close\n"
        ...     "2024-01-02,10,11,10.5\n2024-01-03,11,12,11.5\n"
        ... )
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "x.csv"
        >>> _ = p.write_text(csv_text)
        >>> series = load_ohlcv_csv(p, symbol="TEST")
        >>> series.price_column, len(series)
        ('adj close', 2)
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"no such file: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path} has no data rows")

    header = [c.strip().lower() for c in rows[0]]

    # Locate the date column.
    if date_column is not None:
        if date_column.lower() not in header:
            raise ValueError(f"date column {date_column!r} not in {header}")
        date_index = header.index(date_column.lower())
    else:
        date_index = next(
            (i for i, c in enumerate(header) if c in {"date", "time", "timestamp", "datetime"}),
            -1,
        )
        if date_index < 0:
            # Fall back to the first column that parses as a date.
            for i in range(len(header)):
                if (
                    len(rows) > 1
                    and _parse_date(rows[1][i] if i < len(rows[1]) else "") is not None
                ):
                    date_index = i
                    break
        if date_index < 0:
            raise ValueError(f"no date column found in {header}")

    # Locate the price column, preferring an adjusted close.
    if price_column is not None:
        if price_column.lower() not in header:
            raise ValueError(f"price column {price_column!r} not in {header}")
        price_index = header.index(price_column.lower())
    else:
        price_index = -1
        for candidate in _CLOSE_PREFERENCE:
            if candidate in header:
                price_index = header.index(candidate)
                break
        if price_index < 0:
            raise ValueError(
                f"no recognisable price column in {header}. Expected one of "
                f"{_CLOSE_PREFERENCE}, or pass price_column= explicitly."
            )

    volume_index = header.index("volume") if "volume" in header else -1

    dates: list[np.datetime64] = []
    prices: list[float] = []
    volumes: list[float] = []
    skipped = 0
    for row in rows[1:]:
        if len(row) <= max(date_index, price_index):
            skipped += 1
            continue
        stamp = _parse_date(row[date_index])
        if stamp is None:
            skipped += 1
            continue
        try:
            value = float(row[price_index].replace(",", "").strip())
        except ValueError:
            skipped += 1
            continue
        if not np.isfinite(value):
            skipped += 1
            continue
        dates.append(stamp)
        prices.append(value)
        if volume_index >= 0 and volume_index < len(row):
            try:
                volumes.append(float(row[volume_index].replace(",", "").strip() or 0.0))
            except ValueError:
                volumes.append(0.0)

    if not prices:
        raise ValueError(f"no usable rows parsed from {path}")

    order = np.argsort(np.array(dates, dtype="datetime64[D]"))
    date_array = np.array(dates, dtype="datetime64[D]")[order]
    price_array = np.asarray(prices, dtype=np.float64)[order]
    volume_array = (
        np.asarray(volumes, dtype=np.float64)[order] if len(volumes) == len(prices) else None
    )

    chosen = header[price_index]
    return PriceSeries(
        symbol=symbol or path.stem.upper(),
        dates=date_array,
        prices=price_array,
        price_column=chosen,
        volume=volume_array,
        source=str(path),
        detail={
            "rows_skipped": str(skipped),
            "dividend_adjusted": str(chosen.startswith("adj")),
            "columns_found": ",".join(header),
        },
    )


def align(
    series: dict[str, tuple[NDArray[np.datetime64], NDArray[np.float64]]],
) -> tuple[NDArray[np.datetime64], dict[str, NDArray[np.float64]]]:
    """Restrict several dated series to the dates they all share.

    Purpose
        Make cross-series statistics meaningful. Two series with different
        holiday calendars, or a daily series against a monthly one, will
        mis-pair if simply zipped -- and a correlation from mis-paired dates is
        not a correlation of anything.
    Inputs
        ``series`` -- name to ``(dates, values)``, each sorted ascending.
    Outputs
        ``(common_dates, {name: values_on_common_dates})``.
    Failure modes
        Returns empty arrays if the intersection is empty, which is a real
        answer: a daily equity index and a quarterly GDP series genuinely share
        few dates, and silently resampling would hide that.

    Example
        >>> import numpy as np
        >>> d1 = np.array(['2024-01-01','2024-01-02','2024-01-03'], dtype='datetime64[D]')
        >>> d2 = np.array(['2024-01-02','2024-01-03','2024-01-04'], dtype='datetime64[D]')
        >>> dates, out = align({'a': (d1, np.array([1.,2.,3.])),
        ...                     'b': (d2, np.array([9.,8.,7.]))})
        >>> len(dates), out['a'].tolist(), out['b'].tolist()
        (2, [2.0, 3.0], [9.0, 8.0])
    """
    if not series:
        return np.array([], dtype="datetime64[D]"), {}

    common: NDArray[np.datetime64] | None = None
    for dates, _ in series.values():
        common = dates if common is None else np.intersect1d(common, dates)
    assert common is not None

    out: dict[str, NDArray[np.float64]] = {}
    for name, (dates, values) in series.items():
        index = np.searchsorted(dates, common)
        index = np.clip(index, 0, dates.size - 1)
        out[name] = values[index]
    return common, out


def to_returns(prices: NDArray[np.float64], *, method: str = "log") -> NDArray[np.float64]:
    """Convert a price level to returns.

    ``"log"`` returns are additive across time, which is what every statistic in
    this package assumes. ``"simple"`` returns are additive across *assets*,
    which is what portfolio weighting assumes. The distinction matters at high
    volatility, where the two diverge materially.
    """
    prices = np.asarray(prices, dtype=np.float64).ravel()
    if prices.size < 2:
        return np.zeros(0)
    if method == "log":
        if np.any(prices <= 0):
            raise ValueError("log returns require strictly positive prices")
        return np.diff(np.log(prices))
    if method == "simple":
        return np.diff(prices) / prices[:-1]
    raise ValueError(f"unknown method {method!r}; use 'log' or 'simple'")
