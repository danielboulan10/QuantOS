"""A named catalogue of tradeable and macroeconomic series.

Purpose
-------
Nobody should have to memorise that the high-yield credit spread is
``BAMLH0A0HYM2``. This module gives every series a plain name, records whether it
is a *price level* or a *rate*, and groups them into bundles you can analyse in
one command:

    quantos analyse --bundle risk-appetite

The level/rate distinction is not cosmetic. Log returns are meaningful for an
index and meaningless for a yield, which can be zero or negative, and confusing
the two is one of the easiest ways to produce a plausible-looking but wrong
volatility number. :class:`Series` records which is which, and
:meth:`Series.transform` applies the correct one.

Individual stocks and ETFs
--------------------------
FRED does not carry single names. For AAPL, SPY, QQQ or anything else, download a
CSV (Yahoo Finance's "Download" button, or a broker export) and load it with
:func:`quantos.data.loader.load_ohlcv_csv`. Everything downstream — the risk
metrics, the GARCH fit, the portfolio construction — takes plain arrays and does
not care where they came from.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["BUNDLES", "CATALOG", "Kind", "Series", "describe_bundle", "resolve"]


class Kind(enum.Enum):
    """How a series must be transformed before it can be analysed."""

    #: A price or index level: strictly positive, use log returns.
    LEVEL = "level"
    #: A yield, spread or rate in percent: use first differences.
    RATE = "rate"
    #: An index level that is not a tradeable price (CPI, dollar index).
    INDEX = "index"


@dataclass(frozen=True)
class Series:
    """One catalogued series."""

    key: str
    fred_id: str
    name: str
    kind: Kind
    #: What this series tells you, in one line.
    reads_as: str

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the analysis-ready series: log returns or first differences."""
        if self.kind is Kind.RATE:
            return np.diff(values)
        if np.any(values <= 0):
            raise ValueError(
                f"{self.key} is catalogued as a {self.kind.value} but contains "
                "non-positive values; log returns are undefined"
            )
        return np.diff(np.log(values))

    @property
    def units(self) -> str:
        return "percentage points" if self.kind is Kind.RATE else "log return"


def _s(key: str, fred_id: str, name: str, kind: Kind, reads_as: str) -> tuple[str, Series]:
    return key, Series(key, fred_id, name, kind, reads_as)


#: Every series QuantOS knows by name.
CATALOG: dict[str, Series] = dict(
    [
        # --- equity ---------------------------------------------------------
        _s("spx", "SP500", "S&P 500", Kind.LEVEL, "the US large-cap equity market"),
        _s(
            "nasdaq", "NASDAQCOM", "Nasdaq Composite", Kind.LEVEL, "US technology and growth equity"
        ),
        _s("wilshire", "WILL5000PR", "Wilshire 5000", Kind.LEVEL, "the total US equity market"),
        _s(
            "vix",
            "VIXCLS",
            "VIX",
            Kind.INDEX,
            "30-day implied volatility on the S&P 500; the 'fear gauge'",
        ),
        # --- rates ----------------------------------------------------------
        _s(
            "ust10y",
            "DGS10",
            "10-year Treasury yield",
            Kind.RATE,
            "the long-end risk-free rate; the discount rate for everything",
        ),
        _s(
            "ust2y",
            "DGS2",
            "2-year Treasury yield",
            Kind.RATE,
            "the market's near-term policy-rate expectation",
        ),
        _s("ust3m", "DGS3MO", "3-month Treasury yield", Kind.RATE, "the short end of the curve"),
        _s(
            "fedfunds",
            "DFF",
            "Effective federal funds rate",
            Kind.RATE,
            "the policy rate actually transacted overnight",
        ),
        _s(
            "curve10y2y",
            "T10Y2Y",
            "10y minus 2y spread",
            Kind.RATE,
            "the yield curve; sustained inversion has preceded every US recession "
            "since 1970, though with long and variable lead times",
        ),
        _s(
            "real10y",
            "DFII10",
            "10-year TIPS yield",
            Kind.RATE,
            "the real (inflation-adjusted) long rate",
        ),
        _s(
            "breakeven10y",
            "T10YIE",
            "10-year breakeven inflation",
            Kind.RATE,
            "market-implied average inflation over ten years",
        ),
        # --- credit ---------------------------------------------------------
        _s(
            "hy_spread",
            "BAMLH0A0HYM2",
            "High-yield credit spread",
            Kind.RATE,
            "compensation demanded for junk-bond default risk; the cleanest "
            "single read on risk appetite",
        ),
        _s(
            "ig_spread",
            "BAMLC0A0CM",
            "Investment-grade credit spread",
            Kind.RATE,
            "the same for high-quality corporate credit",
        ),
        # --- macro ----------------------------------------------------------
        _s(
            "cpi",
            "CPIAUCSL",
            "CPI, all urban consumers",
            Kind.INDEX,
            "the headline consumer price index (monthly)",
        ),
        _s(
            "unemployment",
            "UNRATE",
            "Unemployment rate",
            Kind.RATE,
            "the headline US unemployment rate (monthly)",
        ),
        _s(
            "payrolls",
            "PAYEMS",
            "Nonfarm payrolls",
            Kind.INDEX,
            "total US non-farm employment (monthly)",
        ),
        _s("gdp", "GDPC1", "Real GDP", Kind.INDEX, "inflation-adjusted US output (quarterly)"),
        _s(
            "recession",
            "USREC",
            "NBER recession indicator",
            Kind.RATE,
            "1 during an NBER-dated recession, 0 otherwise",
        ),
        # --- FX and commodities --------------------------------------------
        _s(
            "dollar",
            "DTWEXBGS",
            "Trade-weighted dollar index",
            Kind.INDEX,
            "the dollar against a broad basket",
        ),
        _s("wti", "DCOILWTICO", "WTI crude oil", Kind.LEVEL, "the US benchmark oil price"),
        _s("gold", "IQ12260", "Gold price", Kind.LEVEL, "gold, the classic real-asset hedge"),
    ]
)


#: Curated groups, each assembled to answer one question.
BUNDLES: dict[str, tuple[str, list[str]]] = {
    "equity": (
        "The US equity complex and its implied volatility.",
        ["spx", "nasdaq", "vix"],
    ),
    "rates": (
        "The Treasury curve, from the policy rate to the long end.",
        ["fedfunds", "ust2y", "ust10y", "curve10y2y"],
    ),
    "risk-appetite": (
        "What investors charge for risk: equity vol against credit spreads. "
        "These co-move sharply in a crisis and diverge in a slow grind.",
        ["spx", "vix", "hy_spread", "ig_spread"],
    ),
    "inflation": (
        "Realised inflation, market expectations, and the real rate that connects them.",
        ["cpi", "breakeven10y", "real10y", "ust10y"],
    ),
    "macro": (
        "The classic recession dashboard: growth, labour, the curve, credit.",
        ["unemployment", "curve10y2y", "hy_spread", "payrolls"],
    ),
    "crossasset": (
        "Equity, bonds, dollar and oil -- the four inputs to almost any top-down allocation.",
        ["spx", "ust10y", "dollar", "wti"],
    ),
}


def resolve(name: str) -> Series:
    """Look up a series by catalogue key or by raw FRED id.

    Raises
    ------
        :class:`KeyError` with the available keys, since a typo here otherwise
        surfaces as an opaque HTTP error much later.
    """
    key = name.strip().lower()
    if key in CATALOG:
        return CATALOG[key]
    for series in CATALOG.values():
        if series.fred_id.upper() == name.strip().upper():
            return series
    # Allow any raw FRED id through, treated as a level unless it looks like a rate.
    upper = name.strip().upper()
    if upper.isalnum() or "_" in upper:
        return Series(
            key=upper.lower(),
            fred_id=upper,
            name=upper,
            kind=Kind.LEVEL,
            reads_as="uncatalogued FRED series, assumed to be a price level",
        )
    raise KeyError(
        f"unknown series {name!r}. Known keys: {', '.join(sorted(CATALOG))}. "
        "Any raw FRED series ID also works."
    )


def describe_bundle(name: str) -> str:
    """One-line description of a bundle, for the CLI."""
    if name not in BUNDLES:
        raise KeyError(f"unknown bundle {name!r}; choose from {', '.join(sorted(BUNDLES))}")
    return BUNDLES[name][0]
