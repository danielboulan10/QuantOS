"""Instrument types, and what analysis each one admits.

Why this abstraction exists
---------------------------
"Run the analysis on this instrument" is not one operation. A stock, an ETF, an
option and a futures contract share a price series and almost nothing else:

===========  ==================================================================
Equity       Has dividends. Returns require an adjusted price or they are wrong.
             Idiosyncratic risk decomposes against a market factor.
ETF          Same as equity mechanically, but the useful questions differ:
             tracking error, holdings overlap, and whether it is a wrapper for
             something that already has a benchmark.
Option       Has an expiry, a strike, and a *non-linear* payoff. Its risk is
             Greeks, not beta. Volatility is an input and an output at once.
Future       Has an expiry and a term structure. Its return has a roll component
             that a spot series does not show, and carry that can dominate.
Index/Rate   Not directly tradeable. A "return" on a yield is meaningless.
===========  ==================================================================

Applying equity analytics to an option produces numbers that are all defined and
all wrong -- a Sharpe ratio on option returns is dominated by the payoff's
convexity, not by any edge. So the report generator asks the instrument what it
supports, and omits what does not apply rather than printing something plausible.

That refusal is the design principle. A research tool that always fills every
field trains its user to stop reading.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Analysis",
    "AssetClass",
    "FutureSpec",
    "Instrument",
    "OptionSpec",
    "supported_analyses",
]


class AssetClass(enum.Enum):
    """What kind of thing this is, which determines what can be computed."""

    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    RATE = "rate"
    OPTION = "option"
    FUTURE = "future"
    COMMODITY = "commodity"
    FX = "fx"

    @property
    def is_tradeable(self) -> bool:
        """Whether holding it produces an investable return."""
        return self not in (AssetClass.INDEX, AssetClass.RATE)

    @property
    def has_linear_payoff(self) -> bool:
        """Whether return statistics mean what they usually mean.

        False for options: their return distribution is dominated by convexity,
        so a Sharpe ratio computed on option returns measures the payoff shape
        rather than any skill or edge.
        """
        return self is not AssetClass.OPTION

    @property
    def has_expiry(self) -> bool:
        return self in (AssetClass.OPTION, AssetClass.FUTURE)


class Analysis(enum.Enum):
    """A unit of research the report can perform."""

    RETURN_DISTRIBUTION = "return_distribution"
    RISK_METRICS = "risk_metrics"
    VOLATILITY_MODEL = "volatility_model"
    REGIME_DETECTION = "regime_detection"
    FACTOR_EXPOSURE = "factor_exposure"
    SIGNAL_BATTERY = "signal_battery"
    OPTION_ANALYTICS = "option_analytics"
    TERM_STRUCTURE = "term_structure"
    EXECUTION_COST = "execution_cost"
    STATIONARITY = "stationarity"


#: Which analyses apply to which asset class. Everything else is skipped, with
#: the reason printed, rather than computed and quietly misinterpreted.
_APPLICABLE: dict[AssetClass, set[Analysis]] = {
    AssetClass.EQUITY: {
        Analysis.RETURN_DISTRIBUTION,
        Analysis.RISK_METRICS,
        Analysis.VOLATILITY_MODEL,
        Analysis.REGIME_DETECTION,
        Analysis.FACTOR_EXPOSURE,
        Analysis.SIGNAL_BATTERY,
        Analysis.OPTION_ANALYTICS,
        Analysis.EXECUTION_COST,
        Analysis.STATIONARITY,
    },
    AssetClass.ETF: {
        Analysis.RETURN_DISTRIBUTION,
        Analysis.RISK_METRICS,
        Analysis.VOLATILITY_MODEL,
        Analysis.REGIME_DETECTION,
        Analysis.FACTOR_EXPOSURE,
        Analysis.SIGNAL_BATTERY,
        Analysis.OPTION_ANALYTICS,
        Analysis.EXECUTION_COST,
        Analysis.STATIONARITY,
    },
    AssetClass.INDEX: {
        Analysis.RETURN_DISTRIBUTION,
        Analysis.VOLATILITY_MODEL,
        Analysis.REGIME_DETECTION,
        Analysis.FACTOR_EXPOSURE,
        Analysis.SIGNAL_BATTERY,
        Analysis.STATIONARITY,
        Analysis.OPTION_ANALYTICS,
    },
    AssetClass.RATE: {
        Analysis.RETURN_DISTRIBUTION,
        Analysis.VOLATILITY_MODEL,
        Analysis.REGIME_DETECTION,
        Analysis.STATIONARITY,
        Analysis.FACTOR_EXPOSURE,
    },
    AssetClass.OPTION: {Analysis.OPTION_ANALYTICS, Analysis.EXECUTION_COST},
    AssetClass.FUTURE: {
        Analysis.RETURN_DISTRIBUTION,
        Analysis.RISK_METRICS,
        Analysis.VOLATILITY_MODEL,
        Analysis.TERM_STRUCTURE,
        Analysis.SIGNAL_BATTERY,
        Analysis.EXECUTION_COST,
        Analysis.STATIONARITY,
    },
    AssetClass.COMMODITY: {
        Analysis.RETURN_DISTRIBUTION,
        Analysis.RISK_METRICS,
        Analysis.VOLATILITY_MODEL,
        Analysis.REGIME_DETECTION,
        Analysis.SIGNAL_BATTERY,
        Analysis.TERM_STRUCTURE,
        Analysis.STATIONARITY,
    },
    AssetClass.FX: {
        Analysis.RETURN_DISTRIBUTION,
        Analysis.RISK_METRICS,
        Analysis.VOLATILITY_MODEL,
        Analysis.REGIME_DETECTION,
        Analysis.SIGNAL_BATTERY,
        Analysis.STATIONARITY,
    },
}


def supported_analyses(asset_class: AssetClass) -> set[Analysis]:
    """Which analyses are meaningful for this asset class."""
    return set(_APPLICABLE.get(asset_class, set()))


@dataclass(frozen=True)
class OptionSpec:
    """The contract terms of an option, for analytics that need them."""

    strike: float
    expiry: date
    is_call: bool = True
    #: Observed market price, if you have one. Enables implied-vol inversion.
    market_price: float | None = None
    contract_multiplier: float = 100.0

    def years_to_expiry(self, as_of: date) -> float:
        """Act/365 year fraction. Negative if already expired."""
        return (self.expiry - as_of).days / 365.0

    def moneyness(self, spot: float) -> float:
        r"""Log-moneyness :math:`\ln(S/K)`, the natural surface coordinate."""
        return float(np.log(spot / self.strike))


@dataclass(frozen=True)
class FutureSpec:
    """One futures contract in a term structure."""

    expiry: date
    price: float
    label: str = ""
    open_interest: float | None = None

    def years_to_expiry(self, as_of: date) -> float:
        return (self.expiry - as_of).days / 365.0


@dataclass
class Instrument:
    """Everything the report generator needs to know about one thing.

    Example
        >>> import numpy as np
        >>> from datetime import date
        >>> dates = np.array(['2024-01-02', '2024-01-03'], dtype='datetime64[D]')
        >>> inst = Instrument('SPY', AssetClass.ETF, dates, np.array([470.0, 472.0]))
        >>> inst.symbol, inst.asset_class.is_tradeable
        ('SPY', True)
    """

    symbol: str
    asset_class: AssetClass
    dates: NDArray[np.datetime64]
    prices: NDArray[np.float64]

    name: str = ""
    #: Where the data came from, carried into the report for provenance.
    source: str = ""
    #: True if the price series accounts for dividends.
    dividend_adjusted: bool = False
    currency: str = "USD"
    #: Average daily volume, if known. Enables execution-cost estimates.
    average_daily_volume: float | None = None

    option: OptionSpec | None = None
    term_structure: list[FutureSpec] = field(default_factory=list)
    detail: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.prices.size)

    @property
    def start(self) -> str:
        return str(self.dates[0])[:10] if len(self) else ""

    @property
    def end(self) -> str:
        return str(self.dates[-1])[:10] if len(self) else ""

    @property
    def latest(self) -> float:
        return float(self.prices[-1]) if len(self) else float("nan")

    @property
    def display_name(self) -> str:
        return self.name or self.symbol

    def returns(self) -> NDArray[np.float64]:
        """Analysis-ready changes: log returns for levels, differences for rates.

        A rate series is differenced because its log is undefined once it goes
        negative -- which two-year yields and curve spreads routinely do.
        """
        if len(self) < 2:
            return np.zeros(0)
        if self.asset_class is AssetClass.RATE or np.any(self.prices <= 0):
            return np.diff(self.prices)
        return np.diff(np.log(self.prices))

    def supports(self, analysis: Analysis) -> bool:
        return analysis in supported_analyses(self.asset_class)

    def skip_reason(self, analysis: Analysis) -> str:
        """Why an analysis does not apply -- printed instead of a fake number."""
        if self.supports(analysis):
            return ""
        reasons = {
            (
                AssetClass.RATE,
                Analysis.RISK_METRICS,
            ): "a yield is not held, so it has no return, Sharpe ratio or drawdown",
            (
                AssetClass.INDEX,
                Analysis.RISK_METRICS,
            ): "an index is not directly investable; analyse a tracking ETF instead",
            (
                AssetClass.OPTION,
                Analysis.RISK_METRICS,
            ): "option returns are dominated by payoff convexity, so a Sharpe "
            "ratio measures the contract's shape rather than any edge",
            (
                AssetClass.OPTION,
                Analysis.SIGNAL_BATTERY,
            ): "momentum and mean-reversion signals on an option price conflate "
            "the underlying's move with time decay and vol changes",
            (
                AssetClass.RATE,
                Analysis.SIGNAL_BATTERY,
            ): "signals are reported on tradeable instruments; a yield needs a "
            "bond or futures position to express",
        }
        key = (self.asset_class, analysis)
        article = "an" if self.asset_class.value[0] in "aeiou" else "a"
        return reasons.get(
            key, f"{analysis.value} does not apply to {article} {self.asset_class.value}"
        )

    def data_quality_warnings(self) -> list[str]:
        """Problems with the data that would bias the analysis if unstated."""
        warnings: list[str] = []
        if self.asset_class in (AssetClass.EQUITY, AssetClass.ETF) and not self.dividend_adjusted:
            warnings.append(
                "prices are NOT dividend-adjusted: every ex-dividend date injects "
                "a spurious negative return, biasing return and volatility "
                "estimates downward. Prefer an adjusted-close column."
            )
        if len(self) < 250:
            warnings.append(
                f"only {len(self)} observations (~{len(self) / 252:.1f} years). "
                "Volatility estimates are usable; Sharpe ratios and any "
                "significance claim are not."
            )
        if len(self) >= 2:
            gaps = np.diff(self.dates.astype("datetime64[D]").astype(int))
            if gaps.size and int(np.max(gaps)) > 10:
                warnings.append(
                    f"largest gap between observations is {int(np.max(gaps))} days; "
                    "the series may be incomplete or irregularly sampled."
                )
            returns = self.returns()
            if returns.size:
                extreme = int(np.sum(np.abs(returns) > 0.5))
                if extreme and self.asset_class.has_linear_payoff:
                    warnings.append(
                        f"{extreme} observation(s) move more than 50% in a day, "
                        "which usually indicates a split or a data error rather "
                        "than a real move."
                    )
        return warnings
