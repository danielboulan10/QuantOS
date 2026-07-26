"""Build factor return series aligned to an instrument's own dates.

Why alignment happens here rather than at each call site
---------------------------------------------------------
Factor series and instrument series almost never share a calendar. They come
from different sources, have different holidays, and routinely end on different
days -- the high-yield credit series used here typically lags the equity close by
one session.

Two ways of handling that are wrong in ways that do not announce themselves:

**Truncating the instrument to the factors' overlap.** This was the original
behaviour, and one short factor cost most of the sample: the credit series
returns only about three years, which cut an AAPL report from 2,151 observations
to 747. Volatility, Sharpe, drawdown and GARCH were then all computed on a third
of the available history in order to satisfy a regression they have nothing to do
with. It also changed conclusions -- the Sharpe standard error nearly doubled,
flipping the verdict on whether it was distinguishable from zero.

**Pairing the trailing N observations of each series.** Cheap and wrong whenever
two series end on different days, which silently regresses Monday's return on
Friday's factor.

So this module returns factors on the *instrument's* date grid, with ``NaN``
wherever a factor has no observation for that day. Callers keep their full
history, and the regression drops incomplete rows itself -- every row it does use
is a genuine same-day pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["DEFAULT_FACTORS", "FactorSet", "build_factors"]

#: Label -> FRED series id. Rates and credit are quoted in percentage points and
#: are differenced; the market series is a level and is converted to returns.
DEFAULT_FACTORS: dict[str, str] = {
    "market": "SP500",
    "rates_10y": "DGS10",
    "credit_hy": "BAMLH0A0HYM2",
}

#: Labels whose FRED series are already in percentage points, so a first
#: difference is the change in yield/spread rather than a return.
_DIFFERENCE_PREFIXES = ("rates", "credit")


@dataclass
class FactorSet:
    """Factor returns on an instrument's date grid, plus what was dropped."""

    #: Label -> array the same length as the instrument's return series.
    columns: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    #: Days on which every factor is present.
    complete_days: int = 0
    #: Length of the instrument's return series.
    total_days: int = 0
    unavailable: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.columns)

    @property
    def coverage(self) -> float:
        return self.complete_days / self.total_days if self.total_days else 0.0

    def note(self) -> str:
        """A one-line description of the coverage, or empty when it is complete."""
        if not self.columns or self.total_days == 0:
            return ""
        if self.complete_days < 150:
            return (
                f"only {self.complete_days} days have every factor present; the "
                "factor regression is skipped and every other analysis uses the "
                "full history"
            )
        if self.coverage < 0.8:
            return (
                f"the factor regression uses the {self.complete_days:,} days where all "
                f"factors are present, out of {self.total_days:,}. Every other "
                "analysis uses the full history"
            )
        return ""

    @property
    def usable(self) -> bool:
        return bool(self.columns) and self.complete_days >= 150


def build_factors(
    dates: NDArray[np.datetime64],
    prices: NDArray[np.float64],
    *,
    start: str = "2000-01-01",
    offline: bool = False,
    wanted: dict[str, str] | None = None,
) -> FactorSet:
    """Fetch factors and align them to ``dates``.

    Inputs
        ``dates``/``prices`` -- the instrument. Factors are returned on the
        *return* grid, i.e. one shorter than ``dates``.
    Outputs
        A :class:`FactorSet` whose columns are all the same length as the
        instrument's returns, padded with ``NaN`` where a factor is missing.
    Failure modes
        A factor that cannot be fetched is recorded in ``unavailable`` and
        omitted; it is not an error, because the rest of the report does not
        depend on it.
    """
    from quantos.data.fred import FredClient, FredError

    wanted = wanted or DEFAULT_FACTORS
    client = FredClient(offline=offline)

    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        return FactorSet()
    return_dates = np.asarray(dates)[1:]

    columns: dict[str, NDArray[np.float64]] = {}
    unavailable: list[str] = []

    for label, series_id in wanted.items():
        try:
            fetched = client.get(series_id).since(start)
        except (FredError, KeyError):
            unavailable.append(label)
            continue

        values = np.asarray(fetched.values, dtype=float)
        if values.size < 2:
            unavailable.append(label)
            continue

        if label.startswith(_DIFFERENCE_PREFIXES):
            # Already a rate or spread in percentage points: difference it.
            changes = np.diff(values)
        else:
            previous = values[:-1]
            positive = previous > 0
            changes = np.where(
                positive, np.diff(values) / np.where(positive, previous, 1.0), np.nan
            )

        lookup = dict(zip(np.asarray(fetched.dates)[1:], changes, strict=True))
        columns[label] = np.array([lookup.get(day, np.nan) for day in return_dates], dtype=float)

    if not columns:
        return FactorSet(unavailable=unavailable, total_days=int(return_dates.size))

    stacked = np.column_stack(list(columns.values()))
    complete = int(np.sum(np.all(np.isfinite(stacked), axis=1)))

    return FactorSet(
        columns=columns,
        complete_days=complete,
        total_days=int(return_dates.size),
        unavailable=unavailable,
    )
