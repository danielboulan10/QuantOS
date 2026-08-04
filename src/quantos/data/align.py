r"""Put a second series onto an instrument's own dates, without moving anything.

Alignment sounds like plumbing and is not. It is where two of this repository's
worst bugs lived, and both were silent -- the numbers stayed plausible and only
the sample changed underneath them.

**Aligning the wrong way truncates.** The first factor loader aligned the
instrument onto the factor calendar, which cut 2,151 observations to 747 and
flipped a Sharpe verdict from "distinguishable from zero" to "not". Nothing
raised. The rule this module enforces is that the *instrument's* dates are the
grid: a macro series that is missing on some of them contributes NaN there, and
the instrument is never shortened to suit it.

**Differencing the wrong way changes the units.** A yield quoted at 4.25 means
4.25 percent; its day-on-day difference is a move in percentage points, and a
100bp shock is 1.0 of those, not 0.01. A price level differences relatively. Both
appear as "change in the factor" and mixing them is a 100x error that produces
numbers still small enough to look reasonable. :class:`FactorKind` makes the
choice explicit at the point where the series is declared, once, rather than at
every call site.

This logic previously lived inside a CLI command, which meant the only way to
test it was to run the CLI, and the only way to reuse it was to copy it.

Example
    >>> import numpy as np
    >>> grid = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    >>> source = np.array(["2024-01-01", "2024-01-03"], dtype="datetime64[D]")
    >>> align_to_grid(grid, source, np.array([4.0, 4.5]))
    array([4. , nan, 4.5])
    >>> coverage(grid, source)
    0.666...
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MACRO_FACTORS",
    "FactorKind",
    "MacroFactor",
    "align_to_grid",
    "coverage",
    "factor_changes",
    "simple_returns",
]


def simple_returns(prices: ArrayLike, *, pad: bool = True) -> NDArray[np.float64]:
    """Period-on-period simple returns.

    Args:
        prices: the price or level series.
        pad: prepend a zero so the result is the same length as the input,
            which is what every caller aligning returns against dates wants.
            ``False`` returns ``n-1`` values.

    This existed inline at seven call sites, each re-deriving the same
    ``np.diff(p) / p[:-1]`` and each choosing its own padding convention. Two of
    them disagreed about whether the result was ``n`` or ``n-1`` long, which is
    an off-by-one waiting for the first caller who zips it against dates.

    Example
        >>> simple_returns([100.0, 110.0, 99.0])
        array([ 0. ,  0.1, -0.1])
        >>> simple_returns([100.0, 110.0], pad=False)
        array([0.1])
    """
    values = np.asarray(prices, dtype=float).ravel()
    if values.size < 2:
        return np.zeros(values.size if pad else 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        changes = np.diff(values) / values[:-1]

    if not pad:
        return np.asarray(changes, dtype=float)

    out = np.zeros(values.size)
    out[1:] = changes
    return out


def align_to_grid(
    grid: ArrayLike,
    source_dates: ArrayLike,
    source_values: ArrayLike,
) -> NDArray[np.float64]:
    """Place ``source_values`` on ``grid``, NaN where the source has no observation.

    The grid is the instrument's own dates and is never modified. This direction
    is the whole point: aligning the other way would drop instrument
    observations to match a shorter macro history, which shortens the sample
    silently.

    Args:
        grid: the dates to align onto, ascending.
        source_dates: dates of the series being aligned.
        source_values: values matching ``source_dates``.

    Returns
    -------
        An array the length of ``grid``.

    Raises
    ------
        ValueError: if the source dates and values disagree in length.
    """
    target = np.asarray(grid, dtype="datetime64[D]")
    dates = np.asarray(source_dates, dtype="datetime64[D]")
    values = np.asarray(source_values, dtype=float).ravel()

    if dates.size != values.size:
        raise ValueError(
            f"{dates.size} source dates against {values.size} values; a series "
            "whose dates and values disagree cannot be aligned, only guessed at"
        )

    # A dict lookup rather than searchsorted: exact date matching only. An
    # interpolating or nearest-date join would invent observations on days the
    # source did not trade, which is how a holiday becomes a data point.
    position = {date: index for index, date in enumerate(dates)}
    return np.array(
        [values[position[date]] if date in position else np.nan for date in target],
        dtype=float,
    )


def coverage(grid: ArrayLike, source_dates: ArrayLike) -> float:
    """Fraction of the grid for which the source has an observation.

    Worth reporting next to any aligned series. A factor present on 40% of the
    instrument's days is not a factor, it is a subsample, and a regression run
    on it is answering a question about different dates than the caller thinks.
    """
    target = np.asarray(grid, dtype="datetime64[D]")
    if target.size == 0:
        return 0.0
    present = np.isin(target, np.asarray(source_dates, dtype="datetime64[D]"))
    return float(np.mean(present))


class FactorKind(Enum):
    """How a macro series must be differenced to become a change.

    The distinction is not cosmetic. A yield differences *absolutely* -- 4.25 to
    4.35 is a 10bp move -- while a price level differences *relatively*. Using
    the wrong one is a unit error of roughly 100x that still produces small,
    plausible-looking numbers.
    """

    #: Quoted in percent (4.25 means 4.25%). Differences are percentage-point
    #: moves, divided by 100 so a 100bp shock is 0.01 in the same units.
    YIELD = "yield"
    #: An index or price level. Differences are relative.
    LEVEL = "level"


class MacroFactor:
    """A named macro series and how to turn it into a change series."""

    __slots__ = ("description", "kind", "name", "series_id")

    def __init__(self, name: str, series_id: str, kind: FactorKind, description: str) -> None:
        self.name = name
        self.series_id = series_id
        self.kind = kind
        self.description = description

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"MacroFactor({self.name!r}, {self.series_id!r}, {self.kind.value!r})"


#: The macro factors QuantOS knows how to fetch, keyed by the name used in a
#: scenario. Series identifiers are FRED's. Declaring the kind here means a
#: caller cannot get the differencing wrong by forgetting which is which.
MACRO_FACTORS: dict[str, MacroFactor] = {
    factor.name: factor
    for factor in (
        MacroFactor(
            "rates",
            "DGS10",
            FactorKind.YIELD,
            "10-year constant-maturity Treasury yield. The discount-rate channel.",
        ),
        MacroFactor(
            "credit",
            "BAMLH0A0HYM2",
            FactorKind.YIELD,
            "ICE BofA US high-yield option-adjusted spread. The refinancing channel, "
            "which usually reprices before earnings do.",
        ),
        MacroFactor(
            "oil",
            "DCOILWTICO",
            FactorKind.LEVEL,
            "WTI spot. Producers and consumers respond with opposite signs, so an "
            "index-level answer averages the effect away.",
        ),
        MacroFactor(
            "dollar",
            "DTWEXBGS",
            FactorKind.LEVEL,
            "Trade-weighted broad dollar index. Pressures foreign revenue and "
            "commodity prices at once.",
        ),
    )
}


def factor_changes(
    grid: ArrayLike,
    source_dates: ArrayLike,
    source_values: ArrayLike,
    kind: FactorKind,
) -> NDArray[np.float64]:
    """Align a macro series onto a grid and difference it in the right units.

    The two operations are combined deliberately. Splitting them invites a
    caller to align correctly and then difference a yield relatively, which is
    the 100x error this module exists to prevent.

    Returns
    -------
        An array the length of ``grid``, with NaN in the first position and
        wherever the source had no observation.

    Example
        >>> import numpy as np
        >>> grid = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
        >>> factor_changes(grid, grid, [4.25, 4.35], FactorKind.YIELD)
        array([  nan, 0.001])
    """
    level = align_to_grid(grid, source_dates, source_values)
    change = np.full(level.size, np.nan)

    if level.size < 2:
        return change

    with np.errstate(divide="ignore", invalid="ignore"):
        if kind is FactorKind.YIELD:
            change[1:] = np.diff(level) / 100.0
        else:
            change[1:] = np.diff(level) / level[:-1]
    return change
