r"""Replay a portfolio through crises that actually happened.

A simulated shock tells you what your model thinks. A dated historical window
tells you what the market did, including the parts no model contains: that
correlations among risk assets converge exactly when diversification is needed,
that a hedge which held for a decade can invert, and that recovery takes years
rather than the quarters a mean-reversion assumption implies.

The design constraint here is honesty about coverage, and it is the reason most
of this module is not arithmetic. Two failure modes matter:

**Survivorship.** An instrument that did not exist in 2008 cannot be stress
tested against 2008. A tool that quietly returns a number anyway -- by
substituting a proxy, or by testing whatever overlap happens to exist -- produces
the most dangerous output available: a confident figure for a scenario the asset
never experienced. Every result here carries its coverage, and a window that is
not covered returns :data:`None` rather than a number.

**Partial coverage.** Worse than no coverage, because it looks like coverage. An
ETF launched in November 2008 has data inside the GFC window and would report a
drawdown of a few percent, having missed the entire collapse. Windows are
therefore accepted only if the instrument's history begins before the crisis
does, with a small grace period, and the fraction covered is reported alongside.

Example
    >>> import numpy as np
    >>> dates = np.arange("2006-01-01", "2024-01-01", dtype="datetime64[D]")
    >>> rng = np.random.default_rng(0)
    >>> prices = 100 * np.exp(np.cumsum(rng.standard_normal(dates.size) * 0.01))
    >>> report = stress_test(dates, prices)
    >>> covered = [r for r in report.results if r.covered]
    >>> len(covered) >= 4
    True
    >>> report.worst.crisis.name in {c.name for c in CRISES}
    True

References
----------
    Longin & Solnik (2001), "Extreme Correlation of International Equity
    Markets", *Journal of Finance* 56(2) -- correlation rises in the lower tail
    and the rise is not explained by volatility alone.
    Ang & Chen (2002), "Asymmetric Correlations of Equity Portfolios", *JFE*
    63(3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CRISES",
    "CorrelationBreakdown",
    "Crisis",
    "CrisisResult",
    "PairCorrelation",
    "StressReport",
    "correlation_breakdown",
    "stress_test",
]


# --------------------------------------------------------------------------- #
# The windows
#
# Dates are the episode as it was actually experienced, peak to trough, not
# calendar quarters. Each is bounded by the reference index's own high and low,
# so the window is the drawdown rather than a round-numbered approximation of it.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Crisis:
    """A dated historical episode, with what it did to the S&P 500."""

    name: str
    start: str
    end: str
    #: Roughly how far the S&P 500 fell, peak to trough, for orientation.
    reference_drawdown: float
    description: str

    @property
    def start_date(self) -> np.datetime64:
        return np.datetime64(self.start, "D")

    @property
    def end_date(self) -> np.datetime64:
        return np.datetime64(self.end, "D")


CRISES: tuple[Crisis, ...] = (
    Crisis(
        "dot-com unwind",
        "2000-03-24",
        "2002-10-09",
        -0.49,
        "Two and a half years of grinding decline, not a crash. The Nasdaq fell "
        "78% and took fifteen years to recover its high. The lesson is duration: "
        "a drawdown that arrives slowly is no easier to hold.",
    ),
    Crisis(
        "global financial crisis",
        "2007-10-09",
        "2009-03-09",
        -0.567,
        "The reference case. Equity, credit and property fell together while "
        "correlations among supposedly diversifying assets converged toward one, "
        "which is the risk a correlation matrix estimated in calm cannot show.",
    ),
    Crisis(
        "COVID crash",
        "2020-02-19",
        "2020-03-23",
        -0.339,
        "The fastest 30% decline on record: 33 calendar days. Anything relying on "
        "rebalancing, stop-losses or an orderly exit was tested on speed rather "
        "than on depth.",
    ),
    Crisis(
        "2022 inflation shock",
        "2022-01-03",
        "2022-10-12",
        -0.255,
        "The one that broke the 60/40 portfolio. Bonds fell WITH equities because "
        "the shock was the discount rate itself, so the hedge and the risk asset "
        "shared a cause. A correlation estimated over the prior decade was not "
        "merely stale, it had the sign wrong.",
    ),
    Crisis(
        "regional banking crisis",
        "2023-03-01",
        "2023-05-04",
        -0.075,
        "Shallow in the index and severe underneath it. Regional bank equity fell "
        "by more than half while the S&P 500 finished the period higher, which is "
        "why an index-level stress test can report almost nothing while a "
        "concentrated book is destroyed.",
    ),
)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CrisisResult:
    """What one crisis did to one portfolio -- or why it could not be tested."""

    crisis: Crisis
    covered: bool
    #: Fraction of the crisis window for which data exists. Reported even when
    #: `covered` is False, because "we have 4% of it" and "we have none of it"
    #: are different conversations.
    coverage: float
    reason: str = ""

    total_return: float = float("nan")
    max_drawdown: float = float("nan")
    worst_day: float = float("nan")
    worst_day_date: str = ""
    days_underwater: int = 0
    #: Trading days from the trough back to the pre-crisis high, or None if it
    #: had not recovered by the end of the available history.
    days_to_recover: int | None = None
    annualised_volatility: float = float("nan")
    #: Volatility during the crisis divided by volatility in the year before it.
    volatility_multiple: float = float("nan")

    def summary(self) -> str:
        if not self.covered:
            return f"  {self.crisis.name:<26} NOT TESTABLE -- {self.reason}"
        recovery = (
            f"{self.days_to_recover}d to recover"
            if self.days_to_recover is not None
            else "never recovered in sample"
        )
        return (
            f"  {self.crisis.name:<26} "
            f"{self.max_drawdown:>7.1%} drawdown   "
            f"worst day {self.worst_day:>6.1%}   "
            f"vol x{self.volatility_multiple:.1f}   {recovery}"
        )


@dataclass
class StressReport:
    """Every crisis, tested or explicitly not."""

    results: list[CrisisResult]
    history_start: str
    history_end: str
    notes: list[str] = field(default_factory=list)

    @property
    def tested(self) -> list[CrisisResult]:
        return [result for result in self.results if result.covered]

    @property
    def worst(self) -> CrisisResult:
        """The tested crisis with the deepest drawdown.

        Raises if nothing was testable, rather than returning a placeholder: a
        "worst case" derived from no data is the single most misleading thing
        this module could produce.
        """
        tested = self.tested
        if not tested:
            raise ValueError(
                "no crisis window is covered by this history, so there is no worst case to report"
            )
        return min(tested, key=lambda result: result.max_drawdown)

    def summary(self) -> str:
        lines = [
            f"HISTORICAL STRESS TEST -- data from {self.history_start} to {self.history_end}",
            "=" * 78,
            "",
        ]
        lines.extend(result.summary() for result in self.results)

        tested = self.tested
        lines.append("")
        if tested:
            worst = self.worst
            lines.append(
                f"  Worst experienced: {worst.crisis.name}, "
                f"{worst.max_drawdown:.1%} peak to trough."
            )
            lines.append(f"  {len(tested)} of {len(self.results)} windows testable.")
        else:
            lines.append("  Nothing testable: this history does not reach any crisis window.")

        lines.extend(["", *(f"  {note}" for note in self.notes)])
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The measurements
# --------------------------------------------------------------------------- #
def _drawdown_curve(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    running_max = np.maximum.accumulate(prices)
    return prices / running_max - 1.0


def _analyse_window(
    dates: NDArray[np.datetime64],
    prices: NDArray[np.float64],
    crisis: Crisis,
    *,
    periods_per_year: float = 252.0,
) -> CrisisResult:
    """Measure one crisis, or explain why it cannot be measured."""
    inside = (dates >= crisis.start_date) & (dates <= crisis.end_date)
    window_length = float((crisis.end_date - crisis.start_date).astype(int))

    if not inside.any():
        return CrisisResult(crisis, False, 0.0, "history does not reach this window")

    span = (dates[inside][-1] - dates[inside][0]).astype(int)
    coverage = float(span) / window_length if window_length > 0 else 0.0

    # Partial coverage is the dangerous case: an instrument listed halfway
    # through a crisis has data inside the window and would report a shallow
    # drawdown, having missed the collapse entirely. Require the history to
    # begin before the crisis did, with a fortnight of grace for a listing that
    # is genuinely at the boundary.
    grace = np.timedelta64(14, "D")
    if dates[0] > crisis.start_date + grace:
        return CrisisResult(
            crisis,
            False,
            coverage,
            f"history begins {str(dates[0])[:10]}, after the crisis started "
            f"{crisis.start}; {coverage:.0%} of the window is present and testing "
            "it would miss the decline",
        )

    window_prices = prices[inside]
    if window_prices.size < 10:
        return CrisisResult(
            crisis, False, coverage, f"only {window_prices.size} observations in the window"
        )

    window_dates = dates[inside]
    returns = np.diff(window_prices) / window_prices[:-1]
    drawdown = _drawdown_curve(window_prices)

    trough_index = int(np.argmin(drawdown))
    worst_index = int(np.argmin(returns))

    # Recovery is measured against the pre-crisis high, over the whole history
    # after the trough -- a recovery that happens after the window closes is
    # still a recovery, and cutting the search at the window boundary would
    # report "never recovered" for every crisis that ended in one.
    before_crisis = dates <= crisis.start_date
    peak = float(np.max(prices[before_crisis])) if before_crisis.any() else float(window_prices[0])

    trough_position = int(np.flatnonzero(inside)[trough_index])
    at_or_above_peak = np.flatnonzero(prices[trough_position:] >= peak)
    days_to_recover = int(at_or_above_peak[0]) if at_or_above_peak.size else None

    crisis_vol = float(np.std(returns, ddof=1)) * np.sqrt(periods_per_year)

    # Volatility in the year before the crisis, as the baseline it multiplied.
    before = (dates < crisis.start_date) & (dates >= crisis.start_date - np.timedelta64(365, "D"))
    if before.sum() > 30:
        prior_prices = prices[before]
        prior_returns = np.diff(prior_prices) / prior_prices[:-1]
        prior_vol = float(np.std(prior_returns, ddof=1)) * np.sqrt(periods_per_year)
        multiple = crisis_vol / prior_vol if prior_vol > 0 else float("nan")
    else:
        multiple = float("nan")

    return CrisisResult(
        crisis=crisis,
        covered=True,
        coverage=min(coverage, 1.0),
        total_return=float(window_prices[-1] / window_prices[0] - 1.0),
        max_drawdown=float(np.min(drawdown)),
        worst_day=float(returns[worst_index]),
        worst_day_date=str(window_dates[worst_index + 1])[:10],
        days_underwater=int(np.sum(drawdown < -0.01)),
        days_to_recover=days_to_recover,
        annualised_volatility=crisis_vol,
        volatility_multiple=multiple,
    )


def stress_test(
    dates: ArrayLike,
    prices: ArrayLike,
    *,
    crises: tuple[Crisis, ...] = CRISES,
    periods_per_year: float = 252.0,
) -> StressReport:
    """Replay a price history through every crisis window.

    Args:
        dates: observation dates, ascending, as ``datetime64[D]`` or parseable.
        prices: the price or portfolio value at each date.
        crises: windows to test. Defaults to :data:`CRISES`.
        periods_per_year: annualisation factor.

    Returns
    -------
        A :class:`StressReport` in which untestable windows are marked untestable
        rather than silently omitted -- an omitted window reads as a window that
        passed.
    """
    date_array = np.asarray(dates, dtype="datetime64[D]")
    price_array = np.asarray(prices, dtype=float).ravel()

    if date_array.size != price_array.size:
        raise ValueError(
            f"{date_array.size} dates against {price_array.size} prices; these must align"
        )
    if price_array.size < 30:
        raise ValueError("a stress test over fewer than 30 observations is not one")
    if np.any(np.diff(date_array.astype(int)) < 0):
        raise ValueError("dates must be ascending")

    results = [
        _analyse_window(date_array, price_array, crisis, periods_per_year=periods_per_year)
        for crisis in crises
    ]

    notes: list[str] = []
    untestable = [result for result in results if not result.covered]
    if untestable:
        notes.append(
            f"{len(untestable)} of {len(results)} windows are not testable on this "
            "history. They are listed rather than dropped: a window silently "
            "omitted reads as a window that was survived."
        )
    partial = [result for result in untestable if result.coverage > 0.1]
    if partial:
        notes.append(
            "Some untestable windows have partial data. Partial coverage is worse "
            "than none, because it produces a shallow drawdown that looks like a "
            "result: an instrument listed midway through a crash reports only the "
            "part it was present for."
        )
    notes.append(
        "A historical replay assumes the position was held throughout. It does "
        "not model margin calls, redemptions, borrow being pulled, or the "
        "decision to sell at the bottom -- which is what actually converts a "
        "drawdown into a loss."
    )

    return StressReport(
        results=results,
        history_start=str(date_array[0])[:10],
        history_end=str(date_array[-1])[:10],
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# The thing a correlation matrix cannot tell you
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairCorrelation:
    """One pair, before and during."""

    left: str
    right: str
    calm: float
    stressed: float

    @property
    def change(self) -> float:
        return self.stressed - self.calm

    @property
    def flipped_sign(self) -> bool:
        """Whether the relationship reversed -- a hedge that stopped hedging."""
        return self.calm * self.stressed < 0


@dataclass
class CorrelationBreakdown:
    """Pairwise correlation in calm versus crisis, decomposed by pair type."""

    crisis: Crisis
    pairs: list[PairCorrelation]
    #: Mean correlation among the assets tagged as risk assets.
    within_risk_calm: float = float("nan")
    within_risk_stressed: float = float("nan")
    #: Mean correlation between a risk asset and everything else.
    cross_calm: float = float("nan")
    cross_stressed: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def sign_flips(self) -> list[PairCorrelation]:
        return [pair for pair in self.pairs if pair.flipped_sign]

    def summary(self) -> str:
        lines = [f"CORRELATION THROUGH {self.crisis.name.upper()}", "-" * 60]
        for pair in sorted(self.pairs, key=lambda p: -abs(p.change)):
            flag = "  <- SIGN FLIP" if pair.flipped_sign else ""
            lines.append(
                f"  {pair.left}-{pair.right:<6} "
                f"{pair.calm:+.2f} -> {pair.stressed:+.2f}  "
                f"({pair.change:+.2f}){flag}"
            )
        if np.isfinite(self.within_risk_calm):
            lines.extend(
                [
                    "",
                    f"  among risk assets  {self.within_risk_calm:+.2f} -> "
                    f"{self.within_risk_stressed:+.2f}",
                    f"  risk vs the rest   {self.cross_calm:+.2f} -> {self.cross_stressed:+.2f}",
                ]
            )
        lines.extend(["", *(f"  {note}" for note in self.notes)])
        return "\n".join(lines)


def correlation_breakdown(
    dates: ArrayLike,
    returns_by_asset: dict[str, ArrayLike],
    crisis: Crisis,
    *,
    risk_assets: set[str] | None = None,
) -> CorrelationBreakdown:
    r"""Pairwise correlation before and during a crisis, decomposed by pair type.

    The single most expensive assumption in portfolio construction is that a
    correlation matrix estimated in calm markets describes the one that will
    apply in a crisis. It does not, and the way it fails is not the way the
    slogan says.

    **The average pairwise correlation is the wrong statistic, and the first
    version of this function used it.** On SPY, QQQ, IWM, EFA, TLT and GLD
    through the global financial crisis it reported that correlation *fell*, and
    concluded that "the assets genuinely diversified here" -- the opposite of the
    risk being looked for. The average had netted two effects that move in
    opposite directions:

    ==============================  ==========  ==========
    pair type                       calm        crisis
    ==============================  ==========  ==========
    equity-equity                   +0.84       **+0.90**
    equity vs bonds/gold            +0.05       **-0.20**
    ==============================  ==========  ==========

    Correlations among the risk assets did converge toward one, which is where
    the concentration is and therefore where it hurts. Bonds meanwhile
    diversified *harder*, because a flight to quality is a bid for Treasuries.
    Averaging the two produces a number that is a summary of nothing.

    So the decomposition is the output. Tag which assets are risk assets and the
    two are reported separately; the per-pair table is always returned, because
    the pair that matters is usually a specific one.

    Sign flips are called out for the same reason. Through 2022 the SPY-TLT
    correlation went from -0.40 to +0.03 while TLT fell 31% against SPY's 24% --
    the hedge did not merely weaken, it inverted, and a 60/40 portfolio
    optimised on the previous decade had the sign wrong rather than the
    magnitude.

    Args:
        dates: observation dates, aligned with every return series.
        returns_by_asset: name to return series. At least two.
        crisis: the window to measure.
        risk_assets: names to treat as risk assets for the decomposition. If
            omitted, only the per-pair table is produced -- guessing which
            assets are meant to be the hedge would be inventing the answer.
    """
    date_array = np.asarray(dates, dtype="datetime64[D]")
    names = sorted(returns_by_asset)
    if len(names) < 2:
        raise ValueError("correlation needs at least two assets")

    matrix = np.column_stack([np.asarray(returns_by_asset[name], dtype=float) for name in names])
    if matrix.shape[0] != date_array.size:
        raise ValueError("returns and dates must align")

    inside = (date_array >= crisis.start_date) & (date_array <= crisis.end_date)
    calm_mask = (date_array < crisis.start_date) & (
        date_array >= crisis.start_date - np.timedelta64(730, "D")
    )

    if inside.sum() < 20 or calm_mask.sum() < 60:
        return CorrelationBreakdown(
            crisis, [], notes=["not enough data on both sides of the window"]
        )

    calm_corr = np.asarray(np.corrcoef(matrix[calm_mask], rowvar=False), dtype=float)
    crisis_corr = np.asarray(np.corrcoef(matrix[inside], rowvar=False), dtype=float)

    pairs = [
        PairCorrelation(names[i], names[j], float(calm_corr[i, j]), float(crisis_corr[i, j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]

    breakdown = CorrelationBreakdown(crisis=crisis, pairs=pairs)

    if risk_assets:
        within = [p for p in pairs if p.left in risk_assets and p.right in risk_assets]
        cross = [p for p in pairs if (p.left in risk_assets) != (p.right in risk_assets)]
        if within:
            breakdown.within_risk_calm = float(np.mean([p.calm for p in within]))
            breakdown.within_risk_stressed = float(np.mean([p.stressed for p in within]))
        if cross:
            breakdown.cross_calm = float(np.mean([p.calm for p in cross]))
            breakdown.cross_stressed = float(np.mean([p.stressed for p in cross]))

        if within and breakdown.within_risk_stressed > breakdown.within_risk_calm + 0.03:
            breakdown.notes.append(
                "Correlation among the risk assets ROSE into the crisis: the "
                "diversification measured in calm was not there when it was needed."
            )

    flips = breakdown.sign_flips
    if flips:
        breakdown.notes.append(
            f"{len(flips)} pair(s) changed sign. A hedge that inverts is worse than "
            "one that weakens, because the position was sized as though it offset "
            "the risk it was in fact adding to."
        )
    if not breakdown.notes:
        breakdown.notes.append("No pair moved enough to change how the book should be sized.")

    return breakdown
