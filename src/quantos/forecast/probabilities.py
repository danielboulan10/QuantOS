r"""Probabilities that mean something, computed from simulated paths.

Which probabilities are worth reporting
---------------------------------------
The number every retail tool leads with -- the chance a stock goes up -- is the
least informative one available. Over a month, driftless dynamics put it within a
point or two of 50% for essentially every liquid security, and the honest report
says so. A site quoting "73% chance AAPL rises" is either using a drift estimate
whose standard error swamps it, or inventing.

The probabilities that differ enormously between securities, and that actually
bear on a decision, are all about **paths and tails**:

**Touching a level before the horizon** (:func:`first_passage_probability`). The
chance of hitting a stop at any point is far higher than the chance of *finishing*
below it, because a path can dip and recover. People consistently underestimate
this, and it is the number that decides whether a stop gets hit.

**Drawdown within the holding period** (:func:`drawdown_probability`). Peak-to-
trough inside the window, which is what a holder experiences rather than the
endpoint they are marked at.

**Conditional loss** (:func:`ProbabilityReport.expected_shortfall`). Not "how
often does it lose" but "when it loses, how much" -- the quantity that sizes a
position.

**Long versus short asymmetry** (:func:`long_short_comparison`). Almost no retail
tool shows it, and it is not a detail. A short's loss is unbounded while its gain
caps at 100%; equity returns are right-skewed in the log, so the tail that hurts
a short is the fatter one; and borrow cost accrues daily against the position.
The same forecast therefore implies materially different risk on the two sides.

Every number here is a summary of :class:`~quantos.forecast.paths.PathEnsemble`,
so every number inherits that ensemble's stated assumptions -- and is testable
against outcomes by :mod:`quantos.forecast.calibration`. A probability that has
not been calibration-checked is a number with a percent sign on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantos.forecast.paths import PathEnsemble

__all__ = [
    "LongShortComparison",
    "ProbabilityReport",
    "drawdown_probability",
    "first_passage_probability",
    "long_short_comparison",
    "probability_report",
]


def _share(mask: NDArray[np.bool_]) -> float:
    return float(np.mean(mask)) if mask.size else float("nan")


def first_passage_probability(
    ensemble: PathEnsemble, level: float, *, direction: str = "down"
) -> float:
    r"""Probability of touching ``level`` at any point before the horizon.

    This is the quantity a stop-loss actually responds to, and it is strictly
    larger than the probability of *ending* beyond the level -- a path may breach
    and recover. Computing it needs the paths themselves; there is no way to read
    it off terminal quantiles, which is why the ensemble is simulated rather than
    summarised analytically.
    """
    highs, lows = ensemble.running_extremes()
    if direction == "down":
        return _share(lows[:, -1] <= level)
    if direction == "up":
        return _share(highs[:, -1] >= level)
    raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")


def _median_worst_drawdown(ensemble: PathEnsemble) -> float:
    """Depth exceeded by the worst drawdown on half the paths."""
    running_max = np.maximum.accumulate(ensemble.paths, axis=1)
    worst = np.min(ensemble.paths / running_max - 1.0, axis=1)
    return float(-np.median(worst))


def drawdown_probability(ensemble: PathEnsemble, depth: float) -> float:
    r"""Probability of a peak-to-trough fall of at least ``depth`` (e.g. ``0.10``).

    Measured *within* the path rather than from the starting price, because that
    is what a holder lives through: an instrument can finish the month up 3% and
    still have been 15% underwater in between.
    """
    if not 0.0 < depth < 1.0:
        raise ValueError(f"depth must lie in (0, 1), got {depth}")
    running_max = np.maximum.accumulate(ensemble.paths, axis=1)
    drawdowns = ensemble.paths / running_max - 1.0
    worst = np.min(drawdowns, axis=1)
    return _share(worst <= -depth)


@dataclass
class ProbabilityReport:
    """The forward-looking numbers for one instrument at one horizon."""

    symbol: str
    spot: float
    horizon_days: int
    engine: str

    probability_up: float
    #: level -> probability of finishing at or beyond it.
    terminal_thresholds: dict[str, float] = field(default_factory=dict)
    #: level -> probability of *touching* it at any point.
    touch_thresholds: dict[str, float] = field(default_factory=dict)
    #: depth -> probability of a drawdown at least that deep.
    drawdowns: dict[str, float] = field(default_factory=dict)

    median_return: float = float("nan")
    expected_shortfall_95: float = float("nan")
    expected_gain_given_gain: float = float("nan")
    expected_loss_given_loss: float = float("nan")
    #: Pointwise quantiles of the terminal price.
    terminal_quantiles: dict[float, float] = field(default_factory=dict)
    #: Depth the worst drawdown exceeds with probability 0.5. Horizon-adaptive.
    median_worst_drawdown: float = float("nan")
    annualised_volatility: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def direction_verdict(self) -> str:
        """Plain language about direction, which is usually 'no information'."""
        distance = abs(self.probability_up - 0.5)
        if distance < 0.03:
            return (
                f"Direction is a coin flip ({self.probability_up:.0%} up). That is the "
                "honest answer over this horizon, not a gap in the model: expected "
                "return cannot be estimated precisely enough from price history to say "
                "otherwise."
            )
        return (
            f"{self.probability_up:.0%} chance of finishing higher. This departs from "
            "50% only because a drift was supplied; it is not evidence of direction."
        )

    @property
    def risk_verdict(self) -> str:
        """Lead with the median worst drawdown, not a fixed threshold.

        A fixed 10% threshold stops carrying information as the horizon grows: on
        a volatile name over 160 days the answer is "100% chance", which is true
        and tells a reader nothing. The depth that is exceeded half the time
        stays informative at every horizon and volatility, and is directly
        comparable between instruments.
        """
        depth = self.median_worst_drawdown
        if not np.isfinite(depth) or self.horizon_days <= 0:
            return "not computable"

        # Severity is judged on a horizon-normalised depth, because drawdown
        # scales roughly with sigma*sqrt(T): 17.8% over 21 days is alarming and
        # the same figure over 160 days is ordinary. Dividing by sqrt(T in years)
        # puts every horizon on one comparable scale, so the wording means the
        # same thing whether the reader asked for a month or a year.
        scaled = depth / np.sqrt(self.horizon_days / 252.0)
        if scaled > 0.35:
            severity = "high"
        elif scaled > 0.18:
            severity = "moderate"
        else:
            severity = "low"
        return (
            f"{severity.capitalize()} path risk: a coin flip that the worst drawdown over "
            f"the next {self.horizon_days} days exceeds {depth:.1%}."
        )

    def summary(self) -> str:
        lines = [
            f"{self.symbol} — {self.horizon_days}-day forward distribution "
            f"({self.engine}, {self.annualised_volatility:.1%} annualised vol)",
            f"  spot {self.spot:,.2f}   median outcome {self.median_return:+.2%}",
            "",
            "  DIRECTION",
            f"    {self.direction_verdict}",
            "",
            "  WHERE IT MIGHT FINISH",
        ]
        for label, probability in self.terminal_thresholds.items():
            lines.append(f"    finishes {label:>7s}   {probability:6.1%}")
        lines += ["", "  WHAT IT MIGHT TOUCH ALONG THE WAY"]
        for label, probability in self.touch_thresholds.items():
            lines.append(f"    touches  {label:>7s}   {probability:6.1%}")
        lines += ["", "  DRAWDOWN WITHIN THE PERIOD"]
        for label, probability in self.drawdowns.items():
            lines.append(f"    at least {label:>7s}   {probability:6.1%}")
        lines += [
            "",
            "  IF IT GOES AGAINST YOU",
            f"    average loss, given a loss      {self.expected_loss_given_loss:+.2%}",
            f"    average of the worst 5%         {self.expected_shortfall_95:+.2%}",
            f"    average gain, given a gain      {self.expected_gain_given_gain:+.2%}",
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def probability_report(
    ensemble: PathEnsemble,
    *,
    symbol: str = "",
    moves: tuple[float, ...] = (0.05, 0.10, 0.20),
    drawdown_depths: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> ProbabilityReport:
    """Summarise an ensemble into the numbers a reader can act on.

    Example
        >>> import numpy as np
        >>> from quantos.forecast.paths import simulate_garch_paths
        >>> rng = np.random.default_rng(0)
        >>> r = rng.normal(0, 0.02, 1200)
        >>> e = simulate_garch_paths(r, 100.0, 21, n_paths=5000)
        >>> report = probability_report(e, symbol="TEST")
        >>> 0.4 < report.probability_up < 0.6      # driftless: near a coin flip
        True
        >>> report.touch_thresholds["-10%"] >= report.terminal_thresholds["-10%"]
        True
    """
    terminal = ensemble.terminal
    spot = ensemble.spot
    returns = ensemble.terminal_returns
    simple = terminal / spot - 1.0

    terminal_thresholds: dict[str, float] = {}
    touch_thresholds: dict[str, float] = {}
    for move in moves:
        up_level, down_level = spot * (1 + move), spot * (1 - move)
        terminal_thresholds[f"+{move:.0%}"] = _share(terminal >= up_level)
        terminal_thresholds[f"-{move:.0%}"] = _share(terminal <= down_level)
        touch_thresholds[f"+{move:.0%}"] = first_passage_probability(
            ensemble, up_level, direction="up"
        )
        touch_thresholds[f"-{move:.0%}"] = first_passage_probability(
            ensemble, down_level, direction="down"
        )

    losses = simple[simple < 0]
    gains = simple[simple > 0]
    tail = np.quantile(simple, 0.05)

    horizon_years = ensemble.horizon / 252.0
    annualised = (
        float(np.std(returns, ddof=1) / np.sqrt(horizon_years))
        if horizon_years > 0
        else float("nan")
    )

    return ProbabilityReport(
        symbol=symbol or "instrument",
        spot=spot,
        horizon_days=ensemble.horizon,
        engine=str(ensemble.assumptions.get("engine", ensemble.engine)),
        probability_up=_share(terminal > spot),
        terminal_thresholds=terminal_thresholds,
        touch_thresholds=touch_thresholds,
        drawdowns={
            f"{depth:.0%}": drawdown_probability(ensemble, depth) for depth in drawdown_depths
        },
        median_return=float(np.median(simple)),
        median_worst_drawdown=float(
            -np.median(
                np.min(ensemble.paths / np.maximum.accumulate(ensemble.paths, axis=1) - 1.0, axis=1)
            )
        ),
        expected_shortfall_95=float(np.mean(simple[simple <= tail]))
        if simple.size
        else float("nan"),
        expected_gain_given_gain=float(np.mean(gains)) if gains.size else float("nan"),
        expected_loss_given_loss=float(np.mean(losses)) if losses.size else float("nan"),
        terminal_quantiles={
            level: float(np.quantile(terminal, level))
            for level in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
        },
        annualised_volatility=annualised,
        notes=list(ensemble.notes),
    )


@dataclass
class LongShortComparison:
    """The same forecast, seen from both sides of the trade."""

    symbol: str
    horizon_days: int
    borrow_cost_annual: float

    long_probability_of_profit: float
    short_probability_of_profit: float
    long_expected_shortfall_95: float
    short_expected_shortfall_95: float
    long_worst_case: float
    short_worst_case: float
    long_stop_hit_probability: float
    short_stop_hit_probability: float
    stop_distance: float

    @property
    def asymmetry_verdict(self) -> str:
        return (
            f"The short's worst simulated outcome is {self.short_worst_case:+.1%} against "
            f"the long's {self.long_worst_case:+.1%}. A short's loss is unbounded while "
            "its gain caps at 100%, the fat tail in equity returns points upward in log "
            f"space, and borrow accrues at {self.borrow_cost_annual:.1%} a year whether or "
            "not the view is right. Equal-sized positions are not equal risk."
        )

    def summary(self) -> str:
        rows = [
            (
                "probability of profit",
                self.long_probability_of_profit,
                self.short_probability_of_profit,
            ),
            (
                "average of worst 5%",
                self.long_expected_shortfall_95,
                self.short_expected_shortfall_95,
            ),
            ("worst simulated path", self.long_worst_case, self.short_worst_case),
            (
                f"chance of a {self.stop_distance:.0%} stop being hit",
                self.long_stop_hit_probability,
                self.short_stop_hit_probability,
            ),
        ]
        lines = [
            f"{self.symbol} — long versus short over {self.horizon_days} days",
            f"  {'':38s} {'LONG':>10s} {'SHORT':>10s}",
        ]
        lines += [f"  {label:38s} {long:9.1%} {short:9.1%}" for label, long, short in rows]
        lines += ["", f"  {self.asymmetry_verdict}"]
        return "\n".join(lines)


def long_short_comparison(
    ensemble: PathEnsemble,
    *,
    symbol: str = "",
    borrow_cost_annual: float = 0.005,
    stop_distance: float = 0.08,
) -> LongShortComparison:
    r"""Compare the risk of the long and short side of the same forecast.

    Method
        The short's return is :math:`-(S_T/S_0 - 1)` less accrued borrow, and its
        stop is a rise rather than a fall. Both are computed from the same paths,
        so the difference between them is entirely the asymmetry of the position
        rather than any difference in view.
    Inputs
        ``borrow_cost_annual`` -- a plausible general-collateral rate. Hard-to-
        borrow names cost far more, and the cost is the least predictable part of
        a short.

    Example
        >>> import numpy as np
        >>> from quantos.forecast.paths import simulate_garch_paths
        >>> rng = np.random.default_rng(3)
        >>> r = rng.normal(0, 0.025, 1200)
        >>> e = simulate_garch_paths(r, 100.0, 63, n_paths=5000)
        >>> c = long_short_comparison(e, symbol="TEST")
        >>> c.short_worst_case < c.long_worst_case   # the short's tail is worse
        True
    """
    spot = ensemble.spot
    horizon_years = ensemble.horizon / 252.0
    long_returns = ensemble.terminal / spot - 1.0
    short_returns = -long_returns - borrow_cost_annual * horizon_years

    long_tail = np.quantile(long_returns, 0.05)
    short_tail = np.quantile(short_returns, 0.05)

    return LongShortComparison(
        symbol=symbol or "instrument",
        horizon_days=ensemble.horizon,
        borrow_cost_annual=borrow_cost_annual,
        long_probability_of_profit=_share(long_returns > 0),
        short_probability_of_profit=_share(short_returns > 0),
        long_expected_shortfall_95=float(np.mean(long_returns[long_returns <= long_tail])),
        short_expected_shortfall_95=float(np.mean(short_returns[short_returns <= short_tail])),
        long_worst_case=float(np.min(long_returns)),
        short_worst_case=float(np.min(short_returns)),
        long_stop_hit_probability=first_passage_probability(
            ensemble, spot * (1 - stop_distance), direction="down"
        ),
        short_stop_hit_probability=first_passage_probability(
            ensemble, spot * (1 + stop_distance), direction="up"
        ),
        stop_distance=stop_distance,
    )
