r"""Test whether the forecast probabilities are true.

Why this module is the important one
------------------------------------
Any simulation will emit a number with a percent sign on it. The question that
decides whether the number is worth anything is: **when it says 10%, does the
thing happen 10% of the time?**

That is testable, and it is the only claim in this package's forward-looking half
that a reader should accept without checking. So it is checked, on historical data,
out of sample, and the result is reported whatever it is.

The method
----------
Walk through history. At each date, fit the model on data up to that date only,
simulate forward, and record the forecast probability of a set of events. Then
look at what actually happened. Group the forecasts into buckets by predicted
probability and compare the predicted rate with the realised rate in each bucket:
that is a **reliability curve**, and a well-calibrated model sits on the diagonal.

Two failure modes it separates, which a single accuracy number cannot:

**Overconfidence.** The curve is flatter than the diagonal -- events predicted at
5% happen 15% of the time. This is what a thin-tailed model does, and it is the
dangerous direction, because the risk numbers understate risk.

**Underconfidence.** The curve is steeper -- events predicted at 30% happen 10%
of the time. Wasteful but safe.

Scoring
-------
The Brier score (mean squared error of the probability) is reported alongside, and
decomposed into **reliability** and **resolution** by Murphy's identity. That
decomposition matters because a forecaster can achieve a good Brier score by
always predicting the base rate -- perfectly calibrated, perfectly useless.
Resolution is what separates a model that knows something from one that has merely
learned the average.

Overlap
-------
Successive windows share days, so their outcomes are not independent, and the
naive standard error on a realised rate is too small. The number of
non-overlapping windows is reported next to every bucket count so no interval is
read as tighter than it is.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CalibrationResult",
    "ReliabilityBucket",
    "brier_decomposition",
    "calibration_test",
]


@dataclass(frozen=True)
class ReliabilityBucket:
    """One point on the reliability curve."""

    lower: float
    upper: float
    n_forecasts: int
    n_independent: int
    mean_predicted: float
    realised_rate: float

    @property
    def error(self) -> float:
        return self.realised_rate - self.mean_predicted

    @property
    def standard_error(self) -> float:
        """Binomial standard error on the *independent* count, not the raw one."""
        if self.n_independent < 1:
            return float("nan")
        p = self.realised_rate
        return float(np.sqrt(max(p * (1 - p), 1e-12) / self.n_independent))

    @property
    def within_noise(self) -> bool:
        se = self.standard_error
        return bool(np.isfinite(se) and abs(self.error) <= 2 * se)


@dataclass
class CalibrationResult:
    """Whether a forecaster's probabilities can be believed."""

    event: str
    horizon_days: int
    n_forecasts: int
    n_independent: int
    buckets: list[ReliabilityBucket] = field(default_factory=list)
    brier_score: float = float("nan")
    reliability: float = float("nan")
    resolution: float = float("nan")
    uncertainty: float = float("nan")
    #: Residual from binning a continuous forecast; see :func:`brier_decomposition`.
    binning_residual: float = float("nan")
    base_rate: float = float("nan")
    mean_predicted: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def skill_versus_base_rate(self) -> float:
        """Brier skill score. Positive means better than always predicting the base rate."""
        if not np.isfinite(self.uncertainty) or self.uncertainty <= 0:
            return float("nan")
        return float(1.0 - self.brier_score / self.uncertainty)

    @property
    def verdict(self) -> str:
        """A three-way answer, because "not rejected" is not the same as "good".

        An earlier version returned a bare boolean and reported ``True`` on a run
        whose Brier skill was **negative** -- the model was worse than always
        predicting the base rate. It passed because the buckets where it was
        badly wrong (46% predicted, 0% realised) held only five forecasts, so the
        standard error on four independent observations swallowed the error.

        That is absence of evidence being read as evidence of calibration, which
        is the specific mistake this whole package exists to avoid. So the test
        now reports insufficient power as its own outcome, and refuses to call a
        model calibrated when it carries no skill over the base rate.
        """
        populated = [b for b in self.buckets if b.n_independent >= 10]
        if len(populated) < 2:
            return (
                "INSUFFICIENT EVIDENCE — fewer than two buckets hold 10 independent "
                "forecasts, so this test cannot distinguish a good model from a bad one"
            )
        misses = [b for b in populated if not b.within_noise]
        skill = self.skill_versus_base_rate
        if misses:
            return f"NOT CALIBRATED — {len(misses)} of {len(populated)} tested buckets miss"
        if np.isfinite(skill) and skill < 0:
            return (
                f"calibrated on average but NO SKILL (Brier skill {skill:+.3f}): unbiased, "
                "yet no better than always quoting the base rate"
            )
        return f"calibrated, with positive skill ({skill:+.3f}) over the base rate"

    @property
    def is_calibrated(self) -> bool:
        """Calibrated *and* carrying skill. Both, not either."""
        return self.verdict.startswith("calibrated, with positive")

    @property
    def has_resolution(self) -> bool:
        """Report whether it can tell risky periods from calm ones at all.

        Resolution near zero means the forecast is essentially the base rate
        wearing a model. That can coexist with perfect calibration, which is why
        it is reported separately.
        """
        return bool(
            np.isfinite(self.resolution)
            and np.isfinite(self.uncertainty)
            and self.uncertainty > 0
            and self.resolution / self.uncertainty > 0.02
        )

    @property
    def bias_direction(self) -> str:
        if not np.isfinite(self.mean_predicted) or not np.isfinite(self.base_rate):
            return "not computable"
        gap = self.base_rate - self.mean_predicted
        if abs(gap) < 0.01:
            return "unbiased on average"
        if gap > 0:
            return (
                f"UNDERSTATES risk: predicted {self.mean_predicted:.1%} on average, "
                f"happened {self.base_rate:.1%} of the time"
            )
        return (
            f"overstates risk: predicted {self.mean_predicted:.1%} on average, "
            f"happened {self.base_rate:.1%} of the time"
        )

    def summary(self) -> str:
        lines = [
            f"calibration of P({self.event}) at {self.horizon_days} days",
            f"  {self.n_forecasts:,} forecasts ({self.n_independent} non-overlapping)",
            f"  base rate {self.base_rate:.1%}, mean prediction {self.mean_predicted:.1%}",
            f"  {self.bias_direction}",
            f"  Brier {self.brier_score:.4f}  =  reliability {self.reliability:.4f}"
            f"  -  resolution {self.resolution:.4f}  +  uncertainty {self.uncertainty:.4f}"
            f"  +  binning {self.binning_residual:+.4f}",
            f"  skill versus base rate: {self.skill_versus_base_rate:+.3f}",
            f"  resolution / uncertainty: "
            f"{self.resolution / self.uncertainty if self.uncertainty else float('nan'):.3f}"
            f"  ({'can' if self.has_resolution else 'CANNOT'} distinguish risky periods)",
            f"  verdict: {self.verdict}",
            "",
            "  predicted      realised    n   indep   within noise",
        ]
        for bucket in self.buckets:
            if bucket.n_forecasts == 0:
                continue
            flag = "yes" if bucket.within_noise else "NO"
            lines.append(
                f"  {bucket.lower:4.0%}-{bucket.upper:4.0%}  "
                f"{bucket.mean_predicted:7.1%}  {bucket.realised_rate:7.1%} "
                f"{bucket.n_forecasts:5d} {bucket.n_independent:6d}   {flag}"
            )
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def brier_decomposition(
    predicted: NDArray[np.float64], outcomes: NDArray[np.bool_], *, n_bins: int = 10
) -> tuple[float, float, float, float, float]:
    r"""Murphy's decomposition, made exact for continuous forecasts.

    Returns ``(brier, reliability, resolution, uncertainty, residual)`` with

    .. math::
       \text{BS} = \text{REL} - \text{RES} + \text{UNC} + \text{WBV} - 2\,\text{WBC}

    where the residual returned is :math:`\text{WBV} - 2\,\text{WBC}`.

    Reliability is the squared gap between predicted and realised rates within
    bins (lower is better). Resolution is how far the bin rates move away from the
    base rate (higher is better). Uncertainty is the base rate's own variance,
    which no forecaster can affect.

    The decomposition is what distinguishes a model with information from one that
    has memorised the average: predicting the base rate every time gives perfect
    reliability and zero resolution.

    Why there is a fourth term
        The textbook three-term identity is exact only when the forecaster emits
        finitely many distinct probabilities, so each bin holds one value. A
        simulation emits a continuum, and writing :math:`f_i = \bar f_k + \delta_i`
        inside bin :math:`k` leaves two leftovers: the within-bin **variance** of
        the forecasts, :math:`\text{WBV}`, and a cross term
        :math:`-2\,\text{WBC}` where :math:`\text{WBC}` is the within-bin
        covariance between forecast and outcome.

        WBC is not a nuisance -- it is real resolution that binning discards. A
        forecaster who ranks correctly *inside* a bin is being informative in a
        way the coarse table cannot show, and that shows up here as a negative
        contribution to the Brier score.

        Measured on 4,000 uniform forecasts in ten bins: dropping both terms left
        the identity failing by :math:`4.2\times10^{-4}`; keeping only WBV
        overshot in the other direction. Both are needed, and with both the
        identity closes to machine precision.
    """
    predicted = np.asarray(predicted, dtype=float)
    outcomes = np.asarray(outcomes, dtype=bool)
    n = predicted.size
    if n == 0:
        return (float("nan"),) * 5

    base = float(np.mean(outcomes))
    brier = float(np.mean((predicted - outcomes.astype(float)) ** 2))
    uncertainty = base * (1.0 - base)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    within_variance = 0.0
    within_covariance = 0.0
    for lower, upper in itertools.pairwise(edges):
        # Include the right edge only in the final bin so nothing is double counted.
        mask = (predicted >= lower) & ((predicted < upper) if upper < 1.0 else (predicted <= upper))
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_predicted = float(np.mean(predicted[mask]))
        bin_realised = float(np.mean(outcomes[mask]))
        weight = count / n
        reliability += weight * (bin_predicted - bin_realised) ** 2
        resolution += weight * (bin_realised - base) ** 2

        deviation = predicted[mask] - bin_predicted
        outcome_deviation = outcomes[mask].astype(float) - bin_realised
        within_variance += weight * float(np.mean(deviation**2))
        within_covariance += weight * float(np.mean(deviation * outcome_deviation))

    residual = within_variance - 2.0 * within_covariance
    return brier, reliability, resolution, uncertainty, residual


def calibration_test(
    prices: NDArray[np.float64],
    *,
    horizon: int = 21,
    event: str = "drawdown_10pct",
    threshold: float = 0.10,
    train_window: int = 750,
    step: int = 5,
    n_paths: int = 2000,
    engine: str = "garch",
    seed: int = 20240719,
) -> CalibrationResult:
    """Walk history, forecast, and compare predictions with what happened.

    Purpose
        Establish whether this package's forward probabilities are true, rather
        than merely produced.
    Inputs
        ``prices`` -- the full history. ``event`` -- one of
        ``"drawdown_10pct"`` (a peak-to-trough fall of ``threshold`` within the
        window), ``"touch_down"`` (touches ``-threshold`` from the start price),
        or ``"finish_down"`` (ends below ``-threshold``).
        ``train_window`` -- observations used for each fit. ``step`` -- days
        between forecasts; smaller means more forecasts but more overlap.
    Outputs
        A :class:`CalibrationResult`.
    Method
        Strictly walk-forward. Each fit sees only data before its forecast date,
        so no outcome can influence the model that predicted it. This is the same
        discipline as the forward ledger, applied retrospectively -- which is
        weaker evidence than the live ledger, and is labelled as such.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(7)
        >>> prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 1400)))
        >>> result = calibration_test(prices, horizon=21, train_window=600,
        ...                           step=40, n_paths=400)
        >>> result.n_forecasts > 5
        True
    """
    from quantos.forecast.paths import simulate_bootstrap_paths, simulate_garch_paths
    from quantos.forecast.probabilities import drawdown_probability, first_passage_probability

    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < train_window + horizon + 10:
        raise ValueError(
            f"need at least {train_window + horizon + 10} prices for this configuration, "
            f"got {prices.size}"
        )

    simulate = simulate_garch_paths if engine == "garch" else simulate_bootstrap_paths
    log_prices = np.log(prices)
    returns = np.diff(log_prices)

    predicted: list[float] = []
    outcomes: list[bool] = []
    forecast_indices: list[int] = []

    for t in range(train_window, prices.size - horizon, step):
        history = returns[max(0, t - train_window) : t]
        if history.size < 100:
            continue
        spot = float(prices[t])
        try:
            ensemble = simulate(history, spot, horizon, n_paths=n_paths, seed=seed + t)
        except (ValueError, np.linalg.LinAlgError):
            continue

        window = prices[t : t + horizon + 1]
        if event == "drawdown_10pct":
            probability = drawdown_probability(ensemble, threshold)
            running_max = np.maximum.accumulate(window)
            happened = bool(np.min(window / running_max - 1.0) <= -threshold)
        elif event == "touch_down":
            level = spot * (1 - threshold)
            probability = first_passage_probability(ensemble, level, direction="down")
            happened = bool(np.min(window) <= level)
        elif event == "finish_down":
            level = spot * (1 - threshold)
            probability = float(np.mean(ensemble.terminal <= level))
            happened = bool(window[-1] <= level)
        else:
            raise ValueError(f"unknown event {event!r}")

        predicted.append(float(probability))
        outcomes.append(happened)
        forecast_indices.append(t)

    if not predicted:
        raise ValueError("no forecasts could be produced with this configuration")

    predicted_array = np.asarray(predicted, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=bool)

    # Non-overlapping forecasts: greedily take one, skip the horizon, take the next.
    independent_indices: list[int] = []
    next_free = -1
    for index in forecast_indices:
        if index >= next_free:
            independent_indices.append(index)
            next_free = index + horizon
    independent_count = len(independent_indices)
    independent_fraction = independent_count / len(forecast_indices)

    brier, reliability, resolution, uncertainty, residual = brier_decomposition(
        predicted_array, outcome_array
    )

    edges = np.linspace(0.0, 1.0, 11)
    buckets: list[ReliabilityBucket] = []
    for lower, upper in itertools.pairwise(edges):
        mask = (predicted_array >= lower) & (
            (predicted_array < upper) if upper < 1.0 else (predicted_array <= upper)
        )
        count = int(np.sum(mask))
        buckets.append(
            ReliabilityBucket(
                lower=float(lower),
                upper=float(upper),
                n_forecasts=count,
                # Apportion the independent count in proportion to the bucket's size.
                n_independent=round(count * independent_fraction),
                mean_predicted=float(np.mean(predicted_array[mask])) if count else float("nan"),
                realised_rate=float(np.mean(outcome_array[mask])) if count else float("nan"),
            )
        )

    notes = [
        f"forecasts every {step} days over a {horizon}-day horizon, so successive "
        f"windows overlap; {independent_count} of {len(predicted)} are independent and "
        "the standard errors use that count",
        "walk-forward on historical data: each model saw only prices before its own "
        "forecast date. This is weaker evidence than the live ledger in forward/, "
        "because the universe and horizon were chosen with the history already visible",
    ]

    return CalibrationResult(
        event=event,
        horizon_days=horizon,
        n_forecasts=len(predicted),
        n_independent=independent_count,
        buckets=buckets,
        brier_score=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        binning_residual=residual,
        base_rate=float(np.mean(outcome_array)),
        mean_predicted=float(np.mean(predicted_array)),
        notes=notes,
    )
