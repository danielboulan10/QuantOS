r"""Generate a thousand factors, test every one, and report what survives.

The usual pitch for a factor lab is "generate a thousand signals, test each,
publish the best." That pitch describes a machine for producing false
discoveries, and the arithmetic of why is not subtle. Search a thousand
independent worthless signals at the 5% level and roughly fifty come back
"significant". The best of the thousand will show a t-statistic near 3.2 purely
from the extreme-value behaviour of the maximum. Publish it and you have
published noise, with a confidence interval attached to make it look otherwise.

So this module does generate the thousand factors -- systematically, from a
grammar rather than by hand, so the search is reproducible and its *size* is
known exactly, which is the number every correction needs. Then it applies the
machinery already in this repository to the whole search rather than to the
winner:

- :func:`~quantos.core.stats.multipletest.whites_reality_check` tests the null
  that *no* factor in the universe has skill, correctly accounting for the fact
  that the best was selected from many.
- :func:`~quantos.core.stats.multipletest.hansen_spa` does the same with
  studentisation, so a high-variance factor cannot win on noise alone.
- :func:`~quantos.core.stats.multipletest.stepm` identifies *which* factors
  survive, not just whether any do.
- :func:`~quantos.strategy.validation.deflated_sharpe_ratio` asks the question
  that matters about the winner specifically: given that it was chosen as the
  best of N, is its Sharpe distinguishable from what luck would produce?

The expected verdict on real single-instrument data is that nothing survives.
That is the result, not a failure of the lab. A search this size over a series
this short cannot support a discovery, and reporting so is the entire point.
The uncorrected best t-statistic is reported *alongside* the corrected verdict
precisely so the size of the illusion is visible.

Example
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> returns = rng.standard_normal(1500) * 0.01
    >>> report = run_factor_lab(returns, n_factors=200, seed=1)
    >>> report.best.t_statistic > 2.0            # the winner looks significant
    True
    >>> report.survivors                          # ... and nothing survives
    []
    >>> "no factor survives" in report.summary()
    True

References
----------
    White (2000), "A Reality Check for Data Snooping", *Econometrica* 68(5).
    Hansen (2005), "A Test for Superior Predictive Ability", *JBES* 23(4).
    Harvey, Liu & Zhu (2016), "... and the Cross-Section of Expected Returns",
    *Review of Financial Studies* 29(1) -- which argues that a t-statistic of 2
    is far too weak a bar once the published search is accounted for, and
    proposes roughly 3.0 as a minimum.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.special import ndtr
from quantos.core.stats.multipletest import hansen_spa, stepm, whites_reality_check
from quantos.strategy.validation import deflated_sharpe_ratio

__all__ = [
    "FactorResult",
    "FactorSpec",
    "LabReport",
    "compute_signal",
    "evaluate_factor",
    "generate_factor_grid",
    "run_factor_lab",
]

# --------------------------------------------------------------------------- #
# The grammar
#
# Factors are generated from a small grammar rather than written out, for two
# reasons. The search becomes reproducible from a seed and a spec, and its size
# is known exactly -- and the *size of the search* is the input every correction
# below needs. A hand-written list of "the fifty factors I tried" is always an
# undercount, because the ones that were tried and abandoned never make the list.
# That undercount is the single most common way a corrected result is still
# wrong.
# --------------------------------------------------------------------------- #

#: Base transforms applied to the return series. Each maps a window of past
#: returns to a scalar signal, using only information available at that point.
TRANSFORMS = (
    "momentum",  # cumulative return over the window
    "reversal",  # negated recent return
    "volatility",  # realised volatility, as a risk-off signal
    "vol_change",  # change in volatility between two windows
    "skew",  # third moment, sign-carrying
    "kurtosis",  # tail activity
    "trend",  # slope of a fitted line through the log path
    "acceleration",  # difference of two momentum windows
    "drawdown",  # distance below the running maximum
    "up_ratio",  # fraction of positive days
)

#: Lookback windows in trading days, spanning a week to roughly a year.
WINDOWS = (5, 10, 21, 42, 63, 126, 252)

#: Post-processing applied to the raw signal.
SCALINGS = ("raw", "zscore", "rank", "sign")

#: Holding periods in trading days.
HOLDS = (1, 5, 21)


@dataclass(frozen=True)
class FactorSpec:
    """One point in the search space. Fully determines the signal."""

    transform: str
    window: int
    scaling: str
    hold: int

    @property
    def name(self) -> str:
        return f"{self.transform}_{self.window}d_{self.scaling}_h{self.hold}"


@dataclass(frozen=True)
class FactorResult:
    """What one factor achieved, before any correction for the search."""

    spec: FactorSpec
    mean: float
    sharpe: float
    t_statistic: float
    #: Naive two-sided p-value, ignoring that many factors were tried. Reported
    #: only so the gap against the corrected verdict is visible.
    naive_p_value: float
    n_observations: int
    turnover: float

    @property
    def naively_significant(self) -> bool:
        return self.naive_p_value < 0.05


@dataclass
class LabReport:
    """The result of the whole search, corrected for the whole search."""

    n_factors: int
    n_observations: int
    results: list[FactorResult]
    best: FactorResult
    #: Factors surviving Romano-Wolf StepM at the stated level. Usually empty.
    survivors: list[str] = field(default_factory=list)
    reality_check_p: float = float("nan")
    spa_p: float = float("nan")
    deflated_sharpe_p: float = float("nan")
    #: The same test with the trial variance estimated under the null instead of
    #: taken from the search. The two disagree sharply when the search contains
    #: real skill -- see the note in :func:`run_factor_lab`.
    deflated_sharpe_p_null_variance: float = float("nan")
    n_naively_significant: int = 0
    expected_false_positives: float = 0.0
    alpha: float = 0.05
    notes: list[str] = field(default_factory=list)

    @property
    def any_survivor(self) -> bool:
        return bool(self.survivors)

    def summary(self) -> str:
        lines = [
            f"FACTOR LAB -- {self.n_factors:,} factors over {self.n_observations:,} observations",
            "=" * 66,
            "",
            "Best factor, uncorrected:",
            f"  {self.best.spec.name}",
            f"  Sharpe {self.best.sharpe:6.2f}    "
            f"t = {self.best.t_statistic:5.2f}    "
            f"naive p = {self.best.naive_p_value:.4f}",
            "",
            f"  {self.n_naively_significant} of {self.n_factors} factors clear "
            f"p < {self.alpha:.2f} on their own.",
            f"  Pure chance predicts about {self.expected_false_positives:.0f}.",
            "",
            "Corrected for the size of the search:",
            f"  White's Reality Check   p = {self.reality_check_p:.4f}",
            f"  Hansen's SPA            p = {self.spa_p:.4f}",
            f"  Deflated Sharpe         p = {self.deflated_sharpe_p:.4f}"
            f"   (null variance: {self.deflated_sharpe_p_null_variance:.4f})",
            "",
        ]

        if self.survivors:
            lines.append(f"  SURVIVORS ({len(self.survivors)}) after Romano-Wolf StepM:")
            lines.extend(f"    {name}" for name in self.survivors)
        else:
            lines.append("  VERDICT: no factor survives the correction.")

        lines.extend(["", *(f"  {note}" for note in self.notes)])
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Generating and computing
# --------------------------------------------------------------------------- #
def generate_factor_grid(
    n_factors: int | None = None, *, seed: int | None = None
) -> list[FactorSpec]:
    """Enumerate the search space, optionally subsampling to ``n_factors``.

    The full grid is |TRANSFORMS| x |WINDOWS| x |SCALINGS| x |HOLDS| = 840
    specifications. Subsampling is done with a seeded generator so the search is
    reproducible: an unreproducible search cannot be corrected for, because its
    size is unknown.

    Example
        >>> grid = generate_factor_grid()
        >>> len(grid)
        840
        >>> generate_factor_grid(10, seed=0)[0].name
        'drawdown_21d_sign_h5'
    """
    full = [
        FactorSpec(transform, window, scaling, hold)
        for transform, window, scaling, hold in product(TRANSFORMS, WINDOWS, SCALINGS, HOLDS)
    ]
    if n_factors is None or n_factors >= len(full):
        return full

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(full), size=n_factors, replace=False)
    return [full[index] for index in chosen]


def _rolling(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Stack of trailing windows, one row per observation, NaN before warm-up."""
    n = values.size
    out = np.full((n, window), np.nan)
    for offset in range(window):
        out[window - 1 :, offset] = values[offset : n - window + 1 + offset]
    return out


def compute_signal(returns: NDArray[np.float64], spec: FactorSpec) -> NDArray[np.float64]:
    """Signal at each point, using only returns strictly before that point.

    The look-ahead discipline is the whole game. Every window ends at ``t-1``
    and the resulting position is applied to the return at ``t``; the shift is
    applied once, here, rather than left to each caller to remember.
    """
    windows = _rolling(returns, spec.window)
    transform = spec.transform

    # Warm-up rows are all-NaN by construction, so every nan-aware reduction
    # warns about an empty slice. Expected here and only here; suppressed at the
    # site rather than with a global filter, which would also hide real ones.
    with (
        np.errstate(invalid="ignore", divide="ignore"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        if transform == "momentum":
            raw = np.nansum(windows, axis=1)
        elif transform == "reversal":
            raw = -np.nansum(windows[:, -min(5, spec.window) :], axis=1)
        elif transform == "volatility":
            raw = -np.nanstd(windows, axis=1)
        elif transform == "vol_change":
            half = max(2, spec.window // 2)
            raw = np.nanstd(windows[:, -half:], axis=1) - np.nanstd(windows[:, :half], axis=1)
            raw = -raw
        elif transform == "skew":
            centred = windows - np.nanmean(windows, axis=1, keepdims=True)
            scale = np.nanstd(windows, axis=1)
            raw = np.nanmean(centred**3, axis=1) / np.maximum(scale**3, 1e-12)
        elif transform == "kurtosis":
            centred = windows - np.nanmean(windows, axis=1, keepdims=True)
            scale = np.nanstd(windows, axis=1)
            raw = -(np.nanmean(centred**4, axis=1) / np.maximum(scale**4, 1e-12))
        elif transform == "trend":
            x = np.arange(spec.window, dtype=float)
            x = x - x.mean()
            path = np.nancumsum(windows, axis=1)
            raw = (path * x).sum(axis=1) / max((x**2).sum(), 1e-12)
        elif transform == "acceleration":
            half = max(2, spec.window // 2)
            raw = np.nansum(windows[:, -half:], axis=1) - np.nansum(windows[:, :half], axis=1)
        elif transform == "drawdown":
            path = np.nancumsum(windows, axis=1)
            raw = path[:, -1] - np.nanmax(path, axis=1)
        elif transform == "up_ratio":
            raw = np.nanmean(windows > 0, axis=1) - 0.5
        else:  # pragma: no cover - the grammar is closed
            raise ValueError(f"unknown transform {transform!r}")

    signal = _scale(raw, spec.scaling)

    if spec.hold > 1:
        # Hold the position: average the signal over the holding period, which
        # is what actually trading it at that frequency would produce.
        held = _rolling(signal, spec.hold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            signal = np.nanmean(held, axis=1)

    # The single shift that makes the whole thing out of sample.
    shifted = np.full_like(signal, np.nan)
    shifted[1:] = signal[:-1]
    return shifted


def _scale(raw: NDArray[np.float64], scaling: str) -> NDArray[np.float64]:
    """Post-process a raw signal, expanding-window so nothing looks ahead."""
    if scaling == "raw":
        return raw
    if scaling == "sign":
        return np.sign(raw)

    finite = np.isfinite(raw)
    out = np.full_like(raw, np.nan)
    if not finite.any():
        return out

    # Expanding statistics: at each point, use only what was knowable then. A
    # full-sample z-score would leak the future into every observation, which is
    # the most common look-ahead bug in factor research and does not announce
    # itself -- the signal simply looks better than it is.
    values = np.where(finite, raw, np.nan)
    count = np.cumsum(finite)
    total = np.nancumsum(values)
    mean = total / np.maximum(count, 1)

    if scaling == "zscore":
        total_sq = np.nancumsum(values**2)
        variance = np.divide(total_sq, np.maximum(count, 1)) - mean**2
        out = np.divide(values - mean, np.sqrt(np.maximum(variance, 1e-24)))
        return np.where(count > 20, out, np.nan)

    if scaling == "rank":
        # Expanding percentile rank. O(n^2) in the worst case, but n is a few
        # thousand and correctness matters more than speed here.
        ranks = np.full_like(raw, np.nan)
        seen: list[float] = []
        for index, value in enumerate(values):
            if np.isfinite(value):
                seen.append(float(value))
                if len(seen) > 20:
                    below = sum(1 for other in seen if other < value)
                    ranks[index] = 2.0 * below / len(seen) - 1.0
        return ranks

    raise ValueError(f"unknown scaling {scaling!r}")  # pragma: no cover


def evaluate_factor(
    returns: NDArray[np.float64], spec: FactorSpec, *, periods_per_year: float = 252.0
) -> tuple[FactorResult, NDArray[np.float64]]:
    """Score one factor, returning both the summary and the per-period P&L.

    The P&L series is returned because the multiple-testing tests need the whole
    matrix, not the summaries -- they resample it.
    """
    signal = compute_signal(returns, spec)
    usable = np.isfinite(signal) & np.isfinite(returns)

    pnl = np.zeros_like(returns)
    pnl[usable] = signal[usable] * returns[usable]

    active = pnl[usable]
    n = active.size
    if n < 30 or not np.any(active != 0.0):
        empty = FactorResult(spec, 0.0, 0.0, 0.0, 1.0, n, 0.0)
        return empty, pnl

    mean = float(np.mean(active))
    sd = float(np.std(active, ddof=1))
    if sd <= 0:
        empty = FactorResult(spec, mean, 0.0, 0.0, 1.0, n, 0.0)
        return empty, pnl

    t_statistic = mean / (sd / np.sqrt(n))
    sharpe = mean / sd * np.sqrt(periods_per_year)
    p_value = 2.0 * (1.0 - float(ndtr(abs(t_statistic))))

    positions = signal[usable]
    turnover = float(np.mean(np.abs(np.diff(positions)))) if positions.size > 1 else 0.0

    return (
        FactorResult(spec, mean, sharpe, float(t_statistic), p_value, n, turnover),
        pnl,
    )


# --------------------------------------------------------------------------- #
# The lab
# --------------------------------------------------------------------------- #
def run_factor_lab(
    returns: ArrayLike,
    *,
    n_factors: int | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    periods_per_year: float = 252.0,
    seed: int | None = None,
) -> LabReport:
    """Run the whole search and correct for the whole search.

    Args:
        returns: the instrument's return series.
        n_factors: how many of the 840-point grid to search. ``None`` uses all.
        alpha: family-wise error rate for StepM.
        n_bootstrap: resamples for the Reality Check, SPA and StepM.
        periods_per_year: annualisation factor for the reported Sharpe.
        seed: makes the subsample and the bootstrap reproducible.

    Returns
    -------
        A :class:`LabReport`. Read ``survivors`` before ``best``: the second is
        what the search found, the first is what the evidence supports.
    """
    values = np.asarray(returns, dtype=float).ravel()
    if values.size < 300:
        raise ValueError(
            f"a factor search over {values.size} observations cannot support a "
            "conclusion; at least 300 are needed for the block bootstrap to mean "
            "anything"
        )

    specs = generate_factor_grid(n_factors, seed=seed)
    rng = np.random.default_rng(seed)

    results: list[FactorResult] = []
    pnl_columns: list[NDArray[np.float64]] = []
    for spec in specs:
        result, pnl = evaluate_factor(values, spec, periods_per_year=periods_per_year)
        results.append(result)
        pnl_columns.append(pnl)

    performance = np.column_stack(pnl_columns)

    # Put every factor on a common risk basis before comparing them.
    #
    # This was missing in the first version and it silently broke the whole
    # comparison. The grammar produces P&L series spanning a 1,900x range in
    # standard deviation -- a `raw`-scaled momentum signal is a number like
    # 0.4, a `sign`-scaled one is exactly +/-1 -- so a max-of-means statistic
    # across the columns ranks LEVERAGE, not skill. With a signal planted
    # deliberately, the true factor ranked 143rd of 200 by P&L scale and the
    # tests could not see it.
    #
    # Dividing each column by its own standard deviation is a per-column
    # constant. It leaves every factor's own t-statistic and Sharpe untouched --
    # those are already scale-free -- and only makes the cross-factor comparison
    # about the thing being compared. The constant is computed in sample, which
    # is worth stating plainly: it uses the whole series. It cannot manufacture
    # skill, because it is a single positive multiplier per column that cannot
    # change the sign or the timing of anything, but it is not a quantity a
    # trader would have known in advance either.
    scale = np.std(performance, axis=0)
    performance = performance / np.where(scale > 0, scale, 1.0)
    # Trim the warm-up: the longest window plus the longest hold is dead for
    # every factor, and leaving those zeros in would dilute every mean equally
    # while making the bootstrap resample rows that carry no information.
    warmup = max(spec.window + spec.hold for spec in specs) + 1
    performance = performance[warmup:]

    best_index = int(np.argmax([result.t_statistic for result in results]))
    best = results[best_index]

    reality = whites_reality_check(performance, n_bootstrap=n_bootstrap, rng=rng)
    spa = hansen_spa(performance, n_bootstrap=n_bootstrap, rng=np.random.default_rng(seed))
    step = stepm(performance, alpha=alpha, n_bootstrap=n_bootstrap, rng=np.random.default_rng(seed))

    survivors = [specs[index].name for index in np.flatnonzero(step)]

    best_pnl = pnl_columns[best_index]
    best_active = best_pnl[np.isfinite(best_pnl) & (best_pnl != 0.0)]
    # The deflated Sharpe works in PER-PERIOD units, so the variance of trial
    # Sharpes must be given in per-period units too. Passing the variance of the
    # annualised Sharpes -- which is what FactorResult.sharpe stores -- inflates
    # it by `periods_per_year` and puts the expected maximum at an impossible
    # 4.53 per-period, which returned p = 1.0000 on a factor with t = 11.4. The
    # unit mismatch is invisible in the output: a p-value of 1 looks like a
    # verdict rather than a bug.
    trial_sharpes = np.array([result.sharpe for result in results], dtype=float)
    trial_variance = float(np.var(trial_sharpes, ddof=1)) / periods_per_year
    deflated = deflated_sharpe_ratio(
        best_active,
        n_trials=len(specs),
        trial_sharpe_variance=trial_variance,
        periods_per_year=periods_per_year,
    )
    # Both variants are reported because they disagree in an instructive way.
    # The deflated Sharpe deflates against the expected maximum of `n_trials`
    # draws UNDER THE NULL of no skill. Feeding it the observed spread of trial
    # Sharpes therefore assumes that spread is pure noise -- and when the search
    # actually contains something, the skill itself inflates the spread and
    # raises the bar the winner must clear.
    #
    # Measured on a deliberately planted signal: the true factor has t = 11.4
    # and a per-period Sharpe of 0.257, the search-derived variance puts the
    # expected null maximum at 0.285, and the test returns p = 0.90. Estimating
    # the variance under the null instead gives 0.062 and p < 0.0001. The
    # search-derived version is self-defeating in exactly the case where there
    # is something to find, and a p-value of 0.90 looks like a verdict rather
    # than a broken assumption.
    deflated_null = deflated_sharpe_ratio(
        best_active, n_trials=len(specs), periods_per_year=periods_per_year
    )

    naive_hits = sum(1 for result in results if result.naively_significant)
    expected = alpha * len(specs)

    notes = [
        f"Searching {len(specs):,} factors and reporting the best without correction "
        f"would have produced a Sharpe of {best.sharpe:.2f} (t = {best.t_statistic:.2f}).",
        "The corrected tests above ask a different question: whether ANY factor in "
        "the searched universe has skill, given that the best of many was selected.",
    ]
    if not survivors:
        notes.append(
            "Nothing survives. That is the expected outcome of a search this size "
            "over a series this short, and it is the result -- not a failure to "
            "find one."
        )
    else:
        notes.append(
            f"{len(survivors)} factor(s) survive family-wise correction at "
            f"alpha={alpha:.2f}. Treat this as a hypothesis to test out of sample, "
            "not a finding: StepM controls the error rate of THIS search, and says "
            "nothing about a series it has not seen."
        )
    if not survivors and min(reality.p_value, spa.p_value) < alpha:
        notes.append(
            "Note the disagreement: a global test rejects 'no factor has skill' at "
            f"p = {min(reality.p_value, spa.p_value):.3f}, yet StepM identifies no "
            "individual factor. These are different questions and both answers are "
            "correct. Evidence that SOMETHING in a large correlated family works is "
            "much cheaper to obtain than evidence about WHICH one, and only the "
            "second is tradeable."
        )
    if deflated.p_value > 0.5 > deflated_null.p_value:
        notes.append(
            "The two deflated-Sharpe figures disagree because the search contains "
            "skill: the deflation uses the spread of trial Sharpes as if it were "
            "pure noise, so real skill inflates the very benchmark the winner is "
            "measured against. Read the null-variance figure here, and prefer the "
            "search-derived one when the search is genuinely fishing."
        )
    notes.append(
        "Harvey, Liu & Zhu (2016) argue that t > 3.0 is the right bar for a "
        "published factor once the profession's collective search is counted. "
        f"This search alone was {len(specs):,} wide."
    )

    return LabReport(
        n_factors=len(specs),
        n_observations=int(performance.shape[0]),
        results=results,
        best=best,
        survivors=survivors,
        reality_check_p=float(reality.p_value),
        spa_p=float(spa.p_value),
        deflated_sharpe_p=float(deflated.p_value),
        deflated_sharpe_p_null_variance=float(deflated_null.p_value),
        n_naively_significant=naive_hits,
        expected_false_positives=expected,
        alpha=alpha,
        notes=notes,
    )
