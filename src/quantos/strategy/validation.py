r"""Guarding against backtest overfitting.

The premise
-----------
A backtested Sharpe ratio is not an estimate of future performance. It is the
*maximum* over however many configurations were tried, and the maximum of a set
of noisy estimates is biased upward by an amount that grows with the number of
trials. Bailey and Lopez de Prado show that with 100 trials on 5 years of daily
data, a strategy with **zero** true skill produces an expected best in-sample
Sharpe ratio above 1.5. Report that number without adjustment and it looks like
a discovery.

This module implements the corrections. They are not optional extras -- a
backtest result presented without them carries almost no information.

============================================  ================================
:func:`deflated_sharpe_ratio`                 Adjust for trials, skew, kurtosis
:func:`minimum_track_record_length`           How long until significance?
:func:`probability_of_backtest_overfitting`    CSCV: does IS rank predict OOS?
:class:`PurgedKFold`                          CV without leakage
:class:`CombinatorialPurgedCV`                Many backtest paths, not one
:func:`walk_forward_splits`                   Expanding/rolling anchored splits
============================================  ================================

Why standard k-fold cross-validation is invalid here
----------------------------------------------------
Three separate problems, each fatal:

1. **Serial correlation.** Adjacent observations are dependent, so a training
   sample containing :math:`t-1` and a test sample containing :math:`t` are not
   independent. :class:`PurgedKFold` *purges* training observations whose label
   horizon overlaps the test set.
2. **Overlapping labels.** A 20-day forward return observed daily means each
   label draws on 20 days of future data. Any training observation within 20 days
   of a test observation leaks. Purging removes exactly those.
3. **Serial dependence after the test set.** Even non-overlapping labels leak
   through autocorrelation immediately after the test window, so an *embargo*
   removes a further fraction of observations.

Shuffling, which ordinary k-fold does by default, is worse still: it puts future
observations in the training set. A strategy validated by shuffled k-fold on time
series has been given the answer.

References
----------
Bailey, D. H. & Lopez de Prado, M. (2014), "The deflated Sharpe ratio",
    *J. Portfolio Management* 40(5), 94-107.
Bailey, D. H., Borwein, J., Lopez de Prado, M. & Zhu, Q. J. (2017), "The
    probability of backtest overfitting", *J. Computational Finance* 20(4).
Lopez de Prado, M. (2018), *Advances in Financial Machine Learning*, ch. 7, 11-12.
Harvey, C. R. & Liu, Y. (2015), "Backtesting", *J. Portfolio Management* 42(1).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.special import ndtr, ndtri

__all__ = [
    "CombinatorialPurgedCV",
    "DeflatedSharpeResult",
    "PBOResult",
    "PurgedKFold",
    "SharpeStatistics",
    "deflated_sharpe_ratio",
    "minimum_track_record_length",
    "probability_of_backtest_overfitting",
    "sharpe_ratio_with_moments",
    "walk_forward_splits",
]

_EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Sharpe ratio statistics                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SharpeStatistics:
    """Sharpe ratio together with the moments its standard error depends on."""

    sharpe: float
    skewness: float
    excess_kurtosis: float
    n_obs: int
    periods_per_year: float = 252.0

    @property
    def annualised(self) -> float:
        return float(self.sharpe * np.sqrt(self.periods_per_year))

    @property
    def standard_error(self) -> float:
        r"""Standard error accounting for non-normality (Lo 2002; Mertens 2002).

        .. math::
            \sigma_{\hat{SR}} = \sqrt{\frac{1 - \gamma_3 \hat{SR}
                + \frac{\gamma_4 - 1}{4}\hat{SR}^2}{n - 1}}

        The naive :math:`\sqrt{1/n}` is *wrong in the dangerous direction* for
        typical strategies: negative skew and excess kurtosis both inflate the
        true standard error, so the naive version overstates significance. A
        short-volatility strategy is precisely where the difference matters most.
        """
        if self.n_obs < 2:
            return float("nan")
        variance = (
            1.0 - self.skewness * self.sharpe + 0.25 * self.excess_kurtosis * self.sharpe**2
        ) / (self.n_obs - 1)
        return float(np.sqrt(max(variance, 0.0)))

    @property
    def t_statistic(self) -> float:
        se = self.standard_error
        return float(self.sharpe / se) if se > 0 else float("nan")

    @property
    def p_value(self) -> float:
        """One-sided p-value against the null of zero Sharpe ratio."""
        return float(1.0 - ndtr(np.array(self.t_statistic)))


def sharpe_ratio_with_moments(
    returns: ArrayLike, *, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> SharpeStatistics:
    """Sharpe ratio plus the higher moments needed for honest inference.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> stats = sharpe_ratio_with_moments(rng.standard_normal(1000) * 0.01)
        >>> bool(abs(stats.sharpe) < 0.15)
        True
    """
    from quantos.core.stats.descriptive import kurtosis, skewness

    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)] - risk_free / periods_per_year
    if r.size < 3:
        raise ValueError("need at least 3 returns")
    sd = float(np.std(r, ddof=1))
    if sd == 0:
        raise ValueError("returns have zero variance")
    return SharpeStatistics(
        sharpe=float(np.mean(r) / sd),
        skewness=skewness(r),
        excess_kurtosis=kurtosis(r, excess=True),
        n_obs=int(r.size),
        periods_per_year=periods_per_year,
    )


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated Sharpe ratio and the benchmark it was deflated against."""

    observed_sharpe: float
    #: Expected maximum Sharpe under the null of no skill, given n_trials.
    expected_maximum: float
    deflated_sharpe: float
    p_value: float
    n_trials: int
    n_obs: int

    @property
    def is_significant(self) -> bool:
        """Whether the strategy survives at the 5% level after deflation."""
        return self.p_value < 0.05


def deflated_sharpe_ratio(
    returns: ArrayLike,
    *,
    n_trials: int,
    trial_sharpe_variance: float | None = None,
    periods_per_year: float = 252.0,
) -> DeflatedSharpeResult:
    r"""Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Purpose
        Answer the only question that matters about a backtest: given that I
        selected this strategy as the best of ``n_trials``, is its Sharpe ratio
        distinguishable from what pure luck would have produced?
    Method
        Under the null of zero skill, the expected maximum of ``N`` trial Sharpe
        ratios is approximately

        .. math::
            \mathbb{E}[\max SR] \approx \sqrt{V}\left[(1-\gamma)
                \Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
                + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{Ne}\right)\right]

        with :math:`\gamma` the Euler-Mascheroni constant and :math:`V` the
        variance of Sharpe ratios across trials -- the extreme-value result for
        the maximum of ``N`` Gaussians. The observed Sharpe is then tested
        against *that* benchmark rather than against zero, using the
        non-normality-adjusted standard error from
        :attr:`SharpeStatistics.standard_error`.
    Inputs
        ``n_trials`` -- **the number of configurations actually tried**, including
        every parameter you swept and discarded. Understating it is the easiest
        way to make this test say what you want; if you cannot count the trials,
        the honest answer is that the backtest cannot be evaluated.
        ``trial_sharpe_variance`` -- variance of Sharpe across trials; defaults to
        the estimated sampling variance of the single observed track record.
    Outputs
        :class:`DeflatedSharpeResult`.
    Failure modes
        Raises for ``n_trials < 1``. With ``n_trials = 1`` this reduces to the
        ordinary non-normality-adjusted significance test.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # Five years of daily returns, 2.4 annualised Sharpe. Tested once, it
        >>> # is convincing; as the best of 500 attempts, it is not.
        >>> good = rng.standard_normal(1260) * 0.01 + 0.0015
        >>> deflated_sharpe_ratio(good, n_trials=1).is_significant
        True
        >>> deflated_sharpe_ratio(good, n_trials=500).is_significant
        False
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    stats = sharpe_ratio_with_moments(returns, periods_per_year=periods_per_year)
    variance = (
        trial_sharpe_variance if trial_sharpe_variance is not None else stats.standard_error**2
    )
    if variance <= 0:
        raise ValueError("trial Sharpe variance must be positive")

    n = float(n_trials)
    if n_trials == 1:
        expected_max = 0.0
    else:
        expected_max = float(
            np.sqrt(variance)
            * (
                (1.0 - _EULER_MASCHERONI) * float(ndtri(np.array(1.0 - 1.0 / n)))
                + _EULER_MASCHERONI * float(ndtri(np.array(1.0 - 1.0 / (n * np.e))))
            )
        )

    se = stats.standard_error
    z = (stats.sharpe - expected_max) / se if se > 0 else float("nan")
    deflated = float(ndtr(np.array(z)))

    return DeflatedSharpeResult(
        observed_sharpe=stats.sharpe,
        expected_maximum=expected_max,
        deflated_sharpe=deflated,
        p_value=float(1.0 - deflated),
        n_trials=n_trials,
        n_obs=stats.n_obs,
    )


def minimum_track_record_length(
    returns: ArrayLike, *, target_sharpe: float = 0.0, confidence: float = 0.95
) -> float:
    r"""Observations needed for the Sharpe ratio to beat ``target_sharpe``.

    .. math::
        \text{MinTRL} = 1 + \left[1 - \gamma_3 \hat{SR}
            + \frac{\gamma_4 - 1}{4}\hat{SR}^2\right]
            \left(\frac{\Phi^{-1}(\alpha)}{\hat{SR} - SR^*}\right)^2

    Purpose
        Turn "is this significant?" into the more useful "how much more data
        would I need?". Frequently the answer is decades, which is itself the
        finding -- a strategy whose edge cannot be established within the
        available history is not tradeable on the strength of its backtest,
        however good the point estimate.
    Outputs
        Required number of periods, or ``inf`` if the observed Sharpe does not
        exceed the target at all.
    """
    stats = sharpe_ratio_with_moments(returns)
    excess = stats.sharpe - target_sharpe
    if excess <= 0:
        return float("inf")
    z = float(ndtri(np.array(confidence)))
    adjustment = (
        1.0 - stats.skewness * stats.sharpe + 0.25 * stats.excess_kurtosis * stats.sharpe**2
    )
    return float(1.0 + adjustment * (z / excess) ** 2)


# --------------------------------------------------------------------------- #
# Probability of backtest overfitting                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PBOResult:
    """Combinatorially-symmetric cross-validation result."""

    pbo: float
    #: Out-of-sample performance of the in-sample-best configuration, per split.
    oos_performance: NDArray[np.float64]
    #: Logit of the OOS rank of the IS-best configuration, per split.
    logits: NDArray[np.float64]
    n_splits: int
    n_configurations: int
    #: Slope of OOS on IS performance. Negative means selection actively hurts.
    performance_degradation: float = float("nan")

    @property
    def is_overfit(self) -> bool:
        """Conventional threshold: PBO above 0.5 means selection is not informative."""
        return self.pbo > 0.5


def probability_of_backtest_overfitting(performance: ArrayLike, *, n_splits: int = 16) -> PBOResult:
    r"""Probability of Backtest Overfitting via CSCV (Bailey et al. 2017).

    Purpose
        Ask whether in-sample performance *ranking* has any out-of-sample
        predictive content. Deflation (:func:`deflated_sharpe_ratio`) asks whether
        the best result is lucky; PBO asks whether the *selection procedure*
        works at all. They are complementary, and PBO is the harsher test.
    Method
        Split the return matrix into ``S`` contiguous blocks. For every one of the
        :math:`\binom{S}{S/2}` ways to assign half the blocks to "in sample" and
        half to "out of sample":

        1. Find the configuration with the best in-sample Sharpe ratio.
        2. Record that configuration's *rank* out of sample.

        PBO is the fraction of splits where the in-sample winner lands in the
        **bottom half** out of sample. Under a working selection procedure this
        should be well below 0.5; at 0.5 the in-sample ranking is worthless; above
        0.5 it is actively misleading, which does happen and is worth knowing.

        Using combinations of blocks rather than a single split is what makes the
        estimate stable, and keeping blocks contiguous is what preserves the serial
        dependence that makes the exercise realistic.
    Inputs
        ``performance`` -- ``(T, N)`` matrix of per-period returns, one column per
        configuration.
    Outputs
        :class:`PBOResult`.
    Failure modes
        Raises for fewer than 2 configurations or fewer than ``n_splits``
        observations. ``n_splits`` above 20 is refused: the binomial coefficient
        grows explosively (:math:`\binom{24}{12} = 2.7` million).

    Sampling variability -- read this before quoting a single PBO
        PBO is itself an estimate, and a noisy one. Measured over 12 *independent*
        datasets of 50 skill-less configurations, this implementation returns a
        mean of 0.449 -- correctly centred on the theoretical 0.5 -- but the
        individual values ranged from **0.10 to 0.70**. A single PBO of 0.22 is
        therefore not evidence that a selection procedure works, and a single 0.65
        is not proof it fails.

        Two consequences for practice. Report PBO alongside
        ``performance_degradation`` (the slope of out-of-sample on in-sample
        performance), which is more stable. And where the strategy is cheap to
        re-evaluate, compute PBO on several independent samples rather than one.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # 50 configurations with no real skill. Averaged over datasets this
        >>> # centres on 0.5; on any one dataset it is dispersed, so the assertion
        >>> # below is deliberately loose.
        >>> noise = rng.standard_normal((1000, 50)) * 0.01
        >>> res = probability_of_backtest_overfitting(noise, n_splits=10)
        >>> bool(0.0 <= res.pbo <= 1.0), bool(res.performance_degradation < 0.5)
        (True, True)
    """
    perf = np.asarray(performance, dtype=float)
    if perf.ndim != 2:
        raise ValueError("performance must be a 2-D (T, N) array")
    t, n_config = perf.shape
    if n_config < 2:
        raise ValueError("need at least 2 configurations to rank")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 2")
    if n_splits > 20:
        raise ValueError("n_splits > 20 makes the number of combinations impractical")
    if t < n_splits * 2:
        raise ValueError(f"need at least {n_splits * 2} observations")

    # Contiguous blocks preserve serial dependence.
    bounds = np.linspace(0, t, n_splits + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]

    def block_sharpe(rows: NDArray[np.intp]) -> NDArray[np.float64]:
        sub = perf[rows]
        sd = np.std(sub, axis=0, ddof=1)
        mean = np.mean(sub, axis=0)
        return np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0), 0.0)

    logits: list[float] = []
    oos_values: list[float] = []
    is_values: list[float] = []
    half = n_splits // 2

    for combo in itertools.combinations(range(n_splits), half):
        in_rows = np.concatenate([blocks[i] for i in combo])
        out_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])

        is_sharpe = block_sharpe(in_rows)
        oos_sharpe = block_sharpe(out_rows)

        best = int(np.argmax(is_sharpe))
        is_values.append(float(is_sharpe[best]))
        oos_values.append(float(oos_sharpe[best]))

        # Relative rank of the chosen configuration out of sample, in (0, 1).
        rank = float(np.mean(oos_sharpe <= oos_sharpe[best]))
        rank = min(max(rank, 1.0 / (n_config + 1)), 1.0 - 1.0 / (n_config + 1))
        logits.append(float(np.log(rank / (1.0 - rank))))

    logit_array = np.asarray(logits)
    # PBO = P(logit <= 0) = P(the IS winner ranks in the bottom half OOS).
    pbo = float(np.mean(logit_array <= 0.0))

    degradation = float("nan")
    if len(is_values) > 2 and np.std(is_values) > 0:
        degradation = float(np.polyfit(np.asarray(is_values), np.asarray(oos_values), 1)[0])

    return PBOResult(
        pbo=pbo,
        oos_performance=np.asarray(oos_values),
        logits=logit_array,
        n_splits=n_splits,
        n_configurations=n_config,
        performance_degradation=degradation,
    )


# --------------------------------------------------------------------------- #
# Cross-validation splitters                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class PurgedKFold:
    r"""K-fold cross-validation with purging and an embargo.

    Purpose
        Produce train/test splits for time series with *overlapping labels*
        without leaking information from test to train.
    Method
        For each contiguous test fold:

        * **Purge.** Drop every training observation whose label interval
          overlaps the test fold's time span. With a 20-day forward return, an
          observation 15 days before the test set has a label that partly
          consists of test-period data.
        * **Embargo.** Additionally drop a fraction of observations immediately
          *after* the test fold, since serial correlation leaks even where label
          intervals do not overlap.

        Folds are never shuffled.
    Inputs
        ``n_splits`` -- number of folds. ``embargo_fraction`` -- fraction of the
        sample length to embargo after each test fold (Lopez de Prado suggests
        0.01, i.e. 1%). ``label_horizon`` -- number of periods each label spans;
        this is what drives purging.
    Outputs
        Iterator of ``(train_indices, test_indices)``.

    Example
        >>> import numpy as np
        >>> cv = PurgedKFold(n_splits=4, label_horizon=10, embargo_fraction=0.01)
        >>> splits = list(cv.split(np.arange(400)))
        >>> len(splits)
        4
        >>> train, test = splits[1]
        >>> # No training index lies within the label horizon of the test fold.
        >>> bool(np.min(np.abs(train[:, None] - test[None, :])) > 1)
        True
    """

    n_splits: int = 5
    label_horizon: int = 1
    embargo_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.label_horizon < 1:
            raise ValueError("label_horizon must be >= 1")
        if not 0.0 <= self.embargo_fraction < 0.5:
            raise ValueError("embargo_fraction must lie in [0, 0.5)")

    def split(self, x: ArrayLike) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield purged, embargoed train/test index pairs."""
        n = len(np.asarray(x))
        if n < self.n_splits * (self.label_horizon + 2):
            raise ValueError(
                f"sample of {n} is too short for {self.n_splits} folds with "
                f"label_horizon={self.label_horizon}"
            )
        indices = np.arange(n)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        embargo = int(np.ceil(n * self.embargo_fraction))

        for i in range(self.n_splits):
            test_start, test_stop = bounds[i], bounds[i + 1]
            test = indices[test_start:test_stop]

            # Purge before: any label starting within label_horizon of the test
            # start reaches into the test period.
            purge_start = max(0, test_start - self.label_horizon)
            # Purge + embargo after.
            purge_stop = min(n, test_stop + self.label_horizon + embargo)

            train = np.concatenate([indices[:purge_start], indices[purge_stop:]])
            if train.size == 0:
                continue
            yield train, test

    def get_n_splits(self) -> int:
        return self.n_splits


@dataclass
class CombinatorialPurgedCV:
    r"""Combinatorial purged cross-validation (Lopez de Prado 2018, ch. 12).

    Purpose
        A single backtest gives one performance path, and one path gives no
        estimate of the *variance* of the strategy's performance. CPCV generates
        many. With ``n_groups`` blocks and ``n_test_groups`` held out per split,
        it produces :math:`\binom{N}{k}` distinct train/test partitions, which
        combine into :math:`\binom{N}{k} k / N` complete backtest paths.
    Why it matters
        It converts "my strategy earned a 1.8 Sharpe" into a *distribution* of
        Sharpe ratios, from which a confidence interval follows. A strategy whose
        CPCV Sharpe distribution straddles zero is not established by its single
        best path, no matter how good that path looks.
    Inputs
        ``n_groups`` -- contiguous blocks. ``n_test_groups`` -- blocks held out
        per split (2 is the usual choice). Purging and embargo as
        :class:`PurgedKFold`.

    Example
        >>> import numpy as np
        >>> cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_horizon=5)
        >>> len(list(cv.split(np.arange(600))))       # C(6,2)
        15
        >>> cv.n_backtest_paths()
        5
    """

    n_groups: int = 6
    n_test_groups: int = 2
    label_horizon: int = 1
    embargo_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.n_groups < 3:
            raise ValueError("n_groups must be >= 3")
        if not 1 <= self.n_test_groups < self.n_groups:
            raise ValueError("n_test_groups must lie in [1, n_groups)")

    def n_splits(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    def n_backtest_paths(self) -> int:
        r"""Number of complete backtest paths, :math:`\binom{N}{k}k/N`."""
        return self.n_splits() * self.n_test_groups // self.n_groups

    def split(self, x: ArrayLike) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield every purged, embargoed combinatorial train/test pair."""
        n = len(np.asarray(x))
        indices = np.arange(n)
        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        groups = [indices[bounds[i] : bounds[i + 1]] for i in range(self.n_groups)]
        embargo = int(np.ceil(n * self.embargo_fraction))

        for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test = np.concatenate([groups[i] for i in combo])
            # Purge every training index near ANY held-out block, since the test
            # set is not contiguous here.
            blocked = np.zeros(n, dtype=bool)
            for i in combo:
                start = max(0, bounds[i] - self.label_horizon)
                stop = min(n, bounds[i + 1] + self.label_horizon + embargo)
                blocked[start:stop] = True
            train = indices[~blocked]
            if train.size == 0:
                continue
            yield train, test


def walk_forward_splits(
    n: int,
    *,
    n_splits: int = 5,
    min_train: int | None = None,
    anchored: bool = True,
    gap: int = 0,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    r"""Walk-forward train/test splits, expanding or rolling.

    Purpose
        The split scheme that most closely mimics live deployment: train on the
        past, test on the immediately following period, roll forward.
    Inputs
        ``anchored=True`` expands the training window from a fixed start
        (uses all history); ``False`` rolls a fixed-length window (adapts to
        regime change but discards data). ``gap`` inserts a buffer between train
        and test, serving the same purpose as the embargo in
        :class:`PurgedKFold`.
    Outputs
        List of ``(train_indices, test_indices)``, chronologically ordered.

    Example
        >>> splits = walk_forward_splits(1000, n_splits=4)
        >>> [(len(tr), len(te)) for tr, te in splits]
        [(200, 200), (400, 200), (600, 200), (800, 200)]
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    test_size = n // (n_splits + 1)
    if test_size < 1:
        raise ValueError(f"sample of {n} is too short for {n_splits} splits")
    if min_train is None:
        min_train = test_size

    out: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    indices = np.arange(n)
    for i in range(n_splits):
        train_stop = test_size * (i + 1)
        test_start = train_stop + gap
        test_stop = min(n, test_start + test_size)
        if test_start >= n or train_stop < min_train:
            continue
        train_start = 0 if anchored else max(0, train_stop - min_train)
        out.append((indices[train_start:train_stop], indices[test_start:test_stop]))
    return out
