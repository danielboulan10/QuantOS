r"""Measuring whether a simulated market looks like a real one.

The validation problem
----------------------
An agent-based market simulator is trivially easy to write and very hard to
*validate*. Anyone can produce a price series; the question is whether it has
the statistical signature of a real market. The literature has converged on a
list of "stylised facts" -- robust, model-free properties observed across
essentially every liquid market, asset class and epoch. A simulator that
reproduces them is doing something right; one that does not is generating
noise with a plausible-looking chart.

The critical point is that **none of these properties is programmed in**. No
agent in :mod:`quantos.sim.agents` is told to produce fat tails or clustered
volatility. If they appear, they are emergent consequences of the interaction
between order-book mechanics, inventory-averse liquidity provision, adverse
selection and trend-following. That is what makes the measurement meaningful
rather than circular.

The facts measured here
-----------------------
=========================================  ===================================
1. Returns are **not** normally distributed  Excess kurtosis > 0; JB rejects
2. Tails follow a **power law**, index ~3    Hill estimator in [2, 5]
3. Returns are **nearly uncorrelated**       ACF of r within noise bands
4. **Volatility clusters**                   ACF of |r| positive, slow decay
5. Volatility is **long-memory**             Hurst exponent of |r| > 0.5
6. **Leverage effect**                       corr(r_t, |r_{t+k}|) < 0
7. **Aggregational Gaussianity**             kurtosis falls as horizon grows
8. **Volume-volatility correlation**         positive
=========================================  ===================================

Fact 3 combined with fact 4 is the discriminating pair: it is easy to build a
simulator with fat tails but no clustering (draw i.i.d. from a t-distribution),
and easy to build one with clustering but predictable returns (a naive momentum
loop). Having both simultaneously requires the volatility dynamics to be
genuine while leaving the *direction* unpredictable, which is a real constraint.

References
----------
Cont, R. (2001), "Empirical properties of asset returns: stylized facts and
    statistical issues", *Quantitative Finance* 1(2), 223-236.
Gopikrishnan, P. et al. (1999), "Scaling of the distribution of fluctuations of
    financial market indices", *Phys. Rev. E* 60(5), 5305-5316.
Bouchaud, J.-P., Matacz, A. & Potters, M. (2001), "Leverage effect in financial
    markets", *Phys. Rev. Lett.* 87, 228701.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["StylizedFact", "StylizedFactsReport", "analyse_stylized_facts", "hurst_exponent"]


@dataclass(frozen=True)
class StylizedFact:
    """One measured property, with the criterion it was judged against."""

    name: str
    value: float
    passed: bool
    criterion: str
    detail: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.value:.4f}   ({self.criterion})"


@dataclass(frozen=True)
class StylizedFactsReport:
    """The full battery of stylised-fact measurements."""

    facts: tuple[StylizedFact, ...]
    n_returns: int

    @property
    def n_passed(self) -> int:
        return sum(1 for f in self.facts if f.passed)

    @property
    def score(self) -> float:
        """Fraction of facts reproduced, in ``[0, 1]``."""
        return self.n_passed / len(self.facts) if self.facts else 0.0

    def __getitem__(self, name: str) -> StylizedFact:
        for fact in self.facts:
            if fact.name == name:
                return fact
        raise KeyError(name)

    def __str__(self) -> str:  # pragma: no cover - display
        header = (
            f"Stylised facts: {self.n_passed}/{len(self.facts)} reproduced "
            f"(n={self.n_returns} returns)"
        )
        return "\n".join([header, "-" * len(header), *(str(f) for f in self.facts)])


def hurst_exponent(x: ArrayLike, *, min_window: int = 8, n_windows: int = 20) -> float:
    r"""Hurst exponent by rescaled-range (R/S) analysis.

    .. math:: \mathbb{E}[R(n)/S(n)] \sim c\, n^{H}

    so :math:`H` is the slope of :math:`\log(R/S)` against :math:`\log n`.

    Interpretation: :math:`H = 0.5` is a random walk (independent increments);
    :math:`H > 0.5` indicates long-memory persistence; :math:`H < 0.5`
    anti-persistence. Applied to :math:`|r|` this is the standard evidence for
    long-memory *volatility*, with empirical estimates around 0.7-0.9. Applied
    to raw returns it should come out near 0.5 -- and if it does not, the
    simulated market has exploitable serial dependence, which is a bug in the
    agent population rather than a feature.

    Known limitation
        R/S is biased upward in small samples and sensitive to the window
        range. It is used here as a *comparative* diagnostic (returns versus
        absolute returns on the same sample), which cancels most of the bias,
        not as a precise estimate.
    """
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    if n < 64:
        return float("nan")

    max_window = n // 2
    if max_window <= min_window:
        return float("nan")
    windows = np.unique(np.geomspace(min_window, max_window, num=n_windows).astype(int))

    logs_n: list[float] = []
    logs_rs: list[float] = []
    for w in windows:
        n_chunks = n // w
        if n_chunks < 1:
            continue
        chunks = a[: n_chunks * w].reshape(n_chunks, w)
        deviations = np.cumsum(chunks - chunks.mean(axis=1, keepdims=True), axis=1)
        ranges = deviations.max(axis=1) - deviations.min(axis=1)
        stds = chunks.std(axis=1, ddof=0)
        valid = stds > 0
        if not np.any(valid):
            continue
        logs_n.append(float(np.log(w)))
        logs_rs.append(float(np.log(np.mean(ranges[valid] / stds[valid]))))

    if len(logs_n) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.asarray(logs_n), np.asarray(logs_rs), 1)
    return float(slope)


def analyse_stylized_facts(
    returns: ArrayLike,
    *,
    volumes: ArrayLike | None = None,
    acf_lags: int = 50,
    leverage_lags: int = 20,
) -> StylizedFactsReport:
    r"""Run the full stylised-facts battery on a return series.

    Purpose
        Answer "does this simulated market look real?" with numbers rather than
        by eyeballing a chart.
    Inputs
        ``returns`` -- log returns at the finest available frequency.
        ``volumes`` -- optional per-period volume, enabling fact 8.
    Outputs
        :class:`StylizedFactsReport`.
    Failure modes
        Raises if fewer than 500 returns are supplied: several of these
        estimators (Hill, Hurst, the ACF noise bands) are meaningless on short
        samples, and returning a confident-looking report from 100 observations
        would be worse than refusing.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # A GARCH process should reproduce facts 1, 3 and 4 by construction.
        >>> n = 20000
        >>> r = np.zeros(n); v = np.full(n, 1e-4)
        >>> for t in range(1, n):
        ...     v[t] = 1e-6 + 0.1 * r[t-1]**2 + 0.88 * v[t-1]
        ...     r[t] = np.sqrt(v[t]) * rng.standard_normal()
        >>> report = analyse_stylized_facts(r)
        >>> report["fat_tails"].passed, report["volatility_clustering"].passed
        (True, True)
    """
    from quantos.core.stats.descriptive import autocorrelation, hill_estimator, kurtosis
    from quantos.core.stats.hypothesis import jarque_bera

    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 500:
        raise ValueError(f"need at least 500 returns for a meaningful report, got {r.size}")
    if np.std(r) == 0:
        raise ValueError("returns are constant; nothing to measure")

    facts: list[StylizedFact] = []
    absolute = np.abs(r)
    lags = min(acf_lags, r.size // 4)

    # --- Fact 1: heavy tails ------------------------------------------------
    excess_kurt = kurtosis(r, excess=True)
    jb = jarque_bera(r)
    facts.append(
        StylizedFact(
            "fat_tails",
            float(excess_kurt),
            passed=bool(excess_kurt > 0.5),
            criterion="excess kurtosis > 0.5",
            detail={"jarque_bera_p": jb.p_value, "jarque_bera_stat": jb.statistic},
        )
    )

    # --- Fact 2: power-law tail index ---------------------------------------
    tail_index = hill_estimator(absolute[absolute > 0])
    facts.append(
        StylizedFact(
            "power_law_tail",
            float(tail_index),
            passed=bool(2.0 <= tail_index <= 5.0),
            criterion="Hill tail index in [2, 5] (empirical ~3)",
            detail={"n_positive": float(np.sum(absolute > 0))},
        )
    )

    # --- Fact 3: returns nearly uncorrelated --------------------------------
    acf_returns = autocorrelation(r, lags)[1:]
    # Bartlett's band for white noise.
    band = 1.96 / np.sqrt(r.size)
    fraction_outside = float(np.mean(np.abs(acf_returns) > band))
    facts.append(
        StylizedFact(
            "uncorrelated_returns",
            fraction_outside,
            # Under the null, 5% should exceed the 95% band by chance; allow
            # generous slack for the multiplicity across lags.
            passed=bool(fraction_outside < 0.25),
            criterion="<25% of return ACF lags outside the 95% noise band",
            detail={
                "mean_abs_acf": float(np.mean(np.abs(acf_returns))),
                "acf_lag1": float(acf_returns[0]),
                "noise_band": float(band),
            },
        )
    )

    # --- Fact 4: volatility clustering --------------------------------------
    acf_absolute = autocorrelation(absolute, lags)[1:]
    mean_abs_acf = float(np.mean(acf_absolute))
    facts.append(
        StylizedFact(
            "volatility_clustering",
            mean_abs_acf,
            passed=bool(mean_abs_acf > band),
            criterion="mean ACF of |returns| exceeds the noise band",
            detail={
                "acf_abs_lag1": float(acf_absolute[0]),
                "acf_abs_lag10": float(acf_absolute[min(9, lags - 2)]),
                "noise_band": float(band),
            },
        )
    )

    # --- Fact 5: long memory in volatility ----------------------------------
    hurst_abs = hurst_exponent(absolute)
    hurst_raw = hurst_exponent(r)
    facts.append(
        StylizedFact(
            "long_memory_volatility",
            float(hurst_abs),
            passed=bool(np.isfinite(hurst_abs) and hurst_abs > 0.55),
            criterion="Hurst exponent of |returns| > 0.55",
            detail={"hurst_raw_returns": float(hurst_raw)},
        )
    )

    # --- Fact 6: leverage effect --------------------------------------------
    k = min(leverage_lags, r.size // 10)
    correlations = [float(np.corrcoef(r[:-j], absolute[j:])[0, 1]) for j in range(1, k + 1)]
    leverage = float(np.mean(correlations))
    # Require *significance*, not merely a negative sign. A sign test on a
    # correlation that is pure noise passes half the time by construction --
    # during development, i.i.d. Gaussian returns "reproduced" the leverage
    # effect at -0.0014. Note that a symmetric GARCH also correctly fails this:
    # the leverage effect needs an asymmetric volatility response (GJR), and a
    # simulated market exhibiting it must be getting it from agent behaviour.
    leverage_threshold = -1.96 / np.sqrt(r.size)
    facts.append(
        StylizedFact(
            "leverage_effect",
            leverage,
            passed=bool(leverage < leverage_threshold),
            criterion=(
                f"mean corr(r_t, |r_(t+k)|) < {leverage_threshold:.4f}, i.e. significantly negative"
            ),
            detail={
                "corr_lag1": correlations[0] if correlations else float("nan"),
                "threshold": float(leverage_threshold),
            },
        )
    )

    # --- Fact 7: aggregational Gaussianity ----------------------------------
    # Note on interpretation: this fact converges *slowly* when volatility is
    # highly persistent, because aggregated returns inherit the clustering. A
    # GARCH process with alpha+beta = 0.98 legitimately shows kurtosis still
    # rising at h=20 on a 30k sample. So a failure here is often a correct
    # statement about persistence rather than a defect in the market -- read it
    # alongside `long_memory_volatility`.
    horizons = [1, 10, 50]
    kurtoses: dict[str, float] = {}
    for h in horizons:
        if r.size // h < 200:
            continue
        aggregated = r[: (r.size // h) * h].reshape(-1, h).sum(axis=1)
        kurtoses[f"kurtosis_h{h}"] = float(kurtosis(aggregated, excess=True))
    values = list(kurtoses.values())
    # Only meaningful if there was excess kurtosis at the base horizon to lose.
    applicable = len(values) >= 2 and values[0] > 0.5
    facts.append(
        StylizedFact(
            "aggregational_gaussianity",
            float(values[-1]) if values else float("nan"),
            passed=bool(applicable and values[-1] < values[0]),
            criterion=(
                "excess kurtosis decreases with aggregation horizon"
                if applicable
                else "not applicable: no excess kurtosis at the base horizon"
            ),
            detail=kurtoses,
        )
    )

    # --- Fact 8: volume-volatility correlation ------------------------------
    if volumes is not None:
        v = np.asarray(volumes, dtype=float).ravel()
        m = min(v.size, absolute.size)
        if m > 100 and np.std(v[:m]) > 0:
            corr = float(np.corrcoef(v[:m], absolute[:m])[0, 1])
            facts.append(
                StylizedFact(
                    "volume_volatility_correlation",
                    corr,
                    passed=bool(corr > 0.0),
                    criterion="corr(volume, |return|) > 0",
                )
            )

    return StylizedFactsReport(facts=tuple(facts), n_returns=int(r.size))
