r"""Test whether a published Value-at-Risk number is actually true.

The gap this closes
-------------------
Every research page in this repository reports VaR and CVaR. Nothing tested
whether those numbers hold. That is the same gap
:mod:`quantos.forecast.calibration` closed for the forward probabilities, left
open for the risk figures -- and a 99% VaR that is breached 4% of the time is not
a conservative estimate, it is a wrong one.

A VaR model is a sequence of *predictions*: on each day it claims the loss will
not exceed a threshold, with probability :math:`1-\alpha`. That is falsifiable
against what happened, and there is a standard way to do it.

Two tests, because one is not enough
-------------------------------------
**Kupiec unconditional coverage.** Are there the right *number* of exceedances?
A likelihood ratio against the binomial null:

.. math::
   LR_{uc} = -2\log\frac{(1-p)^{n-x} p^{x}}
                        {(1-\hat\pi)^{n-x}\hat\pi^{x}}, \quad \hat\pi = x/n

distributed :math:`\chi^2_1`. This is the test everyone runs.

**Christoffersen independence.** Are the exceedances *clustered*? This is the one
that matters and the one people skip. A 99% VaR breached exactly 1% of the time,
but with every breach falling in the same fortnight, has passed Kupiec and is
still useless -- that pattern is precisely a risk model that fails when it is
needed. The test is a likelihood ratio against a first-order Markov chain: under
a correct model an exceedance today says nothing about an exceedance tomorrow.

The two combine into a **conditional coverage** statistic,
:math:`LR_{cc} = LR_{uc} + LR_{ind}`, distributed :math:`\chi^2_2`.

Why the historical estimator needs help
----------------------------------------
A historical VaR cannot produce a loss larger than the worst one in its window,
which is a strange property for a tail measure: it is most confident exactly
where it has least information. Extreme value theory replaces the empirical tail
with a fitted generalised Pareto distribution
(:func:`fit_generalised_pareto`), which *can* extrapolate beyond the sample --
and reports a shape parameter that says how heavy the tail is rather than
assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ExceedanceTest",
    "GParetoFit",
    "VarBacktest",
    "backtest_var",
    "christoffersen_independence",
    "evt_value_at_risk",
    "fit_generalised_pareto",
    "kupiec_coverage",
]


def _chi2_sf(statistic: float, degrees: int) -> float:
    """Upper tail of a chi-square, via this package's incomplete gamma."""
    from quantos.core.special import gammaincc

    if not np.isfinite(statistic) or statistic < 0:
        return float("nan")
    return float(gammaincc(np.array(degrees / 2.0), np.array(statistic / 2.0)))


@dataclass(frozen=True)
class ExceedanceTest:
    """One likelihood-ratio test on a sequence of VaR breaches."""

    name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    null_hypothesis: str
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def rejects(self) -> bool:
        """Rejected at 5%. A rejection means the VaR model is wrong."""
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    def summary(self) -> str:
        verdict = "REJECTED" if self.rejects else "not rejected"
        return f"{self.name:28s} LR {self.statistic:8.3f}  p {self.p_value:7.4f}  {verdict}"


def kupiec_coverage(exceedances: NDArray[np.bool_], confidence: float) -> ExceedanceTest:
    r"""Test whether the *number* of VaR breaches matches the model's claim.

    Inputs
        ``exceedances`` -- boolean array, ``True`` where the loss exceeded VaR.
        ``confidence`` -- e.g. ``0.99`` for a 99% VaR, so the expected breach
        rate is 1%.
    Outputs
        An :class:`ExceedanceTest`; a small p-value means the count is wrong in
        one direction or the other.
    Failure modes
        With zero exceedances the likelihood ratio is still defined and the test
        correctly rejects when the sample is long enough -- a VaR never breached
        in 2,000 days at 99% is too conservative, not perfect.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> honest = rng.random(4000) < 0.01     # exactly the promised rate
        >>> bool(kupiec_coverage(honest, 0.99).rejects)
        False
        >>> broken = rng.random(4000) < 0.05     # five times too many
        >>> bool(kupiec_coverage(broken, 0.99).rejects)
        True
    """
    breaches = np.asarray(exceedances, dtype=bool)
    n = int(breaches.size)
    if n == 0:
        return ExceedanceTest(
            "Kupiec unconditional", float("nan"), float("nan"), 1, "breach rate equals 1 - c"
        )

    x = int(np.sum(breaches))
    p = 1.0 - confidence
    observed = x / n

    def log_likelihood(rate: float) -> float:
        # Guard the boundary: 0*log(0) is 0 here, not a NaN.
        first = (n - x) * np.log(1 - rate) if (n - x) > 0 and rate < 1 else 0.0
        second = x * np.log(rate) if x > 0 and rate > 0 else 0.0
        return float(first + second)

    if observed in (0.0, 1.0):
        statistic = -2.0 * (
            log_likelihood(p) - log_likelihood(min(max(observed, 1e-12), 1 - 1e-12))
        )
    else:
        statistic = -2.0 * (log_likelihood(p) - log_likelihood(observed))

    return ExceedanceTest(
        name="Kupiec unconditional",
        statistic=float(statistic),
        p_value=_chi2_sf(statistic, 1),
        degrees_of_freedom=1,
        null_hypothesis=f"breach rate equals {p:.1%}",
        detail={
            "observed_rate": observed,
            "expected_rate": p,
            "n_breaches": float(x),
            "n": float(n),
        },
    )


def christoffersen_independence(exceedances: NDArray[np.bool_]) -> ExceedanceTest:
    r"""Test whether breaches *cluster* -- the failure Kupiec cannot see.

    Under a correct model, an exceedance today carries no information about
    tomorrow. The alternative is a first-order Markov chain with transition
    probabilities :math:`\pi_{01}` (breach after a calm day) and :math:`\pi_{11}`
    (breach after a breach); independence means they are equal.

    Why this matters more than the count
        A 99% VaR breached exactly 20 times in 2,000 days has passed Kupiec
        perfectly. If all 20 fell in one month, the model was wrong for a month
        and right the rest of the time -- which is the only period anyone cares
        about. Clustering is the signature of a risk model that under-reacts to
        volatility regimes, and it is invisible to a count.
    """
    breaches = np.asarray(exceedances, dtype=bool)
    if breaches.size < 2:
        return ExceedanceTest(
            "Christoffersen independence",
            float("nan"),
            float("nan"),
            1,
            "breaches are independent",
        )

    previous, current = breaches[:-1], breaches[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))

    # Transition probabilities under the alternative, and the pooled rate.
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pooled = (n01 + n11) / (n00 + n01 + n10 + n11)

    def term(count: int, probability: float) -> float:
        return count * np.log(probability) if count > 0 and probability > 0 else 0.0

    null = term(n00 + n10, 1 - pooled) + term(n01 + n11, pooled)
    alternative = term(n00, 1 - pi01) + term(n01, pi01) + term(n10, 1 - pi11) + term(n11, pi11)
    statistic = -2.0 * (null - alternative)
    # Numerical noise can make this a hair negative when the fits coincide.
    statistic = max(float(statistic), 0.0)

    return ExceedanceTest(
        name="Christoffersen independence",
        statistic=statistic,
        p_value=_chi2_sf(statistic, 1),
        degrees_of_freedom=1,
        null_hypothesis="breaches are independent of yesterday",
        detail={
            "p_breach_after_calm": pi01,
            "p_breach_after_breach": pi11,
            "n00": float(n00),
            "n01": float(n01),
            "n10": float(n10),
            "n11": float(n11),
        },
    )


@dataclass(frozen=True)
class GParetoFit:
    r"""A generalised Pareto fitted to the tail beyond a threshold."""

    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    n_total: int

    @property
    def tail_fraction(self) -> float:
        return self.n_exceedances / self.n_total if self.n_total else float("nan")

    @property
    def tail_verdict(self) -> str:
        r"""What the shape parameter :math:`\xi` says about the tail.

        Positive shape means a power-law tail with infinite moments above order
        :math:`1/\xi` -- at :math:`\xi > 0.5` the *variance* of the loss
        distribution does not exist, which makes every volatility-based risk
        number a statement about a quantity that is not there.
        """
        if not np.isfinite(self.shape):
            return "not fitted"
        if self.shape <= 0:
            return f"shape {self.shape:+.3f}: thin, bounded tail"
        moments = 1.0 / self.shape
        return (
            f"shape {self.shape:+.3f}: heavy power-law tail, moments beyond order "
            f"{moments:.1f} do not exist"
        )

    def quantile(self, confidence: float, *, exceedance_rate: float | None = None) -> float:
        r"""The VaR implied by the fitted tail.

        .. math:: \text{VaR}_q = u + \frac{\sigma}{\xi}
                  \left[\left(\frac{n}{N_u}(1-q)\right)^{-\xi} - 1\right]

        Unlike a historical quantile this can exceed every loss in the sample,
        which is the entire reason for fitting a tail rather than reading one off.
        """
        rate = exceedance_rate if exceedance_rate is not None else self.tail_fraction
        if not np.isfinite(rate) or rate <= 0:
            return float("nan")
        ratio = (1.0 - confidence) / rate
        if abs(self.shape) < 1e-8:
            return float(self.threshold - self.scale * np.log(ratio))
        return float(self.threshold + self.scale / self.shape * (ratio ** (-self.shape) - 1.0))


def fit_generalised_pareto(
    losses: NDArray[np.float64], *, threshold_quantile: float = 0.90
) -> GParetoFit:
    r"""Fit a generalised Pareto to losses beyond a high threshold.

    Method
        Peaks-over-threshold with probability-weighted moments rather than
        maximum likelihood. PWM is closed-form, cannot fail to converge, and is
        more stable in small samples -- and tail samples are always small, which
        is the whole difficulty. ML has lower asymptotic variance and routinely
        fails to converge on a few dozen exceedances.
    Inputs
        ``losses`` -- positive numbers, larger meaning worse.
    Failure modes
        Fewer than 30 exceedances returns NaN parameters rather than a fit
        nobody should trust.
    """
    losses = np.asarray(losses, dtype=float)
    losses = losses[np.isfinite(losses)]
    if losses.size < 50:
        return GParetoFit(float("nan"), float("nan"), float("nan"), 0, int(losses.size))

    threshold = float(np.quantile(losses, threshold_quantile))
    excess = np.sort(losses[losses > threshold] - threshold)
    if excess.size < 30:
        return GParetoFit(float("nan"), float("nan"), threshold, int(excess.size), int(losses.size))

    n = excess.size
    # Probability-weighted moments: a0 = mean, a1 = E[X(1-F)].
    weights = (np.arange(1, n + 1) - 0.35) / n
    a0 = float(np.mean(excess))
    a1 = float(np.mean(excess * (1.0 - weights)))

    denominator = a0 - 2.0 * a1
    if abs(denominator) < 1e-15:
        return GParetoFit(float("nan"), float("nan"), threshold, n, int(losses.size))

    shape = 2.0 - a0 / denominator
    scale = 2.0 * a0 * a1 / denominator

    return GParetoFit(
        shape=float(shape),
        scale=float(scale),
        threshold=threshold,
        n_exceedances=n,
        n_total=int(losses.size),
    )


def evt_value_at_risk(
    returns: NDArray[np.float64], *, confidence: float = 0.99, threshold_quantile: float = 0.90
) -> float:
    """VaR from a fitted tail rather than an empirical quantile.

    The historical estimator cannot return a loss larger than the worst one it
    has seen. This one can, because it fits a shape to the tail and extrapolates
    along it.
    """
    losses = -np.asarray(returns, dtype=float)
    fit = fit_generalised_pareto(losses, threshold_quantile=threshold_quantile)
    if not np.isfinite(fit.shape):
        return float("nan")
    return fit.quantile(confidence)


@dataclass
class VarBacktest:
    """Everything measured about one VaR model on one series."""

    model: str
    confidence: float
    n_observations: int
    n_breaches: int
    expected_breaches: float
    kupiec: ExceedanceTest
    independence: ExceedanceTest
    conditional_coverage: ExceedanceTest
    #: Mean loss on the days VaR was breached, against what CVaR promised.
    mean_breach_loss: float = float("nan")
    predicted_cvar: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def breach_rate(self) -> float:
        return self.n_breaches / self.n_observations if self.n_observations else float("nan")

    @property
    def verdict(self) -> str:
        if self.kupiec.rejects and self.independence.rejects:
            return "FAILS both: wrong number of breaches, and they cluster"
        if self.kupiec.rejects:
            direction = "too many" if self.breach_rate > 1 - self.confidence else "too few"
            return f"FAILS coverage: {direction} breaches ({self.breach_rate:.2%})"
        if self.independence.rejects:
            return (
                "passes the count but FAILS independence: breaches cluster, so the model "
                "is wrong precisely when it is needed"
            )
        return "passes both tests"

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.model} at {self.confidence:.0%} over {self.n_observations:,} days",
                f"  breaches {self.n_breaches} vs {self.expected_breaches:.1f} expected "
                f"({self.breach_rate:.2%} against {1 - self.confidence:.2%})",
                f"  {self.kupiec.summary()}",
                f"  {self.independence.summary()}",
                f"  {self.conditional_coverage.summary()}",
                f"  -> {self.verdict}",
            ]
        )


def backtest_var(
    returns: NDArray[np.float64],
    var_forecasts: NDArray[np.float64],
    *,
    confidence: float = 0.99,
    model: str = "",
    cvar_forecasts: NDArray[np.float64] | None = None,
) -> VarBacktest:
    """Run the full exceedance battery on a VaR forecast series.

    Inputs
        ``returns`` -- realised returns, aligned with ``var_forecasts``.
        ``var_forecasts`` -- the VaR *as a positive loss magnitude* for each day,
        made before that day's return was known.
    Outputs
        A :class:`VarBacktest` reporting coverage, independence and their
        combination.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> r = rng.normal(0, 0.01, 3000)
        >>> # A correctly specified Gaussian VaR on Gaussian data.
        >>> var = np.full(3000, 0.01 * 2.3263)
        >>> result = backtest_var(r, var, confidence=0.99)
        >>> bool(result.kupiec.rejects)
        False
    """
    returns = np.asarray(returns, dtype=float)
    forecasts = np.asarray(var_forecasts, dtype=float)
    n = min(returns.size, forecasts.size)
    returns, forecasts = returns[:n], forecasts[:n]

    usable = np.isfinite(returns) & np.isfinite(forecasts) & (forecasts > 0)
    returns, forecasts = returns[usable], forecasts[usable]

    breaches = -returns > forecasts
    kupiec = kupiec_coverage(breaches, confidence)
    independence = christoffersen_independence(breaches)

    combined = float(kupiec.statistic + independence.statistic)
    conditional = ExceedanceTest(
        name="Christoffersen conditional",
        statistic=combined,
        p_value=_chi2_sf(combined, 2),
        degrees_of_freedom=2,
        null_hypothesis="correct coverage AND independence",
    )

    notes: list[str] = []
    if returns.size < 250:
        notes.append(
            f"only {returns.size} observations: at {1 - confidence:.0%} the expected breach "
            f"count is {returns.size * (1 - confidence):.1f}, too few for these tests to "
            "have power"
        )

    mean_breach = float(np.mean(-returns[breaches])) if np.any(breaches) else float("nan")
    predicted_cvar = float("nan")
    if cvar_forecasts is not None:
        cvar = np.asarray(cvar_forecasts, dtype=float)[usable]
        predicted_cvar = float(np.mean(cvar[breaches])) if np.any(breaches) else float("nan")

    return VarBacktest(
        model=model or "unnamed",
        confidence=confidence,
        n_observations=int(returns.size),
        n_breaches=int(np.sum(breaches)),
        expected_breaches=float(returns.size * (1 - confidence)),
        kupiec=kupiec,
        independence=independence,
        conditional_coverage=conditional,
        mean_breach_loss=mean_breach,
        predicted_cvar=predicted_cvar,
        notes=notes,
    )
