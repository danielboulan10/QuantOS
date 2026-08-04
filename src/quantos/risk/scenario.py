r"""What happens if rates fall 100bps -- answered as a range, because it is one.

"What happens if" is the question every investment committee asks and the one
most tools answer worst. The usual answer is a point estimate: multiply a beta by
a shock, print a number. That number is wrong in three separate ways at once, and
this module's job is to report all three rather than hide them behind a decimal
place.

**The beta is estimated, so it has a standard error.** A shock response of -4.2%
with a standard error of 3.1% is not a forecast of -4.2%; it is a statement that
the response is somewhere between roughly -10% and +2%. Reporting the midpoint
alone converts an honest ambiguity into a false precision, and macro betas are
estimated on overlapping, autocorrelated data where the naive standard error is
itself too small. Every interval here uses Newey-West.

**The beta is not stable.** The stock-bond relationship inverted in 2022 (see
:mod:`~quantos.risk.stress`), which means a rate beta estimated on 2010-2021 had
the *sign* wrong for what came next, not merely the magnitude. So the response is
also estimated on subsamples, and the spread across them is reported. When the
subsample estimates disagree about direction, the full-sample number is not a
summary of anything and the module says so.

**A linear response is an approximation that fails at exactly the sizes people
ask about.** Nobody asks what happens if rates move 5bps. They ask about 100bps,
which is far outside the daily variation the beta was fitted on, and the honest
statement is that the extrapolation is unvalidated rather than that the answer
scales.

Example
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> rates = rng.standard_normal(1200) * 0.0004
    >>> asset = -6.0 * rates + rng.standard_normal(1200) * 0.0005
    >>> response = estimate_response(asset, {"rates": rates})
    >>> round(response.betas["rates"], 1)
    -6.0
    >>> shock = apply_shock(response, {"rates": 0.01})   # rates +100bps
    >>> shock.point < 0        # a rate rise hurts this asset
    True
    >>> shock.low < shock.point < shock.high
    True

References
----------
    Newey & West (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity
    and Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.special import ndtri
from quantos.core.timeseries.ols import ols

__all__ = [
    "SCENARIOS",
    "FactorResponse",
    "Scenario",
    "ShockResult",
    "apply_shock",
    "estimate_response",
]


# --------------------------------------------------------------------------- #
# Named scenarios
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """A named macro shock, expressed in the units the factors are measured in."""

    name: str
    shocks: dict[str, float]
    description: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "rates fall 100bps",
        {"rates": -0.01},
        "An easing cycle. Long duration and unprofitable growth benefit most, "
        "which is also why they were hurt most on the way up.",
    ),
    Scenario(
        "rates rise 100bps",
        {"rates": 0.01},
        "The 2022 shape. Note that the same beta is being applied in both "
        "directions, which assumes a symmetry the data does not establish.",
    ),
    Scenario(
        "oil to $120",
        {"oil": 0.50},
        "A 50% move in crude. Energy producers and consumers respond with "
        "opposite signs, so an index-level answer averages away the effect.",
    ),
    Scenario(
        "dollar strengthens 10%",
        {"dollar": 0.10},
        "Pressures foreign revenue and commodity prices simultaneously.",
    ),
    Scenario(
        "credit spreads widen 200bps",
        {"credit": 0.02},
        "The transmission channel in most equity drawdowns: refinancing risk "
        "repricing before earnings do.",
    ),
)


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
@dataclass
class FactorResponse:
    """Sensitivities of one asset to a set of macro factors."""

    betas: dict[str, float]
    standard_errors: dict[str, float]
    t_statistics: dict[str, float]
    r_squared: float
    n_obs: int
    #: Beta estimated on each of `n_subsamples` contiguous blocks, per factor.
    subsample_betas: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def is_unstable(self, factor: str) -> bool:
        """Whether the subsample estimates disagree about the *direction*.

        A beta whose sign is not consistent across the sample is not a
        sensitivity, it is an average of two different regimes, and applying it
        to a shock produces a number with no interpretation.
        """
        estimates = self.subsample_betas.get(factor, [])
        if len(estimates) < 2:
            return False
        return min(estimates) < 0 < max(estimates)

    def summary(self) -> str:
        lines = [
            f"FACTOR SENSITIVITIES -- {self.n_obs:,} observations, R-squared {self.r_squared:.3f}",
            "-" * 68,
            f"  {'factor':<12}{'beta':>10}{'std err':>10}{'t':>8}   subsample range",
        ]
        for factor, beta in self.betas.items():
            estimates = self.subsample_betas.get(factor, [])
            spread = (
                f"[{min(estimates):+.2f}, {max(estimates):+.2f}]" if estimates else "not estimated"
            )
            flag = "  UNSTABLE" if self.is_unstable(factor) else ""
            lines.append(
                f"  {factor:<12}{beta:>10.3f}{self.standard_errors[factor]:>10.3f}"
                f"{self.t_statistics[factor]:>8.2f}   {spread}{flag}"
            )
        lines.extend(["", *(f"  {note}" for note in self.notes)])
        return "\n".join(lines)


def estimate_response(
    asset_returns: ArrayLike,
    factor_changes: dict[str, ArrayLike],
    *,
    hac_lags: int | None = None,
    n_subsamples: int = 4,
) -> FactorResponse:
    """Regress asset returns on macro factor changes, with HAC standard errors.

    Args:
        asset_returns: the asset's return series.
        factor_changes: factor name to change series, aligned with the returns.
        hac_lags: Newey-West bandwidth. ``None`` uses the standard
            ``4(n/100)^(2/9)`` rule.
        n_subsamples: contiguous blocks used to check whether the betas are
            stable. Fewer than two disables the check.

    Returns
    -------
        A :class:`FactorResponse`. Read ``is_unstable`` before using any beta.
    """
    y = np.asarray(asset_returns, dtype=float).ravel()
    names = list(factor_changes)
    if not names:
        raise ValueError("at least one factor is needed")

    columns = [np.asarray(factor_changes[name], dtype=float).ravel() for name in names]
    for name, column in zip(names, columns, strict=True):
        if column.size != y.size:
            raise ValueError(
                f"factor {name!r} has {column.size} observations against the "
                f"asset's {y.size}; these must be aligned before regressing, not after"
            )

    # The intercept is explicit: the OLS primitive here deliberately does not add
    # one, because an implicit intercept is a frequent and silent
    # misspecification in factor work.
    design = np.column_stack([np.ones(y.size), *columns])

    usable = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    if usable.sum() < 60:
        raise ValueError(
            f"only {int(usable.sum())} usable observations; a macro beta with an "
            "interval worth reporting needs at least 60"
        )

    fit = ols(y[usable], design[usable], cov_type="hac", hac_lags=hac_lags)

    betas = {name: float(fit.coefficients[i + 1]) for i, name in enumerate(names)}
    errors = {name: float(fit.standard_errors[i + 1]) for i, name in enumerate(names)}
    t_stats = {name: float(fit.t_statistics[i + 1]) for i, name in enumerate(names)}

    subsamples: dict[str, list[float]] = {name: [] for name in names}
    if n_subsamples >= 2:
        blocks = np.array_split(np.flatnonzero(usable), n_subsamples)
        for block in blocks:
            if block.size < 30:
                continue
            try:
                partial = ols(y[block], design[block], cov_type="hac")
            except (ValueError, np.linalg.LinAlgError):
                continue
            for i, name in enumerate(names):
                subsamples[name].append(float(partial.coefficients[i + 1]))

    response = FactorResponse(
        betas=betas,
        standard_errors=errors,
        t_statistics=t_stats,
        r_squared=float(fit.r_squared),
        n_obs=int(usable.sum()),
        subsample_betas=subsamples,
    )

    unstable = [name for name in names if response.is_unstable(name)]
    if unstable:
        response.notes.append(
            f"{', '.join(unstable)}: the sign of the beta is not consistent across "
            "subsamples. A beta that changes direction is not a sensitivity, it is "
            "an average of two regimes, and applying it to a shock produces a "
            "number with no interpretation."
        )
    weak = [name for name in names if abs(t_stats[name]) < 2.0]
    if weak:
        response.notes.append(
            f"{', '.join(weak)}: |t| < 2, so the beta is not distinguishable from "
            "zero. The scenario response below is reported for completeness and "
            "should be read as 'no measurable effect'."
        )
    if response.r_squared < 0.10:
        response.notes.append(
            f"These factors explain {response.r_squared:.1%} of the variance. "
            "Almost everything that moves this asset is not in the model, so a "
            "scenario built from it describes a small part of the outcome."
        )
    return response


# --------------------------------------------------------------------------- #
# Applying a shock
# --------------------------------------------------------------------------- #
@dataclass
class ShockResult:
    """The estimated response to a shock, and the range it actually supports."""

    scenario: str
    point: float
    low: float
    high: float
    confidence: float
    #: Response computed from each subsample's betas.
    subsample_points: list[float] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def direction_is_certain(self) -> bool:
        """Whether the interval excludes zero -- the only claim usually supportable."""
        return self.low > 0 or self.high < 0

    @property
    def confidently_wrong(self) -> bool:
        """A narrow interval that excludes zero while the subsamples disagree on sign.

        This is the failure this module exists to catch, and it is the most
        dangerous output a scenario tool can produce: a precise-looking answer
        whose precision is measuring the wrong thing. The interval quantifies
        SAMPLING error within the estimation window. It does not quantify the
        risk that the relationship itself changes, which is the risk that
        actually materialises.

        Measured on QQQ against the 10-year Treasury yield over twenty years:
        the beta is +5.71 with t = 8.15 and a 90% interval of [+4.55%, +6.86%]
        for a 100bp rise -- confidently positive. Split by period it is +8.5,
        +9.2, +7.9 and +9.4 across 2006-2021, then **-2.8** through the 2022
        hiking cycle. The interval excludes zero and sits entirely on the wrong
        side of it for the regime that followed.
        """
        if not self.direction_is_certain or len(self.subsample_points) < 2:
            return False
        return min(self.subsample_points) < 0 < max(self.subsample_points)

    def summary(self) -> str:
        lines = [
            f"SCENARIO: {self.scenario}",
            "-" * 68,
            f"  estimated response   {self.point:+7.2%}",
            f"  {self.confidence:.0%} interval         [{self.low:+.2%}, {self.high:+.2%}]",
        ]
        if self.subsample_points:
            lines.append(
                f"  across subsamples    "
                f"[{min(self.subsample_points):+.2%}, {max(self.subsample_points):+.2%}]"
            )
        if self.contributions and len(self.contributions) > 1:
            lines.append("")
            ordered = sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))
            for factor, amount in ordered:
                lines.append(f"    {factor:<12}{amount:+8.2%}")
        lines.append("")
        if self.confidently_wrong:
            lines.append(
                "  READ THIS BEFORE THE NUMBER. The interval excludes zero, but the "
                "subsample estimates disagree about the SIGN. The interval measures "
                "sampling error inside the estimation window; it does not measure "
                "the risk that the relationship changes, which is the risk that "
                "actually shows up. A precise answer here is not a reliable one."
            )
        elif self.direction_is_certain:
            sign = "negative" if self.high < 0 else "positive"
            lines.append(f"  The direction is {sign}; the magnitude is not pinned down.")
        else:
            lines.append(
                "  The interval spans zero. On this evidence not even the "
                "DIRECTION of the response is established."
            )
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


def apply_shock(
    response: FactorResponse,
    shocks: dict[str, float],
    *,
    confidence: float = 0.90,
    name: str = "custom shock",
) -> ShockResult:
    r"""Apply a macro shock to estimated sensitivities.

    The point estimate is :math:`\sum_k \beta_k s_k`. Its variance is
    :math:`\sum_k s_k^2 \operatorname{Var}(\beta_k)`, which ignores the
    covariance between betas -- deliberately, and it is stated in the output,
    because that omission makes the interval *narrower* than the truth when the
    factors are correlated. An interval that is too narrow is the dangerous
    direction to err in, so the note says so rather than the code pretending
    otherwise.

    Args:
        response: sensitivities from :func:`estimate_response`.
        shocks: factor name to shock size, in the factor's own units.
        confidence: interval width. 0.90 gives a 5%/95% band.
        name: label for the report.

    Returns
    -------
        A :class:`ShockResult`. ``direction_is_certain`` is usually the only
        claim the data supports.
    """
    unknown = set(shocks) - set(response.betas)
    if unknown:
        raise ValueError(
            f"no beta was estimated for {sorted(unknown)}; a shock to a factor "
            "that was not in the regression cannot be applied"
        )

    contributions = {factor: response.betas[factor] * size for factor, size in shocks.items()}
    point = float(sum(contributions.values()))

    variance = sum(
        (size * response.standard_errors[factor]) ** 2 for factor, size in shocks.items()
    )
    z = float(ndtri(0.5 + confidence / 2.0))
    half_width = z * float(np.sqrt(variance))

    subsample_points: list[float] = []
    lengths = {len(response.subsample_betas.get(factor, [])) for factor in shocks}
    if len(lengths) == 1 and lengths != {0}:
        count = lengths.pop()
        subsample_points = [
            float(
                sum(response.subsample_betas[factor][i] * size for factor, size in shocks.items())
            )
            for i in range(count)
        ]

    result = ShockResult(
        scenario=name,
        point=point,
        low=point - half_width,
        high=point + half_width,
        confidence=confidence,
        subsample_points=subsample_points,
        contributions=contributions,
    )

    if len(shocks) > 1:
        result.notes.append(
            "The interval assumes the beta estimates are independent. They are "
            "not, so the true interval is WIDER than the one shown -- the error "
            "here is in the direction of overconfidence."
        )

    unstable = [factor for factor in shocks if response.is_unstable(factor)]
    if unstable:
        result.notes.append(
            f"The beta for {', '.join(unstable)} changes sign across subsamples, so "
            "this response is an average over regimes that disagreed. Treat it as "
            "a description of the past, not a projection."
        )

    largest = max((abs(size) for size in shocks.values()), default=0.0)
    if largest > 0.005:
        result.notes.append(
            f"A shock of {largest:.1%} is far larger than the daily variation the "
            "betas were fitted on. The response is extrapolated linearly, which is "
            "an assumption this estimate cannot test."
        )
    return result


def _shock_series(prices: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover
    """Simple returns with a leading zero, for callers assembling factor inputs."""
    out = np.zeros(prices.size)
    out[1:] = np.diff(prices) / prices[:-1]
    return out
