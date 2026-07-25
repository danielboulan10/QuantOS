r"""Optimal execution: the Almgren-Chriss framework and the square-root impact law.

The problem
-----------
You must liquidate :math:`X` shares within a horizon :math:`T`. Trading fast
incurs temporary impact; trading slowly leaves you exposed to price risk. These
pull in opposite directions and the resolution is a genuine optimisation rather
than a rule of thumb.

Almgren-Chriss minimise a mean-variance objective over the trading trajectory:

.. math::
    \min_{x(t)} \; \mathbb{E}[\text{cost}] + \lambda \operatorname{Var}[\text{cost}]

with linear temporary impact :math:`\eta v` on trading rate :math:`v`, linear
permanent impact :math:`\gamma`, and volatility :math:`\sigma`. The solution is a
hyperbolic-sine trajectory,

.. math::
    x(t) = X \frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}, \qquad
    \kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}

whose single parameter :math:`\kappa` -- the *urgency* -- has units of inverse
time and a clean interpretation: :math:`1/\kappa` is the timescale over which the
position is worked off. Two limits are worth internalising:

* :math:`\kappa \to 0` (risk-neutral) gives **TWAP**, a straight line. So TWAP is
  not a naive baseline; it is the optimal strategy for a trader who does not care
  about price risk.
* :math:`\kappa T \gg 1` (very risk-averse) front-loads heavily, approaching an
  exponential decay of the remaining position.

Linear versus square-root impact
--------------------------------
Almgren-Chriss assumes impact **linear** in trading rate, which makes the problem
tractable. Empirically, the impact of a completed order of size :math:`Q` scales
as :math:`\sqrt{Q}` -- the "square-root law", one of the most robust regularities
in all of finance, holding across equities, futures and FX over several decades
and orders of magnitude. :func:`fit_square_root_law` estimates it, and
:func:`square_root_impact_cost` uses it, so the two views are both available and
the tension between them is explicit rather than buried.

References
----------
Almgren, R. & Chriss, N. (2001), "Optimal execution of portfolio transactions",
    *J. Risk* 3(2), 5-40.
Almgren, R. et al. (2005), "Direct estimation of equity market impact",
    *Risk* 18(7), 58-62.
Toth, B. et al. (2011), "Anomalous price impact and the critical nature of
    liquidity in financial markets", *Phys. Rev. X* 1, 021006.
Gatheral, J. (2010), "No-dynamic-arbitrage and market impact",
    *Quantitative Finance* 10(7), 749-759.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ExecutionTrajectory",
    "ImpactParameters",
    "SquareRootFit",
    "almgren_chriss_trajectory",
    "efficient_execution_frontier",
    "fit_square_root_law",
    "implementation_shortfall",
    "square_root_impact_cost",
    "twap_trajectory",
    "vwap_trajectory",
]


@dataclass(frozen=True)
class ImpactParameters:
    r"""Market-impact and risk parameters for an execution problem.

    ``temporary_impact`` (:math:`\eta`) is the coefficient on *trading rate*:
    cost per share is :math:`\eta v`. ``permanent_impact`` (:math:`\gamma`) is the
    coefficient on cumulative quantity and cannot be avoided by any schedule --
    it drops out of the optimisation entirely, which is a genuinely useful
    result: **you cannot trade your way out of permanent impact.** Only the
    temporary component responds to scheduling.
    """

    volatility: float
    temporary_impact: float
    permanent_impact: float = 0.0
    #: Fixed per-share bid-ask cost, charged on every share traded.
    spread_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.temporary_impact <= 0:
            raise ValueError("temporary_impact must be positive")


@dataclass(frozen=True)
class ExecutionTrajectory:
    """An execution schedule with its expected cost and risk."""

    #: Time grid, length n+1, from 0 to T.
    times: NDArray[np.float64]
    #: Remaining position at each time, length n+1; starts at X, ends at 0.
    holdings: NDArray[np.float64]
    #: Shares traded in each interval, length n.
    trades: NDArray[np.float64]
    expected_cost: float
    cost_variance: float
    urgency: float
    strategy: str
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def cost_standard_deviation(self) -> float:
        return float(np.sqrt(max(self.cost_variance, 0.0)))

    @property
    def total_quantity(self) -> float:
        return float(self.holdings[0])

    @property
    def half_life(self) -> float:
        """Time by which half the position has been executed."""
        target = 0.5 * self.holdings[0]
        below = np.nonzero(self.holdings <= target)[0]
        return float(self.times[below[0]]) if below.size else float(self.times[-1])

    @property
    def front_loading(self) -> float:
        """Fraction executed in the first half of the horizon.

        0.5 is TWAP; above 0.5 is front-loaded (risk-averse); below 0.5 is
        back-loaded, which is optimal only for a *risk-seeking* trader and is
        almost always a sign of a sign error somewhere.
        """
        midpoint = self.holdings.size // 2
        return float((self.holdings[0] - self.holdings[midpoint]) / self.holdings[0])

    def mean_variance_objective(self, risk_aversion: float) -> float:
        return float(self.expected_cost + risk_aversion * self.cost_variance)


def almgren_chriss_trajectory(
    quantity: float,
    horizon: float,
    params: ImpactParameters,
    *,
    risk_aversion: float = 1e-6,
    n_steps: int = 100,
) -> ExecutionTrajectory:
    r"""The Almgren-Chriss optimal liquidation trajectory.

    Purpose
        Produce the mean-variance optimal schedule for working a large order, and
        the expected cost and cost variance that go with it.
    Method
        Closed form. With :math:`\kappa = \sqrt{\lambda\sigma^2/\eta}`,

        .. math::
            x(t) = X\frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}

        Expected temporary-impact cost and cost variance are then evaluated on the
        discretised trajectory, so the reported numbers correspond to the schedule
        that will actually be sent rather than to its continuous idealisation.
    Inputs
        ``quantity`` -- shares to liquidate (positive; the problem is symmetric so
        pass the absolute size). ``horizon`` -- in the same time units as
        ``volatility``. ``risk_aversion`` (:math:`\lambda`) -- **cost per unit
        variance**, so its scale depends on how prices are quoted; sweep it with
        :func:`efficient_execution_frontier` rather than guessing.
    Outputs
        :class:`ExecutionTrajectory`.
    Numerical note
        For large :math:`\kappa T`, :math:`\sinh` overflows. The ratio is
        evaluated in log space via ``expm1`` when :math:`\kappa T > 30`, where
        :math:`\sinh(a)/\sinh(b) \approx e^{a-b}` to well within double precision.
    Failure modes
        Raises for non-positive ``quantity`` or ``horizon``.

    Example
        >>> p = ImpactParameters(volatility=0.02, temporary_impact=1e-6)
        >>> # Risk-neutral: recovers TWAP, a straight line.
        >>> t = almgren_chriss_trajectory(1e6, 1.0, p, risk_aversion=1e-12)
        >>> bool(abs(t.front_loading - 0.5) < 0.01)
        True
        >>> # Risk-averse: front-loads. Note lambda must be large relative to
        >>> # eta/sigma^2 to bite -- here kappa = sqrt(1.0 * 4e-4 / 1e-6) = 20,
        >>> # so kappa*T = 20 and the schedule is strongly front-loaded.
        >>> t2 = almgren_chriss_trajectory(1e6, 1.0, p, risk_aversion=1.0)
        >>> bool(t2.front_loading > 0.9)
        True
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")

    times = np.linspace(0.0, horizon, n_steps + 1)
    kappa = float(np.sqrt(risk_aversion * params.volatility**2 / params.temporary_impact))

    if kappa * horizon < 1e-8:
        # Risk-neutral limit: the sinh ratio degenerates to a straight line.
        holdings = quantity * (1.0 - times / horizon)
    elif kappa * horizon > 30.0:
        # sinh(k(T-t))/sinh(kT) -> exp(-k t) for large kT, computed directly to
        # avoid overflow in sinh.
        holdings = quantity * np.exp(-kappa * times)
        holdings[-1] = 0.0
    else:
        holdings = quantity * np.sinh(kappa * (horizon - times)) / np.sinh(kappa * horizon)

    trades = -np.diff(holdings)
    dt = horizon / n_steps

    # Temporary impact cost: eta * v * (shares traded), v = trades/dt.
    temporary = float(params.temporary_impact * np.sum(trades**2) / dt)
    permanent = float(0.5 * params.permanent_impact * quantity**2)
    spread = float(params.spread_cost * np.sum(np.abs(trades)))
    expected_cost = temporary + permanent + spread

    # Cost variance from price risk on the *remaining* position over each step.
    cost_variance = float(params.volatility**2 * dt * np.sum(holdings[:-1] ** 2))

    return ExecutionTrajectory(
        times=times,
        holdings=holdings,
        trades=trades,
        expected_cost=expected_cost,
        cost_variance=cost_variance,
        urgency=kappa,
        strategy="almgren_chriss",
        detail={
            "risk_aversion": risk_aversion,
            "temporary_cost": temporary,
            "permanent_cost": permanent,
            "spread_cost": spread,
            "kappa_times_horizon": kappa * horizon,
        },
    )


def twap_trajectory(
    quantity: float, horizon: float, params: ImpactParameters, *, n_steps: int = 100
) -> ExecutionTrajectory:
    """Time-weighted average price: equal shares per interval.

    Identical to :func:`almgren_chriss_trajectory` with zero risk aversion, and
    implemented by that route to make the equivalence explicit rather than
    coincidental.
    """
    trajectory = almgren_chriss_trajectory(
        quantity, horizon, params, risk_aversion=0.0, n_steps=n_steps
    )
    return ExecutionTrajectory(
        times=trajectory.times,
        holdings=trajectory.holdings,
        trades=trajectory.trades,
        expected_cost=trajectory.expected_cost,
        cost_variance=trajectory.cost_variance,
        urgency=0.0,
        strategy="twap",
        detail=trajectory.detail,
    )


def vwap_trajectory(
    quantity: float,
    horizon: float,
    params: ImpactParameters,
    volume_profile: ArrayLike,
) -> ExecutionTrajectory:
    r"""Volume-weighted schedule: trade in proportion to expected volume.

    Purpose
        Minimise *tracking error against the VWAP benchmark*, which is a
        different objective from minimising cost. If you are measured against
        VWAP, matching the volume profile is optimal almost by definition, and
        the Almgren-Chriss trajectory will underperform the benchmark even while
        achieving a lower cost.

        Worth being explicit about, because the choice of benchmark determines the
        optimal strategy. A trader minimising implementation shortfall and one
        minimising VWAP slippage should trade differently, and neither is wrong.
    Inputs
        ``volume_profile`` -- expected volume per interval; normalised internally.
        The classic intraday shape is a U (heavy at the open and close).
    """
    profile = np.asarray(volume_profile, dtype=float).ravel()
    if profile.size < 1 or np.any(profile < 0) or float(np.sum(profile)) <= 0:
        raise ValueError("volume_profile must be non-negative with a positive sum")
    weights = profile / float(np.sum(profile))
    n_steps = weights.size

    times = np.linspace(0.0, horizon, n_steps + 1)
    trades = quantity * weights
    holdings = np.concatenate([[quantity], quantity - np.cumsum(trades)])
    holdings[-1] = 0.0
    dt = horizon / n_steps

    temporary = float(params.temporary_impact * np.sum(trades**2) / dt)
    spread = float(params.spread_cost * np.sum(np.abs(trades)))
    cost_variance = float(params.volatility**2 * dt * np.sum(holdings[:-1] ** 2))

    return ExecutionTrajectory(
        times=times,
        holdings=holdings,
        trades=trades,
        expected_cost=temporary + spread + 0.5 * params.permanent_impact * quantity**2,
        cost_variance=cost_variance,
        urgency=0.0,
        strategy="vwap",
        detail={"n_intervals": float(n_steps)},
    )


def efficient_execution_frontier(
    quantity: float,
    horizon: float,
    params: ImpactParameters,
    *,
    risk_aversions: ArrayLike | None = None,
    n_steps: int = 100,
) -> list[ExecutionTrajectory]:
    r"""Trace the efficient frontier of execution: cost against cost risk.

    Purpose
        There is no single optimal execution -- there is a *frontier*, and the
        choice along it is a preference, not a calculation. Sweeping
        :math:`\lambda` makes the trade-off visible, which is far more useful than
        reporting one trajectory for one arbitrary risk aversion.

        This is the direct analogue of the Markowitz frontier, with expected
        execution cost in place of expected return and cost variance in place of
        portfolio variance.
    Outputs
        Trajectories ordered from risk-neutral (lowest cost, highest risk) to
        highly risk-averse (highest cost, lowest risk).

    Example
        >>> p = ImpactParameters(volatility=0.02, temporary_impact=1e-6)
        >>> frontier = efficient_execution_frontier(1e6, 1.0, p)
        >>> costs = [t.expected_cost for t in frontier]
        >>> risks = [t.cost_variance for t in frontier]
        >>> # Cost rises and risk falls monotonically along the frontier.
        >>> all(a <= b for a, b in zip(costs, costs[1:]))
        True
        >>> all(a >= b for a, b in zip(risks, risks[1:]))
        True
    """
    if risk_aversions is None:
        lambdas = np.concatenate([[0.0], np.geomspace(1e-10, 1e-1, 24)])
    else:
        lambdas = np.asarray(risk_aversions, dtype=float).ravel()
    return [
        almgren_chriss_trajectory(
            quantity, horizon, params, risk_aversion=float(lam), n_steps=n_steps
        )
        for lam in np.sort(lambdas)
    ]


def square_root_impact_cost(
    quantity: float,
    daily_volume: float,
    volatility: float,
    *,
    coefficient: float = 1.0,
    exponent: float = 0.5,
) -> float:
    r"""Impact cost from the square-root law.

    .. math:: \frac{\Delta P}{P} = Y \sigma
              \left(\frac{Q}{V}\right)^{\delta}, \qquad \delta \approx 0.5

    Purpose
        The practitioner's impact estimate. Its remarkable feature is what it does
        *not* depend on: not the execution horizon, not the participation rate,
        not the schedule. Impact is set by the order's size relative to daily
        volume, scaled by volatility. Trading a given order more slowly does not
        reduce it.
    Interpretation
        The exponent near 0.5 -- rather than 1 -- is why large orders are less
        costly per share than linear impact would predict, and it is the empirical
        fact most in tension with the linear-impact assumption underlying
        :func:`almgren_chriss_trajectory`. The coefficient :math:`Y` is typically
        estimated at 0.5-1.0.
    Outputs
        Fractional cost (multiply by price and quantity for currency).
    """
    if daily_volume <= 0:
        raise ValueError("daily_volume must be positive")
    if quantity < 0:
        raise ValueError("quantity must be non-negative")
    return float(coefficient * volatility * (quantity / daily_volume) ** exponent)


@dataclass(frozen=True)
class SquareRootFit:
    r"""Fitted impact law :math:`\\text{impact} = c \\cdot (Q/V)^{\\delta}`."""

    coefficient: float
    exponent: float
    exponent_standard_error: float
    r_squared: float
    n_obs: int

    @property
    def exponent_confidence_interval(self) -> tuple[float, float]:
        """95% interval for the exponent."""
        half = 1.96 * self.exponent_standard_error
        return (self.exponent - half, self.exponent + half)

    @property
    def consistent_with_square_root(self) -> bool:
        """Whether 0.5 lies inside the exponent's 95% interval."""
        lo, hi = self.exponent_confidence_interval
        return bool(lo <= 0.5 <= hi)

    @property
    def consistent_with_linear(self) -> bool:
        """Whether 1.0 lies inside the interval -- the Almgren-Chriss assumption."""
        lo, hi = self.exponent_confidence_interval
        return bool(lo <= 1.0 <= hi)


def fit_square_root_law(
    participation: ArrayLike, impact: ArrayLike, *, volatility: ArrayLike | None = None
) -> SquareRootFit:
    r"""Estimate the impact exponent by log-log regression.

    Purpose
        Test the square-root law on data rather than assuming it. Regressing
        :math:`\log(\text{impact})` on :math:`\log(Q/V)` gives the exponent
        directly as the slope, and its standard error tells you whether 0.5 and
        1.0 can be distinguished at all -- often, on a small sample, they cannot,
        and saying so is more useful than reporting a point estimate of 0.53.
    Inputs
        ``participation`` -- :math:`Q/V` per order, strictly positive.
        ``impact`` -- realised impact per order, strictly positive.
        ``volatility`` -- optional per-order volatility to normalise by, which
        removes the largest confound: high-volatility periods have both bigger
        impact and, usually, bigger orders.
    Outputs
        :class:`SquareRootFit`; check
        :attr:`~SquareRootFit.consistent_with_square_root` and
        :attr:`~SquareRootFit.consistent_with_linear`.
    Failure modes
        Raises if fewer than 10 strictly-positive paired observations remain.
        Non-positive values cannot be log-transformed and are dropped -- note that
        this *censors* the sample, biasing the estimate if small impacts are
        systematically measured as zero or negative.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> q = 10 ** rng.uniform(-4, -1, 800)
        >>> y = 0.8 * q ** 0.5 * np.exp(rng.standard_normal(800) * 0.1)
        >>> fit = fit_square_root_law(q, y)
        >>> fit.consistent_with_square_root, fit.consistent_with_linear
        (True, False)
    """
    from quantos.core.timeseries.ols import ols

    p = np.asarray(participation, dtype=float).ravel()
    y = np.asarray(impact, dtype=float).ravel()
    if p.size != y.size:
        raise ValueError("participation and impact must be the same length")
    if volatility is not None:
        v = np.asarray(volatility, dtype=float).ravel()
        if v.size != y.size:
            raise ValueError("volatility must match the number of observations")
        with np.errstate(divide="ignore", invalid="ignore"):
            y = y / v

    mask = np.isfinite(p) & np.isfinite(y) & (p > 0) & (y > 0)
    if int(np.sum(mask)) < 10:
        raise ValueError(
            f"need at least 10 strictly-positive paired observations, got {int(np.sum(mask))}"
        )
    log_p = np.log(p[mask])
    log_y = np.log(y[mask])

    design = np.column_stack([np.ones(log_p.size), log_p])
    fit = ols(log_y, design)
    return SquareRootFit(
        coefficient=float(np.exp(fit.coefficients[0])),
        exponent=float(fit.coefficients[1]),
        exponent_standard_error=float(fit.standard_errors[1]),
        r_squared=float(fit.r_squared),
        n_obs=int(log_p.size),
    )


def implementation_shortfall(
    fill_prices: ArrayLike,
    fill_quantities: ArrayLike,
    arrival_price: float,
    side: int = 1,
) -> dict[str, float]:
    r"""Implementation shortfall against the arrival price.

    .. math:: IS = \text{side} \times \sum_i q_i (P_i - P_0) \big/ \sum_i q_i

    Purpose
        The honest measure of execution quality. Unlike VWAP slippage it cannot be
        gamed by trading when volume happens to be favourable, because the
        benchmark -- the price at the moment the decision was made -- is fixed
        before trading begins.
    Outputs
        Dictionary with the shortfall in price units and in basis points, the
        volume-weighted average execution price, and the total quantity.

    Example
        >>> out = implementation_shortfall([100.5, 101.0], [100, 100], 100.0)
        >>> round(out["shortfall_bps"], 2)
        75.0
    """
    p = np.asarray(fill_prices, dtype=float).ravel()
    q = np.asarray(fill_quantities, dtype=float).ravel()
    if p.size != q.size:
        raise ValueError("fill_prices and fill_quantities must be the same length")
    total = float(np.sum(q))
    if total <= 0:
        raise ValueError("total filled quantity must be positive")
    if arrival_price <= 0:
        raise ValueError("arrival_price must be positive")

    vwap = float(np.sum(p * q) / total)
    shortfall = float(side * (vwap - arrival_price))
    return {
        "shortfall": shortfall,
        "shortfall_bps": float(shortfall / arrival_price * 1e4),
        "average_execution_price": vwap,
        "total_quantity": total,
        "arrival_price": float(arrival_price),
    }
