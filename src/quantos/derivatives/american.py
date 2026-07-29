r"""American options: a price *interval*, not a price.

Why an interval
---------------
Every option priced elsewhere in this repository is European, where exercise
happens at one known date and a closed form exists. American exercise is an
optimal-stopping problem: at every step the holder compares the payoff now
against the value of waiting, and there is no closed form for that decision.

The standard method is Longstaff-Schwartz least-squares Monte Carlo. It works,
and it is **biased low** by construction: the continuation value is estimated by
regression, the resulting exercise rule is therefore suboptimal, and a suboptimal
rule is worth less than the optimal one. So LSMC gives a *lower bound*.

That is only half an answer. The Andersen-Broadie dual construction turns the
same exercise rule into an **upper bound**, by building a martingale that hedges
away the timing decision. Between them the true price is bracketed, and the width
of the bracket measures how good the exercise rule actually is.

Reporting ``[lower, upper]`` rather than a point is the same disposition this
repository applies everywhere else, in a place where almost nobody applies it.

The trap in the middle
----------------------
LSMC is biased low **only if the regression and the valuation use different
paths**. Fit the continuation-value regression on the same paths you then value,
and the regression has seen each path's future: it can "predict" a continuation
value using information that path actually realised, the exercise rule looks
prescient, and the price is biased *high*.

That in-sample bias is easy to write, invisible in the output, and pushes the
number in the flattering direction. :func:`longstaff_schwartz` therefore takes a
separate regression sample by default, and :func:`price_american` reports both so
the gap can be seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantos.derivatives.black_scholes import OptionType

__all__ = [
    "AmericanPrice",
    "ExerciseRule",
    "dual_upper_bound",
    "longstaff_schwartz",
    "price_american",
    "simulate_gbm_paths",
]


def simulate_gbm_paths(
    spot: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
    n_steps: int,
    n_paths: int,
    *,
    dividend_yield: float = 0.0,
    seed: int = 20240719,
    antithetic: bool = True,
) -> NDArray[np.float64]:
    r"""Risk-neutral geometric Brownian motion paths, shape ``(n_paths, n_steps + 1)``.

    ``antithetic`` pairs each draw with its negation, which removes the odd
    moments of the sampling error exactly. For a payoff that is close to linear in
    the driver -- most options are, away from the wings -- this cuts the standard
    error substantially for no extra work.
    """
    if n_paths % 2 == 1 and antithetic:
        n_paths += 1
    rng = np.random.default_rng(seed)
    dt = time_to_expiry / n_steps
    drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)

    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        shocks = np.concatenate([half, -half], axis=0)
    else:
        shocks = rng.standard_normal((n_paths, n_steps))

    increments = drift + diffusion * shocks
    log_paths = np.cumsum(increments, axis=1)
    return spot * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1))


def _payoff(
    prices: NDArray[np.float64], strike: float, option_type: OptionType
) -> NDArray[np.float64]:
    if option_type is OptionType.CALL:
        return np.maximum(prices - strike, 0.0)
    return np.maximum(strike - prices, 0.0)


def _basis(prices: NDArray[np.float64], strike: float, degree: int = 3) -> NDArray[np.float64]:
    r"""Regression basis: powers of moneyness.

    Moneyness rather than raw price, because a design matrix in raw prices around
    100 has columns of wildly different scale and the normal equations become
    ill-conditioned. Centring on the strike keeps the values near 1.
    """
    x = prices / strike
    return np.column_stack([x**power for power in range(degree + 1)])


@dataclass
class ExerciseRule:
    """The stopping rule LSMC learned, as regression coefficients per step."""

    coefficients: list[NDArray[np.float64] | None]
    strike: float
    option_type: OptionType
    degree: int = 3

    def continuation_value(self, step: int, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimated value of *not* exercising at ``step``."""
        beta = self.coefficients[step]
        if beta is None:
            return np.zeros_like(prices)
        return np.asarray(_basis(prices, self.strike, self.degree) @ beta, dtype=float)

    def should_exercise(self, step: int, prices: NDArray[np.float64]) -> NDArray[np.bool_]:
        immediate = _payoff(prices, self.strike, self.option_type)
        return (immediate > 0) & (immediate >= self.continuation_value(step, prices))


def longstaff_schwartz(
    paths: NDArray[np.float64],
    strike: float,
    rate: float,
    time_to_expiry: float,
    *,
    option_type: OptionType = OptionType.PUT,
    degree: int = 3,
    regression_paths: NDArray[np.float64] | None = None,
) -> tuple[float, ExerciseRule, float]:
    r"""Least-squares Monte Carlo lower bound, and the rule that produced it.

    Method
        Backward induction. At each step, regress the discounted continuation
        value on a polynomial basis of the current price **using only paths that
        are in the money** -- out-of-the-money paths carry no information about
        the exercise boundary and including them wastes the fit on a region where
        the decision is trivial.
    Inputs
        ``regression_paths`` -- a separate sample on which to fit the rule. When
        given, the returned price is an honest lower bound. When omitted the rule
        is fitted in sample and the price is biased *high*, which is offered only
        so the size of that bias can be measured.
    Outputs
        ``(price, rule, standard_error)``.

    Example
        >>> paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 20_000, seed=1)
        >>> price, rule, se = longstaff_schwartz(paths, 100.0, 0.05, 1.0)
        >>> bool(price > 5.0)     # an American put is worth more than nothing
        True
    """
    n_paths, n_nodes = paths.shape
    n_steps = n_nodes - 1
    dt = time_to_expiry / n_steps
    discount = float(np.exp(-rate * dt))

    fitting = regression_paths if regression_paths is not None else paths
    coefficients: list[NDArray[np.float64] | None] = [None] * n_nodes

    # --- learn the rule on the fitting sample -------------------------------- #
    cash_flow = _payoff(fitting[:, -1], strike, option_type)
    for step in range(n_steps - 1, 0, -1):
        cash_flow = cash_flow * discount
        prices = fitting[:, step]
        immediate = _payoff(prices, strike, option_type)
        in_money = immediate > 0
        if int(np.sum(in_money)) <= degree + 1:
            continue

        design = _basis(prices[in_money], strike, degree)
        beta, *_ = np.linalg.lstsq(design, cash_flow[in_money], rcond=None)
        coefficients[step] = beta

        continuation = design @ beta
        exercise = immediate[in_money] >= continuation
        indices = np.where(in_money)[0][exercise]
        cash_flow[indices] = immediate[in_money][exercise]

    rule = ExerciseRule(coefficients, strike, option_type, degree)

    # --- apply it to the valuation sample ------------------------------------ #
    values = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    for step in range(1, n_nodes):
        prices = paths[:, step]
        if step == n_steps:
            stop = alive & (_payoff(prices, strike, option_type) > 0)
        else:
            stop = alive & rule.should_exercise(step, prices)
        if np.any(stop):
            values[stop] = _payoff(prices[stop], strike, option_type) * np.exp(-rate * step * dt)
            alive &= ~stop

    price = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(n_paths))
    return price, rule, standard_error


def dual_upper_bound(
    paths: NDArray[np.float64],
    rule: ExerciseRule,
    rate: float,
    time_to_expiry: float,
    *,
    n_inner: int = 50,
    volatility: float = 0.2,
    dividend_yield: float = 0.0,
    seed: int = 4242,
) -> tuple[float, float]:
    r"""Andersen-Broadie dual upper bound from an exercise rule.

    The idea
        For any martingale :math:`M`, the value of the option is bounded above by

        .. math:: \mathbb{E}\left[\max_t \left(h_t - M_t\right)\right] + M_0

        because a hedger holding :math:`M` no longer needs to time the exercise.
        Choosing :math:`M` as the Doob martingale of the *approximate* value
        process makes the bound tight when the rule is good, and loose when it is
        not -- which is exactly the diagnostic wanted.

    Method
        Along each outer path, estimate the value of following the rule from each
        step by an inner simulation, and accumulate the martingale increments. The
        remaining slack, averaged, is the duality gap.

    Cost
        ``n_outer * n_steps * n_inner`` paths. This is genuinely expensive and is
        why the upper bound is rarely computed -- which is also why computing it
        is worth something.
    """
    n_paths, n_nodes = paths.shape
    n_steps = n_nodes - 1
    dt = time_to_expiry / n_steps
    rng = np.random.default_rng(seed)

    strike, option_type = rule.strike, rule.option_type
    martingale = np.zeros(n_paths)
    best = _payoff(paths[:, 0], strike, option_type).astype(float)
    previous_value = _approximate_value(paths[:, 0], 0, rule, rate, dt, n_steps)

    for step in range(1, n_nodes):
        prices = paths[:, step]
        discount = np.exp(-rate * step * dt)

        current_value = (
            _payoff(prices, strike, option_type).astype(float)
            if step == n_steps
            else _approximate_value(prices, step, rule, rate, dt, n_steps)
        )

        # Continuation estimated by a short inner simulation from the previous node.
        expected = _inner_expectation(
            paths[:, step - 1],
            step - 1,
            rule,
            rate,
            dt,
            n_steps,
            n_inner=n_inner,
            volatility=volatility,
            dividend_yield=dividend_yield,
            rng=rng,
        )
        martingale += discount * (current_value - expected)
        best = np.maximum(best, discount * _payoff(prices, strike, option_type) - martingale)
        previous_value = current_value

    del previous_value
    upper = float(np.mean(best))
    standard_error = float(np.std(best, ddof=1) / np.sqrt(n_paths))
    return upper, standard_error


def _approximate_value(
    prices: NDArray[np.float64],
    step: int,
    rule: ExerciseRule,
    rate: float,
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """Value of following the rule from ``step``, as the rule itself estimates it."""
    immediate = _payoff(prices, rule.strike, rule.option_type).astype(float)
    if step >= n_steps:
        return immediate
    continuation = rule.continuation_value(step, prices)
    return np.asarray(np.maximum(immediate, np.maximum(continuation, 0.0)), dtype=float)


def _inner_expectation(
    prices: NDArray[np.float64],
    step: int,
    rule: ExerciseRule,
    rate: float,
    dt: float,
    n_steps: int,
    *,
    n_inner: int,
    volatility: float,
    dividend_yield: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """One-step-ahead expectation of the approximate value, by inner simulation."""
    drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)
    shocks = rng.standard_normal((prices.size, n_inner))
    successors = prices[:, None] * np.exp(drift + diffusion * shocks)

    values = _approximate_value(successors.ravel(), step + 1, rule, rate, dt, n_steps)
    averaged = values.reshape(prices.size, n_inner).mean(axis=1)
    return np.asarray(np.exp(-rate * dt) * averaged, dtype=float)


@dataclass
class AmericanPrice:
    """A bracketed American price, and how much the bracket cost."""

    lower: float
    upper: float
    lower_standard_error: float
    upper_standard_error: float
    european: float
    in_sample_price: float = float("nan")
    n_paths: int = 0
    n_steps: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def duality_gap(self) -> float:
        """Width of the bracket. Small means the exercise rule is near optimal."""
        return self.upper - self.lower

    @property
    def early_exercise_premium(self) -> float:
        """What the right to exercise early is worth over the European option."""
        return self.lower - self.european

    @property
    def in_sample_bias(self) -> float:
        """How much fitting the rule on the valuation paths inflates the price."""
        if not np.isfinite(self.in_sample_price):
            return float("nan")
        return self.in_sample_price - self.lower

    def summary(self) -> str:
        lines = [
            f"American price in [{self.lower:.4f}, {self.upper:.4f}] (gap {self.duality_gap:.4f})",
            f"  lower  {self.lower:.4f} +/- {self.lower_standard_error:.4f}  (LSMC, out of sample)",
            f"  upper  {self.upper:.4f} +/- {self.upper_standard_error:.4f}"
            "  (Andersen-Broadie dual)",
            f"  European {self.european:.4f}, early-exercise premium "
            f"{self.early_exercise_premium:+.4f}",
        ]
        if np.isfinite(self.in_sample_price):
            lines.append(
                f"  in-sample LSMC {self.in_sample_price:.4f}: biased HIGH by "
                f"{self.in_sample_bias:+.4f} because the regression saw each path's own future"
            )
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


def price_american(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.PUT,
    n_paths: int = 40_000,
    n_steps: int = 50,
    n_inner: int = 30,
    degree: int = 3,
    seed: int = 20240719,
    compute_upper: bool = True,
) -> AmericanPrice:
    """Bracket an American option between LSMC and its dual upper bound.

    Also reports the *in-sample* LSMC price, so the bias from fitting the
    exercise rule on the same paths used to value it can be seen rather than
    described.
    """
    from quantos.derivatives.black_scholes import black_scholes_price

    valuation = simulate_gbm_paths(
        spot,
        rate,
        volatility,
        time_to_expiry,
        n_steps,
        n_paths,
        dividend_yield=dividend_yield,
        seed=seed,
    )
    fitting = simulate_gbm_paths(
        spot,
        rate,
        volatility,
        time_to_expiry,
        n_steps,
        n_paths,
        dividend_yield=dividend_yield,
        seed=seed + 977,
    )

    lower, rule, lower_se = longstaff_schwartz(
        valuation,
        strike,
        rate,
        time_to_expiry,
        option_type=option_type,
        degree=degree,
        regression_paths=fitting,
    )
    in_sample, _, _ = longstaff_schwartz(
        valuation, strike, rate, time_to_expiry, option_type=option_type, degree=degree
    )

    european = float(
        black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            volatility,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
    )

    upper, upper_se = (float("nan"), float("nan"))
    notes: list[str] = []
    if compute_upper:
        # The dual is expensive; a subsample keeps it tractable without changing
        # the estimator, only its standard error.
        subset = valuation[: min(2_000, n_paths)]
        upper, upper_se = dual_upper_bound(
            subset,
            rule,
            rate,
            time_to_expiry,
            n_inner=n_inner,
            volatility=volatility,
            dividend_yield=dividend_yield,
            seed=seed + 31,
        )
        if upper < lower:
            notes.append(
                f"the dual bound ({upper:.4f}) came in below the primal ({lower:.4f}); with "
                "finite inner samples this happens when the gap is smaller than the Monte "
                "Carlo error, and the bracket should be read as [lower, lower + noise]"
            )

    if option_type is OptionType.CALL and dividend_yield <= 0:
        notes.append(
            "an American call on a non-dividend-paying asset is never exercised early, "
            "so this should equal the European price -- a useful check rather than a use case"
        )

    return AmericanPrice(
        lower=lower,
        upper=upper,
        lower_standard_error=lower_se,
        upper_standard_error=upper_se,
        european=european,
        in_sample_price=in_sample,
        n_paths=n_paths,
        n_steps=n_steps,
        notes=notes,
    )
