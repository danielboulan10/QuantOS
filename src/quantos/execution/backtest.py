r"""Route a real order through the matching engine and check the theory's prediction.

The question
------------
:mod:`quantos.execution.almgren_chriss` predicts what an execution will cost. It
does so from a model: linear temporary impact, linear permanent impact, a
volatility term, and a closed-form optimal trajectory. Every one of those is an
assumption.

This module tests the prediction. It takes a parent order, splits it on the
schedule the theory recommends, and sends the child orders into the *actual*
limit order book from :mod:`quantos.exchange` -- where they walk a real ladder of
resting liquidity, move the price by consuming it, and are filled at whatever
prices the queue actually holds. The realised implementation shortfall is then
compared against what was predicted.

This is a falsifiable claim about the repository's own model, which is the
interesting kind. A prediction nobody checks against an independent mechanism is
a parameter fit.

What is genuinely tested, and what is not
------------------------------------------
**Tested.** The shape of the cost curve as a function of participation and
urgency; whether the square-root law's exponent shows up in book-walking
behaviour; whether the optimal trajectory actually beats TWAP under the same
book; and whether the model's *ordering* of strategies survives contact with a
mechanism it knows nothing about.

**Not tested.** Whether real markets behave like this book. The book is populated
by a resting-liquidity model, not by real participants, so agreement here shows
the theory is consistent with a plausible microstructure -- not that its
parameters are right for any actual venue. Overstating that would be the whole
point of the exercise thrown away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantos.core.types import AgentId, Nanos, Order, OrderId, Quantity, Side, Ticks
from quantos.exchange.book import LimitOrderBook
from quantos.execution.almgren_chriss import ExecutionTrajectory, ImpactParameters

__all__ = [
    "ExecutionOutcome",
    "LiquidityProfile",
    "calibrate_impact",
    "compare_strategies",
    "execute_trajectory",
]


@dataclass
class LiquidityProfile:
    """How much resting size sits at each price level, and how it replenishes.

    A book is not a wall of infinite liquidity, and it is not static either:
    consuming a level attracts replacement. ``replenish_rate`` is the fraction of
    consumed size that returns before the next child order arrives, which is what
    makes trading slowly cheaper than trading fast in this simulation. With no
    replenishment, splitting an order buys nothing and the whole exercise is
    trivial.
    """

    #: Contracts resting at the touch.
    depth_at_touch: int = 500
    #: Multiplicative decay of size per tick away from the touch.
    depth_decay: float = 0.85
    #: Number of price levels populated on each side.
    levels: int = 40
    #: Tick size in price units.
    tick: float = 0.01
    #: Fraction of consumed depth restored between child orders, in [0, 1].
    replenish_rate: float = 0.35
    #: Permanent drift per contract consumed, in ticks. The book remembers.
    permanent_impact_ticks: float = 0.002

    def depth_at(self, level: int) -> int:
        return max(1, int(self.depth_at_touch * self.depth_decay**level))


@dataclass
class ExecutionOutcome:
    """What an execution actually cost, against what was predicted."""

    strategy: str
    quantity: int
    filled: int
    #: Volume-weighted average price actually achieved.
    average_price: float
    arrival_price: float
    #: Realised shortfall in basis points of arrival price, positive = cost.
    realised_shortfall_bps: float
    predicted_shortfall_bps: float
    #: Worst single child order's slippage, which the model does not predict.
    worst_child_bps: float
    n_children: int
    permanent_move_bps: float
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Report whether the whole parent order filled.

        A partial fill's shortfall is not comparable with a complete one: it is
        the cost of the *cheap* part of the order, because execution stops when
        liquidity runs out. Comparing the two ranks the schedule that gave up
        first as the cheapest, which an earlier test in this module did.
        """
        return self.filled >= self.quantity

    @property
    def prediction_error_bps(self) -> float:
        return self.realised_shortfall_bps - self.predicted_shortfall_bps

    @property
    def relative_error(self) -> float:
        if abs(self.predicted_shortfall_bps) < 1e-9:
            return float("nan")
        return self.prediction_error_bps / self.predicted_shortfall_bps

    def summary(self) -> str:
        partial = "" if self.complete else f"  PARTIAL {self.filled}/{self.quantity}"
        return (
            f"{self.strategy:22s} {self.n_children:3d} children  "
            f"realised {self.realised_shortfall_bps:7.2f} bps  "
            f"predicted {self.predicted_shortfall_bps:7.2f} bps  "
            f"error {self.prediction_error_bps:+7.2f}{partial}"
        )


def _build_book(profile: LiquidityProfile, mid_ticks: int) -> LimitOrderBook:
    """Populate a fresh book with resting liquidity on both sides."""
    book = LimitOrderBook()
    order_id = 1
    for level in range(profile.levels):
        size = profile.depth_at(level)
        for side, price in (
            (Side.BUY, mid_ticks - 1 - level),
            (Side.SELL, mid_ticks + 1 + level),
        ):
            book.add(
                Order(
                    OrderId(order_id),
                    AgentId("liquidity"),
                    side,
                    Quantity(size),
                    Ticks(price),
                )
            )
            order_id += 1
    return book


def execute_trajectory(
    trajectory: ExecutionTrajectory,
    *,
    profile: LiquidityProfile | None = None,
    arrival_ticks: int = 10_000,
    predicted_shortfall_bps: float = float("nan"),
    strategy: str = "",
    seed: int = 20240719,
) -> ExecutionOutcome:
    """Send a trajectory's child orders into a real book and measure the result.

    Purpose
        Turn a modelled cost into a measured one.
    Method
        The book is rebuilt once, then each child order crosses it as a market
        order, consuming resting size level by level. Between children, part of
        the consumed depth is replenished and the mid is moved by a permanent
        impact term proportional to volume already executed -- so a fast schedule
        walks further up a thinner book, which is the mechanism the theory
        models in closed form.
    Failure modes
        If the book is exhausted the outcome records a partial fill rather than
        raising, because running out of liquidity is a result.
    """
    profile = profile or LiquidityProfile()
    trades = np.asarray(trajectory.trades, dtype=float)
    trades = np.abs(trades[np.isfinite(trades)])
    trades = trades[trades > 0]
    total = round(float(np.sum(trades)))
    if total <= 0:
        raise ValueError("the trajectory contains no trades to execute")

    arrival_price = arrival_ticks * profile.tick
    book = _build_book(profile, arrival_ticks)

    executed_value = 0.0
    executed_quantity = 0
    worst_child_bps = 0.0
    cumulative = 0
    next_order_id = 10_000_000

    for child in trades:
        size = round(child)
        if size <= 0:
            continue
        fills, remaining = book.match(Side.BUY, Quantity(size), None, Nanos(0))
        if not fills:
            break

        child_value = sum(int(traded) * int(maker.price) * profile.tick for maker, traded in fills)
        child_quantity = sum(int(traded) for _, traded in fills)
        executed_value += child_value
        executed_quantity += child_quantity
        cumulative += child_quantity

        child_average = child_value / child_quantity
        child_bps = (child_average - arrival_price) / arrival_price * 1e4
        worst_child_bps = max(worst_child_bps, child_bps)

        # Permanent impact: the book re-forms around a higher mid after volume.
        drift = int(profile.permanent_impact_ticks * cumulative)
        # Replenishment: some consumed depth returns before the next child.
        for level in range(profile.levels):
            price = arrival_ticks + drift + 1 + level
            wanted = int(profile.depth_at(level) * profile.replenish_rate)
            resting = book.size_at(Ticks(price))
            if wanted > resting:
                book.add(
                    Order(
                        OrderId(next_order_id),
                        AgentId("replenish"),
                        Side.SELL,
                        Quantity(wanted - resting),
                        Ticks(price),
                    )
                )
                next_order_id += 1

        if remaining > 0:
            break

    notes: list[str] = []
    if executed_quantity < total:
        notes.append(
            f"only {executed_quantity} of {total} filled: the book was exhausted, which "
            "is itself a result -- the schedule demanded more liquidity than existed"
        )

    average_price = executed_value / executed_quantity if executed_quantity else float("nan")
    realised_bps = (
        (average_price - arrival_price) / arrival_price * 1e4 if executed_quantity else float("nan")
    )
    final_mid = book.best_ask
    permanent_bps = (
        (float(final_mid) * profile.tick - arrival_price) / arrival_price * 1e4
        if final_mid is not None
        else float("nan")
    )

    return ExecutionOutcome(
        strategy=strategy or getattr(trajectory, "strategy", "unnamed"),
        quantity=total,
        filled=executed_quantity,
        average_price=average_price,
        arrival_price=arrival_price,
        realised_shortfall_bps=realised_bps,
        predicted_shortfall_bps=predicted_shortfall_bps,
        worst_child_bps=worst_child_bps,
        n_children=int(trades.size),
        permanent_move_bps=permanent_bps,
        notes=notes,
    )


def calibrate_impact(
    quantity: int,
    *,
    profile: LiquidityProfile | None = None,
    volatility: float = 0.02,
    horizon: float = 1.0,
    n_periods: int = 20,
    arrival_ticks: int = 10_000,
) -> ImpactParameters:
    """Fit the model's impact coefficient to this book, from one observation.

    Why calibration comes first
        An uncalibrated model predicted 2.60 bps where the book charged 23.37 --
        a 9x gap that says nothing about whether the *theory* is right, only that
        its units were never matched to this venue. Comparing an uncalibrated
        prediction to a measurement is not a test of anything.

        So one execution is used to pin the temporary-impact coefficient, and the
        model is then asked to predict the *rest* of the curve. That is the real
        question: given one point, does the shape follow?

    Method
        Execute a TWAP of ``quantity`` and solve for the ``temporary_impact``
        that reproduces its realised cost, holding the other terms fixed.
    """
    from quantos.execution.almgren_chriss import twap_trajectory

    profile = profile or LiquidityProfile()
    arrival_price = arrival_ticks * profile.tick
    notional = quantity * arrival_price

    seed_parameters = ImpactParameters(
        volatility=volatility,
        temporary_impact=1e-6,
        permanent_impact=1e-7,
        spread_cost=0.5 * profile.tick,
    )
    reference = twap_trajectory(quantity, horizon, seed_parameters, n_steps=n_periods)
    measured = execute_trajectory(
        reference, profile=profile, arrival_ticks=arrival_ticks, strategy="calibration"
    )
    measured_cost = measured.realised_shortfall_bps / 1e4 * notional

    # expected_cost = eta * sum(trades^2)/dt + permanent + spread; solve for eta.
    trades = np.asarray(reference.trades, dtype=float)
    dt = horizon / n_periods
    quadratic = float(np.sum(trades**2) / dt)
    permanent = float(0.5 * seed_parameters.permanent_impact * quantity**2)
    spread = float(seed_parameters.spread_cost * np.sum(np.abs(trades)))

    eta = max((measured_cost - permanent - spread) / quadratic, 1e-12)
    return ImpactParameters(
        volatility=volatility,
        temporary_impact=eta,
        permanent_impact=seed_parameters.permanent_impact,
        spread_cost=seed_parameters.spread_cost,
    )


def compare_strategies(
    quantity: int,
    *,
    n_periods: int = 20,
    horizon: float = 1.0,
    volatility: float = 0.02,
    profile: LiquidityProfile | None = None,
    urgencies: tuple[float, ...] = (1e-3, 1e-2, 1e-1),
    arrival_ticks: int = 10_000,
    calibrate: bool = True,
) -> dict[str, object]:
    """Run several schedules through the same book and report what happened.

    The comparison is the point. Each strategy faces an identical book, so any
    difference in realised cost comes from the schedule alone -- which is exactly
    the claim Almgren-Chriss makes and which a closed form cannot verify about
    itself.

    Units
        ``ExecutionTrajectory.expected_cost`` is a total in currency, so it is
        converted to basis points of arrival notional before being compared with
        the realised shortfall. Comparing the two without that conversion would
        produce a "prediction error" that is really a units mismatch -- the kind
        of bug that survives review because both numbers look plausible.
    """
    from quantos.execution.almgren_chriss import almgren_chriss_trajectory, twap_trajectory

    profile = profile or LiquidityProfile()
    arrival_price = arrival_ticks * profile.tick
    notional = quantity * arrival_price

    # Urgencies are chosen so the schedules actually differ. With kappa*T below
    # about 0.5 the sinh trajectory is indistinguishable from a straight line, so
    # a "comparison" across small risk aversions compares four copies of TWAP --
    # which an earlier version of this function did, reporting a 0.01 bps spread
    # between strategies and calling it a result.
    parameters = (
        calibrate_impact(
            quantity,
            profile=profile,
            volatility=volatility,
            horizon=horizon,
            n_periods=n_periods,
            arrival_ticks=arrival_ticks,
        )
        if calibrate
        else ImpactParameters(
            volatility=volatility,
            temporary_impact=1e-6,
            permanent_impact=1e-7,
            spread_cost=0.5 * profile.tick,
        )
    )

    def as_bps(trajectory: ExecutionTrajectory) -> float:
        return float(trajectory.expected_cost) / notional * 1e4 if notional else float("nan")

    outcomes: list[ExecutionOutcome] = []

    twap = twap_trajectory(quantity, horizon, parameters, n_steps=n_periods)
    outcomes.append(
        execute_trajectory(
            twap,
            profile=profile,
            arrival_ticks=arrival_ticks,
            predicted_shortfall_bps=as_bps(twap),
            strategy="TWAP",
        )
    )

    for urgency in urgencies:
        trajectory = almgren_chriss_trajectory(
            quantity, horizon, parameters, risk_aversion=urgency, n_steps=n_periods
        )
        outcomes.append(
            execute_trajectory(
                trajectory,
                profile=profile,
                arrival_ticks=arrival_ticks,
                predicted_shortfall_bps=as_bps(trajectory),
                strategy=f"Almgren-Chriss (lambda={urgency:.0e})",
            )
        )

    realised = {o.strategy: o.realised_shortfall_bps for o in outcomes}
    best = min(realised, key=lambda k: realised[k])
    twap_cost = realised.get("TWAP", float("nan"))
    beat_twap = [k for k, v in realised.items() if k != "TWAP" and v < twap_cost]

    # Did the schedules actually differ? If front-loading is identical there is
    # nothing to compare and any ranking is noise.
    spread_bps = max(realised.values()) - min(realised.values())
    differentiated = spread_bps > 0.5

    errors = [abs(o.prediction_error_bps) for o in outcomes if np.isfinite(o.prediction_error_bps)]
    mean_error = float(np.mean(errors)) if errors else float("nan")

    if not differentiated:
        verdict = (
            f"the schedules are indistinguishable ({spread_bps:.2f} bps apart), so this "
            "comparison establishes nothing -- raise the risk aversion until the "
            "trajectories differ"
        )
    else:
        verdict = (
            f"schedules span {spread_bps:.2f} bps; cheapest is {best} at "
            f"{realised[best]:.2f} bps. After calibrating on one execution the model's "
            f"mean absolute error across the rest is {mean_error:.2f} bps"
        )

    return {
        "outcomes": outcomes,
        "realised_bps": realised,
        "cheapest": best,
        "beat_twap": beat_twap,
        "spread_bps": spread_bps,
        "differentiated": differentiated,
        "mean_absolute_error_bps": mean_error,
        "parameters": parameters,
        "verdict": verdict,
    }
