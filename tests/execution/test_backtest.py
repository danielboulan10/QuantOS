"""Execution theory against the actual matching engine.

The interesting test is the last one. It documents a case where the closed-form
model gets the *ordering of strategies wrong*, and shows exactly which of its
assumptions is responsible -- which is what routing orders through an independent
mechanism is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.execution.almgren_chriss import ImpactParameters, twap_trajectory
from quantos.execution.backtest import (
    LiquidityProfile,
    calibrate_impact,
    compare_strategies,
    execute_trajectory,
)


def parameters(**kwargs) -> ImpactParameters:
    base = {
        "volatility": 0.02,
        "temporary_impact": 1e-6,
        "permanent_impact": 1e-7,
        "spread_cost": 0.005,
    }
    return ImpactParameters(**{**base, **kwargs})


# --------------------------------------------------------------------------- #
# The mechanism
# --------------------------------------------------------------------------- #
def test_an_execution_fills_and_costs_something():
    trajectory = twap_trajectory(5_000, 1.0, parameters(), n_steps=10)
    outcome = execute_trajectory(trajectory, strategy="TWAP")

    assert outcome.filled > 0
    assert outcome.average_price > outcome.arrival_price, "buying walks up the book"
    assert outcome.realised_shortfall_bps > 0


def test_a_larger_order_costs_more():
    """Walking further into the book must be more expensive per share."""
    small = execute_trajectory(twap_trajectory(2_000, 1.0, parameters(), n_steps=10))
    large = execute_trajectory(twap_trajectory(30_000, 1.0, parameters(), n_steps=10))
    assert large.realised_shortfall_bps > small.realised_shortfall_bps


def test_more_children_costs_less_when_liquidity_replenishes():
    """Splitting only helps because consumed depth comes back.

    Both executions must complete before their costs can be compared. An earlier
    version of this test used an order large enough that the four-child schedule
    exhausted the book at 3,312 of 20,000 -- and its shortfall then looked
    *cheaper*, because a partial fill only pays for the cheap part.
    """
    few = execute_trajectory(twap_trajectory(4_000, 1.0, parameters(), n_steps=4))
    many = execute_trajectory(twap_trajectory(4_000, 1.0, parameters(), n_steps=40))

    assert few.complete and many.complete, "an incomplete fill is not comparable"
    assert many.realised_shortfall_bps < few.realised_shortfall_bps


def test_a_partial_fill_is_flagged_as_incomplete():
    """So it can never be silently compared against a finished execution."""
    thin = LiquidityProfile(depth_at_touch=20, levels=5, replenish_rate=0.0)
    outcome = execute_trajectory(
        twap_trajectory(50_000, 1.0, parameters(), n_steps=5), profile=thin
    )
    assert not outcome.complete
    assert "PARTIAL" in outcome.summary()


def test_without_replenishment_splitting_buys_nothing():
    """Guards the test above by removing the mechanism that makes it true."""
    static = LiquidityProfile(replenish_rate=0.0, permanent_impact_ticks=0.0)
    few = execute_trajectory(twap_trajectory(2_000, 1.0, parameters(), n_steps=4), profile=static)
    many = execute_trajectory(twap_trajectory(2_000, 1.0, parameters(), n_steps=40), profile=static)
    assert few.complete and many.complete
    assert many.realised_shortfall_bps == pytest.approx(few.realised_shortfall_bps, rel=0.05)


def test_exhausting_the_book_is_reported_not_raised():
    thin = LiquidityProfile(depth_at_touch=20, levels=5, replenish_rate=0.0)
    outcome = execute_trajectory(
        twap_trajectory(50_000, 1.0, parameters(), n_steps=5), profile=thin
    )
    assert outcome.filled < outcome.quantity
    assert any("exhausted" in note for note in outcome.notes)


def test_an_empty_trajectory_is_refused():
    class Empty:
        trades = np.zeros(5)

    with pytest.raises(ValueError, match="no trades"):
        execute_trajectory(Empty())


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_makes_the_model_reproduce_its_own_observation():
    """After fitting on one execution the prediction must match that execution.

    Comparing an uncalibrated prediction with a measurement tests units, not
    theory: before this step the model predicted 2.60 bps where the book charged
    23.37, a 9x gap that said nothing about whether the theory was right.
    """
    profile = LiquidityProfile()
    fitted = calibrate_impact(20_000, profile=profile)
    assert fitted.temporary_impact > 0

    result = compare_strategies(20_000, n_periods=20, profile=profile)
    twap = next(o for o in result["outcomes"] if o.strategy == "TWAP")
    assert abs(twap.prediction_error_bps) < 0.01


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def test_the_comparison_detects_when_schedules_are_indistinguishable():
    """A ranking over four copies of TWAP is noise, and must be reported as such.

    An earlier version used risk aversions so small that every trajectory was a
    straight line, then reported a 0.01 bps difference as though it meant
    something.
    """
    result = compare_strategies(
        10_000, n_periods=20, urgencies=(1e-9, 1e-8, 1e-7), profile=LiquidityProfile()
    )
    assert not result["differentiated"]
    assert "establishes nothing" in result["verdict"]


def test_realistic_urgencies_do_differentiate():
    result = compare_strategies(20_000, n_periods=20, urgencies=(1e-3, 1e-2, 1e-1))
    assert result["differentiated"]
    assert result["spread_bps"] > 1.0


# --------------------------------------------------------------------------- #
# The finding
# --------------------------------------------------------------------------- #
def test_the_model_misranks_schedules_when_permanent_impact_is_schedule_dependent():
    r"""Almgren-Chriss's permanent term is schedule-invariant. Real books are not.

    The closed form charges :math:`\tfrac12 \lambda X^2` for permanent impact,
    which depends only on total size -- so the theory says front-loading is
    always more expensive, paying extra temporary impact for reduced timing risk.

    This book instead drifts upward with *cumulative* volume, so trading late is
    penalised. Measured on identical books:

    ==========================  =========  ============  ====================
    Book's permanent drift      TWAP       Aggressive    Front-loading
    ==========================  =========  ============  ====================
    0.002 ticks per contract    23.37 bps  7.85 bps      **helps** by 15.52
    0.000 (the AC assumption)   4.38 bps   6.53 bps      hurts by 2.15
    ==========================  =========  ============  ====================

    With the drift removed the model's ordering is restored, which localises the
    disagreement precisely: it is not that Almgren-Chriss is wrong, it is that
    one of its assumptions does not hold on a venue whose permanent impact
    accrues with volume already done.
    """
    drifting = compare_strategies(
        20_000, n_periods=20, profile=LiquidityProfile(permanent_impact_ticks=0.002)
    )["realised_bps"]
    flat = compare_strategies(
        20_000, n_periods=20, profile=LiquidityProfile(permanent_impact_ticks=0.0)
    )["realised_bps"]

    def aggressive(costs: dict[str, float]) -> float:
        return next(v for k, v in costs.items() if "1e-01" in k)

    # With drift, front-loading wins -- contradicting the model.
    assert aggressive(drifting) < drifting["TWAP"] - 5.0

    # Without it, the model's ordering holds.
    assert aggressive(flat) > flat["TWAP"]


def test_the_prediction_error_is_reported_rather_than_hidden():
    result = compare_strategies(20_000, n_periods=20)
    assert np.isfinite(result["mean_absolute_error_bps"])
    assert any(abs(o.prediction_error_bps) > 1.0 for o in result["outcomes"]), (
        "the aggressive schedule should visibly diverge from the prediction"
    )
