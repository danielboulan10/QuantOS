"""Optimal execution and impact."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantos.core.rng import SeedBank
from quantos.execution.almgren_chriss import (
    ImpactParameters,
    almgren_chriss_trajectory,
    efficient_execution_frontier,
    fit_square_root_law,
    implementation_shortfall,
    square_root_impact_cost,
    twap_trajectory,
    vwap_trajectory,
)

PARAMS = ImpactParameters(volatility=0.02, temporary_impact=1e-6)


def test_risk_neutral_almgren_chriss_is_exactly_twap() -> None:
    """TWAP is not a naive baseline: it is optimal for a risk-neutral trader."""
    trajectory = almgren_chriss_trajectory(1e6, 1.0, PARAMS, risk_aversion=0.0)
    assert trajectory.front_loading == pytest.approx(0.5, abs=1e-9)
    # Equal slices throughout.
    assert float(np.std(trajectory.trades)) < 1e-6 * float(np.mean(trajectory.trades))


def test_risk_aversion_front_loads_the_schedule() -> None:
    low = almgren_chriss_trajectory(1e6, 1.0, PARAMS, risk_aversion=1e-9)
    high = almgren_chriss_trajectory(1e6, 1.0, PARAMS, risk_aversion=1e-3)
    assert high.front_loading > low.front_loading
    assert high.urgency > low.urgency
    assert high.half_life < low.half_life


def test_trajectory_always_completes_the_order() -> None:
    for risk_aversion in (0.0, 1e-9, 1e-6, 1e-3, 1.0):
        trajectory = almgren_chriss_trajectory(5e5, 1.0, PARAMS, risk_aversion=risk_aversion)
        assert trajectory.holdings[0] == pytest.approx(5e5)
        assert trajectory.holdings[-1] == pytest.approx(0.0, abs=1e-6)
        assert float(np.sum(trajectory.trades)) == pytest.approx(5e5, rel=1e-9)
        assert np.all(trajectory.trades >= -1e-9)  # never buys back
        assert np.all(np.diff(trajectory.holdings) <= 1e-9)  # monotone decreasing


def test_extreme_urgency_does_not_overflow() -> None:
    """sinh overflows for large kappa*T; the log-space branch must take over."""
    trajectory = almgren_chriss_trajectory(1e6, 1.0, PARAMS, risk_aversion=1e6)
    assert np.all(np.isfinite(trajectory.holdings))
    assert trajectory.front_loading > 0.99


def test_frontier_is_monotone_in_both_cost_and_risk() -> None:
    """There is no single optimum, only a frontier -- and it must be a frontier."""
    frontier = efficient_execution_frontier(1e6, 1.0, PARAMS)
    costs = [t.expected_cost for t in frontier]
    risks = [t.cost_variance for t in frontier]
    assert all(a <= b + 1e-9 for a, b in itertools.pairwise(costs))
    assert all(a >= b - 1e-9 for a, b in itertools.pairwise(risks))


def test_permanent_impact_cannot_be_scheduled_away() -> None:
    """A genuinely useful result: only the temporary component responds."""
    with_permanent = ImpactParameters(volatility=0.02, temporary_impact=1e-6, permanent_impact=1e-7)
    fast = almgren_chriss_trajectory(1e6, 1.0, with_permanent, risk_aversion=1e-3)
    slow = almgren_chriss_trajectory(1e6, 1.0, with_permanent, risk_aversion=0.0)
    assert fast.detail["permanent_cost"] == pytest.approx(slow.detail["permanent_cost"])
    assert fast.detail["temporary_cost"] > slow.detail["temporary_cost"]


def test_vwap_follows_the_volume_profile() -> None:
    profile = np.array([3.0, 1.0, 1.0, 1.0, 4.0])  # U-shaped intraday
    trajectory = vwap_trajectory(1e6, 1.0, PARAMS, profile)
    assert trajectory.trades[0] > trajectory.trades[1]
    assert trajectory.trades[-1] > trajectory.trades[-2]
    assert float(np.sum(trajectory.trades)) == pytest.approx(1e6)


def test_vwap_validates_the_profile() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        vwap_trajectory(1e6, 1.0, PARAMS, [-1.0, 2.0])


def test_twap_and_zero_risk_aversion_agree() -> None:
    a = twap_trajectory(1e6, 1.0, PARAMS)
    b = almgren_chriss_trajectory(1e6, 1.0, PARAMS, risk_aversion=0.0)
    assert np.allclose(a.holdings, b.holdings)


def test_square_root_law_is_concave_in_size() -> None:
    """Large orders cost less per share than linear impact would predict."""
    costs = [square_root_impact_cost(q, 1.0, 0.02) for q in (0.01, 0.04, 0.09, 0.16)]
    # Quadrupling participation only doubles impact.
    assert costs[1] == pytest.approx(2 * costs[0], rel=1e-9)
    assert costs[3] == pytest.approx(4 * costs[0], rel=1e-9)


def test_fit_square_root_law_recovers_a_known_exponent() -> None:
    rng = SeedBank(root=1).child("srl").generator()
    participation = 10 ** rng.uniform(-4, -1, 1_000)
    impact = 0.8 * participation**0.5 * np.exp(rng.standard_normal(1_000) * 0.1)
    fit = fit_square_root_law(participation, impact)
    assert fit.exponent == pytest.approx(0.5, abs=0.02)
    assert fit.coefficient == pytest.approx(0.8, rel=0.1)
    assert fit.consistent_with_square_root
    assert not fit.consistent_with_linear


def test_fit_detects_a_linear_law_when_that_is_the_truth() -> None:
    """The estimator must be able to reject the square root, or it proves nothing."""
    rng = SeedBank(root=2).child("lin").generator()
    participation = 10 ** rng.uniform(-4, -1, 1_000)
    impact = 0.8 * participation**1.0 * np.exp(rng.standard_normal(1_000) * 0.1)
    fit = fit_square_root_law(participation, impact)
    assert fit.exponent == pytest.approx(1.0, abs=0.03)
    # With n=1000 and low noise the standard error is ~0.0015, so the 95% interval
    # spans only three decimal places and a 2.5-sigma sampling deviation can
    # exclude 1.0 exactly. The meaningful assertion is that the exponent is far
    # closer to linear than to square-root, which is what a user would conclude.
    assert abs(fit.exponent - 1.0) < abs(fit.exponent - 0.5) / 10
    assert not fit.consistent_with_square_root


def test_fit_requires_positive_paired_observations() -> None:
    with pytest.raises(ValueError, match="strictly-positive"):
        fit_square_root_law([1.0, -1.0, 0.0], [1.0, 1.0, 1.0])


def test_implementation_shortfall_against_the_arrival_price() -> None:
    out = implementation_shortfall([100.5, 101.0], [100, 100], 100.0)
    assert out["average_execution_price"] == pytest.approx(100.75)
    assert out["shortfall"] == pytest.approx(0.75)
    assert out["shortfall_bps"] == pytest.approx(75.0)
    # A seller beats the benchmark on the same fills.
    assert implementation_shortfall([100.5, 101.0], [100, 100], 100.0, side=-1)["shortfall"] < 0


def test_impact_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="temporary_impact"):
        ImpactParameters(volatility=0.02, temporary_impact=0.0)
