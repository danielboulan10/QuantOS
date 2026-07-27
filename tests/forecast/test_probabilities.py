"""Validation for the forward probability outputs.

The properties tested are the ones that make these numbers trustworthy: touching
a level is more likely than finishing beyond it, drawdown is measured peak-to-
trough rather than from the start, direction is honestly reported as a coin flip,
and the short side is riskier than the long side for the same forecast.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.forecast.paths import PathEnsemble, simulate_garch_paths
from quantos.forecast.probabilities import (
    drawdown_probability,
    first_passage_probability,
    long_short_comparison,
    probability_report,
)


def ensemble(n: int = 20_000, horizon: int = 21, sigma: float = 0.02, seed: int = 0):
    returns = np.random.default_rng(seed).normal(0.0, sigma, 1500)
    return simulate_garch_paths(returns, 100.0, horizon, n_paths=n, seed=seed + 1)


# --------------------------------------------------------------------------- #
# First passage versus terminal
# --------------------------------------------------------------------------- #
def test_touching_a_level_is_at_least_as_likely_as_finishing_beyond_it():
    """The defining inequality. A path can breach and recover; the reverse cannot happen."""
    e = ensemble()
    for move in (0.02, 0.05, 0.10, 0.20):
        touch_down = first_passage_probability(e, 100.0 * (1 - move), direction="down")
        finish_down = float(np.mean(e.terminal <= 100.0 * (1 - move)))
        assert touch_down >= finish_down - 1e-12, f"violated at {move:.0%}"

        touch_up = first_passage_probability(e, 100.0 * (1 + move), direction="up")
        finish_up = float(np.mean(e.terminal >= 100.0 * (1 + move)))
        assert touch_up >= finish_up - 1e-12


def test_touch_probability_is_strictly_larger_for_a_near_level():
    """Not just >=: for a level the path is likely to graze, it must be materially larger."""
    e = ensemble()
    touch = first_passage_probability(e, 95.0, direction="down")
    finish = float(np.mean(e.terminal <= 95.0))
    assert touch > finish * 1.2


def test_first_passage_is_monotone_in_the_level():
    e = ensemble()
    probabilities = [
        first_passage_probability(e, 100.0 * (1 - m), direction="down")
        for m in (0.02, 0.05, 0.10, 0.20, 0.40)
    ]
    assert probabilities == sorted(probabilities, reverse=True)


def test_an_unreachable_level_has_probability_zero():
    paths = np.array([[100.0, 101.0, 102.0], [100.0, 99.0, 100.5]])
    e = PathEnsemble(paths=paths, spot=100.0, horizon=2, engine="fixture")
    assert first_passage_probability(e, 50.0, direction="down") == 0.0
    assert first_passage_probability(e, 200.0, direction="up") == 0.0


def test_a_bad_direction_is_refused():
    with pytest.raises(ValueError, match="direction must be"):
        first_passage_probability(ensemble(n=100), 95.0, direction="sideways")


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def test_drawdown_is_measured_peak_to_trough_not_from_the_start():
    """A path that rises then falls has a drawdown even if it finishes up."""
    # Up 20%, then back to 105: a 12.5% drawdown, but a +5% total return.
    paths = np.array([[100.0, 120.0, 105.0]])
    e = PathEnsemble(paths=paths, spot=100.0, horizon=2, engine="fixture")

    assert drawdown_probability(e, 0.10) == 1.0
    assert drawdown_probability(e, 0.20) == 0.0
    assert e.terminal[0] > e.spot  # finished higher, and still drew down


def test_drawdown_probability_is_monotone_in_depth():
    e = ensemble()
    probabilities = [drawdown_probability(e, d) for d in (0.02, 0.05, 0.10, 0.20)]
    assert probabilities == sorted(probabilities, reverse=True)


def test_drawdown_is_at_least_as_likely_as_touching_the_same_level_down():
    """Drawdown from a running peak subsumes a fall from the start."""
    e = ensemble()
    for depth in (0.05, 0.10):
        assert (
            drawdown_probability(e, depth)
            >= first_passage_probability(e, 100.0 * (1 - depth), direction="down") - 1e-12
        )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_depth_is_refused(bad):
    with pytest.raises(ValueError, match="depth must lie"):
        drawdown_probability(ensemble(n=100), bad)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def test_direction_is_a_coin_flip_and_says_so():
    """The honest headline. A driftless simulation must not imply a view."""
    report = probability_report(ensemble(n=40_000), symbol="TEST")
    assert 0.47 < report.probability_up < 0.53
    assert "coin flip" in report.direction_verdict


def test_a_volatile_instrument_carries_more_path_risk_than_a_calm_one():
    """The comparison that makes the numbers useful."""
    calm = probability_report(ensemble(sigma=0.008, seed=5), symbol="CALM")
    wild = probability_report(ensemble(sigma=0.045, seed=5), symbol="WILD")

    assert wild.drawdowns["10%"] > calm.drawdowns["10%"] * 3
    assert wild.touch_thresholds["-10%"] > calm.touch_thresholds["-10%"]
    assert wild.expected_shortfall_95 < calm.expected_shortfall_95


def test_conditional_moments_have_the_right_signs():
    report = probability_report(ensemble(n=40_000), symbol="TEST")
    assert report.expected_loss_given_loss < 0
    assert report.expected_gain_given_gain > 0
    assert report.expected_shortfall_95 <= report.expected_loss_given_loss


def test_terminal_quantiles_are_ordered():
    report = probability_report(ensemble())
    levels = sorted(report.terminal_quantiles)
    values = [report.terminal_quantiles[level] for level in levels]
    assert values == sorted(values)


def test_annualised_volatility_recovers_the_input():
    report = probability_report(ensemble(sigma=0.02, horizon=63, n=40_000))
    assert report.annualised_volatility == pytest.approx(0.02 * np.sqrt(252), rel=0.12)


def test_the_summary_leads_with_direction_then_risk():
    text = probability_report(ensemble(), symbol="TEST").summary()
    assert text.index("DIRECTION") < text.index("WHAT IT MIGHT TOUCH")
    assert "coin flip" in text


def test_risk_verdict_scales_with_volatility():
    calm = probability_report(ensemble(sigma=0.006, seed=9))
    wild = probability_report(ensemble(sigma=0.05, seed=9))
    assert "low" in calm.risk_verdict.lower()
    assert "high" in wild.risk_verdict.lower()


def test_the_risk_verdict_is_horizon_normalised():
    """The same wording must mean the same thing at any horizon.

    Drawdown grows with sqrt(T), so a fixed threshold drifts: a depth that reads
    "moderate" over a month would read "low" over a year purely because the
    window is longer. Severity is judged on the horizon-normalised depth so the
    same volatility earns the same label either way.
    """
    short = probability_report(ensemble(sigma=0.03, horizon=21, seed=21))
    long = probability_report(ensemble(sigma=0.03, horizon=160, seed=21))

    assert long.median_worst_drawdown > short.median_worst_drawdown  # deeper, as expected
    severity = lambda verdict: verdict.split()[0].lower()  # noqa: E731
    assert severity(short.risk_verdict) == severity(long.risk_verdict)


def test_median_worst_drawdown_grows_with_the_horizon():
    short = probability_report(ensemble(horizon=21, seed=22))
    long = probability_report(ensemble(horizon=252, seed=22))
    assert long.median_worst_drawdown > short.median_worst_drawdown


# --------------------------------------------------------------------------- #
# Long versus short
# --------------------------------------------------------------------------- #
def test_the_short_side_has_the_worse_tail():
    """The asymmetry that almost no retail tool shows.

    A short loses without bound while its gain caps at 100%, and in log space the
    fat tail points upward -- against the short.
    """
    comparison = long_short_comparison(ensemble(horizon=63, sigma=0.03), symbol="TEST")
    assert comparison.short_worst_case < comparison.long_worst_case
    assert comparison.short_expected_shortfall_95 < comparison.long_expected_shortfall_95


def test_borrow_cost_reduces_the_shorts_probability_of_profit():
    e = ensemble(horizon=252, sigma=0.02, n=20_000)
    free = long_short_comparison(e, borrow_cost_annual=0.0)
    expensive = long_short_comparison(e, borrow_cost_annual=0.20)
    assert expensive.short_probability_of_profit < free.short_probability_of_profit


def test_a_long_stop_and_a_short_stop_are_different_events():
    """The long's stop is a fall; the short's is a rise. They must not be the same number."""
    comparison = long_short_comparison(ensemble(horizon=63, sigma=0.035, seed=11))
    assert comparison.long_stop_hit_probability > 0
    assert comparison.short_stop_hit_probability > 0
    # For a roughly symmetric log distribution they are close but not identical.
    assert comparison.long_stop_hit_probability != comparison.short_stop_hit_probability


def test_the_asymmetry_verdict_names_the_mechanisms():
    verdict = long_short_comparison(ensemble(), symbol="TEST").asymmetry_verdict
    assert "unbounded" in verdict
    assert "borrow" in verdict
