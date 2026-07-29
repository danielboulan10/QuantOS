"""Validation for American option pricing.

The benchmark test anchors the method against a published number. The rest test
the two things that make this module worth having: that the price is *bracketed*
rather than asserted, and that the in-sample bias -- the easiest and most
flattering mistake in least-squares Monte Carlo -- is measured rather than
avoided by luck.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.derivatives.american import (
    longstaff_schwartz,
    price_american,
    simulate_gbm_paths,
)
from quantos.derivatives.black_scholes import OptionType


# --------------------------------------------------------------------------- #
# Path simulation
# --------------------------------------------------------------------------- #
def test_paths_start_at_spot_and_have_the_right_shape():
    paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 1000)
    assert paths.shape == (1000, 51)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_the_risk_neutral_drift_is_recovered():
    r"""Under Q the discounted price is a martingale: E[S_T] = S e^{(r-q)T}."""
    spot, rate, expiry = 100.0, 0.05, 1.0
    paths = simulate_gbm_paths(spot, rate, 0.2, expiry, 50, 200_000)
    assert float(np.mean(paths[:, -1])) == pytest.approx(spot * np.exp(rate * expiry), rel=0.01)


def test_antithetic_sampling_reduces_the_standard_error():
    """Pairing each draw with its negation removes odd sampling error exactly.

    The standard error must be computed on the **pair averages**, not on all
    draws pooled. An earlier version of this test took the naive standard
    deviation over every path and measured 0.06426 with antithetic sampling
    against 0.06424 without -- no benefit whatsoever, because that statistic
    cannot see the pairing it is meant to measure. Averaging within pairs first
    gives 0.04907, a 24% reduction.
    """

    def estimator_error(antithetic: bool) -> float:
        paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 20_000, seed=3, antithetic=antithetic)
        payoff = np.maximum(100.0 - paths[:, -1], 0.0)
        if antithetic:
            half = payoff.size // 2
            payoff = 0.5 * (payoff[:half] + payoff[half:])
        return float(np.std(payoff, ddof=1) / np.sqrt(payoff.size))

    assert estimator_error(True) < 0.9 * estimator_error(False)


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def test_matches_the_longstaff_schwartz_published_value():
    """Longstaff-Schwartz (2001) Table 1: S=36, K=40, r=6%, sigma=20%, T=1 -> 4.478.

    An independently published number is the strongest check available for a
    method with no closed form.
    """
    result = price_american(
        36.0, 40.0, 1.0, 0.20, rate=0.06, n_paths=40_000, n_steps=50, compute_upper=False
    )
    assert result.lower == pytest.approx(4.478, abs=3 * result.lower_standard_error + 0.02)


# --------------------------------------------------------------------------- #
# Structural identities
# --------------------------------------------------------------------------- #
def test_an_american_call_without_dividends_equals_the_european_call():
    """The textbook identity: early exercise is never optimal, so there is no premium.

    Any implementation that exercises early here is finding value that does not
    exist, which is the failure mode of a badly conditioned regression.
    """
    result = price_american(
        100.0,
        100.0,
        1.0,
        0.25,
        rate=0.05,
        option_type=OptionType.CALL,
        n_paths=40_000,
        n_steps=50,
        compute_upper=False,
    )
    assert result.early_exercise_premium == pytest.approx(0.0, abs=3 * result.lower_standard_error)
    assert any("never exercised early" in note for note in result.notes)


def test_a_dividend_makes_early_exercise_worth_something():
    """Guards the test above by removing the condition that made the premium zero."""
    result = price_american(
        100.0,
        100.0,
        1.0,
        0.25,
        rate=0.05,
        dividend_yield=0.08,
        option_type=OptionType.CALL,
        n_paths=40_000,
        n_steps=50,
        compute_upper=False,
    )
    assert result.early_exercise_premium > 0.1


def test_an_american_put_is_worth_more_than_the_european_put():
    """Early exercise always has value for a put, because the strike earns interest."""
    result = price_american(
        90.0, 100.0, 1.0, 0.25, rate=0.06, n_paths=40_000, n_steps=50, compute_upper=False
    )
    assert result.lower > result.european
    assert result.early_exercise_premium > 0


def test_the_price_never_falls_below_intrinsic():
    result = price_american(
        80.0, 100.0, 1.0, 0.25, rate=0.05, n_paths=20_000, n_steps=50, compute_upper=False
    )
    assert result.lower >= 20.0 - 3 * result.lower_standard_error


def test_the_put_price_rises_with_volatility():
    prices = [
        price_american(
            100.0, 100.0, 1.0, sigma, rate=0.05, n_paths=20_000, n_steps=40, compute_upper=False
        ).lower
        for sigma in (0.15, 0.25, 0.40)
    ]
    assert prices == sorted(prices)


# --------------------------------------------------------------------------- #
# The bracket
# --------------------------------------------------------------------------- #
def test_the_price_is_bracketed_by_the_two_bounds():
    """The point of the module: an interval, not a point.

    The lower bound follows a suboptimal rule, so it under-prices. The dual
    construction hedges away the timing decision, so it over-prices. The true
    value lies between them.
    """
    result = price_american(
        36.0, 40.0, 1.0, 0.20, rate=0.06, n_paths=20_000, n_steps=40, n_inner=30
    )
    assert np.isfinite(result.upper)
    assert result.upper >= result.lower - 3 * (
        result.lower_standard_error + result.upper_standard_error
    )
    assert result.duality_gap == pytest.approx(result.upper - result.lower)
    # The published value must sit inside the bracket.
    assert result.lower - 0.05 <= 4.478 <= result.upper + 0.05


def test_the_bracket_contains_the_european_price_from_below():
    result = price_american(
        36.0, 40.0, 1.0, 0.20, rate=0.06, n_paths=20_000, n_steps=40, n_inner=20
    )
    assert result.european < result.lower, "an American put dominates its European counterpart"


# --------------------------------------------------------------------------- #
# The in-sample trap
# --------------------------------------------------------------------------- #
def test_fitting_the_rule_on_the_valuation_paths_biases_the_price_high():
    r"""The easy, flattering mistake, measured.

    LSMC is biased *low* only when the regression is fitted on separate paths.
    Reuse the valuation paths and the regression has seen each path's own future:
    the exercise rule looks prescient and the price goes up. Nothing raises, and
    the number moves in the direction that makes the result look better.
    """
    result = price_american(
        36.0, 40.0, 1.0, 0.20, rate=0.06, n_paths=20_000, n_steps=50, compute_upper=False
    )
    assert np.isfinite(result.in_sample_price)
    assert result.in_sample_bias > 0, "the in-sample price must be the higher one"
    assert "biased HIGH" in result.summary()


def test_the_out_of_sample_price_is_the_one_reported_as_the_lower_bound():
    """The default must be the honest estimator, not the flattering one."""
    result = price_american(
        36.0, 40.0, 1.0, 0.20, rate=0.06, n_paths=20_000, n_steps=50, compute_upper=False
    )
    assert result.lower < result.in_sample_price


def test_the_in_sample_bias_is_positive_on_average_across_seeds():
    """The bias is real in expectation and smaller than run-to-run noise.

    Measured over twelve seeds: mean +0.0056 with a standard deviation of 0.0055,
    positive on eleven of twelve. So a single-seed assertion is wrong roughly one
    time in twelve -- an earlier version of this test asserted it per-seed and
    duly failed. Averaging is the correct comparison, and the ratio of the bias to
    its own spread is itself the reason the separate regression sample matters:
    the effect is easy to miss and always points the flattering way.
    """
    differences = []
    for seed in range(12):
        valuation = simulate_gbm_paths(36.0, 0.06, 0.20, 1.0, 50, 20_000, seed=seed)
        fitting = simulate_gbm_paths(36.0, 0.06, 0.20, 1.0, 50, 20_000, seed=seed + 500)
        in_sample, _, _ = longstaff_schwartz(valuation, 40.0, 0.06, 1.0)
        out_of_sample, _, _ = longstaff_schwartz(
            valuation, 40.0, 0.06, 1.0, regression_paths=fitting
        )
        differences.append(in_sample - out_of_sample)

    bias = np.array(differences)
    assert float(bias.mean()) > 0, "in-sample valuation must inflate the price on average"
    assert int((bias > 0).sum()) >= 9, "and do so on most individual seeds"


# --------------------------------------------------------------------------- #
# The learned rule
# --------------------------------------------------------------------------- #
def test_the_exercise_rule_only_ever_exercises_in_the_money():
    paths = simulate_gbm_paths(36.0, 0.06, 0.20, 1.0, 50, 5_000, seed=4)
    _, rule, _ = longstaff_schwartz(paths, 40.0, 0.06, 1.0)

    prices = np.linspace(20.0, 60.0, 200)
    for step in (10, 25, 45):
        exercising = prices[rule.should_exercise(step, prices)]
        if exercising.size:
            assert float(np.max(exercising)) <= 40.0, "a put must be in the money to exercise"


def test_the_exercise_boundary_is_below_the_strike_and_rises_toward_expiry():
    """Economically the boundary must approach the strike as time runs out."""
    paths = simulate_gbm_paths(36.0, 0.06, 0.20, 1.0, 50, 20_000, seed=5)
    _, rule, _ = longstaff_schwartz(paths, 40.0, 0.06, 1.0)

    prices = np.linspace(20.0, 40.0, 400)

    def boundary(step: int) -> float:
        exercising = prices[rule.should_exercise(step, prices)]
        return float(np.max(exercising)) if exercising.size else float("nan")

    early, late = boundary(5), boundary(48)
    assert np.isfinite(early) and np.isfinite(late)
    assert early <= 40.0 and late <= 40.0
    assert late >= early - 1e-9, "the boundary should not fall as expiry approaches"
