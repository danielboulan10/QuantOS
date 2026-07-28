"""Validation for the VaR exceedance tests.

Testing a test needs known answers, so the sequences here are constructed with
the property being detected: a correct model, one that breaches too often, and
one that breaches the right number of times but all at once. The last is the case
Kupiec cannot see and Christoffersen exists for.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.risk.var_backtest import (
    backtest_var,
    christoffersen_independence,
    evt_value_at_risk,
    fit_generalised_pareto,
    kupiec_coverage,
)


# --------------------------------------------------------------------------- #
# Kupiec
# --------------------------------------------------------------------------- #
def test_a_correct_breach_rate_is_not_rejected():
    rng = np.random.default_rng(0)
    breaches = rng.random(5000) < 0.01
    assert not kupiec_coverage(breaches, 0.99).rejects


def test_too_many_breaches_are_rejected():
    rng = np.random.default_rng(1)
    assert kupiec_coverage(rng.random(5000) < 0.05, 0.99).rejects


def test_too_few_breaches_are_also_rejected():
    """An over-conservative VaR is wrong too, and wastes capital."""
    rng = np.random.default_rng(2)
    assert kupiec_coverage(rng.random(5000) < 0.0005, 0.99).rejects


def test_zero_breaches_does_not_crash_and_still_rejects():
    """The likelihood ratio has a 0*log(0) term that must be handled, not NaN."""
    result = kupiec_coverage(np.zeros(3000, dtype=bool), 0.99)
    assert np.isfinite(result.statistic)
    assert result.rejects


def test_the_size_of_the_kupiec_test_is_close_to_nominal():
    """A test that over-rejects correct models is worse than no test.

    This is the check most often skipped, and the one that decides whether a
    rejection means anything.
    """
    rng = np.random.default_rng(3)
    rejections = sum(kupiec_coverage(rng.random(2000) < 0.01, 0.99).rejects for _ in range(300))
    rate = rejections / 300
    assert rate < 0.12, f"empirical size {rate:.1%} at a nominal 5%"


def test_an_empty_series_reports_nan_rather_than_raising():
    assert np.isnan(kupiec_coverage(np.zeros(0, dtype=bool), 0.99).statistic)


# --------------------------------------------------------------------------- #
# Christoffersen -- the test that earns its place
# --------------------------------------------------------------------------- #
def test_independent_breaches_are_not_rejected():
    rng = np.random.default_rng(4)
    assert not christoffersen_independence(rng.random(5000) < 0.01).rejects


def test_clustered_breaches_are_rejected_even_at_the_right_rate():
    """The whole point. The count is correct; the pattern is not.

    Twenty breaches in 2,000 days is exactly 1%, so Kupiec passes. Putting them
    all in one run is a model that was wrong for a month and right otherwise --
    and the month is the only part anyone cares about.
    """
    breaches = np.zeros(2000, dtype=bool)
    breaches[900:920] = True  # every breach consecutive

    assert not kupiec_coverage(breaches, 0.99).rejects, "the count is correct"
    assert christoffersen_independence(breaches).rejects, "but the clustering is not"


def test_the_clustered_case_is_caught_by_conditional_coverage():
    breaches = np.zeros(2000, dtype=bool)
    breaches[900:920] = True
    returns = np.where(breaches, -0.10, 0.001)
    var = np.full(2000, 0.05)

    result = backtest_var(returns, var, confidence=0.99, model="clustered")
    assert not result.kupiec.rejects
    assert result.independence.rejects
    assert result.conditional_coverage.rejects
    assert "cluster" in result.verdict


def test_independence_statistic_is_never_negative():
    """Numerical noise can push a likelihood ratio a hair below zero."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        breaches = rng.random(500) < 0.02
        assert christoffersen_independence(breaches).statistic >= 0.0


def test_a_single_observation_reports_nan():
    assert np.isnan(christoffersen_independence(np.array([True])).statistic)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_a_correctly_specified_gaussian_var_passes_on_gaussian_data():
    """Guards every rejection below: the battery must not reject a correct model."""
    from quantos.core.special import ndtri

    rng = np.random.default_rng(6)
    sigma = 0.01
    returns = rng.normal(0, sigma, 6000)
    var = np.full(6000, float(ndtri(np.array(0.99))) * sigma)

    result = backtest_var(returns, var, confidence=0.99, model="correct")
    assert not result.kupiec.rejects
    assert not result.independence.rejects
    assert result.verdict == "passes both tests"


def test_a_gaussian_var_is_rejected_on_fat_tailed_data():
    """The documented empirical failure, reproduced on synthetic data.

    The returns are standardised to unit variance, so the rejection is caused by
    tail *shape* alone rather than by a scale mismatch. That makes the effect
    small -- a standardised t(3) has a 99% quantile only about 13% beyond the
    Gaussian one, giving roughly a 1.2% breach rate against 1% -- so it needs a
    long sample to detect. At 6,000 observations the test returned p = 0.08 and
    correctly did not reject; the effect is real but the evidence was not there.
    """
    from quantos.core.special import ndtri

    rng = np.random.default_rng(7)
    df = 3.0
    n = 40_000
    returns = rng.standard_t(df, n) / np.sqrt(df / (df - 2)) * 0.01
    var = np.full(n, float(ndtri(np.array(0.99))) * 0.01)

    result = backtest_var(returns, var, confidence=0.99, model="gaussian on fat tails")
    assert result.breach_rate > 0.011
    assert result.kupiec.rejects


def test_a_short_sample_is_flagged_as_underpowered():
    rng = np.random.default_rng(8)
    result = backtest_var(rng.normal(0, 0.01, 100), np.full(100, 0.03), confidence=0.99)
    assert any("too few" in note for note in result.notes)


def test_unusable_forecasts_are_dropped_rather_than_failing():
    returns = np.array([0.01, -0.02, 0.03, -0.04])
    var = np.array([0.05, np.nan, -1.0, 0.05])
    assert backtest_var(returns, var, confidence=0.99).n_observations == 2


# --------------------------------------------------------------------------- #
# Extreme value theory
# --------------------------------------------------------------------------- #
def test_the_gpd_fit_recovers_a_known_shape():
    """Parameter recovery on data drawn from the distribution being fitted."""
    rng = np.random.default_rng(9)
    shape, scale = 0.25, 1.0
    uniform = rng.random(40_000)
    sample = scale / shape * ((1 - uniform) ** -shape - 1)

    fit = fit_generalised_pareto(sample, threshold_quantile=0.90)
    assert fit.shape == pytest.approx(shape, abs=0.08)
    assert fit.n_exceedances > 3000


def test_a_thin_tail_fits_a_nonpositive_shape():
    rng = np.random.default_rng(10)
    fit = fit_generalised_pareto(rng.exponential(1.0, 20_000))
    assert fit.shape == pytest.approx(0.0, abs=0.08)
    assert "thin" in fit.tail_verdict or "heavy" in fit.tail_verdict


def test_the_shape_parameter_reports_which_moments_exist():
    rng = np.random.default_rng(11)
    uniform = rng.random(30_000)
    heavy = 1.0 / 0.6 * ((1 - uniform) ** -0.6 - 1)  # shape 0.6 -> no variance
    fit = fit_generalised_pareto(heavy)
    assert fit.shape > 0.4
    assert "do not exist" in fit.tail_verdict


def test_the_historical_quantile_is_capped_by_its_worst_observation_and_evt_is_not():
    """The structural difference, stated precisely.

    "EVT gives a larger number" is not true at every confidence: at 99.9% on this
    sample the empirical quantile sits on an unusually extreme order statistic
    (0.1054) while the smoothed fit gives 0.0926. An earlier version of this test
    asserted the wrong thing and failed.

    What *is* always true is the bound. A historical estimator can never exceed
    the worst loss it has seen, so it flattens out exactly where the tail matters
    most; the fitted tail keeps extrapolating.
    """
    rng = np.random.default_rng(12)
    returns = rng.standard_t(3.0, 5000) * 0.01
    worst = float(-np.min(returns))

    for confidence in (0.9999, 0.99999):
        historical = float(-np.quantile(returns, 1 - confidence))
        evt = evt_value_at_risk(returns, confidence=confidence)
        assert historical <= worst + 1e-12, "the empirical quantile cannot exceed the sample"
        assert evt > historical, f"EVT should extrapolate beyond it at {confidence}"

    # And far enough out, EVT exceeds the worst observation outright.
    assert evt_value_at_risk(returns, confidence=0.99999) > worst


def test_too_few_exceedances_returns_nan_rather_than_a_fit_nobody_should_trust():
    fit = fit_generalised_pareto(np.random.default_rng(13).normal(0, 1, 60))
    assert not np.isfinite(fit.shape)


def test_an_evt_var_on_a_short_series_is_nan_not_a_guess():
    assert np.isnan(evt_value_at_risk(np.random.default_rng(14).normal(0, 0.01, 40)))
