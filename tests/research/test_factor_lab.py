"""Validation for the factor lab.

Two properties matter here and they pull against each other. The lab must not
find things that are not there -- that is the whole point of it. But a lab that
*always* says no is not a test, it is a constant, and it would pass every
no-false-positive check ever written. So the suite below is built in pairs:
every test that the lab rejects noise is matched by one that it accepts a signal
planted deliberately.

The planted signal is the same one throughout: tomorrow's return depends on the
sign of the trailing 21-day momentum. It is in the grammar
(`momentum_21d_sign_h1`), so a working lab should recover the exact rule, not
merely something correlated with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.research.factor_lab import (
    HOLDS,
    SCALINGS,
    TRANSFORMS,
    WINDOWS,
    FactorSpec,
    compute_signal,
    evaluate_factor,
    generate_factor_grid,
    run_factor_lab,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def planted_signal(*, seed: int = 7, n: int = 2000, edge: float = 0.0022) -> np.ndarray:
    """Returns with a real, exploitable rule: follow 21-day momentum."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * 0.01
    returns = np.zeros(n)
    for t in range(1, n):
        trailing = returns[max(0, t - 21) : t].sum()
        returns[t] = edge * np.sign(trailing) + noise[t]
    return returns


def pure_noise(*, seed: int = 0, n: int = 1500) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n) * 0.01


@pytest.fixture(scope="module")
def noise_report():
    return run_factor_lab(pure_noise(), n_factors=200, seed=1, n_bootstrap=400)


@pytest.fixture(scope="module")
def signal_report():
    return run_factor_lab(planted_signal(), n_factors=200, seed=1, n_bootstrap=400)


# --------------------------------------------------------------------------- #
# The search space
# --------------------------------------------------------------------------- #
def test_the_full_grid_is_the_product_of_the_grammar():
    """The size of the search is the input every correction needs, so it is fixed."""
    grid = generate_factor_grid()
    assert len(grid) == len(TRANSFORMS) * len(WINDOWS) * len(SCALINGS) * len(HOLDS) == 840
    assert len({spec.name for spec in grid}) == len(grid), "names must be unique"


def test_subsampling_is_reproducible_from_the_seed():
    """An unreproducible search cannot be corrected for: its size is unknown."""
    first = generate_factor_grid(50, seed=3)
    second = generate_factor_grid(50, seed=3)
    different = generate_factor_grid(50, seed=4)

    assert [s.name for s in first] == [s.name for s in second]
    assert [s.name for s in first] != [s.name for s in different]
    assert len(set(first)) == 50, "sampling is without replacement"


def test_asking_for_more_factors_than_exist_returns_the_whole_grid():
    assert len(generate_factor_grid(5_000)) == 840


# --------------------------------------------------------------------------- #
# Look-ahead, which is the only bug that matters here
# --------------------------------------------------------------------------- #
def test_the_signal_at_time_t_does_not_depend_on_the_return_at_time_t():
    """The one property that makes any of this meaningful.

    A factor that peeks produces a beautiful backtest and no information. The
    test perturbs a single return and requires every earlier signal to be
    unchanged -- and the signal at that same index too, since positions are
    applied to the *next* return.
    """
    returns = pure_noise(n=800)
    spec = FactorSpec("momentum", 21, "zscore", 1)
    baseline = compute_signal(returns, spec)

    poked = returns.copy()
    poked[500] += 5.0  # a violent, unmissable change
    after = compute_signal(poked, spec)

    unchanged = np.isclose(baseline[:501], after[:501], equal_nan=True)
    assert unchanged.all(), "a future return leaked into a past signal"
    assert not np.isclose(baseline[502], after[502], equal_nan=True), (
        "the signal must respond to the change eventually, or it uses no data"
    )


@pytest.mark.parametrize("scaling", SCALINGS)
def test_no_scaling_leaks_the_future(scaling: str):
    """`zscore` and `rank` are the dangerous ones: the obvious implementation
    of both uses full-sample statistics, which puts the future into every
    observation. Expanding windows are the fix, and this is what checks it."""
    returns = pure_noise(n=700)
    spec = FactorSpec("momentum", 21, scaling, 1)

    prefix = compute_signal(returns[:400], spec)
    full = compute_signal(returns, spec)

    assert np.allclose(prefix, full[:400], equal_nan=True), (
        f"{scaling} scaling changes past values when future data is appended"
    )


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_every_transform_in_the_grammar_produces_a_usable_signal(transform: str):
    """A transform that silently returns all-NaN would be a factor that never
    trades, quietly shrinking the search below its stated size."""
    returns = pure_noise(n=900)
    signal = compute_signal(returns, FactorSpec(transform, 21, "zscore", 1))

    finite = np.isfinite(signal)
    assert finite.sum() > 500, f"{transform} produced almost nothing"
    assert np.std(signal[finite]) > 0, f"{transform} is constant, so it cannot be a signal"


# --------------------------------------------------------------------------- #
# No false positives
# --------------------------------------------------------------------------- #
def test_nothing_survives_on_pure_noise(noise_report):
    """The headline result. There is nothing there, and the lab says so."""
    assert noise_report.survivors == []
    assert not noise_report.any_survivor
    assert noise_report.reality_check_p > 0.10
    assert noise_report.spa_p > 0.10
    assert "no factor survives" in noise_report.summary()


def test_the_winner_on_noise_still_looks_significant_uncorrected(noise_report):
    """This is the illusion being demonstrated, so it has to actually appear.

    If the best of two hundred worthless factors did not clear a naive p < 0.05,
    the module would be arguing against a problem it had failed to reproduce.
    """
    assert noise_report.best.t_statistic > 1.96
    assert noise_report.best.naively_significant
    assert noise_report.best.naive_p_value < 0.05


def test_the_expected_false_positive_count_is_reported(noise_report):
    assert noise_report.expected_false_positives == pytest.approx(0.05 * 200)
    assert noise_report.n_naively_significant >= 1


# --------------------------------------------------------------------------- #
# ... but the lab is not a constant
# --------------------------------------------------------------------------- #
def test_a_planted_signal_is_found(signal_report):
    """The matching half of the noise test.

    Without this, every no-false-positive test above would be satisfied by a
    function that returns "nothing survives" unconditionally.
    """
    assert signal_report.any_survivor
    assert signal_report.reality_check_p < 0.05
    assert signal_report.spa_p < 0.05


def test_the_exact_planted_rule_is_recovered_as_the_best_factor(signal_report):
    """Not merely 'something significant' -- the actual generating rule.

    The data was built from the sign of trailing 21-day momentum, and
    `momentum_21d_sign_h1` is that rule expressed in the grammar.
    """
    assert signal_report.best.spec.name == "momentum_21d_sign_h1"
    assert signal_report.best.t_statistic > 8.0


def test_the_survivors_are_variants_of_the_true_signal(signal_report):
    """Correlated variants surviving alongside the truth is correct behaviour.

    Family-wise control does not promise a unique answer; a hundred rescalings
    of a real signal are all real. What it promises is that the probability of
    ANY false rejection stays bounded, which the noise test checks.
    """
    assert "momentum_21d_sign_h1" in signal_report.survivors
    assert len(signal_report.survivors) > 1


# --------------------------------------------------------------------------- #
# Two defects found by the tests above, now pinned
# --------------------------------------------------------------------------- #
def test_factors_are_compared_at_equal_risk_not_equal_leverage():
    """The bug that made the whole comparison meaningless.

    The grammar produces P&L series spanning a ~1,900x range in standard
    deviation, because a `raw` momentum signal is a number like 0.4 and a `sign`
    signal is exactly +/-1. Without normalisation the max-of-means statistic
    ranks leverage: with a signal planted deliberately, the true factor sat 143rd
    of 200 by P&L scale and White's Reality Check returned p = 0.0975 while
    Hansen's SPA -- which studentises, and so was partly protected -- returned
    p < 0.0001. After normalisation both return p < 0.0001.

    The check here is the invariant that makes the fix safe: dividing a column by
    a positive constant cannot change that factor's own t-statistic.
    """
    returns = planted_signal()
    spec = FactorSpec("momentum", 21, "sign", 1)

    result, pnl = evaluate_factor(returns, spec)

    # The same mask the module uses. An earlier version of this test also
    # dropped genuine zero-P&L observations -- days when the momentum sign was
    # exactly zero -- and then compared the result against a t-statistic
    # computed over a different sample, disagreeing in the fourth decimal for
    # reasons that had nothing to do with scaling.
    usable = np.isfinite(compute_signal(returns, spec)) & np.isfinite(returns)
    active = pnl[usable]
    scaled = active / np.std(active)

    original_t = np.mean(active) / (np.std(active, ddof=1) / np.sqrt(active.size))
    scaled_t = np.mean(scaled) / (np.std(scaled, ddof=1) / np.sqrt(scaled.size))

    assert original_t == pytest.approx(scaled_t, rel=1e-9), "scaling changed the t-statistic"
    assert result.t_statistic == pytest.approx(float(original_t), rel=1e-9)


def test_the_deflated_sharpe_is_reported_both_ways_because_they_disagree(signal_report):
    """A p-value of 0.90 on a t = 11.4 signal is a broken assumption, not a verdict.

    The deflated Sharpe deflates against the expected maximum of n_trials draws
    *under the null of no skill*. Handing it the observed spread of trial Sharpes
    treats that spread as pure noise -- so when the search really does contain
    something, the skill inflates the benchmark the winner must clear. It is
    self-defeating in exactly the case where there is something to find, which is
    why both figures are printed.
    """
    assert signal_report.deflated_sharpe_p > 0.5, "the search-derived variance is contaminated"
    assert signal_report.deflated_sharpe_p_null_variance < 0.01, "the null-variance form works"
    assert "inflates the very benchmark" in signal_report.summary()


def test_on_noise_the_two_deflated_sharpes_agree(noise_report):
    """The guard for the test above: the disagreement is caused by skill.

    With nothing planted there is no skill to contaminate the spread, so the two
    figures should not tell opposite stories, and the explanatory note should not
    appear.
    """
    assert noise_report.deflated_sharpe_p > 0.05
    assert noise_report.deflated_sharpe_p_null_variance > 0.05
    assert "inflates the very benchmark" not in noise_report.summary()


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_a_series_too_short_to_support_a_conclusion_is_refused():
    """Returning a confident verdict from 100 observations would be the failure."""
    with pytest.raises(ValueError, match="cannot support a conclusion"):
        run_factor_lab(pure_noise(n=100), n_factors=20)


def test_the_report_states_the_uncorrected_result_alongside_the_corrected_one(noise_report):
    """The gap between the two is the finding; hiding either one loses it."""
    text = noise_report.summary()
    assert "without correction" in text
    assert f"{noise_report.best.sharpe:.2f}" in text
    assert "Reality Check" in text and "SPA" in text


def test_running_the_lab_twice_with_the_same_seed_gives_the_same_answer():
    """A search that cannot be reproduced cannot be corrected for."""
    returns = pure_noise(n=800)
    first = run_factor_lab(returns, n_factors=40, seed=11, n_bootstrap=200)
    second = run_factor_lab(returns, n_factors=40, seed=11, n_bootstrap=200)

    assert first.best.spec.name == second.best.spec.name
    assert first.reality_check_p == pytest.approx(second.reality_check_p)
    assert first.survivors == second.survivors


# --------------------------------------------------------------------------- #
# Gaps found by mutation testing
# --------------------------------------------------------------------------- #
def test_the_skew_transform_measures_skew_and_carries_its_sign():
    """Mutation testing changed the exponent from 3 to 6 and nothing failed.

    Skew is the transform whose *sign* is the information -- a right-skewed
    window and a left-skewed one must not produce the same signal -- so an
    exponent that quietly becomes even would destroy exactly the content the
    factor exists to capture while still producing plausible-looking numbers.
    """
    n = 400
    right = np.full(n, -0.001)
    right[::40] = 0.05  # rare large gains, occasional
    left = -right

    spec = FactorSpec("skew", 63, "raw", 1)
    right_signal = compute_signal(right, spec)
    left_signal = compute_signal(left, spec)

    tail = ~np.isnan(right_signal)
    assert np.nanmean(right_signal[tail]) > 0, "right-skewed data must score positive"
    assert np.nanmean(left_signal[tail]) < 0, "and its mirror must score negative"
    # An even exponent would make the two identical.
    assert np.nanmean(right_signal[tail]) != pytest.approx(np.nanmean(left_signal[tail]))


def test_the_kurtosis_transform_responds_to_tails_and_not_to_spread():
    """The fourth moment is standardised, so scaling the series must not move it.

    Changing the exponent from 4 to 8 survived mutation testing, which means
    nothing was checking that this transform measures tail *shape* rather than
    magnitude.
    """
    rng = np.random.default_rng(3)
    calm = rng.standard_normal(600) * 0.01
    wild = calm.copy()
    wild[::50] *= 12.0  # same body, much heavier tails

    spec = FactorSpec("kurtosis", 126, "raw", 1)
    calm_signal = np.nanmean(compute_signal(calm, spec)[200:])
    wild_signal = np.nanmean(compute_signal(wild, spec)[200:])
    scaled_signal = np.nanmean(compute_signal(calm * 10.0, spec)[200:])

    # The transform is negated, so fatter tails score LOWER.
    assert wild_signal < calm_signal, "heavier tails must change the signal"
    assert scaled_signal == pytest.approx(calm_signal, rel=1e-6), (
        "a standardised moment must be invariant to a change of units"
    )


def test_a_factor_that_never_moves_scores_zero_rather_than_dividing_by_zero():
    """A constant signal has no standard deviation and no t-statistic.

    Returning a t of inf, or raising, would either poison the maximum across the
    search or abort the whole run because one of 840 factors was degenerate.
    """
    flat = np.zeros(800)
    result, pnl = evaluate_factor(flat, FactorSpec("momentum", 21, "sign", 1))

    assert result.t_statistic == 0.0
    assert result.sharpe == 0.0
    assert result.naive_p_value == 1.0
    assert not result.naively_significant
    assert np.all(pnl == 0.0)


def test_the_minimum_length_guard_is_checked_at_its_stated_boundary():
    """299 observations is refused and 300 is not, as documented."""
    with pytest.raises(ValueError, match="at least 300"):
        run_factor_lab(pure_noise(n=299), n_factors=10, n_bootstrap=50)

    report = run_factor_lab(pure_noise(n=300), n_factors=10, seed=0, n_bootstrap=50)
    assert report.n_factors == 10


def test_the_warm_up_period_is_excluded_from_the_tested_sample():
    """Rows where every factor is still NaN carry no information.

    Leaving them in would dilute every mean identically -- which looks harmless
    -- while giving the block bootstrap rows of structural zeros to resample,
    understating the variance of the maximum and making every correction too
    permissive.
    """
    returns = pure_noise(n=1000)
    report = run_factor_lab(returns, n_factors=30, seed=5, n_bootstrap=100)

    longest = max(spec.window + spec.hold for spec in generate_factor_grid(30, seed=5))
    assert report.n_observations == returns.size - longest - 1
    assert report.n_observations < returns.size


def test_a_bigger_search_is_reported_as_a_bigger_search():
    """The size of the search is the input every correction depends on."""
    returns = pure_noise(n=900)
    small = run_factor_lab(returns, n_factors=20, seed=2, n_bootstrap=100)
    large = run_factor_lab(returns, n_factors=120, seed=2, n_bootstrap=100)

    assert small.n_factors == 20
    assert large.n_factors == 120
    assert large.expected_false_positives > small.expected_false_positives
    assert "120" in large.summary()
