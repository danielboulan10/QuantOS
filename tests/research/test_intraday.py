"""Validation for intraday volatility estimation.

Simulated data is used throughout, because it is the only way to know the true
integrated variance and so the only way to tell a working estimator from a
plausible one. The tests inject a known volatility, a known noise level and known
jumps, and require each estimator to recover what it claims to measure -- and,
just as importantly, to *fail* to recover what it claims to be robust to.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.research.intraday import (
    MU_1,
    MU_43,
    bipower_variation,
    epps_curve,
    intraday_seasonality,
    jump_test,
    noise_variance,
    optimal_sampling_interval,
    realized_variance,
    signature_plot,
    two_scale_realized_variance,
    volatility_report,
)

MINUTES = 390  # one US equity session
SECONDS = 23_400


def efficient_path(n: int, sigma: float, *, seed: int, start: float = 100.0) -> np.ndarray:
    """A driftless GBM session with annualised volatility ``sigma``."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma / np.sqrt(252 * n), n)
    # Rescale so the path realises exactly sigma, removing sampling noise from
    # the comparison: the estimator is under test, not the random draw.
    steps *= (sigma / np.sqrt(252)) / np.sqrt(np.sum(steps**2))
    return start * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))


def add_noise(prices: np.ndarray, noise_sd: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return prices * np.exp(rng.normal(0.0, noise_sd, prices.size))


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
def test_bipower_constants_match_their_definitions():
    """Both constants are moments of |Z|; check against direct integration."""
    z = np.random.default_rng(11).standard_normal(2_000_000)
    assert pytest.approx(float(np.mean(np.abs(z))), abs=2e-3) == MU_1
    assert pytest.approx(float(np.mean(np.abs(z) ** (4 / 3))), abs=2e-3) == MU_43


# --------------------------------------------------------------------------- #
# Recovery in the absence of noise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.45])
def test_realized_variance_recovers_volatility_without_noise(sigma):
    prices = efficient_path(MINUTES, sigma, seed=1)
    assert np.sqrt(realized_variance(prices)) == pytest.approx(sigma, rel=1e-9)


def test_bipower_matches_realized_variance_on_a_continuous_path():
    """With no jumps the two estimate the same thing, so they must agree."""
    prices = efficient_path(MINUTES, 0.25, seed=2)
    rv = realized_variance(prices)
    bv = bipower_variation(prices)
    assert np.sqrt(bv) == pytest.approx(np.sqrt(rv), rel=0.10)


# --------------------------------------------------------------------------- #
# Noise: the central problem
# --------------------------------------------------------------------------- #
def test_noise_inflates_the_naive_estimator_by_the_predicted_amount():
    """Validates the module's central formula, not just its direction.

    The docstring claims :math:`E[RV_n] = IV + 2n E[u^2]`. That is testable: the
    inflated estimate must match the prediction, not merely be larger. Asserting
    only "bigger than sigma" would pass for a noise model that was wrong by an
    order of magnitude.
    """
    sigma, noise_sd = 0.30, 1e-4
    clean = efficient_path(SECONDS, sigma, seed=3)
    noisy = add_noise(clean, noise_sd, seed=4)

    assert np.sqrt(realized_variance(clean)) == pytest.approx(sigma, rel=1e-9)

    integrated = sigma**2 / 252  # per-session variance
    predicted = np.sqrt((integrated + 2 * SECONDS * noise_sd**2) * 252)
    assert np.sqrt(realized_variance(noisy)) == pytest.approx(predicted, rel=0.02)
    # At this noise level that is a ~50% overstatement of volatility.
    assert np.sqrt(realized_variance(noisy)) > 1.4 * sigma


def test_two_scale_estimator_recovers_volatility_through_noise():
    """The estimator that is supposed to survive noise must actually survive it."""
    sigma, noise_sd = 0.30, 1e-4
    noisy = add_noise(efficient_path(SECONDS, sigma, seed=5), noise_sd, seed=6)

    corrected = np.sqrt(two_scale_realized_variance(noisy))
    naive = np.sqrt(realized_variance(noisy))

    assert corrected == pytest.approx(sigma, abs=0.05)
    assert abs(corrected - sigma) < 0.2 * abs(naive - sigma)


def test_noise_variance_is_recovered():
    noise_sd = 2e-4
    noisy = add_noise(efficient_path(SECONDS, 0.20, seed=7), noise_sd, seed=8)
    estimated = np.sqrt(noise_variance(noisy))
    assert estimated == pytest.approx(noise_sd, rel=0.15)


def test_signature_plot_slopes_up_under_noise_and_is_flat_without():
    """The diagnostic must distinguish the two cases it exists to distinguish."""
    clean = efficient_path(SECONDS, 0.25, seed=9)
    noisy = add_noise(clean, 1.5e-4, seed=10)

    flat = signature_plot(clean)
    sloped = signature_plot(noisy)

    assert flat.noise_is_material is False
    assert sloped.noise_is_material is True
    # The noisy curve must decline as sampling coarsens; the clean one must not.
    assert sloped.volatilities[0] > 2 * sloped.volatilities[-1]
    assert flat.volatilities[0] == pytest.approx(flat.volatilities[-1], rel=0.25)
    assert sloped.suggested_step > flat.suggested_step


def test_optimal_sampling_coarsens_as_noise_grows():
    """More noise must push the recommended interval out, not in."""
    clean = efficient_path(SECONDS, 0.25, seed=12)
    steps = [optimal_sampling_interval(add_noise(clean, sd, seed=13)) for sd in (1e-5, 5e-5, 2e-4)]
    assert steps == sorted(steps), f"expected monotonically coarser sampling, got {steps}"
    assert steps[-1] > steps[0]


# --------------------------------------------------------------------------- #
# Jumps
# --------------------------------------------------------------------------- #
def test_bipower_ignores_a_jump_that_realized_variance_counts():
    """The defining property of bipower variation."""
    sigma = 0.20
    prices = efficient_path(MINUTES, sigma, seed=14)
    jumped = prices.copy()
    jumped[MINUTES // 2 :] *= 1.05  # a clean 5% gap

    rv_error = abs(np.sqrt(realized_variance(jumped)) - sigma)
    bv_error = abs(np.sqrt(bipower_variation(jumped)) - sigma)

    assert np.sqrt(realized_variance(jumped)) > 1.8 * sigma
    # Bipower is contaminated too -- the jump enters through its two adjacent
    # products -- but by far less. Claiming it is *unaffected* would overstate
    # the estimator: in finite samples it retains a modest upward bias.
    assert bv_error < 0.3 * rv_error


def test_jump_test_has_power_against_a_real_jump():
    detected = 0
    for seed in range(40):
        prices = efficient_path(MINUTES, 0.20, seed=seed)
        prices[MINUTES // 2 :] *= 1.04
        if jump_test(prices).has_jump:
            detected += 1
    assert detected >= 32, f"detected only {detected}/40 injected jumps"


def test_jump_test_size_is_close_to_nominal():
    """A test that fires on continuous paths is not a jump test.

    Checking size is the step most often skipped, and it is the one that decides
    whether a rejection means anything. At a nominal 5% the empirical rate should
    sit near 5%; the tolerance below is wide because 300 replications carry a
    standard error of about 1.3 percentage points.
    """
    rejections = 0
    trials = 300
    for seed in range(trials):
        prices = efficient_path(MINUTES, 0.20, seed=1000 + seed)
        if jump_test(prices).has_jump:
            rejections += 1
    rate = rejections / trials
    assert rate < 0.12, f"empirical size {rate:.1%} at a nominal 5%: the test over-rejects"


def test_jump_share_is_reported_and_bounded():
    prices = efficient_path(MINUTES, 0.20, seed=21)
    prices[200:] *= 1.06
    result = jump_test(prices)
    assert 0.0 <= result.jump_share <= 1.0
    assert result.jump_share > 0.2
    assert "jump detected" in result.verdict


# --------------------------------------------------------------------------- #
# Cross-asset and seasonality
# --------------------------------------------------------------------------- #
def test_epps_effect_appears_under_non_synchronous_observation():
    """Correlation must attenuate at fine sampling, and recover at coarse.

    Non-synchronicity is simulated by holding each series stale for a few ticks
    at a time -- which is what a low trade rate does to an observed price.
    """
    rng = np.random.default_rng(31)
    n = 20_000
    common = rng.normal(0, 1e-4, n)
    a_steps = 0.8 * common + 0.6 * rng.normal(0, 1e-4, n)
    b_steps = 0.8 * common + 0.6 * rng.normal(0, 1e-4, n)
    a = 100 * np.exp(np.cumsum(a_steps))
    b = 100 * np.exp(np.cumsum(b_steps))

    # Stale-price observation: each series only updates every few ticks.
    def stale(series: np.ndarray, hold: int) -> np.ndarray:
        held = series.copy()
        for offset in range(0, series.size, hold):
            held[offset : offset + hold] = series[offset]
        return held

    _, correlations = epps_curve(stale(a, 7), stale(b, 5))
    fine = correlations[0]
    coarse = float(np.nanmedian(correlations[-4:]))

    assert fine < coarse - 0.10, (
        f"expected attenuation at fine sampling: fine={fine:.3f} coarse={coarse:.3f}"
    )
    assert coarse > 0.4


def test_intraday_seasonality_finds_an_injected_u_shape():
    rng = np.random.default_rng(41)
    sessions = []
    for _ in range(30):
        position = np.linspace(0.0, 1.0, MINUTES)
        # U-shaped volatility: busy at the open and close, quiet at midday.
        shape = 1.0 + 1.5 * (2 * position - 1) ** 2
        steps = rng.normal(0.0, 1e-4, MINUTES) * shape
        sessions.append(100 * np.exp(np.cumsum(steps)))

    seasonality = intraday_seasonality(sessions)
    assert seasonality.n_sessions == 30
    assert seasonality.is_u_shaped is True
    assert seasonality.relative_volatility[0] > seasonality.relative_volatility[6]


def test_flat_sessions_are_not_reported_as_u_shaped():
    """Guards the test above."""
    rng = np.random.default_rng(42)
    sessions = [100 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, MINUTES))) for _ in range(30)]
    assert intraday_seasonality(sessions).is_u_shaped is False


# --------------------------------------------------------------------------- #
# The combined report
# --------------------------------------------------------------------------- #
def test_report_flags_noise_and_prefers_the_corrected_estimate():
    noisy = add_noise(efficient_path(SECONDS, 0.30, seed=51), 1e-4, seed=52)
    report = volatility_report(noisy)

    assert report.n_observations == SECONDS + 1
    assert report.noise_detected is True
    assert any("microstructure noise dominates" in note for note in report.notes)
    assert report.best_estimate == pytest.approx(0.30, abs=0.06)
    assert report.optimal_step > 1
    assert "noise-contaminated" in report.summary()


def test_report_is_quiet_on_clean_data():
    """On clean data the report must not invent noise, and must not "correct" for it."""
    report = volatility_report(efficient_path(MINUTES, 0.22, seed=53))
    assert report.noise_detected is False
    assert not any("noise dominates" in note for note in report.notes)
    assert report.optimal_step == 1
    # Realised variance is exact here, so the reported estimate should be too.
    assert report.best_estimate == pytest.approx(0.22, abs=0.005)
