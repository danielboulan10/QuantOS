"""Validation for the forward path simulators.

Each test injects a known property and requires the simulation to reproduce it:
the right terminal variance, the right tail weight, the right first-passage
ordering. A simulator that merely runs would pass none of these.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantos.forecast.paths import (
    PathEnsemble,
    compare_engines,
    simulate_bootstrap_paths,
    simulate_garch_paths,
)

TRADING_DAYS = 252


def gaussian_returns(n: int = 1500, sigma: float = 0.02, *, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, n)


def clustered_returns(
    n: int = 2000, *, seed: int = 0, innovation_df: float | None = None
) -> np.ndarray:
    """A series with genuine GARCH dynamics, so a fit has something to find.

    ``innovation_df`` makes the innovations Student-t, which is what lets a test
    distinguish the ``t`` and ``normal`` simulation paths -- on Gaussian
    innovations maximum likelihood correctly estimates a huge df and the two
    become the same thing.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.10, 0.85
    variance = omega / (1 - alpha - beta)
    out = np.empty(n)
    for i in range(n):
        variance = omega + alpha * (out[i - 1] ** 2 if i else 0.0) + beta * variance
        if innovation_df is None:
            shock = rng.standard_normal()
        else:
            shock = rng.standard_t(innovation_df) / np.sqrt(innovation_df / (innovation_df - 2.0))
        out[i] = np.sqrt(variance) * shock
    return out


# --------------------------------------------------------------------------- #
# Shape and invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon", [1, 21, 160])
def test_every_path_starts_at_spot(horizon):
    ensemble = simulate_garch_paths(gaussian_returns(), 100.0, horizon, n_paths=500)
    assert ensemble.paths.shape == (500, horizon + 1)
    assert np.all(ensemble.paths[:, 0] == 100.0)


def test_prices_stay_positive():
    """Log-space simulation must make a negative price impossible."""
    ensemble = simulate_garch_paths(gaussian_returns(sigma=0.06), 10.0, 160, n_paths=2000)
    assert np.all(ensemble.paths > 0)


def test_simulation_is_reproducible_from_its_seed():
    a = simulate_garch_paths(gaussian_returns(), 100.0, 21, n_paths=300, seed=7)
    b = simulate_garch_paths(gaussian_returns(), 100.0, 21, n_paths=300, seed=7)
    np.testing.assert_array_equal(a.paths, b.paths)

    c = simulate_garch_paths(gaussian_returns(), 100.0, 21, n_paths=300, seed=8)
    assert not np.array_equal(a.paths, c.paths)


def test_too_little_history_is_refused():
    with pytest.raises(ValueError, match="at least 100 returns"):
        simulate_garch_paths(gaussian_returns(n=50), 100.0, 21)


def test_a_nonpositive_horizon_is_refused():
    with pytest.raises(ValueError, match="horizon must be positive"):
        simulate_garch_paths(gaussian_returns(), 100.0, 0)


# --------------------------------------------------------------------------- #
# Does the simulation reproduce the volatility it was given?
# --------------------------------------------------------------------------- #
def test_terminal_variance_scales_with_the_horizon():
    r"""Variance of the terminal log return must grow about linearly in time.

    For a driftless walk :math:`\operatorname{Var}(\log S_T/S_0) = T\sigma^2`, so
    the standard deviation ratio between a 4x horizon should be about 2. Getting
    this wrong -- by, say, forgetting to standardise the Student-t -- would break
    every probability downstream while still producing plausible-looking paths.
    """
    returns = gaussian_returns(n=3000, sigma=0.015, seed=3)
    short = simulate_garch_paths(returns, 100.0, 21, n_paths=40_000, seed=1)
    long = simulate_garch_paths(returns, 100.0, 84, n_paths=40_000, seed=1)

    ratio = np.std(long.terminal_returns) / np.std(short.terminal_returns)
    assert ratio == pytest.approx(2.0, rel=0.12)


def test_constant_vol_fallback_recovers_the_input_volatility():
    """With no ARCH effects the engine falls back, and must still be unbiased."""
    sigma = 0.018
    returns = gaussian_returns(n=3000, sigma=sigma, seed=5)
    ensemble = simulate_garch_paths(returns, 100.0, 63, n_paths=40_000, seed=2)

    realised = float(np.std(ensemble.terminal_returns) / np.sqrt(63))
    assert realised == pytest.approx(sigma, rel=0.08)


def test_no_arch_effects_triggers_the_fallback_and_says_so():
    """Fitting GARCH to a series without clustering is the failure being avoided."""
    ensemble = simulate_garch_paths(gaussian_returns(n=2000, seed=11), 100.0, 21, n_paths=500)
    assert ensemble.engine == "constant-vol"
    assert any("no ARCH effects" in note for note in ensemble.notes)


def test_clustered_series_actually_fits_garch():
    """Guards the test above: real clustering must not be discarded."""
    ensemble = simulate_garch_paths(clustered_returns(seed=4), 100.0, 21, n_paths=500)
    assert ensemble.engine == "garch"
    assert "persistence" in ensemble.assumptions


def test_standardised_t_keeps_the_variance_it_was_given():
    r"""A raw t has variance :math:`\nu/(\nu-2)`; the standardised one has 1."""
    from quantos.forecast.paths import _standardised_t

    rng = np.random.default_rng(0)
    for df in (3.0, 4.0, 8.0):
        draws = _standardised_t(rng, df, (400_000,))
        assert float(np.var(draws)) == pytest.approx(1.0, rel=0.05)


def test_infinite_variance_degrees_of_freedom_are_refused():
    from quantos.forecast.paths import _standardised_t

    with pytest.raises(ValueError, match="degrees of freedom must exceed 2"):
        _standardised_t(np.random.default_rng(0), 2.0, (10,))


# --------------------------------------------------------------------------- #
# Fat tails
# --------------------------------------------------------------------------- #
def _kurtosis(values: np.ndarray) -> float:
    centred = values - np.mean(values)
    return float(np.mean(centred**4) / np.var(values) ** 2)


def test_student_t_innovations_produce_fatter_tails_than_a_normal():
    """The reason the t is used at all: a normal cannot make the tails.

    The input series is built with genuinely fat-tailed (t) innovations. On a
    series with *Gaussian* innovations this test would correctly find no
    difference, because maximum likelihood estimates a very large df and the two
    simulations converge -- which is the model being right, not the test failing.
    """
    returns = clustered_returns(n=3000, seed=6, innovation_df=4.0)
    fat = simulate_garch_paths(returns, 100.0, 21, n_paths=40_000, distribution="t", seed=3)
    thin = simulate_garch_paths(returns, 100.0, 21, n_paths=40_000, distribution="normal", seed=3)

    assert fat.assumptions["innovation_df"] < 12.0, "ML should detect the fat tails"
    assert thin.assumptions["innovation_df"] == float("inf")
    assert _kurtosis(fat.terminal_returns) > _kurtosis(thin.terminal_returns)


def test_asking_for_normal_does_not_secretly_give_fat_tails():
    """The inverted-default bug, pinned.

    ``distribution="normal"`` once fell through to ``t(5)`` because a normal fit
    carries no ``df``, so it produced *higher* kurtosis than ``distribution="t"``.
    """
    returns = clustered_returns(n=2500, seed=6, innovation_df=4.0)
    thin = simulate_garch_paths(returns, 100.0, 21, n_paths=30_000, distribution="normal", seed=3)
    fat = simulate_garch_paths(returns, 100.0, 21, n_paths=30_000, distribution="t", seed=3)
    assert _kurtosis(thin.terminal_returns) < _kurtosis(fat.terminal_returns)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def test_paths_are_driftless_by_default():
    """The default must not smuggle in a return forecast."""
    ensemble = simulate_garch_paths(
        gaussian_returns(n=2500, seed=8) + 0.002,  # a strongly drifting history
        100.0,
        63,
        n_paths=40_000,
        seed=4,
    )
    # The historical mean is removed, so the median outcome sits at the spot.
    assert float(np.median(ensemble.terminal_returns)) == pytest.approx(0.0, abs=0.01)


def test_an_explicit_drift_is_honoured():
    ensemble = simulate_garch_paths(
        gaussian_returns(n=2500, seed=8), 100.0, 63, n_paths=40_000, drift=0.001, seed=4
    )
    assert float(np.mean(ensemble.terminal_returns)) == pytest.approx(0.063, abs=0.01)


def test_bootstrap_removes_the_historical_mean_by_default():
    drifting = gaussian_returns(n=2500, seed=9) + 0.003
    ensemble = simulate_bootstrap_paths(drifting, 100.0, 42, n_paths=20_000, seed=5)
    assert float(np.median(ensemble.terminal_returns)) == pytest.approx(0.0, abs=0.015)
    assert any("driftless" in note for note in ensemble.notes)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_reproduces_the_per_step_volatility_exactly():
    """Each simulated step is drawn from the sample, so step variance must match."""
    returns = gaussian_returns(n=3000, sigma=0.02, seed=12)
    centred = returns - np.mean(returns)
    ensemble = simulate_bootstrap_paths(centred, 100.0, 63, n_paths=20_000, block=21, seed=6)
    steps = np.diff(np.log(ensemble.paths), axis=1)
    assert float(np.std(steps)) == pytest.approx(float(np.std(centred, ddof=1)), rel=0.01)


def test_iid_bootstrap_recovers_the_horizon_variance():
    """With block=1 the draws are independent, so i.i.d. scaling must hold."""
    returns = gaussian_returns(n=3000, sigma=0.02, seed=12)
    centred = returns - np.mean(returns)
    ensemble = simulate_bootstrap_paths(centred, 100.0, 63, n_paths=40_000, block=1, seed=6)
    realised = float(np.std(ensemble.terminal_returns) / np.sqrt(63))
    assert realised == pytest.approx(float(np.std(centred, ddof=1)), rel=0.03)


def test_block_horizon_variance_follows_the_samples_own_autocorrelation():
    r"""Documents why a long block does *not* give i.i.d. horizon variance.

    Terminal variance carries the factor :math:`1 + 2\sum_k (1 - k/b)\rho_k`,
    where the autocorrelations are the *sample's*, including finite-sample noise.
    On this fixture each :math:`|\rho_k| < 0.03`, yet they accumulate to a 16%
    variance shortfall at ``block=21`` -- which the simulation reproduces, because
    reproducing the sample's dependence is exactly what a block bootstrap is for.

    An earlier version of this test asserted i.i.d. scaling and failed. The
    simulator was right; the expectation was wrong.
    """
    block = 21
    returns = gaussian_returns(n=3000, sigma=0.02, seed=12)
    centred = returns - np.mean(returns)
    variance = float(np.var(centred, ddof=1))

    rho = [float(np.corrcoef(centred[:-k], centred[k:])[0, 1]) for k in range(1, block)]
    predicted_factor = 1.0 + 2.0 * sum((1 - k / block) * rho[k - 1] for k in range(1, block))

    ensemble = simulate_bootstrap_paths(centred, 100.0, 63, n_paths=40_000, block=block, seed=6)
    observed_factor = float(np.var(ensemble.terminal_returns) / (63 * variance))

    assert observed_factor == pytest.approx(predicted_factor, rel=0.05)
    assert predicted_factor < 0.95, "this fixture is chosen to show the effect"


def test_block_bootstrap_preserves_volatility_clustering():
    """Blocks exist to keep clustering, which i.i.d. resampling destroys.

    The property is autocorrelation in *squared* returns -- the signature of
    clustering. An earlier version of this test instead asserted that blocking
    raises the drawdown probability, which is not a theorem: blocking makes the
    drawdown distribution more dispersed, and whether a given threshold's
    probability rises or falls depends on where the threshold sits.
    """
    returns = clustered_returns(n=3000, seed=13)
    iid = simulate_bootstrap_paths(returns, 100.0, 252, n_paths=4000, block=1, seed=7)
    blocked = simulate_bootstrap_paths(returns, 100.0, 252, n_paths=4000, block=42, seed=7)

    def squared_autocorrelation(ensemble) -> float:
        steps = np.diff(np.log(ensemble.paths), axis=1)
        squared = steps**2
        centred = squared - squared.mean(axis=1, keepdims=True)
        numerator = np.mean(np.sum(centred[:, 1:] * centred[:, :-1], axis=1))
        denominator = np.mean(np.sum(centred**2, axis=1))
        return float(numerator / denominator)

    assert squared_autocorrelation(blocked) > squared_autocorrelation(iid) + 0.02
    assert abs(squared_autocorrelation(iid)) < 0.05, "i.i.d. draws should show none"


def test_bootstrap_cannot_exceed_its_sample_and_says_so():
    returns = gaussian_returns(n=2000, sigma=0.01, seed=14)
    ensemble = simulate_bootstrap_paths(returns, 100.0, 5, n_paths=5000, seed=8)
    worst_step = float(np.min(np.diff(np.log(ensemble.paths), axis=1)))
    assert worst_step >= float(np.min(returns - np.mean(returns))) - 1e-12
    assert any("sample extremes" in note for note in ensemble.notes)


# --------------------------------------------------------------------------- #
# Engine comparison
# --------------------------------------------------------------------------- #
def test_the_two_engines_broadly_agree_on_a_well_behaved_series():
    returns = gaussian_returns(n=3000, sigma=0.015, seed=15)
    comparison = compare_engines(returns, 100.0, 21, n_paths=20_000)
    assert comparison["max_relative_gap"] < 0.05
    assert "agree" in str(comparison["verdict"])


# --------------------------------------------------------------------------- #
# Fan chart / quantiles
# --------------------------------------------------------------------------- #
def test_quantile_bands_are_ordered_and_widen_with_time():
    ensemble = simulate_garch_paths(gaussian_returns(n=2000), 100.0, 63, n_paths=10_000)
    bands = ensemble.quantile_bands()
    levels = sorted(bands)
    for lower, upper in itertools.pairwise(levels):
        assert np.all(bands[lower] <= bands[upper] + 1e-9)
    width_early = bands[0.95][5] - bands[0.05][5]
    width_late = bands[0.95][-1] - bands[0.05][-1]
    assert width_late > width_early


def test_running_extremes_bound_the_paths():
    ensemble = simulate_garch_paths(gaussian_returns(n=1200), 100.0, 21, n_paths=500)
    highs, lows = ensemble.running_extremes()
    assert np.all(highs >= ensemble.paths - 1e-9)
    assert np.all(lows <= ensemble.paths + 1e-9)
    assert np.all(np.diff(highs, axis=1) >= -1e-9)  # monotone


def test_summary_mentions_the_engine_and_assumptions():
    text = simulate_garch_paths(clustered_returns(seed=16), 100.0, 21, n_paths=500).summary()
    assert "paths" in text
    assert "GARCH" in text or "persistence" in text


def test_ensemble_can_be_built_directly_for_downstream_tests():
    paths = np.array([[100.0, 110.0], [100.0, 90.0]])
    ensemble = PathEnsemble(paths=paths, spot=100.0, horizon=1, engine="fixture")
    assert ensemble.n_paths == 2
    np.testing.assert_allclose(ensemble.terminal, [110.0, 90.0])
