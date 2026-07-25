"""Statistics, time series, and estimator-recovery tests.

Two kinds of assertion here:

* **Oracle agreement** where SciPy implements the same quantity.
* **Parameter recovery** where it does not: simulate from a known process and
  check the estimator returns what went in. This is the stronger test, because it
  validates the whole pipeline rather than one formula.

Statistical tests are also checked for **size** -- the false-positive rate under
the null. An estimator that finds cointegration in independent random walks 28%
of the time at a nominal 5% is worse than useless, and only a size check catches
that. It caught exactly that here; see ``docs/ddr/DDR-004``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as ss

from quantos.core.distributions import (
    Beta,
    Binomial,
    ChiSquare,
    Exponential,
    FisherF,
    Gamma,
    Laplace,
    Normal,
    Poisson,
    StudentT,
)
from quantos.core.rng import SeedBank, spawn_key
from quantos.core.stats import bootstrap as bs
from quantos.core.stats import descriptive as ds
from quantos.core.stats import hypothesis as hp
from quantos.core.stats import multipletest as mt
from quantos.core.timeseries.cointegration import engle_granger, hedge_ratio, johansen
from quantos.core.timeseries.garch import fit_garch, fit_gjr_garch
from quantos.core.timeseries.ols import ols
from quantos.core.timeseries.ou import OUParameters, expected_time_to_mean, fit_ou, simulate_ou


# --------------------------------------------------------------------------- #
# Distributions vs SciPy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ours,theirs,grid",
    [
        (Normal(0.5, 2.0), ss.norm(0.5, 2.0), np.linspace(-10, 12, 400)),
        (StudentT(5.0), ss.t(5.0), np.linspace(-8, 8, 400)),
        (StudentT(2.5, 1.0, 3.0), ss.t(2.5, 1.0, 3.0), np.linspace(-20, 22, 400)),
        (ChiSquare(7.0), ss.chi2(7.0), np.linspace(0.01, 60, 400)),
        (FisherF(4.0, 9.0), ss.f(4.0, 9.0), np.linspace(0.01, 30, 400)),
        (Exponential(1.5), ss.expon(scale=1 / 1.5), np.linspace(0.001, 12, 400)),
        (Gamma(2.5, 1.3), ss.gamma(2.5, scale=1.3), np.linspace(0.01, 40, 400)),
        (Beta(2.0, 5.0), ss.beta(2.0, 5.0), np.linspace(0.001, 0.999, 400)),
        (Laplace(1.0, 2.0), ss.laplace(1.0, 2.0), np.linspace(-15, 17, 400)),
    ],
)
def test_continuous_cdf_matches_scipy(ours, theirs, grid) -> None:
    assert np.allclose(ours.cdf(grid), theirs.cdf(grid), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize(
    "ours,theirs,grid",
    [
        (Normal(0.5, 2.0), ss.norm(0.5, 2.0), np.linspace(-10, 12, 400)),
        (StudentT(5.0), ss.t(5.0), np.linspace(-8, 8, 400)),
        (Gamma(2.5, 1.3), ss.gamma(2.5, scale=1.3), np.linspace(0.05, 40, 400)),
        (Beta(2.0, 5.0), ss.beta(2.0, 5.0), np.linspace(0.01, 0.99, 400)),
    ],
)
def test_continuous_pdf_matches_scipy(ours, theirs, grid) -> None:
    assert np.allclose(ours.pdf(grid), theirs.pdf(grid), rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize(
    "ours,theirs,support",
    [
        (Poisson(6.0), ss.poisson(6.0), np.arange(0, 40.0)),
        (Binomial(30, 0.3), ss.binom(30, 0.3), np.arange(0, 31.0)),
        (Binomial(500, 0.02), ss.binom(500, 0.02), np.arange(0, 40.0)),
    ],
)
def test_discrete_matches_scipy(ours, theirs, support) -> None:
    assert np.allclose(ours.cdf(support), theirs.cdf(support), rtol=1e-11, atol=1e-13)
    assert np.allclose(ours.pdf(support), theirs.pmf(support), rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize(
    "dist", [Normal(), StudentT(6.0), ChiSquare(4.0), Gamma(2.0), Beta(2.0, 3.0)]
)
def test_cdf_ppf_round_trip(dist) -> None:
    p = np.linspace(0.005, 0.995, 200)
    assert np.allclose(dist.cdf(dist.ppf(p)), p, atol=1e-9)


@pytest.mark.parametrize(
    "dist", [Normal(1.0, 2.0), StudentT(8.0), Exponential(2.0), Gamma(3.0, 0.5), Poisson(4.0)]
)
def test_sample_moments_match_analytic(dist) -> None:
    """A sampler inconsistent with its own CDF is the classic silent bug."""
    rng = SeedBank(root=1).child("moments").generator()
    draws = dist.sample(400_000, rng)
    assert abs(float(np.mean(draws)) - dist.mean) < 6.0 * dist.std / np.sqrt(400_000)
    assert abs(float(np.var(draws)) / dist.variance - 1.0) < 0.05


def test_upper_tail_uses_the_survival_branch() -> None:
    """``sf`` must not be computed as ``1 - cdf``: p-values live in the tail."""
    assert float(Normal().sf(9.0)) > 0.0
    assert 1.0 - float(Normal().cdf(9.0)) == 0.0
    assert float(ChiSquare(3.0).sf(200.0)) > 0.0


def test_student_t_moments_reflect_tail_conditions() -> None:
    assert np.isnan(StudentT(0.5).mean)  # undefined for df <= 1
    assert StudentT(1.5).variance == np.inf  # infinite for 1 < df <= 2
    assert StudentT(3.0).excess_kurtosis == np.inf
    assert StudentT(6.0).excess_kurtosis == pytest.approx(3.0)


def test_distribution_validation() -> None:
    with pytest.raises(ValueError, match="sigma"):
        Normal(0.0, -1.0)
    with pytest.raises(ValueError, match="df"):
        StudentT(0.0)
    with pytest.raises(ValueError, match="p must lie"):
        Binomial(10, 1.5)


# --------------------------------------------------------------------------- #
# Reproducibility contract
# --------------------------------------------------------------------------- #
def test_seed_bank_is_reproducible_and_order_independent() -> None:
    """The four guarantees in quantos.core.rng, asserted."""
    # 1. Bit-exact reproducibility.
    a = SeedBank(root=42).child("agents").child("mm").generator().standard_normal(5)
    b = SeedBank(root=42).child("agents").child("mm").generator().standard_normal(5)
    assert np.array_equal(a, b)

    # 2. Stream independence between siblings.
    c = SeedBank(root=42).child("agents").child("other").generator().standard_normal(5)
    assert not np.array_equal(a, c)

    # 3. Order independence: drawing from a sibling first changes nothing.
    bank = SeedBank(root=42).child("agents")
    _ = bank.child("other").generator().standard_normal(1000)
    d = bank.child("mm").generator().standard_normal(5)
    assert np.array_equal(a, d)

    # 4. Refactor stability: the path, not construction order, keys the stream.
    assert np.array_equal(
        a, SeedBank(root=42).child("agents").child("mm").generator().standard_normal(5)
    )


def test_spawn_key_is_stable_across_processes() -> None:
    """``hash()`` is salted per process; the key must not be."""
    assert spawn_key("sim/agents/mm_01") == spawn_key("sim/agents/mm_01")
    assert spawn_key("a") != spawn_key("b")
    assert 0 <= spawn_key("anything") < 2**128


def test_seed_bank_is_immutable() -> None:
    bank = SeedBank(root=1)
    child = bank.child("x")
    assert bank.path == ()
    assert child.path == ("x",)


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def test_welford_matches_numpy_exactly(gaussian_returns: np.ndarray) -> None:
    moments = ds.RunningMoments()
    moments.update_many(gaussian_returns)
    assert moments.mean == pytest.approx(float(np.mean(gaussian_returns)), rel=1e-14)
    assert moments.variance == pytest.approx(float(np.var(gaussian_returns, ddof=1)), rel=1e-12)
    assert moments.skewness == pytest.approx(float(ss.skew(gaussian_returns)), rel=1e-9)
    assert moments.excess_kurtosis == pytest.approx(float(ss.kurtosis(gaussian_returns)), rel=1e-8)


def test_welford_merge_is_exact(gaussian_returns: np.ndarray) -> None:
    """Parallel accumulation must be exact, not approximate."""
    whole = ds.RunningMoments()
    whole.update_many(gaussian_returns)
    left, right = ds.RunningMoments(), ds.RunningMoments()
    left.update_many(gaussian_returns[:700])
    right.update_many(gaussian_returns[700:])
    merged = left.merge(right)
    assert merged.count == whole.count
    assert merged.variance == pytest.approx(whole.variance, rel=1e-12)
    assert merged.excess_kurtosis == pytest.approx(whole.excess_kurtosis, rel=1e-9)


def test_welford_survives_a_large_offset() -> None:
    """The naive E[X^2]-E[X]^2 formula returns a negative variance here."""
    x = 1e9 + np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    moments = ds.RunningMoments()
    moments.update_many(x)
    assert moments.variance == pytest.approx(2.5, rel=1e-9)
    naive = float(np.mean(x**2) - np.mean(x) ** 2)
    assert naive <= 0.0 or abs(naive - 2.5) > 1.0


def test_autocorrelation_is_positive_semidefinite_and_correct() -> None:
    rng = SeedBank(root=3).child("acf").generator()
    n = 20_000
    ar = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        ar[t] = 0.6 * ar[t - 1] + shocks[t]
    acf = ds.autocorrelation(ar, 10)
    assert acf[0] == pytest.approx(1.0)
    # AR(1) theory: rho_k = phi^k.
    assert acf[1] == pytest.approx(0.6, abs=0.02)
    assert acf[2] == pytest.approx(0.36, abs=0.03)


def test_hill_estimator_recovers_a_known_tail_index() -> None:
    """Student-t with df=3 has tail index 3."""
    draws = np.abs(StudentT(3.0).sample(200_000, SeedBank(root=5).child("hill").generator()))
    assert 2.4 < ds.hill_estimator(draws) < 3.6


def test_ewma_matches_a_manual_recursion() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = ds.ewma(x, alpha=0.5, adjust=False)
    expected = [1.0, 1.5, 2.25, 3.125]
    assert np.allclose(out, expected)


def test_ewma_requires_exactly_one_parameter() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ds.ewma([1.0, 2.0])
    with pytest.raises(ValueError, match="exactly one"):
        ds.ewma([1.0, 2.0], halflife=2.0, alpha=0.5)


# --------------------------------------------------------------------------- #
# Hypothesis tests
# --------------------------------------------------------------------------- #
def test_jarque_bera_matches_scipy(gaussian_returns: np.ndarray) -> None:
    result = hp.jarque_bera(gaussian_returns)
    reference = ss.jarque_bera(gaussian_returns)
    assert result.statistic == pytest.approx(float(reference.statistic), rel=1e-9)
    assert result.p_value == pytest.approx(float(reference.pvalue), rel=1e-7)


def test_kolmogorov_smirnov_matches_scipy(gaussian_returns: np.ndarray) -> None:
    standardised = (gaussian_returns - gaussian_returns.mean()) / gaussian_returns.std(ddof=1)
    result = hp.ks_test(standardised, Normal())
    reference = ss.kstest(standardised, "norm")
    assert result.statistic == pytest.approx(float(reference.statistic), rel=1e-10)
    # SciPy uses the exact Kolmogorov distribution at this sample size; we use
    # the asymptotic series with Stephens' small-sample correction. They agree to
    # ~0.5%, which is a documented approximation difference, not an error.
    assert result.p_value == pytest.approx(float(reference.pvalue), rel=0.02)


def test_t_test_matches_scipy(gaussian_returns: np.ndarray) -> None:
    result = hp.t_test(gaussian_returns, mu0=0.0)
    reference = ss.ttest_1samp(gaussian_returns, 0.0)
    assert result.statistic == pytest.approx(float(reference.statistic), rel=1e-11)
    assert result.p_value == pytest.approx(float(reference.pvalue), rel=1e-9)


def test_welch_matches_scipy() -> None:
    rng = SeedBank(root=8).child("welch").generator()
    a = rng.standard_normal(300) * 1.0
    b = rng.standard_normal(200) * 3.0 + 0.5
    result = hp.welch_t_test(a, b)
    reference = ss.ttest_ind(a, b, equal_var=False)
    assert result.statistic == pytest.approx(float(reference.statistic), rel=1e-11)
    assert result.p_value == pytest.approx(float(reference.pvalue), rel=1e-9)


def test_adf_distinguishes_a_random_walk_from_white_noise() -> None:
    rng = SeedBank(root=11).child("adf").generator()
    assert not hp.augmented_dickey_fuller(np.cumsum(rng.standard_normal(800))).rejects_at(0.05)
    assert hp.augmented_dickey_fuller(rng.standard_normal(800)).rejects_at(0.05)


def test_adf_returns_dickey_fuller_critical_values_not_normal_ones() -> None:
    """The tau distribution is shifted left of the normal; using normal critical
    values here is a common error that manufactures stationarity."""
    rng = SeedBank(root=12).child("adf_crit").generator()
    result = hp.augmented_dickey_fuller(rng.standard_normal(500))
    assert set(result.critical_values) == {"1%", "5%", "10%"}
    assert result.critical_values["5%"] < -2.5  # not -1.96
    assert result.critical_values["1%"] < result.critical_values["5%"]


def test_kpss_null_is_the_reverse_of_adf() -> None:
    rng = SeedBank(root=13).child("kpss").generator()
    # Stationary: KPSS should NOT reject.
    assert not hp.kpss(rng.standard_normal(800)).rejects_at(0.05)
    # Integrated: KPSS SHOULD reject, and by a wide margin.
    integrated = hp.kpss(np.cumsum(rng.standard_normal(800)))
    assert integrated.rejects_at(0.05)
    assert integrated.statistic > integrated.critical_values["1%"]


@pytest.mark.statistical
def test_kpss_bandwidth_choice_dominates_on_power() -> None:
    """The Newey-West bandwidth beats Schwert's against a unit root.

    Schwert's 12*(n/100)^(1/4) over-estimates the long-run variance of an
    integrated series and deflates the statistic. Measured here rather than
    assumed, because this is the parameter that decides whether KPSS works.
    """
    n = 800
    newey_west = int(np.ceil(4.0 * (n / 100.0) ** 0.25))
    schwert = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    power = {}
    size = {}
    for lags in (newey_west, schwert):
        reject_walk = 0
        reject_noise = 0
        for s in range(40):
            g = SeedBank(root=7000 + s).child("kpss_bw").generator()
            reject_noise += hp.kpss(g.standard_normal(n), lags=lags).rejects_at(0.05)
            reject_walk += hp.kpss(np.cumsum(g.standard_normal(n)), lags=lags).rejects_at(0.05)
        power[lags] = reject_walk / 40
        size[lags] = reject_noise / 40
    assert power[newey_west] >= power[schwert]
    assert power[newey_west] > 0.9
    assert size[newey_west] < 0.15


def test_engle_arch_detects_clustering_and_clears_after_garch(
    garch_returns: np.ndarray,
) -> None:
    """The model-adequacy loop, end to end."""
    assert hp.engle_arch(garch_returns, 5).p_value < 1e-10
    fitted = fit_garch(garch_returns)
    residuals = fitted.standardised_residuals(garch_returns - garch_returns.mean())
    assert hp.engle_arch(residuals, 5).p_value > 0.01


def test_variance_ratio_direction_identifies_the_strategy_family() -> None:
    rng = SeedBank(root=14).child("vr").generator()
    shocks = rng.standard_normal(4_000)
    momentum = np.zeros(4_000)
    reversion = np.zeros(4_000)
    for t in range(1, 4_000):
        momentum[t] = 0.3 * momentum[t - 1] + shocks[t]
        reversion[t] = -0.3 * reversion[t - 1] + shocks[t]

    trending = hp.variance_ratio(momentum, 4)
    contrarian = hp.variance_ratio(reversion, 4)
    assert trending.detail["variance_ratio"] > 1.0
    assert trending.p_value < 1e-6
    assert contrarian.detail["variance_ratio"] < 1.0
    assert contrarian.p_value < 1e-6


@pytest.mark.statistical
def test_variance_ratio_has_approximately_correct_size() -> None:
    """False-positive rate under the random-walk null, over 200 replications."""
    rejections = sum(
        hp.variance_ratio(
            SeedBank(root=1000 + s).child("size").generator().standard_normal(1_000), 4
        ).rejects_at(0.05)
        for s in range(200)
    )
    assert rejections / 200 < 0.10  # nominal 5%, robust version is conservative


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def test_ols_recovers_known_coefficients() -> None:
    rng = SeedBank(root=20).child("ols").generator()
    design = np.column_stack([np.ones(600), rng.standard_normal(600), rng.standard_normal(600)])
    truth = np.array([1.0, 2.0, -0.5])
    y = design @ truth + rng.standard_normal(600) * 0.2
    fit = ols(y, design)
    assert np.allclose(fit.coefficients, truth, atol=0.05)
    assert 0.9 < fit.r_squared < 1.0
    assert fit.f_test()[1] < 1e-50


def test_hac_standard_errors_exceed_classical_under_autocorrelation() -> None:
    """The overlapping-returns problem, set up correctly.

    A subtlety worth stating, because getting it wrong is easy: HAC corrects the
    autocorrelation of the **score** x_t * e_t, not of the residual alone. With an
    i.i.d. regressor independent of the error, Cov(x_t e_t, x_{t-j} e_{t-j}) = 0
    for j > 0 even when e is strongly autocorrelated -- so HAC and classical
    standard errors agree, and a test built that way measures nothing. (An earlier
    version of this test did exactly that and found HAC *smaller*.)

    The real predictive-regression setup needs a **persistent regressor** as well
    as overlapping returns -- a dividend yield or valuation ratio predicting
    multi-period forward returns. Then the score is autocorrelated and the
    classical standard error is badly understated.
    """
    rng = SeedBank(root=21).child("hac").generator()
    n, horizon = 2_000, 20

    # Persistent regressor: AR(1) with phi = 0.97, like a valuation ratio.
    x = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + shocks[t]

    # Overlapping forward returns: adjacent observations share 19 of 20 terms.
    noise = rng.standard_normal(n + horizon)
    y = np.array([noise[i : i + horizon].sum() for i in range(n)])
    design = np.column_stack([np.ones(n), x])

    classical = ols(y, design, cov_type="classical")
    white = ols(y, design, cov_type="white")
    hac = ols(y, design, cov_type="hac", hac_lags=horizon)

    # White's correction handles heteroskedasticity but not serial correlation,
    # so it is close to classical here; only HAC captures the overlap.
    assert hac.standard_errors[1] > 2.0 * classical.standard_errors[1]
    assert hac.standard_errors[1] > 2.0 * white.standard_errors[1]
    # The t-statistic is correspondingly deflated -- often from "significant" to not.
    assert abs(hac.t_statistics[1]) < 0.5 * abs(classical.t_statistics[1])


def test_ols_rejects_rank_deficient_designs() -> None:
    """Perfect collinearity is a modelling error, not something to regularise."""
    x = np.random.default_rng(0).standard_normal(100)
    design = np.column_stack([np.ones(100), x, 2.0 * x])
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        ols(x, design)


# --------------------------------------------------------------------------- #
# GARCH
# --------------------------------------------------------------------------- #
def test_garch_recovers_known_parameters(garch_returns: np.ndarray) -> None:
    fitted = fit_garch(garch_returns)
    assert fitted.converged
    assert fitted.alpha == pytest.approx(0.08, abs=0.04)
    assert fitted.beta == pytest.approx(0.90, abs=0.05)
    assert 0.9 < fitted.persistence < 1.0
    assert fitted.half_life > 5.0


def test_garch_variance_targeting_pins_the_unconditional_variance(
    garch_returns: np.ndarray,
) -> None:
    fitted = fit_garch(garch_returns, variance_targeting=True)
    sample = float(np.var(garch_returns - garch_returns.mean()))
    assert fitted.unconditional_variance == pytest.approx(sample, rel=1e-6)


def test_garch_forecast_decays_toward_the_unconditional_variance(
    garch_returns: np.ndarray,
) -> None:
    fitted = fit_garch(garch_returns)
    path = fitted.forecast(400)
    target = fitted.unconditional_variance
    assert abs(path[-1] - target) < abs(path[0] - target)


def test_gjr_garch_detects_the_leverage_effect() -> None:
    """A symmetric GARCH cannot express this; gamma must come out positive."""
    rng = SeedBank(root=22).child("gjr").generator()
    n = 6_000
    r = np.zeros(n)
    v = np.zeros(n)
    v[0] = 1e-4
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        leverage = 0.12 if r[t - 1] < 0 else 0.0
        v[t] = 2e-6 + (0.03 + leverage) * r[t - 1] ** 2 + 0.88 * v[t - 1]
        r[t] = np.sqrt(v[t]) * shocks[t]
    fitted = fit_gjr_garch(r)
    assert fitted.gamma > 0.05
    assert fitted.leverage_ratio > 2.0


def test_garch_rejects_short_samples() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        fit_garch(np.random.default_rng(0).standard_normal(50))


# --------------------------------------------------------------------------- #
# Ornstein-Uhlenbeck
# --------------------------------------------------------------------------- #
def test_ou_estimation_recovers_known_parameters() -> None:
    rng = SeedBank(root=30).child("ou").generator()
    path = simulate_ou(4.0, 0.0, 0.5, n=40_000, dt=1 / 252, rng=rng)
    fitted = fit_ou(path, dt=1 / 252)
    assert fitted.theta == pytest.approx(4.0, rel=0.15)
    assert fitted.sigma == pytest.approx(0.5, rel=0.05)
    assert abs(fitted.mu) < 0.05
    # Stationary dispersion must match the empirical spread of the path.
    assert fitted.stationary_std == pytest.approx(float(np.std(path)), rel=0.05)


def test_ou_half_life_matches_theory() -> None:
    params = OUParameters(theta=4.0, mu=0.0, sigma=0.5, dt=1 / 252)
    assert params.half_life == pytest.approx(np.log(2) / 4.0)
    assert params.stationary_std == pytest.approx(0.5 / np.sqrt(8.0))


def test_ou_first_passage_matches_monte_carlo() -> None:
    """Analytic quadrature against a direct simulation of the hitting time."""
    params = OUParameters(theta=4.0, mu=0.0, sigma=0.5, dt=1 / 252)
    analytic = expected_time_to_mean(params, 2 * params.stationary_std)

    rng = SeedBank(root=31).child("fpt").generator()
    dt = 1 / 5040
    decay = float(np.exp(-4.0 * dt))
    shock = float(0.5 * np.sqrt((1 - decay**2) / 8.0))
    times = []
    for _ in range(3_000):
        x = 2 * params.stationary_std
        t = 0.0
        while x > 0 and t < 20.0:
            x = x * decay + shock * rng.standard_normal()
            t += dt
        times.append(t)
    # Discretisation biases the simulation slightly upward; 8% is generous.
    assert analytic == pytest.approx(float(np.mean(times)), rel=0.08)


def test_ou_reports_non_reversion_rather_than_raising() -> None:
    walk = np.cumsum(SeedBank(root=32).child("walk").generator().standard_normal(2_000))
    fitted = fit_ou(walk)
    assert not fitted.is_mean_reverting
    assert fitted.half_life == np.inf


# --------------------------------------------------------------------------- #
# Cointegration
# --------------------------------------------------------------------------- #
def test_engle_granger_finds_a_real_relationship(
    cointegrated_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    a, b = cointegrated_pair
    result = engle_granger(a, b)
    assert result.is_cointegrated
    assert result.direction_agreement
    assert result.beta == pytest.approx(2.0 / 3.0, abs=0.05)


def test_engle_granger_rejects_independent_walks(
    independent_walks: tuple[np.ndarray, np.ndarray],
) -> None:
    """The spurious-regression null."""
    assert not engle_granger(*independent_walks).is_cointegrated


@pytest.mark.statistical
def test_engle_granger_size_is_controlled() -> None:
    """False-positive rate on independent walks, over 200 pairs."""
    false_positives = 0
    for s in range(200):
        rng = SeedBank(root=2000 + s).child("eg_size").generator()
        a = np.cumsum(rng.standard_normal(500))
        b = np.cumsum(rng.standard_normal(500))
        false_positives += engle_granger(a, b).is_cointegrated
    # Taking the weaker of two regression directions makes this conservative.
    assert false_positives / 200 < 0.05


def test_hedge_ratio_methods_agree_on_clean_data() -> None:
    rng = SeedBank(root=40).child("hedge").generator()
    x = np.cumsum(rng.standard_normal(2_000))
    y = 2.0 * x + rng.standard_normal(2_000) * 0.5
    assert hedge_ratio(y, x, method="ols") == pytest.approx(2.0, abs=0.05)
    assert hedge_ratio(y, x, method="tls") == pytest.approx(2.0, abs=0.05)


def test_johansen_recovers_rank_and_the_cointegrating_vector() -> None:
    rng = SeedBank(root=41).child("johansen").generator()
    factor = np.cumsum(rng.standard_normal(1_500))
    data = np.column_stack(
        [
            factor + rng.standard_normal(1_500) * 0.3,
            2.0 * factor + rng.standard_normal(1_500) * 0.3,
            np.cumsum(rng.standard_normal(1_500)),
        ]
    )
    result = johansen(data)
    assert result.rank() == 1
    vector = result.cointegrating_vector(0)
    # y0 - 0.5 * y1 is stationary; the third series must carry ~zero weight.
    assert vector[0] == pytest.approx(1.0)
    assert vector[1] == pytest.approx(-0.5, abs=0.05)
    assert abs(vector[2]) < 0.05


def test_johansen_finds_rank_two_when_there_are_two_relationships() -> None:
    rng = SeedBank(root=42).child("johansen2").generator()
    f1 = np.cumsum(rng.standard_normal(1_500))
    f2 = np.cumsum(rng.standard_normal(1_500))
    data = np.column_stack(
        [
            f1 + rng.standard_normal(1_500) * 0.2,
            f1 + rng.standard_normal(1_500) * 0.2,
            f2 + rng.standard_normal(1_500) * 0.2,
            f2 + rng.standard_normal(1_500) * 0.2,
        ]
    )
    assert johansen(data).rank() == 2


@pytest.mark.statistical
def test_johansen_size_is_controlled() -> None:
    """Regression test for a 28% false-positive rate caused by a wrong table.

    See ``docs/ddr/DDR-004``: the critical values are simulated for this exact
    VECM specification rather than taken from a publication whose treatment of
    the constant differed.
    """
    spurious = 0
    for s in range(150):
        rng = SeedBank(root=3000 + s).child("joh_size").generator()
        data = np.column_stack([np.cumsum(rng.standard_normal(800)) for _ in range(3)])
        spurious += johansen(data).rank() >= 1
    assert spurious / 150 < 0.12


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_standard_error_matches_the_analytic_one() -> None:
    rng = SeedBank(root=50).child("boot").generator()
    x = rng.standard_normal(600) * 1.0
    result = bs.bootstrap_statistic(x, np.mean, n_replicates=800, rng=rng, method="iid")
    assert result.standard_error == pytest.approx(float(np.std(x, ddof=1) / np.sqrt(600)), rel=0.10)


def test_block_bootstrap_inflates_the_error_under_dependence() -> None:
    r"""For AR(1), the long-run variance ratio is :math:`(1+\rho)/(1-\rho)`.

    With :math:`\rho = 0.7` the standard error should be about
    :math:`\sqrt{1.7/0.3} = 2.38` times the i.i.d. one. Getting this wrong is why
    so many backtest confidence intervals are far too narrow.
    """
    rng = SeedBank(root=51).child("boot_ar").generator()
    n = 4_000
    ar = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + shocks[t]

    iid = bs.bootstrap_statistic(ar, np.mean, n_replicates=400, rng=rng, method="iid")
    block = bs.bootstrap_statistic(ar, np.mean, n_replicates=400, rng=rng, method="stationary")
    ratio = block.standard_error / iid.standard_error
    assert 1.7 < ratio < 3.2


def test_politis_white_block_length_scales_with_dependence() -> None:
    rng = SeedBank(root=52).child("pw").generator()
    white = bs.politis_white_block_length(rng.standard_normal(2_000))
    n = 2_000
    ar = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + shocks[t]
    dependent = bs.politis_white_block_length(ar)
    assert white < 6.0
    assert dependent > 3.0 * white


def test_bootstrap_intervals_are_ordered_sensibly() -> None:
    rng = SeedBank(root=53).child("ci").generator()
    x = rng.standard_normal(500) + 0.1
    result = bs.bootstrap_statistic(x, np.mean, n_replicates=600, rng=rng)
    for method in (result.percentile_interval, result.basic_interval, result.bca_interval):
        lo, hi = method(0.95)
        assert lo < result.point_estimate < hi


# --------------------------------------------------------------------------- #
# Multiple testing
# --------------------------------------------------------------------------- #
def test_correction_power_ordering() -> None:
    """BH is more powerful than Holm, which dominates Bonferroni."""
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.7, 0.9])
    counts = {
        f.__name__: f(p).n_rejected
        for f in (mt.bonferroni, mt.holm, mt.benjamini_hochberg, mt.benjamini_yekutieli)
    }
    assert counts["benjamini_hochberg"] >= counts["holm"] >= counts["bonferroni"]
    assert counts["benjamini_yekutieli"] <= counts["benjamini_hochberg"]


def test_adjusted_p_values_never_decrease_below_the_raw_values() -> None:
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.6])
    for f in (mt.bonferroni, mt.holm, mt.benjamini_hochberg, mt.benjamini_yekutieli):
        assert np.all(f(p).adjusted >= p - 1e-15)
        assert np.all(f(p).adjusted <= 1.0)


def test_reality_check_and_spa_find_nothing_in_noise() -> None:
    rng = SeedBank(root=60).child("snoop").generator()
    junk = rng.standard_normal((600, 40)) * 0.01
    assert mt.whites_reality_check(junk, n_bootstrap=300, rng=rng).p_value > 0.05
    assert mt.hansen_spa(junk, n_bootstrap=300, rng=rng).p_value > 0.05


def test_spa_detects_a_planted_edge() -> None:
    rng = SeedBank(root=61).child("spa").generator()
    performance = rng.standard_normal((800, 20)) * 0.01
    performance[:, 7] += 0.004
    result = mt.hansen_spa(performance, n_bootstrap=400, rng=rng)
    assert result.p_value < 0.05
    assert result.best_index == 7


def test_stepm_identifies_which_strategies_are_superior() -> None:
    """SPA says 'at least one'; StepM says which -- the question you actually have."""
    rng = SeedBank(root=62).child("stepm").generator()
    performance = rng.standard_normal((800, 20)) * 0.01
    performance[:, 3] += 0.005
    performance[:, 11] += 0.005
    rejected = set(np.nonzero(mt.stepm(performance, n_bootstrap=400, rng=rng))[0].tolist())
    assert {3, 11} <= rejected
    assert len(rejected) <= 5  # few false discoveries
