"""Risk metrics and portfolio construction."""

from __future__ import annotations

import numpy as np
import pytest

from quantos.core.linalg import (
    condition_number,
    effective_rank,
    marchenko_pastur_edge,
    nearest_correlation,
    safe_cholesky,
)
from quantos.core.rng import SeedBank
from quantos.risk.metrics import (
    conditional_value_at_risk,
    drawdown_series,
    max_drawdown,
    performance_report,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
    value_at_risk,
)
from quantos.risk.portfolio import (
    diversification_ratio,
    hierarchical_risk_parity,
    kelly_weights,
    ledoit_wolf_shrinkage,
    mean_variance,
    minimum_variance,
    risk_contributions,
    risk_parity,
)


def test_max_drawdown_locates_peak_and_trough() -> None:
    depth, peak, trough = max_drawdown([100.0, 120.0, 90.0, 110.0])
    assert (round(depth, 10), peak, trough) == (-0.25, 1, 2)


def test_drawdown_requires_positive_equity() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        drawdown_series([100.0, 0.0, 50.0])


def test_cvar_is_subadditive_where_var_is_not() -> None:
    """The reason CVaR can be allocated across a portfolio and VaR cannot."""
    rng = SeedBank(root=1).child("coherent").generator()
    a = rng.standard_normal(50_000) * 0.01
    b = rng.standard_normal(50_000) * 0.01
    assert conditional_value_at_risk(a + b) <= (
        conditional_value_at_risk(a) + conditional_value_at_risk(b) + 1e-12
    )


def test_cvar_exceeds_var_and_sees_tail_shape() -> None:
    """Two series with equal VaR but different tails must differ in CVaR."""
    mild = np.concatenate([np.full(95, 0.01), np.full(5, -0.05)])
    severe = np.concatenate([np.full(95, 0.01), np.full(5, -0.50)])
    assert value_at_risk(mild, confidence=0.95) == pytest.approx(
        value_at_risk(severe, confidence=0.95), abs=0.03
    )
    assert conditional_value_at_risk(severe) > 5.0 * conditional_value_at_risk(mild)


def test_var_is_reported_as_a_positive_loss() -> None:
    rng = SeedBank(root=2).child("var").generator()
    losses = rng.standard_normal(10_000) * 0.02 - 0.001
    assert value_at_risk(losses, confidence=0.99) > 0.0
    assert conditional_value_at_risk(losses, confidence=0.99) > value_at_risk(
        losses, confidence=0.99
    )


def test_parametric_var_understates_fat_tails() -> None:
    from quantos.core.distributions import StudentT

    fat = StudentT(3.0).sample(200_000, SeedBank(root=3).child("fat").generator()) * 0.01
    assert value_at_risk(fat, confidence=0.99, method="parametric") < value_at_risk(
        fat, confidence=0.99, method="historical"
    )


def test_sortino_penalises_downside_only() -> None:
    """Upside volatility must not be punished."""
    rng = SeedBank(root=4).child("sortino").generator()
    base = rng.standard_normal(5_000) * 0.01 + 0.0005
    upside_only = np.where(base > 0, base * 3.0, base)
    assert sortino_ratio(upside_only) > sortino_ratio(base)
    assert sharpe_ratio(upside_only) < sortino_ratio(upside_only)


def test_ulcer_index_distinguishes_recovery_speed() -> None:
    """Same maximum drawdown, very different time underwater."""
    quick = np.array([100.0, 80.0] + [100.0] * 50)
    slow = np.array([100.0, 80.0] + [80.5] * 48 + [100.0, 100.0])
    assert max_drawdown(quick)[0] == pytest.approx(max_drawdown(slow)[0], abs=0.01)
    assert ulcer_index(slow) > 3.0 * ulcer_index(quick)


def test_autocorrelation_adjusted_sharpe_is_lower_under_momentum() -> None:
    """Positive autocorrelation makes the naive sqrt(252) OVERSTATE the Sharpe."""
    rng = SeedBank(root=5).child("ac").generator()
    n = 5_000
    r = np.zeros(n)
    shocks = rng.standard_normal(n) * 0.01
    for t in range(1, n):
        r[t] = 0.3 * r[t - 1] + shocks[t] + 0.0004
    assert sharpe_ratio(r, adjust_autocorrelation=True) < sharpe_ratio(r)


def test_performance_report_is_internally_consistent() -> None:
    rng = SeedBank(root=6).child("report").generator()
    r = rng.standard_normal(2_000) * 0.01 + 0.0004
    report = performance_report(r)
    assert report.n_periods == 2_000
    assert report.max_drawdown < 0
    assert report.cvar_95 >= report.var_95
    assert report.cvar_99 >= report.var_99
    assert 0.0 < report.hit_rate < 1.0
    assert "Sharpe" in str(report)


# --------------------------------------------------------------------------- #
def test_nearest_correlation_repairs_an_indefinite_matrix() -> None:
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.min(np.linalg.eigvalsh(bad)) < 0
    fixed = nearest_correlation(bad)
    assert np.min(np.linalg.eigvalsh(fixed)) > -1e-9
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.allclose(fixed, fixed.T)


def test_safe_cholesky_reports_the_jitter_it_needed() -> None:
    """Repair is fine; silent repair is not."""
    clean = safe_cholesky(np.array([[4.0, 1.0], [1.0, 3.0]]))
    assert not clean.was_repaired
    singular = safe_cholesky(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert singular.was_repaired
    assert singular.jitter > 0


def test_effective_rank_measures_spanned_directions() -> None:
    assert effective_rank(np.eye(5)) == pytest.approx(5.0)
    assert effective_rank(np.ones((5, 5))) == pytest.approx(1.0)


def test_marchenko_pastur_edge() -> None:
    lo, hi = marchenko_pastur_edge(100, 500)
    assert hi == pytest.approx((1 + np.sqrt(0.2)) ** 2)
    assert lo == pytest.approx((1 - np.sqrt(0.2)) ** 2)


def test_ledoit_wolf_improves_conditioning_without_a_tuning_parameter() -> None:
    rng = SeedBank(root=10).child("lw").generator()
    returns = rng.standard_normal((80, 60)) * 0.01
    shrunk, delta = ledoit_wolf_shrinkage(returns)
    assert 0.0 < delta <= 1.0
    assert condition_number(shrunk) < condition_number(np.cov(returns, rowvar=False))
    assert np.allclose(shrunk, shrunk.T)


def test_risk_parity_gives_exact_inverse_volatility_when_uncorrelated() -> None:
    """Regression test: the textbook fixed point OSCILLATES and returned 1:1."""
    weights = risk_parity(np.diag([0.04, 0.16])).weights
    assert weights[0] / weights[1] == pytest.approx(2.0, abs=1e-9)
    three = risk_parity(np.diag([0.01, 0.04, 0.09])).weights
    assert three[0] / three[2] == pytest.approx(3.0, abs=1e-9)


def test_risk_parity_equalises_contributions_and_satisfies_euler() -> None:
    cov = np.array([[0.04, 0.02, 0.01], [0.02, 0.09, 0.03], [0.01, 0.03, 0.16]])
    solution = risk_parity(cov)
    contributions = risk_contributions(solution.weights, cov)
    assert np.ptp(contributions) < 1e-12
    # Euler's theorem: contributions sum exactly to portfolio volatility.
    assert float(np.sum(contributions)) == pytest.approx(solution.volatility, rel=1e-12)


def test_minimum_variance_tilts_to_the_lower_volatility_asset() -> None:
    solution = minimum_variance(np.array([[0.04, 0.01], [0.01, 0.09]]))
    assert solution.weights[0] > solution.weights[1]
    assert float(np.sum(solution.weights)) == pytest.approx(1.0)
    assert np.all(solution.weights >= 0)


def test_hrp_works_when_the_covariance_is_singular() -> None:
    """T < N. Markowitz is undefined here; HRP never inverts anything."""
    rng = SeedBank(root=11).child("hrp").generator()
    returns = rng.standard_normal((30, 40)) * 0.01
    solution = hierarchical_risk_parity(returns=returns)
    assert np.all(solution.weights > 0)
    assert float(np.sum(solution.weights)) == pytest.approx(1.0)
    assert solution.effective_positions > 5.0


def test_long_only_solutions_stay_on_the_simplex() -> None:
    cov = np.array([[0.04, 0.02, 0.01], [0.02, 0.09, 0.03], [0.01, 0.03, 0.16]])
    mu = np.array([0.08, 0.12, 0.05])
    for solution in (minimum_variance(cov), mean_variance(mu, cov), risk_parity(cov)):
        assert np.all(solution.weights >= -1e-12)
        assert float(np.sum(solution.weights)) == pytest.approx(1.0, abs=1e-9)


def test_kelly_fraction_scales_the_position_linearly() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    mu = np.array([0.08, 0.12])
    full = kelly_weights(mu, cov, fraction=1.0)
    half = kelly_weights(mu, cov, fraction=0.5)
    assert np.allclose(half, full * 0.5)
    capped = kelly_weights(mu, cov, fraction=1.0, max_leverage=0.5)
    assert float(np.sum(np.abs(capped))) == pytest.approx(0.5)


def test_diversification_ratio_is_one_for_a_single_asset() -> None:
    assert diversification_ratio([1.0], np.array([[0.04]])) == pytest.approx(1.0)
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    assert diversification_ratio([0.5, 0.5], cov) > 1.0


def test_shrinkage_helps_out_of_sample() -> None:
    """The error-maximisation problem, and its remedy, in one assertion."""
    rng = SeedBank(root=12).child("oos").generator()
    factor_train = rng.standard_normal((120, 1))
    factor_test = rng.standard_normal((2_000, 1))
    betas = rng.uniform(0.4, 1.6, 40)
    idio = rng.uniform(0.005, 0.03, 40)
    train = factor_train @ betas[None, :] * 0.01 + rng.standard_normal((120, 40)) * idio
    test = factor_test @ betas[None, :] * 0.01 + rng.standard_normal((2_000, 40)) * idio

    unshrunk = minimum_variance(returns=train, shrink=False).weights
    shrunk = minimum_variance(returns=train, shrink=True).weights
    assert float(np.std(test @ shrunk)) <= float(np.std(test @ unshrunk))
