"""Validation for the real-data analysis battery.

This module had no test file at all and sat at 29.8% line coverage, reached only
incidentally through the CLI. That is the same shape as the module mutation
testing later found scoring 0% -- code that runs in production and is checked by
nothing.

The tests below concentrate on the *judgements* rather than the arithmetic: the
statistics are already covered where they are implemented, but the decisions
about which of them apply to which kind of series, and how two tests should be
read together, live only here.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.data.analysis import (
    CrossSectionReport,
    SeriesReport,
    analyse_cross_section,
    analyse_series,
)


def business_dates(n: int, start: str = "2015-01-01") -> np.ndarray:
    origin = np.datetime64(start, "D")
    return origin + np.arange(n, dtype="timedelta64[D]")


def geometric_series(n: int, *, drift: float = 0.0004, vol: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n) * vol + drift
    return 100.0 * np.exp(np.cumsum(steps))


# --------------------------------------------------------------------------- #
# What applies to which kind of series
# --------------------------------------------------------------------------- #
def test_a_tradeable_level_gets_return_and_risk_statistics():
    n = 1500
    report = analyse_series("SPY", "S&P 500 ETF", "level", business_dates(n), geometric_series(n))

    assert report.n_observations == n
    assert np.isfinite(report.annualised_return)
    assert np.isfinite(report.annualised_volatility)
    assert np.isfinite(report.sharpe)
    assert report.max_drawdown <= 0.0
    assert report.var_95 > 0 and report.cvar_95 >= report.var_95


def test_a_yield_series_gets_no_sharpe_ratio():
    """The 'return' of a yield is not a return.

    A 10-year Treasury yield moving 4.25 to 4.35 has not earned anything. A
    Sharpe ratio computed on those differences is a number with no
    interpretation, and printing one invites a comparison against an equity
    Sharpe that means something entirely different.
    """
    n = 1200
    rng = np.random.default_rng(3)
    yields = 4.0 + np.cumsum(rng.standard_normal(n) * 0.03)

    report = analyse_series("DGS10", "10-year Treasury", "rate", business_dates(n), yields)

    assert not np.isfinite(report.sharpe), "a yield must not be given a Sharpe ratio"
    assert not np.isfinite(report.annualised_return)
    # Distributional statistics still apply: the differences are a real series.
    assert np.isfinite(report.annualised_volatility)


def test_the_cvar_is_never_smaller_than_the_var_it_conditions_on():
    """CVaR is the mean loss beyond VaR, so it cannot be the smaller number.

    A sign or ordering slip here is invisible in isolation and inverts every
    tail statement the report makes.
    """
    n = 2000
    report = analyse_series("X", "Test", "level", business_dates(n), geometric_series(n, seed=5))

    assert report.cvar_95 >= report.var_95
    assert report.cvar_99 >= report.var_99
    assert report.var_99 >= report.var_95, "a further tail is a larger loss"


def test_fat_tails_are_detected_when_they_are_planted():
    """The battery must respond to the thing it measures.

    Without this every tail statistic could return a constant and the tests
    above would still pass.
    """
    n = 2000
    rng = np.random.default_rng(11)
    calm = np.exp(np.cumsum(rng.standard_normal(n) * 0.01)) * 100

    steps = rng.standard_normal(n) * 0.01
    steps[::40] *= 8.0  # occasional violent days
    wild = np.exp(np.cumsum(steps)) * 100

    calm_report = analyse_series("C", "Calm", "level", business_dates(n), calm)
    wild_report = analyse_series("W", "Wild", "level", business_dates(n), wild)

    assert wild_report.excess_kurtosis > calm_report.excess_kurtosis
    assert wild_report.cvar_99 > calm_report.cvar_99


# --------------------------------------------------------------------------- #
# Reading two tests together, which is the part only this module does
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("adf_rejects", "kpss_rejects", "expected"),
    [
        (True, False, "stationary (both tests agree)"),
        (False, True, "unit root / integrated (both tests agree)"),
        (True, True, "tests disagree"),
        (False, False, "inconclusive"),
    ],
)
def test_the_stationarity_verdict_reads_adf_and_kpss_jointly(adf_rejects, kpss_rejects, expected):
    """ADF and KPSS have opposite null hypotheses, which is the whole point.

    Reporting either alone is how a series gets called stationary on the
    strength of one test failing to reject. All four combinations mean something
    different and the fourth -- neither rejecting -- is 'not enough data', not
    'stationary'.
    """
    report = SeriesReport(
        key="X",
        name="X",
        kind="level",
        n_observations=100,
        start="2020-01-01",
        end="2024-01-01",
        latest=1.0,
        adf_rejects_unit_root=adf_rejects,
        kpss_rejects_stationarity=kpss_rejects,
    )
    assert expected in report.stationarity_verdict


def test_a_random_walk_is_not_called_stationary_and_its_returns_are():
    """The end-to-end version of the test above, on data with a known answer."""
    n = 1500
    prices = geometric_series(n, drift=0.0, seed=7)

    level = analyse_series("P", "Price", "level", business_dates(n), prices)
    assert "stationary (both tests agree)" not in level.stationarity_verdict


def test_the_tail_severity_ratio_is_reported_and_guards_against_a_zero_var():
    n = 1500
    report = analyse_series("X", "X", "level", business_dates(n), geometric_series(n, seed=9))
    assert "CVaR/VaR at 99%" in report.tail_severity

    degenerate = SeriesReport(
        key="X",
        name="X",
        kind="level",
        n_observations=10,
        start="2020-01-01",
        end="2020-01-10",
        latest=1.0,
    )
    assert degenerate.tail_severity == "n/a", "no VaR means no ratio, not a division by zero"


# --------------------------------------------------------------------------- #
# Cross-section
# --------------------------------------------------------------------------- #
def test_the_correlation_matrix_is_symmetric_with_a_unit_diagonal():
    rng = np.random.default_rng(2)
    n = 800
    market = rng.standard_normal(n) * 0.01
    series = {
        "A": market + rng.standard_normal(n) * 0.004,
        "B": market + rng.standard_normal(n) * 0.004,
        "C": rng.standard_normal(n) * 0.01,
    }

    report = analyse_cross_section(series)

    assert report.correlation.shape == (3, 3)
    assert np.allclose(np.diag(report.correlation), 1.0)
    assert np.allclose(report.correlation, report.correlation.T)


def test_the_most_correlated_pair_is_the_one_that_shares_a_factor():
    rng = np.random.default_rng(4)
    n = 900
    market = rng.standard_normal(n) * 0.01
    report = analyse_cross_section(
        {
            "TWIN_A": market + rng.standard_normal(n) * 0.002,
            "TWIN_B": market + rng.standard_normal(n) * 0.002,
            "LONER": rng.standard_normal(n) * 0.01,
        }
    )

    left, right, value = report.most_correlated()
    assert {left, right} == {"TWIN_A", "TWIN_B"}
    assert value > 0.8


def test_the_strongest_pair_is_found_by_magnitude_not_by_sign():
    """A -0.9 relationship is stronger than a +0.3 one.

    Ranking by the raw value rather than its absolute value would report the
    weaker pair, and a hedge is exactly the case where the strong relationship
    is negative.
    """
    report = CrossSectionReport(
        names=["A", "B", "C"],
        correlation=np.array([[1.0, 0.3, -0.9], [0.3, 1.0, 0.1], [-0.9, 0.1, 1.0]]),
        n_common_dates=500,
        start="2020-01-01",
        end="2022-01-01",
    )

    left, right, value = report.most_correlated()
    assert {left, right} == {"A", "C"}
    assert value == pytest.approx(-0.9)


def test_cointegration_is_tested_on_levels_and_not_on_returns():
    """Testing cointegration on returns is a category error.

    Returns are already stationary, so the test rejects the unit root every time
    and reports a relationship that is an artefact of the transform. The API
    takes levels separately so a caller cannot pass the wrong one by accident.
    """
    n = 900
    rng = np.random.default_rng(6)
    base = np.cumsum(rng.standard_normal(n) * 0.01) + 4.0
    partner = 2.0 * base + rng.standard_normal(n) * 0.02  # genuinely cointegrated

    levels = {"A": np.exp(base), "B": np.exp(partner)}
    transformed = {name: np.diff(np.log(values)) for name, values in levels.items()}

    report = analyse_cross_section(transformed, levels, dates=business_dates(n - 1))

    assert report.cointegration, "level pairs must be tested"
    (is_cointegrated, _statistic, hedge_ratio) = next(iter(report.cointegration.values()))
    assert is_cointegrated
    assert hedge_ratio > 0


def test_unrelated_random_walks_are_not_reported_as_cointegrated():
    """The guard for the test above. Two independent random walks are the
    textbook spurious-regression case and must come back negative."""
    n = 900
    rng = np.random.default_rng(8)
    levels = {
        "A": np.exp(np.cumsum(rng.standard_normal(n) * 0.01) + 4.0),
        "B": np.exp(np.cumsum(rng.standard_normal(n) * 0.01) + 4.0),
    }
    transformed = {name: np.diff(np.log(values)) for name, values in levels.items()}

    report = analyse_cross_section(transformed, levels, dates=business_dates(n - 1))
    is_cointegrated, _, _ = next(iter(report.cointegration.values()))
    assert not is_cointegrated


def test_a_single_series_has_no_cross_section_to_report():
    report = analyse_cross_section({"ONLY": np.zeros(100)})
    assert report.correlation.shape == (1, 1)
    assert report.most_correlated() == ("", "", 0.0)
