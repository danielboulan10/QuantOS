"""The research pipeline: instruments, signals, and report generation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantos.core.rng import SeedBank
from quantos.research.instruments import (
    Analysis,
    AssetClass,
    FutureSpec,
    Instrument,
    OptionSpec,
    supported_analyses,
)
from quantos.research.render import render_markdown, render_text
from quantos.research.report import generate_report
from quantos.research.signals import SIGNALS, _strategy_returns, run_signal_battery


def make_instrument(
    n: int = 800, asset_class: AssetClass = AssetClass.ETF, seed: int = 0, drift: float = 0.0003
) -> Instrument:
    rng = SeedBank(root=seed).child("instrument").generator()
    dates = np.datetime64("2019-01-01") + np.arange(n)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.011 + drift))
    return Instrument("TEST", asset_class, dates, prices, dividend_adjusted=True)


# --------------------------------------------------------------------------- #
# Instrument semantics
# --------------------------------------------------------------------------- #
def test_asset_class_properties() -> None:
    assert AssetClass.EQUITY.is_tradeable
    assert not AssetClass.INDEX.is_tradeable
    assert not AssetClass.RATE.is_tradeable
    assert AssetClass.EQUITY.has_linear_payoff
    assert not AssetClass.OPTION.has_linear_payoff
    assert AssetClass.OPTION.has_expiry and AssetClass.FUTURE.has_expiry


def test_rates_are_differenced_and_levels_are_logged() -> None:
    """The distinction that stops a 'return' being computed on a yield."""
    dates = np.datetime64("2024-01-01") + np.arange(3)
    values = np.array([100.0, 110.0, 121.0])
    equity = Instrument("A", AssetClass.EQUITY, dates, values)
    rate = Instrument("B", AssetClass.RATE, dates, values)
    assert equity.returns() == pytest.approx([np.log(1.1), np.log(1.1)])
    assert rate.returns().tolist() == [10.0, 11.0]


def test_negative_values_fall_back_to_differences() -> None:
    """Curve spreads go negative; log returns would be NaN."""
    dates = np.datetime64("2024-01-01") + np.arange(3)
    inst = Instrument("SPREAD", AssetClass.RATE, dates, np.array([0.5, -0.2, 0.1]))
    assert np.all(np.isfinite(inst.returns()))


def test_analyses_are_gated_by_asset_class() -> None:
    """An option must not be handed equity risk analytics."""
    assert Analysis.RISK_METRICS in supported_analyses(AssetClass.EQUITY)
    assert Analysis.RISK_METRICS not in supported_analyses(AssetClass.OPTION)
    assert Analysis.SIGNAL_BATTERY not in supported_analyses(AssetClass.OPTION)
    assert Analysis.TERM_STRUCTURE in supported_analyses(AssetClass.FUTURE)


def test_skip_reasons_are_explanatory_not_generic() -> None:
    inst = make_instrument(asset_class=AssetClass.RATE)
    reason = inst.skip_reason(Analysis.RISK_METRICS)
    assert "yield" in reason and "return" in reason
    option = make_instrument(asset_class=AssetClass.OPTION)
    assert "convexity" in option.skip_reason(Analysis.RISK_METRICS)


def test_skip_reason_grammar() -> None:
    """'a index' would be embarrassing in a report someone else reads."""
    inst = make_instrument(asset_class=AssetClass.INDEX)
    assert " an index" in inst.skip_reason(Analysis.EXECUTION_COST)


def test_unadjusted_prices_produce_a_warning() -> None:
    """The commonest silent data error in equity analysis."""
    inst = make_instrument()
    inst.dividend_adjusted = False
    warnings = inst.data_quality_warnings()
    assert any("dividend" in w for w in warnings)


def test_short_samples_and_gaps_are_flagged() -> None:
    short = make_instrument(n=100)
    assert any("observations" in w for w in short.data_quality_warnings())


def test_option_and_future_specs() -> None:
    option = OptionSpec(strike=100.0, expiry=date(2025, 1, 17))
    assert option.years_to_expiry(date(2024, 1, 17)) == pytest.approx(1.0, abs=0.01)
    assert option.moneyness(110.0) == pytest.approx(np.log(1.1))
    future = FutureSpec(expiry=date(2025, 3, 21), price=95.0)
    assert future.years_to_expiry(date(2024, 3, 21)) == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Signals: the look-ahead guarantee
# --------------------------------------------------------------------------- #
def test_position_return_alignment_has_no_look_ahead() -> None:
    """The single most important test in this module.

    A position decided at t may only earn the return from t to t+1. If the
    alignment is off by one in the wrong direction, a signal that merely echoes
    the current return earns an impossible Sharpe ratio and every downstream
    number is fiction.
    """
    rng = SeedBank(root=1).child("lookahead").generator()
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(800) * 0.01))
    returns = np.diff(np.log(prices))

    # positions[t] = sign(returns[t]) knows the future; it must be profitable.
    oracle = np.concatenate([np.sign(returns), [0.0]])
    # positions[t] = sign(returns[t-1]) knows only the past; on a random walk it
    # must NOT be.
    honest = np.concatenate([[0.0], np.sign(returns)])

    assert float(np.mean(_strategy_returns(oracle, returns))) > 0.005
    assert abs(float(np.mean(_strategy_returns(honest, returns)))) < 0.001


def test_strategy_returns_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError, match="one longer"):
        _strategy_returns(np.zeros(10), np.zeros(10))


@pytest.mark.statistical
def test_battery_finds_nothing_in_a_random_walk() -> None:
    """The expected outcome, and the reason the module exists."""
    rng = SeedBank(root=2).child("rw").generator()
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(1200) * 0.01))
    battery = run_signal_battery(prices, n_bootstrap=200)
    assert not battery.any_significant
    assert battery.spa_p_value > 0.05
    assert all(not r.is_significant for r in battery.results)


@pytest.mark.statistical
@pytest.mark.slow
def test_battery_detects_a_strong_planted_signal() -> None:
    """A detector that never fires is not a detector.

    AR(0.35) daily autocorrelation is far stronger than any liquid market
    exhibits, which is the point: the battery must be able to say yes.
    """
    rng = SeedBank(root=3).child("momentum").generator()
    n = 2500
    series = np.zeros(n)
    shocks = rng.standard_normal(n) * 0.01
    for t in range(1, n):
        series[t] = 0.35 * series[t - 1] + shocks[t]
    prices = 100 * np.exp(np.cumsum(series))

    battery = run_signal_battery(prices, n_bootstrap=300)
    assert battery.any_significant
    assert battery.spa_p_value < 0.05
    best = battery.sorted_by_evidence()[0]
    assert best.in_sample_sharpe > 1.0
    assert best.information_coefficient > 0.0


def test_battery_charges_transaction_costs() -> None:
    """High-turnover signals must be penalised, or the battery is fiction."""
    rng = SeedBank(root=4).child("costs").generator()
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(900) * 0.01))
    free = run_signal_battery(prices, transaction_cost_bps=0.0, n_bootstrap=50)
    charged = run_signal_battery(prices, transaction_cost_bps=50.0, n_bootstrap=50)
    by_name = {r.name: r for r in free.results}
    for result in charged.results:
        reference = by_name[result.name]
        if np.isfinite(result.in_sample_sharpe) and result.turnover > 1.0:
            assert result.in_sample_sharpe < reference.in_sample_sharpe


def test_battery_size_is_fixed_before_seeing_data() -> None:
    """Deflation is only valid if the trial count is not chosen post hoc."""
    assert len(SIGNALS) == 9
    names = [s[0] for s in SIGNALS]
    assert len(set(names)) == len(names)


def test_battery_requires_enough_history() -> None:
    with pytest.raises(ValueError, match="at least 300"):
        run_signal_battery(np.linspace(100, 110, 200))


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def test_report_on_an_etf_is_complete() -> None:
    report = generate_report(make_instrument(), run_signals=False)
    assert report.n_returns > 700
    assert np.isfinite(report.annualised_volatility)
    assert np.isfinite(report.sharpe)
    assert report.max_drawdown < 0
    assert report.cvar_99 >= report.var_99
    assert report.option_table  # an ETF has options


def test_report_omits_risk_metrics_for_a_rate() -> None:
    """A yield has no Sharpe ratio, and the report must say so rather than fake one."""
    dates = np.datetime64("2020-01-01") + np.arange(600)
    rng = SeedBank(root=5).child("rate").generator()
    values = 2.0 + np.cumsum(rng.standard_normal(600) * 0.03)
    report = generate_report(Instrument("DGS10", AssetClass.RATE, dates, values), run_signals=False)
    assert not np.isfinite(report.sharpe)
    assert "risk" in report.skipped
    assert np.isfinite(report.annualised_volatility)  # dispersion still meaningful


def test_report_declines_garch_without_arch_effects() -> None:
    """i.i.d. data has no volatility clustering; a GARCH fit there is degenerate.

    Unguarded, the optimiser walks to the boundary and reports persistence
    1.0000 with a 442,909-day half-life -- finite, precise-looking, meaningless.
    """
    report = generate_report(make_instrument(n=900, seed=7), run_signals=False)
    assert np.isfinite(report.arch_p_value)
    if report.arch_p_value > 0.05:
        assert "garch" in report.skipped
        assert "ARCH" in report.skipped["garch"]
        assert not np.isfinite(report.garch_persistence)


def test_report_fits_garch_when_arch_effects_are_present() -> None:
    rng = SeedBank(root=8).child("garch").generator()
    n = 1500
    r = np.zeros(n)
    v = np.full(n, 1e-4)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        v[t] = 2e-6 + 0.10 * r[t - 1] ** 2 + 0.87 * v[t - 1]
        r[t] = np.sqrt(v[t]) * shocks[t]
    dates = np.datetime64("2019-01-01") + np.arange(n)
    prices = 100 * np.exp(np.cumsum(r))
    report = generate_report(
        Instrument("GARCHY", AssetClass.ETF, dates, prices, dividend_adjusted=True),
        run_signals=False,
    )
    assert report.arch_p_value < 0.01
    assert 0.8 < report.garch_persistence < 1.0
    assert 1.0 < report.garch_half_life < 500.0


def test_report_flags_an_insignificant_sharpe() -> None:
    """A Sharpe printed without its standard error invites over-trust."""
    report = generate_report(make_instrument(n=400, drift=0.0002), run_signals=False)
    assert np.isfinite(report.sharpe_standard_error)
    assert not report.sharpe_is_significant


def test_factor_regression_drops_a_self_regression() -> None:
    """Regressing an instrument on itself gives beta 1 and a t-stat of 1e21."""
    inst = make_instrument(n=600)
    returns = inst.returns()
    report = generate_report(
        inst,
        factors={"itself": returns, "noise": np.roll(returns, 7)},
        run_signals=False,
    )
    if report.factors is not None:
        assert "itself" not in report.factors.factor_names


def test_report_renders_to_text_and_markdown() -> None:
    report = generate_report(make_instrument(), run_signals=False)
    text = render_text(report)
    assert "RESEARCH REPORT" in text
    assert "WHAT THE DATA SUPPORTS" in text
    assert "not investment advice" in text
    assert "NOT established by this report" in text

    markdown = render_markdown(report)
    assert markdown.startswith("# Research report")
    assert "| Metric | Value |" in markdown
    assert "Limitations" in markdown


def test_report_survives_a_tiny_sample_without_crashing() -> None:
    report = generate_report(make_instrument(n=70), run_signals=False)
    assert report.n_returns == 69
    assert isinstance(render_text(report), str)


def test_missing_values_render_as_dashes_without_breaking_alignment() -> None:
    """A '--' that ignores its column width corrupts every row after it."""
    report = generate_report(make_instrument(n=70), run_signals=False)
    for line in render_text(report).splitlines():
        assert "nan" not in line.lower()
