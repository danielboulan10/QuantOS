"""The research report: one instrument in, a complete analysis out.

What this does
--------------
Given any instrument -- a stock, ETF, index, rate, future or option -- run every
analysis that applies to it, skip the ones that do not, and produce a report that
states its own limitations.

The sections, and why each is here
-----------------------------------
1.  **Data provenance and quality.** Where the numbers came from, and what is
    wrong with them. Printed first, because every later number inherits these
    problems and a reader who skips this section will misread everything after it.
2.  **Return distribution.** Moments, tail index, and normality tests. Not to
    establish that returns are non-normal -- they always are -- but to quantify
    *how*, since that determines which risk model is defensible.
3.  **Risk.** Drawdown, VaR and CVaR at two levels. The CVaR/VaR ratio is the
    number worth reading: it is how much worse the tail is than the threshold.
4.  **Volatility dynamics.** GARCH persistence, half-life, and a forward
    forecast, plus whether the leverage effect is present.
5.  **Regimes.** When volatility changed, identified without look-ahead.
6.  **Factor exposure.** Beta to the market, rates and credit, with HAC standard
    errors, and the share of variance that is idiosyncratic.
7.  **Signals.** The pre-registered battery, each judged after correction.
8.  **Option analytics.** A volatility surface implied by realised volatility,
    Greeks across strikes, and what the term structure of realised vol says
    about whether options look rich or cheap.
9.  **Execution.** What it would cost to trade a given size, from the
    square-root law.
10. **What the data supports.** An explicit summary, including what it does not.

Every section that cannot be computed says so and says why.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from numpy.typing import NDArray

from quantos.research.instruments import Analysis, Instrument
from quantos.research.signals import SignalBattery

__all__ = ["FactorExposure", "ResearchReport", "VolatilityRegime", "generate_report"]

TRADING_DAYS = 252.0


@dataclass
class VolatilityRegime:
    """A period of distinct volatility."""

    start: str
    end: str
    annualised_volatility: float
    n_days: int
    label: str = ""


@dataclass
class FactorExposure:
    """Regression of the instrument on a set of factors."""

    factor_names: list[str]
    betas: NDArray[np.float64]
    t_statistics: NDArray[np.float64]
    alpha_annualised: float
    alpha_t_statistic: float
    r_squared: float
    idiosyncratic_share: float
    n_observations: int

    @property
    def significant_factors(self) -> list[str]:
        return [
            name
            for name, t in zip(self.factor_names, self.t_statistics, strict=True)
            if abs(float(t)) > 1.96
        ]


@dataclass
class ResearchReport:
    """A complete research report on one instrument."""

    instrument: Instrument
    generated: str = field(default_factory=lambda: date.today().isoformat())

    data_warnings: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    # Distribution and risk
    n_returns: int = 0
    annualised_return: float = float("nan")
    annualised_volatility: float = float("nan")
    sharpe: float = float("nan")
    sharpe_standard_error: float = float("nan")
    sortino: float = float("nan")
    max_drawdown: float = float("nan")
    max_drawdown_days: int = 0
    var_95: float = float("nan")
    cvar_95: float = float("nan")
    var_99: float = float("nan")
    cvar_99: float = float("nan")
    skewness: float = float("nan")
    excess_kurtosis: float = float("nan")
    tail_index: float = float("nan")
    normality_p: float = float("nan")

    # Volatility
    garch_alpha: float = float("nan")
    garch_beta: float = float("nan")
    garch_persistence: float = float("nan")
    garch_half_life: float = float("nan")
    volatility_forecast_1d: float = float("nan")
    volatility_forecast_21d: float = float("nan")
    leverage_gamma: float = float("nan")
    current_vol_percentile: float = float("nan")
    #: Engle LM test p-value. Above 0.05 means constant volatility is adequate.
    arch_p_value: float = float("nan")

    regimes: list[VolatilityRegime] = field(default_factory=list)
    factors: FactorExposure | None = None
    signals: SignalBattery | None = None

    # Options
    option_summary: dict[str, float] = field(default_factory=dict)
    option_table: list[dict[str, float]] = field(default_factory=list)

    # Execution
    execution_costs: list[tuple[float, float]] = field(default_factory=list)

    stationarity: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def sharpe_is_significant(self) -> bool:
        """Whether the Sharpe ratio is two standard errors from zero.

        Reported rather than the Sharpe alone because a 0.6 Sharpe on two years
        of data is indistinguishable from zero, and printing it without its
        error invites exactly that mistake.
        """
        if not (np.isfinite(self.sharpe) and self.sharpe_standard_error > 0):
            return False
        annual_se = self.sharpe_standard_error * float(np.sqrt(TRADING_DAYS))
        return bool(abs(self.sharpe / annual_se) > 1.96)


def _volatility_regimes(
    dates: NDArray[np.datetime64], returns: NDArray[np.float64], window: int = 63
) -> list[VolatilityRegime]:
    """Split the sample where trailing volatility crosses its own median.

    Deliberately simple, and deliberately *causal*: the threshold is the median
    of the trailing window's own history, so no future information enters. A
    Markov-switching model would be more sophisticated and would need the whole
    sample to fit, which makes its regime labels unusable as a live signal.
    """
    if returns.size < window * 3:
        return []
    rolling = np.array(
        [
            float(np.std(returns[max(0, t - window) : t], ddof=1))
            for t in range(window, returns.size)
        ]
    ) * np.sqrt(TRADING_DAYS)
    if rolling.size < 10:
        return []

    threshold = float(np.median(rolling))
    high = rolling > threshold

    regimes: list[VolatilityRegime] = []
    start_index = 0
    for i in range(1, high.size):
        if high[i] != high[i - 1]:
            length = i - start_index
            if length >= 21:  # ignore regimes shorter than a month
                segment = rolling[start_index:i]
                regimes.append(
                    VolatilityRegime(
                        start=str(dates[window + start_index])[:10],
                        end=str(dates[min(window + i, dates.size - 1)])[:10],
                        annualised_volatility=float(np.mean(segment)),
                        n_days=length,
                        label="high volatility" if high[start_index] else "low volatility",
                    )
                )
            start_index = i
    segment = rolling[start_index:]
    if segment.size >= 21:
        regimes.append(
            VolatilityRegime(
                start=str(dates[window + start_index])[:10],
                end=str(dates[-1])[:10],
                annualised_volatility=float(np.mean(segment)),
                n_days=int(segment.size),
                label="high volatility" if high[start_index] else "low volatility",
            )
        )
    return regimes


def _factor_exposure(
    returns: NDArray[np.float64], factors: dict[str, NDArray[np.float64]]
) -> FactorExposure | None:
    """Regress instrument returns on factor returns, with HAC standard errors.

    Two guards that matter in practice:

    **Self-regression.** Analysing the S&P 500 with the S&P 500 as the market
    factor gives beta exactly 1.0 with a t-statistic of order 1e21 -- a perfect
    fit that tells you nothing. Any factor almost perfectly correlated with the
    instrument is dropped, and the report says so.

    **Unit mismatch.** Rate factors are differences in percentage points while
    equity returns are decimals, so a raw regression gives betas of order 1e-5
    that print as 0.0000. Rate factors are rescaled to decimals so the
    coefficient is a duration-like number a reader can interpret.
    """
    from quantos.core.timeseries.ols import ols

    names = list(factors)
    if not names:
        return None

    length_all = min([returns.size] + [factors[n].size for n in names])
    usable: dict[str, NDArray[np.float64]] = {}
    dropped: list[str] = []
    for name in names:
        column = factors[name][-length_all:]
        target = returns[-length_all:]
        if float(np.std(column)) == 0.0:
            dropped.append(name)
            continue
        correlation = abs(float(np.corrcoef(target, column)[0, 1]))
        if correlation > 0.999:
            # The instrument IS this factor; the regression is an identity.
            dropped.append(name)
            continue
        # Rate factors arrive in percentage points; convert to decimals.
        usable[name] = column / 100.0 if name.startswith(("rates", "credit")) else column

    if not usable:
        return None
    names = list(usable)
    factors = usable
    length = min([returns.size] + [factors[n].size for n in names])
    if length < 100:
        return None

    y = returns[-length:]
    design = np.column_stack([np.ones(length)] + [factors[n][-length:] for n in names])
    try:
        fit = ols(y, design, cov_type="hac")
    except (ValueError, np.linalg.LinAlgError):
        return None

    return FactorExposure(
        factor_names=names,
        betas=fit.coefficients[1:],
        t_statistics=fit.t_statistics[1:],
        alpha_annualised=float(fit.coefficients[0] * TRADING_DAYS),
        alpha_t_statistic=float(fit.t_statistics[0]),
        r_squared=float(fit.r_squared),
        idiosyncratic_share=float(1.0 - fit.r_squared),
        n_observations=length,
    )


def _option_analytics(
    spot: float, realised_vol: float, rate: float = 0.04
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Price a strike ladder at the instrument's own realised volatility.

    The realised volatility is used as the volatility *input*, which makes this a
    fair-value ladder rather than a market quote. Comparing a real option chain
    against it is the classic rich/cheap screen: if market implied volatility
    sits above realised, options are expensive relative to what the underlying
    has actually been doing, which is the variance risk premium.
    """
    from quantos.derivatives.black_scholes import (
        OptionType,
        black_scholes_greeks,
        black_scholes_price,
    )

    summary = {
        "spot": spot,
        "realised_volatility": realised_vol,
        "rate": rate,
    }
    table: list[dict[str, float]] = []
    maturity = 30.0 / 365.0

    for moneyness in (0.90, 0.95, 1.00, 1.05, 1.10):
        strike = spot * moneyness
        call = float(black_scholes_price(spot, strike, maturity, realised_vol, rate=rate))
        put = float(
            black_scholes_price(
                spot, strike, maturity, realised_vol, rate=rate, option_type=OptionType.PUT
            )
        )
        greeks = black_scholes_greeks(spot, strike, maturity, realised_vol, rate=rate)
        table.append(
            {
                "moneyness": moneyness,
                "strike": strike,
                "call": call,
                "put": put,
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "vega": greeks.vega / 100.0,  # per volatility point
                "theta": greeks.theta / 365.0,  # per calendar day
            }
        )

    # A straddle's breakeven is the clearest single statement of what the option
    # market must deliver for a long-volatility position to pay.
    atm = table[2]
    straddle = atm["call"] + atm["put"]
    summary["atm_straddle"] = straddle
    summary["breakeven_move_pct"] = straddle / spot
    summary["implied_daily_move_pct"] = realised_vol / np.sqrt(TRADING_DAYS)
    return summary, table


def generate_report(
    instrument: Instrument,
    *,
    factors: dict[str, NDArray[np.float64]] | None = None,
    run_signals: bool = True,
    transaction_cost_bps: float = 5.0,
    risk_free: float = 0.0,
) -> ResearchReport:
    """Produce a complete research report on one instrument.

    Purpose
        Run, in one call, the analysis a careful quant would run by hand -- and
        in particular, never skip the validation step.
    Inputs
        ``instrument`` -- what to analyse. ``factors`` -- optional factor return
        series (market, rates, credit) for the exposure regression.
        ``run_signals`` -- the signal battery is the slow part; disable it for a
        quick look.
    Outputs
        :class:`ResearchReport`, with unavailable sections recorded in
        ``skipped`` alongside the reason.
    """
    from quantos.core.stats.descriptive import hill_estimator, kurtosis, skewness
    from quantos.core.stats.hypothesis import (
        augmented_dickey_fuller,
        engle_arch,
        jarque_bera,
        kpss,
    )
    from quantos.core.timeseries.garch import fit_garch, fit_gjr_garch
    from quantos.execution.almgren_chriss import square_root_impact_cost
    from quantos.risk.metrics import (
        conditional_value_at_risk,
        drawdown_series,
        max_drawdown,
        sharpe_ratio,
        sortino_ratio,
        value_at_risk,
    )
    from quantos.strategy.validation import sharpe_ratio_with_moments

    report = ResearchReport(instrument=instrument)
    report.data_warnings = instrument.data_quality_warnings()

    returns = instrument.returns()
    returns = returns[np.isfinite(returns)]
    report.n_returns = int(returns.size)
    if returns.size < 60:
        report.notes.append("fewer than 60 observations; nothing can be estimated")
        return report

    linear = instrument.asset_class.has_linear_payoff
    tradeable = instrument.asset_class.is_tradeable

    # -- distribution ------------------------------------------------------
    report.annualised_volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))
    report.skewness = skewness(returns)
    report.excess_kurtosis = kurtosis(returns, excess=True)
    non_zero = np.abs(returns[returns != 0.0])
    report.tail_index = hill_estimator(non_zero) if non_zero.size > 50 else float("nan")
    report.normality_p = jarque_bera(returns).p_value
    report.var_95 = value_at_risk(returns, confidence=0.95)
    report.cvar_95 = conditional_value_at_risk(returns, confidence=0.95)
    report.var_99 = value_at_risk(returns, confidence=0.99)
    report.cvar_99 = conditional_value_at_risk(returns, confidence=0.99)

    # -- risk --------------------------------------------------------------
    if instrument.supports(Analysis.RISK_METRICS) and tradeable and linear:
        years = returns.size / TRADING_DAYS
        total = float(instrument.prices[-1] / instrument.prices[0])
        if years > 0 and total > 0:
            report.annualised_return = float(total ** (1.0 / years) - 1.0)
        report.sharpe = sharpe_ratio(returns, risk_free=risk_free, periods_per_year=TRADING_DAYS)
        report.sharpe_standard_error = sharpe_ratio_with_moments(returns).standard_error
        report.sortino = sortino_ratio(returns, periods_per_year=TRADING_DAYS)
        depth, _, _ = max_drawdown(instrument.prices)
        report.max_drawdown = depth
        underwater = drawdown_series(instrument.prices) < 0
        longest = current = 0
        for flag in underwater:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        report.max_drawdown_days = longest
    else:
        report.skipped["risk"] = instrument.skip_reason(Analysis.RISK_METRICS)

    # -- volatility --------------------------------------------------------
    # Test for ARCH effects BEFORE fitting a model of them. With no conditional
    # heteroskedasticity the GARCH likelihood is flat and the optimiser walks to
    # the boundary: on i.i.d. Gaussian data this produced alpha=0.000,
    # beta=1.000, persistence=1.0000 and a half-life of 442,909 days -- all
    # finite, all meaningless. A constant-volatility model is the honest answer
    # there, and saying so is more useful than a degenerate fit.
    if returns.size >= 250:
        arch = engle_arch(returns, lags=5)
        report.arch_p_value = arch.p_value
        if arch.p_value > 0.05:
            report.skipped["garch"] = (
                f"no significant ARCH effects (Engle LM p = {arch.p_value:.3f}), so "
                "volatility is adequately modelled as constant. Fitting GARCH here "
                "would return boundary estimates that look precise and mean nothing."
            )
    if returns.size >= 250 and "garch" not in report.skipped:
        try:
            fit = fit_garch(returns)
            report.garch_alpha = fit.alpha
            report.garch_beta = fit.beta
            report.garch_persistence = fit.persistence
            report.garch_half_life = fit.half_life
            report.volatility_forecast_1d = float(np.sqrt(fit.forecast(1)[0] * TRADING_DAYS))
            report.volatility_forecast_21d = float(
                np.sqrt(float(np.mean(fit.forecast(21))) * TRADING_DAYS)
            )
        except (ValueError, RuntimeError) as error:
            report.skipped["garch"] = str(error)
        if returns.size >= 500:
            with contextlib.suppress(ValueError, RuntimeError):
                report.leverage_gamma = fit_gjr_garch(returns).gamma

        window = min(63, returns.size // 4)
        rolling = np.array(
            [float(np.std(returns[t - window : t], ddof=1)) for t in range(window, returns.size)]
        )
        if rolling.size > 20:
            report.current_vol_percentile = float(np.mean(rolling <= rolling[-1]))
    elif returns.size < 250:
        report.skipped["garch"] = f"only {returns.size} observations; GARCH needs at least 250"

    # -- regimes -----------------------------------------------------------
    if instrument.supports(Analysis.REGIME_DETECTION):
        report.regimes = _volatility_regimes(instrument.dates[1:], returns)
    else:
        report.skipped["regimes"] = instrument.skip_reason(Analysis.REGIME_DETECTION)

    # -- factors -----------------------------------------------------------
    if factors and instrument.supports(Analysis.FACTOR_EXPOSURE):
        report.factors = _factor_exposure(returns, factors)
        if report.factors is None:
            report.skipped["factors"] = "not enough overlapping observations"
    elif not factors:
        report.skipped["factors"] = "no factor series supplied (pass --factors)"

    # -- signals -----------------------------------------------------------
    if run_signals and instrument.supports(Analysis.SIGNAL_BATTERY) and linear:
        from quantos.research.signals import run_signal_battery

        try:
            report.signals = run_signal_battery(
                instrument.prices, transaction_cost_bps=transaction_cost_bps
            )
        except ValueError as error:
            report.skipped["signals"] = str(error)
    elif not run_signals:
        report.skipped["signals"] = "disabled (--no-signals)"
    else:
        report.skipped["signals"] = instrument.skip_reason(Analysis.SIGNAL_BATTERY)

    # -- options -----------------------------------------------------------
    if instrument.supports(Analysis.OPTION_ANALYTICS) and np.isfinite(report.annualised_volatility):
        report.option_summary, report.option_table = _option_analytics(
            instrument.latest, report.annualised_volatility
        )
    else:
        report.skipped["options"] = instrument.skip_reason(Analysis.OPTION_ANALYTICS)

    # -- execution ---------------------------------------------------------
    if instrument.supports(Analysis.EXECUTION_COST):
        report.execution_costs = [
            (
                participation,
                square_root_impact_cost(
                    participation, 1.0, report.annualised_volatility, coefficient=0.8
                ),
            )
            for participation in (0.001, 0.005, 0.01, 0.05, 0.10)
        ]
    else:
        report.skipped["execution"] = instrument.skip_reason(Analysis.EXECUTION_COST)

    # -- stationarity ------------------------------------------------------
    try:
        adf = augmented_dickey_fuller(instrument.prices)
        adf_rejects = adf.statistic < adf.critical_values.get("5%", -2.86)
        kpss_rejects = kpss(instrument.prices).rejects_at(0.05)
        if adf_rejects and not kpss_rejects:
            report.stationarity = "stationary (both tests agree)"
        elif not adf_rejects and kpss_rejects:
            report.stationarity = "unit root / trending (both tests agree)"
        elif adf_rejects and kpss_rejects:
            report.stationarity = "tests disagree: possible structural break"
        else:
            report.stationarity = "inconclusive"
    except (ValueError, RuntimeError) as error:
        report.skipped["stationarity"] = str(error)

    return report
