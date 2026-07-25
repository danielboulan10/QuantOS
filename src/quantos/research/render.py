"""Render a :class:`~quantos.research.report.ResearchReport` as text or Markdown.

Presentation is kept separate from computation so the same report can be printed
to a terminal, written to a Markdown file for the research journal, or consumed
programmatically without any of the three constraining the others.

One editorial rule runs through this module: **every number is printed with the
thing that qualifies it.** A Sharpe ratio appears with its standard error, a
signal with its deflated p-value, a volatility forecast with the persistence that
determines how fast it decays. A number without its qualifier invites the reader
to over-trust it, and a research tool that does that is worse than no tool.
"""

from __future__ import annotations

import numpy as np

from quantos.research.report import ResearchReport

__all__ = ["render_markdown", "render_text"]


def _rule(title: str, char: str = "=") -> str:
    return f"\n{title}\n{char * len(title)}"


def _fmt(value: float, spec: str = ".2%") -> str:
    """Format, or a clear dash when the quantity does not exist.

    The dash respects any width in ``spec``, so a missing value does not shift
    every subsequent column and corrupt the table.
    """
    if np.isfinite(value):
        return format(value, spec)
    width = "".join(c for c in spec.split(".", maxsplit=1)[0] if c.isdigit())
    align = ">" if ">" in spec else ""
    return format("--", f"{align}{width}") if width else "--"


def _t_stat(value: float) -> str:
    """Format a t-statistic, capping absurd magnitudes.

    A t-statistic of 1e21 means the regression is an identity, not that the
    result is astronomically significant. Printing the raw number also overflows
    the column and corrupts the table alignment.
    """
    if not np.isfinite(value):
        return "--"
    if abs(value) > 1e4:
        return ">1e4"
    return f"{value:.2f}"


def render_text(report: ResearchReport) -> str:
    """Render for a terminal."""
    inst = report.instrument
    out: list[str] = []
    add = out.append

    add(_rule(f"RESEARCH REPORT: {inst.display_name} ({inst.symbol})"))
    add(f"asset class      {inst.asset_class.value}")
    add(f"observations     {len(inst):,}  from {inst.start} to {inst.end}")
    add(f"latest price     {inst.latest:,.4f} {inst.currency}")
    if inst.source:
        add(f"source           {inst.source}")
    add(f"generated        {report.generated}")

    if report.data_warnings:
        add(_rule("DATA QUALITY", "-"))
        for warning in report.data_warnings:
            add(f"  ! {warning}")

    # -- distribution ------------------------------------------------------
    add(_rule("RETURN DISTRIBUTION", "-"))
    add(f"  annualised volatility    {_fmt(report.annualised_volatility)}")
    add(f"  skewness                 {_fmt(report.skewness, '.3f')}")
    add(f"  excess kurtosis          {_fmt(report.excess_kurtosis, '.2f')}")
    add(f"  Hill tail index          {_fmt(report.tail_index, '.2f')}   (equities ~3)")
    add(f"  Jarque-Bera p-value      {_fmt(report.normality_p, '.3g')}")
    if np.isfinite(report.excess_kurtosis) and report.excess_kurtosis > 1:
        add("    -> returns are materially fat-tailed; a Gaussian VaR will understate losses")

    # -- risk --------------------------------------------------------------
    if "risk" in report.skipped:
        add(_rule("RISK", "-"))
        add(f"  skipped: {report.skipped['risk']}")
    else:
        add(_rule("RISK", "-"))
        add(f"  annualised return        {_fmt(report.annualised_return)}")
        add(f"  Sharpe ratio             {_fmt(report.sharpe, '.3f')}")
        if np.isfinite(report.sharpe_standard_error):
            annual_se = report.sharpe_standard_error * np.sqrt(252.0)
            add(f"    standard error         {annual_se:.3f}")
            verdict = (
                "distinguishable from zero"
                if report.sharpe_is_significant
                else "NOT distinguishable from zero"
            )
            add(f"    -> {verdict}")
        add(f"  Sortino ratio            {_fmt(report.sortino, '.3f')}")
        add(f"  maximum drawdown         {_fmt(report.max_drawdown)}")
        add(f"    longest underwater     {report.max_drawdown_days:,} trading days")
        add(f"  VaR 95% / CVaR 95%       {_fmt(report.var_95)} / {_fmt(report.cvar_95)}")
        add(f"  VaR 99% / CVaR 99%       {_fmt(report.var_99)} / {_fmt(report.cvar_99)}")
        if np.isfinite(report.cvar_99) and report.var_99 > 0:
            add(
                f"    CVaR/VaR at 99%        {report.cvar_99 / report.var_99:.2f}  "
                "(how much worse the tail is than the threshold)"
            )

    # -- volatility --------------------------------------------------------
    add(_rule("VOLATILITY DYNAMICS", "-"))
    if np.isfinite(report.arch_p_value):
        add(f"  Engle ARCH-LM p-value    {_fmt(report.arch_p_value, '.3g')}")
    if "garch" in report.skipped:
        add(f"  {report.skipped['garch']}")
    else:
        add(
            f"  GARCH(1,1)               alpha {_fmt(report.garch_alpha, '.3f')}  "
            f"beta {_fmt(report.garch_beta, '.3f')}"
        )
        add(f"  persistence              {_fmt(report.garch_persistence, '.4f')}")
        add(f"  shock half-life          {_fmt(report.garch_half_life, '.1f')} days")
        add(f"  forecast, 1 day          {_fmt(report.volatility_forecast_1d)} annualised")
        add(f"  forecast, 21 days        {_fmt(report.volatility_forecast_21d)} annualised")
        if np.isfinite(report.current_vol_percentile):
            add(
                f"  current vol percentile   {report.current_vol_percentile:.0%} of its own history"
            )
        if np.isfinite(report.leverage_gamma) and report.leverage_gamma > 0.01:
            add(
                f"  leverage effect (gamma)  {report.leverage_gamma:.3f}  "
                "-> down moves raise volatility more than up moves"
            )
        if np.isfinite(report.garch_persistence) and report.garch_persistence > 0.99:
            add(
                "    ! persistence above 0.99: volatility is nearly integrated, so "
                "long-horizon forecasts are unreliable"
            )

    # -- regimes -----------------------------------------------------------
    if report.regimes:
        add(_rule("VOLATILITY REGIMES", "-"))
        add(f"  {'period':<26}{'days':>7}{'annualised vol':>17}  regime")
        for regime in report.regimes[-8:]:
            add(
                f"  {regime.start} to {regime.end}{regime.n_days:>7,}"
                f"{regime.annualised_volatility:>17.2%}  {regime.label}"
            )

    # -- factors -----------------------------------------------------------
    add(_rule("FACTOR EXPOSURE", "-"))
    if report.factors is None:
        add(f"  skipped: {report.skipped.get('factors', 'unavailable')}")
    else:
        f = report.factors
        add(f"  {'factor':<20}{'beta':>12}{'t-stat':>10}   (HAC standard errors)")
        for name, beta, t in zip(f.factor_names, f.betas, f.t_statistics, strict=True):
            marker = " *" if abs(float(t)) > 1.96 else ""
            add(f"  {name:<20}{float(beta):>12.4f}{_t_stat(float(t)):>10}{marker}")
        add(
            f"  {'alpha (annual)':<20}{f.alpha_annualised:>12.2%}{_t_stat(f.alpha_t_statistic):>10}"
        )
        add(f"  R-squared            {f.r_squared:>10.4f}")
        add(f"  idiosyncratic share  {f.idiosyncratic_share:>10.2%}")
        if abs(f.alpha_t_statistic) < 1.96:
            add("    -> alpha is not distinguishable from zero")

    # -- signals -----------------------------------------------------------
    add(_rule("SIGNAL BATTERY", "-"))
    if report.signals is None:
        add(f"  skipped: {report.skipped.get('signals', 'unavailable')}")
    else:
        battery = report.signals
        add(
            f"  {battery.n_signals} pre-registered signals, {battery.n_observations:,} "
            "observations, costs charged on every position change.\n"
        )
        add(
            f"  {'signal':<22}{'IS Sharpe':>11}{'deflated p':>12}"
            f"{'OOS Sharpe':>12}{'turnover':>10}  verdict"
        )
        for result in battery.sorted_by_evidence():
            add(
                f"  {result.name:<22}{_fmt(result.in_sample_sharpe, '>11.3f')}"
                f"{_fmt(result.deflated_p_value, '>12.3f')}"
                f"{_fmt(result.out_of_sample_sharpe, '>12.3f')}"
                f"{_fmt(result.turnover, '>10.1f')}  {result.verdict}"
            )
        add("")
        add(f"  Hansen SPA over the battery   p = {_fmt(battery.spa_p_value, '.4f')}")
        if battery.any_significant:
            add("  -> at least one signal survives every correction. Inspect it, then")
            add("     test it on a different instrument and a different period.")
        else:
            add("  -> nothing survives correction for multiple testing.")
        for note in battery.notes:
            add(f"  note: {note}")

    # -- options -----------------------------------------------------------
    if report.option_table:
        summary = report.option_summary
        add(_rule("OPTION ANALYTICS (30-day, priced at realised volatility)", "-"))
        add(
            f"  spot {summary['spot']:,.2f}   realised vol "
            f"{summary['realised_volatility']:.2%}   rate {summary['rate']:.2%}"
        )
        add(
            f"  ATM straddle {summary['atm_straddle']:,.2f}  "
            f"-> breakeven move {summary['breakeven_move_pct']:.2%} over 30 days"
        )
        add(f"  implied daily move {summary['implied_daily_move_pct']:.2%}\n")
        add(
            f"  {'moneyness':>10}{'strike':>11}{'call':>10}{'put':>10}"
            f"{'delta':>9}{'gamma':>10}{'vega/pt':>10}{'theta/day':>11}"
        )
        for row in report.option_table:
            add(
                f"  {row['moneyness']:>10.0%}{row['strike']:>11.2f}{row['call']:>10.3f}"
                f"{row['put']:>10.3f}{row['delta']:>9.3f}{row['gamma']:>10.5f}"
                f"{row['vega']:>10.4f}{row['theta']:>11.4f}"
            )
        add(
            "\n  These are fair values at the instrument's OWN realised volatility,"
            "\n  not market quotes. Compare a real chain against them: implied above"
            "\n  realised is the variance risk premium, and it is usually positive."
        )

    # -- execution ---------------------------------------------------------
    if report.execution_costs:
        add(_rule("EXECUTION COST (square-root law)", "-"))
        add(f"  {'participation':>14}{'impact (bps)':>15}")
        for participation, cost in report.execution_costs:
            add(f"  {participation:>14.1%}{cost * 1e4:>15.1f}")
        add(
            "\n  Impact depends on size relative to daily volume, not on how long"
            "\n  you take. Trading the same order more slowly does not reduce it."
        )

    # -- summary -----------------------------------------------------------
    add(_rule("WHAT THE DATA SUPPORTS"))
    claims: list[str] = []
    if np.isfinite(report.annualised_volatility):
        claims.append(
            f"Volatility is {report.annualised_volatility:.1%} annualised"
            + (
                f", currently at the {report.current_vol_percentile:.0%} percentile "
                "of its own history"
                if np.isfinite(report.current_vol_percentile)
                else ""
            )
            + "."
        )
    if np.isfinite(report.excess_kurtosis) and report.excess_kurtosis > 1:
        claims.append(
            f"Returns are fat-tailed (excess kurtosis {report.excess_kurtosis:.1f}), so "
            "size positions on CVaR rather than on standard deviation."
        )
    if np.isfinite(report.garch_half_life):
        claims.append(
            f"Volatility shocks decay with a {report.garch_half_life:.0f}-day half-life, "
            "so a volatility spike is informative about roughly the next month."
        )
    if report.sharpe_is_significant:
        claims.append(
            f"The historical Sharpe ratio of {report.sharpe:.2f} IS statistically "
            "distinguishable from zero over this sample."
        )
    elif np.isfinite(report.sharpe):
        claims.append(
            f"The historical Sharpe ratio of {report.sharpe:.2f} is NOT distinguishable "
            "from zero over this sample."
        )
    if report.factors and report.factors.significant_factors:
        claims.append(
            "Returns load significantly on "
            + ", ".join(report.factors.significant_factors)
            + f", leaving {report.factors.idiosyncratic_share:.0%} idiosyncratic."
        )
    if report.signals is not None and not report.signals.any_significant:
        claims.append(
            f"None of {report.signals.n_signals} standard signals predicts this "
            "instrument after correcting for multiple testing."
        )
    for claim in claims:
        add(f"  - {claim}")

    add("\n  NOT established by this report:")
    add("  - anything about the future; every number above is descriptive")
    add("  - fundamental value, earnings quality, or anything not in the price")
    add("  - liquidity, borrow availability, or capacity")
    if report.skipped:
        add("\n  Sections skipped:")
        for name, reason in sorted(report.skipped.items()):
            add(f"  - {name}: {reason}")
    add("\n  This is not investment advice.")
    return "\n".join(out)


def render_markdown(report: ResearchReport) -> str:
    """Render for the research journal: a Markdown document."""
    inst = report.instrument
    out: list[str] = [
        f"# Research report: {inst.display_name} ({inst.symbol})",
        "",
        f"*Generated {report.generated} by [QuantOS](https://github.com/danielboulan10/QuantOS).*",
        "",
        "| | |",
        "|---|---|",
        f"| Asset class | {inst.asset_class.value} |",
        f"| Observations | {len(inst):,} |",
        f"| Period | {inst.start} to {inst.end} |",
        f"| Latest | {inst.latest:,.4f} {inst.currency} |",
        f"| Source | {inst.source or 'n/a'} |",
        "",
    ]
    if report.data_warnings:
        out += ["## Data quality", ""]
        out += [f"- {w}" for w in report.data_warnings]
        out += [""]

    out += [
        "## Risk and distribution",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Annualised volatility | {_fmt(report.annualised_volatility)} |",
        f"| Annualised return | {_fmt(report.annualised_return)} |",
        f"| Sharpe ratio | {_fmt(report.sharpe, '.3f')} |",
        f"| Maximum drawdown | {_fmt(report.max_drawdown)} |",
        f"| VaR 99% / CVaR 99% | {_fmt(report.var_99)} / {_fmt(report.cvar_99)} |",
        f"| Skewness | {_fmt(report.skewness, '.3f')} |",
        f"| Excess kurtosis | {_fmt(report.excess_kurtosis, '.2f')} |",
        f"| Hill tail index | {_fmt(report.tail_index, '.2f')} |",
        "",
    ]

    if np.isfinite(report.garch_persistence):
        out += [
            "## Volatility dynamics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| GARCH persistence | {_fmt(report.garch_persistence, '.4f')} |",
            f"| Shock half-life | {_fmt(report.garch_half_life, '.1f')} days |",
            f"| 1-day forecast | {_fmt(report.volatility_forecast_1d)} |",
            f"| 21-day forecast | {_fmt(report.volatility_forecast_21d)} |",
            "",
        ]

    if report.signals is not None:
        battery = report.signals
        out += [
            "## Signal battery",
            "",
            f"{battery.n_signals} pre-registered signals. The battery size is fixed "
            "before any data is seen, which is what makes the deflated Sharpe "
            "correction valid.",
            "",
            "| Signal | IS Sharpe | Deflated p | OOS Sharpe | Verdict |",
            "|---|---|---|---|---|",
        ]
        for r in battery.sorted_by_evidence():
            out.append(
                f"| `{r.name}` | {_fmt(r.in_sample_sharpe, '.3f')} | "
                f"{_fmt(r.deflated_p_value, '.3f')} | "
                f"{_fmt(r.out_of_sample_sharpe, '.3f')} | {r.verdict} |"
            )
        out += ["", f"Hansen SPA over the battery: **p = {_fmt(battery.spa_p_value, '.4f')}**", ""]

    out += [
        "## Limitations",
        "",
        "- Every number is descriptive of the past; none is a forecast.",
        "- Nothing here reflects fundamentals, liquidity, borrow or capacity.",
        "- This is not investment advice.",
        "",
    ]
    return "\n".join(out)
