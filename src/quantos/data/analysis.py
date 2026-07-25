"""Run the QuantOS toolkit over real market and macroeconomic series.

This is where the rest of the platform meets actual data. Everything here is
composition -- the estimators, risk measures and tests are the same ones the
simulated market uses, applied to series pulled from FRED or loaded from CSV.
That reuse is the point: an estimator validated against simulated ground truth
and then run unchanged on real data is a much stronger claim than either half
alone.

What it computes, and why each one earns its place
--------------------------------------------------
* **Risk metrics** — annualised return and volatility, Sharpe with the
  autocorrelation adjustment, drawdown, VaR and CVaR at two levels, skew and
  kurtosis. The CVaR/VaR gap is the number to look at: it is how much worse the
  tail is than the threshold suggests.
* **Distributional tests** — Jarque-Bera and the Hill tail index, to say *how*
  non-normal the series is rather than merely that it is.
* **GARCH(1,1)** — persistence and half-life, plus a one-step volatility
  forecast. Fitted with the same MLE the simulated data uses.
* **Stationarity** — ADF and KPSS together, since they test complementary nulls
  and disagreement is itself informative.
* **Cross-series** — correlation of transformed series, and a cointegration test
  on any pair of price levels, which is the actual basis for pairs trading.

Every number is accompanied by the caveat that applies to it. A Sharpe ratio
printed without its sampling error, or a cointegration verdict without its
critical value, is decoration rather than analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["CrossSectionReport", "SeriesReport", "analyse_cross_section", "analyse_series"]

TRADING_DAYS = 252.0


@dataclass
class SeriesReport:
    """Everything QuantOS can say about one real series."""

    key: str
    name: str
    kind: str
    n_observations: int
    start: str
    end: str
    latest: float

    annualised_return: float = float("nan")
    annualised_volatility: float = float("nan")
    sharpe: float = float("nan")
    sharpe_autocorr_adjusted: float = float("nan")
    max_drawdown: float = float("nan")
    var_95: float = float("nan")
    cvar_95: float = float("nan")
    var_99: float = float("nan")
    cvar_99: float = float("nan")
    skewness: float = float("nan")
    excess_kurtosis: float = float("nan")
    tail_index: float = float("nan")
    jarque_bera_p: float = float("nan")

    garch_alpha: float = float("nan")
    garch_beta: float = float("nan")
    garch_persistence: float = float("nan")
    garch_half_life: float = float("nan")
    volatility_forecast: float = float("nan")

    adf_statistic: float = float("nan")
    adf_rejects_unit_root: bool = False
    kpss_rejects_stationarity: bool = False

    notes: list[str] = field(default_factory=list)

    @property
    def stationarity_verdict(self) -> str:
        """Read ADF and KPSS jointly, as they are meant to be read."""
        if self.adf_rejects_unit_root and not self.kpss_rejects_stationarity:
            return "stationary (both tests agree)"
        if not self.adf_rejects_unit_root and self.kpss_rejects_stationarity:
            return "unit root / integrated (both tests agree)"
        if self.adf_rejects_unit_root and self.kpss_rejects_stationarity:
            return "tests disagree -- possible fractional integration or a break"
        return "inconclusive -- neither test rejects; likely too little data"

    @property
    def tail_severity(self) -> str:
        """How much worse the tail is than the VaR threshold implies."""
        if not np.isfinite(self.var_99) or self.var_99 <= 0:
            return "n/a"
        ratio = self.cvar_99 / self.var_99
        return f"CVaR/VaR at 99% = {ratio:.2f}"


def analyse_series(
    key: str,
    name: str,
    kind: str,
    dates: NDArray[np.datetime64],
    values: NDArray[np.float64],
    *,
    periods_per_year: float = TRADING_DAYS,
) -> SeriesReport:
    """Run the full single-series battery.

    ``kind`` is ``"level"``, ``"index"`` or ``"rate"``; it decides whether the
    series is transformed by log returns or first differences. Return and Sharpe
    statistics are only reported for tradeable levels, because the "return" of a
    yield series is not a return and printing a Sharpe ratio for it would be
    meaningless.
    """
    from quantos.core.stats.descriptive import hill_estimator, kurtosis, skewness
    from quantos.core.stats.hypothesis import augmented_dickey_fuller, jarque_bera, kpss
    from quantos.core.timeseries.garch import fit_garch
    from quantos.risk.metrics import (
        conditional_value_at_risk,
        max_drawdown,
        sharpe_ratio,
        value_at_risk,
    )

    values = np.asarray(values, dtype=float)
    report = SeriesReport(
        key=key,
        name=name,
        kind=kind,
        n_observations=int(values.size),
        start=str(dates[0])[:10] if dates.size else "",
        end=str(dates[-1])[:10] if dates.size else "",
        latest=float(values[-1]) if values.size else float("nan"),
    )
    if values.size < 60:
        report.notes.append("fewer than 60 observations; most statistics suppressed")
        return report

    tradeable = kind == "level"
    changes = np.diff(values) if kind == "rate" else None
    if changes is None:
        if np.any(values <= 0):
            report.notes.append("non-positive values present; treated as differences")
            changes = np.diff(values)
            tradeable = False
        else:
            changes = np.diff(np.log(values))

    changes = changes[np.isfinite(changes)]
    if changes.size < 60:
        report.notes.append("too few usable changes after cleaning")
        return report

    # -- distribution ------------------------------------------------------
    report.annualised_volatility = float(np.std(changes, ddof=1) * np.sqrt(periods_per_year))
    report.skewness = skewness(changes)
    report.excess_kurtosis = kurtosis(changes, excess=True)
    report.tail_index = hill_estimator(np.abs(changes[changes != 0.0]))
    report.jarque_bera_p = jarque_bera(changes).p_value
    report.var_95 = value_at_risk(changes, confidence=0.95)
    report.cvar_95 = conditional_value_at_risk(changes, confidence=0.95)
    report.var_99 = value_at_risk(changes, confidence=0.99)
    report.cvar_99 = conditional_value_at_risk(changes, confidence=0.99)

    if tradeable:
        years = changes.size / periods_per_year
        total = float(values[-1] / values[0])
        report.annualised_return = (
            float(total ** (1.0 / years) - 1.0) if years > 0 else float("nan")
        )
        report.sharpe = sharpe_ratio(changes, periods_per_year=periods_per_year)
        report.sharpe_autocorr_adjusted = sharpe_ratio(
            changes, periods_per_year=periods_per_year, adjust_autocorrelation=True
        )
        report.max_drawdown = max_drawdown(values)[0]
    else:
        report.notes.append(
            f"kind={kind}: return and Sharpe statistics omitted, since changes in a "
            "rate or non-tradeable index are not investable returns"
        )

    # -- volatility dynamics ----------------------------------------------
    if changes.size >= 250:
        try:
            fit = fit_garch(changes)
            report.garch_alpha = fit.alpha
            report.garch_beta = fit.beta
            report.garch_persistence = fit.persistence
            report.garch_half_life = fit.half_life
            report.volatility_forecast = float(np.sqrt(fit.forecast(1)[0] * periods_per_year))
        except (ValueError, RuntimeError) as error:  # pragma: no cover - data dependent
            report.notes.append(f"GARCH fit failed: {error}")

    # -- stationarity ------------------------------------------------------
    try:
        adf = augmented_dickey_fuller(values)
        report.adf_statistic = adf.statistic
        report.adf_rejects_unit_root = adf.statistic < adf.critical_values.get("5%", -2.86)
        report.kpss_rejects_stationarity = kpss(values).rejects_at(0.05)
    except (ValueError, RuntimeError) as error:  # pragma: no cover - data dependent
        report.notes.append(f"stationarity tests failed: {error}")

    return report


@dataclass
class CrossSectionReport:
    """Relationships between several real series."""

    names: list[str]
    correlation: NDArray[np.float64]
    n_common_dates: int
    start: str
    end: str
    #: (a, b) -> (is_cointegrated, statistic, hedge_ratio) for level pairs.
    cointegration: dict[tuple[str, str], tuple[bool, float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def most_correlated(self) -> tuple[str, str, float]:
        """The strongest off-diagonal pair."""
        best = ("", "", 0.0)
        n = len(self.names)
        for i in range(n):
            for j in range(i + 1, n):
                value = float(self.correlation[i, j])
                if abs(value) > abs(best[2]):
                    best = (self.names[i], self.names[j], value)
        return best


def analyse_cross_section(
    transformed: dict[str, NDArray[np.float64]],
    levels: dict[str, NDArray[np.float64]] | None = None,
    *,
    dates: NDArray[np.datetime64] | None = None,
) -> CrossSectionReport:
    """Correlation across series, plus cointegration tests on level pairs.

    ``transformed`` holds the analysis-ready series (returns or differences);
    ``levels`` holds the raw price levels for the subset where cointegration is
    meaningful. Testing cointegration on *returns* is a category error -- returns
    are already stationary, so the test always rejects the unit root and tells
    you nothing.
    """
    from quantos.core.timeseries.cointegration import engle_granger

    names = list(transformed)
    if not names:
        return CrossSectionReport(
            names=[], correlation=np.zeros((0, 0)), n_common_dates=0, start="", end=""
        )

    matrix = np.vstack([transformed[n] for n in names])
    correlation = np.corrcoef(matrix) if len(names) > 1 else np.ones((1, 1))

    report = CrossSectionReport(
        names=names,
        correlation=np.atleast_2d(correlation),
        n_common_dates=int(matrix.shape[1]),
        start=str(dates[0])[:10] if dates is not None and dates.size else "",
        end=str(dates[-1])[:10] if dates is not None and dates.size else "",
    )

    if levels:
        level_names = list(levels)
        for i in range(len(level_names)):
            for j in range(i + 1, len(level_names)):
                a, b = level_names[i], level_names[j]
                try:
                    result = engle_granger(levels[a], levels[b])
                    report.cointegration[(a, b)] = (
                        result.is_cointegrated,
                        result.statistic,
                        result.beta,
                    )
                except (ValueError, RuntimeError):  # pragma: no cover - data dependent
                    continue
        if not report.cointegration:
            report.notes.append("no level pairs were testable for cointegration")

    return report
