r"""A battery of standard signals, each reported with the evidence against it.

What makes this different from a screener
-----------------------------------------
Any tool can compute a 12-month momentum score and print it. The difficulty is
not computing signals; it is knowing which of them survives the fact that you
computed *many*.

So this module never reports a signal's backtested Sharpe ratio on its own. Every
signal returns:

* its in-sample Sharpe ratio, and
* the **deflated** Sharpe ratio, adjusted for the number of signals in the
  battery, and
* a purged, embargoed out-of-sample Sharpe, and
* whether it survives a Hansen SPA test run over the whole battery jointly.

The battery is deliberately fixed and pre-registered. Its size is known before
any data is seen, which is exactly the condition under which the deflation
correction is valid. A tool that let you keep adding signals until one looked
good would invalidate its own statistics -- so this one does not.

The expected outcome, stated in advance
---------------------------------------
On most instruments, **nothing here will be significant.** That is the correct
result and the reason the module exists. A research tool whose value is measured
by how many opportunities it reports is a tool optimised to mislead you.

References
----------
Jegadeesh, N. & Titman, S. (1993), *J. Finance* 48(1) -- momentum.
Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012), *J. Financial Economics*
    104(2) -- time-series momentum.
Bailey, D. H. & Lopez de Prado, M. (2014), *J. Portfolio Management* 40(5).
Hansen, P. R. (2005), *JBES* 23(4) -- superior predictive ability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["SignalBattery", "SignalResult", "run_signal_battery"]


@dataclass
class SignalResult:
    """One signal's performance, with every correction applied."""

    name: str
    description: str
    #: What a positive value of this signal predicts.
    reads_as: str

    in_sample_sharpe: float = float("nan")
    deflated_p_value: float = float("nan")
    out_of_sample_sharpe: float = float("nan")
    hit_rate: float = float("nan")
    turnover: float = float("nan")
    n_trades: int = 0
    #: Correlation between the signal and the NEXT period's return.
    information_coefficient: float = float("nan")
    survives_spa: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def is_significant(self) -> bool:
        """Survives deflation *and* the joint test. Both, not either."""
        return bool(self.deflated_p_value < 0.05 and self.survives_spa)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.in_sample_sharpe):
            return "not computable"
        if self.is_significant:
            return "SURVIVES all corrections"
        if self.deflated_p_value < 0.05:
            return "survives deflation, fails the joint test"
        if self.in_sample_sharpe > 0.5:
            return "looks good in sample, does not survive"
        return "no evidence"


@dataclass
class SignalBattery:
    """The whole battery, plus the joint tests that judge it."""

    results: list[SignalResult]
    n_signals: int
    n_observations: int
    #: Hansen SPA p-value over the battery: is ANY signal genuinely superior?
    spa_p_value: float = float("nan")
    best_signal: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def any_significant(self) -> bool:
        return any(r.is_significant for r in self.results)

    def sorted_by_evidence(self) -> list[SignalResult]:
        """Ordered by deflated p-value -- by evidence, not by raw performance."""
        return sorted(
            self.results,
            key=lambda r: r.deflated_p_value if np.isfinite(r.deflated_p_value) else 1.0,
        )


# --------------------------------------------------------------------------- #
# The signals themselves. Each maps a price history to a position in [-1, 1].
# Every one is computed using ONLY information available at time t, which is
# enforced by construction: each returns a position for t+1 from data up to t.
# --------------------------------------------------------------------------- #
def _time_series_momentum(prices: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """Long if the trailing return over ``lookback`` is positive."""
    positions = np.zeros(prices.size)
    for t in range(lookback, prices.size):
        positions[t] = np.sign(prices[t] - prices[t - lookback])
    return positions


def _moving_average_crossover(
    prices: NDArray[np.float64], fast: int, slow: int
) -> NDArray[np.float64]:
    """Long when the fast moving average is above the slow one."""
    positions = np.zeros(prices.size)
    for t in range(slow, prices.size):
        fast_mean = float(np.mean(prices[t - fast : t]))
        slow_mean = float(np.mean(prices[t - slow : t]))
        positions[t] = np.sign(fast_mean - slow_mean)
    return positions


def _mean_reversion(
    prices: NDArray[np.float64], lookback: int, threshold: float
) -> NDArray[np.float64]:
    """Fade deviations beyond ``threshold`` standard deviations."""
    positions = np.zeros(prices.size)
    for t in range(lookback, prices.size):
        window = prices[t - lookback : t]
        sigma = float(np.std(window, ddof=1))
        if sigma <= 0:
            continue
        z = (prices[t] - float(np.mean(window))) / sigma
        if abs(z) > threshold:
            positions[t] = -np.sign(z)
    return positions


def _volatility_scaled_momentum(
    prices: NDArray[np.float64], lookback: int, vol_window: int
) -> NDArray[np.float64]:
    r"""Momentum sized inversely to realised volatility.

    Scaling by :math:`1/\sigma` targets constant *risk* rather than constant
    notional, which is what makes time-series momentum work across assets with
    very different volatilities. It is also, on its own, a large part of the
    strategy's historical Sharpe ratio.
    """
    positions = np.zeros(prices.size)
    returns = np.diff(np.log(np.maximum(prices, 1e-300)))
    start = max(lookback, vol_window) + 1
    for t in range(start, prices.size):
        sigma = float(np.std(returns[t - vol_window - 1 : t - 1], ddof=1))
        if sigma <= 0:
            continue
        direction = np.sign(prices[t] - prices[t - lookback])
        target = 0.01 / (sigma * np.sqrt(252.0))
        positions[t] = float(np.clip(direction * target, -1.0, 1.0))
    return positions


def _volatility_breakout(prices: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Long on a break above the trailing high, short below the trailing low."""
    positions = np.zeros(prices.size)
    for t in range(window, prices.size):
        window_slice = prices[t - window : t]
        if prices[t] > float(np.max(window_slice)):
            positions[t] = 1.0
        elif prices[t] < float(np.min(window_slice)):
            positions[t] = -1.0
    return positions


#: The pre-registered battery. Its size is fixed before any data is seen, which
#: is the condition under which the deflated Sharpe correction is valid.
SIGNALS: list[tuple[str, str, str, object]] = [
    (
        "momentum_21d",
        "Sign of the trailing one-month return",
        "recent winners keep winning over the next month",
        lambda p: _time_series_momentum(p, 21),
    ),
    (
        "momentum_126d",
        "Sign of the trailing six-month return",
        "the classic medium-horizon momentum effect (Jegadeesh-Titman)",
        lambda p: _time_series_momentum(p, 126),
    ),
    (
        "momentum_252d",
        "Sign of the trailing twelve-month return",
        "time-series momentum (Moskowitz-Ooi-Pedersen)",
        lambda p: _time_series_momentum(p, 252),
    ),
    (
        "ma_cross_50_200",
        "50-day moving average above the 200-day",
        "the 'golden cross'; a slow trend filter",
        lambda p: _moving_average_crossover(p, 50, 200),
    ),
    (
        "ma_cross_20_50",
        "20-day moving average above the 50-day",
        "a faster trend filter, with correspondingly higher turnover",
        lambda p: _moving_average_crossover(p, 20, 50),
    ),
    (
        "mean_reversion_5d",
        "Fade moves beyond 1.5 sd of the trailing week",
        "short-horizon overreaction reverses",
        lambda p: _mean_reversion(p, 5, 1.5),
    ),
    (
        "mean_reversion_21d",
        "Fade moves beyond 2 sd of the trailing month",
        "monthly reversal",
        lambda p: _mean_reversion(p, 21, 2.0),
    ),
    (
        "vol_scaled_momentum",
        "12-month momentum, sized inversely to realised volatility",
        "trend, at constant risk rather than constant notional",
        lambda p: _volatility_scaled_momentum(p, 252, 63),
    ),
    (
        "breakout_63d",
        "Break above/below the trailing quarter's range",
        "range breakouts persist",
        lambda p: _volatility_breakout(p, 63),
    ),
]


def _strategy_returns(
    positions: NDArray[np.float64], returns: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply positions to returns with a one-period lag.

    The lag is the entire point. A position decided using data through time ``t``
    can only earn the return from ``t`` to ``t+1``. Aligning them without the lag
    is look-ahead bias, and it is the commonest way a backtest produces an
    impossible Sharpe ratio.

    Index bookkeeping, since it is easy to get wrong by one and impossible to
    notice afterwards:

    * ``positions`` has length ``n`` and is aligned with *prices*: ``positions[t]``
      is decided from prices up to and including ``t``.
    * ``returns`` has length ``n - 1``: ``returns[t] = log(P[t+1] / P[t])``, the
      return earned *after* ``t``.

    So ``positions[t]`` earns ``returns[t]``, and the product runs over
    ``positions[:-1]`` (the last position has no subsequent return to earn).
    """
    if positions.size != returns.size + 1:
        raise ValueError(
            f"positions must be one longer than returns (prices vs diffs); "
            f"got {positions.size} and {returns.size}"
        )
    return positions[:-1] * returns


def run_signal_battery(
    prices: NDArray[np.float64],
    *,
    transaction_cost_bps: float = 5.0,
    periods_per_year: float = 252.0,
    n_bootstrap: int = 400,
    seed: int = 20240719,
) -> SignalBattery:
    """Run every signal, then judge them jointly.

    Purpose
        Answer "does anything predict this instrument?" in a way that accounts
        for having asked several questions at once.
    Method
        For each signal: generate positions using only past data, lag them by one
        period, apply transaction costs on every position change, and compute the
        in-sample Sharpe ratio. Then:

        * **Deflate** each Sharpe by the number of signals in the battery.
        * Compute an **out-of-sample** Sharpe on a purged, embargoed split.
        * Run **Hansen's SPA** over the whole battery to ask whether the best of
          them is better than nothing, given that it was chosen by looking.
    Inputs
        ``transaction_cost_bps`` -- round-trip cost per unit turnover. The default
        5 bp is optimistic for a small-cap and pessimistic for a liquid ETF; it
        matters enormously for the high-turnover signals, which is why it is
        charged rather than assumed away.
    Outputs
        :class:`SignalBattery`.
    Failure modes
        Raises if fewer than 300 observations: with less, the 252-day signals
        have no history and the out-of-sample split is meaningless.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> prices = 100 * np.exp(np.cumsum(rng.standard_normal(800) * 0.01))
        >>> battery = run_signal_battery(prices, n_bootstrap=100)
        >>> # A random walk is unpredictable, so nothing should survive.
        >>> battery.any_significant
        False
    """
    from quantos.core.rng import SeedBank
    from quantos.core.stats.multipletest import hansen_spa
    from quantos.risk.metrics import sharpe_ratio
    from quantos.strategy.validation import deflated_sharpe_ratio

    prices = np.asarray(prices, dtype=float).ravel()
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 300:
        raise ValueError(
            f"need at least 300 observations to run the battery, got {prices.size}. "
            "The 252-day signals would have no history."
        )

    returns = np.diff(np.log(prices))
    rng = SeedBank(root=seed).child("signal_battery").generator()
    cost = transaction_cost_bps * 1e-4

    results: list[SignalResult] = []
    performance_columns: list[NDArray[np.float64]] = []

    for name, description, reads_as, generator in SIGNALS:
        result = SignalResult(name=name, description=description, reads_as=reads_as)
        try:
            positions = np.asarray(generator(prices), dtype=np.float64)  # type: ignore[operator]
        except (ValueError, IndexError) as error:  # pragma: no cover - defensive
            result.notes.append(f"could not compute: {error}")
            results.append(result)
            continue

        gross = _strategy_returns(positions, returns)
        # Charge costs on every change in position.
        turnover = np.abs(np.diff(np.concatenate([[0.0], positions[:-1]])))
        net = gross - turnover[: gross.size] * cost

        active = net[positions[:-1] != 0.0]
        if active.size < 100 or float(np.std(net, ddof=1)) == 0.0:
            result.notes.append("too few active periods to evaluate")
            results.append(result)
            continue

        result.in_sample_sharpe = sharpe_ratio(net, periods_per_year=periods_per_year)
        result.hit_rate = float(np.mean(active > 0))
        result.turnover = float(np.mean(turnover) * periods_per_year)
        result.n_trades = int(np.sum(np.abs(np.diff(positions)) > 1e-12))
        # Same alignment as _strategy_returns: positions[t] predicts returns[t].
        if float(np.std(positions[:-1])) > 0 and float(np.std(returns)) > 0:
            result.information_coefficient = float(np.corrcoef(positions[:-1], returns)[0, 1])
        deflated = deflated_sharpe_ratio(net, n_trials=len(SIGNALS))
        result.deflated_p_value = deflated.p_value

        # Purged out-of-sample split: train on the first 70%, test on the last
        # 30%, with a gap so no signal's lookback window straddles the boundary.
        split = int(0.7 * net.size)
        # The embargo scales with the sample. A fixed 252 days consumed the
        # entire test set on a three-year history, so the out-of-sample column
        # silently read '--' for every signal.
        embargo = min(252, max(21, net.size // 10))
        if net.size - split - embargo > 100:
            oos = net[split + embargo :]
            if float(np.std(oos, ddof=1)) > 0:
                result.out_of_sample_sharpe = sharpe_ratio(oos, periods_per_year=periods_per_year)

        results.append(result)
        performance_columns.append(net)

    battery = SignalBattery(
        results=results, n_signals=len(SIGNALS), n_observations=int(returns.size)
    )

    # Joint test over the whole battery.
    if len(performance_columns) >= 2:
        length = min(column.size for column in performance_columns)
        matrix = np.column_stack([column[-length:] for column in performance_columns])
        try:
            spa = hansen_spa(matrix, n_bootstrap=n_bootstrap, rng=rng)
            battery.spa_p_value = spa.p_value
            evaluated = [r for r in results if np.isfinite(r.in_sample_sharpe)]
            if evaluated and 0 <= spa.best_index < len(evaluated):
                battery.best_signal = evaluated[spa.best_index].name
            # A signal "survives SPA" only if the joint test rejects at all.
            if spa.p_value < 0.05:
                for result in evaluated:
                    result.survives_spa = result.name == battery.best_signal
        except (ValueError, RuntimeError) as error:  # pragma: no cover - data dependent
            battery.notes.append(f"joint SPA test failed: {error}")
    else:
        battery.notes.append("too few evaluable signals for a joint test")

    if not battery.any_significant:
        battery.notes.append(
            "No signal survives correction for multiple testing. This is the "
            "usual and expected outcome on a liquid instrument, and it is a "
            "result, not a failure of the search."
        )
    return battery
