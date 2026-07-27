r"""Baselines a forecasting model must beat before it may claim anything.

Why this module exists before the model does
---------------------------------------------
A sequence model reported in isolation is unfalsifiable. "Our transformer
achieves 0.34 pinball loss" means nothing without the number a random walk
achieves on the same data, and in volatility forecasting the boring baselines are
strong: a great deal of published deep-learning work on financial time series
fails to beat an exponentially weighted moving average once the comparison is
actually run.

So the baselines are written first and the model is measured against them. If the
model wins, the margin is the claim. If it loses, that is a finding worth
publishing, and it is the one this repository would publish.

What is being forecast
----------------------
**Volatility, not direction.** Direction is not reliably forecastable from price
history -- the signal battery in :mod:`quantos.research.signals` demonstrates that
on real instruments, and the forward ledger tests it live. Volatility is
genuinely forecastable because it clusters. So that is the target, and a model
that does well here is doing something real rather than fitting noise.

The scoring rules
-----------------
**Quantile (pinball) loss** for distributional forecasts, because a point
forecast of a random quantity is the wrong object. Pinball loss is *proper*: it
is minimised by reporting your true belief, so a model cannot improve its score by
hedging.

**QLIKE** for variance forecasts, which is standard in the volatility literature
and, unlike mean squared error, is robust to the fact that realised variance is a
noisy proxy for the latent quantity being predicted. MSE on variance rewards
underprediction; QLIKE does not.

Both are reported. A model that wins on one and loses on the other has not won.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ForecastScore",
    "ewma_volatility_forecast",
    "garch_volatility_forecast",
    "historical_volatility_forecast",
    "pinball_loss",
    "qlike",
    "random_walk_volatility_forecast",
    "score_forecast",
]

TRADING_DAYS = 252


def pinball_loss(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    quantile: float,
) -> float:
    r"""Quantile loss at level :math:`\tau`, the proper score for a quantile.

    .. math::
       L_\tau(y, \hat q) = \max\left(\tau (y - \hat q), (\tau - 1)(y - \hat q)\right)

    Being *proper* is the point: the loss is minimised in expectation by
    reporting the true :math:`\tau`-quantile, so a model cannot game it by
    shading its forecasts toward the middle. A symmetric loss like MSE would
    reward exactly that.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}")
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def qlike(realised_variance: NDArray[np.float64], forecast_variance: NDArray[np.float64]) -> float:
    r"""QLIKE loss for variance forecasts.

    .. math:: \text{QLIKE} = \frac{1}{n}\sum \left(\frac{\sigma^2_t}{\hat h_t}
              - \log\frac{\sigma^2_t}{\hat h_t} - 1\right)

    Preferred to mean squared error because realised variance is a *noisy proxy*
    for the latent variance a model is trying to predict. Under MSE that noise
    interacts with the asymmetry of the squared error and systematically rewards
    forecasts that are too low; QLIKE is robust to it, which is why the volatility
    literature settled on it. Zero is a perfect forecast.
    """
    realised = np.asarray(realised_variance, dtype=float)
    forecast = np.asarray(forecast_variance, dtype=float)
    valid = np.isfinite(realised) & np.isfinite(forecast) & (forecast > 0) & (realised > 0)
    if not np.any(valid):
        return float("nan")
    ratio = realised[valid] / forecast[valid]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


@dataclass
class ForecastScore:
    """How one forecaster did, on both scoring rules."""

    name: str
    n_forecasts: int
    qlike: float
    pinball_mean: float
    pinball_by_quantile: dict[float, float] = field(default_factory=dict)
    #: Root mean squared error on volatility, reported for readability only.
    rmse_volatility: float = float("nan")
    bias: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def beats(self, other: ForecastScore) -> bool:
        """Strictly better on *both* rules. Winning one and losing the other is not winning."""
        return bool(self.qlike < other.qlike and self.pinball_mean < other.pinball_mean)

    def summary(self) -> str:
        return (
            f"{self.name:26s} QLIKE {self.qlike:7.4f}  pinball {self.pinball_mean:8.5f}  "
            f"RMSE(vol) {self.rmse_volatility:7.4f}  bias {self.bias:+7.4f}"
        )


def score_forecast(
    name: str,
    realised_returns: NDArray[np.float64],
    forecast_volatility: NDArray[np.float64],
    *,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> ForecastScore:
    r"""Score a volatility forecaster on QLIKE and pinball loss.

    The pinball part treats each forecast as implying a distribution of the next
    return, :math:`N(0, \hat\sigma^2)` scaled to the horizon, and scores the
    quantiles of that distribution against what actually happened. A forecaster
    that is well calibrated in the tails scores better than one that only gets the
    centre right, which point-error metrics cannot see.
    """
    from quantos.core.special import ndtri

    realised = np.asarray(realised_returns, dtype=float)
    forecast = np.asarray(forecast_volatility, dtype=float)
    valid = np.isfinite(realised) & np.isfinite(forecast) & (forecast > 0)
    realised, forecast = realised[valid], forecast[valid]
    if realised.size == 0:
        return ForecastScore(
            name=name, n_forecasts=0, qlike=float("nan"), pinball_mean=float("nan")
        )

    by_quantile: dict[float, float] = {}
    for level in quantiles:
        z = float(ndtri(np.array(level)))
        by_quantile[level] = pinball_loss(realised, z * forecast, level)

    return ForecastScore(
        name=name,
        n_forecasts=int(realised.size),
        qlike=qlike(realised**2, forecast**2),
        pinball_mean=float(np.mean(list(by_quantile.values()))),
        pinball_by_quantile=by_quantile,
        rmse_volatility=float(np.sqrt(np.mean((np.abs(realised) - forecast) ** 2))),
        bias=float(np.mean(forecast - np.abs(realised))),
    )


# --------------------------------------------------------------------------- #
# The baselines themselves. Each maps a history to a one-step volatility forecast.
# --------------------------------------------------------------------------- #
def random_walk_volatility_forecast(returns: NDArray[np.float64], window: int = 1) -> float:
    """Yesterday's absolute return. The weakest sensible baseline.

    Included because it is the floor: a model that cannot beat "tomorrow looks
    like today" is not forecasting.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < window:
        return float("nan")
    return float(np.sqrt(np.mean(returns[-window:] ** 2)))


def historical_volatility_forecast(returns: NDArray[np.float64], window: int = 252) -> float:
    """Trailing standard deviation. Ignores clustering entirely."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 20:
        return float("nan")
    return float(np.std(returns[-window:], ddof=1))


def ewma_volatility_forecast(returns: NDArray[np.float64], lam: float = 0.94) -> float:
    r"""Exponentially weighted moving average -- RiskMetrics' :math:`\lambda = 0.94`.

    The baseline that matters. It has **no fitted parameters**, costs nothing, and
    is genuinely hard to beat at a one-day horizon: it is a GARCH(1,1) with
    :math:`\omega = 0`, :math:`\alpha = 1 - \lambda`, :math:`\beta = \lambda` and
    persistence pinned at exactly 1.

    Any model claiming to forecast volatility must clear this line before its
    architecture is worth discussing.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 20:
        return float("nan")
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must lie in (0, 1), got {lam}")

    weights = lam ** np.arange(returns.size - 1, -1, -1)
    weights /= weights.sum()
    return float(np.sqrt(np.sum(weights * returns**2)))


def garch_volatility_forecast(returns: NDArray[np.float64]) -> float:
    """One-step GARCH(1,1) forecast, gated on an ARCH test.

    Falls back to the sample standard deviation when there are no ARCH effects to
    model -- fitting GARCH to a series without clustering produces persistence
    near 1.0 and is worse than the simple thing.
    """
    from quantos.core.stats.hypothesis import engle_arch
    from quantos.core.timeseries.garch import fit_garch

    returns = np.asarray(returns, dtype=float)
    if returns.size < 250:
        return historical_volatility_forecast(returns)

    centred = returns - float(np.mean(returns))
    if engle_arch(centred, lags=5).p_value >= 0.05:
        return historical_volatility_forecast(returns)

    try:
        fitted = fit_garch(centred, distribution="t")
    except (ValueError, np.linalg.LinAlgError):
        return historical_volatility_forecast(returns)
    if not fitted.converged:
        return historical_volatility_forecast(returns)

    variance = (
        fitted.omega
        + fitted.alpha * float(centred[-1]) ** 2
        + fitted.beta * float(fitted.conditional_variance[-1])
    )
    return float(np.sqrt(max(variance, 1e-12)))
