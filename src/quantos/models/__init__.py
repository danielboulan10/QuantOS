"""Forecasting models, and the baselines they must beat.

:mod:`quantos.models.baselines`
    Random walk, trailing volatility, EWMA and GARCH, scored on QLIKE and pinball
    loss. Written before the model, because a model reported without them is
    unfalsifiable -- and in volatility forecasting the boring baselines are strong.
:mod:`quantos.models.sequence`
    A small attention model in NumPy with hand-derived gradients, verified against
    finite differences.
"""

from quantos.models.baselines import (
    ForecastScore,
    ewma_volatility_forecast,
    garch_volatility_forecast,
    historical_volatility_forecast,
    pinball_loss,
    qlike,
    random_walk_volatility_forecast,
    score_forecast,
)
from quantos.models.sequence import AttentionVolatilityModel, make_windows

__all__ = [
    "AttentionVolatilityModel",
    "ForecastScore",
    "ewma_volatility_forecast",
    "garch_volatility_forecast",
    "historical_volatility_forecast",
    "make_windows",
    "pinball_loss",
    "qlike",
    "random_walk_volatility_forecast",
    "score_forecast",
]
