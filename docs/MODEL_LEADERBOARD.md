# Volatility forecasting leaderboard

Every forecaster scored on the same walk-forward split, each fit using only
observations preceding the point it forecasts. Reproduce with:

```bash
python scripts/benchmark_models.py --tickers SPY --train 1000 --step 10
```

## Result

SPY, 5,030 daily returns, 2006–2026. **Lower is better** on both losses.

| Forecaster | QLIKE | Pinball | RMSE(vol) | Bias |
|---|---:|---:|---:|---:|
| **GARCH(1,1)-t** | **1.5147** | **0.002260** | 0.00737 | +0.00245 |
| EWMA (λ=0.94) | 1.5972 | 0.002292 | 0.00793 | +0.00239 |
| attention (NumPy) | 1.7629 | 0.002358 | 0.00868 | +0.00276 |
| trailing 252d | 1.9431 | 0.002422 | 0.00931 | +0.00322 |
| random walk (yesterday) | 585.61 | 0.002553 | 0.00831 | −0.00038 |

## What this says

**The attention model loses.** It is beaten by GARCH(1,1) and by an exponentially
weighted moving average that has no fitted parameters at all. This is reported
because it is the result; publishing only the runs where a new model wins is how
the literature ends up full of methods nobody can reproduce.

**It did learn something.** It beats a trailing 252-day standard deviation and a
random walk comfortably, so the architecture and the hand-derived gradients are
working — the model has picked up volatility clustering. It simply does not pick
up more of it than a well-specified parametric model with three parameters.

**Why this outcome is unsurprising.** EWMA is already a GARCH(1,1) with ω = 0,
α = 1−λ, β = λ and persistence pinned at exactly 1. Daily volatility is close to
that process, so the parametric models are nearly correctly specified and there is
little structure left for a flexible model to find — while the flexible model pays
for its capacity in estimation error. A great deal of published deep learning on
financial time series does not clear this bar either, and often does not report it.

**The random walk's QLIKE of 585 is not a bug.** QLIKE contains σ²/ĥ, and using a
single day's squared return as the forecast makes ĥ near zero whenever that day
was quiet, so the ratio explodes. It is precisely why nobody forecasts volatility
with one observation, and why QLIKE is the right loss for exposing it — RMSE
ranks the random walk *third*, which is misleading.

## What would change the result

- **A longer horizon.** The parametric edge is largest at one day. Over a month
  there is more structure — volatility-of-volatility, regime persistence — that a
  flexible model might capture.
- **More inputs.** This model sees only past returns. Realised volatility from
  intraday data, option-implied volatility, or cross-asset signals all carry
  information a returns-only model cannot see.
- **More data.** A few thousand daily observations is a small training set. The
  same architecture on intraday bars has two orders of magnitude more.

Until one of those is tried and measured, the honest summary is the one at the
top: **GARCH wins, and the neural model does not earn a claim.**

## Method notes

- **Losses.** QLIKE is standard for variance forecasts and robust to realised
  variance being a noisy proxy for the latent quantity; MSE on variance
  systematically rewards underprediction. Pinball loss scores the implied
  quantiles and is *proper*, so a forecaster cannot improve it by hedging.
- **Refitting.** Baselines are recomputed at every forecast point. The attention
  model is retrained every 40 forecasts rather than every one — a full retrain per
  point would take hours, and periodic refitting is what a desk does anyway.
- **Splits.** The model's own validation split is chronological, never shuffled.
  A shuffled split would place future observations in training and leak the answer
  backwards, which is the most common way a financial sequence model reports a
  score it did not earn.
