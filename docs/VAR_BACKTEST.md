# Does the VaR this site publishes actually hold?

Every research page reports Value-at-Risk and CVaR. Until now nothing tested
whether those numbers were true — the same gap that
[calibration testing](../src/quantos/forecast/calibration.py) closed for the
forward probabilities, left open for the risk figures.

A VaR is a falsifiable claim: *the loss will not exceed this threshold, 99% of
the time*. Two standard tests decide it.

- **Kupiec** — are there the right *number* of breaches?
- **Christoffersen** — do the breaches **cluster**? This is the one people skip,
  and the one that matters. A 99% VaR breached exactly 1% of the time with every
  breach in the same fortnight has passed Kupiec and is useless: that pattern is
  a model that is wrong precisely when it is needed.

## Result: SPY, 4,528 days out of sample, 2008 and 2020 included

Rolling 500-day estimation, 99% VaR, forecasts made before each day's return.

| Model | Breach rate | Kupiec | Independence | Verdict |
|---|---:|---|---|---|
| Gaussian | **2.52%** | rejected | rejected | fails both |
| Historical | 1.55% | rejected | rejected | fails both |
| EVT (GPD tail) | 1.55% | rejected | rejected | fails both |
| GARCH(1,1), normal quantile | 2.47% | rejected | **not rejected** | fails coverage only |
| GARCH(1,1), t quantile | 1.97% | rejected | rejected | fails both |

**Nothing passes.** The promised breach rate is 1%; the best model delivers 1.55%.

This is not a bug in the implementation, and it was not tuned until something
passed — which would have been the exact failure this repository exists to avoid.
It is the well-documented empirical result that VaR models understate tail risk
over periods containing crises, reproduced from scratch.

## What each row teaches

**Gaussian VaR is the worst, by a lot.** 2.52% against a promised 1% — it breaches
two and a half times too often. The tail-index estimate explains why: fitting a
generalised Pareto to SPY losses gives a shape of **+0.193**, a power-law tail
where moments beyond order 5.2 do not exist. A normal distribution cannot
represent that, and the site says so on every page it prints a Gaussian number.

**The conditional model fixes the timing, not the level.** GARCH with a normal
quantile is the only model whose breaches do *not* cluster (independence p =
0.215). That is the strongest single result here: it isolates volatility
clustering as the cause of the clustering failure, and shows a conditional
variance model removes it. Its breach *count* is still far too high.

**A fatter quantile helps the level and is not enough.** Applying the fitted
Student-t quantile instead of the Gaussian one moved coverage from 2.47% to
1.97%. Still rejected.

## The same test on TSLA, 3,543 days

| Model | Breach rate | Verdict |
|---|---:|---|
| Gaussian | 1.86% | fails coverage |
| Historical | 1.27% | **passes both** |
| EVT (GPD tail) | 1.19% | **passes both** |

The contrast is informative. TSLA is always volatile, so a 500-day window is
roughly right most of the time. SPY spends years calm and then is not, and a
trailing window is systematically wrong at exactly the transitions.

## Why EVT is here at all

A historical VaR **cannot return a loss larger than the worst one in its
window** — it is most confident exactly where it has least information. Fitting a
generalised Pareto to the tail gives an estimator that extrapolates: measured on
a t(3) sample whose worst loss was 0.1377, the empirical 99.999% quantile
saturates at 0.1373 while the fitted tail gives 0.327.

It also reports a **shape parameter** rather than assuming one, which is how the
"moments beyond order 5.2 do not exist" statement above is arrived at rather than
asserted.

## Reproducing

```python
from quantos.risk.var_backtest import backtest_var, evt_value_at_risk
result = backtest_var(returns, var_forecasts, confidence=0.99, model="mine")
print(result.summary())
```

## What would change these results

- **A longer estimation window with a shorter refit interval**, so the conditional
  model reacts faster at regime transitions.
- **Filtered historical simulation** — standardise returns by conditional
  volatility, bootstrap the residuals, rescale. It combines the GARCH timing fix
  with an empirical tail and is the obvious next thing to test.
- **A 95% threshold instead of 99%.** More breaches means more power and less
  reliance on the extreme tail; several of these models may pass there, and that
  would itself be worth reporting.
