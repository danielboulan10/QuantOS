# A t-statistic of 8, with the sign wrong

**Daniel Boulan** · August 2026 · [reproduce](#reproducing-this)

> QQQ's sensitivity to the 10-year Treasury yield, estimated on twenty years of
> daily data, is +5.71 with t = 8.15 and a 90% confidence interval that excludes
> zero. Every conventional check passes. Split by period, the beta is positive in
> four consecutive regimes and then **negative** through the 2022 hiking cycle.
> The interval was never measuring the thing that went wrong.

## The problem

"What happens if rates fall 100bps" is the question every investment committee
asks. The standard answer is a beta times a shock, and the sophisticated version
of that answer adds a confidence interval.

The interval is the part worth examining. A confidence interval on a regression
coefficient quantifies **sampling error**: how much the estimate would move if
you drew a different sample *from the same process*. It says nothing at all about
whether the process is the same one that will generate the future. When the
relationship itself changes, the interval is not merely too narrow — it is
answering a different question, and answering it precisely.

## Method

QQQ daily returns, 2006-08 to 2026-08 (4,953 usable observations after
alignment). The factor is the daily change in the 10-year constant-maturity
Treasury yield (FRED `DGS10`), in decimal units, so a beta multiplied by 0.01 is
the response to a 100bp move.

Ordinary least squares with an explicit intercept and **Newey-West** standard
errors, because macro betas are fitted on autocorrelated data where the classical
standard error is itself too small. Betas are also estimated on four contiguous
subsamples, and separately on hand-chosen regime windows.

### A first attempt that was not identified

The obvious factor is a bond ETF's return — TLT falls when rates rise, so `−TLT`
looks like a rate change. It is not identified. A bond ETF's return mixes the
discount-rate channel with flight-to-quality, and over a sample where risk-on and
risk-off dominate, the fitted sign says a rate *fall* hurts equities. The
regression is picking up the equity–bond risk relationship, not the rate
sensitivity. The actual yield series is the right regressor and it reverses the
apparent answer.

Macro series are aligned onto the instrument's own dates rather than the reverse.
Aligning the other way truncates the instrument to the macro calendar and
silently shortens the sample — a bug this repository already had once, in the
factor loader, where it cut 2,151 observations to 747 and flipped a Sharpe
verdict.

## Result

Full sample:

```
  beta                 +5.71
  Newey-West std err    0.700
  t                     +8.15
  R-squared             0.056

SCENARIO: rates rise 100bps
  estimated response   +5.71%
  90% interval        [+4.55%, +6.86%]
```

By period:

| period | beta | t | implied response to +100bp |
|---|---:|---:|---:|
| 2006–2009 (GFC) | +8.54 | +7.02 | +8.5% |
| 2010–2014 (ZIRP) | +9.23 | +11.60 | +9.2% |
| 2015–2019 | +7.86 | +7.79 | +7.9% |
| 2020–2021 (COVID / QE) | +9.40 | +2.77 | +9.4% |
| **2022–2023 (hiking cycle)** | **−2.81** | −1.66 | **−2.8%** |
| 2023–2026 | −1.20 | −1.01 | −1.2% |

Four consecutive regimes agreeing at high significance, then a reversal. The
full-sample estimate is a weighted average dominated by the first four, and it is
on the wrong side of zero for everything after 2021.

The same exercise on utilities (XLU) is messier still: +5.4, +4.0, **−4.9**,
+11.3, −4.4, −5.0. There is no stable number to report.

### Why the interval does not help

The 90% interval is [+4.55%, +6.86%]. It excludes zero, so a reader applying the
usual standard concludes the direction is established. It is 2.3 percentage
points wide, so the same reader concludes the estimate is fairly precise.

Both conclusions are correct statements about sampling error and neither is a
statement about 2022. The subsample estimates span [−1.70%, +9.12%] — nearly
eleven points, five times the width of the interval, and crucially *straddling
zero*.

This is the case the module now detects and names. `confidently_wrong` is true
when the interval excludes zero **and** the subsample estimates disagree about
the sign, and when it fires the warning is printed above the number rather than
below it:

> READ THIS BEFORE THE NUMBER. The interval excludes zero, but the subsample
> estimates disagree about the SIGN. The interval measures sampling error inside
> the estimation window; it does not measure the risk that the relationship
> changes, which is the risk that actually shows up.

### Two smaller honesty problems, for completeness

**R² = 0.056.** Rate changes explain 5.6% of QQQ's daily variance. Almost
everything that moves the asset is not in the model, so even a stable beta would
describe a small part of any outcome.

**The shock is an extrapolation.** A 100bp move is roughly 20 times the daily
standard deviation of the yield change the beta was fitted on. The response is
extrapolated linearly and this estimate cannot test that assumption. Nobody asks
what happens if rates move 5bps, which is the range where the model is actually
supported.

**Multi-factor intervals are too narrow.** When several factors are shocked
together, the reported variance ignores the covariance between beta estimates.
That makes the interval *narrower* than the truth when the factors are
correlated, which is the direction of overconfidence, and it is stated in the
output rather than silently accepted.

## What this does not establish

- **Not that rate betas are useless.** Within a regime they are strongly
  significant and economically sensible. The claim is that the regime boundary is
  where the risk lives, and a confidence interval is blind to it by construction.
- **Not that the regimes were identifiable in advance.** The 2022 break is
  obvious in hindsight. Nothing here detects a break as it happens; the
  subsample check detects that breaks have *occurred*, which is a much weaker and
  much more honest claim.
- **The regime windows are chosen by hand.** They correspond to well-known
  monetary episodes, but choosing them after seeing the data is itself a
  specification search. The four contiguous equal-sized subsamples, which involve
  no choice, tell the same story and are what the code uses.
- **One asset, one factor.** The mechanism — that a discount-rate shock and a
  growth shock move equities and bonds in opposite relative directions — is the
  same one in note [002](002-the-hedge-inverted.md), which is corroboration
  rather than independent evidence.

## The general point

A narrow interval on an unstable parameter is worse than a wide interval on a
stable one, because it invites a decision. Reporting the point estimate alone is
obviously bad. Reporting the point estimate with an interval looks careful and
can be worse, because the interval supplies confidence about the wrong quantity.

The minimum honest addition is cheap: estimate the same parameter on subsamples
and show the spread next to the interval. Where the two disagree, the interval is
not the thing to read.

## Reproducing this

```bash
quantos scenario --ticker QQQ
quantos scenario --ticker XLU --factors rates credit
pytest tests/risk/test_scenario.py
```

Source: [`risk/scenario.py`](../../src/quantos/risk/scenario.py). A CI claim
re-derives the full-sample beta, its interval and the subsample spread from live
data, and fails if the sign ever stops flipping.

## References

Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55(3).
