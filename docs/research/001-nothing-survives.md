# Nothing survives an 840-factor search

**Daniel Boulan** · August 2026 · [reproduce](#reproducing-this)

> Searching 840 systematically generated factors on SPY produces a best factor
> with t = 2.23 and 104 factors clearing a naive p < 0.05. None survives
> Romano-Wolf StepM. Two defects found while building the search turned out to
> be more instructive than the search itself: the comparison was ranking
> leverage rather than skill, and the deflated Sharpe ratio is self-defeating in
> exactly the case where there is something to find.

## The problem

The standard pitch for a factor lab is: generate a thousand signals, test each,
publish the best. This describes a machine for producing false discoveries, and
the arithmetic is not subtle. Search a thousand independent worthless signals at
the 5% level and roughly fifty return "significant". The best of the thousand
shows a t-statistic near 3.2 purely from the extreme-value behaviour of a
maximum over many draws.

The correction for this is old and well understood — White (2000), Hansen (2005),
Romano & Wolf (2005). What is less often done is applying it to a search whose
size is actually known.

## Method

Factors are generated from a grammar rather than written by hand:

| dimension | values | count |
|---|---|---:|
| transform | momentum, reversal, volatility, vol change, skew, kurtosis, trend, acceleration, drawdown, up-ratio | 10 |
| lookback | 5, 10, 21, 42, 63, 126, 252 trading days | 7 |
| scaling | raw, expanding z-score, expanding rank, sign | 4 |
| holding period | 1, 5, 21 days | 3 |

Product: **840 factors.**

The grammar is not tidiness. It makes the search reproducible from a seed and its
size *exactly* known — and the size of the search is the input every correction
needs. A hand-written list of "the factors I tried" is always an undercount,
because the ones tried and abandoned never appear on it. That undercount is the
commonest way a corrected result is still wrong.

Every scaling uses expanding-window statistics. A full-sample z-score puts the
future into every observation; it is the most common look-ahead bug in factor
research and it does not announce itself, because the signal simply looks better
than it is. A test asserts that appending future data cannot change a past
signal value, for each of the four scalings.

**Data.** SPY daily total-return-adjusted bars, 2016-08 to 2026-08 (2,511
observations, 2,237 after the longest warm-up).

## Result

```
Best factor, uncorrected:
  up_ratio_252d_zscore_h21
  Sharpe 0.71    t = 2.23    naive p = 0.0255

  104 of 840 factors clear p < 0.05 on their own.
  Pure chance predicts about 42.

Corrected for the size of the search:
  White's Reality Check   p = 0.186
  Hansen's SPA            p = 0.010
  Romano-Wolf StepM       0 survivors
```

The winner looks publishable and is not.

### The two corrections disagree, and both are right

SPA rejects "no factor in this universe has skill" at p = 0.010. StepM, which
controls the family-wise error rate, identifies no individual factor at all.

These answer different questions and the gap between them is the interesting
part. Evidence that *something* in a large correlated family works is far cheaper
to obtain than evidence about *which one*. Only the second is tradeable. A lab
that reported the SPA p-value alone would be reporting a result that cannot be
acted on.

### 104 naive hits against 42 expected

More factors clear the naive bar than chance predicts, which is not evidence of
skill: the 840 factors are heavily correlated (a 21-day momentum and a 42-day
momentum are nearly the same signal), so the effective number of independent
tests is far below 840 and the count is not distributed the way the naive
calculation assumes. This is precisely why the bootstrap-based corrections are
used instead of counting.

## The two defects, which were the interesting part

Both were found by a test that plants a signal deliberately, and neither was
visible in the output.

### The comparison ranked leverage, not skill

The grammar produces P&L series spanning a **1,900× range in standard
deviation** — a `raw`-scaled momentum signal is a number like 0.4, a `sign`-scaled
one is exactly ±1. A max-of-means statistic across those columns compares
position sizes, not skill.

With a signal planted deliberately (tomorrow's return follows the sign of
trailing 21-day momentum), the true factor sat **143rd of 200** by P&L scale.
White's Reality Check returned p = 0.0975 and failed to find it. Hansen's SPA,
which studentises, returned p < 0.0001 and was partly protected — which is
Hansen's own argument for studentisation, arrived at accidentally.

Normalising each factor to equal risk fixes it. The fix is safe because dividing
a column by a positive constant cannot change that factor's own t-statistic; the
constant is computed in sample, which is stated at the site rather than left for
a reader to discover.

After the fix: Reality Check p < 0.0001, and the search recovers
`momentum_21d_sign_h1` — the exact generating rule — at t = 11.4.

### The deflated Sharpe ratio defeats itself when there is skill

The deflated Sharpe (Bailey & López de Prado 2014) deflates an observed Sharpe
against the expected maximum of `n_trials` draws **under the null of no skill**.
The natural thing to hand it is the observed spread of trial Sharpes from the
search. That is wrong, and wrong in a way that only bites when it matters.

On the planted signal:

| trial variance from | expected null maximum | p-value |
|---|---:|---:|
| the search itself | 0.285 | **0.90** |
| estimated under the null | 0.062 | **< 0.0001** |

The true factor has a per-period Sharpe of 0.257 and t = 11.4. Feeding the
search-derived variance treats the spread of trial Sharpes as pure noise — so the
skill inflates the benchmark the winner must clear, and the winner fails to clear
a bar it created. A p-value of 0.90 reads as a verdict rather than a broken
assumption.

(An earlier version also passed *annualised* trial variance into a function
working in per-period units, a 252× error that returned p = 1.0000. That was a
plain bug; the above is not.)

Both variants are now reported side by side, because they disagree precisely when
there is something to find.

## What this does not establish

- **Not that no factor works on SPY.** It establishes that this search, on this
  instrument, over this sample, does not support a discovery. A different
  universe, a cross-section rather than a single instrument, or a longer sample
  could all give a different answer.
- **Not that the factors are worthless.** Momentum has a substantial literature
  behind it. What fails here is finding it *by search* on one instrument over
  2,237 observations — the sample is too short relative to the search.
- **Nothing about transaction costs.** All P&L is gross. Turnover is computed but
  not charged, and several surviving-adjacent factors trade daily.
- **The corrections themselves have assumptions.** The stationary bootstrap
  assumes the dependence structure is stable across the sample, which the
  companion note [003](003-confidently-wrong.md) gives direct reason to doubt.

Harvey, Liu & Zhu (2016) argue that once the profession's collective search is
counted, t > 3.0 is the right bar for a published factor. This search alone was
840 wide.

## Reproducing this

```bash
quantos factors --ticker SPY
quantos factors                 # the same search on pure noise
pytest tests/research/test_factor_lab.py
```

Source: [`research/factor_lab.py`](../../src/quantos/research/factor_lab.py).
The claims verifier re-derives both the negative result on noise and the
positive result on the planted signal — the second is the guard, because a lab
that always says no is not a test, it is a constant.

## References

White, H. (2000). "A Reality Check for Data Snooping." *Econometrica* 68(5).
Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *JBES* 23(4).
Romano, J. P., & Wolf, M. (2005). "Stepwise Multiple Testing as Formalized Data
Snooping." *Econometrica* 73(4).
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio."
*Journal of Portfolio Management* 40(5).
Harvey, C. R., Liu, Y., & Zhu, H. (2016). "… and the Cross-Section of Expected
Returns." *Review of Financial Studies* 29(1).
