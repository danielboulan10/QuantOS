# The stock–bond hedge is not a constant

**Daniel Boulan** · August 2026 · [reproduce](#reproducing-this)

> Treasuries hedged equities in 2008 and again in 2020, and hedged *harder* than
> their calm-market correlation implied. In 2022 the relationship inverted and
> TLT fell 31.2% against SPY's 24.1%. The statistic that would have shown the
> risk in advance is not the one usually reported: average pairwise correlation
> across a mixed portfolio nets out the two effects that matter and moves in the
> wrong direction.

## The problem

A 60/40 portfolio is a bet on a negative stock–bond correlation. So is risk
parity, so is most target-date construction, and so is the standard risk model
underneath them. The correlation is estimated from history and then treated as a
property of the assets.

It is not a property of the assets. It is a property of *what kind of shock is
arriving*, and the two kinds point opposite ways:

- A **growth** shock hurts equities and helps bonds — earnings fall, the policy
  response is easing, and money moves to safety. Correlation negative.
- A **discount-rate** shock hurts both — the same rise in the discount rate
  reprices equity cash flows and bond coupons. Correlation positive.

2008 and 2020 were growth shocks. 2022 was a discount-rate shock.

## Method

Daily total-return-adjusted bars for SPY and TLT from a public feed, 2006-08 to
2026-08 (5,029 observations). Five crisis windows, each dated peak-to-trough on
the S&P 500 rather than by calendar quarter:

| window | dates |
|---|---|
| dot-com unwind | 2000-03-24 → 2002-10-09 |
| global financial crisis | 2007-10-09 → 2009-03-09 |
| COVID crash | 2020-02-19 → 2020-03-23 |
| 2022 inflation shock | 2022-01-03 → 2022-10-12 |
| regional banking crisis | 2023-03-01 → 2023-05-04 |

Correlation is measured inside each window and over the two calendar years
immediately preceding it. The dot-com window is **not testable** on this feed,
which reaches only to 2006 — reported as untestable rather than dropped, because
an omitted row reads as a row that was survived.

## Result

| window | corr before | corr during | SPY | TLT |
|---|---:|---:|---:|---:|
| dot-com unwind | *no data* | | | |
| global financial crisis | −0.18 | **−0.47** | −54.8% | **+24.9%** |
| COVID crash | −0.39 | **−0.50** | −33.4% | **+14.2%** |
| **2022 inflation shock** | −0.40 | **+0.03** | −24.1% | **−31.2%** |
| regional banking crisis | +0.04 | −0.35 | +2.6% | +4.3% |

In 2008 and 2020 the hedge did not merely hold; it *tightened* — exactly the
behaviour a flight to quality produces, and the reason a correlation estimated in
calm markets understates how well Treasuries work in a growth shock.

In 2022 it inverted. The sign was wrong, not the magnitude, and the hedge leg
lost more than the leg it was hedging.

## The statistic that hid it

The first version of this analysis reported one number: mean pairwise
correlation across the portfolio. On SPY, QQQ, IWM, EFA, TLT and GLD through the
global financial crisis that number **fell**, from +0.16 to −0.01, and the
analysis concluded the assets had diversified.

That is the opposite of the risk being looked for, and the average was the
reason. Decomposing the same pairs:

| pair type | calm | crisis | change |
|---|---:|---:|---:|
| equity–equity (6 pairs) | +0.84 | **+0.90** | +0.06 |
| equity vs bonds/gold (9 pairs) | +0.05 | **−0.20** | −0.25 |

Both effects are real and they point in opposite directions:

- **Correlations among the risk assets converged.** EFA–QQQ went +0.75 → +0.88,
  QQQ–SPY +0.86 → +0.92. This is where a portfolio's concentration is, so this is
  where convergence hurts.
- **Bonds and gold diversified harder.** SPY–TLT went −0.18 → −0.47.

Averaging fifteen pairs of two different kinds produces a summary of neither. The
decomposition is now the output and the average is not reported at all.

The practical form of the warning is a **sign flip**: a pair whose correlation
reverses rather than merely weakens. A hedge that weakens is a smaller offset. A
hedge that inverts is a position that was sized as though it reduced the risk it
was in fact adding to.

## What this does not establish

- **Not that bonds stopped being a hedge.** One episode. The 2023 banking window
  in the same table shows the correlation going negative again immediately
  afterwards. The claim is that the relationship is regime-dependent, not that it
  has permanently changed sign.
- **Not a prediction.** Knowing that the correlation depends on the type of shock
  does not tell you which type is coming. It tells you that a single number
  estimated from history is the wrong object to build a portfolio on.
- **Two assets, two decades.** SPY and TLT are one point on the duration and
  credit spectrum. The mechanism should apply to any long-duration hedge but is
  not tested here on any other.
- **The replay assumes the position was held throughout.** It does not model
  margin calls, redemptions, or the decision to sell at the bottom — which is
  what actually converts a drawdown into a loss.

## An aside on survivorship

Building this surfaced a trap worth naming. An instrument that did not exist in
2008 cannot be stress tested against 2008, and the dangerous case is not zero
coverage but *partial* coverage: an ETF listed in November 2008 has 25% of the
GFC window in its history and would report a shallow drawdown, having missed the
collapse. That number is indistinguishable from a real one.

Windows are therefore accepted only when the instrument's history begins before
the crisis does, with a fortnight of grace, and the fraction covered is reported
either way. A CI claim asserts that the November-2008 case is refused.

## Reproducing this

```bash
quantos stress --ticker SPY --against TLT GLD
pytest tests/risk/test_stress.py
```

Source: [`risk/stress.py`](../../src/quantos/risk/stress.py). The 2022 inversion
and the 2008 tightening are both re-derived from live data by the claims
verifier, so if either stops being true this note fails CI rather than quietly
becoming wrong.

## References

Longin, F., & Solnik, B. (2001). "Extreme Correlation of International Equity
Markets." *Journal of Finance* 56(2).
Ang, A., & Chen, J. (2002). "Asymmetric Correlations of Equity Portfolios."
*Journal of Financial Economics* 63(3).
