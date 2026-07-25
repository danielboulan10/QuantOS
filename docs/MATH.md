# Mathematical Reference

Derivations for the results in this repository that are not one-line lookups.
Each links to its implementation.

## 1. Why `erfc`'s branch points are set by cancellation

[`core/special.py`](../src/quantos/core/special.py)

Two representations are available:

- **Series** (all terms positive, so no cancellation *within* it):
  `erf(x) = (2x/√π) e^{-x²} Σ (2x²)ⁿ / (1·3·5···(2n+1))`
- **Continued fraction** (computes the small tail *directly*):
  `erfc(x) = e^{-x²}/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + ⋯))))`

The naive scheme picks the series where it converges fastest (|x| ≲ 3). That is
the wrong criterion. Writing `erfc(x) = 1 - erf(x)` has relative error
`ε · erf(x)/erfc(x)`, which passes 1e-15 at x ≈ 1.2 and reaches 1e-12 by x = 2.5.
So the *positive* axis must leave the series at 1.2, well before convergence
degrades.

The negative axis has no such problem: `erfc(-y) = 1 + erf(y)` adds two positive
quantities, so the series serves it out to its own accuracy limit.

Consequence: `erfc(26.68)` returns the correct subnormal `1.46e-311` where SciPy
underflows to `0.0`. Tail accuracy is where every VaR number and p-value lives.

## 2. `ndtri` by Halley refinement, and the cancellation trap

Acklam's rational approximation gives 9 correct digits. One Halley step,

```
u = (Φ(x) − p)·√(2π)·e^{x²/2},    x ← x − u/(1 + xu/2)
```

is cubically convergent, so 9 digits become machine precision. The advantage over
a memorised high-order approximation (AS 241) is that accuracy is *inherited from*
`ndtr`: if the CDF is right, the quantile is right.

The trap: for `p > ½` both `Φ(x)` and `p` are close to 1 and their difference
loses every significant digit. Rewriting via the upper tail,

```
Φ(x) − p = (1 − Q(x)) − (1 − r) = r − Q(x),    r = 1 − p
```

keeps both operands small. Measured effect: 9.6e-10 → 3e-16 relative error near
p = 1.

## 3. Ornstein-Uhlenbeck: exact discrete MLE

[`core/timeseries/ou.py`](../src/quantos/core/timeseries/ou.py)

`dX = θ(μ − X)dt + σ dW` sampled at spacing Δt is **exactly** an AR(1):

```
X_{t+Δ} = X_t e^{−θΔ} + μ(1 − e^{−θΔ}) + ε,   Var(ε) = σ²(1 − e^{−2θΔ})/(2θ)
```

so regressing `X_{t+1}` on `X_t` and inverting gives the exact transition-density
MLE:

```
θ = −ln(φ)/Δt,    μ = c/(1 − φ),    σ = σ_ε √(2θ/(1 − φ²))
```

An Euler discretisation instead biases θ downward, and the bias does **not**
vanish as the sample grows — only as Δt → 0.

### Expected first-passage time to the mean

Standardise to `U = (X − μ)/σ_stat` and rescale time by θ, giving
`dU = −U dt + √2 dW`. The hitting time of the origin solves `T'' − uT' = −1`,
`T(0) = 0`. With `v = T'` this is first-order linear with integrating factor
`e^{−u²/2}`:

```
T'(u) = e^{u²/2} ∫_u^∞ e^{−s²/2} ds = √(π/2) e^{u²/2} erfc(u/√2)
```

the constant fixed by requiring `T'` bounded as `u → ∞`. Integrating again,

```
T(u) = √(π/2) ∫_0^{|u|} e^{v²/2} erfc(v/√2) dv,   answer = T(u)/θ
```

Verified against Monte Carlo: analytic 0.3563 vs simulated 0.3654 at 2σ.

Sanity check worth remembering: reverting from 2 stationary standard deviations
takes about **two half-lives**.

## 4. Avellaneda-Stoikov, and why the inventory term is the whole model

[`sim/agents.py`](../src/quantos/sim/agents.py)

Maximising exponential utility of terminal wealth while quoting two-sidedly gives
a reservation price and an optimal total spread:

```
r(s,q,t) = s − q γ σ² (T−t)
δᵃ + δᵇ = γσ²(T−t) + (2/γ) ln(1 + γ/κ)
```

The `−qγσ²(T−t)` term shifts **both** quotes down when long. A maker quoting
symmetrically about the mid accumulates inventory in whichever direction the
market trends — precisely the wrong direction, since it is being adversely
selected. This one term converts a strategy that blows up on a trend into one that
survives, and its magnitude is the price charged for bearing inventory risk.

Quotes are anchored on the **microprice** `(Q_b P_a + Q_a P_b)/(Q_a + Q_b)`, not
the mid: queue imbalance predicts the next mid move, so mid-anchoring quotes
systematically on the wrong side of fair value when the book is lopsided.

## 5. Kyle's linear demand, and why informed traders need a target position

An informed trader with demand `q* = β(V − P)` has a *downward-sloping* demand
curve: it buys as the gap opens and **sells back as it closes**.

An agent that instead buys whenever price is below value accumulates without
bound, hits its position limit, and goes silent — at which point nothing pushes
price toward value. Measured consequence in this repository: the mid moved 7 ticks
while the fundamental moved 49, correlation 0.01. With target-position demand,
correlation 0.78.

## 6. Deflated Sharpe Ratio

[`strategy/validation.py`](../src/quantos/strategy/validation.py)

Under the null of zero skill, the expected maximum of N trial Sharpe ratios is the
extreme-value result for the maximum of N Gaussians:

```
E[max SR] ≈ √V · [ (1−γ)Φ⁻¹(1 − 1/N) + γΦ⁻¹(1 − 1/(Ne)) ]
```

with γ the Euler-Mascheroni constant. The observed Sharpe is tested against *that*
benchmark, not against zero, using the non-normality-adjusted standard error

```
σ_SR = √( (1 − γ₃·SR + (γ₄−1)/4·SR²) / (n−1) )
```

Note the direction of the correction: negative skew and excess kurtosis both
**inflate** the true standard error, so the naive `√(1/n)` overstates
significance — worst for exactly the short-volatility strategies most likely to be
presented with a high Sharpe.

## 7. Hierarchical Risk Parity

[`risk/portfolio.py`](../src/quantos/risk/portfolio.py)

1. `d_ij = √(½(1 − ρ_ij))` — a proper metric on correlations.
2. Single-linkage hierarchy; reorder so similar assets are adjacent
   (quasi-diagonalisation).
3. Recursive bisection, splitting between halves inversely to cluster variance.

The point is what is *absent*: no matrix inverse anywhere. Markowitz inverts the
covariance and so loads maximally on its smallest eigenvalues — the directions
estimated with least precision. HRP is therefore defined even when T < N, at the
cost of any claim to in-sample optimality, which is the right trade when the
inputs are estimates.

## 8. Risk parity needs a damped iteration

Equal risk contribution means `w_i(Σw)_i` constant. The obvious update
`w_i ← (Σw)_i⁻¹` has the right fixed point but **oscillates**: on
`Σ = diag(0.04, 0.16)` it flips between (0.8, 0.2) and (0.5, 0.5) forever.

The damped update `w_i ← normalise(√(w_i/(Σw)_i))` takes a geometric mean of the
current iterate and the target. Squaring shows it has the same fixed point, and it
converges monotonically — landing exactly on the 2:1 inverse-volatility solution
in one step for the diagonal case.

## 9. Almgren-Chriss, and its two informative limits

[`execution/almgren_chriss.py`](../src/quantos/execution/almgren_chriss.py)

```
x(t) = X sinh(κ(T−t))/sinh(κT),    κ = √(λσ²/η)
```

- `κ → 0` (risk-neutral) gives **TWAP**. A straight line is not a naive baseline;
  it is optimal for a trader indifferent to price risk.
- `κT ≫ 1` front-loads heavily, approaching exponential decay of the position.

Permanent impact drops out of the optimisation entirely: **you cannot schedule
your way out of permanent impact.** Only the temporary component responds.

The model assumes impact linear in trading *rate*. Empirically the impact of a
completed order scales as `√Q` — one of the most robust regularities in finance.
Both views are implemented so the tension is explicit.

## 10. Purging and embargo

Standard k-fold is invalid for time series with overlapping labels for three
separate reasons: serial correlation, label overlap, and post-test dependence. A
20-day forward return sampled daily means every label draws on 20 days of future
data, so any training observation within 20 days of a test observation leaks.

*Purging* removes training observations whose label interval overlaps the test
span. *Embargo* additionally removes a fraction immediately after, because serial
correlation leaks where label intervals do not overlap. Shuffling — k-fold's
default — is worse still: it puts future observations in the training set.
