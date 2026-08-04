# Changelog

Notable changes to QuantOS. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [semantic](https://semver.org/spec/v2.0.0.html).

Entries record what changed **and what it revealed**. Several of the most useful
changes here are ones where the measurement came out badly and was kept.

## [Unreleased]

## [1.5.0] — 2026-08-03

### Added
- **Factor research lab** ([`research/factor_lab.py`](src/quantos/research/factor_lab.py)).
  840 factors from a grammar, then the multiple-testing machinery applied to the
  whole search rather than to the winner. `quantos factors`.
- **Historical stress testing** ([`risk/stress.py`](src/quantos/risk/stress.py))
  against 2008, COVID, 2022 and the 2023 banking episode, with survivorship
  refused rather than papered over. `quantos stress`.
- **Scenario engine** ([`risk/scenario.py`](src/quantos/risk/scenario.py)):
  macro shocks answered as a range, with HAC intervals. `quantos scenario`.
- **Lattice option pricing** ([`derivatives/lattice.py`](src/quantos/derivatives/lattice.py)):
  CRR binomial and Boyle trinomial. `quantos lattice`.
- **Three research notes** in [`docs/research/`](docs/research/).
- Architecture diagrams, real screenshots, a self-invalidating recorded demo,
  a link checker in CI, and release tags back to v1.0.0.
- Claims verified in CI: 15 → **24**.

### Results kept because they were unflattering
- **Nothing survives an 840-factor search on SPY.** Best t = 2.23; 104 of 840
  clear a naive p < 0.05 against 42 expected. Zero survivors.
- **The stock–bond hedge inverted in 2022** — TLT −31.2% against SPY −24.1%,
  correlation −0.40 → +0.03. It held and tightened in 2008 and 2020.
- **A macro beta with t = 8.15 had the sign wrong** for the regime that
  followed: QQQ's rate beta was +8 to +9 across four consecutive periods
  2006–2021, then −2.8 through the 2022 hiking cycle.
- **The published Longstaff-Schwartz value is 0.0087 low**, which is the
  downward bias LSMC's own theory predicts. Measured against a converged lattice.

### Fixed
- The factor comparison ranked **leverage, not skill** — a 1,900× spread in P&L
  scale meant the max-of-means statistic compared position sizes. Found by a
  planted-signal test; the true factor had ranked 143rd of 200.
- The deflated Sharpe was fed annualised variance while working in per-period
  units (252×), returning p = 1.0000 on a t = 11.4 signal. Fixing the units
  exposed a deeper issue: the deflation treats the trial spread as pure noise, so
  real skill inflates the benchmark. Both variants are now reported.
- Correlation was summarised by a single average that **moved the wrong way**
  through the GFC. Replaced with the decomposition.
- The scenario engine's first factor was a bond ETF return, which is not
  identified — it mixes the discount-rate channel with flight to quality and
  reverses the apparent sign. Replaced with the actual Treasury yield.

### Changed
- Mutation scores: lattice **100%** (first module to reach it), stress 24% → 68%,
  factor lab 36% → 56%, calculator 56% → 84%.

## [1.4.0] — 2026-08-03

### Added
- **Investment calculator** ([`planning/calculator.py`](src/quantos/planning/calculator.py)).
  Reproduces the compound-interest figure every online calculator prints, then
  gives the distribution behind it. Available as `quantos plan`, as a
  [page on the site](https://danielboulan10.github.io/QuantOS/calculator.html)
  where the arithmetic runs live in the browser, and as a library function.
- Solvers for the three inverse questions: required return, required
  contribution, required years.
- Two new claims in the CI verifier (17 total).

### Verified
- The published schedule of a widely used online calculator is reproduced
  **year by year to under half a cent** over ten years. Their monthly rate
  convention is undocumented; it is the effective `(1+r)^(1/12)-1`, not the
  nominal `r/12`, and the nominal reading misses the ten-year figure by over $9.

### Result kept because it was unflattering
- $20,000 plus $1,000/month for ten years at 6% projects to **$198,290**. At a
  realistic 15% volatility that figure is reached **43%** of the time; the median
  is $187,439. The printed number is roughly the 56th percentile, not the middle.
  CI fails if this ever reverses.

### Changed
- Mutation score for the new module raised 56% → 84% by closing four real gaps
  (the growth/deposit split, the first-year deposit column, daily compounding,
  the inflation note). Two of the tests written to close them asserted false
  things and were corrected rather than deleted.

## [1.3.0] — 2026-07-28

### Added
- **Heston stochastic volatility** by Fourier inversion, with Gauss-Legendre
  quadrature via Golub-Welsch — both written here.
- **American options priced as an interval**: Longstaff-Schwartz lower bound plus
  the Andersen-Broadie dual upper bound, so the answer is a bracket rather than a
  point estimate asserted with unearned confidence.
- **VaR backtesting**: Kupiec unconditional coverage, Christoffersen
  independence, and peaks-over-threshold EVT.
- **Options market making**: Greek-space inventory, generalised Avellaneda-Stoikov
  quoting, P&L attribution.
- **Execution backtest** with impact calibration.
- **Mutation testing** ([`scripts/mutation_test.py`](scripts/mutation_test.py)),
  sandboxed to a temporary copy of the repository.
- **Claim verification in CI** ([`scripts/verify_claims.py`](scripts/verify_claims.py)).

### Results kept because they were unflattering
- **No standard VaR model passes on SPY.** The best delivers 1.55% breaches
  against a promised 1%. Published in [VAR_BACKTEST.md](docs/VAR_BACKTEST.md)
  rather than quietly dropped.
- **Heston's own 1993 formulation overprices by 93% at T=5**, silently, from a
  complex-logarithm branch cut. Both formulations are kept so the failure can be
  reproduced.
- **Almgren-Chriss misranks execution schedules** when permanent impact is
  schedule-dependent — the permanent term is schedule-invariant, so the model
  cannot distinguish orderings it is being asked to rank.
- **Mutation testing found a module scoring 0%.** No test file existed for it.
  Coverage had not noticed.

### Fixed
- Murphy's Brier decomposition did not close (gap 4.2e-4). The within-bin
  variance and covariance terms were missing; it now closes to 2.8e-17.
- Calibration returned a passing verdict on a model with negative skill. Replaced
  with a three-way verdict including an explicit *insufficient evidence* state.
- `distribution="normal"` produced fatter tails than `"t"` — a fallback was
  silently using Student-t innovations regardless of the request.
- The first claim verifier had three false positives from regex matching prose
  ("...from finance"). Rewritten using `ast`.

## [1.2.0] — 2026-07-26

### Added
- **Forward-testing ledger** ([`live/ledger.py`](src/quantos/live/ledger.py)):
  append-only, hash-chained, written before the outcome is known. Overlapping
  horizons discounted by greedy interval scheduling instead of double-counted.
  Updated daily in CI — [the record](forward/RECORD.md).
- **Forward probability engine**: 20,000 simulated paths, first-passage and
  drawdown probabilities, then calibration testing of those probabilities.
- **Static site** generated daily and published to GitHub Pages, with a PWA
  manifest and service worker.
- **SVI volatility surface** fitting with Durrleman butterfly and calendar
  arbitrage conditions.
- **Noise-corrected intraday volatility**: signature plots, Epps curves,
  seasonality.
- **C++ order book** behind the existing interface.

### Results kept because they were unflattering
- **A NumPy attention model loses to GARCH.** It is on
  [the leaderboard](docs/MODEL_LEADERBOARD.md) with its loss recorded, and CI
  fails if it ever starts winning without the leaderboard being updated.
- **Forecast probabilities have real skill for ordinary moves and none for rare
  ones.** The calibration curve is published including the part where it fails.

### Measured
- The obvious C++ integration was benchmarked at **1.2x** the pure Python book —
  not worth a build step. Profiling showed 83% of runtime was Python object
  churn, not matching. Batching the tape across the boundary reached ~35x. The
  first table quoted a single run to six significant figures it had not earned;
  it now reports a median and its spread.

### Fixed
- Requesting `range=max` silently returned monthly bars. The feed now checks the
  response's `dataGranularity` against what was asked and refuses a mismatch.
- Factor alignment truncated instruments to the factor grid (2,151 → 747
  observations, which flipped a Sharpe verdict). Factors are now aligned onto the
  instrument's own grid with NaN padding.
- `sharpe_standard_error` was stored per-period beside an annualised `sharpe`.

## [1.1.0] — 2026-07-25

### Added
- **`quantos research`**: a full report on any listed symbol — US equities, ETFs,
  indices, foreign listings, crypto — with no API key.
- **Real data layer**: split- and dividend-adjusted daily bars, FRED macro
  series, Fama-French factors.
- **Figure gallery** generated from real data.

### Changed
- The README's price-discovery claim was cherry-picked from a favourable seed.
  Replaced with the distribution across seeds: mean correlation **0.29**,
  standard deviation **0.48** — the simulation does *not* reliably discover its
  own fundamental, and the README now says so.

## [1.0.0] — 2026-07-25

Initial public release.

### Added
- Numerical core with no dependencies beyond NumPy: `erf`, `ndtri`, incomplete
  gamma and beta, all validated against SciPy as a test-only oracle.
- Reproducibility contract: RNG streams keyed by semantic path, so adding an
  agent cannot perturb another's numbers.
- Limit order book with price-time priority, intrusive linked lists for O(1)
  cancel, five property-tested invariants.
- Discrete-event market simulation with agents, latency, and a stylised-fact
  battery.
- Black-Scholes with the full Greek set through vanna/volga/charm.
- Backtest validation: deflated Sharpe, PBO, purged and combinatorial CV.
- White's Reality Check, Hansen's SPA, Romano-Wolf StepM.
- GARCH/GJR-GARCH MLE, exact OU estimation, Engle-Granger and Johansen
  cointegration, OLS with HAC standard errors.

[Unreleased]: https://github.com/danielboulan10/QuantOS/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/danielboulan10/QuantOS/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/danielboulan10/QuantOS/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/danielboulan10/QuantOS/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/danielboulan10/QuantOS/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/danielboulan10/QuantOS/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/danielboulan10/QuantOS/releases/tag/v1.0.0
