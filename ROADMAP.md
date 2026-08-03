# Roadmap

What is built, what is being built, and — the part most roadmaps omit — what has
been deliberately ruled out and why.

Dates are targets, not commitments. Items move to the
[changelog](CHANGELOG.md) when they land.

## Shipped

| | |
|---|---|
| ✅ | Numerical core with no dependency beyond NumPy, validated against SciPy as a test-only oracle |
| ✅ | Research report on any listed symbol, no API key |
| ✅ | Forward-testing ledger — hash-chained, written before outcomes are known, updated daily in CI |
| ✅ | Forecast distributions with calibration testing, including where calibration fails |
| ✅ | Options: Black-Scholes with full Greeks, American LSMC bracketed by a dual upper bound, Heston by Fourier inversion |
| ✅ | VaR backtesting with EVT — and the published finding that no standard model passes |
| ✅ | SVI volatility surface with arbitrage conditions |
| ✅ | Limit order book, matching engine, agent-based simulation |
| ✅ | Execution: Almgren-Chriss with impact calibration |
| ✅ | Backtest validation: deflated Sharpe, PBO, purged and combinatorial CV, Reality Check / SPA / StepM |
| ✅ | Static site and PWA, rebuilt daily |
| ✅ | Investment calculator matched to a published schedule, with the distribution it hides |
| ✅ | Mutation testing and claim verification in CI |

## In progress

### Factor research lab — *the data-mining trap, demonstrated*
Generate on the order of a thousand systematic factors, test every one, and then
run the multiple-testing machinery already in this repository over the results.
The expected outcome is that the best factor found is **not** significant once
the search is accounted for. That is the point: a lab that publishes the winner
without the correction is producing noise, and this one will show the arithmetic
of why.

### Portfolio stress testing against real crises
Replay a portfolio through dated windows — 2008, the COVID crash, the dot-com
unwind, 2022 inflation, the 2023 regional banking episode — and report drawdown,
time to recovery, worst single day, and the correlation breakdown that makes
diversification fail exactly when it is needed. Historical windows, not
hypotheticals.

### Lattice option pricing
CRR binomial and trinomial trees from scratch, to sit beside the existing
closed-form, Fourier and Monte Carlo methods. Four independent routes to the same
number is a stronger validation than any single one.

### Scenario engine
"What happens if rates fall 100bps" answered from historical factor betas with
HAC standard errors — and reported as a range with its uncertainty attached,
because a point estimate for a macro shock is a fiction.

### Research notes
Written from results this repository actually produced, not chosen topics.
Method, data, result, limitations. The first four are already sitting in the
codebase waiting to be written up.

## Considered and deferred

**Native mobile app.** The site is already an installable PWA that works
offline. A native app would add a build pipeline and an app-store dependency for
capability the PWA already has. Revisit only if something genuinely needs the
native layer.

**Splitting into eight repositories.** A common suggestion for looking
professional. It would make the project look *larger* and be *worse*: the whole
argument here is that one system holds together end to end, that the special
functions under the VaR test are the same ones under the option Greeks. Eight
thin repositories break that and multiply the CI surface. Declined on purpose.

## Ruled out

**LLM-based earnings-call and filing analysis, as usually proposed.** Sending a
10-K to a language model and printing the summary produces confident text with no
error bar, and nothing in this repository could verify it. What *is* planned is
the deterministic version analysts actually rely on: a structural diff of
risk-factor sections between filings, and tone scored with the Loughran-McDonald
finance lexicon — reproducible, auditable, and with published accuracy. That is
strictly more useful than an unverifiable summary, and it fits the one
constraint this project does not bend on.

**Scraped alternative data — satellite imagery, shipping manifests, app-store
ranks.** Attractive on paper. In practice these are either paid feeds,
terms-of-service violations, or scrapes that break within weeks and quietly
return stale numbers. A backtest built on a feed that silently degrades is worse
than no backtest. Public, documented, citable sources only.

**Anything resembling a trade recommendation.** The site carries a disclaimer on
every page, enforced by CI. The forecast is deliberately a distribution rather
than a direction, because over these horizons the direction is not estimable and
drawing a sloping median line would misrepresent the evidence.

## Contributing

Issues and pull requests are welcome. The bar for a change is the same as the bar
for everything already here: a claim needs a check that fails when the claim
becomes false. See [CONTRIBUTING.md](CONTRIBUTING.md).
