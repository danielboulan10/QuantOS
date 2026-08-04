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
| ✅ | Factor research lab — 840 factors, corrected for the size of the search |
| ✅ | Historical stress testing against 2008, COVID, 2022, with survivorship refused |
| ✅ | Scenario engine answering macro shocks as a range, and flagging false precision |
| ✅ | Lattice option pricing — a fourth independent route to the same number |
| ✅ | Three published research notes, all negative results |
| ✅ | Mutation testing and claim verification in CI |

## In progress

### SEC filing analysis, the deterministic way
A structural diff of risk-factor sections between consecutive 10-K filings, and
tone scored with the Loughran-McDonald finance lexicon. Reproducible, auditable,
and with published accuracy — see *Ruled out* below for why this rather than an
LLM summary.

### Cross-sectional factor research
The [factor lab](src/quantos/research/factor_lab.py) currently searches one
instrument at a time, and note [001](docs/research/001-nothing-survives.md) says
plainly that a single series over 2,237 observations is too short to support a
discovery. A cross-section of several hundred names is the natural next step and
changes what the search can conclude.

### Detecting a regime break as it happens
Note [003](docs/research/003-confidently-wrong.md) detects that breaks have
occurred. Detecting one *while* it happens is a much harder and much more useful
problem — and one where a negative result would be worth as much as a positive.

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
