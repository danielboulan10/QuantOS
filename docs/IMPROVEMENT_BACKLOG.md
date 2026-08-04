# Improvement backlog

Ranked recommendations for QuantOS, each with why it matters, expected impact,
what it costs, how to do it, and where it sits. Written to be actionable by
someone other than the author.

Priority is **P0** (do next), **P1** (this quarter), **P2** (worth doing),
**P3** (only if the constraint changes). Effort is in focused days.

A note on what is *not* here: several obvious-sounding items are in
[ROADMAP.md](../ROADMAP.md) under **Ruled out**, with reasons. Scraped
alternative data, LLM-summarised filings and an eight-repository split are all
declined on purpose rather than forgotten.

---

## P0 — Do next

### 1. Performance attribution (Brinson-Fachler)

**Why it matters.** This is the single most common thing an analyst does that
QuantOS cannot: decompose a portfolio's return against a benchmark into
*allocation* (were you in the right sectors), *selection* (did you pick the right
names within them), and *interaction*. Every investment committee memo contains
this table. Its absence is the clearest gap between "quant library" and
"something a desk uses".

**Expected impact.** High on real-world usefulness and on the internship story —
it is the one module that could plausibly be used at work. Moderate on technical
impressiveness; the arithmetic is not hard, and the value is in getting the
conventions right.

**Tradeoffs.** Needs holdings and benchmark weights, which this repository has no
source for — so the API takes them as input and the CLI needs a CSV loader.
Multi-period linking (Carino, Menchero) is genuinely subtle and is where most
implementations quietly go wrong: naively summing single-period effects does not
reconcile to the total return, and the residual is often material.

**Plan.** `src/quantos/attribution/brinson.py`. Single-period Brinson-Fachler
first, with a test that the three effects sum exactly to the active return.
Then Carino smoothing for multi-period, with a test that the linked effects
reconcile to the compounded active return — that reconciliation *is* the
validation. Holdings loader in `data/`. ~3 days.

### 2. Raise the mutation score on the four newest modules

**Why it matters.** `curve` sits at 52%, `scenario` at 56%, `factor_lab` at 56%.
This repository publishes the argument that coverage lies and the mutation score
is the real measure — so leaving the newest modules mid-range undercuts the
loudest claim it makes.

**Expected impact.** Direct on engineering credibility, because the number is
already published in the README. Low on capability.

**Tradeoffs.** Some survivors are semantically equivalent mutations and chasing
them produces tests that assert nothing. The discipline is to read each survivor
and close only the real ones, which is slower than chasing the percentage.

**Plan.** `make mutation` per module, read the survivor list, write tests only
where a real change would go unnoticed. Target 70%+ on each, and record any
survivor deliberately left alone with its reason. ~1.5 days.

### 3. `quantos research` should include the curve and the stress test

**Why it matters.** Four substantial modules were added and none of them appear
in the flagship command. A user who types the one command everything points at
gets none of the newest work.

**Expected impact.** High on perceived coherence for very little work — this is
integration, not construction.

**Tradeoffs.** The report is already long. Sections must be conditional on asset
class (a curve section is noise for an equity) and the runtime grows.

**Plan.** Add a rates section when the instrument is a bond ETF or the user asks;
add the crisis replay for any instrument with enough history. Gate both on
`--sections`. ~0.5 days.

---

## P1 — This quarter

### 4. SEC filing risk-factor diff and Loughran-McDonald tone

**Why it matters.** The deterministic, defensible version of "earnings
intelligence". Comparing Item 1A between consecutive 10-Ks and scoring tone with
the standard finance lexicon is what analysts actually do by hand, it is
reproducible, and its accuracy can be published. An LLM summary cannot be
verified by anything in this repository, which is why it is ruled out.

**Expected impact.** High on real-world usefulness and on differentiation —
almost no student project does the auditable version.

**Tradeoffs.** EDGAR's full-text structure is inconsistent across filers and
years; section extraction will fail on some documents and must say so rather than
returning a partial diff. The Loughran-McDonald lexicon is a data file, which
brushes against the no-dependency rule (it is data, not a package, so it is
allowed — but it needs a provenance note and a licence check).

**Plan.** `data/edgar.py` for retrieval and caching; `research/filings.py` for
section extraction, structural diff and tone scoring. Validate extraction against
a hand-labelled sample of 20 filings and **publish the extraction failure rate**.
~5 days.

### 5. Cross-sectional factor research

**Why it matters.** Research note 001 says plainly that a search over one
instrument and 2,237 observations cannot support a discovery. A cross-section of
several hundred names changes what the search is able to conclude, and makes the
existing multiple-testing machinery answer a question worth asking.

**Expected impact.** High on research quality. It converts an honest negative
result into a genuine research capability.

**Tradeoffs.** Needs a universe with survivorship handling — a cross-section
built from today's index members is the classic look-ahead, and getting it wrong
would undermine the module's own argument. Bulk data fetching also strains a
public endpoint.

**Plan.** Extend `factor_lab` to a panel. Point-in-time universe construction
first, with the survivorship bias measured and published as a delta against the
naive version — that measurement is itself the finding. ~4 days.

### 6. Split `cli/main.py`

**Why it matters.** 1,646 lines. The business logic has already moved out (see
the [engineering audit](ENGINEERING_REPORT.md)), so what remains is genuinely
presentation — but it is the first file a reviewer opens after the README.

**Expected impact.** Moderate on maintainability, moderate on first impression.
Zero on capability.

**Tradeoffs.** Real regression risk for no user-visible benefit, and it trades
one navigable file for twenty plus a registry. Deferred once already for exactly
this reason; the right trigger is the next time it grows.

**Plan.** `cli/commands/{research,curve,risk,options,sim}.py`, each exporting
`register(subparsers)`. Keep `main.py` as entry point and dispatch. Verify by
snapshotting `--help` output before and after and diffing. ~1 day.

---

## P2 — Worth doing

### 7. Liquidity analytics (Amihud, Roll spread, turnover)

**Why it matters.** The reports say repeatedly that they establish nothing about
"liquidity, borrow availability, or capacity". That honesty is good; measuring
some of it would be better, and Amihud illiquidity is computable from the daily
bars already fetched.

**Impact** moderate. **Tradeoffs:** daily-bar estimators are noisy and the good
ones need intraday data, which is only available by CSV here.
**Plan:** `research/liquidity.py`, validated against the intraday module's
effective-spread estimates on a sample where both are available. ~2 days.

### 8. Interactive charts on the site

**Why it matters.** The SVG charts are static. Hovering a fan chart to read a
quantile is the difference between a figure and a tool.

**Impact** moderate on presentation, low on substance. **Tradeoffs:** DDR-002
forbids a charting library, so this is hand-written SVG plus inline JavaScript —
maintainable only if kept small. **Plan:** a single `tooltip.js` inlined at build
time, driven by data attributes the SVG renderer already emits. ~2 days.

### 9. Property-based tests on the numerical core

**Why it matters.** Hypothesis is already a dependency and used in the order
book. The special functions and distributions are the ideal target — monotonicity,
round-trip inverses, and known bounds hold for *all* inputs, and the SciPy oracle
gives an exact comparison.

**Impact** moderate on confidence in the foundation everything rests on.
**Tradeoffs:** slower CI; needs care to avoid generating inputs where the oracle
itself is unreliable. **Plan:** `tests/core/test_special_properties.py`. ~1.5 days.

### 10. Benchmark regression gate in CI

**Why it matters.** `benchmarks/` exists and CI runs it, but nothing fails when
something gets slower. A performance claim that is not enforced decays.

**Impact** low now, compounding later. **Tradeoffs:** CI runners are noisy, so a
naive threshold produces flaky failures — the README already documents the
order-book throughput ranging 8.1M to 16.5M across runs on the same tape.
**Plan:** record medians of several runs; fail only on a >25% regression against
a committed baseline. ~1 day.

---

## P3 — Only if the constraint changes

### 11. Native mobile app

The site is already an installable PWA that works offline. A native app adds a
build pipeline and an app-store dependency for capability the PWA has. Revisit
only if something genuinely needs the native layer.

### 12. Real-time / streaming data

Everything here is daily bars from a cached public endpoint. Streaming would need
a paid feed, a persistent process, and a different failure model. The research
questions this repository asks do not require it.

### 13. Multi-currency and cross-border conventions

Day-count conventions, holiday calendars and FX hedging are real institutional
requirements and a large amount of unglamorous, high-precision work. Worth doing
only if someone is actually using QuantOS on non-USD instruments.

---

## Standing principles for anything added

1. **A claim needs a check that fails when the claim becomes false.** Anything
   that puts a number in a README or docstring adds a check to
   [`scripts/verify_claims.py`](../scripts/verify_claims.py).
2. **Validate against something independent** — a published benchmark, a closed
   form, an analytic identity, or a case constructed so the answer is known.
   "The tests pass" is not validation.
3. **A negative result is a result.** Seven are published here because they came
   out badly and were kept.
4. **No runtime dependency beyond NumPy** ([DDR-002](ddr/DDR-002-numpy-only-runtime.md)).
5. **State what a module cannot do**, in its own docstring, where someone about
   to misuse it will read it.
