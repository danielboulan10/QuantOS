# DDR-005: The published site is static, and the web layer stays dependency-free

**Status:** accepted
**Date:** 2026-07-26
**Supersedes:** nothing. **Constrains:** `quantos/web`, `scripts/build_site.py`

## Context

QuantOS needed to become a website that many people could use. Two constraints
were already in force and pulled against that:

- **DDR-002** fixes the runtime dependencies at NumPy alone. Every special
  function, statistical test, optimiser and chart is implemented here and
  validated against SciPy, which appears only as a test-time oracle.
- The price feed is a public but **undocumented** endpoint that rate-limits.

A conventional answer — Flask or FastAPI on a hosted server, fetching per
request — breaks the first constraint and walks straight into the second.

## Decision

Two decisions, which together resolve both constraints.

**1. The web layer is written against the standard library, so DDR-002 holds
unchanged.** `http.server`, hand-written HTML, and this repository's own SVG
renderer. No Flask, no Jinja, no JavaScript charting library. The runtime
dependency set is still exactly `numpy`.

**2. The published site is generated as static files and served without a
runtime.** `scripts/build_site.py` renders every page once a day in CI and
publishes the output to GitHub Pages.

## Why static rather than a server

The research pages are expensive to compute and **identical for every visitor**:
the same GARCH fit, the same 20,000 simulated paths, the same charts. A server
would recompute that per request. Worse, it would place the undocumented upstream
endpoint in the path of every page load, so a traffic spike becomes a rate-limit
which becomes a broken site — the failure mode arriving exactly when the site is
succeeding.

Building once a day fixes three things at once:

| Problem | Server | Static |
|---|---|---|
| Upstream rate limits | once per **visitor** | once per **instrument per day** |
| Page latency | seconds (GARCH + 20k paths) | instant |
| Hosting | runtime, scaling, cost | files, free |

## What was given up

**Only the pre-built universe is browsable.** A visitor cannot analyse an
arbitrary ticker on the published site. This is the real cost, and it is
mitigated rather than eliminated: `quantos serve` runs the *same renderer*
against a live fetch locally, so anyone who installs the package gets the full
tool. The published site is a shop window over a universe chosen to span asset
classes.

**Data is up to a day stale.** For daily-bar research this is immaterial — the
underlying bars are themselves end-of-day — but it would be disqualifying for
anything intraday, and the site says so.

**No personalisation, no accounts, no stored state.** Not a loss for a research
tool, and it removes an entire category of obligation around user data.

## Alternatives considered

**Flask or FastAPI on a hosted runtime.** Rejected: adds runtime dependencies for
a presentation layer, which is not a good enough reason to weaken DDR-002, and
keeps the upstream endpoint on the request path.

**Server with an aggressive cache.** This is the static approach with extra
machinery, a cache to invalidate, and a runtime to keep alive.

**Client-side computation in the browser.** Would need the entire numerical stack
reimplemented in JavaScript — a second implementation of every routine this
repository validates, with no oracle.

## Consequences

- Adding an instrument means editing `DEFAULT_UNIVERSE` and waiting for the next
  build.
- The build must fail loudly rather than publish a stub: the workflow refuses to
  deploy fewer than ten pages, and checks that the disclaimer is present on both
  the index **and** a research page. A near-empty build almost always means the
  data source refused us, and quietly replacing a working site with a broken one
  is the worst available outcome.
- The site is installable (web manifest, service worker), so the "app" is the
  same artifact rather than a second codebase.
- Because the disclaimer must appear on pages reached directly by link, it is
  defined once in `quantos.web.server.DISCLAIMER` and rendered on every page.

## What would change this decision

Wanting arbitrary tickers for visitors who have not installed anything. That
needs a licensed data source that permits redistribution and can absorb request
volume — at which point a server becomes justified, and it would be scoped as an
optional extra so the research core stays dependency-free.
