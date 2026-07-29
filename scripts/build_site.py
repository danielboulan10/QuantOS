#!/usr/bin/env python3
"""Generate the published site: one static page per instrument, no server.

Why static
----------
The research pages are expensive to produce and identical for every visitor: the
same GARCH fit, the same 20,000 simulated paths, the same charts. Computing them
per request wastes the work and puts an undocumented upstream data endpoint in
the path of every page load -- which is the fastest way to get rate-limited into
a broken site.

So the pages are built once a day and served as files. That fixes three problems
at once:

**Rate limiting.** The upstream endpoint is touched once per instrument per day
by a build job, never by a visitor.

**Latency.** A page that took several seconds to compute loads instantly.

**Hosting.** Static files need no server, no runtime, no scaling and no cost.

The trade-off is that only the pre-built universe is browsable. Anyone wanting an
arbitrary ticker runs ``quantos serve`` locally, which is the same renderer
against a live fetch. See ``docs/ddr/DDR-005-static-site.md``.

Output
------
    index.html            search over the universe, resolved client-side
    <TICKER>.html         one page per instrument
    methodology.html      what every number means and how it is computed
    universe.json         the index the search box filters
    manifest.webmanifest  installable as an app
    sw.js                 offline support
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - script convenience
    sys.path.insert(0, str(ROOT / "src"))

#: Spread across asset classes so the site demonstrates that the same analysis
#: behaves differently on equities, bonds, commodities and crypto.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    # Broad equity
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "EFA",
    "EEM",
    # Sectors
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLU",
    "XLP",
    # Large caps
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "V",
    "JNJ",
    "WMT",
    "XOM",
    "KO",
    "PG",
    "DIS",
    "NFLX",
    "AMD",
    # Rates and credit
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    # Commodities and currency
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "UUP",
    # Crypto
    "BTC-USD",
    "ETH-USD",
    # Volatility
    "^VIX",
    "^GSPC",
)


from quantos.web.server import DISCLAIMER  # noqa: E402 - needs the sys.path line above


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _shell(title: str, body: str, *, style: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Quantitative research on any listed security.">
<link rel="manifest" href="manifest.webmanifest">
<meta name="theme-color" content="#0d1117">
<title>{title}</title><style>{style}</style></head><body><div class="wrap">
{body}
</div>
<script>
if ('serviceWorker' in navigator) {{
  window.addEventListener('load', function () {{
    navigator.serviceWorker.register('sw.js').catch(function () {{}});
  }});
}}
</script>
</body></html>"""


def build_index(entries: list[dict], style: str) -> str:
    """The landing page: a search box over the pre-built universe."""
    from quantos.web.server import _search_form

    rows = "".join(
        f'<a class="row" href="{e["ticker"]}.html" data-search="{e["ticker"]} {e["name"]}">'
        f'<span class="tk">{e["ticker"]}</span>'
        f'<span class="nm">{e["name"]}</span>'
        f'<span class="cl">{e["asset_class"]}</span>'
        f'<span class="vol">{e["annualised_volatility"]:.0%}</span>'
        f'<span class="dd">{e["median_worst_drawdown"]:.0%}</span></a>'
        for e in entries
    )
    body = f"""
<h1>QuantOS</h1>
<p class="sub">Quantitative research on any listed security — return distribution,
tail risk, volatility forecasting, factor exposures, a pre-registered signal
battery corrected for multiple testing, and a simulated forward distribution whose
probabilities have been calibration-tested.</p>

{_search_form(link_style="static")}

<p class="hint">Type any ticker, or pick from the {len(entries)} rebuilt daily below.
Columns are annualised volatility and the median worst drawdown over the next 160
trading days — a coin flip that the drawdown is at least that deep.</p>

<input type="text" id="filter" placeholder="Filter this list…" aria-label="Filter"
       oninput="filterRows(this.value)" class="filter">

<div class="table" id="rows">
  <div class="row head"><span class="tk">Ticker</span><span class="nm">Name</span>
    <span class="cl">Class</span><span class="vol">Vol</span><span class="dd">Drawdown</span></div>
  {rows}
</div>
<p class="hint" id="empty" style="display:none">No match in the pre-built universe.
Press Enter in the search box above to try it anyway — it will exist if the last
build covered it.</p>

<h2>What this answers well</h2>
<div class="panel"><ul>
<li><b>How risky is this, really?</b> Fat-tailed VaR and CVaR, and the probability of
    <i>touching</i> a level — the number a stop-loss actually responds to — rather than
    only the chance of finishing beyond it.</li>
<li><b>How volatile will it be next month?</b> Volatility clusters and is genuinely
    forecastable. This is the part of the analysis with real predictive content.</li>
<li><b>How different is shorting?</b> Unbounded loss, borrow cost, and a skewed
    distribution. The same forecast implies different risk on the two sides.</li>
</ul></div>

<h2>What it will not tell you</h2>
<div class="panel"><ul>
<li><b>Where the price is going.</b> Direction is not reliably forecastable from past
    prices. Every page says so, and shows the coin-flip probability rather than
    manufacturing a target.</li>
<li><b>Which rare events are coming.</b> Calibration testing found real skill for
    ordinary moves and <b>none</b> for rare ones. Both results are published.</li>
</ul></div>

<footer><b>{DISCLAIMER}</b><br><br>
Built {_now()} · <a href="methodology.html">Methodology</a> ·
<a href="https://github.com/danielboulan10/QuantOS">Source</a></footer>

<script>
function filterRows(q) {{
  q = (q || '').trim().toLowerCase();
  var shown = 0;
  document.querySelectorAll('#rows .row:not(.head)').forEach(function (row) {{
    var hit = !q || row.dataset.search.toLowerCase().indexOf(q) !== -1;
    row.style.display = hit ? '' : 'none';
    if (hit) shown++;
  }});
  document.getElementById('empty').style.display = shown ? 'none' : '';
}}
</script>"""
    return _shell("QuantOS", body, style=style)


def build_methodology(style: str) -> str:
    body = f"""
<h1>Methodology</h1>
<p class="sub">What each number is, how it is computed, and how far it can be trusted.
Every claim below links to the code that produces it.</p>
<p><a href="index.html">&larr; Back to search</a></p>

<h2>Prices</h2>
<div class="panel">
Daily bars, <b>total-return adjusted</b> for dividends. This is not cosmetic: over ten
years Verizon shows a (1.85)% annual return on raw prices and 3.59% adjusted — the
sign of the conclusion flips. Berkshire, which pays no dividend, shows a gap of
exactly zero, which is the control that confirms the mechanism.
</div>

<h2>Risk statistics</h2>
<div class="panel">
Volatility, Sharpe, drawdown, VaR and CVaR are computed on the full available history.
The Sharpe ratio is always shown with its standard error, because a Sharpe of 0.5 over
ten years is about 1.6 standard errors from zero and reporting it alone invites the
wrong conclusion. The Hill tail index measures how fat the tails are; equities sit
around 3, which is why a Gaussian VaR understates losses.
</div>

<h2>Volatility forecasting</h2>
<div class="panel">
GARCH(1,1) fitted by maximum likelihood, but only after an Engle ARCH-LM test confirms
there is clustering to model. Fitting GARCH to a series without ARCH effects yields
persistence near 1.0 and half-lives of centuries, so the test is a gate rather than a
diagnostic. Volatility clustering is real and forecastable, and this is the part of the
analysis with genuine predictive content.
</div>

<h2>The forward distribution</h2>
<div class="panel">
20,000 simulated paths from the fitted model with Student-t innovations, plus a
model-free block bootstrap as a cross-check. Paths are <b>driftless by design</b>:
expected return cannot be estimated from price history to any useful precision over
these horizons, and a noisy drift estimate would turn every directional probability
into a function of that noise. That is why the probability of finishing higher is
always near 50% — which is the honest answer, not a missing feature.
<br><br>
Probabilities of <i>touching</i> a level are always at least as large as probabilities of
<i>finishing</i> beyond it, because a path can breach and recover. The touch column is
the one that matters for a stop.
</div>

<h2>Calibration — how much to trust the probabilities</h2>
<div class="panel">
A probability that has not been calibration-tested is a number with a percent sign on
it. Walk-forward over 20 years of SPY, with every model fitted only on data preceding
its own forecast:
<br><br>
<table>
<tr><th>Event</th><th>Base rate</th><th>Brier skill</th><th>Verdict</th></tr>
<tr><td>5% drawdown within a month</td><td>22.5%</td><td>+0.041</td>
    <td class="good">calibrated, real skill</td></tr>
<tr><td>Touch -5% within a month</td><td>15.1%</td><td>+0.026</td>
    <td class="good">calibrated, real skill</td></tr>
<tr><td>10% drawdown within a month</td><td>4.5%</td><td class="bad">-0.067</td>
    <td class="bad">no skill over the base rate</td></tr>
</table>
<br>
Volatility clustering predicts ordinary moves, not crashes. Treat the moderate-event
probabilities as informative and the rare-event ones as unbiased but uninformative.
Both results are published because publishing only the first would be dishonest.
</div>

<h2>The signal battery</h2>
<div class="panel">
Nine pre-registered signals, each judged only after correcting for the fact that nine
were tried — deflated Sharpe ratios and Hansen's SPA. On most liquid instruments
nothing survives, which is the expected outcome and is reported as a result rather
than hidden. A separate <a
href="https://github.com/danielboulan10/QuantOS/blob/main/forward/RECORD.md">live
forward record</a> logs predictions before their outcomes exist and scores them later,
with overlapping forecasts discounted to their independent count.
</div>

<h2>Known limitations</h2>
<div class="panel"><ul>
<li>Data comes from a public but undocumented endpoint and can break or be wrong.</li>
<li>Only the pre-built universe is browsable here; run <code>quantos serve</code>
    locally for arbitrary tickers.</li>
<li>Simulations cannot produce a regime the history does not contain.</li>
<li>No transaction costs, no borrow availability, no capacity, no taxes.</li>
</ul></div>

<footer><b>{DISCLAIMER}</b><br><br>
<a href="index.html">Back to search</a> ·
<a href="https://github.com/danielboulan10/QuantOS">Source</a></footer>"""
    return _shell("QuantOS — Methodology", body, style=style)


EXTRA_STYLE = """
.filter { width:100%; padding:10px 14px; font-size:15px; border-radius:8px;
  border:1px solid var(--line); background:var(--panel); color:var(--ink); margin:6px 0 14px; }
.table { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.row { display:grid; grid-template-columns:5.5rem 1fr 5rem 4rem 5.5rem; gap:10px;
  padding:9px 14px; border-bottom:1px solid var(--line); text-decoration:none;
  color:var(--ink); font-size:14px; align-items:center; }
.row:last-child { border-bottom:0; }
.row:hover { background:var(--panel); }
.row.head { color:var(--muted); font-size:12px; text-transform:uppercase;
  letter-spacing:.05em; font-weight:600; }
.row .tk { font-weight:650; }
.row .nm { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.row .cl { color:var(--muted); font-size:12px; }
.row .vol, .row .dd { text-align:right; font-variant-numeric:tabular-nums; }
@media (max-width:640px) {
  .row { grid-template-columns:4.5rem 1fr 4.5rem; }
  .row .cl, .row .vol { display:none; }
}
"""

SERVICE_WORKER = """// Cache-first for the shell, network-first for pages, so an installed
// copy keeps working offline and still updates when the daily build lands.
const CACHE = 'quantos-v1';
const SHELL = ['index.html', 'methodology.html', 'manifest.webmanifest'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((c) => c.put(event.request, copy)).catch(() => {});
        return response;
      })
      .catch(() => caches.match(event.request).then((hit) => hit || caches.match('index.html')))
  );
});
"""


NOT_FOUND = r"""
<h1>Page not found</h1>
<p class="sub">That address does not exist on this site. GitHub Pages paths are
<b>case-sensitive</b>, so <code>spy.html</code> is not the same file as
<code>SPY.html</code>.</p>
<p class="hint" id="guess"></p>
<p><a href="index.html">&larr; Back to the search page</a></p>
<script>
// Most 404s here are a ticker typed in the wrong case, or without the .html
// suffix. Both are recoverable: normalise and offer the corrected address
// rather than leaving the visitor at a dead end.
(function () {
  var path = location.pathname.split('/').pop() || '';
  var guess = path.replace(/\.html$/i, '').toUpperCase();
  if (!guess) return;
  var target = encodeURIComponent(guess) + '.html';
  document.getElementById('guess').innerHTML =
    'Did you mean <a href="' + target + '"><code>' + guess + '</code></a>? '
    + 'Redirecting there in a moment&hellip;';
  setTimeout(function () { location.replace(target); }, 2500);
})();
</script>
"""


def manifest() -> str:
    return json.dumps(
        {
            "name": "QuantOS",
            "short_name": "QuantOS",
            "description": "Quantitative research on any listed security.",
            "start_url": "index.html",
            "scope": ".",
            "display": "standalone",
            "background_color": "#0d1117",
            "theme_color": "#0d1117",
            "icons": [],
        },
        indent=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "site"))
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N")
    args = parser.parse_args()

    from quantos.web.server import _STYLE, render_page

    universe = tuple(args.tickers) if args.tickers else DEFAULT_UNIVERSE
    if args.limit:
        universe = universe[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style = _STYLE + EXTRA_STYLE

    entries: list[dict] = []
    failures: list[str] = []
    started = time.perf_counter()

    for index, ticker in enumerate(universe, 1):
        try:
            rendered = render_page(ticker, link_style="static")
            if rendered.status != 200:
                failures.append(ticker)
                print(
                    f"  [{index:2d}/{len(universe)}] {ticker:9s} skipped (status {rendered.status})"
                )
                continue
            (out / f"{ticker}.html").write_text(rendered.body, encoding="utf-8")

            import numpy as np

            from quantos.data.market import fetch_prices
            from quantos.forecast import probability_report, simulate_garch_paths

            price, info = fetch_prices(ticker, start="2015-01-01")
            returns = np.diff(np.log(price.prices))
            ensemble = simulate_garch_paths(returns, float(price.prices[-1]), 160, n_paths=4000)
            report = probability_report(ensemble, symbol=ticker)
            entries.append(
                {
                    "ticker": ticker,
                    "name": info.name or ticker,
                    "asset_class": info.asset_class,
                    "annualised_volatility": float(np.std(returns, ddof=1) * np.sqrt(252)),
                    "median_worst_drawdown": report.median_worst_drawdown,
                    "last_price": float(price.prices[-1]),
                    "as_of": price.end,
                }
            )
            print(f"  [{index:2d}/{len(universe)}] {ticker:9s} ok")
        except Exception as error:
            failures.append(ticker)
            print(f"  [{index:2d}/{len(universe)}] {ticker:9s} FAILED — {error}")

    if not entries:
        print("no pages were built; refusing to publish an empty site")
        return 1

    entries.sort(key=lambda e: e["ticker"])
    (out / "index.html").write_text(build_index(entries, style), encoding="utf-8")
    (out / "methodology.html").write_text(build_methodology(style), encoding="utf-8")
    (out / "universe.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    (out / "manifest.webmanifest").write_text(manifest(), encoding="utf-8")
    (out / "sw.js").write_text(SERVICE_WORKER, encoding="utf-8")
    # GitHub Pages serves this for any unmatched path under the site root, so a
    # mistyped ticker recovers instead of dead-ending on GitHub's generic page.
    (out / "404.html").write_text(
        _shell("QuantOS — page not found", NOT_FOUND, style=style), encoding="utf-8"
    )
    # Tell GitHub Pages not to run Jekyll over the output.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    elapsed = time.perf_counter() - started
    total = sum(f.stat().st_size for f in out.glob("*")) / 1e6
    print(f"\nbuilt {len(entries)} pages in {elapsed:.0f}s, {total:.1f} MB -> {out}")
    if failures:
        print(f"  {len(failures)} failed: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
