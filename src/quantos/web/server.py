"""A search bar for the research pipeline: type a ticker, get the analysis.

Why this exists
---------------
The whole pipeline was reachable only from a terminal, which meant that showing
someone what it does required them to install it first. This is the same analysis
behind a text box.

Why it is stdlib only
---------------------
``http.server`` rather than Flask or FastAPI, and hand-written HTML rather than a
template engine, because DDR-002 fixes the runtime dependencies at NumPy alone
and a web view is not a good enough reason to break that. Charts are the
repository's own SVG renderer, inlined into the page, so there is no JavaScript
charting library and no network request from the browser.

What this is not
----------------
Not a production web service. It binds to localhost by default, runs one request
at a time, and has no authentication -- it is a local viewer for a research tool,
and the docstring says so rather than leaving someone to discover it by
deploying it.
"""

from __future__ import annotations

import html
import http.server
import json
import socketserver
import traceback
import urllib.parse
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["render_landing", "render_page", "serve"]

_STYLE = """
:root {
  --bg:#0d1117; --panel:#161b22; --line:#30363d; --ink:#e6edf3;
  --muted:#8b949e; --accent:#58a6ff; --good:#3fb950; --warn:#d29922; --bad:#f85149;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de; --ink:#1f2328;
          --muted:#59636e; --accent:#0969da; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:28px 20px 80px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em;
     color:var(--muted); margin:34px 0 12px; font-weight:600; }
a { color:var(--accent); }
.sub { color:var(--muted); margin:0 0 22px; font-size:14px; }
form { display:flex; gap:10px; margin:0 0 8px; }
input[type=text] { flex:1; padding:13px 16px; font-size:17px; border-radius:8px;
  border:1px solid var(--line); background:var(--panel); color:var(--ink); }
input[type=text]:focus { outline:2px solid var(--accent); outline-offset:-1px; }
button { padding:13px 22px; font-size:15px; font-weight:600; border-radius:8px;
  border:0; background:var(--accent); color:#fff; cursor:pointer; }
.hint { color:var(--muted); font-size:13px; margin:0 0 26px; }
.hint code { background:var(--panel); padding:2px 6px; border-radius:4px;
  border:1px solid var(--line); cursor:pointer; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
        padding:14px 16px; }
.card .k { color:var(--muted); font-size:11px; text-transform:uppercase;
           letter-spacing:.06em; margin-bottom:6px; }
.card .v { font-size:21px; font-weight:650; font-variant-numeric:tabular-nums; }
.card .n { color:var(--muted); font-size:12px; margin-top:4px; }
table { width:100%; border-collapse:collapse; font-size:14px;
        font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); }
th:first-child,td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
     letter-spacing:.05em; }
.scroll { overflow-x:auto; }
.good{color:var(--good)} .warn{color:var(--warn)} .bad{color:var(--bad)} .mut{color:var(--muted)}
.panel { background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:16px 18px; }
.panel li { margin:5px 0; }
.chart { background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:10px; overflow-x:auto; }
.chart svg { max-width:100%; height:auto; display:block; }
.err { border-left:3px solid var(--bad); }
footer { margin-top:44px; padding-top:18px; border-top:1px solid var(--line);
         color:var(--muted); font-size:13px; }
"""

_SEARCH_FORM = """
<form action="/research" method="get" autocomplete="off">
  <input type="text" name="ticker" placeholder="Ticker — AAPL, SPY, ^GSPC, BTC-USD, VOD.L"
         value="{value}" autofocus aria-label="Ticker symbol">
  <button type="submit">Research</button>
</form>
<p class="hint">Try
  <code onclick="go('NVDA')">NVDA</code>
  <code onclick="go('SPY')">SPY</code>
  <code onclick="go('^GSPC')">^GSPC</code>
  <code onclick="go('BTC-USD')">BTC-USD</code>
  <code onclick="go('VOD.L')">VOD.L</code>
  &nbsp;· no API key required · daily bars, delayed, total-return adjusted</p>
<script>function go(t){location.href='/research?ticker='+encodeURIComponent(t);}</script>
"""


def _search_form(value: str = "") -> str:
    """Render the search box.

    A plain ``replace`` rather than ``str.format``: the inline script contains
    braces, which ``format`` would try to read as replacement fields.
    """
    return _SEARCH_FORM.replace("{value}", html.escape(value))


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}%}"


def _num(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:,.{digits}f}"


def _card(key: str, value: str, note: str = "", css: str = "") -> str:
    note_html = f'<div class="n">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="card"><div class="k">{html.escape(key)}</div>'
        f'<div class="v {css}">{value}</div>{note_html}</div>'
    )


@dataclass
class _Rendered:
    status: int
    body: str


def render_landing(message: str = "") -> str:
    """The search page."""
    banner = (
        f'<div class="panel err" style="margin-bottom:20px">{html.escape(message)}</div>'
        if message
        else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantOS — quantitative research</title><style>{_STYLE}</style></head><body><div class="wrap">
<h1>QuantOS</h1>
<p class="sub">Type a ticker. Get the full quantitative research report — return
distribution, tail risk, GARCH volatility dynamics and forecast, regimes, factor
exposures, and a pre-registered signal battery corrected for multiple testing.</p>
{banner}
{_search_form()}
<h2>What this answers well</h2>
<div class="panel"><ul>
<li><b>How risky is this, really?</b> Fat-tailed VaR and CVaR, not a Gaussian approximation.</li>
<li><b>How volatile will it be next month?</b> Volatility genuinely is forecastable, and the
    GARCH forecast is the part of this report with real predictive content.</li>
<li><b>What drives it?</b> Factor betas with HAC standard errors.</li>
<li><b>Does any standard signal work on it?</b> Nine of them, judged after correcting for
    the fact that nine were tried.</li>
</ul></div>
<h2>What it will not tell you</h2>
<div class="panel"><ul>
<li><b>Where the price is going.</b> Direction is not reliably forecastable from past prices,
    and this tool reports that rather than manufacturing a target.</li>
<li>Anything not in the price: earnings quality, fundamentals, borrow, capacity.</li>
</ul></div>
<footer>Research tool, not investment advice. Data is delayed daily bars from a public
endpoint. <a href="https://github.com/danielboulan10/QuantOS">Source</a>.</footer>
</div></body></html>"""


def _as_years(dates: Any) -> np.ndarray:
    """Dates as fractional calendar years, for a readable axis.

    The chart renderer takes floats, and passing raw ``datetime64[D]`` values
    labels the axis with day counts since the epoch -- "17,036" rather than
    "2016". Fractional years are the smallest change that makes the axis mean
    something to a reader.
    """
    days = np.asarray(dates, dtype="datetime64[D]").astype(float)
    return 1970.0 + days / 365.2425


def _price_chart(dates: Any, prices: Any, symbol: str) -> str:
    from quantos.viz.svg import line_chart

    figure = line_chart(
        {symbol: (_as_years(dates), np.asarray(prices, dtype=float))},
        title=f"{symbol} — total-return adjusted",
        x_label="year",
        y_label="price",
        x_tick_style="year",
    )
    return str(figure.render())


def _volatility_history(dates: Any, prices: Any, report: Any) -> str:
    """Rolling realised volatility, with the GARCH forecast as a reference line.

    An earlier version plotted the full-sample volatility and the forecast as two
    constants against an invented x-axis, which drew two flat lines and told the
    reader nothing. What is actually informative is where volatility has been:
    the forecast only means something relative to that history.
    """
    from quantos.viz.svg import line_chart

    prices = np.asarray(prices, dtype=float)
    if prices.size < 90:
        return ""

    window = 21
    returns = np.diff(np.log(prices))
    # Rolling standard deviation via cumulative sums: O(n) rather than O(n*window).
    squared = np.concatenate([[0.0], np.cumsum(returns**2)])
    running = np.concatenate([[0.0], np.cumsum(returns)])
    count = float(window)
    mean_square = (squared[window:] - squared[:-window]) / count
    mean = (running[window:] - running[:-window]) / count
    rolling = np.sqrt(np.maximum(mean_square - mean**2, 0.0)) * np.sqrt(252.0)

    x = _as_years(dates)[window:]
    series = {"realised, 21-day rolling": (x, rolling)}
    if np.isfinite(report.volatility_forecast_21d):
        series["GARCH forecast for the next 21 days"] = (
            np.array([x[0], x[-1]]),
            np.full(2, report.volatility_forecast_21d),
        )
    figure = line_chart(
        series,
        title="realised volatility, and what is forecast next",
        x_label="year",
        y_label="annualised",
        x_tick_style="year",
    )
    return str(figure.render())


def _fan_chart(dates: Any, prices: Any, ensemble: Any, symbol: str) -> str:
    """400 bars of history, then the simulated forward distribution."""
    from quantos.viz.svg import fan_chart

    history_len = min(400, len(prices))
    hx = _as_years(dates)[-history_len:]
    hy = np.asarray(prices, dtype=float)[-history_len:]
    step = float(np.median(np.diff(hx))) if hx.size > 1 else 1 / 252
    fx = hx[-1] + step * np.arange(ensemble.horizon + 1)

    return str(
        fan_chart(
            hx,
            hy,
            fx,
            ensemble.quantile_bands((0.05, 0.25, 0.50, 0.75, 0.95)),
            title=f"{symbol} — {history_len} bars of history, {ensemble.horizon} simulated forward",
            x_label="year",
            y_label="price",
            x_tick_style="year",
        ).render()
    )


def _forecast_section(price: Any, report: Any, info: Any) -> str:
    """The forward-looking half: fan chart, probabilities, long vs short."""
    from quantos.forecast import (
        long_short_comparison,
        probability_report,
        simulate_garch_paths,
    )

    returns = np.diff(np.log(np.asarray(price.prices, dtype=float)))
    if returns.size < 200:
        return ""

    horizon = 160
    ensemble = simulate_garch_paths(returns, float(price.prices[-1]), horizon, n_paths=20_000)
    probabilities = probability_report(ensemble, symbol=info.ticker)
    sides = long_short_comparison(ensemble, symbol=info.ticker)

    def rows(mapping: dict[str, float]) -> str:
        return "".join(
            f"<tr><td>{html.escape(label)}</td><td>{value:.1%}</td></tr>"
            for label, value in mapping.items()
        )

    chart = _fan_chart(price.dates, price.prices, ensemble, info.ticker)

    return f"""
<h2>Forward distribution — {horizon} trading days</h2>
<div class="chart">{chart}</div>
<div class="panel" style="margin-top:14px">
<b>{html.escape(probabilities.direction_verdict)}</b><br><br>
{html.escape(probabilities.risk_verdict)}
</div>

<h2>Probabilities that mean something</h2>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px">
  <div>
    <h3 style="font-size:13px;color:var(--muted)">Where it might finish</h3>
    <table>{rows(probabilities.terminal_thresholds)}</table>
  </div>
  <div>
    <h3 style="font-size:13px;color:var(--muted)">What it might touch on the way</h3>
    <table>{rows(probabilities.touch_thresholds)}</table>
  </div>
  <div>
    <h3 style="font-size:13px;color:var(--muted)">Drawdown within the period</h3>
    <table>{rows(probabilities.drawdowns)}</table>
  </div>
</div>
<p class="hint">Touching a level is always more likely than finishing beyond it —
a path can dip and recover. The touch column is the one a stop-loss responds to.</p>

<h2>Buying versus shorting — the same forecast, both sides</h2>
<div class="scroll"><table>
<tr><th>Measure</th><th>Long</th><th>Short</th></tr>
<tr><td>probability of profit</td><td>{sides.long_probability_of_profit:.1%}</td>
    <td>{sides.short_probability_of_profit:.1%}</td></tr>
<tr><td>average of the worst 5%</td><td>{sides.long_expected_shortfall_95:.1%}</td>
    <td>{sides.short_expected_shortfall_95:.1%}</td></tr>
<tr><td>worst simulated outcome</td><td>{sides.long_worst_case:.1%}</td>
    <td>{sides.short_worst_case:.1%}</td></tr>
<tr><td>chance an {sides.stop_distance:.0%} stop is hit</td>
    <td>{sides.long_stop_hit_probability:.1%}</td>
    <td>{sides.short_stop_hit_probability:.1%}</td></tr>
</table></div>
<div class="panel" style="margin-top:12px">{html.escape(sides.asymmetry_verdict)}</div>

<div class="panel" style="margin-top:14px">
<b>How much to trust these.</b> They come from {ensemble.n_paths:,} simulated paths
({html.escape(str(ensemble.assumptions.get("engine", "")))}), driftless by design —
expected return cannot be estimated precisely enough over this horizon to justify
tilting them. Calibration was tested on 20 years of SPY: probabilities of moderate
events (a 5% drawdown in a month) came back <b>calibrated with positive skill</b>,
while rare-event probabilities (a 10% drawdown in a month) were unbiased on average
but carried <b>no skill over the base rate</b> — volatility clustering predicts
ordinary moves, not crashes. Read the 5% rows with more confidence than the 20% rows.
</div>
"""


def render_page(ticker: str) -> _Rendered:
    """Run the pipeline for one ticker and render it."""
    from quantos.data.market import MarketDataError, fetch_prices
    from quantos.research.instruments import AssetClass, Instrument
    from quantos.research.report import generate_report

    try:
        price, info = fetch_prices(ticker, start="2015-01-01")
    except MarketDataError as error:
        return _Rendered(404, render_landing(f"{ticker}: {error}"))

    if len(price) < 260:
        return _Rendered(
            400,
            render_landing(
                f"{info.ticker} has only {len(price)} daily bars; at least ~260 "
                "are needed before these statistics mean anything."
            ),
        )

    instrument = Instrument(
        symbol=info.ticker,
        name=info.name,
        asset_class=AssetClass(info.asset_class),
        dates=price.dates,
        prices=price.prices,
        source=price.source,
        dividend_adjusted=True,
        average_daily_volume=(
            float(np.nanmean(price.volume)) if price.volume is not None else None
        ),
    )
    from quantos.data.factors import build_factors

    factor_set = build_factors(price.dates, price.prices, start="2015-01-01")
    report = generate_report(
        instrument,
        factors=factor_set.columns if factor_set.usable else None,
        run_signals=True,
    )

    sharpe_class = "good" if abs(report.sharpe) > 2 * report.sharpe_standard_error else "mut"
    sharpe_note = (
        "distinguishable from zero"
        if abs(report.sharpe) > 2 * report.sharpe_standard_error
        else "NOT distinguishable from zero"
    )

    cards = "".join(
        [
            _card("Latest", f"{price.prices[-1]:,.2f}", f"{info.currency} · {price.end}"),
            _card("Annualised vol", _pct(report.annualised_volatility)),
            _card(
                "Sharpe",
                _num(report.sharpe, 2),
                f"± {report.sharpe_standard_error:.2f} — {sharpe_note}",
                sharpe_class,
            ),
            _card(
                "Max drawdown",
                _pct(report.max_drawdown),
                f"{report.max_drawdown_days} days under",
                "bad",
            ),
            _card("CVaR 99%", _pct(report.cvar_99), "average loss beyond VaR", "warn"),
            _card("Tail index", _num(report.tail_index, 2), "equities ~3"),
        ]
    )

    # -- the forward-looking section, kept deliberately narrow ---------------- #
    forecast_rows = ""
    if np.isfinite(report.volatility_forecast_21d):
        spot = float(price.prices[-1])
        sigma_21 = report.volatility_forecast_21d * np.sqrt(21.0 / 252.0)
        for label, z in (("68% (1 sd)", 1.0), ("95% (2 sd)", 1.96)):
            low, high = spot * np.exp(-z * sigma_21), spot * np.exp(z * sigma_21)
            forecast_rows += (
                f"<tr><td>{label}</td><td>{low:,.2f}</td><td>{high:,.2f}</td>"
                f"<td>±{z * sigma_21:.1%}</td></tr>"
            )

    vol_chart = _volatility_history(price.dates, price.prices, report)
    try:
        forward_html = _forecast_section(price, report, info)
    except Exception:
        traceback.print_exc()
        forward_html = (
            '<div class="panel err">The forward distribution could not be simulated '
            "for this instrument; the historical analysis below is unaffected.</div>"
        )

    forecast_html = f"""
<h2>What can actually be forecast</h2>
<div class="cards">
  {_card("Vol forecast, 1 day", _pct(report.volatility_forecast_1d))}
  {_card("Vol forecast, 21 days", _pct(report.volatility_forecast_21d))}
  {_card("Shock half-life", _num(report.garch_half_life, 1) + " d", "how long a spike persists")}
  {_card("Vol percentile", _pct(report.current_vol_percentile, 0), "versus its own history")}
</div>
<div class="chart" style="margin-top:14px">{vol_chart}</div>
<h2>Implied 21-day range, from the volatility forecast</h2>
<div class="scroll"><table>
<tr><th>Confidence</th><th>Low</th><th>High</th><th>Move</th></tr>
{forecast_rows}
</table></div>
<div class="panel" style="margin-top:14px">
<b>Read this correctly.</b> These are ranges implied by forecast <i>volatility</i>,
centred on today's price — not a prediction of direction. Volatility is genuinely
forecastable: it clusters, and the GARCH persistence of
{_num(report.garch_persistence, 4)} with a {_num(report.garch_half_life, 1)}-day
half-life is that predictability measured on this instrument's own history.
Direction is not, on this evidence — see the signal battery below, where every
signal is judged only after correcting for the fact that several were tried.
</div>"""

    signals_html = ""
    if report.signals and report.signals.results:
        rows = ""
        for result in report.signals.sorted_by_evidence():
            css = "good" if result.is_significant else "mut"
            rows += (
                f"<tr><td>{html.escape(result.name)}</td>"
                f"<td>{_num(result.in_sample_sharpe, 3)}</td>"
                f"<td>{_num(result.deflated_p_value, 3)}</td>"
                f"<td>{_num(result.out_of_sample_sharpe, 3)}</td>"
                f'<td class="{css}">{html.escape(result.verdict)}</td></tr>'
            )
        verdict = (
            "At least one signal survived every correction."
            if report.signals.any_significant
            else "Nothing survives correction for multiple testing — the usual and "
            "expected outcome on a liquid instrument, and a result rather than a failure."
        )
        signals_html = f"""
<h2>Signal battery — {len(report.signals.results)} pre-registered signals</h2>
<div class="scroll"><table>
<tr><th>Signal</th><th>IS Sharpe</th><th>Deflated p</th><th>OOS Sharpe</th><th>Verdict</th></tr>
{rows}</table></div>
<div class="panel" style="margin-top:12px">Hansen SPA over the battery:
<b>p = {_num(report.signals.spa_p_value, 4)}</b>. {verdict}</div>"""

    factors_html = ""
    if report.factors:
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{_num(beta, 4)}</td><td>{_num(t, 2)}</td></tr>"
            for name, beta, t in zip(
                report.factors.factor_names,
                report.factors.betas,
                report.factors.t_statistics,
                strict=True,
            )
        )
        factors_html = f"""
<h2>Factor exposure (HAC standard errors)</h2>
<div class="scroll"><table><tr><th>Factor</th><th>Beta</th><th>t</th></tr>{rows}
<tr><td>alpha (annual)</td><td>{_pct(report.factors.alpha_annualised)}</td>
<td>{_num(report.factors.alpha_t_statistic, 2)}</td></tr></table></div>
<p class="hint">R² {_num(report.factors.r_squared, 3)} —
{_pct(report.factors.idiosyncratic_share)} idiosyncratic.</p>"""

    notes = "".join(f"<li>{html.escape(note)}</li>" for note in report.notes)
    skipped = "".join(
        f"<li><b>{html.escape(k)}</b> skipped — {html.escape(v)}</li>"
        for k, v in report.skipped.items()
    )

    heading = html.escape(info.name or info.ticker)
    price_chart = _price_chart(price.dates, price.prices, info.ticker)

    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(info.ticker)} — QuantOS</title><style>{_STYLE}</style></head>
<body><div class="wrap">
<h1>{heading} <span class="mut">({html.escape(info.ticker)})</span></h1>
<p class="sub">{html.escape(info.asset_class)} · {html.escape(info.exchange)} ·
{len(price):,} daily bars, {price.start} to {price.end} · {html.escape(price.price_column)}</p>
{_search_form(info.ticker)}
<div class="cards">{cards}</div>
<div class="chart" style="margin-top:18px">{price_chart}</div>
{forecast_html}
{forward_html}
{factors_html}
{signals_html}
<h2>What the data supports</h2>
<div class="panel"><ul>{notes or "<li>--</li>"}{skipped}</ul></div>
<footer>Generated {report.generated} from {html.escape(price.source)}.
Every number is descriptive of the past except the volatility forecast, which is
explicitly labelled. Research tool, not investment advice.
<a href="/">New search</a> ·
<a href="https://github.com/danielboulan10/QuantOS">Source</a></footer>
</div></body></html>"""
    return _Rendered(200, body)


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "QuantOS"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"  {self.address_string()} {format % args}")

    def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._send(200, render_landing())
            return
        if parsed.path == "/healthz":
            self._send(200, json.dumps({"ok": True}), "application/json")
            return
        if parsed.path != "/research":
            self._send(404, render_landing("No such page."))
            return

        ticker = (query.get("ticker") or [""])[0].strip()
        if not ticker:
            self._send(200, render_landing("Enter a ticker symbol."))
            return
        # Tickers are alphanumerics plus a handful of separators. Rejecting the
        # rest keeps arbitrary strings out of the fetch path and the cache
        # filename, and gives a clearer message than a downstream 404.
        if len(ticker) > 24 or not all(c.isalnum() or c in "^.-=" for c in ticker):
            self._send(400, render_landing(f"{ticker!r} does not look like a ticker symbol."))
            return

        try:
            rendered = render_page(ticker)
            self._send(rendered.status, rendered.body)
        except Exception:
            traceback.print_exc()
            self._send(
                500,
                render_landing(
                    f"Something failed while analysing {ticker}. The traceback is in "
                    "the terminal running the server."
                ),
            )


class _Server(socketserver.TCPServer):
    allow_reuse_address = True


def serve(host: str = "127.0.0.1", port: int = 8000, *, open_browser: bool = True) -> int:
    """Run the local research viewer.

    Binds to localhost by default. There is no authentication and requests are
    handled one at a time, so this is a local viewer rather than a deployable
    service; binding it to a public interface would expose an unauthenticated
    endpoint that makes outbound requests on a caller's behalf.
    """
    with _Server((host, port), _Handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"QuantOS research viewer on {url}")
        print("  type a ticker in the search bar; Ctrl-C to stop")
        if open_browser:
            try:
                import webbrowser

                webbrowser.open(url)
            except Exception:  # pragma: no cover - headless environments
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0
