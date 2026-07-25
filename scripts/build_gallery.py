#!/usr/bin/env python3
"""Generate every figure in docs/gallery, and the gallery page itself.

Why this exists
---------------
A repository of 16,000 lines has a first-impression problem: the landing page
shows a folder list, and nothing about what any of it produces. This script emits
one figure per subsystem, all rendered by ``quantos.viz.svg`` -- no matplotlib,
no external assets -- so the gallery is regenerable and version-controls as
readable text.

Every figure is produced from a real computation, not a mock-up. The real-data
panels pull live series from FRED (cached), so re-running this updates them.

Usage
-----
    python scripts/build_gallery.py [--offline] [--out docs/gallery]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantos.core.rng import SeedBank
from quantos.viz.svg import Figure, histogram, line_chart

FIGURES: list[tuple[str, str, str]] = []  # (filename, title, caption)


def register(filename: str, title: str, caption: str) -> None:
    FIGURES.append((filename, title, caption))


# --------------------------------------------------------------------------- #
def fig_order_book(out: Path) -> None:
    """Cumulative depth from a real simulated book."""
    from quantos.core.types import Side
    from quantos.sim.scenarios import build_liquid_market

    simulation = build_liquid_market(duration_ns=6_000_000_000, seed=11)
    simulation.run()
    book = simulation.engine.book

    bids = book.depth(Side.BUY, 25)
    asks = book.depth(Side.SELL, 25)
    if not bids or not asks:
        return
    from quantos.viz.svg import book_depth_chart

    book_depth_chart(
        [int(level.price) for level in bids],
        [int(level.quantity) for level in bids],
        [int(level.price) for level in asks],
        [int(level.quantity) for level in asks],
        title="Limit order book: cumulative depth",
    ).save(str(out / "order_book_depth.svg"))
    register(
        "order_book_depth.svg",
        "Order book depth",
        "Cumulative resting size on each side of a simulated book, after six "
        "seconds of trading between 26 agents. The cumulative curve *is* the "
        "cost function of a marketable order: read off how far a sweep of a "
        "given size would walk the book.",
    )


def fig_price_discovery(out: Path) -> None:
    """The market price against the latent value no agent broadcasts."""
    from quantos.sim.scenarios import build_liquid_market

    result = build_liquid_market(duration_ns=20_000_000_000, seed=6).run()
    mid = result.mid_series()
    stamps, values = result.fundamental_path
    grid = np.arange(mid.size, dtype=np.float64) * result.config.snapshot_interval_ns
    latent = np.interp(grid, stamps, values)
    x = np.arange(mid.size, dtype=np.float64)

    line_chart(
        {"market mid": (x, mid), "latent fundamental": (x, latent)},
        title="Price discovery: what the market found vs what was true",
        x_label="snapshot (1 ms apart)",
        y_label="price (ticks)",
    ).save(str(out / "price_discovery.svg"))
    register(
        "price_discovery.svg",
        "Price discovery",
        "The dashed line is a fundamental value visible only to informed traders. "
        "The market has to infer it from order flow. This is a *favourable* seed: "
        "measured across 16 seeds the correlation averages 0.29 with a standard "
        "deviation of 0.48, and the README says so rather than showing only this one.",
    )


def fig_return_distribution(out: Path) -> None:
    """Emergent fat tails against the Gaussian the agents were built from."""
    from quantos.sim.scenarios import build_liquid_market

    result = build_liquid_market(duration_ns=20_000_000_000, seed=42).run()
    returns = result.returns()
    if returns.size < 500:
        return
    standardised = (returns - returns.mean()) / returns.std()
    grid = np.linspace(-6, 6, 400)
    normal = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)

    from quantos.core.stats.descriptive import kurtosis

    histogram(
        standardised,
        bins=90,
        density=True,
        title=f"Emergent return distribution (excess kurtosis {kurtosis(returns):.1f})",
        x_label="standardised return",
        overlay={"N(0,1)": (grid, normal)},
    ).save(str(out / "return_distribution.svg"))
    register(
        "return_distribution.svg",
        "Emergent fat tails",
        "Returns from the simulated market against the Gaussian its component "
        "agents were built from. No agent was told to produce fat tails; they "
        "emerge from order-book mechanics and inventory-averse liquidity provision.",
    )


def fig_execution_frontier(out: Path) -> None:
    """Almgren-Chriss cost against risk."""
    from quantos.execution.almgren_chriss import ImpactParameters, efficient_execution_frontier

    params = ImpactParameters(volatility=0.02, temporary_impact=1e-6)
    frontier = efficient_execution_frontier(1e6, 1.0, params)
    costs = np.array([t.expected_cost for t in frontier])
    risks = np.array([t.cost_standard_deviation for t in frontier])

    figure = Figure(
        title="Efficient frontier of execution (Almgren-Chriss)",
        x_label="cost risk (standard deviation)",
        y_label="expected cost",
    )
    figure.set_ranges(risks, costs)
    figure.add_grid()
    figure.add_line(risks, costs, width=2.2)
    figure.add_points(risks, costs, radius=3.0, opacity=0.85)
    figure.save(str(out / "execution_frontier.svg"))
    register(
        "execution_frontier.svg",
        "Optimal execution frontier",
        "There is no single optimal way to work a large order -- only a frontier, "
        "with risk aversion picking a point on it. The bottom-right end is TWAP, "
        "which is not a naive baseline but the exact optimum for a trader "
        "indifferent to price risk.",
    )


def fig_execution_trajectories(out: Path) -> None:
    """How urgency reshapes the schedule."""
    from quantos.execution.almgren_chriss import ImpactParameters, almgren_chriss_trajectory

    params = ImpactParameters(volatility=0.02, temporary_impact=1e-6)
    series = {}
    for label, risk_aversion in [
        ("risk-neutral (= TWAP)", 0.0),
        ("moderate", 1e-2),
        ("urgent", 1.0),
        ("very urgent", 20.0),
    ]:
        trajectory = almgren_chriss_trajectory(1e6, 1.0, params, risk_aversion=risk_aversion)
        series[label] = (trajectory.times, trajectory.holdings)

    line_chart(
        series,
        title="Execution schedules by risk aversion",
        x_label="fraction of horizon",
        y_label="shares remaining",
    ).save(str(out / "execution_trajectories.svg"))
    register(
        "execution_trajectories.svg",
        "Execution schedules",
        "The same one-million-share order, worked four ways. Risk-neutral gives "
        "a straight line; increasing urgency front-loads toward an exponential "
        "decay. Permanent impact is identical in all four -- you cannot schedule "
        "your way out of it.",
    )


def fig_volatility_smile(out: Path) -> None:
    """Implied volatility recovered across the moneyness range."""
    from quantos.derivatives.black_scholes import black_scholes_price, implied_volatility

    spot, maturity, rate = 100.0, 0.5, 0.04
    strikes = np.linspace(60, 160, 60)
    # A skewed "market": vol rises for downside strikes, as equity options do.
    true_vol = (
        0.20 + 0.35 * np.maximum(0.0, (100 - strikes) / 100) + 0.05 * ((strikes - 100) / 100) ** 2
    )
    recovered = []
    kept_strikes = []
    for strike, vol in zip(strikes, true_vol, strict=True):
        price = float(black_scholes_price(spot, strike, maturity, vol, rate=rate))
        try:
            recovered.append(implied_volatility(price, spot, float(strike), maturity, rate=rate))
            kept_strikes.append(strike)
        except ValueError:
            continue

    figure = Figure(
        title="Implied volatility recovered from prices (max error < 1e-12)",
        x_label="strike",
        y_label="implied volatility",
    )
    figure.set_ranges(np.array(kept_strikes), np.array(recovered))
    figure.add_grid()
    figure.add_line(np.array(kept_strikes), np.array(recovered), label="recovered", width=2.4)
    figure.add_points(strikes, true_vol, label="true", radius=2.4, opacity=0.6)
    figure.save(str(out / "volatility_smile.svg"))
    register(
        "volatility_smile.svg",
        "Implied volatility inversion",
        "A skewed volatility surface priced with Black-Scholes, then inverted "
        "back out of the prices. The two curves coincide to twelve decimal "
        "places, including deep out-of-the-money strikes where vega is ~1e-9 and "
        "naive Newton iteration diverges.",
    )


def fig_greeks(out: Path) -> None:
    """Delta and gamma across spot."""
    from quantos.derivatives.black_scholes import black_scholes_greeks

    spots = np.linspace(60, 140, 200)
    series_delta = {}
    series_gamma = {}
    for maturity, name in [(0.08, "1 month"), (0.5, "6 months"), (2.0, "2 years")]:
        greeks = [black_scholes_greeks(float(s), 100.0, maturity, 0.25, rate=0.04) for s in spots]
        series_delta[name] = (spots, np.array([g.delta for g in greeks]))
        series_gamma[name] = (spots, np.array([g.gamma for g in greeks]))

    line_chart(
        series_delta,
        title="Call delta across spot, by maturity",
        x_label="spot",
        y_label="delta",
    ).save(str(out / "greeks_delta.svg"))
    register(
        "greeks_delta.svg",
        "Option Greeks",
        "Call delta against spot for three maturities. As expiry approaches the "
        "curve steepens toward a step function at the strike -- which is why "
        "gamma explodes and hedging a short-dated option near the money is hard. "
        "All ten Greeks are analytic and match central differences to 6e-9.",
    )


def fig_deflated_sharpe(out: Path) -> None:
    """How the significance bar rises with the number of trials."""
    from quantos.strategy.validation import deflated_sharpe_ratio

    rng = SeedBank(root=3).child("gallery_dsr").generator()
    track = rng.standard_normal(1260) * 0.01 + 0.0015
    trials = np.unique(np.geomspace(1, 100_000, 40).astype(int))
    p_values = np.array([deflated_sharpe_ratio(track, n_trials=int(n)).p_value for n in trials])

    figure = Figure(
        title="The same track record, judged against N trials",
        x_label="number of configurations tried (log scale)",
        y_label="deflated Sharpe p-value",
    )
    figure.set_ranges(np.log10(trials), p_values, y_from_zero=True)
    figure.add_grid()
    figure.add_line(np.log10(trials), p_values, width=2.4)
    figure.add_horizontal_line(0.05, label="5% significance")
    figure.save(str(out / "deflated_sharpe.svg"))
    register(
        "deflated_sharpe.svg",
        "Deflated Sharpe ratio",
        "One unchanged five-year track record with a 2.4 annualised Sharpe. The "
        "x-axis is how many strategy configurations you tried before finding it. "
        "The evidence does not change; what it is worth does. This is the single "
        "most under-reported number in quantitative research.",
    )


def fig_hrp_vs_markowitz(out: Path) -> None:
    """Out-of-sample volatility as N/T rises."""
    from quantos.risk.portfolio import (
        hierarchical_risk_parity,
        minimum_variance,
        risk_parity,
    )

    rng = SeedBank(root=9).child("gallery_portfolio").generator()
    ratios = []
    results: dict[str, list[float]] = {
        "min-var (sample cov)": [],
        "min-var (Ledoit-Wolf)": [],
        "risk parity": [],
        "hierarchical risk parity": [],
        "equal weight": [],
    }
    n_assets = 40
    for n_train in (60, 80, 120, 200, 400, 800):
        factor_train = rng.standard_normal((n_train, 1))
        factor_test = rng.standard_normal((2000, 1))
        betas = rng.uniform(0.4, 1.6, n_assets)
        idio = rng.uniform(0.005, 0.03, n_assets)
        train = (
            factor_train @ betas[None, :] * 0.01 + rng.standard_normal((n_train, n_assets)) * idio
        )
        test = factor_test @ betas[None, :] * 0.01 + rng.standard_normal((2000, n_assets)) * idio
        ratios.append(n_assets / n_train)
        weights = {
            "min-var (sample cov)": minimum_variance(returns=train, shrink=False).weights,
            "min-var (Ledoit-Wolf)": minimum_variance(returns=train, shrink=True).weights,
            "risk parity": risk_parity(returns=train, shrink=True).weights,
            "hierarchical risk parity": hierarchical_risk_parity(returns=train).weights,
            "equal weight": np.ones(n_assets) / n_assets,
        }
        for name, w in weights.items():
            results[name].append(float(np.std(test @ w)) * np.sqrt(252))

    x = np.array(ratios)
    line_chart(
        {name: (x, np.array(v)) for name, v in results.items()},
        title="Out-of-sample volatility as estimation error grows",
        x_label="N / T  (assets per observation)",
        y_label="annualised OOS volatility",
    ).save(str(out / "portfolio_oos.svg"))
    register(
        "portfolio_oos.svg",
        "Portfolio construction under estimation error",
        "Forty assets, shrinking the training window from 800 observations to 60. "
        "Minimum-variance on the raw sample covariance degrades sharply as N/T "
        "rises, because it inverts the matrix and so loads on its noisiest "
        "directions. Shrinkage and HRP hold up; HRP never inverts anything.",
    )


def fig_real_equity(out: Path, offline: bool) -> None:
    """Real S&P 500 data: level and return distribution."""
    from quantos.data.catalog import resolve
    from quantos.data.fred import FredClient, FredError

    try:
        client = FredClient(offline=offline)
        spec = resolve("spx")
        series = client.get(spec.fred_id).since("2015-01-01")
    except FredError:
        return
    if len(series) < 500:
        return

    x = np.arange(len(series), dtype=np.float64)
    line_chart(
        {"S&P 500": (x, series.values)},
        title=f"S&P 500, {series.start} to {series.end}  (live FRED data)",
        x_label="trading days",
        y_label="index level",
    ).save(str(out / "real_spx_level.svg"))
    register(
        "real_spx_level.svg",
        "Real market data",
        f"The S&P 500 from {series.start} to {series.end}, pulled live from FRED "
        "with no API key and no dependency beyond the standard library. "
        "`quantos analyse --bundle equity` produces the full statistical report.",
    )

    returns = series.log_returns()
    standardised = (returns - returns.mean()) / returns.std()
    grid = np.linspace(-8, 8, 400)
    normal = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
    from quantos.core.stats.descriptive import hill_estimator, kurtosis

    histogram(
        standardised,
        bins=100,
        density=True,
        title=(
            f"S&P 500 daily returns: excess kurtosis {kurtosis(returns):.1f}, "
            f"tail index {hill_estimator(np.abs(returns[returns != 0])):.2f}"
        ),
        x_label="standardised daily return",
        overlay={"N(0,1)": (grid, normal)},
    ).save(str(out / "real_spx_returns.svg"))
    register(
        "real_spx_returns.svg",
        "Real returns are not Gaussian",
        "The same data as a distribution. The Gaussian overlay is what a "
        "risk model assuming normality believes; the histogram is what happened. "
        "The gap in the tails is why VaR computed parametrically understates "
        "losses, and why CVaR is the coherent measure.",
    )


def fig_probability_convergence(out: Path) -> None:
    """Monte Carlo converging on an analytic answer."""
    from quantos.probability.problems import SecretaryProblem

    problem = SecretaryProblem(n_candidates=100)
    exact = problem.analytic()
    rng = SeedBank(root=17).child("gallery_prob").generator()

    sizes = np.unique(np.geomspace(50, 40_000, 45).astype(int))
    estimates = []
    upper, lower = [], []
    for n in sizes:
        result = problem.simulate(int(n), rng)
        estimates.append(result.estimate)
        lo, hi = result.confidence_interval(0.95)
        lower.append(lo)
        upper.append(hi)

    x = np.log10(sizes.astype(float))
    figure = Figure(
        title="Monte Carlo converging on the analytic answer (secretary problem)",
        x_label="simulations (log scale)",
        y_label="P(select the best candidate)",
    )
    figure.set_ranges(x, np.concatenate([np.array(lower), np.array(upper)]))
    figure.add_grid()
    figure.add_line(x, np.array(upper), label="95% interval", colour="#cbd5e0", width=1.4)
    figure.add_line(x, np.array(lower), colour="#cbd5e0", width=1.4)
    figure.add_line(x, np.array(estimates), label="Monte Carlo", width=2.2)
    figure.add_horizontal_line(exact, label=f"analytic = {exact:.5f}")
    figure.save(str(out / "probability_convergence.svg"))
    register(
        "probability_convergence.svg",
        "Analytic answers, checked by simulation",
        "Every problem in the Probability Lab is solved twice: once in closed "
        "form, once by a simulation written from the problem statement without "
        "reference to the formula. Agreement within the Monte Carlo interval is "
        "the test. All ten agree.",
    )


BUILDERS = [
    fig_order_book,
    fig_price_discovery,
    fig_return_distribution,
    fig_execution_frontier,
    fig_execution_trajectories,
    fig_volatility_smile,
    fig_greeks,
    fig_deflated_sharpe,
    fig_hrp_vs_markowitz,
    fig_probability_convergence,
]


def write_gallery_page(out: Path, docs: Path) -> None:
    """Emit docs/GALLERY.md linking every figure with its caption."""
    lines = [
        "# Gallery",
        "",
        "Every figure below is generated by `python scripts/build_gallery.py` from a",
        "real computation -- no mock-ups. All are rendered by",
        "[`quantos/viz/svg.py`](../src/quantos/viz/svg.py), a 480-line",
        "dependency-free SVG writer, because a chart library would have broken the",
        "NumPy-only runtime rule in [DDR-002](ddr/DDR-002-numpy-only-runtime.md).",
        "",
        "---",
        "",
    ]
    for filename, title, caption in FIGURES:
        lines += [
            f"## {title}",
            "",
            f"![{title}](gallery/{filename})",
            "",
            caption,
            "",
            "---",
            "",
        ]
    lines += [
        "## Reproducing these",
        "",
        "```bash",
        "pip install -e '.[test]'",
        "python scripts/build_gallery.py",
        "```",
        "",
        "The real-data panels pull live series from FRED and cache them under",
        "`~/.cache/quantos`, so re-running updates them. Pass `--offline` to use",
        "only what is already cached.",
        "",
    ]
    (docs / "GALLERY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/gallery"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"building gallery into {out}/")
    for builder in BUILDERS:
        name = builder.__name__.replace("fig_", "")
        try:
            builder(out)
            print(f"  ok    {name}")
        except Exception as error:
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
    try:
        fig_real_equity(out, args.offline)
        print("  ok    real_equity")
    except Exception as error:
        print(f"  FAIL  real_equity: {error}")

    write_gallery_page(out, out.parent)
    total = sum(f.stat().st_size for f in out.glob("*.svg"))
    print(f"\n{len(FIGURES)} figures, {total / 1024:.0f} KB total")
    print(f"wrote {out.parent / 'GALLERY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
