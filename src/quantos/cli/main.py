"""The ``quantos`` command-line interface.

Design
------
The CLI exists so that every claim in the README can be reproduced by one
command. Each subcommand corresponds to a subsystem, prints numbers rather than
prose, and takes ``--seed`` so its output is reproducible.

Implemented with :mod:`argparse` rather than a third-party framework, in keeping
with the zero-runtime-dependency rule (``docs/ddr/DDR-002``).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from quantos import __version__
from quantos.core.logging import configure_logging, get_logger
from quantos.data.fred import FredError
from quantos.data.market import MarketDataError

_LOG = get_logger(__name__)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from quantos.sim.world import SimulationResult

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _supports_colour() -> bool:
    return sys.stdout.isatty()


def _heading(text: str) -> str:
    if not _supports_colour():
        return f"\n{text}\n{'=' * len(text)}"
    return f"\n{_BOLD}{text}{_RESET}\n{_DIM}{'=' * len(text)}{_RESET}"


# --------------------------------------------------------------------------- #
# Subcommands                                                                 #
# --------------------------------------------------------------------------- #
def cmd_probability(args: argparse.Namespace) -> int:
    """Verify every probability problem's analytic answer against Monte Carlo."""
    from quantos.core.rng import SeedBank
    from quantos.probability.problems import ALL_PROBLEMS

    print(_heading("Probability Lab: analytic vs Monte Carlo"))
    print(
        "Each problem is solved twice -- once in closed form, once by simulation\n"
        "written from the problem statement. Agreement within the Monte Carlo\n"
        f"confidence interval is the test. n = {args.samples:,} per problem.\n"
    )
    bank = SeedBank(root=args.seed)
    selected = [p for p in ALL_PROBLEMS if not args.problem or args.problem in p.name]
    if not selected:
        print(f"no problem matches {args.problem!r}")
        return 2

    # Bonferroni-correct the confidence level for the number of problems checked.
    # Without it, ten independent 99% intervals agree all-together only 0.99^10 =
    # 90% of the time, so this command would report a spurious failure on roughly
    # one run in ten. The repository implements multiple-testing corrections; it
    # should apply one to its own verification.
    family_alpha = 0.01
    level = 1.0 - family_alpha / len(selected)
    print(
        f"Confidence level Bonferroni-corrected to {level:.5f} for "
        f"{len(selected)} simultaneous checks (family-wise alpha {family_alpha}).\n"
    )

    failures = 0
    for problem in selected:
        result = problem.verify(args.samples, bank.child(problem.name).generator(), level=level)
        print(result)
        failures += 0 if result.agrees else 1
    if failures:
        print(f"\n{failures} problem(s) disagreed with their analytic solution.")
        return 1
    print("\nAll checked problems agree with their closed-form solutions.")
    return 0


def cmd_book(args: argparse.Namespace) -> int:
    """Benchmark the limit order book and assert its invariants."""
    import random

    from quantos.core.types import AgentId, Order, OrderId, Quantity, Side, Ticks
    from quantos.exchange.book import LimitOrderBook

    print(_heading("Order book: throughput and invariants"))
    book = LimitOrderBook()
    rng = random.Random(args.seed)
    live: list[int] = []
    order_id = 0
    operations = args.operations

    start = time.perf_counter()
    for _ in range(operations):
        roll = rng.random()
        if roll < 0.55 or not live:
            order_id += 1
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            best_ask, best_bid = book.best_ask, book.best_bid
            price = rng.randint(9_950, 10_050)
            if side is Side.BUY and best_ask is not None:
                price = min(price, int(best_ask) - 1)
            if side is Side.SELL and best_bid is not None:
                price = max(price, int(best_bid) + 1)
            if price <= 0:
                continue
            try:
                book.add(
                    Order(
                        OrderId(order_id),
                        AgentId(f"a{order_id % 8}"),
                        side,
                        Quantity(rng.randint(1, 100)),
                        Ticks(price),
                    )
                )
                live.append(order_id)
            except Exception:
                pass
        elif roll < 0.9:
            victim = live.pop(rng.randrange(len(live)))
            with contextlib.suppress(Exception):
                book.cancel(OrderId(victim))
        else:
            target = live[rng.randrange(len(live))]
            with contextlib.suppress(Exception):
                book.amend(OrderId(target), Quantity(rng.randint(1, 120)))
    elapsed = time.perf_counter() - start

    book.check_invariants()
    print(f"operations           {operations:,}")
    print(f"wall time            {elapsed:.3f} s")
    print(f"throughput           {operations / elapsed:,.0f} ops/s")
    print(f"resting orders       {len(book):,}")
    print(f"heap slack           {book.heap_slack:,}  (stale price-heap entries)")
    print(f"best bid / ask       {book.best_bid} / {book.best_ask}")
    print("\ncheck_invariants()   PASSED")
    print(
        f"{_DIM}Invariants verified: uncrossed book, cached level quantities and\n"
        f"counts consistent with the linked lists, order index complete, and\n"
        f"queue links intact in both directions.{_RESET}"
        if _supports_colour()
        else "Invariants verified: uncrossed book, cached quantities, index, links."
    )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run an agent-based market simulation and measure its stylised facts."""
    from quantos.research.features.microstructure import (
        kyle_lambda,
        price_discovery_efficiency,
    )
    from quantos.sim.scenarios import SCENARIOS, build_liquid_market, build_stressed_market
    from quantos.sim.stylized_facts import analyse_stylized_facts

    builders = {"liquid": build_liquid_market, "stressed": build_stressed_market}
    if args.scenario not in builders:
        print(f"unknown scenario {args.scenario!r}; choose from {sorted(builders)}")
        return 2

    spec = SCENARIOS[args.scenario]
    print(_heading(f"Market simulation: {spec.name}"))
    print(f"{spec.description}")
    print(f"calibrated for: {spec.calibration_target}\n")

    duration = int(args.seconds * 1_000_000_000)
    start = time.perf_counter()
    simulation = builders[args.scenario](duration_ns=duration, seed=args.seed)
    result = simulation.run()
    elapsed = time.perf_counter() - start

    print(f"market time          {args.seconds:.1f} s")
    print(f"wall time            {elapsed:.1f} s")
    print(f"events processed     {result.events_processed:,}")
    print(f"trades printed       {len(result.trades):,}")
    print(f"agents               {len(simulation.agents)}")

    mid = result.mid_series()
    spread = result.spread_series()
    print(f"mean quoted spread   {float(np.nanmean(spread)):.2f} ticks")
    print(f"mid price range      {float(np.nanmin(mid)):.1f} - {float(np.nanmax(mid)):.1f}")

    # Price discovery against the latent value no agent broadcast.
    if result.fundamental_path is not None:
        stamps, values = result.fundamental_path
        grid = np.arange(len(mid)) * simulation.config.snapshot_interval_ns
        interpolated = np.interp(grid, stamps, values)
        finite = np.isfinite(mid)
        if int(np.sum(finite)) > 200:
            discovery = price_discovery_efficiency(mid[finite], interpolated[finite])
            print(_heading("Price discovery"))
            print(
                "The fundamental value is known to the simulator and to no agent's\n"
                "public actions. How much of it did the price incorporate?\n"
            )
            print(f"correlation          {discovery.correlation:.4f}")
            print(f"regression beta      {discovery.beta:.4f}   (1.0 = full incorporation)")
            print(f"tracking RMSE        {discovery.rmse:.2f} ticks")
            print(f"fundamental moved    {float(values.max() - values.min()):.1f} ticks")
            print(f"efficient            {discovery.is_efficient}")

    print(_heading("Agent profit and loss"))
    print(f"{'agent':<22}{'pnl':>12}{'position':>10}{'fills':>9}{'fees':>11}")
    for name, stats in sorted(result.agent_summary.items()):
        print(
            f"{name:<22}{stats['pnl']:>12.2f}{stats['position']:>10.0f}"
            f"{stats['n_fills']:>9.0f}{stats['fees_paid']:>11.2f}"
        )

    returns = result.returns()
    if returns.size >= 500:
        print(_heading("Stylised facts (emergent, not programmed)"))
        print(analyse_stylized_facts(returns))

    signed = result.signed_volume_series()
    if signed.size > 100:
        prices = result.trade_price_series()
        lam = kyle_lambda(np.diff(prices), signed[1:])
        print(_heading("Microstructure"))
        print(f"Kyle's lambda        {lam.lambda_:.3e} ticks per lot  (t={lam.t_statistic:.1f})")
        print(f"implied depth        {lam.market_depth:,.0f} lots to move one tick")

    if args.output:
        _write_simulation_charts(result, Path(args.output))
        print(f"\ncharts written to {args.output}")
    return 0


def _write_simulation_charts(result: SimulationResult, directory: Path) -> None:
    """Emit SVG charts for a simulation result."""
    from quantos.viz.svg import histogram, line_chart

    directory.mkdir(parents=True, exist_ok=True)
    mid = result.mid_series()
    steps = np.arange(mid.size, dtype=np.float64)
    series = {"mid price": (steps, mid)}
    if result.fundamental_path is not None:
        stamps, values = result.fundamental_path
        grid = steps * (result.config.snapshot_interval_ns if result.config else 1_000_000)
        series["fundamental (latent)"] = (steps, np.interp(grid, stamps, values))
    line_chart(
        series,
        title="Price discovery: market price vs latent fundamental",
        x_label="snapshot",
        y_label="price (ticks)",
    ).save(str(directory / "price_discovery.svg"))

    returns = result.returns()
    if returns.size > 500:
        standardised = (returns - returns.mean()) / returns.std()
        grid = np.linspace(-6, 6, 300)
        normal = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
        histogram(
            standardised,
            bins=80,
            density=True,
            title="Return distribution vs Gaussian",
            x_label="standardised return",
            overlay={"N(0,1)": (grid, normal)},
        ).save(str(directory / "return_distribution.svg"))


def cmd_options(args: argparse.Namespace) -> int:
    """Price options, show Greeks, and round-trip implied volatility."""
    from quantos.derivatives.black_scholes import (
        OptionType,
        black_scholes_greeks,
        black_scholes_price,
        implied_volatility,
        put_call_parity_check,
    )

    s, k, t, v, r, q = (
        args.spot,
        args.strike,
        args.maturity,
        args.volatility,
        args.rate,
        args.dividend,
    )
    print(_heading("Black-Scholes-Merton"))
    print(f"S={s}  K={k}  T={t}  sigma={v}  r={r}  q={q}\n")

    call = float(black_scholes_price(s, k, t, v, rate=r, dividend_yield=q))
    put = float(
        black_scholes_price(s, k, t, v, rate=r, dividend_yield=q, option_type=OptionType.PUT)
    )
    print(f"call                 {call:.10f}")
    print(f"put                  {put:.10f}")
    print(
        f"put-call parity      "
        f"{put_call_parity_check(call, put, s, k, t, rate=r, dividend_yield=q):+.3e}"
        "   (model-free identity; must be ~0)"
    )

    print(_heading("Greeks (call)"))
    greeks = black_scholes_greeks(s, k, t, v, rate=r, dividend_yield=q)
    for name, value in greeks.as_dict().items():
        if name == "price":
            continue
        print(f"{name:<20} {value:>16.8f}")

    print(_heading("Implied volatility round-trip"))
    print("Recovering sigma from the price, across the moneyness range.\n")
    print(f"{'strike':>8}{'price':>16}{'implied vol':>16}{'error':>12}")
    for strike in [k * m for m in (0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5)]:
        price = float(black_scholes_price(s, strike, t, v, rate=r, dividend_yield=q))
        try:
            recovered = implied_volatility(price, s, strike, t, rate=r, dividend_yield=q)
            print(f"{strike:>8.1f}{price:>16.10g}{recovered:>16.12f}{abs(recovered - v):>12.2e}")
        except ValueError as error:
            print(f"{strike:>8.1f}{price:>16.10g}{'not identifiable':>16}  {str(error)[:40]}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Demonstrate backtest-overfitting controls on synthetic strategies."""
    from quantos.core.rng import SeedBank
    from quantos.core.stats.multipletest import hansen_spa, stepm, whites_reality_check
    from quantos.strategy.validation import (
        deflated_sharpe_ratio,
        probability_of_backtest_overfitting,
        sharpe_ratio_with_moments,
    )

    rng = SeedBank(root=args.seed).child("validate").generator()
    n_periods, n_configs = 1260, args.configurations

    print(_heading("Backtest overfitting controls"))
    print(
        f"Simulating a research process: {n_configs} strategy configurations tested\n"
        f"over {n_periods} daily observations. Exactly ONE has a real edge, of\n"
        f"{args.edge * 1e4:.0f} bp per day.\n"
    )
    performance = rng.standard_normal((n_periods, n_configs)) * 0.01
    truth = 3
    performance[:, truth] += args.edge

    sharpes = performance.mean(axis=0) / performance.std(axis=0, ddof=1)
    best = int(np.argmax(sharpes))
    found_the_real_one = best == truth
    print(f"planted edge in configuration    #{truth}")
    print(f"best in-sample Sharpe found in   #{best}")
    print(f"best in-sample Sharpe (annual)   {sharpes[best] * np.sqrt(252):.3f}")
    print(f"planted config Sharpe (annual)   {sharpes[truth] * np.sqrt(252):.3f}")
    if found_the_real_one:
        print("  -> the search found the genuine edge; the tests below should confirm it")
    else:
        print(
            f"  -> the search found NOISE: config #{best} has no edge at all, and it\n"
            f"     beat the real one in sample. Every test below should decline to\n"
            f"     call it significant. Try --edge 0.0025 to see the other regime."
        )

    print(_heading("1. Naive vs deflated Sharpe ratio"))
    stats = sharpe_ratio_with_moments(performance[:, best])
    print(f"annualised Sharpe                {stats.annualised:.4f}")
    print(f"naive one-sided p-value          {stats.p_value:.4f}")
    naive_verdict = "SIGNIFICANT" if stats.p_value < 0.05 else "not significant"
    print(f"  -> {naive_verdict} if you forget you ran {n_configs} trials")
    deflated = deflated_sharpe_ratio(performance[:, best], n_trials=n_configs)
    print(f"\nexpected max Sharpe under H0     {deflated.expected_maximum:.4f} (per period)")
    print(f"deflated Sharpe p-value          {deflated.p_value:.4f}")
    print(f"  -> {'SIGNIFICANT' if deflated.is_significant else 'not significant'} after deflation")

    print(_heading("2. Probability of backtest overfitting (CSCV)"))
    pbo = probability_of_backtest_overfitting(performance, n_splits=10)
    print(f"PBO                              {pbo.pbo:.4f}")
    print(f"OOS-on-IS degradation slope      {pbo.performance_degradation:+.4f}")
    print(f"  -> selection {'is not' if pbo.is_overfit else 'is'} informative out of sample")
    print(
        f"{_DIM if _supports_colour() else ''}PBO is itself noisy across datasets; "
        f"see the docstring.{_RESET if _supports_colour() else ''}"
    )

    print(_heading("3. Data-snooping tests on the whole universe"))
    reality = whites_reality_check(performance, n_bootstrap=500, rng=rng)
    spa = hansen_spa(performance, n_bootstrap=500, rng=rng)
    print(f"White's Reality Check p-value    {reality.p_value:.4f}")
    print(f"Hansen SPA p-value               {spa.p_value:.4f}   (best: #{spa.best_index})")
    rejected = np.nonzero(stepm(performance, n_bootstrap=500, rng=rng))[0].tolist()
    print(f"Romano-Wolf StepM rejects        {rejected}   (planted: [{truth}])")

    print(_heading("What just happened"))
    if found_the_real_one and truth in rejected:
        print(
            "The edge was strong enough to win in sample AND to survive every\n"
            "correction. StepM named it by index. This is the outcome you want,\n"
            "and it is rarer than the literature implies."
        )
    elif found_the_real_one:
        print(
            "The search found the right configuration, but the corrections still\n"
            f"cannot certify it against {n_configs} trials. The edge is real and\n"
            "the evidence is insufficient -- which is a legitimate finding, not a\n"
            "failure of the tests. minimum_track_record_length() says how much\n"
            "more data would settle it."
        )
    else:
        print(
            f"The best in-sample result was pure noise, and it beat a genuine\n"
            f"{args.edge * 1e4:.0f}bp/day edge. Every method declined to certify it, which\n"
            "is exactly correct. Note what the naive p-value said: 'significant'.\n"
            "That single number, reported without the trial count, is how most\n"
            "spurious strategies reach production."
        )
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Compare portfolio construction methods out of sample."""
    from quantos.core.linalg import condition_number, effective_rank, marchenko_pastur_edge
    from quantos.core.rng import SeedBank
    from quantos.risk.portfolio import (
        hierarchical_risk_parity,
        ledoit_wolf_shrinkage,
        minimum_variance,
        risk_parity,
    )

    rng = SeedBank(root=args.seed).child("portfolio").generator()
    n_assets, n_train, n_test = args.assets, args.train, 1000

    # A one-factor market plus idiosyncratic noise: the structure real equity
    # covariance matrices actually have.
    factor_train = rng.standard_normal((n_train, 1))
    factor_test = rng.standard_normal((n_test, 1))
    betas = rng.uniform(0.4, 1.6, n_assets)
    idio = rng.uniform(0.005, 0.03, n_assets)
    train = factor_train @ betas[None, :] * 0.01 + rng.standard_normal((n_train, n_assets)) * idio
    test = factor_test @ betas[None, :] * 0.01 + rng.standard_normal((n_test, n_assets)) * idio

    print(_heading("Portfolio construction, out of sample"))
    print(
        f"{n_assets} assets, {n_train} training observations "
        f"(N/T = {n_assets / n_train:.2f}), {n_test} test observations.\n"
        "One-factor return structure. Lower out-of-sample volatility is better.\n"
    )
    sample_cov = np.cov(train, rowvar=False)
    shrunk, delta = ledoit_wolf_shrinkage(train)
    lo, hi = marchenko_pastur_edge(n_assets, n_train)
    print(f"sample covariance condition      {condition_number(sample_cov):.3e}")
    print(f"shrunk covariance condition      {condition_number(shrunk):.3e}")
    print(f"Ledoit-Wolf shrinkage intensity  {delta:.4f}")
    print(f"effective rank                   {effective_rank(sample_cov):.2f} of {n_assets}")
    print(f"Marchenko-Pastur noise band      [{lo:.3f}, {hi:.3f}] (eigenvalues inside are noise)")

    methods = {
        "equal weight": np.ones(n_assets) / n_assets,
        "min-var (sample cov)": minimum_variance(returns=train, shrink=False).weights,
        "min-var (LW shrunk)": minimum_variance(returns=train, shrink=True).weights,
        "risk parity": risk_parity(returns=train, shrink=True).weights,
        "hierarchical risk parity": hierarchical_risk_parity(returns=train).weights,
    }
    print(f"\n{'method':<26}{'in-sample vol':>15}{'OOS vol':>11}{'eff. positions':>16}")
    for name, weights in methods.items():
        in_sample = float(np.std(train @ weights))
        out_sample = float(np.std(test @ weights))
        effective = float(1.0 / np.sum(weights**2))
        print(f"{name:<26}{in_sample:>15.6f}{out_sample:>11.6f}{effective:>16.1f}")
    print(
        "\nThe gap between in-sample and out-of-sample volatility for the\n"
        "unshrunk minimum-variance portfolio is the error-maximisation problem\n"
        "in one number."
    )
    return 0


def cmd_execution(args: argparse.Namespace) -> int:
    """Show the Almgren-Chriss efficient frontier of execution."""
    from quantos.execution.almgren_chriss import (
        ImpactParameters,
        efficient_execution_frontier,
        square_root_impact_cost,
    )

    params = ImpactParameters(volatility=args.volatility, temporary_impact=1e-6)
    print(_heading("Optimal execution: Almgren-Chriss frontier"))
    print(
        f"Liquidating {args.quantity:,.0f} shares over {args.horizon} unit(s) of time.\n"
        "There is no single optimum -- only a frontier. Risk aversion picks a point.\n"
    )
    print(
        f"{'risk aversion':>16}{'urgency (kappa)':>18}{'E[cost]':>14}"
        f"{'sd[cost]':>14}{'front-load':>12}{'half-life':>11}"
    )
    for trajectory in efficient_execution_frontier(
        args.quantity,
        args.horizon,
        params,
        risk_aversions=[0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    ):
        print(
            f"{trajectory.detail['risk_aversion']:>16.1e}{trajectory.urgency:>18.4f}"
            f"{trajectory.expected_cost:>14.2f}{trajectory.cost_standard_deviation:>14.2f}"
            f"{trajectory.front_loading:>12.3f}{trajectory.half_life:>11.4f}"
        )
    print(
        "\nRisk aversion 0 is exactly TWAP (front-load 0.500): a straight line is\n"
        "optimal for a trader indifferent to price risk, not a naive baseline."
    )

    print(_heading("Square-root impact law"))
    print("Impact depends on size relative to daily volume -- not on the schedule.\n")
    print(f"{'participation':>15}{'impact (bps)':>15}")
    for participation in (0.001, 0.005, 0.01, 0.05, 0.10, 0.25):
        cost = square_root_impact_cost(participation, 1.0, args.volatility, coefficient=0.8)
        print(f"{participation:>15.1%}{cost * 1e4:>15.2f}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Run the QuantOS toolkit over real market and macroeconomic series."""
    import numpy as np

    from quantos.data.analysis import analyse_cross_section, analyse_series
    from quantos.data.catalog import BUNDLES, CATALOG, Kind, describe_bundle, resolve
    from quantos.data.fred import FredClient, FredError
    from quantos.data.loader import align

    if args.list:
        print(_heading("Catalogued series"))
        print(f"{'key':<16}{'FRED id':<16}{'kind':<8}name")
        print("-" * 78)
        for key, series in sorted(CATALOG.items()):
            print(f"{key:<16}{series.fred_id:<16}{series.kind.value:<8}{series.name}")
        print(_heading("Bundles"))
        for name, (description, members) in sorted(BUNDLES.items()):
            print(f"  {name:<16}{', '.join(members)}")
            print(f"  {'':<16}{description}")
        print(
            "\nIndividual stocks and ETFs are not on FRED. Download a CSV and pass\n"
            "--csv PATH.csv (Yahoo Finance's Download button works as-is)."
        )
        return 0

    keys: list[str] = []
    if args.bundle:
        if args.bundle not in BUNDLES:
            print(f"unknown bundle {args.bundle!r}; choose from {', '.join(sorted(BUNDLES))}")
            return 2
        keys = list(BUNDLES[args.bundle][1])
    if args.series:
        keys.extend(s.strip() for s in args.series.split(",") if s.strip())
    if not keys and not args.csv:
        keys = list(BUNDLES["risk-appetite"][1])

    print(_heading("QuantOS on real data"))
    if args.bundle:
        print(f"bundle: {args.bundle} -- {describe_bundle(args.bundle)}\n")
    print(f"window: from {args.start}\n")

    client = FredClient(offline=args.offline)
    loaded: dict[str, tuple] = {}

    for key in keys:
        try:
            spec = resolve(key)
        except KeyError as error:
            print(f"  skipping {key}: {error}")
            continue
        try:
            fetched = client.get(spec.fred_id).since(args.start)
        except FredError as error:
            print(f"  skipping {spec.name}: {error}")
            continue
        if len(fetched) < 60:
            print(f"  skipping {spec.name}: only {len(fetched)} observations after {args.start}")
            continue
        loaded[spec.key] = (spec, fetched.dates, fetched.values)

    if args.csv:
        from quantos.data.loader import load_ohlcv_csv

        for path in args.csv.split(","):
            try:
                price = load_ohlcv_csv(path.strip()).since(args.start)
            except ValueError as error:
                print(f"  skipping {path}: {error}")
                continue
            from quantos.data.catalog import Series as CatalogSeries

            spec = CatalogSeries(
                key=price.symbol.lower(),
                fred_id=price.symbol,
                name=price.symbol,
                kind=Kind.LEVEL,
                reads_as=f"loaded from {path.strip()}",
            )
            loaded[spec.key] = (spec, price.dates, price.prices)
            adjusted = price.detail.get("dividend_adjusted") == "True"
            print(
                f"  loaded {price.symbol}: {len(price)} rows from column "
                f"'{price.price_column}'"
                + ("" if adjusted else "  [NOT dividend-adjusted -- returns will be biased]")
            )

    if not loaded:
        print("\nno series could be loaded.")
        return 1

    reports = []
    for key, (spec, dates, values) in loaded.items():
        report = analyse_series(key, spec.name, spec.kind.value, dates, values)
        reports.append((spec, report))

    for spec, report in reports:
        print(_heading(f"{report.name}  [{spec.fred_id}]"))
        print(f"  {spec.reads_as}")
        print(
            f"  {report.n_observations:,} observations, {report.start} to {report.end}, "
            f"latest {report.latest:,.4g}"
        )
        print()
        # Units matter here. A *level* series has returns, so percentages are
        # right. A *rate* or non-tradeable index has first differences measured
        # in the units of the series itself -- annualising those by sqrt(252) and
        # printing "%" produced "VIX annualised volatility 128%", which reads as
        # a return and is not one.
        is_return = np.isfinite(report.sharpe)
        if np.isfinite(report.annualised_return):
            print(f"  annualised return      {report.annualised_return:>10.2%}")
        if is_return:
            print(f"  annualised volatility  {report.annualised_volatility:>10.2%}")
            print(f"  Sharpe ratio           {report.sharpe:>10.3f}")
            print(f"    autocorr-adjusted    {report.sharpe_autocorr_adjusted:>10.3f}")
            print(f"  maximum drawdown       {report.max_drawdown:>10.2%}")
            print(f"  VaR 95% / CVaR 95%     {report.var_95:>10.2%} / {report.cvar_95:.2%}")
            print(f"  VaR 99% / CVaR 99%     {report.var_99:>10.2%} / {report.cvar_99:.2%}")
        else:
            unit = "index pts" if spec.kind is Kind.INDEX else "pp"
            daily = report.annualised_volatility / np.sqrt(252.0)
            print(f"  daily change, std dev  {daily:>10.4f} {unit}")
            print(f"  annualised             {report.annualised_volatility:>10.4f} {unit}")
            print(f"  VaR 95% / CVaR 95%     {report.var_95:>10.4f} / {report.cvar_95:.4f} {unit}")
            print(f"  VaR 99% / CVaR 99%     {report.var_99:>10.4f} / {report.cvar_99:.4f} {unit}")
        print(f"    {report.tail_severity}")
        print(f"  skew / excess kurtosis {report.skewness:>10.2f} / {report.excess_kurtosis:.2f}")
        print(f"  Hill tail index        {report.tail_index:>10.2f}   (empirical equities ~3)")
        print(f"  Jarque-Bera p-value    {report.jarque_bera_p:>10.3g}")
        if np.isfinite(report.garch_persistence):
            print()
            print(
                f"  GARCH(1,1)             alpha {report.garch_alpha:.3f}  "
                f"beta {report.garch_beta:.3f}  persistence {report.garch_persistence:.4f}"
            )
            print(f"    shock half-life      {report.garch_half_life:>10.1f} days")
            if is_return:
                print(f"    1-step vol forecast  {report.volatility_forecast:>10.2%} annualised")
            else:
                unit = "index pts" if spec.kind is Kind.INDEX else "pp"
                print(
                    f"    1-step vol forecast  {report.volatility_forecast:>10.4f} "
                    f"{unit} annualised"
                )
        print()
        print(f"  stationarity           {report.stationarity_verdict}")
        for note in report.notes:
            print(f"  note: {note}")

    if len(loaded) > 1:
        common, aligned = align({k: (d, v) for k, (_, d, v) in loaded.items()})
        if common.size > 60:
            transformed = {}
            levels = {}
            for key, (spec, _, _) in loaded.items():
                try:
                    transformed[key] = spec.transform(aligned[key])
                except ValueError:
                    transformed[key] = np.diff(aligned[key])
                if spec.kind is Kind.LEVEL:
                    levels[key] = aligned[key]

            cross = analyse_cross_section(transformed, levels, dates=common)
            print(_heading("Cross-series"))
            print(f"  {cross.n_common_dates:,} common dates, {cross.start} to {cross.end}\n")
            width = max(len(n) for n in cross.names) + 2
            print(" " * width + "".join(f"{n[:9]:>11}" for n in cross.names))
            for i, name in enumerate(cross.names):
                row = "".join(f"{cross.correlation[i, j]:>11.3f}" for j in range(len(cross.names)))
                print(f"{name:<{width}}{row}")
            a, b, value = cross.most_correlated()
            if a:
                print(f"\n  strongest pair: {a} / {b} at {value:+.3f}")
            if cross.cointegration:
                print("\n  cointegration (Engle-Granger, both directions):")
                for (x, y), (is_coint, stat, beta) in cross.cointegration.items():
                    verdict = "COINTEGRATED" if is_coint else "not cointegrated"
                    print(f"    {x:<12} / {y:<12} stat {stat:>7.3f}  beta {beta:>8.4f}  {verdict}")
        else:
            print(_heading("Cross-series"))
            print(f"  only {common.size} common dates -- too few to compare.")
            print("  Series with different frequencies (daily vs monthly) rarely align.")

    if args.output:
        _write_analysis_charts(loaded, Path(args.output))
        print(f"\ncharts written to {args.output}")

    print(_heading("Caveats"))
    print(
        "  Data: Federal Reserve Bank of St. Louis (FRED), retrieved and cached\n"
        "  locally. These are historical statistics, not forecasts, and nothing\n"
        "  here is investment advice. Sharpe ratios computed on an index exclude\n"
        "  dividends, financing and transaction costs."
    )
    return 0


def _write_analysis_charts(loaded: dict, directory: Path) -> None:
    """Emit SVG charts for a real-data analysis."""
    import numpy as np

    from quantos.viz.svg import histogram, line_chart

    directory.mkdir(parents=True, exist_ok=True)
    for key, (spec, dates, values) in loaded.items():
        x = np.arange(values.size, dtype=np.float64)
        line_chart(
            {spec.name: (x, values)},
            title=f"{spec.name} ({spec.fred_id})",
            x_label=f"observations, {str(dates[0])[:10]} to {str(dates[-1])[:10]}",
            y_label=spec.kind.value,
        ).save(str(directory / f"{key}_level.svg"))

        try:
            changes = spec.transform(values)
        except ValueError:
            changes = np.diff(values)
        if changes.size > 200:
            standardised = (changes - changes.mean()) / changes.std()
            grid = np.linspace(-6, 6, 300)
            normal = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
            histogram(
                standardised,
                bins=70,
                density=True,
                title=f"{spec.name}: distribution of changes vs Gaussian",
                x_label="standardised change",
                overlay={"N(0,1)": (grid, normal)},
            ).save(str(directory / f"{key}_distribution.svg"))


def cmd_research(args: argparse.Namespace) -> int:
    """Produce a full research report on one instrument."""
    import numpy as np

    from quantos.data.catalog import Kind, resolve
    from quantos.data.fred import FredClient, FredError
    from quantos.data.loader import load_ohlcv_csv
    from quantos.data.market import MarketDataError, fetch_prices
    from quantos.research.instruments import AssetClass, Instrument
    from quantos.research.render import render_markdown, render_text
    from quantos.research.report import generate_report

    client = FredClient(offline=args.offline)
    instrument: Instrument

    if args.ticker:
        try:
            price, info = fetch_prices(
                args.ticker,
                start=args.start,
                range_key=args.range,
                offline=args.offline,
                refresh=args.refresh,
            )
        except MarketDataError as error:
            print(f"could not fetch {args.ticker}: {error}")
            return 1
        print(f"resolved {info.describe()}")
        print(f"  {len(price)} daily bars, {price.start}..{price.end}, {price.price_column}")
        if "warning" in price.detail:
            print(f"  warning: {price.detail['warning']}")
        print()
        # The asset class comes from the venue rather than from a flag, so the
        # report gates its own analyses correctly without being told.
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
    elif args.csv:
        try:
            price = load_ohlcv_csv(args.csv, symbol=args.symbol).since(args.start)
        except ValueError as error:
            print(f"could not load {args.csv}: {error}")
            return 1
        instrument = Instrument(
            symbol=price.symbol,
            asset_class=AssetClass(args.asset_class),
            dates=price.dates,
            prices=price.prices,
            source=price.source,
            dividend_adjusted=price.detail.get("dividend_adjusted") == "True",
            average_daily_volume=(
                float(np.mean(price.volume)) if price.volume is not None else None
            ),
        )
    elif args.series:
        try:
            spec = resolve(args.series)
            fetched = client.get(spec.fred_id).since(args.start)
        except (KeyError, FredError) as error:
            print(f"could not load {args.series}: {error}")
            return 1
        mapping = {
            Kind.LEVEL: AssetClass.INDEX,
            Kind.INDEX: AssetClass.INDEX,
            Kind.RATE: AssetClass.RATE,
        }
        instrument = Instrument(
            symbol=spec.fred_id,
            name=spec.name,
            asset_class=mapping[spec.kind],
            dates=fetched.dates,
            prices=fetched.values,
            source=f"FRED:{spec.fred_id}",
            dividend_adjusted=True,
        )
    else:
        print(
            "supply --ticker SYMBOL (e.g. --ticker AAPL), --csv PATH, or --series KEY.\n"
            "See `quantos analyse --list` for the built-in macro series."
        )
        return 2

    if len(instrument) < 60:
        print(f"only {len(instrument)} observations after {args.start}; need 60+")
        return 1

    factors: dict[str, np.ndarray] = {}
    if not args.no_factors:
        from quantos.data.factors import build_factors

        factor_set = build_factors(
            instrument.dates, instrument.prices, start=args.start, offline=args.offline
        )
        note = factor_set.note()
        if note:
            print(f"  note: {note}")
        if factor_set.usable:
            factors = factor_set.columns
        if factor_set.unavailable:
            print(f"  note: factors unavailable: {', '.join(factor_set.unavailable)}")

    report = generate_report(
        instrument,
        factors=factors or None,
        run_signals=not args.no_signals,
        transaction_cost_bps=args.cost_bps,
    )
    print(render_text(report))

    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
        print(f"\nMarkdown report written to {path}")
    return 0


def cmd_forward(args: argparse.Namespace) -> int:
    """Record today's predictions, settle the ones whose horizon has elapsed."""
    from quantos.data.loader import load_ohlcv_csv
    from quantos.live.ledger import Ledger, default_ledger_path
    from quantos.live.runner import run_daily

    ledger = Ledger(args.ledger or default_ledger_path())

    if args.score_only:
        scored = ledger.score(signal=args.signal, symbol=args.symbol)
        print(f"forward-testing ledger: {ledger.path}")
        print(f"  records          {len(ledger.read_all())}")
        try:
            ledger.verify_chain()
            print("  hash chain       intact")
        except Exception as error:
            print(f"  hash chain       BROKEN: {error}")
            return 1
        for key in (
            "n_predictions",
            "n_settled",
            "n_open",
            "hit_rate",
            "hit_rate_95_low",
            "hit_rate_95_high",
            "mean_return",
            "total_return",
        ):
            if key in scored:
                value = scored[key]
                shown = f"{value:.4f}" if isinstance(value, float) else str(value)
                print(f"  {key:16s} {shown}")
        if "note" in scored:
            print(f"  note: {scored['note']}")
        if scored.get("n_settled", 0) > 0:
            print(
                "\nNo deflation, no purging and no embargo were applied, because none "
                "are needed: every prediction here was written down before its outcome "
                "existed."
            )
        return 0

    if args.ticker:
        from quantos.data.market import MarketDataError, fetch_prices

        try:
            series, info = fetch_prices(args.ticker, start=args.start, offline=args.offline)
        except MarketDataError as error:
            print(f"could not fetch {args.ticker}: {error}")
            return 1
        print(f"resolved {info.describe()}")
    elif args.csv:
        try:
            series = load_ohlcv_csv(args.csv, symbol=args.symbol).since(args.start)
        except ValueError as error:
            print(f"could not load {args.csv}: {error}")
            return 1
    else:
        print(
            "pass --ticker SYMBOL or --csv PATH to record predictions, "
            "or --score-only to read the ledger back"
        )
        return 1

    result = run_daily(ledger, series, horizon_days=args.horizon)
    print(f"forward cycle for {result['symbol']} as of {series.end}")
    print(f"  settled          {result['settled']}")
    print(f"  newly recorded   {result['recorded']}")
    print(f"  already present  {result['already_present']}")
    print(f"  open positions   {result['open_after']}")
    print(f"  hash chain       {'intact' if result['chain_valid'] else 'BROKEN'}")
    print(f"\nledger: {ledger.path}")
    return 0


def cmd_surface(args: argparse.Namespace) -> int:
    """Fit a volatility surface to a real option chain."""
    import datetime

    import numpy as np

    from quantos.data.loader import load_ohlcv_csv
    from quantos.data.options import ChainFilter, load_option_chain_csv
    from quantos.research.vol_surface import fit_surface, variance_risk_premium

    as_of = None
    if args.as_of:
        try:
            as_of = datetime.date.fromisoformat(args.as_of)
        except ValueError:
            print(f"--as-of must be YYYY-MM-DD, got {args.as_of!r}")
            return 1

    try:
        chain = load_option_chain_csv(
            args.chain,
            symbol=args.symbol,
            as_of=as_of,
            spot=args.spot,
            rate=args.rate,
            chain_filter=ChainFilter(otm_only=not args.keep_itm),
        )
    except ValueError as error:
        print(f"could not load {args.chain}: {error}")
        return 1

    print(chain.summary())
    if len(chain) == 0:
        return 1

    print()
    surface = fit_surface(chain)
    print(surface.summary())

    if surface.smiles:
        times, vols = surface.term_structure()
        _, skews = surface.skew_structure()
        print("\nterm structure")
        for t, v, k in zip(times, vols, skews, strict=True):
            print(f"  T={t:5.3f}  ATM {v:6.2%}  skew {k:+7.3f}")

    if args.underlying:
        try:
            price = load_ohlcv_csv(args.underlying, symbol=chain.symbol)
        except ValueError as error:
            print(f"\ncould not load {args.underlying}: {error}")
            return 0
        returns = np.diff(np.log(price.prices))
        premium = variance_risk_premium(chain, returns[-args.realised_days :])
        print()
        print(premium.summary())
        for note in premium.notes:
            print(f"  note: {note}")
    return 0


def cmd_intraday(args: argparse.Namespace) -> int:
    """Estimate volatility from intraday data, correcting for microstructure noise."""
    from quantos.data.intraday import load_intraday_csv
    from quantos.research.intraday import (
        epps_curve,
        intraday_seasonality,
        signature_plot,
        volatility_report,
    )

    try:
        bars = load_intraday_csv(args.csv, symbol=args.symbol, price_column=args.price_column)
    except ValueError as error:
        print(f"could not load {args.csv}: {error}")
        return 1

    print(bars.summary())

    sessions = bars.sessions(min_observations=40)
    if not sessions:
        print("\nno session has enough observations for these estimators")
        return 1

    # Estimate within sessions, never across the overnight gap.
    target = sessions[args.session] if args.session < len(sessions) else sessions[-1]
    print(f"\n--- session {min(args.session, len(sessions) - 1)} of {len(sessions)} ---")
    print(volatility_report(target).summary())
    print()
    print(signature_plot(target).summary())

    if len(sessions) >= 5:
        seasonality = intraday_seasonality(sessions)
        print(f"\nintraday seasonality across {seasonality.n_sessions} sessions")
        for position, level in zip(
            seasonality.time_of_day, seasonality.relative_volatility, strict=True
        ):
            bar = "#" * round(level * 20)
            print(f"  {position:5.2f} of session  {level:5.2f}  {bar}")
        print(
            "  U-shaped: "
            + ("yes, as volume and volatility usually are" if seasonality.is_u_shaped else "no")
        )

    if args.compare:
        try:
            other = load_intraday_csv(args.compare, price_column=args.price_column)
        except ValueError as error:
            print(f"\ncould not load {args.compare}: {error}")
            return 0
        steps, correlations = epps_curve(target, other.sessions(min_observations=40)[0])
        print(f"\nEpps curve against {other.symbol}")
        for step, correlation in zip(steps, correlations, strict=True):
            print(f"  every {step:5d} obs   correlation {correlation:+.4f}")
    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    """Estimate macro sensitivities, then answer 'what happens if' as a range."""
    from quantos.data.align import MACRO_FACTORS, coverage, factor_changes, simple_returns
    from quantos.data.fred import FredClient, FredError
    from quantos.data.market import fetch_prices
    from quantos.risk.scenario import SCENARIOS, apply_shock, estimate_response

    series, info = fetch_prices(args.ticker, range_key=args.range)
    dates = np.asarray(series.dates, dtype="datetime64[D]")
    returns = simple_returns(series.prices)

    client = FredClient()
    factors: dict[str, ArrayLike] = {}
    for name in args.factors:
        factor = MACRO_FACTORS[name]
        try:
            macro = client.get(factor.series_id)
        except (FredError, OSError) as error:
            _LOG.warning("skipping factor %s (%s): %s", name, factor.series_id, error)
            print(f"  skipping {name}: {error}")
            continue

        macro_dates = np.asarray(macro.dates, dtype="datetime64[D]")
        present = coverage(dates, macro_dates)
        if present < 0.5:
            print(
                f"  skipping {name}: present on only {present:.0%} of "
                f"{info.ticker}'s trading days, which would make the regression "
                "a statement about different dates than it appears to be"
            )
            continue

        factors[name] = factor_changes(dates, macro_dates, macro.values, factor.kind)

    if not factors:
        print("no macro factors available; nothing to estimate")
        return 1

    usable = np.isfinite(returns) & np.all(
        np.isfinite(np.column_stack(list(factors.values()))), axis=1
    )
    response = estimate_response(
        returns[usable], {name: np.asarray(v)[usable] for name, v in factors.items()}
    )

    print(f"{info.describe()}\n")
    print(response.summary())
    print()

    for scenario in (s for s in SCENARIOS if set(s.shocks) <= set(factors)):
        print(apply_shock(response, scenario.shocks, name=scenario.name).summary())
        print()
    return 0


def cmd_lattice(args: argparse.Namespace) -> int:
    """Price on a lattice, and show what the step count is actually worth."""
    from quantos.derivatives.black_scholes import OptionType
    from quantos.derivatives.lattice import (
        averaged_binomial_price,
        binomial_price,
        convergence_path,
        trinomial_price,
    )

    option_type = OptionType.PUT if args.put else OptionType.CALL
    shared = {
        "rate": args.rate,
        "dividend_yield": args.dividend_yield,
        "option_type": option_type,
        "american": args.american,
    }
    args_positional = (args.spot, args.strike, args.expiry, args.volatility)

    print(binomial_price(*args_positional, n_steps=args.steps, **shared).summary())
    print()
    print(averaged_binomial_price(*args_positional, n_steps=args.steps, **shared).summary())
    print()
    print(trinomial_price(*args_positional, n_steps=args.steps, **shared).summary())

    if args.american or args.put:
        return 0

    counts, binomial, trinomial = convergence_path(
        *args_positional,
        rate=args.rate,
        dividend_yield=args.dividend_yield,
        option_type=option_type,
        steps=range(30, 151),
    )
    flips = int(np.sum(np.diff(np.sign(binomial)) != 0))
    tri_flips = int(np.sum(np.diff(np.sign(trinomial)) != 0))
    print(
        f"\nAcross {counts.size} step counts the binomial error changes sign "
        f"{flips} times and the trinomial {tri_flips}. Convergence is not "
        "monotone, so a price quoted at one convenient step count is a point on "
        "an oscillation rather than a converged value."
    )
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    """Replay an instrument through crises that actually happened."""
    import numpy as np

    from quantos.data.align import align_to_grid, simple_returns
    from quantos.data.market import fetch_prices
    from quantos.risk.stress import CRISES, correlation_breakdown, stress_test

    series, info = fetch_prices(args.ticker, range_key=args.range)
    dates = np.asarray(series.dates, dtype="datetime64[D]")
    prices = np.asarray(series.prices, dtype=float)

    print(f"{info.describe()}\n")
    report = stress_test(dates, prices)
    print(report.summary())

    if not args.against:
        return 0

    # Correlation needs a second leg, and the comparison assets must be aligned
    # onto the same dates -- a shorter history would otherwise silently shift
    # every observation.
    returns: dict[str, ArrayLike] = {}
    for symbol in [args.ticker, *args.against]:
        other, _ = fetch_prices(symbol, range_key=args.range)
        aligned = align_to_grid(dates, np.asarray(other.dates, dtype="datetime64[D]"), other.prices)
        returns[symbol] = np.nan_to_num(simple_returns(aligned))

    print()
    for crisis in CRISES:
        breakdown = correlation_breakdown(
            dates, returns, crisis, risk_assets={args.ticker} | set(args.risk or [])
        )
        if breakdown.pairs:
            print(breakdown.summary())
            print()
    return 0


def cmd_factors(args: argparse.Namespace) -> int:
    """Search a large factor grid, and correct for having searched it."""
    import numpy as np

    from quantos.research.factor_lab import run_factor_lab

    if args.ticker:
        from quantos.data.market import fetch_prices

        series, info = fetch_prices(args.ticker)
        returns = series.log_returns()
        print(f"{info.describe()} -- {len(returns):,} daily returns\n")
    else:
        rng = np.random.default_rng(args.seed)
        returns = rng.standard_normal(args.n_synthetic) * 0.01
        print(
            f"No ticker given, so the search runs on {args.n_synthetic:,} synthetic\n"
            "returns with no signal in them whatsoever. Whatever the best factor\n"
            "looks like below, it is nothing.\n"
        )

    report = run_factor_lab(
        returns,
        n_factors=args.n_factors,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(report.summary())
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Project a savings plan, then show the distribution the projection hides."""
    from quantos.planning import investment_schedule, required_contribution, simulate_plan

    plan = investment_schedule(
        args.start,
        args.years,
        args.rate,
        contribution=args.contribution,
        frequency=args.frequency,
    )
    print(plan.summary())

    if args.volatility <= 0:
        print("\npass --volatility to see the distribution behind the projection")
        return 0

    outcome = simulate_plan(
        args.start,
        args.years,
        args.rate,
        args.volatility,
        contribution=args.contribution,
        frequency=args.frequency,
        n_paths=args.paths,
        seed=args.seed,
    )
    print()
    print(outcome.summary())

    if args.target is not None:
        needed = required_contribution(
            args.start, args.years, args.rate, args.target, frequency=args.frequency
        )
        print(
            f"\nto reach {args.target:,.0f} at a fixed {args.rate:.2%}, "
            f"contribute {needed:,.2f} per period"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the local research viewer: a search bar for the whole pipeline."""
    from quantos.web.server import serve

    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_doctor(args: argparse.Namespace) -> int:
    """Import every module and report the environment."""
    import importlib
    import pkgutil

    import quantos

    print(_heading("QuantOS environment check"))
    print(f"quantos              {__version__}")
    print(f"python               {sys.version.split()[0]}")
    print(f"numpy                {np.__version__}")
    try:
        import scipy

        print(
            f"scipy                {scipy.__version__}  "
            f"(test-only oracle, not a runtime dependency)"
        )
    except ImportError:
        print("scipy                not installed (correct for a runtime install)")

    print(_heading("Module import check"))
    failures = []
    modules = sorted(
        module.name for module in pkgutil.walk_packages(quantos.__path__, prefix="quantos.")
    )
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"  ok    {name}")
        except Exception as error:
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
            failures.append(name)
    print(f"\n{len(modules) - len(failures)}/{len(modules)} modules import cleanly")
    return 1 if failures else 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a short tour of every subsystem."""
    print(_heading("QuantOS tour"))
    print(
        "Running one demonstration from each subsystem. Every number below is\n"
        f"reproducible with --seed {args.seed}.\n"
    )
    steps = [
        ("order book", cmd_book, {"operations": 200_000, "seed": args.seed}),
        (
            "options",
            cmd_options,
            {
                "spot": 100.0,
                "strike": 100.0,
                "maturity": 1.0,
                "volatility": 0.2,
                "rate": 0.05,
                "dividend": 0.0,
            },
        ),
        ("probability", cmd_probability, {"samples": 50_000, "seed": args.seed, "problem": None}),
        (
            "validation",
            cmd_validate,
            {"seed": args.seed, "configurations": 200, "edge": 0.0025},
        ),
        ("portfolio", cmd_portfolio, {"seed": args.seed, "assets": 60, "train": 120}),
        (
            "execution",
            cmd_execution,
            {
                "quantity": 1_000_000.0,
                "horizon": 1.0,
                "volatility": 0.02,
            },
        ),
        (
            "market simulation",
            cmd_simulate,
            {
                "scenario": "liquid",
                "seconds": args.seconds,
                "seed": args.seed,
                "output": None,
            },
        ),
    ]
    for _, handler, kwargs in steps:
        code = handler(argparse.Namespace(**kwargs))
        if code != 0:
            return code
    print(_heading("Tour complete"))
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser, including every subcommand."""
    parser = argparse.ArgumentParser(
        prog="quantos",
        description="QuantOS -- a quantitative research platform.",
        epilog="Every subcommand is deterministic given --seed.",
    )
    parser.add_argument("--version", action="version", version=f"quantos {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error"],
        help=(
            "diagnostic detail on stderr; defaults to QUANTOS_LOG_LEVEL, then "
            "warning. Use debug to see cache hits, fetch timings and why a "
            "factor was skipped."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="short tour of every subsystem")
    p.add_argument("--seed", type=int, default=20240719)
    p.add_argument("--seconds", type=float, default=5.0, help="market seconds to simulate")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("book", help="order book throughput and invariant check")
    p.add_argument("--operations", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=20240719)
    p.set_defaults(func=cmd_book)

    p = sub.add_parser("simulate", help="agent-based market simulation")
    p.add_argument("--scenario", default="liquid", choices=["liquid", "stressed"])
    p.add_argument("--seconds", type=float, default=30.0, help="market seconds")
    p.add_argument("--seed", type=int, default=20240719)
    p.add_argument("--output", default=None, help="directory for SVG charts")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("options", help="Black-Scholes prices, Greeks, implied vol")
    p.add_argument("--spot", type=float, default=100.0)
    p.add_argument("--strike", type=float, default=100.0)
    p.add_argument("--maturity", type=float, default=1.0)
    p.add_argument("--volatility", type=float, default=0.2)
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--dividend", type=float, default=0.0)
    p.set_defaults(func=cmd_options)

    p = sub.add_parser("probability", help="verify the probability lab")
    p.add_argument("--samples", type=int, default=100_000)
    p.add_argument("--problem", default=None, help="substring filter on problem name")
    p.add_argument("--seed", type=int, default=20240719)
    p.set_defaults(func=cmd_probability)

    p = sub.add_parser("validate", help="backtest overfitting controls")
    p.add_argument("--configurations", type=int, default=500)
    p.add_argument(
        "--edge",
        type=float,
        default=0.0006,
        help="daily drift of the one genuinely profitable configuration "
        "(default 0.0006; try 0.0025 for a detectable edge)",
    )
    p.add_argument("--seed", type=int, default=20240719)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("portfolio", help="portfolio construction out of sample")
    p.add_argument("--assets", type=int, default=80)
    p.add_argument("--train", type=int, default=150)
    p.add_argument("--seed", type=int, default=20240719)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("execution", help="Almgren-Chriss execution frontier")
    p.add_argument("--quantity", type=float, default=1_000_000.0)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--volatility", type=float, default=0.02)
    p.set_defaults(func=cmd_execution)

    p = sub.add_parser("analyse", help="run the toolkit on real market/macro data")
    p.add_argument(
        "--bundle",
        default=None,
        help="a named bundle: equity, rates, risk-appetite, inflation, macro, crossasset",
    )
    p.add_argument(
        "--series",
        default=None,
        help="comma-separated catalogue keys or raw FRED ids (e.g. spx,vix,DGS10)",
    )
    p.add_argument(
        "--csv", default=None, help="comma-separated CSV files for stocks/ETFs FRED does not carry"
    )
    p.add_argument("--start", default="2015-01-01", help="earliest date to include")
    p.add_argument("--list", action="store_true", help="list the catalogue and exit")
    p.add_argument("--offline", action="store_true", help="use only cached data")
    p.add_argument("--output", default=None, help="directory for SVG charts")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("research", help="full research report on any ticker")
    p.add_argument(
        "--ticker",
        default=None,
        help="any listed symbol: AAPL, SPY, ^GSPC, VOD.L, BTC-USD. No API key needed.",
    )
    p.add_argument(
        "--range",
        default="10y",
        choices=["1y", "2y", "5y", "10y", "20y"],
        help="how much history to request for --ticker",
    )
    p.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    p.add_argument("--csv", default=None, help="CSV file for a stock, ETF or future")
    p.add_argument("--series", default=None, help="a catalogue key or FRED id")
    p.add_argument("--symbol", default=None, help="override the symbol name")
    p.add_argument(
        "--asset-class",
        default="equity",
        choices=["equity", "etf", "index", "rate", "future", "commodity", "fx"],
        help="what kind of instrument the CSV holds",
    )
    p.add_argument("--start", default="2015-01-01")
    p.add_argument(
        "--cost-bps", type=float, default=5.0, help="round-trip cost charged to every signal"
    )
    p.add_argument("--no-signals", action="store_true", help="skip the signal battery")
    p.add_argument("--no-factors", action="store_true", help="skip the factor regression")
    p.add_argument("--markdown", default=None, help="also write a Markdown report here")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("forward", help="append-only forward testing: predict today, score later")
    p.add_argument("--ticker", default=None, help="any listed symbol, e.g. --ticker AAPL")
    p.add_argument("--csv", default=None, help="price file to form predictions from")
    p.add_argument("--symbol", default=None)
    p.add_argument("--offline", action="store_true", help="use only cached prices")
    p.add_argument("--signal", default=None, help="restrict scoring to one signal")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--horizon", type=int, default=30, help="calendar days to hold")
    p.add_argument("--ledger", default=None, help="ledger path (default ~/.quantos)")
    p.add_argument(
        "--score-only", action="store_true", help="report the existing record, write nothing"
    )
    p.set_defaults(func=cmd_forward)

    p = sub.add_parser("surface", help="fit an SVI volatility surface to an option chain")
    p.add_argument("--chain", required=True, help="option chain CSV")
    p.add_argument("--symbol", default=None)
    p.add_argument("--spot", type=float, default=None, help="underlying price")
    p.add_argument(
        "--as-of",
        default=None,
        help="quote date (YYYY-MM-DD); defaults to today. Needed for a historical chain.",
    )
    p.add_argument("--rate", type=float, default=0.0)
    p.add_argument("--keep-itm", action="store_true", help="do not filter in-the-money quotes")
    p.add_argument(
        "--underlying", default=None, help="price CSV, to measure the variance risk premium"
    )
    p.add_argument("--realised-days", type=int, default=63)
    p.set_defaults(func=cmd_surface)

    p = sub.add_parser("intraday", help="noise-corrected intraday volatility estimation")
    p.add_argument("--csv", required=True, help="intraday bars or ticks")
    p.add_argument("--symbol", default=None)
    p.add_argument("--price-column", default=None, help="override the price column")
    p.add_argument("--session", type=int, default=0, help="which session to analyse in detail")
    p.add_argument("--compare", default=None, help="second intraday file, for the Epps curve")
    p.set_defaults(func=cmd_intraday)

    from quantos.data.align import MACRO_FACTORS

    p = sub.add_parser("scenario", help="what happens if rates fall 100bps, as a range")
    p.add_argument("--ticker", required=True)
    p.add_argument("--range", default="20y")
    p.add_argument(
        "--factors",
        nargs="*",
        default=["rates"],
        choices=sorted(MACRO_FACTORS),
        help="macro factors to estimate sensitivities to",
    )
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("lattice", help="binomial and trinomial option pricing")
    p.add_argument("--spot", type=float, default=100.0)
    p.add_argument("--strike", type=float, default=105.0)
    p.add_argument("--expiry", type=float, default=1.0, help="years")
    p.add_argument("--volatility", type=float, default=0.25)
    p.add_argument("--rate", type=float, default=0.04)
    p.add_argument("--dividend-yield", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--put", action="store_true")
    p.add_argument("--american", action="store_true")
    p.set_defaults(func=cmd_lattice)

    p = sub.add_parser("stress", help="replay an instrument through real historical crises")
    p.add_argument("--ticker", required=True)
    p.add_argument("--range", default="20y", help="history length; 20y reaches 2008")
    p.add_argument("--against", nargs="*", default=None, help="assets to correlate against")
    p.add_argument("--risk", nargs="*", default=None, help="which of those are risk assets")
    p.set_defaults(func=cmd_stress)

    p = sub.add_parser("factors", help="search a large factor grid, corrected for the search")
    p.add_argument("--ticker", default=None, help="instrument to search; omit for pure noise")
    p.add_argument("--n-factors", type=int, default=None, help="grid subset; omit for all 840")
    p.add_argument("--alpha", type=float, default=0.05, help="family-wise error rate")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--n-synthetic", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_factors)

    p = sub.add_parser("plan", help="investment projection, with the risk the projection hides")
    p.add_argument("--start", type=float, default=20_000.0, help="starting amount")
    p.add_argument("--years", type=float, default=10.0)
    p.add_argument("--rate", type=float, default=0.06, help="annual return, e.g. 0.06")
    p.add_argument("--contribution", type=float, default=1_000.0, help="added each period")
    p.add_argument("--frequency", default="monthly", help="monthly, quarterly, annually, ...")
    p.add_argument("--volatility", type=float, default=0.15, help="annual volatility; 0 disables")
    p.add_argument("--target", type=float, default=None, help="solve for the contribution needed")
    p.add_argument("--paths", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("serve", help="search bar: type a ticker in your browser")
    p.add_argument("--host", default="127.0.0.1", help="bind address (localhost by default)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="environment and import check")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``quantos`` console script.

    The CLI is the application, so it is the one place in this package that
    configures logging -- a library that does so at import steals the root
    logger from whatever imported it.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logs go to stderr, so a report on stdout stays pipeable.
    configure_logging(getattr(args, "log_level", None))
    _LOG.debug("running %s", getattr(args, "command", "?"))

    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (MarketDataError, FredError) as error:
        # A data failure is the common one and is not a bug. Report it as a
        # message rather than a traceback, which tells the user nothing they
        # can act on, and log the detail for anyone who wants it.
        _LOG.debug("data error", exc_info=True)
        print(f"\ncould not fetch data: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
