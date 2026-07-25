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

if TYPE_CHECKING:
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

    p = sub.add_parser("doctor", help="environment and import check")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``quantos`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
