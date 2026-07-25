#!/usr/bin/env python3
"""Performance benchmarks with recorded metadata.

Every benchmark records the interpreter, platform and NumPy version alongside the
timing, because a number without its environment is not comparable to anything.

Run:  python benchmarks/run_benchmarks.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import contextlib

from quantos import __version__
from quantos.core.types import AgentId, Order, OrderId, Quantity, Side, Ticks
from quantos.derivatives.black_scholes import black_scholes_price, implied_volatility
from quantos.exchange.book import LimitOrderBook, OrderNotFound


@dataclass
class Result:
    name: str
    operations: int
    seconds: float
    unit: str = "ops/s"
    notes: str = ""

    @property
    def rate(self) -> float:
        return self.operations / self.seconds if self.seconds > 0 else float("inf")


@dataclass
class Suite:
    metadata: dict[str, str] = field(default_factory=dict)
    results: list[Result] = field(default_factory=list)


def environment() -> dict[str, str]:
    return {
        "quantos": __version__,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def bench_order_book(n: int = 500_000) -> Result:
    """Mixed add/cancel/amend, the realistic workload.

    95%+ of real orders are cancelled rather than filled, so this weights cancels
    heavily -- which is what makes the O(1) intrusive-list cancel worth having.
    """
    book = LimitOrderBook()
    rng = random.Random(12345)
    live: list[int] = []
    order_id = 0
    start = time.perf_counter()
    for _ in range(n):
        roll = rng.random()
        if roll < 0.55 or not live:
            order_id += 1
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            price = rng.randint(9_950, 10_050)
            opposing = book.best_ask if side is Side.BUY else book.best_bid
            if opposing is not None:
                price = (
                    min(price, int(opposing) - 1)
                    if side is Side.BUY
                    else max(price, int(opposing) + 1)
                )
            if price <= 0:
                continue
            try:
                book.add(
                    Order(
                        OrderId(order_id),
                        AgentId("a"),
                        side,
                        Quantity(rng.randint(1, 100)),
                        Ticks(price),
                    )
                )
                live.append(order_id)
            except Exception:
                pass
        elif roll < 0.90:
            with contextlib.suppress(OrderNotFound):
                book.cancel(OrderId(live.pop(rng.randrange(len(live)))))
        else:
            target = live[rng.randrange(len(live))]
            if target in book:
                book.amend(OrderId(target), Quantity(rng.randint(1, 120)))
    elapsed = time.perf_counter() - start
    book.check_invariants()
    return Result(
        "order_book_mixed", n, elapsed, notes=f"{len(book):,} resting, heap slack {book.heap_slack}"
    )


def bench_order_book_cancel_only(n: int = 200_000) -> Result:
    """Isolates cancellation, the operation the data structure is designed for."""
    book = LimitOrderBook()
    for i in range(1, n + 1):
        book.add(Order(OrderId(i), AgentId("a"), Side.BUY, Quantity(1), Ticks(10_000 - (i % 40))))
    order = list(range(1, n + 1))
    random.Random(7).shuffle(order)  # cancel from random queue positions
    start = time.perf_counter()
    for oid in order:
        book.cancel(OrderId(oid))
    elapsed = time.perf_counter() - start
    return Result(
        "order_book_cancel_random_position",
        n,
        elapsed,
        notes="O(1) via intrusive linked list + id map",
    )


def bench_matching(n: int = 100_000) -> Result:
    from quantos.core.types import OrderType, TimeInForce
    from quantos.exchange.matching import MatchingEngine

    engine = MatchingEngine()
    for i in range(1, 20_001):
        engine.submit(
            Order(OrderId(i), AgentId("mm"), Side.SELL, Quantity(5), Ticks(10_000 + (i % 200)))
        )
    start = time.perf_counter()
    for i in range(1_000_000, 1_000_000 + n):
        engine.submit(
            Order(
                OrderId(i),
                AgentId("t"),
                Side.BUY,
                Quantity(1),
                None,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
            )
        )
        engine.drain_maker_fills()
    elapsed = time.perf_counter() - start
    return Result(
        "matching_market_orders", n, elapsed, notes=f"{len(engine.trades):,} trades printed"
    )


def bench_special_functions(n: int = 5_000_000) -> Result:
    from quantos.core import special as sf

    x = np.linspace(-8.0, 8.0, n)
    start = time.perf_counter()
    sf.ndtr(x)
    elapsed = time.perf_counter() - start
    return Result(
        "ndtr_vectorised",
        n,
        elapsed,
        unit="values/s",
        notes="erfc continued fraction + series, NumPy only",
    )


def bench_black_scholes(n: int = 2_000_000) -> Result:
    rng = np.random.default_rng(0)
    spot = rng.uniform(50, 150, n)
    start = time.perf_counter()
    black_scholes_price(spot, 100.0, 1.0, 0.2, rate=0.05)
    elapsed = time.perf_counter() - start
    return Result("black_scholes_vectorised", n, elapsed, unit="prices/s")


def bench_implied_vol(n: int = 20_000) -> Result:
    """Scalar and iterative, so much slower -- and worth measuring separately."""
    rng = np.random.default_rng(0)
    strikes = rng.uniform(60, 160, n)
    prices = [float(black_scholes_price(100.0, k, 1.0, 0.25, rate=0.05)) for k in strikes]
    start = time.perf_counter()
    for k, p in zip(strikes, prices, strict=False):
        implied_volatility(p, 100.0, float(k), 1.0, rate=0.05)
    elapsed = time.perf_counter() - start
    return Result(
        "implied_volatility_scalar",
        n,
        elapsed,
        unit="solves/s",
        notes="safeguarded Newton, converged to 1e-12 in sigma",
    )


def bench_simulation(seconds: float = 5.0) -> Result:
    from quantos.sim.scenarios import build_liquid_market

    start = time.perf_counter()
    result = build_liquid_market(duration_ns=int(seconds * 1e9), seed=1).run()
    elapsed = time.perf_counter() - start
    return Result(
        "market_simulation_events",
        result.events_processed,
        elapsed,
        unit="events/s",
        notes=f"{len(result.trades):,} trades, {seconds}s market time",
    )


BENCHMARKS = [
    bench_order_book,
    bench_order_book_cancel_only,
    bench_matching,
    bench_special_functions,
    bench_black_scholes,
    bench_implied_vol,
    bench_simulation,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="skip the slow ones")
    args = parser.parse_args()

    suite = Suite(metadata=environment())
    print("QuantOS benchmarks")
    print("=" * 72)
    for key, value in suite.metadata.items():
        print(f"  {key:<16} {value}")
    print("=" * 72)
    print(f"{'benchmark':<38}{'rate':>16}{'unit':>12}")
    print("-" * 72)

    for benchmark in BENCHMARKS:
        if args.quick and benchmark in (bench_simulation, bench_implied_vol):
            continue
        result = benchmark()
        suite.results.append(result)
        print(f"{result.name:<38}{result.rate:>16,.0f}{result.unit:>12}")
        if result.notes:
            print(f"{'':<38}  {result.notes}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "metadata": suite.metadata,
                    "results": [asdict(r) | {"rate": r.rate} for r in suite.results],
                },
                indent=2,
            )
        )
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
