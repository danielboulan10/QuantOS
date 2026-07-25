#!/usr/bin/env python3
"""Measure the distribution of price-discovery efficiency across seeds.

Why this script exists
----------------------
A single simulation run reports one correlation between the market price and the
latent fundamental. Quoting that number is exactly the selective reporting this
repository is built to guard against -- and it caught us: an early README quoted
0.78, which turned out to be the best of the three seeds that had been run. The
same configuration produces 0.24 on another seed.

So the honest summary of an agent-based model's realism is a *distribution* over
seeds, not a point. This script computes it, and the README quotes what it finds.

Usage
-----
    python scripts/measure_price_discovery.py --seeds 20 --seconds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantos.research.features.microstructure import (
    price_discovery_efficiency,
)
from quantos.sim.agents import MarketMaker
from quantos.sim.scenarios import build_liquid_market
from quantos.sim.stylized_facts import analyse_stylized_facts


def measure_one(seed: int, seconds: float, learning_rate: float | None = None) -> dict[str, float]:
    """Run one simulation and return its discovery and stylised-fact metrics.

    ``learning_rate`` overrides the market makers' Glosten-Milgrom order-flow
    learning rate, so the ablation (0.0 versus the default) can be run from the
    command line.
    """
    simulation = build_liquid_market(duration_ns=int(seconds * 1e9), seed=seed)
    if learning_rate is not None:
        for agent in simulation.agents.values():
            if isinstance(agent, MarketMaker):
                agent.learning_rate = learning_rate
    result = simulation.run()
    mid = result.mid_series()
    stamps, values = result.fundamental_path
    grid = np.arange(mid.size) * result.config.snapshot_interval_ns
    latent = np.interp(grid, stamps, values)
    finite = np.isfinite(mid)

    if int(np.sum(finite)) < 200:
        # The book stayed one-sided for most of the run, so the mid is largely
        # unobservable. Reporting a correlation from a handful of points would be
        # worse than reporting nothing.
        raise RuntimeError(
            f"seed {seed}: only {int(np.sum(finite))} observable mid prices; "
            "the book was one-sided for most of the run"
        )
    discovery = price_discovery_efficiency(mid[finite], latent[finite])
    facts = analyse_stylized_facts(result.returns())
    return {
        "seed": float(seed),
        "correlation": discovery.correlation,
        "beta": discovery.beta,
        "rmse": discovery.rmse,
        "fundamental_range": float(values.max() - values.min()),
        "price_range": float(np.nanmax(mid) - np.nanmin(mid)),
        "spread": float(np.nanmean(result.spread_series())),
        "trades": float(len(result.trades)),
        "facts_score": facts.score,
        "excess_kurtosis": facts["fat_tails"].value,
        "vol_clustering": facts["volatility_clustering"].value,
        "hurst": facts["long_memory_volatility"].value,
    }


def summarise(name: str, values: np.ndarray) -> str:
    """One row of the distribution table."""
    return (
        f"  {name:<20} mean {values.mean():>8.3f}   median {np.median(values):>8.3f}   "
        f"min {values.min():>8.3f}   max {values.max():>8.3f}   sd {values.std(ddof=1):>7.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="override the market makers' order-flow learning rate; pass 0.0 to "
        "ablate Glosten-Milgrom learning entirely",
    )
    args = parser.parse_args()

    print(
        f"Measuring {args.seeds} independent runs of build_liquid_market at "
        f"{args.seconds}s of market time each.\n"
    )
    rows: list[dict[str, float]] = []
    for i in range(args.seeds):
        seed = args.start_seed + i
        try:
            row = measure_one(seed, args.seconds, args.learning_rate)
        except RuntimeError as error:
            print(f"  seed {seed:>4}  SKIPPED: {error}", flush=True)
            continue
        rows.append(row)
        print(
            f"  seed {seed:>4}  corr {row['correlation']:+.4f}  beta {row['beta']:+.4f}  "
            f"rmse {row['rmse']:6.2f}  fundamental {row['fundamental_range']:6.1f}  "
            f"facts {row['facts_score']:.2f}",
            flush=True,
        )

    print(f"\nDistribution over {args.seeds} seeds")
    print("-" * 100)
    for key in (
        "correlation",
        "beta",
        "rmse",
        "fundamental_range",
        "price_range",
        "spread",
        "excess_kurtosis",
        "vol_clustering",
        "hurst",
        "facts_score",
    ):
        print(summarise(key, np.array([r[key] for r in rows])))

    correlations = np.array([r["correlation"] for r in rows])
    print(f"\nFraction of runs with correlation > 0.5: {float(np.mean(correlations > 0.5)):.2f}")
    print(f"Fraction of runs with correlation > 0.0: {float(np.mean(correlations > 0.0)):.2f}")

    # Is dispersion explained by how far the fundamental happened to travel? A
    # fundamental that barely moves gives the market nothing to discover, so the
    # correlation is dominated by microstructure noise.
    ranges = np.array([r["fundamental_range"] for r in rows])
    if ranges.std() > 0:
        print(
            f"\ncorr(discovery correlation, fundamental range) = "
            f"{float(np.corrcoef(correlations, ranges)[0, 1]):+.3f}"
        )
        print(
            "A positive value means the runs where the fundamental moved further\n"
            "are the runs where the market tracked it better -- i.e. the dispersion\n"
            "is largely about how much signal there was to find, not instability\n"
            "in the market mechanism."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
