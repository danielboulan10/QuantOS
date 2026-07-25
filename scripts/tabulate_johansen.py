#!/usr/bin/env python3
"""Tabulate critical values for the Johansen trace and max-eigenvalue statistics.

Why this script exists
----------------------
Published Johansen critical-value tables are specific to the *deterministic
specification* of the VECM -- no constant, unrestricted constant, constant
restricted to the cointegrating space, and the trend variants -- and the tables
are widely reproduced without their specification attached. Using the wrong one
is not a small error: it produced a 28% false-positive rate against a nominal
5% during the development of :mod:`quantos.core.timeseries.cointegration`.

Rather than trust a remembered table, we simulate the null distribution of the
statistic *as our own estimator computes it*. The asymptotic distribution
depends only on ``k - r`` (the number of common stochastic trends under the
null) and the deterministic terms, so one table serves every application.

The output is pasted into ``quantos/core/timeseries/cointegration.py`` with a
pointer back to this script, so the numbers are auditable and regenerable
rather than magic.

Usage
-----
    python scripts/tabulate_johansen.py --replications 20000 --sample 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantos.core.rng import SeedBank
from quantos.core.timeseries.cointegration import _johansen_statistics

LEVELS = (0.90, 0.95, 0.99)


def simulate_null(
    n_series: int, n_obs: int, replications: int, lags: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trace and max-eigenvalue statistics for r = 0 under independent walks.

    Under the null of *no* cointegration among ``k`` series, each series is an
    independent random walk. The r=0 statistic from such a system is a draw from
    the null distribution for ``k - r = k``.
    """
    bank = SeedBank(root=seed).child(f"johansen_k{n_series}")
    trace = np.empty(replications)
    max_eigen = np.empty(replications)
    for i in range(replications):
        rng = bank.child(f"rep_{i:07d}").generator()
        data = np.cumsum(rng.standard_normal((n_obs, n_series)), axis=0)
        stats = _johansen_statistics(data, lags=lags)
        trace[i] = stats.trace_statistics[0]
        max_eigen[i] = stats.max_eigen_statistics[0]
    return trace, max_eigen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=20000)
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--max-series", type=int, default=8)
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()

    print(
        f"# Simulated with scripts/tabulate_johansen.py: "
        f"{args.replications} replications, T={args.sample}, lags={args.lags}, "
        f"seed={args.seed}"
    )
    print(f"# Rows are k - r = 1 .. {args.max_series}; columns are the 90/95/99% quantiles.")

    trace_rows: list[list[float]] = []
    eigen_rows: list[list[float]] = []
    for k in range(1, args.max_series + 1):
        trace, max_eigen = simulate_null(k, args.sample, args.replications, args.lags, args.seed)
        trace_rows.append([float(np.quantile(trace, level)) for level in LEVELS])
        eigen_rows.append([float(np.quantile(max_eigen, level)) for level in LEVELS])
        print(
            f"k-r={k}: trace {np.round(trace_rows[-1], 4).tolist()}  "
            f"maxeig {np.round(eigen_rows[-1], 4).tolist()}",
            flush=True,
        )

    print("\n_JOHANSEN_TRACE_CRIT = np.array(")
    print("    [")
    for row in trace_rows:
        print(f"        [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")
    print("    ]\n)")
    print("\n_JOHANSEN_MAXEIG_CRIT = np.array(")
    print("    [")
    for row in eigen_rows:
        print(f"        [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")
    print("    ]\n)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
