# ERS-001: Master Architecture

## 1. Purpose

QuantOS is a research platform, not a trading system. Its purpose is to make
quantitative claims *checkable*: to provide an environment where an estimator can
be validated against a known answer, a strategy's edge can be distinguished from
selection bias, and a numerical routine's accuracy can be measured rather than
assumed.

Everything in the architecture follows from that.

## 2. Dependency structure

Packages form a strict layering, with dependencies pointing **inward**:

```
                          ┌───────────────────────────┐
                          │   cli  ·  viz  ·  api     │  presentation
                          └────────────┬──────────────┘
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
    ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────┐
    │  strategy  risk  │   │  research         │   │  derivatives     │
    │  execution       │   │  (features,       │   │  probability     │
    │                  │   │   journal)        │   │                  │
    └────────┬─────────┘   └─────────┬─────────┘   └────────┬─────────┘
             │                       │                      │
             └───────────┬───────────┴──────────────────────-┘
                         ▼
              ┌─────────────────────┐
              │        sim          │   agents, clock, world, scenarios
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │      exchange       │   book, matching, fees
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │        core         │   special, distributions, stats,
              │                     │   optimize, numerics, linalg,
              │                     │   timeseries, types, rng
              └─────────────────────┘
```

`core` depends on NumPy alone. Nothing depends on `cli` or `viz`. `exchange` knows
nothing about agents; `sim` knows nothing about strategies; `research` knows
nothing about how the data it analyses was produced, which is why the same
estimators run on simulated and (in principle) historical data.

`quantos doctor` walks the whole tree and imports every module, so a layering
violation that creates a cycle fails immediately.

## 3. Load-bearing decisions

Each is recorded as a DDR with alternatives and consequences:

| DDR | Decision |
|---|---|
| [DDR-001](../ddr/DDR-001-integer-prices.md) | Prices are integer ticks; timestamps are integer nanoseconds |
| [DDR-002](../ddr/DDR-002-numpy-only-runtime.md) | NumPy is the only runtime dependency; SciPy is a test oracle |
| [DDR-003](../ddr/DDR-003-agents-cannot-touch-the-venue.md) | Agents return actions; only the world touches the exchange |
| [DDR-004](../ddr/DDR-004-simulated-critical-values.md) | Critical values are simulated for our own estimator |

## 4. Interfaces

Extension happens by implementing one protocol, not by subclassing a framework:

| To add a… | Implement | Nothing else changes |
|---|---|---|
| distribution | `core.distributions.Distribution` | every MC engine, test and VaR path |
| agent | `sim.agents.Agent` (three handlers) | the world schedules and settles it |
| fee schedule | `exchange.fees.FeeModel` | the matching engine charges it |
| optimiser | the `OptimizeResult`-returning signature | every calibration |
| CV scheme | a `split(x)` generator | every validation routine |

## 5. Reproducibility

A research platform whose results cannot be reproduced is a source of anecdotes.
Three mechanisms:

1. **Hierarchical seeding.** `core.rng.SeedBank` derives every stream from one
   root by *semantic path*, so streams are independent, order-independent, and
   stable under refactoring.
2. **Integer time with total event ordering.** `sim.clock.SimulationClock` keys
   its heap on `(timestamp, priority, sequence)`, all integral, so simultaneous
   events resolve identically on every platform.
3. **CI enforcement.** The `reproducibility` job runs the same seed twice and
   requires byte-identical output.

## 6. Verification strategy

Four independent kinds of check, because different errors hide from different
tests:

| Kind | Catches | Example |
|---|---|---|
| Oracle comparison | Wrong formulas | `special` vs SciPy *and* `math` |
| Analytic ↔ Monte Carlo | Wrong derivations | the whole probability lab |
| Property-based invariants | Unconsidered sequences | order-book invariants under Hypothesis |
| Parameter recovery | Broken pipelines | GARCH, OU, Johansen, Kyle's lambda |
| **Size checks** | Wrong critical values | every statistical test's false-positive rate |

The last is the one most projects omit, and the one that caught the most serious
defect found during development (DDR-004).

Doctests run as part of the suite, so every worked example in every docstring is
executed. Several incorrect values were caught that way.

## 7. Non-goals

Stated explicitly, because scope creep in a project like this is unlimited:

- **Not a trading system.** No broker connectivity, no order routing to a real
  venue, no risk limits enforced in anger.
- **Not a data platform.** No vendor adapters, no tick store. Everything is
  simulated or synthetic, by design — ground truth is the point.
- **Not a machine-learning library.** The ML surface is deliberately thin;
  labelling and cross-validation are included because *those* are where financial
  ML goes wrong, not the model fitting.
- **Not optimised for latency.** Pure Python. The interfaces are shaped so a
  compiled core could replace the hot paths without an API change; that work has
  not been done, and the benchmarks exist to show where it would matter.

## 8. Why this exists

The recurring failure in quantitative research is not bad mathematics. It is
**unfalsifiable claims**: a backtest with no accounting for the number of
configurations tried, an estimator applied where its assumptions do not hold, a
critical value borrowed from a paper whose specification differed, a simulation
whose realism is asserted rather than measured.

Every subsystem here exists to make one such claim checkable. That is the design
principle, and where a choice traded convenience for checkability, checkability
won.
