# DDR-001: Prices are integers, timestamps are integer nanoseconds

- **Status:** Accepted
- **Affects:** `core/types.py`, all of `exchange/`, all of `sim/`

## Context

Prices could be `float` (natural to read, matches market data feeds) or integer
tick counts (matches how exchanges actually work internally).

## Decision

Every price in QuantOS is an integer count of ticks. Every timestamp and duration
is an integer nanosecond count. Conversion to decimals happens only at the
presentation boundary, through `TickSizeRule`.

## Rationale

An order book is fundamentally a hash map keyed by price. If `0.1 + 0.2 != 0.3`,
two orders a trader intends to rest at the same level can land in different
buckets — and the bug is invisible until a level fails to aggregate correctly
under a specific sequence of arithmetic. Real exchanges quote in integer ticks for
exactly this reason.

Timestamps have the same problem plus one more: a discrete-event simulator must
order simultaneous events *deterministically*, and float timestamps break ties
inconsistently across platforms and compiler flags. `SimulationClock` keys its
heap on `(timestamp, priority, sequence)` with all three integral, which makes the
ordering total and the simulation bit-reproducible.

## Consequences

- **Positive:** exact price equality, exact level aggregation, exact tie-breaking,
  bit-reproducible simulations, and `Ticks`/`Nanos` `NewType`s that make a unit
  mix-up a type error rather than a silent 100x.
- **Negative:** every log line and chart axis needs conversion, and a reader
  seeing `10001` must know the tick size to interpret it. Accepted: the conversion
  is one call and lives at the boundary.
- **Negative:** instruments with different tick sizes cannot share a naive
  comparison. `TickSizeRule` is per-instrument for this reason.

## Alternatives considered

**Decimal.** Correct but 50-100x slower than `int` for arithmetic, which matters
in a matching loop.

**Floats with an epsilon comparison.** Rejected: it makes price equality
non-transitive, so `a == b` and `b == c` no longer imply `a == c`, and a hash map
keyed on such a comparison is not well defined.

**Fixed-point floats (price * 100 stored as float).** Rejected as strictly worse
than integers with no compensating benefit.
