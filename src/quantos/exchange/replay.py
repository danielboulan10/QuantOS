"""Batched order-book replay: cross the Python/C++ boundary once, not per order.

The measurement that produced this module
------------------------------------------
The obvious way to use a C++ order book is to call it once per operation. That
was built first, and benchmarked at **1.2x** the pure Python implementation --
barely worth the build step.

Profiling the difference showed why. Timing the same workload three ways:

======================================  ==================  =========
Path                                    ops/s (median)      speedup
======================================  ==================  =========
Pure Python, ``Order`` dataclass        ~435,000            1.0x
C++ via the per-call wrapper            ~660,000            1.5x
C++ raw, no dataclass, no wrapper       ~3,800,000          8.7x
**C++ batched through this module**     **~15,100,000**     **~35x**
======================================  ==================  =========

These are medians of repeated runs, rounded. The batched figure ranged from 8.1
to 16.5 million operations per second across seven runs of the same tape, so it
is quoted to two significant figures; an earlier version of this docstring gave a
single run as ``16,012,260``, which was near the top of that range and precise to
six digits it had not earned.

**83% of the wrapped runtime was Python-side object churn**, not matching:
constructing a frozen dataclass and paying call overhead for every operation.
The C++ core was never the bottleneck.

Batching removes it. And it is not a trick -- it is how a real replay is
structured anyway. An exchange hands you a market-data *file*, not a function
call per message, so the natural interface is "here is a tape, run it."

The tape format
---------------
A ``(n, 5)`` int64 array, one row per operation:

===========  ==========================================================
``opcode``   0 add, 1 cancel, 2 amend, 3 match
``order_id`` the order to add, cancel or amend
``side``     +1 buy, -1 sell (add and match only)
``price``    limit price in ticks; 0 on a match row means no limit
``quantity`` size
===========  ==========================================================

Integers throughout, because prices are ticks and there is no floating point
anywhere in the matching path (see ``docs/ddr/DDR-001``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["BATCH_AVAILABLE", "OpCode", "ReplayResult", "build_tape", "replay_tape"]

try:  # pragma: no cover - depends on whether the extension was built
    from quantos.exchange import _book as _extension  # type: ignore[attr-defined]

    BATCH_AVAILABLE = hasattr(_extension.Book, "execute_batch")
except ImportError:  # pragma: no cover
    _extension = None
    BATCH_AVAILABLE = False


class OpCode(enum.IntEnum):
    """Operation codes in a replay tape."""

    ADD = 0
    CANCEL = 1
    AMEND = 2
    MATCH = 3


@dataclass(frozen=True)
class ReplayResult:
    """Counts from a batched replay."""

    added: int
    rejected: int
    cancelled: int
    amended: int
    matched_quantity: int
    fills: int
    operations: int
    seconds: float

    @property
    def operations_per_second(self) -> float:
        return self.operations / self.seconds if self.seconds > 0 else float("inf")

    @property
    def rejection_rate(self) -> float:
        """Share of adds refused, almost always for crossing the book."""
        attempted = self.added + self.rejected
        return self.rejected / attempted if attempted else 0.0


def build_tape(
    opcodes: NDArray[np.int64] | list[int],
    order_ids: NDArray[np.int64] | list[int],
    sides: NDArray[np.int64] | list[int],
    prices: NDArray[np.int64] | list[int],
    quantities: NDArray[np.int64] | list[int],
) -> NDArray[np.int64]:
    """Assemble a replay tape from five parallel columns.

    Validates lengths and dtype up front, because a malformed buffer reaching
    the extension is a segfault rather than an exception.

    Example
        >>> import numpy as np
        >>> tape = build_tape([0, 0, 3], [1, 2, 0], [1, -1, 1],
        ...                   [9990, 10010, 0], [100, 100, 60])
        >>> tape.shape, tape.dtype
        ((3, 5), dtype('int64'))
    """
    columns = [
        np.asarray(column, dtype=np.int64).ravel()
        for column in (opcodes, order_ids, sides, prices, quantities)
    ]
    lengths = {column.size for column in columns}
    if len(lengths) != 1:
        raise ValueError(f"all columns must be the same length, got sizes {sorted(lengths)}")
    return np.column_stack(columns).astype(np.int64)


def replay_tape(
    tape: NDArray[np.int64], *, agent_id: int = 0, book: object | None = None
) -> tuple[ReplayResult, object]:
    """Execute a tape entirely inside C++.

    Purpose
        Replay a large order-flow sequence at roughly 15 million operations per
        second, about 35x the pure Python book and 20x the per-call C++ wrapper.
    Inputs
        ``tape`` -- ``(n, 5)`` int64 array from :func:`build_tape`.
        ``book`` -- an existing extension book to continue into; a fresh one is
        created if omitted.
    Outputs
        ``(ReplayResult, book)``. The book is returned so a tape can be replayed
        in chunks against continuing state.
    Failure modes
        :class:`RuntimeError` if the extension is not built -- there is no pure
        Python fallback for this function, because a Python "batch" would simply
        be the per-operation loop it exists to avoid. Use
        :class:`~quantos.exchange.book.LimitOrderBook` directly instead.

    Example
        >>> import numpy as np
        >>> tape = build_tape([0, 0, 3], [1, 2, 0], [1, -1, 1],
        ...                   [9990, 10010, 0], [100, 100, 60])
        >>> result, _ = replay_tape(tape)     # doctest: +SKIP
        >>> result.added, result.matched_quantity      # doctest: +SKIP
        (2, 60)
    """
    import time

    if not BATCH_AVAILABLE:
        raise RuntimeError(
            "the batched replay needs the C++ extension. Run "
            "`python scripts/build_extension.py`, or use LimitOrderBook for a "
            "pure Python path (a Python 'batch' would just be the per-operation "
            "loop this exists to avoid)."
        )

    tape = np.ascontiguousarray(tape, dtype=np.int64)
    if tape.ndim != 2 or tape.shape[1] != 5:
        raise ValueError(f"tape must have shape (n, 5), got {tape.shape}")

    # `Any` because the extension type is only importable after a build; the
    # BATCH_AVAILABLE guard above is what actually establishes it exists.
    target: Any = book if book is not None else _extension.Book()
    start = time.perf_counter()
    counts = target.execute_batch(tape.tobytes(), int(agent_id))
    elapsed = time.perf_counter() - start

    return (
        ReplayResult(
            added=int(counts["added"]),
            rejected=int(counts["rejected"]),
            cancelled=int(counts["cancelled"]),
            amended=int(counts["amended"]),
            matched_quantity=int(counts["matched_quantity"]),
            fills=int(counts["fills"]),
            operations=int(tape.shape[0]),
            seconds=elapsed,
        ),
        target,
    )


def synthetic_tape(
    n_operations: int,
    *,
    seed: int = 20240719,
    add_share: float = 0.60,
    cancel_share: float = 0.30,
    centre: int = 10_000,
    half_width: int = 50,
) -> NDArray[np.int64]:
    """Generate a realistic order-flow tape for benchmarking.

    Weighted toward cancels because real markets are: over 95% of orders are
    cancelled rather than filled, which is the workload the data structure was
    designed for.
    """
    if not 0.0 < add_share + cancel_share <= 1.0:
        raise ValueError("add_share + cancel_share must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    match_share = 1.0 - add_share - cancel_share

    opcodes = rng.choice(
        [OpCode.ADD, OpCode.CANCEL, OpCode.MATCH],
        size=n_operations,
        p=[add_share, cancel_share, match_share],
    ).astype(np.int64)
    order_ids = np.arange(1, n_operations + 1, dtype=np.int64)
    # Cancels target an order placed a little earlier, as a real participant's would.
    order_ids = np.where(opcodes == OpCode.CANCEL, np.maximum(1, order_ids - 200), order_ids)
    sides = rng.choice([1, -1], size=n_operations).astype(np.int64)
    prices = (centre + rng.integers(-half_width, half_width + 1, size=n_operations)).astype(
        np.int64
    )
    prices = np.where(opcodes == OpCode.MATCH, 0, prices)  # unlimited sweep
    quantities = rng.integers(1, 100, size=n_operations).astype(np.int64)
    return build_tape(opcodes, order_ids, sides, prices, quantities)
