"""The C++ book must be byte-identical to the Python book, or it is worthless.

Why this test carries the whole optional-backend design
-------------------------------------------------------
``docs/ddr/DDR-002`` rejects optional accelerators on the grounds that two code
paths produce two behaviours, and results depending on what happens to be
installed are worse than results that are merely slower.

The order book is the single exception, and the reason is narrow and checkable:
matching is **exact integer arithmetic**. No floating point appears anywhere in
``_book.cpp``. Two correct implementations must therefore agree exactly on every
input -- not approximately, not to a tolerance, exactly.

So the accelerator is only admissible if that claim is *tested*, which is what
this module does: identical random operation sequences into both books, then a
full structural comparison of the results. Without these tests the C++ backend
would be an unverified second behaviour, and DDR-002 would rule it out.
"""

from __future__ import annotations

import contextlib
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quantos.core.types import AgentId, Order, OrderId, Quantity, Side, Ticks
from quantos.exchange.book import BookError, LimitOrderBook, OrderNotFound
from quantos.exchange.fastbook import EXTENSION_AVAILABLE, FastLimitOrderBook

pytestmark = pytest.mark.skipif(
    not EXTENSION_AVAILABLE,
    reason="C++ extension not built; run `python scripts/build_extension.py`",
)


def snapshot(book) -> dict:
    """A complete, comparable description of a book's state."""
    return {
        "best_bid": book.best_bid,
        "best_ask": book.best_ask,
        "n_orders": len(book),
        "bid_depth": [
            (int(level.price), int(level.quantity), level.order_count)
            for level in book.depth(Side.BUY, 10_000)
        ],
        "ask_depth": [
            (int(level.price), int(level.quantity), level.order_count)
            for level in book.depth(Side.SELL, 10_000)
        ],
    }


OPERATIONS = st.one_of(
    st.tuples(
        st.just("add"),
        st.sampled_from([Side.BUY, Side.SELL]),
        st.integers(min_value=9_950, max_value=10_050),
        st.integers(min_value=1, max_value=100),
    ),
    st.tuples(st.just("cancel"), st.integers(min_value=0, max_value=999)),
    st.tuples(
        st.just("amend"),
        st.integers(min_value=0, max_value=999),
        st.integers(min_value=1, max_value=100),
    ),
    st.tuples(
        st.just("match"),
        st.sampled_from([Side.BUY, Side.SELL]),
        st.integers(min_value=1, max_value=200),
    ),
)


def replay(book, operations: list[tuple]) -> list:
    """Apply an operation sequence, returning the observable result of each."""
    live: list[int] = []
    next_id = 0
    trace: list = []

    for operation in operations:
        kind = operation[0]
        if kind == "add":
            _, side, price, size = operation
            next_id += 1
            opposing = book.best_ask if side is Side.BUY else book.best_bid
            if opposing is not None:
                price = (
                    min(price, int(opposing) - 1)
                    if side is Side.BUY
                    else max(price, int(opposing) + 1)
                )
            if price <= 0:
                trace.append(("add", "skipped"))
                continue
            try:
                book.add(
                    Order(
                        OrderId(next_id),
                        AgentId(f"a{next_id % 5}"),
                        side,
                        Quantity(size),
                        Ticks(price),
                    )
                )
                live.append(next_id)
                trace.append(("add", next_id, price, size))
            except BookError as error:
                trace.append(("add", "rejected", type(error).__name__))
        elif kind == "cancel":
            if not live:
                trace.append(("cancel", "none-live"))
                continue
            victim = live.pop(operation[1] % len(live))
            try:
                trace.append(("cancel", victim, int(book.cancel(OrderId(victim)))))
            except OrderNotFound:
                trace.append(("cancel", victim, "not-found"))
        elif kind == "amend":
            if not live:
                trace.append(("amend", "none-live"))
                continue
            target = live[operation[1] % len(live)]
            try:
                trace.append(
                    ("amend", target, int(book.amend(OrderId(target), Quantity(operation[2]))))
                )
            except OrderNotFound:
                trace.append(("amend", target, "not-found"))
        else:
            _, side, size = operation
            fills, remaining = book.match(side, Quantity(size), None, 0)
            trace.append(
                (
                    "match",
                    [(int(node.order_id), int(qty), int(node.price)) for node, qty in fills],
                    int(remaining),
                )
            )
    return trace


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(operations=st.lists(OPERATIONS, min_size=1, max_size=80))
def test_identical_under_random_operation_sequences(operations: list[tuple]) -> None:
    """The central claim: same inputs, byte-identical outputs."""
    python_book = LimitOrderBook()
    cpp_book = FastLimitOrderBook()

    python_trace = replay(python_book, operations)
    cpp_trace = replay(cpp_book, operations)

    assert python_trace == cpp_trace
    assert snapshot(python_book) == snapshot(cpp_book)


def test_identical_over_a_long_deterministic_sequence() -> None:
    """20,000 operations, seeded, so a divergence is reproducible."""
    from quantos.exchange.book import OrderNotFound

    rng_a = random.Random(4242)
    rng_b = random.Random(4242)
    results = []

    for book, rng in ((LimitOrderBook(), rng_a), (FastLimitOrderBook(), rng_b)):
        live: list[int] = []
        order_id = 0
        for _ in range(20_000):
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
                with contextlib.suppress(BookError):
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
            elif roll < 0.85:
                victim = live.pop(rng.randrange(len(live)))
                with contextlib.suppress(OrderNotFound):
                    book.cancel(OrderId(victim))
            elif roll < 0.95:
                target = live[rng.randrange(len(live))]
                if target in book:
                    book.amend(OrderId(target), Quantity(rng.randint(1, 120)))
            else:
                side = Side.BUY if rng.random() < 0.5 else Side.SELL
                book.match(side, Quantity(rng.randint(1, 150)), None, 0)
        book.check_invariants()
        results.append(snapshot(book))

    assert results[0] == results[1]


def test_matching_semantics_agree_on_price_time_priority() -> None:
    """Both books must fill the earlier order first at the same price."""
    for book_class in (LimitOrderBook, FastLimitOrderBook):
        book = book_class()
        for i, agent in enumerate(["first", "second", "third"], start=1):
            book.add(Order(OrderId(i), AgentId(agent), Side.SELL, Quantity(10), Ticks(10_001)))
        fills, remaining = book.match(Side.BUY, Quantity(25), Ticks(10_001), 0)
        assert [int(q) for _, q in fills] == [10, 10, 5]
        assert [str(node.agent_id) for node, _ in fills] == ["first", "second", "third"]
        assert remaining == 0


def test_amend_priority_rules_agree() -> None:
    """A reduction keeps queue position; an increase loses it. Both backends."""
    for book_class in (LimitOrderBook, FastLimitOrderBook):
        book = book_class()
        for i in (1, 2):
            book.add(Order(OrderId(i), AgentId(f"a{i}"), Side.BUY, Quantity(10), Ticks(10_000)))
        book.amend(OrderId(1), Quantity(20))  # increase -> to the back
        fills, _ = book.match(Side.SELL, Quantity(5), Ticks(10_000), 0)
        assert str(fills[0][0].agent_id) == "a2"


def test_crossing_orders_are_rejected_by_both() -> None:
    for book_class in (LimitOrderBook, FastLimitOrderBook):
        book = book_class()
        book.add(Order(OrderId(1), AgentId("a"), Side.SELL, Quantity(10), Ticks(10_000)))
        with pytest.raises(BookError, match="crossing"):
            book.add(Order(OrderId(2), AgentId("b"), Side.BUY, Quantity(10), Ticks(10_005)))


def test_cpp_book_reports_no_heap_slack() -> None:
    """A balanced tree cannot accumulate stale entries; the Python heaps could."""
    assert FastLimitOrderBook().heap_slack == 0


# --------------------------------------------------------------------------- #
# Batched replay
# --------------------------------------------------------------------------- #
def test_batched_replay_matches_operation_by_operation() -> None:
    """The batch path must reach the same book state as the per-call path.

    Batching is a performance optimisation, so it is only admissible if it
    changes nothing else. This drives the same tape through both and compares
    the resulting books.
    """
    from quantos.exchange.replay import BATCH_AVAILABLE, OpCode, replay_tape, synthetic_tape

    if not BATCH_AVAILABLE:
        pytest.skip("batch API not built")

    tape = synthetic_tape(20_000, seed=99)
    _, batched = replay_tape(tape)

    stepwise = FastLimitOrderBook()
    for opcode, order_id, side, price, quantity in tape:
        if opcode == OpCode.ADD:
            # A synthetic tape contains crossing and duplicate adds by design; the
            # batched path rejects them too, which is what the counts compare.
            with contextlib.suppress(BookError):
                stepwise.add(
                    Order(
                        OrderId(int(order_id)),
                        AgentId("a"),
                        Side.BUY if side > 0 else Side.SELL,
                        Quantity(int(quantity)),
                        Ticks(int(price)),
                    )
                )
        elif opcode == OpCode.CANCEL:
            with contextlib.suppress(OrderNotFound):
                stepwise.cancel(OrderId(int(order_id)))
        elif opcode == OpCode.MATCH:
            stepwise.match(
                Side.BUY if side > 0 else Side.SELL,
                Quantity(int(quantity)),
                Ticks(int(price)) if price else None,
                0,
            )

    assert batched.best_bid() == (int(stepwise.best_bid) if stepwise.best_bid is not None else None)
    assert batched.best_ask() == (int(stepwise.best_ask) if stepwise.best_ask is not None else None)
    assert batched.order_count() == len(stepwise)
    # Full structural comparison: every resting order, in book order.
    batched_orders = sorted(
        (int(o), int(s), int(p), int(q)) for o, s, p, q, _ in batched.snapshot()
    )
    stepwise_orders = sorted(
        (int(n.order_id), n.side.sign, int(n.price), int(n.remaining))
        for n in stepwise.iter_orders()
    )
    assert batched_orders == stepwise_orders


def test_batched_replay_reports_consistent_counts() -> None:
    from quantos.exchange.replay import BATCH_AVAILABLE, replay_tape, synthetic_tape

    if not BATCH_AVAILABLE:
        pytest.skip("batch API not built")

    tape = synthetic_tape(50_000, seed=7)
    result, book = replay_tape(tape)
    assert result.operations == 50_000
    assert result.added + result.rejected > 0
    assert 0.0 <= result.rejection_rate <= 1.0
    assert book.order_count() <= result.added
    assert result.operations_per_second > 1e6  # the whole point


def test_build_tape_validates_column_lengths() -> None:
    from quantos.exchange.replay import build_tape

    with pytest.raises(ValueError, match="same length"):
        build_tape([0, 0], [1], [1, 1], [100, 100], [10, 10])


def test_replay_rejects_a_malformed_tape() -> None:
    """A bad buffer reaching the extension is a segfault, not an exception."""
    import numpy as np

    from quantos.exchange.replay import BATCH_AVAILABLE, replay_tape

    if not BATCH_AVAILABLE:
        pytest.skip("batch API not built")
    with pytest.raises(ValueError, match=r"shape \(n, 5\)"):
        replay_tape(np.zeros((10, 3), dtype=np.int64))
