"""Property-based tests for the limit order book.

Strategy
--------
Unit tests confirm the cases you thought of. The interesting order-book bugs live
in the cases you did not -- a cancel racing a fill, an amend that empties a level
that is also the best price, a sweep that consumes a level whose heap entry is
already stale. So the core of this module drives *randomised operation sequences*
with Hypothesis and asserts every structural invariant after every operation.

The invariants (from :meth:`LimitOrderBook.check_invariants`)
------------------------------------------------------------
1. The book is never crossed.
2. Each level's cached ``total_quantity`` equals the sum of its nodes'.
3. Each level's cached ``order_count`` equals its node count.
4. The order index is exactly the set of nodes reachable from the levels.
5. Queue links are consistent in both directions, and tails are not stale.

Conservation of quantity is checked separately in
:func:`test_quantity_is_conserved`, since it spans the book and the tape.
"""

from __future__ import annotations

import contextlib

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quantos.core.types import (
    AgentId,
    Liquidity,
    Order,
    OrderId,
    OrderType,
    Quantity,
    Side,
    Ticks,
    TimeInForce,
)
from quantos.exchange.book import BookError, LimitOrderBook, OrderNotFound
from quantos.exchange.matching import MatchingEngine, RejectReason, SelfTradePolicy

# --------------------------------------------------------------------------- #
# Hypothesis strategies
# --------------------------------------------------------------------------- #
PRICES = st.integers(min_value=9_950, max_value=10_050)
SIZES = st.integers(min_value=1, max_value=100)
SIDES = st.sampled_from([Side.BUY, Side.SELL])

# One operation: ("add", side, price, size) | ("cancel", i) | ("amend", i, size)
OPERATIONS = st.one_of(
    st.tuples(st.just("add"), SIDES, PRICES, SIZES),
    st.tuples(st.just("cancel"), st.integers(min_value=0, max_value=999)),
    st.tuples(st.just("amend"), st.integers(min_value=0, max_value=999), SIZES),
)


@settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(operations=st.lists(OPERATIONS, min_size=1, max_size=120))
def test_invariants_hold_after_every_operation(operations: list[tuple]) -> None:
    """Random add/cancel/amend sequences never violate a structural invariant."""
    book = LimitOrderBook()
    live: list[int] = []
    next_id = 0

    for operation in operations:
        if operation[0] == "add":
            _, side, price, size = operation
            next_id += 1
            # Keep the order non-crossing; the book rejects crossing rests by
            # design, and the matching engine is what handles aggression.
            opposing = book.best_ask if side is Side.BUY else book.best_bid
            if opposing is not None:
                price = (
                    min(price, int(opposing) - 1)
                    if side is Side.BUY
                    else max(price, int(opposing) + 1)
                )
            if price <= 0:
                continue
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
        elif operation[0] == "cancel":
            if not live:
                continue
            victim = live.pop(operation[1] % len(live))
            with pytest.raises(OrderNotFound) if victim not in book else _NullContext():
                book.cancel(OrderId(victim))
        else:
            if not live:
                continue
            target = live[operation[1] % len(live)]
            if target in book:
                book.amend(OrderId(target), Quantity(operation[2]))

        book.check_invariants()

    book.check_invariants()


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Price-time priority
# --------------------------------------------------------------------------- #
def test_price_priority_beats_time_priority() -> None:
    """A better price is filled first, regardless of arrival order."""
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("early"), Side.SELL, Quantity(10), Ticks(10_005)))
    engine.submit(Order(OrderId(2), AgentId("late"), Side.SELL, Quantity(10), Ticks(10_001)))

    report = engine.submit(
        Order(OrderId(3), AgentId("taker"), Side.BUY, Quantity(10), Ticks(10_010))
    )
    assert report.filled_quantity == 10
    # Filled against the better-priced later order, at that order's price.
    assert report.fills[0].price == Ticks(10_001)


def test_time_priority_within_a_price_level() -> None:
    """At equal prices, the earlier order fills first -- strict FIFO."""
    engine = MatchingEngine()
    for i, agent in enumerate(["first", "second", "third"], start=1):
        engine.submit(Order(OrderId(i), AgentId(agent), Side.SELL, Quantity(10), Ticks(10_001)))

    report = engine.submit(
        Order(OrderId(99), AgentId("taker"), Side.BUY, Quantity(25), Ticks(10_001))
    )
    maker_fills = engine.drain_maker_fills()
    assert [str(f.agent_id) for f in maker_fills] == ["first", "second", "third"]
    assert [int(f.quantity) for f in maker_fills] == [10, 10, 5]
    assert report.filled_quantity == 25


def test_fills_print_at_the_maker_price_not_the_taker_limit() -> None:
    """Price improvement accrues to the aggressor.

    Getting this backwards silently inflates every backtest's transaction costs
    and understates the value of passive execution.
    """
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("mm"), Side.SELL, Quantity(50), Ticks(10_001)))
    report = engine.submit(
        Order(OrderId(2), AgentId("taker"), Side.BUY, Quantity(50), Ticks(10_020))
    )
    assert report.average_price == 10_001.0
    assert engine.trades[0].price == Ticks(10_001)


def test_amend_down_keeps_queue_position_amend_up_loses_it() -> None:
    """Exchange priority rules, which are economically load-bearing.

    A market maker may shade size down without losing its place, but must go to
    the back of the queue to add size. A simulator that let increases keep
    priority would make queue-position strategies look far better than they are.
    """
    book = LimitOrderBook()
    for i in (1, 2):
        book.add(Order(OrderId(i), AgentId(f"a{i}"), Side.BUY, Quantity(10), Ticks(10_000)))

    # Reduce order 1: it must remain at the head.
    book.amend(OrderId(1), Quantity(5))
    assert book.size_at(Ticks(10_000)) == 15
    engine = MatchingEngine(book=book)
    engine.submit(Order(OrderId(3), AgentId("t"), Side.SELL, Quantity(5), Ticks(10_000)))
    assert str(engine.drain_maker_fills()[0].agent_id) == "a1"

    # Now increase order 2: it must go to the back, behind the remainder of a1.
    book2 = LimitOrderBook()
    for i in (1, 2):
        book2.add(Order(OrderId(i), AgentId(f"a{i}"), Side.BUY, Quantity(10), Ticks(10_000)))
    book2.amend(OrderId(1), Quantity(20))  # increase -> loses priority
    engine2 = MatchingEngine(book=book2)
    engine2.submit(Order(OrderId(3), AgentId("t"), Side.SELL, Quantity(5), Ticks(10_000)))
    assert str(engine2.drain_maker_fills()[0].agent_id) == "a2"


def test_book_refuses_to_rest_a_crossing_order() -> None:
    """A crossed book is an invariant violation, not a recoverable state."""
    book = LimitOrderBook()
    book.add(Order(OrderId(1), AgentId("a"), Side.SELL, Quantity(10), Ticks(10_000)))
    with pytest.raises(BookError, match="crossing"):
        book.add(Order(OrderId(2), AgentId("b"), Side.BUY, Quantity(10), Ticks(10_005)))


def test_cancel_is_o1_from_any_queue_position() -> None:
    """Cancelling from the middle of a long queue must not traverse it.

    Correctness proxy for the O(1) claim: the resulting book state must be exact
    regardless of which position was removed.
    """
    book = LimitOrderBook()
    for i in range(1, 201):
        book.add(Order(OrderId(i), AgentId("a"), Side.BUY, Quantity(1), Ticks(10_000)))
    book.cancel(OrderId(100))
    book.check_invariants()
    assert book.size_at(Ticks(10_000)) == 199
    assert len(book) == 199
    assert OrderId(100) not in book


def test_cancelling_a_missing_order_raises_from_the_book() -> None:
    book = LimitOrderBook()
    with pytest.raises(OrderNotFound):
        book.cancel(OrderId(42))


def test_engine_cancel_tolerates_the_fill_race() -> None:
    """At the venue boundary a cancel racing a fill is normal, not an error."""
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("mm"), Side.SELL, Quantity(10), Ticks(10_001)))
    engine.submit(Order(OrderId(2), AgentId("t"), Side.BUY, Quantity(10), Ticks(10_001)))
    assert engine.cancel(OrderId(1)) is None  # already filled


def test_heap_slack_stays_bounded() -> None:
    """Lazy deletion must not leak: slack is bounded by distinct prices touched."""
    book = LimitOrderBook()
    for order_id in range(1, 2_001):
        book.add(
            Order(
                OrderId(order_id),
                AgentId("a"),
                Side.BUY,
                Quantity(1),
                Ticks(10_000 - (order_id % 50)),
            )
        )
        if order_id > 10:
            with contextlib.suppress(OrderNotFound):
                book.cancel(OrderId(order_id - 10))
    # 50 distinct prices were used, so slack cannot meaningfully exceed that.
    assert book.heap_slack <= 60
    book.check_invariants()


# --------------------------------------------------------------------------- #
# Conservation
# --------------------------------------------------------------------------- #
@settings(max_examples=60, deadline=None)
@given(
    orders=st.lists(
        st.tuples(SIDES, PRICES, SIZES),
        min_size=2,
        max_size=60,
    )
)
def test_quantity_is_conserved(orders: list[tuple]) -> None:
    """Every submitted unit ends up resting, filled, or cancelled -- never lost.

    This is the invariant a matching engine most needs and most easily breaks,
    because partial fills touch three places at once: the maker's remaining
    quantity, the level's cached total, and the taker's residual.
    """
    engine = MatchingEngine(self_trade_policy=SelfTradePolicy.ALLOW)
    submitted = 0
    filled_as_taker = 0
    resting_reported = 0
    cancelled_reported = 0

    for i, (side, price, size) in enumerate(orders, start=1):
        report = engine.submit(
            Order(
                OrderId(i),
                AgentId(f"a{i % 4}"),
                side,
                Quantity(size),
                Ticks(price),
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
            )
        )
        if not report.accepted:
            continue
        submitted += size
        filled_as_taker += report.filled_quantity
        resting_reported += int(report.resting_quantity)
        cancelled_reported += int(report.cancelled_quantity)

    # Each taker fill consumed an equal maker quantity, so total traded volume
    # is twice the taker-side volume.
    traded_volume = sum(int(t.quantity) for t in engine.trades)
    assert traded_volume == filled_as_taker

    resting_on_book = sum(
        int(level.quantity)
        for side in (Side.BUY, Side.SELL)
        for level in engine.book.depth(side, levels=10_000)
    )
    # submitted = (taker-filled) + (maker-filled) + (still resting) + (cancelled)
    assert submitted == filled_as_taker * 2 + resting_on_book + cancelled_reported
    engine.book.check_invariants()


# --------------------------------------------------------------------------- #
# Order types
# --------------------------------------------------------------------------- #
def test_post_only_is_rejected_rather_than_crossing() -> None:
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("mm"), Side.SELL, Quantity(10), Ticks(10_001)))
    report = engine.submit(
        Order(
            OrderId(2),
            AgentId("b"),
            Side.BUY,
            Quantity(10),
            Ticks(10_005),
            order_type=OrderType.POST_ONLY,
        )
    )
    assert not report.accepted
    assert report.reject_reason is RejectReason.POST_ONLY_WOULD_CROSS
    assert len(engine.trades) == 0


def test_post_only_rests_when_it_does_not_cross() -> None:
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("mm"), Side.SELL, Quantity(10), Ticks(10_005)))
    report = engine.submit(
        Order(
            OrderId(2),
            AgentId("b"),
            Side.BUY,
            Quantity(10),
            Ticks(10_001),
            order_type=OrderType.POST_ONLY,
        )
    )
    assert report.accepted
    assert int(report.resting_quantity) == 10


def test_fok_is_all_or_nothing_and_leaves_no_trace() -> None:
    """A partial FOK sweep must never be observable, so the check precedes matching."""
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("mm"), Side.SELL, Quantity(10), Ticks(10_001)))
    report = engine.submit(
        Order(
            OrderId(2),
            AgentId("b"),
            Side.BUY,
            Quantity(50),
            Ticks(10_010),
            time_in_force=TimeInForce.FOK,
        )
    )
    assert not report.accepted
    assert report.reject_reason is RejectReason.FOK_INSUFFICIENT_LIQUIDITY
    assert len(engine.trades) == 0
    assert engine.book.size_at(Ticks(10_001)) == 10  # untouched


def test_fok_fills_completely_when_liquidity_suffices() -> None:
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("mm"), Side.SELL, Quantity(60), Ticks(10_001)))
    report = engine.submit(
        Order(
            OrderId(2),
            AgentId("b"),
            Side.BUY,
            Quantity(50),
            Ticks(10_010),
            time_in_force=TimeInForce.FOK,
        )
    )
    assert report.accepted
    assert report.filled_quantity == 50


def test_ioc_cancels_the_remainder() -> None:
    engine = MatchingEngine()
    engine.submit(Order(OrderId(1), AgentId("mm"), Side.SELL, Quantity(10), Ticks(10_001)))
    report = engine.submit(
        Order(
            OrderId(2),
            AgentId("b"),
            Side.BUY,
            Quantity(30),
            Ticks(10_010),
            time_in_force=TimeInForce.IOC,
        )
    )
    assert report.filled_quantity == 10
    assert int(report.cancelled_quantity) == 20
    assert int(report.resting_quantity) == 0


def test_market_order_sweeps_multiple_levels() -> None:
    engine = MatchingEngine()
    for i, price in enumerate([10_001, 10_002, 10_003], start=1):
        engine.submit(Order(OrderId(i), AgentId("mm"), Side.SELL, Quantity(10), Ticks(price)))
    report = engine.submit(
        Order(
            OrderId(9),
            AgentId("t"),
            Side.BUY,
            Quantity(25),
            None,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
        )
    )
    assert report.filled_quantity == 25
    assert [int(f.price) for f in report.fills] == [10_001, 10_002, 10_003]
    # VWAP across the swept levels.
    assert report.average_price == pytest.approx((10_001 * 10 + 10_002 * 10 + 10_003 * 5) / 25)


def test_gtc_market_order_is_rejected_at_construction() -> None:
    """An unfilled market remainder has no price at which to rest."""
    with pytest.raises(ValueError, match="market orders cannot be GTC"):
        Order(
            OrderId(1),
            AgentId("a"),
            Side.BUY,
            Quantity(10),
            None,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
        )


def test_market_order_with_a_price_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not carry a limit price"):
        Order(
            OrderId(1),
            AgentId("a"),
            Side.BUY,
            Quantity(10),
            Ticks(100),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
        )


def test_self_trade_prevention_stops_wash_trades() -> None:
    """Wash trades pollute the tape and inflate every volume-based statistic."""
    engine = MatchingEngine(self_trade_policy=SelfTradePolicy.CANCEL_RESTING)
    engine.submit(Order(OrderId(1), AgentId("same"), Side.SELL, Quantity(10), Ticks(10_001)))
    report = engine.submit(
        Order(OrderId(2), AgentId("same"), Side.BUY, Quantity(10), Ticks(10_005))
    )
    assert report.filled_quantity == 0
    assert len(engine.trades) == 0


# --------------------------------------------------------------------------- #
# Derived market data
# --------------------------------------------------------------------------- #
def test_top_of_book_derived_quantities() -> None:
    book = LimitOrderBook()
    book.add(Order(OrderId(1), AgentId("a"), Side.BUY, Quantity(100), Ticks(9_999)))
    book.add(Order(OrderId(2), AgentId("b"), Side.SELL, Quantity(300), Ticks(10_001)))
    top = book.top_of_book()

    assert top.mid == 10_000.0
    assert top.spread == 2
    # Microprice leans toward the thinner side -- here the bid, which has less size.
    assert top.microprice == pytest.approx((100 * 10_001 + 300 * 9_999) / 400)
    assert top.microprice < top.mid
    assert top.imbalance == pytest.approx((100 - 300) / 400)


def test_one_sided_book_returns_none_rather_than_guessing() -> None:
    book = LimitOrderBook()
    book.add(Order(OrderId(1), AgentId("a"), Side.BUY, Quantity(100), Ticks(9_999)))
    top = book.top_of_book()
    assert top.mid is None
    assert top.spread is None
    assert top.microprice is None
    assert book.best_ask is None


def test_side_sign_arithmetic() -> None:
    """Side values are +-1 so they compose as multipliers."""
    assert Side.BUY.sign == 1
    assert Side.SELL.sign == -1
    assert Side.BUY.opposite is Side.SELL
    assert Side.BUY.is_aggressive_at(Ticks(101), Ticks(100))
    assert not Side.BUY.is_aggressive_at(Ticks(99), Ticks(100))
    assert Side.SELL.is_aggressive_at(Ticks(99), Ticks(100))
    assert not Side.BUY.is_aggressive_at(Ticks(101), None)


def test_fee_schedule_rebates_makers_and_charges_takers() -> None:
    """The sign convention: positive is a cost, negative is a rebate."""
    from quantos.core.types import Fill, Nanos
    from quantos.exchange.fees import MakerTakerFees

    schedule = MakerTakerFees()
    maker = Fill(
        OrderId(1),
        AgentId("m"),
        Side.BUY,
        Ticks(10_000),
        Quantity(100),
        Nanos(0),
        Liquidity.MAKER,
    )
    taker = Fill(
        OrderId(2),
        AgentId("t"),
        Side.SELL,
        Ticks(10_000),
        Quantity(100),
        Nanos(0),
        Liquidity.TAKER,
    )
    assert schedule.charge(maker) < 0.0
    assert schedule.charge(taker) > 0.0
    # The venue keeps the difference between the taker fee and the maker rebate.
    assert schedule.charge(taker) + schedule.charge(maker) > 0.0
