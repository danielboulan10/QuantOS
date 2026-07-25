"""Matching engine: order lifecycle, the public tape, and fee assignment.

Separation of concerns
----------------------
:class:`~quantos.exchange.book.LimitOrderBook` is a data structure. The
:class:`MatchingEngine` is the *venue*: it owns sequence numbers, decides what
each order type means, applies the fee schedule, publishes the tape, and
enforces self-trade prevention. Nothing here reaches into the book's internals
beyond its public methods.

Order type semantics implemented here
-------------------------------------
============  ==============================================================
``LIMIT``     Cross what it can, rest the remainder (GTC) or cancel it (IOC).
``MARKET``    Sweep the book with no price limit. IOC or FOK only -- an
              unfilled market remainder has no price at which to rest.
``POST_ONLY`` Rejected outright if it would cross. The order type that makes
              maker-rebate strategies expressible without a race: the venue
              guarantees you either post or nothing happens.
``FOK``       Filled entirely or not at all, checked *before* any state
              change so a partial sweep is never observable.
============  ==============================================================

Self-trade prevention
---------------------
An agent matching its own resting order is prevented, defaulting to
*cancel-resting* (the maker is pulled, the aggressor continues). Real venues
offer several STP policies; the important thing for a simulator is that
*something* prevents it, because wash trades otherwise pollute the tape and
inflate every volume-based statistic -- VPIN and Kyle's lambda in particular.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from quantos.core.types import (
    Fill,
    Liquidity,
    Nanos,
    Order,
    OrderId,
    OrderType,
    Quantity,
    Side,
    Ticks,
    TimeInForce,
    TopOfBook,
    Trade,
)
from quantos.exchange.book import LimitOrderBook, OrderNotFound
from quantos.exchange.fees import FeeSchedule

__all__ = ["ExecutionReport", "MatchingEngine", "RejectReason", "SelfTradePolicy"]


class RejectReason(enum.Enum):
    """Why an order was refused. Every rejection carries one of these."""

    NONE = "none"
    POST_ONLY_WOULD_CROSS = "post_only_would_cross"
    FOK_INSUFFICIENT_LIQUIDITY = "fok_insufficient_liquidity"
    IOC_NO_LIQUIDITY = "ioc_no_liquidity"
    DUPLICATE_ORDER_ID = "duplicate_order_id"
    SELF_TRADE = "self_trade"


class SelfTradePolicy(enum.Enum):
    """What to do when an agent's order would match its own resting order."""

    #: Pull the resting order, let the aggressor continue. Venue default.
    CANCEL_RESTING = "cancel_resting"
    #: Cancel the incoming aggressor, leave the book untouched.
    CANCEL_AGGRESSING = "cancel_aggressing"
    #: Permit it. Only for tests that need to exercise wash-trade detection.
    ALLOW = "allow"


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Everything that happened as a result of submitting one order.

    Returned synchronously from :meth:`MatchingEngine.submit`. Agents consume
    this; the tape (:attr:`MatchingEngine.trades`) is the public view.
    """

    order_id: OrderId
    accepted: bool
    fills: tuple[Fill, ...] = ()
    resting_quantity: Quantity = Quantity(0)
    cancelled_quantity: Quantity = Quantity(0)
    reject_reason: RejectReason = RejectReason.NONE
    fees_paid: float = 0.0

    @property
    def filled_quantity(self) -> int:
        return sum(int(f.quantity) for f in self.fills)

    @property
    def average_price(self) -> float | None:
        """Volume-weighted average fill price in ticks, or ``None`` if unfilled."""
        total = self.filled_quantity
        if total == 0:
            return None
        return sum(int(f.price) * int(f.quantity) for f in self.fills) / total


@dataclass
class MatchingEngine:
    """A single-instrument venue wrapping a :class:`LimitOrderBook`.

    Example
        >>> from quantos.core.types import Order, OrderId, AgentId, Side, Quantity, Ticks
        >>> eng = MatchingEngine()
        >>> _ = eng.submit(Order(OrderId(1), AgentId("mm"), Side.SELL,
        ...                      Quantity(100), Ticks(1001)))
        >>> rpt = eng.submit(Order(OrderId(2), AgentId("taker"), Side.BUY,
        ...                        Quantity(60), Ticks(1005)))
        >>> rpt.filled_quantity, rpt.average_price
        (60, 1001.0)
        >>> eng.book.size_at(Ticks(1001))     # 40 left resting
        40
    """

    book: LimitOrderBook = field(default_factory=LimitOrderBook)
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    self_trade_policy: SelfTradePolicy = SelfTradePolicy.CANCEL_RESTING
    trades: list[Trade] = field(default_factory=list)
    #: Maker-side fills are published separately: the taker receives its fills
    #: in the ExecutionReport, but makers learn of theirs asynchronously via
    #: :meth:`drain_maker_fills`, exactly as they would from a live drop-copy
    #: feed. Modelling it this way forces agents to be written as event
    #: handlers -- the same shape they need to run against a real venue.
    _maker_fills: list[Fill] = field(default_factory=list)
    _trade_seq: int = 0

    # -- public API -------------------------------------------------------- #
    def submit(self, order: Order) -> ExecutionReport:
        """Submit an order and return everything that resulted from it."""
        if order.order_id in self.book:
            return ExecutionReport(
                order.order_id, accepted=False, reject_reason=RejectReason.DUPLICATE_ORDER_ID
            )

        if order.order_type is OrderType.POST_ONLY:
            return self._submit_post_only(order)
        if order.time_in_force is TimeInForce.FOK:
            return self._submit_fok(order)
        return self._submit_crossing(order)

    def cancel(self, order_id: OrderId) -> Quantity | None:
        """Cancel a resting order; ``None`` if it was already gone.

        Returning ``None`` rather than raising is deliberate here: at the venue
        boundary a cancel racing a fill is an ordinary occurrence, not an
        error, and agents should not be forced to wrap every cancel in a
        ``try``. The book's own :meth:`~LimitOrderBook.cancel` still raises,
        because *there* a missing order means a corrupted index.
        """
        try:
            return self.book.cancel(order_id)
        except OrderNotFound:
            return None

    def top_of_book(self, timestamp: Nanos = Nanos(0)) -> TopOfBook:
        return self.book.top_of_book(timestamp)

    @property
    def last_price(self) -> Ticks | None:
        return self.trades[-1].price if self.trades else None

    # -- order-type handlers ----------------------------------------------- #
    def _submit_post_only(self, order: Order) -> ExecutionReport:
        opposing = self.book.best_ask if order.side is Side.BUY else self.book.best_bid
        assert order.price is not None  # guaranteed by Order.__post_init__
        if order.side.is_aggressive_at(order.price, opposing):
            return ExecutionReport(
                order.order_id,
                accepted=False,
                reject_reason=RejectReason.POST_ONLY_WOULD_CROSS,
            )
        self.book.add(order)
        return ExecutionReport(order.order_id, accepted=True, resting_quantity=order.quantity)

    def _submit_fok(self, order: Order) -> ExecutionReport:
        """Fill-or-kill: verify sufficient liquidity *before* mutating anything."""
        if self._available_liquidity(order.side, order.price) < int(order.quantity):
            return ExecutionReport(
                order.order_id,
                accepted=False,
                reject_reason=RejectReason.FOK_INSUFFICIENT_LIQUIDITY,
            )
        return self._submit_crossing(order, force_ioc=True)

    def _available_liquidity(self, side: Side, limit_price: Ticks | None) -> int:
        """Marketable quantity on the opposite side at or better than ``limit_price``.

        Read-only: it inspects depth rather than matching, which is what makes
        the FOK pre-check non-destructive.
        """
        depth = self.book.depth(side.opposite, levels=1_000_000)
        total = 0
        for level in depth:
            if limit_price is not None and not side.is_aggressive_at(limit_price, level.price):
                break
            total += int(level.quantity)
        return total

    def _submit_crossing(self, order: Order, *, force_ioc: bool = False) -> ExecutionReport:
        """Match against the book, then rest or cancel the remainder."""
        limit = order.price if order.order_type is not OrderType.MARKET else None
        raw_fills, remaining = self.book.match(order.side, order.quantity, limit, order.timestamp)

        raw_fills, remaining = self._apply_self_trade_policy(order, raw_fills, remaining)

        fills: list[Fill] = []
        fees_paid = 0.0
        for maker_node, traded in raw_fills:
            self._trade_seq += 1
            price = maker_node.price
            self.trades.append(
                Trade(
                    seq=self._trade_seq,
                    price=price,
                    quantity=Quantity(traded),
                    timestamp=order.timestamp,
                    aggressor_side=order.side,
                    maker_order_id=maker_node.order_id,
                    taker_order_id=order.order_id,
                )
            )
            maker_fill = Fill(
                order_id=maker_node.order_id,
                agent_id=maker_node.agent_id,
                side=maker_node.side,
                price=price,
                quantity=Quantity(traded),
                timestamp=order.timestamp,
                liquidity=Liquidity.MAKER,
                trade_seq=self._trade_seq,
            )
            taker_fill = Fill(
                order_id=order.order_id,
                agent_id=order.agent_id,
                side=order.side,
                price=price,
                quantity=Quantity(traded),
                timestamp=order.timestamp,
                liquidity=Liquidity.TAKER,
                trade_seq=self._trade_seq,
            )
            self._maker_fills.append(maker_fill)
            fills.append(taker_fill)
            fees_paid += self.fees.charge(taker_fill)

        # Rest or cancel whatever did not trade.
        resting = 0
        cancelled = 0
        should_rest = (
            remaining > 0
            and not force_ioc
            and order.time_in_force is TimeInForce.GTC
            and order.order_type is not OrderType.MARKET
        )
        if should_rest:
            assert order.price is not None
            self.book.add(
                Order(
                    order_id=order.order_id,
                    agent_id=order.agent_id,
                    side=order.side,
                    quantity=Quantity(remaining),
                    price=order.price,
                    order_type=OrderType.LIMIT,
                    time_in_force=TimeInForce.GTC,
                    timestamp=order.timestamp,
                )
            )
            resting = remaining
        else:
            cancelled = remaining

        reason = RejectReason.NONE
        if not fills and cancelled and order.time_in_force is TimeInForce.IOC:
            reason = RejectReason.IOC_NO_LIQUIDITY

        return ExecutionReport(
            order_id=order.order_id,
            accepted=True,
            fills=tuple(fills),
            resting_quantity=Quantity(resting),
            cancelled_quantity=Quantity(cancelled),
            reject_reason=reason,
            fees_paid=fees_paid,
        )

    def _apply_self_trade_policy(
        self,
        order: Order,
        raw_fills: list[tuple[Any, int]],
        remaining: int,
    ) -> tuple[list[tuple[Any, int]], int]:
        """Filter out any fill where the aggressor would trade with itself."""
        if self.self_trade_policy is SelfTradePolicy.ALLOW:
            return raw_fills, remaining
        if not any(node.agent_id == order.agent_id for node, _ in raw_fills):
            return raw_fills, remaining

        kept = []
        returned = 0
        for node, traded in raw_fills:
            if node.agent_id != order.agent_id:
                kept.append((node, traded))
                continue
            if self.self_trade_policy is SelfTradePolicy.CANCEL_RESTING:
                # The maker was already consumed by book.match; it stays gone,
                # and the aggressor keeps the quantity to continue with.
                returned += traded
            else:  # CANCEL_AGGRESSING
                returned += traded
        return kept, remaining + returned

    def drain_maker_fills(self) -> list[Fill]:
        """Pop and return maker fills accumulated since the last drain.

        Modelling maker fills as an asynchronous feed rather than a return
        value is what forces agents to be written as event handlers -- the same
        shape they must have to run against a live venue.
        """
        out = self._maker_fills
        self._maker_fills = []
        return out
