"""Price-time priority limit order book.

Data structure
--------------
Three cooperating structures, each chosen for one access pattern:

======================  =========================  ==========================
Structure               Purpose                    Cost
======================  =========================  ==========================
``dict[Ticks, _Level]`` Find a price level         O(1)
``_Level`` (intrusive   FIFO queue within a level  O(1) push, pop, unlink
doubly-linked list)
``heapq`` per side      Track the best price       O(log n) push, O(1) peek
``dict[OrderId, _Node]``Locate a resting order     O(1)
======================  =========================  ==========================

**Why a linked list and not ``collections.deque``?** Cancellation. A deque
gives O(1) at the ends but O(n) removal from the middle, and cancels arrive at
arbitrary queue positions -- in real markets 95%+ of orders are cancelled
rather than filled, so middle-removal is the *dominant* operation, not an edge
case. An intrusive doubly-linked list plus the id-to-node map makes cancel
strictly O(1). This is the single most consequential data-structure decision in
the engine, and it is the reason the benchmark in ``benchmarks/`` sustains
seven-figure operations per second in pure Python.

**Why lazy deletion on the price heaps?** ``heapq`` cannot delete an interior
element. Rather than pay O(n) to rebuild, emptied price levels are left in the
heap and skipped when they surface at the top. The heap can therefore hold
stale entries, but it is bounded by the number of *distinct prices ever
touched*, and each stale entry is popped at most once -- so the amortised cost
is O(1) per level, ever. :meth:`LimitOrderBook.heap_slack` exposes the
accumulated slack for the benchmark suite to assert on.

Invariants (asserted by ``tests/exchange/test_book_invariants.py`` under
Hypothesis, over random operation sequences)
-------------------------------------------------------------------------
1. ``best_bid < best_ask`` whenever both sides are non-empty -- the book is
   never crossed after an operation completes.
2. A level's cached ``total_quantity`` equals the sum of its nodes' remaining
   quantities.
3. ``len(order_index)`` equals the number of nodes reachable by walking every
   level of both sides.
4. Total quantity is conserved: every unit that enters is resting, filled, or
   cancelled -- never lost or duplicated.
5. Price-time priority: within a level, fills occur in strict arrival order.

References
----------
Gould, M. et al. (2013), "Limit order books", *Quantitative Finance* 13(11).
Harris, L. (2003), *Trading and Exchanges*, ch. 4-6.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from quantos.core.types import (
    AgentId,
    BookLevel,
    Nanos,
    Order,
    OrderId,
    Quantity,
    Side,
    Ticks,
    TopOfBook,
)

__all__ = ["BookError", "LimitOrderBook", "OrderNotFound"]


class BookError(RuntimeError):
    """Base class for order-book violations."""


class OrderNotFound(BookError):
    """Raised when cancelling or amending an order the book does not hold."""


@dataclass(slots=True)
class _Node:
    """A resting order. Mutable and private: the public type is ``Order``.

    ``prev``/``next`` make this an *intrusive* list node -- the links live on
    the payload rather than in wrapper cells, which halves allocation and makes
    unlinking a pure pointer operation.
    """

    order_id: OrderId
    agent_id: AgentId
    side: Side
    price: Ticks
    remaining: Quantity
    timestamp: Nanos
    sequence: int
    prev: _Node | None = None
    next: _Node | None = None


@dataclass(slots=True)
class _Level:
    """FIFO queue of orders resting at a single price."""

    price: Ticks
    head: _Node | None = None
    tail: _Node | None = None
    total_quantity: int = 0
    order_count: int = 0

    def push_back(self, node: _Node) -> None:
        """Append at the back of the queue -- the time half of price-time priority."""
        node.prev = self.tail
        node.next = None
        if self.tail is None:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
        self.total_quantity += int(node.remaining)
        self.order_count += 1

    def unlink(self, node: _Node) -> None:
        """Remove ``node`` from anywhere in the queue in O(1)."""
        if node.prev is None:
            self.head = node.next
        else:
            node.prev.next = node.next
        if node.next is None:
            self.tail = node.prev
        else:
            node.next.prev = node.prev
        node.prev = node.next = None
        self.total_quantity -= int(node.remaining)
        self.order_count -= 1

    @property
    def is_empty(self) -> bool:
        return self.head is None


class LimitOrderBook:
    """A single-instrument limit order book with price-time priority.

    The book is a *pure data structure*: it knows how to rest, cancel, amend
    and match orders, and nothing about fees, latency, agents or risk. Those
    are layered on by :class:`quantos.exchange.matching.MatchingEngine`. Keeping
    them separate is what allows the book to be benchmarked and property-tested
    in isolation -- and what allows the same book to serve a simulated venue,
    a historical replay, and a unit test without modification.

    Example
        >>> from quantos.core.types import Order, OrderId, AgentId, Side, Quantity, Ticks
        >>> book = LimitOrderBook()
        >>> _ = book.add(Order(OrderId(1), AgentId("mm"), Side.BUY,
        ...                    Quantity(100), Ticks(9_99)))
        >>> _ = book.add(Order(OrderId(2), AgentId("mm"), Side.SELL,
        ...                    Quantity(100), Ticks(10_01)))
        >>> book.best_bid, book.best_ask
        (999, 1001)
        >>> book.top_of_book().spread
        2
    """

    __slots__ = (
        "_ask_heap",
        "_ask_in_heap",
        "_bid_heap",
        "_bid_in_heap",
        "_heap_pops",
        "_levels",
        "_order_index",
        "_sequence",
        "_stale_pops",
    )

    def __init__(self) -> None:
        # One level map keyed by price. A single map (rather than one per side)
        # is safe because a crossed book is never allowed to persist, so a
        # price can only ever be occupied by one side at a time.
        self._levels: dict[int, _Level] = {}
        # Bids negated so heapq's min-heap yields the highest price.
        self._bid_heap: list[int] = []
        self._ask_heap: list[int] = []
        # Which prices currently have an entry in each heap. Without this, a
        # price whose level empties and is later re-created gets a SECOND heap
        # entry while the first is still present, and the heap grows without
        # bound: measured at 1,990 stale entries after 2,000 add/cancel cycles
        # over just 50 distinct prices. Lazy deletion is only sound if each
        # price is in the heap at most once.
        self._bid_in_heap: set[int] = set()
        self._ask_in_heap: set[int] = set()
        self._order_index: dict[int, _Node] = {}
        self._sequence: int = 0
        self._heap_pops: int = 0
        self._stale_pops: int = 0

    # -- introspection ----------------------------------------------------- #
    def __len__(self) -> int:
        """Number of resting orders."""
        return len(self._order_index)

    def __contains__(self, order_id: object) -> bool:
        return int(order_id) in self._order_index  # type: ignore[call-overload]

    @property
    def heap_slack(self) -> int:
        """Stale entries currently sitting in the price heaps.

        Exposed so the benchmark suite can assert the lazy-deletion scheme
        stays bounded rather than degenerating into an unbounded leak.
        """
        return (len(self._bid_heap) + len(self._ask_heap)) - sum(
            1 for lvl in self._levels.values() if not lvl.is_empty
        )

    def _prune(self, heap: list[int], negated: bool) -> int | None:
        """Discard stale heap tops until a live price surfaces.

        Membership is retired in lockstep with the heap pop. If the set said a
        price were present after it had been popped, that price's level would
        become permanently invisible to :attr:`best_bid` / :attr:`best_ask`,
        because :meth:`add` would decline to re-push it.
        """
        membership = self._bid_in_heap if negated else self._ask_in_heap
        while heap:
            price = -heap[0] if negated else heap[0]
            level = self._levels.get(price)
            if level is not None and not level.is_empty:
                return price
            heapq.heappop(heap)
            membership.discard(price)
            self._heap_pops += 1
            self._stale_pops += 1
            self._levels.pop(price, None)
        return None

    @property
    def best_bid(self) -> Ticks | None:
        """Highest resting bid price, or ``None``."""
        price = self._prune(self._bid_heap, negated=True)
        return Ticks(price) if price is not None else None

    @property
    def best_ask(self) -> Ticks | None:
        """Lowest resting ask price, or ``None``."""
        price = self._prune(self._ask_heap, negated=False)
        return Ticks(price) if price is not None else None

    def size_at(self, price: Ticks) -> int:
        """Total resting quantity at ``price`` (0 if the level is empty)."""
        level = self._levels.get(int(price))
        return level.total_quantity if level is not None else 0

    def top_of_book(self, timestamp: Nanos = Nanos(0)) -> TopOfBook:
        """Snapshot the best bid and offer with their sizes."""
        bid = self.best_bid
        ask = self.best_ask
        return TopOfBook(
            timestamp=timestamp,
            bid_price=bid,
            bid_size=Quantity(self.size_at(bid)) if bid is not None else None,
            ask_price=ask,
            ask_size=Quantity(self.size_at(ask)) if ask is not None else None,
        )

    def depth(self, side: Side, levels: int = 5) -> list[BookLevel]:
        """The ``levels`` best price levels on ``side``, best first.

        Complexity is O(k log k) in the number of *occupied* levels, since it
        sorts the live price set. Intended for periodic snapshots and
        visualisation, not for the hot path -- the matching loop uses
        :attr:`best_bid` / :attr:`best_ask`, which are O(1) amortised.
        """
        prices = [
            p
            for p, lvl in self._levels.items()
            if not lvl.is_empty and self._side_of_level(lvl) is side
        ]
        prices.sort(reverse=(side is Side.BUY))
        out: list[BookLevel] = []
        for price in prices[:levels]:
            level = self._levels[price]
            out.append(
                BookLevel(
                    price=Ticks(price),
                    quantity=Quantity(level.total_quantity),
                    order_count=level.order_count,
                )
            )
        return out

    @staticmethod
    def _side_of_level(level: _Level) -> Side | None:
        return level.head.side if level.head is not None else None

    def iter_orders(self) -> list[_Node]:
        """Every resting node, for invariant checks. Not part of the hot path."""
        return list(self._order_index.values())

    # -- mutation ---------------------------------------------------------- #
    def add(self, order: Order) -> OrderId:
        """Rest an order on the book. Does **not** match.

        The caller is responsible for having already crossed the order against
        the opposite side; resting a crossing order raises, because a crossed
        book is an invariant violation rather than a recoverable state.

        Raises
        ------
            :class:`BookError` if ``order_id`` is already resting or the price
            would cross the opposite side.
        """
        if int(order.order_id) in self._order_index:
            raise BookError(f"duplicate resting order id {order.order_id}")
        if order.price is None:
            raise BookError("cannot rest an order without a limit price")

        price = int(order.price)
        opposing = self.best_ask if order.side is Side.BUY else self.best_bid
        if opposing is not None and order.side.is_aggressive_at(Ticks(price), opposing):
            raise BookError(
                f"refusing to rest a crossing order: {order.side.name} at {price} "
                f"against best {opposing}. Match it first."
            )

        self._sequence += 1
        node = _Node(
            order_id=order.order_id,
            agent_id=order.agent_id,
            side=order.side,
            price=Ticks(price),
            remaining=order.quantity,
            timestamp=order.timestamp,
            sequence=self._sequence,
        )

        level = self._levels.get(price)
        if level is None or level.is_empty:
            level = _Level(price=Ticks(price))
            self._levels[price] = level
        # Push only if this price is not already represented in the heap.
        if order.side is Side.BUY:
            if price not in self._bid_in_heap:
                heapq.heappush(self._bid_heap, -price)
                self._bid_in_heap.add(price)
        elif price not in self._ask_in_heap:
            heapq.heappush(self._ask_heap, price)
            self._ask_in_heap.add(price)

        level.push_back(node)
        self._order_index[int(order.order_id)] = node
        return OrderId(int(order.order_id))

    def cancel(self, order_id: OrderId) -> Quantity:
        """Remove a resting order. Returns the quantity that was cancelled.

        O(1): the id map locates the node and the intrusive links splice it out
        without traversing the queue.

        Raises
        ------
            :class:`OrderNotFound` if the order is not resting -- which, note,
            includes the case where it was just fully filled. Callers racing a
            cancel against a fill must handle this exception; that race is real
            in live markets and the simulator reproduces it faithfully.
        """
        node = self._order_index.pop(int(order_id), None)
        if node is None:
            raise OrderNotFound(f"order {order_id} is not resting on the book")
        level = self._levels[int(node.price)]
        level.unlink(node)
        # The level is left in place (and in the heap) even when empty; it is
        # reclaimed lazily by _prune. Deleting it here would be O(n) on the heap.
        return node.remaining

    def amend(self, order_id: OrderId, new_quantity: Quantity) -> Quantity:
        """Change a resting order's quantity, following exchange priority rules.

        **A reduction keeps queue position; an increase loses it.** This is not
        an implementation convenience, it is how real venues work, and it is
        economically load-bearing: a market maker can shade size down while
        holding its place in the queue, but must go to the back to add size.
        A simulator that let increases keep priority would make queue-position
        strategies look far more profitable than they are.

        Returns
        -------
            The new quantity.
        """
        node = self._order_index.get(int(order_id))
        if node is None:
            raise OrderNotFound(f"order {order_id} is not resting on the book")
        if new_quantity <= 0:
            raise ValueError("amended quantity must be positive; cancel instead")

        level = self._levels[int(node.price)]
        if int(new_quantity) < int(node.remaining):
            level.total_quantity -= int(node.remaining) - int(new_quantity)
            node.remaining = new_quantity
            return new_quantity

        if int(new_quantity) == int(node.remaining):
            return new_quantity

        # Size increase: cancel and re-add at the back of the queue.
        level.unlink(node)
        self._sequence += 1
        node.remaining = new_quantity
        node.sequence = self._sequence
        level.push_back(node)
        return new_quantity

    def match(
        self,
        side: Side,
        quantity: Quantity,
        limit_price: Ticks | None,
        timestamp: Nanos,
    ) -> tuple[list[tuple[_Node, int]], int]:
        """Walk the opposite side, consuming liquidity in price-time order.

        Purpose
            The core matching loop. Returns the makers hit and the residual.
        Inputs
            ``side`` -- the *aggressor's* side. ``limit_price`` -- ``None`` for
            a market order (sweep any price), otherwise the worst acceptable
            price.
        Outputs
            ``(fills, remaining)`` where ``fills`` is a list of
            ``(maker_node, quantity)`` pairs in execution order, and
            ``remaining`` is the unfilled aggressor quantity.
        Assumptions
            **Fills print at the resting (maker) order's price, not the
            aggressor's limit.** This is price improvement, and it is how every
            major venue works. Getting it backwards silently inflates every
            backtested strategy's costs and understates the value of passive
            execution.
        Complexity
            O(f + L) where ``f`` is the number of orders filled and ``L`` the
            number of price levels consumed.

        Note this method *mutates* the book (consumed makers are removed) but
        does not emit :class:`~quantos.core.types.Trade` objects -- the
        matching engine owns the tape, sequence numbers and fee assignment.
        """
        remaining = int(quantity)
        fills: list[tuple[_Node, int]] = []
        opposite = side.opposite

        while remaining > 0:
            best = self.best_ask if side is Side.BUY else self.best_bid
            if best is None:
                break
            if limit_price is not None and not side.is_aggressive_at(limit_price, best):
                break

            level = self._levels[int(best)]
            while remaining > 0 and level.head is not None:
                maker = level.head
                traded = min(remaining, int(maker.remaining))
                fills.append((maker, traded))
                remaining -= traded

                if traded == int(maker.remaining):
                    level.unlink(maker)
                    del self._order_index[int(maker.order_id)]
                else:
                    maker.remaining = Quantity(int(maker.remaining) - traded)
                    level.total_quantity -= traded

            if level.is_empty:
                # Drop the exhausted level; _prune will reclaim the heap entry.
                self._levels.pop(int(best), None)

        del opposite
        return fills, remaining

    def clear(self) -> None:
        """Empty the book completely (used between simulation episodes)."""
        self._levels.clear()
        self._bid_heap.clear()
        self._ask_heap.clear()
        self._bid_in_heap.clear()
        self._ask_in_heap.clear()
        self._order_index.clear()
        self._sequence = 0

    # -- diagnostics ------------------------------------------------------- #
    def check_invariants(self) -> None:
        """Assert every structural invariant. Raises :class:`BookError`.

        Called by the property-based tests after *every* operation in a random
        sequence. Cheap enough to enable in simulation debug runs.
        """
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None and int(bid) >= int(ask):
            raise BookError(f"book is crossed: bid {bid} >= ask {ask}")

        seen = 0
        for price, level in self._levels.items():
            total = 0
            count = 0
            node = level.head
            prev = None
            while node is not None:
                if int(node.price) != price:
                    raise BookError(f"node {node.order_id} filed under wrong price")
                if node.prev is not prev:
                    raise BookError(f"broken back-link at order {node.order_id}")
                if int(node.remaining) <= 0:
                    raise BookError(f"order {node.order_id} rests with zero quantity")
                total += int(node.remaining)
                count += 1
                prev, node = node, node.next
            if level.tail is not prev:
                raise BookError(f"level {price} tail pointer is stale")
            if total != level.total_quantity:
                raise BookError(
                    f"level {price} cached quantity {level.total_quantity} != actual {total}"
                )
            if count != level.order_count:
                raise BookError(f"level {price} cached count {level.order_count} != actual {count}")
            seen += count

        if seen != len(self._order_index):
            raise BookError(
                f"order index holds {len(self._order_index)} orders but "
                f"{seen} are reachable from the levels"
            )
