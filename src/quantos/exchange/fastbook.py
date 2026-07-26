"""A :class:`~quantos.exchange.book.LimitOrderBook`-compatible C++ backend.

This wraps the compiled extension in exactly the interface the pure Python book
exposes, so :class:`~quantos.exchange.matching.MatchingEngine` cannot tell them
apart. Selecting a backend is therefore a one-line change with no downstream
effect -- which is the property that makes the equivalence test in
``tests/exchange/test_cpp_equivalence.py`` meaningful rather than decorative.

Agent identity
--------------
The C++ layer stores an integer id, not a Python string, because interning
strings across the boundary on every order would cost more than the matching
itself. This wrapper keeps the string table and translates in both directions,
so callers still see :class:`~quantos.core.types.AgentId`.

What is deliberately *not* here
-------------------------------
``check_invariants``. The C++ structures make its invariants unrepresentable: a
``std::map`` cannot hold a stale price, and a level is erased the moment it
empties, so there is no cached-count-versus-reality gap to check. The pure
Python book needs those assertions because its lazy-deleted heaps *can* drift --
and did, until the duplicate-entry leak was found. Asserting properties that
cannot fail would be theatre; the equivalence test against the Python book is
the real check.
"""

from __future__ import annotations

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
from quantos.exchange.book import BookError, OrderNotFound

__all__ = ["EXTENSION_AVAILABLE", "FastLimitOrderBook"]

try:  # pragma: no cover - import success depends on whether the build ran
    from quantos.exchange import _book as _extension  # type: ignore[attr-defined]

    EXTENSION_AVAILABLE = True
except ImportError:  # pragma: no cover
    _extension = None
    EXTENSION_AVAILABLE = False


class _MakerView:
    """The subset of a resting order that :meth:`match` must return.

    The Python book hands back its internal ``_Node`` objects. The C++ book
    cannot, so ``match`` reconstructs the same three fields the matching engine
    actually reads. Keeping the shape identical is what lets the engine stay
    unaware of which backend it is talking to.
    """

    __slots__ = ("agent_id", "order_id", "price", "remaining", "side")

    def __init__(
        self, order_id: OrderId, agent_id: AgentId, price: Ticks, remaining: Quantity, side: Side
    ) -> None:
        self.order_id = order_id
        self.agent_id = agent_id
        self.price = price
        self.remaining = remaining
        self.side = side


class FastLimitOrderBook:
    """Drop-in replacement for :class:`~quantos.exchange.book.LimitOrderBook`."""

    __slots__ = ("_agent_ids", "_agent_index", "_book", "_owner")

    def __init__(self) -> None:
        if not EXTENSION_AVAILABLE:  # pragma: no cover - guarded by callers
            raise RuntimeError(
                "the C++ extension is not built. Run "
                "`python scripts/build_extension.py`, or use LimitOrderBook."
            )
        self._book = _extension.Book()
        self._agent_ids: list[str] = []
        self._agent_index: dict[str, int] = {}
        #: order id -> agent id, so fills can report who owned the order.
        self._owner: dict[int, int] = {}

    # -- agent id interning ------------------------------------------------ #
    def _intern(self, agent_id: AgentId) -> int:
        key = str(agent_id)
        found = self._agent_index.get(key)
        if found is None:
            found = len(self._agent_ids)
            self._agent_ids.append(key)
            self._agent_index[key] = found
        return found

    # -- introspection ----------------------------------------------------- #
    def __len__(self) -> int:
        return int(self._book.order_count())

    def __contains__(self, order_id: object) -> bool:
        return bool(self._book.contains(int(order_id)))  # type: ignore[call-overload]

    @property
    def heap_slack(self) -> int:
        """Always zero: a balanced tree has no stale entries to accumulate."""
        return 0

    @property
    def best_bid(self) -> Ticks | None:
        price = self._book.best_bid()
        return Ticks(price) if price is not None else None

    @property
    def best_ask(self) -> Ticks | None:
        price = self._book.best_ask()
        return Ticks(price) if price is not None else None

    def size_at(self, price: Ticks) -> int:
        return int(self._book.size_at(int(price)))

    def top_of_book(self, timestamp: Nanos = Nanos(0)) -> TopOfBook:
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
        return [
            BookLevel(price=Ticks(price), quantity=Quantity(quantity), order_count=count)
            for price, quantity, count in self._book.depth(side.sign, levels)
        ]

    # -- mutation ---------------------------------------------------------- #
    def add(self, order: Order) -> OrderId:
        if order.price is None:
            raise BookError("cannot rest an order without a limit price")
        status = self._book.add(
            int(order.order_id),
            self._intern(order.agent_id),
            order.side.sign,
            int(order.price),
            int(order.quantity),
        )
        if status == 1:
            raise BookError(f"duplicate resting order id {order.order_id}")
        if status == 2:
            opposing = self.best_ask if order.side is Side.BUY else self.best_bid
            raise BookError(
                f"refusing to rest a crossing order: {order.side.name} at "
                f"{int(order.price)} against best {opposing}. Match it first."
            )
        self._owner[int(order.order_id)] = self._agent_index[str(order.agent_id)]
        return OrderId(int(order.order_id))

    def cancel(self, order_id: OrderId) -> Quantity:
        quantity = self._book.cancel(int(order_id))
        if quantity < 0:
            raise OrderNotFound(f"order {order_id} is not resting on the book")
        self._owner.pop(int(order_id), None)
        return Quantity(int(quantity))

    def amend(self, order_id: OrderId, new_quantity: Quantity) -> Quantity:
        if new_quantity <= 0:
            raise ValueError("amended quantity must be positive; cancel instead")
        result = self._book.amend(int(order_id), int(new_quantity))
        if result < 0:
            raise OrderNotFound(f"order {order_id} is not resting on the book")
        return Quantity(int(result))

    def match(
        self,
        side: Side,
        quantity: Quantity,
        limit_price: Ticks | None,
        timestamp: Nanos,
    ) -> tuple[list[tuple[_MakerView, int]], int]:
        raw, remaining = self._book.match(
            side.sign,
            int(quantity),
            limit_price is not None,
            int(limit_price) if limit_price is not None else 0,
        )
        maker_side = side.opposite
        fills: list[tuple[_MakerView, int]] = []
        for maker_id, traded, price in raw:
            owner = self._owner.get(int(maker_id))
            agent = AgentId(self._agent_ids[owner]) if owner is not None else AgentId("")
            if not self._book.contains(int(maker_id)):
                self._owner.pop(int(maker_id), None)
            fills.append(
                (
                    _MakerView(
                        order_id=OrderId(int(maker_id)),
                        agent_id=agent,
                        price=Ticks(int(price)),
                        remaining=Quantity(int(traded)),
                        side=maker_side,
                    ),
                    int(traded),
                )
            )
        return fills, int(remaining)

    def clear(self) -> None:
        self._book.clear()
        self._owner.clear()

    def iter_orders(self) -> list[_MakerView]:
        """Every resting order, for diagnostics and equivalence testing."""
        return [
            _MakerView(
                order_id=OrderId(int(order_id)),
                agent_id=AgentId(self._agent_ids[self._owner.get(int(order_id), 0)])
                if self._owner
                else AgentId(""),
                price=Ticks(int(price)),
                remaining=Quantity(int(remaining)),
                side=Side.BUY if side > 0 else Side.SELL,
            )
            for order_id, side, price, remaining, _ in self._book.snapshot()
        ]

    def check_invariants(self) -> None:
        """Verify the book is not crossed.

        The remaining invariants the Python book asserts are structurally
        impossible here -- see the module docstring. This one is kept because it
        is a property of the *matching logic*, not of the data structure.
        """
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None and int(bid) >= int(ask):
            raise BookError(f"book is crossed: bid {bid} >= ask {ask}")
