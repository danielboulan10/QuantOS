"""Shared value types and data contracts.

Design notes
------------
**Prices are integers.** Every price in QuantOS is an integer count of ticks,
never a float. Floating-point prices make exact price-level equality
unreliable, and an order book is fundamentally a hash map keyed by price: if
``0.1 + 0.2 != 0.3`` then two orders that a trader intends to rest at the same
level can land in different buckets. Real exchanges quote in integer ticks for
this reason. Conversion to a human-facing decimal happens only at the
presentation boundary, via :class:`TickSizeRule`.

**Timestamps are integer nanoseconds** since an arbitrary epoch, for the same
reason plus one more: a discrete-event simulator must be able to order events
deterministically, and float timestamps break ties inconsistently across
platforms.

**Everything is frozen.** Market data objects are immutable value types. A
strategy that receives a :class:`Trade` cannot corrupt the tape for every other
subscriber, which removes an entire class of backtest-vs-live divergence.
``slots=True`` keeps the memory cost of that discipline low -- it matters when
a simulation produces ten million events.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final, NewType

__all__ = [
    "NANOS_PER_MICRO",
    "NANOS_PER_MILLI",
    "NANOS_PER_SECOND",
    "AgentId",
    "BookLevel",
    "Fill",
    "Liquidity",
    "Nanos",
    "Order",
    "OrderId",
    "OrderType",
    "Quantity",
    "Side",
    "TickSizeRule",
    "Ticks",
    "TimeInForce",
    "TopOfBook",
    "Trade",
]

Ticks = NewType("Ticks", int)
"""A price, expressed as an integer number of ticks. Never a float."""

Quantity = NewType("Quantity", int)
"""A size, in integer lots/shares."""

Nanos = NewType("Nanos", int)
"""A timestamp or duration in integer nanoseconds."""

OrderId = NewType("OrderId", int)
AgentId = NewType("AgentId", str)

NANOS_PER_SECOND: Final[int] = 1_000_000_000
NANOS_PER_MILLI: Final[int] = 1_000_000
NANOS_PER_MICRO: Final[int] = 1_000


class Side(enum.IntEnum):
    """Order side, valued as the payoff sign.

    Values are ``+1``/``-1`` rather than ``0``/``1`` so that a side can be used
    directly as a signed multiplier: ``pnl = side * (exit - entry) * qty``, and
    ``-side`` is the opposite side. This removes a great many ``if`` statements
    from the matching engine and the position accounting.
    """

    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        return int(self.value)

    def is_aggressive_at(self, limit: Ticks, opposing_best: Ticks | None) -> bool:
        """Report whether an order at ``limit`` would cross ``opposing_best``."""
        if opposing_best is None:
            return False
        return limit >= opposing_best if self is Side.BUY else limit <= opposing_best


class OrderType(enum.Enum):
    """Supported order types, kept deliberately small.

    Deliberately small. Each additional type must earn its place by exercising
    a distinct code path in the matching engine (see ``docs/ers/ERS-003``);
    exotic types that decompose into these are the client's business, not the
    exchange's.
    """

    LIMIT = "limit"
    MARKET = "market"
    #: Never crosses; rejected outright if it would take liquidity. The order
    #: type that makes a maker-rebate strategy expressible without race
    #: conditions.
    POST_ONLY = "post_only"


class TimeInForce(enum.Enum):
    """Order lifetime instructions."""

    #: Rests on the book until cancelled.
    GTC = "gtc"
    #: Fill whatever is immediately available, cancel the remainder.
    IOC = "ioc"
    #: Fill the *entire* quantity immediately or cancel all of it.
    FOK = "fok"


class Liquidity(enum.Enum):
    """Whether a fill added or removed liquidity -- drives the fee schedule."""

    MAKER = "maker"
    TAKER = "taker"


@dataclass(frozen=True, slots=True)
class Order:
    """A client order request.

    Note this is the *request*, not the resting book entry: the book's internal
    node type carries mutable remaining quantity and queue links, and is
    private to :mod:`quantos.exchange.book`. Keeping the two separate is what
    lets this type stay immutable.
    """

    order_id: OrderId
    agent_id: AgentId
    side: Side
    quantity: Quantity
    price: Ticks | None = None
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: Nanos = Nanos(0)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.MARKET:
            if self.price is not None:
                raise ValueError("market orders must not carry a limit price")
            if self.time_in_force is TimeInForce.GTC:
                raise ValueError(
                    "market orders cannot be GTC: an unfilled remainder has no "
                    "resting price. Use IOC (default semantics) or FOK."
                )
        elif self.price is None:
            raise ValueError(f"{self.order_type.value} orders require a limit price")


@dataclass(frozen=True, slots=True)
class Fill:
    """One side of an execution, from the perspective of a single order."""

    order_id: OrderId
    agent_id: AgentId
    side: Side
    price: Ticks
    quantity: Quantity
    timestamp: Nanos
    liquidity: Liquidity
    #: Monotonic sequence number of the parent trade on the public tape.
    trade_seq: int = 0

    @property
    def signed_quantity(self) -> int:
        """Position delta: positive for a buy, negative for a sell."""
        return self.side.sign * self.quantity

    @property
    def notional_ticks(self) -> int:
        return int(self.price) * int(self.quantity)


@dataclass(frozen=True, slots=True)
class Trade:
    """A public tape print: two orders matched.

    ``aggressor_side`` is the side of the order that *removed* liquidity. It is
    the single most valuable field on the tape -- trade sign is the input to
    order-flow imbalance, VPIN, Kyle's lambda and the entire price-impact
    literature. Real feeds usually make you infer it with the Lee-Ready rule;
    a simulator that knows the truth lets us *measure how badly Lee-Ready
    misclassifies*, which is a genuine research question
    (see :mod:`quantos.research.features.microstructure`).
    """

    seq: int
    price: Ticks
    quantity: Quantity
    timestamp: Nanos
    aggressor_side: Side
    maker_order_id: OrderId
    taker_order_id: OrderId

    @property
    def signed_volume(self) -> int:
        return self.aggressor_side.sign * self.quantity


@dataclass(frozen=True, slots=True)
class BookLevel:
    """Aggregate resting interest at one price."""

    price: Ticks
    quantity: Quantity
    order_count: int


@dataclass(frozen=True, slots=True)
class TopOfBook:
    """Best bid and offer with sizes; the most-consumed market data object.

    Either side may be ``None`` when the book is one-sided, which happens
    routinely at the open and after a sweep. Every derived quantity therefore
    returns ``None`` rather than raising -- forcing callers to handle the empty
    book explicitly is the point.
    """

    timestamp: Nanos
    bid_price: Ticks | None
    bid_size: Quantity | None
    ask_price: Ticks | None
    ask_size: Quantity | None

    @property
    def mid(self) -> float | None:
        """Arithmetic mid. ``None`` if either side is empty."""
        if self.bid_price is None or self.ask_price is None:
            return None
        return 0.5 * (int(self.bid_price) + int(self.ask_price))

    @property
    def spread(self) -> int | None:
        """Quoted spread in ticks."""
        if self.bid_price is None or self.ask_price is None:
            return None
        return int(self.ask_price) - int(self.bid_price)

    @property
    def microprice(self) -> float | None:
        r"""Size-weighted mid, :math:`\frac{Q_b P_a + Q_a P_b}{Q_a + Q_b}`.

        Leans toward the side with *less* size, because that side is the one
        likely to be exhausted next. Stoikov (2018) shows this is the correct
        first-order estimate of the future mid conditional on current book
        state, and it materially outperforms the arithmetic mid as a fair-value
        anchor for market making.
        """
        if (
            self.bid_price is None
            or self.ask_price is None
            or self.bid_size is None
            or self.ask_size is None
        ):
            return None
        total = int(self.bid_size) + int(self.ask_size)
        if total == 0:
            return self.mid
        return (
            int(self.bid_size) * int(self.ask_price) + int(self.ask_size) * int(self.bid_price)
        ) / total

    @property
    def imbalance(self) -> float | None:
        r"""Queue imbalance :math:`(Q_b - Q_a)/(Q_b + Q_a) \in [-1, 1]`."""
        if self.bid_size is None or self.ask_size is None:
            return None
        total = int(self.bid_size) + int(self.ask_size)
        if total == 0:
            return None
        return (int(self.bid_size) - int(self.ask_size)) / total


@dataclass(frozen=True, slots=True)
class TickSizeRule:
    """Converts between integer ticks and decimal prices at the UI boundary.

    Kept as an explicit object rather than a module constant because tick size
    is an instrument property, and mixing instruments with different tick sizes
    in one process is normal.
    """

    tick_size: float = 0.01
    reference_price: float = 100.0

    def to_decimal(self, ticks: Ticks) -> float:
        """Integer ticks to a decimal price."""
        return round(int(ticks) * self.tick_size, 10)

    def to_ticks(self, price: float) -> Ticks:
        """Decimal price to the nearest integer tick (round-half-to-even)."""
        return Ticks(round(price / self.tick_size))

    @property
    def reference_ticks(self) -> Ticks:
        return self.to_ticks(self.reference_price)
