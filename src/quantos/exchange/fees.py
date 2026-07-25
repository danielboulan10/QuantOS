"""Exchange fee schedules.

Why this is its own module
--------------------------
Fees are not a detail. A maker-taker schedule is the economic reason market
makers post rather than cross, and the sign of a passive strategy's PnL often
*is* the rebate. Backtests that omit fees do not merely overstate returns --
they invert the ranking of passive against aggressive strategies. Making the
schedule an injected object rather than a constant means every backtest states
its fee assumption explicitly, and swapping venues is a one-line change.

The default is a maker-taker schedule in the range US equity exchanges actually
charge: a 0.20 bp rebate to makers, 0.30 bp charged to takers.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from quantos.core.types import Fill, Liquidity

__all__ = ["FeeSchedule", "FlatFees", "MakerTakerFees", "NoFees", "TieredFees"]


class FeeModel(abc.ABC):
    """Interface for anything that prices a fill.

    Positive is a cost, negative is a rebate. Sticking to one sign convention
    everywhere removes a whole family of PnL sign errors.
    """

    @abc.abstractmethod
    def charge(self, fill: Fill) -> float:
        """Fee for one fill, in currency units. Negative means a rebate."""


@dataclass(frozen=True)
class MakerTakerFees(FeeModel):
    """Proportional maker-taker schedule, quoted in basis points of notional.

    Example
        >>> from quantos.core.types import Fill, OrderId, AgentId, Side, Ticks, Quantity, Nanos
        >>> sched = MakerTakerFees()
        >>> maker = Fill(OrderId(1), AgentId("a"), Side.BUY, Ticks(10_000),
        ...              Quantity(100), Nanos(0), Liquidity.MAKER)
        >>> # 10,000 ticks x 100 lots x $0.01 = $10,000 notional; 0.20bp rebate.
        >>> round(sched.charge(maker), 6)     # a rebate, hence negative
        -0.2
    """

    #: Basis points paid *to* the maker (so a positive number is a rebate).
    maker_rebate_bps: float = 0.20
    #: Basis points charged to the taker.
    taker_fee_bps: float = 0.30
    #: Value of one tick in currency units, for converting notional.
    tick_value: float = 0.01

    def charge(self, fill: Fill) -> float:
        notional = fill.notional_ticks * self.tick_value
        if fill.liquidity is Liquidity.MAKER:
            return -notional * self.maker_rebate_bps * 1e-4
        return notional * self.taker_fee_bps * 1e-4


@dataclass(frozen=True)
class FlatFees(FeeModel):
    """Flat per-share commission, irrespective of liquidity flag."""

    per_share: float = 0.0035

    def charge(self, fill: Fill) -> float:
        return self.per_share * int(fill.quantity)


@dataclass(frozen=True)
class NoFees(FeeModel):
    """Zero fees. Useful for isolating a strategy's raw signal from its costs.

    Never the default: making "free" the explicit opt-in rather than the
    accident is the point.
    """

    def charge(self, fill: Fill) -> float:
        return 0.0


@dataclass(frozen=True)
class TieredFees(FeeModel):
    """Volume-tiered maker-taker, the schedule most real venues actually use.

    ``tiers`` maps a monthly volume threshold to ``(maker_bps, taker_bps)``.
    The applicable tier is the highest threshold not exceeding
    ``running_volume``, which the caller updates as the month progresses.

    Included because tiering creates a genuine strategic effect a flat schedule
    cannot express: the marginal cost of the trade that pushes you into the
    next tier is negative, so optimal execution near a boundary is
    discontinuous.
    """

    tiers: tuple[tuple[int, float, float], ...] = (
        (0, 0.15, 0.30),
        (1_000_000, 0.20, 0.28),
        (10_000_000, 0.25, 0.25),
        (100_000_000, 0.32, 0.20),
    )
    tick_value: float = 0.01
    running_volume: int = 0

    def _rates(self) -> tuple[float, float]:
        maker, taker = self.tiers[0][1], self.tiers[0][2]
        for threshold, m, t in self.tiers:
            if self.running_volume >= threshold:
                maker, taker = m, t
        return maker, taker

    def charge(self, fill: Fill) -> float:
        maker_bps, taker_bps = self._rates()
        notional = fill.notional_ticks * self.tick_value
        if fill.liquidity is Liquidity.MAKER:
            return -notional * maker_bps * 1e-4
        return notional * taker_bps * 1e-4


#: Default schedule used by :class:`~quantos.exchange.matching.MatchingEngine`.
FeeSchedule = MakerTakerFees
