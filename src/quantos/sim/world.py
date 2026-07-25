r"""The simulation world: agents, a venue, latency, and a recorded tape.

What the world is responsible for
---------------------------------
Everything agents are deliberately *not* allowed to do:

1. **Assign order ids.** Agents request actions; the world issues ids. An agent
   cannot forge or guess another's id.
2. **Impose latency.** Every action is delayed by the agent's own latency draw
   before it reaches the matching engine, and every fill notification is delayed
   again on the way back. Zero-latency backtests make queue-position and
   liquidity-taking strategies look far better than they are, because they let a
   strategy react to a quote it could not physically have seen yet.
3. **Deliver market data.** Snapshots are pushed to agents; agents never pull
   from the book. This is the structural guarantee against look-ahead bias.
4. **Record.** The tape, the top-of-book series, and every agent's equity curve
   are captured for the research layer.

Latency model
-------------
Each agent draws its latency from a shifted log-normal: a hard floor (speed of
light plus fixed switch hops) plus a heavy right tail (queueing, GC pauses,
kernel scheduling). Log-normal, not Gaussian, because measured exchange latency
distributions are strongly right-skewed -- the mean is dominated by rare large
delays, and a symmetric model both understates tail risk and, worse, can produce
negative latencies.

Reproducibility
---------------
The entire world derives from one :class:`~quantos.core.rng.SeedBank`. Two runs
with the same seed and configuration produce byte-identical tapes; adding an
agent does not perturb the existing agents' random streams. See
:mod:`quantos.core.rng` for why that property is not free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from quantos.core.rng import SeedBank
from quantos.core.types import (
    AgentId,
    Fill,
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
from quantos.exchange.fees import FeeModel, MakerTakerFees
from quantos.exchange.matching import MatchingEngine
from quantos.sim.agents import Agent, CancelOrder, InstitutionalTrader, MarketMaker, SubmitOrder
from quantos.sim.clock import EventPriority, SimulationClock
from quantos.sim.fundamental import FundamentalValue

__all__ = ["LatencyModel", "MarketSimulation", "SimulationConfig", "SimulationResult"]


@dataclass(frozen=True)
class LatencyModel:
    r"""Shifted log-normal one-way latency, in nanoseconds.

    .. math:: \text{latency} = \text{floor} + \exp(\mu + \sigma Z)

    ``floor_ns`` is the irreducible physical minimum; the log-normal part is
    queueing and jitter. Defaults are loosely calibrated to a co-located
    participant: ~50 microseconds typical, with a tail into the low
    milliseconds.
    """

    floor_ns: int = 20_000
    log_mean: float = 10.0
    log_sigma: float = 0.7

    def draw(self, rng: np.random.Generator) -> Nanos:
        """One latency sample."""
        return Nanos(
            self.floor_ns + int(np.exp(self.log_mean + self.log_sigma * rng.standard_normal()))
        )

    def mean_ns(self) -> float:
        r"""Analytic mean, :math:`\text{floor} + e^{\mu + \sigma^2/2}`."""
        return self.floor_ns + float(np.exp(self.log_mean + 0.5 * self.log_sigma**2))


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    duration_ns: int = 60_000_000_000  # 60 seconds of market time
    initial_price_ticks: int = 10_000
    #: Interval at which top-of-book is sampled into the recorded series.
    snapshot_interval_ns: int = 1_000_000
    tick_value: float = 0.01
    seed: int = 20240719
    #: Seed the book with resting liquidity so the first agents see a market.
    seed_book_levels: int = 10
    seed_book_size: int = 50


@dataclass
class SimulationResult:
    """Everything a simulation produced, ready for the research layer."""

    trades: list[Trade]
    #: Sampled top-of-book snapshots.
    snapshots: list[TopOfBook]
    #: agent_id -> equity curve sampled at the snapshot interval.
    equity_curves: dict[str, list[float]] = field(default_factory=dict)
    #: agent_id -> final state summary.
    agent_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    events_processed: int = 0
    config: SimulationConfig | None = None
    #: (timestamps_ns, values) of the latent fundamental the market never saw.
    fundamental_path: tuple[np.ndarray, np.ndarray] | None = None

    # -- derived series ---------------------------------------------------- #
    def mid_series(self) -> np.ndarray:
        """Sampled mid prices, with gaps forward-filled.

        Forward-filling a one-sided book is the honest choice: the mid is
        genuinely unobservable there, and interpolating would invent a price
        that no one could have traded at.
        """
        out: list[float] = []
        last = np.nan
        for snapshot in self.snapshots:
            mid = snapshot.mid
            if mid is not None:
                last = mid
            out.append(last)
        return np.asarray(out, dtype=float)

    def trade_price_series(self) -> np.ndarray:
        return np.asarray([float(t.price) for t in self.trades], dtype=float)

    def returns(self, *, from_trades: bool = False) -> np.ndarray:
        """Log returns of the mid (or of the trade series)."""
        prices = self.trade_price_series() if from_trades else self.mid_series()
        prices = prices[np.isfinite(prices) & (prices > 0)]
        if prices.size < 2:
            return np.zeros(0)
        return np.diff(np.log(prices))

    def spread_series(self) -> np.ndarray:
        return np.asarray(
            [s.spread if s.spread is not None else np.nan for s in self.snapshots], dtype=float
        )

    def signed_volume_series(self) -> np.ndarray:
        """Aggressor-signed trade volume -- the input to order-flow imbalance."""
        return np.asarray([t.signed_volume for t in self.trades], dtype=float)

    def summary(self) -> str:  # pragma: no cover - display
        r = self.returns()
        lines = [
            f"trades={len(self.trades)}  snapshots={len(self.snapshots)}  "
            f"events={self.events_processed}",
        ]
        if r.size > 1:
            lines.append(
                f"return vol (per snapshot) = {float(np.std(r)):.3e}  "
                f"mean spread = {float(np.nanmean(self.spread_series())):.2f} ticks"
            )
        for agent_id, stats in sorted(self.agent_summary.items()):
            lines.append(
                f"  {agent_id:<22} pnl={stats['pnl']:>12.2f}  pos={stats['position']:>7.0f}  "
                f"fills={stats['n_fills']:>6.0f}  fees={stats['fees_paid']:>9.2f}"
            )
        return "\n".join(lines)


class MarketSimulation:
    """Agent-based market simulation over a :class:`MatchingEngine`.

    Example
        >>> import numpy as np
        >>> from quantos.sim.agents import NoiseTrader, MarketMaker
        >>> cfg = SimulationConfig(duration_ns=2_000_000_000, seed=7)
        >>> sim = MarketSimulation(cfg)
        >>> bank = sim.seed_bank
        >>> for i in range(6):
        ...     _ = sim.add_agent(NoiseTrader(f"noise_{i}",
        ...                       bank.child(f"noise_{i}").generator()))
        >>> _ = sim.add_agent(MarketMaker("mm", bank.child("mm").generator()))
        >>> result = sim.run()
        >>> bool(len(result.trades) > 0)
        True
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        fees: FeeModel | None = None,
        latency: LatencyModel | None = None,
        fundamental: FundamentalValue | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.seed_bank = SeedBank(root=self.config.seed).child("sim")
        # The fundamental value belongs to the world, so that every informed
        # trader observes the *same* process. See quantos/sim/fundamental.py for
        # why an earlier per-agent design produced a market that would not move.
        self.fundamental = (
            fundamental
            if fundamental is not None
            else FundamentalValue(
                initial=float(self.config.initial_price_ticks),
                volatility=2.0,
                rng=self.seed_bank.child("fundamental").generator(),
            )
        )
        self.engine = MatchingEngine(fees=fees or MakerTakerFees(tick_value=self.config.tick_value))
        self.clock = SimulationClock()
        self.latency = latency or LatencyModel()
        self.agents: dict[str, Agent] = {}

        self._next_order_id = 1
        self._order_owner: dict[int, str] = {}
        self._latency_rng = self.seed_bank.child("latency").generator()
        self._snapshots: list[TopOfBook] = []
        self._equity: dict[str, list[float]] = {}

    # -- setup ------------------------------------------------------------- #
    def add_agent(self, agent: Agent) -> Agent:
        """Register an agent. Its wakeup timer is scheduled by :meth:`run`."""
        if str(agent.agent_id) in self.agents:
            raise ValueError(f"duplicate agent id {agent.agent_id!r}")
        self.agents[str(agent.agent_id)] = agent
        self._equity[str(agent.agent_id)] = []
        return agent

    def _seed_book(self) -> None:
        """Rest initial liquidity so the first agents observe a two-sided market.

        Attributed to a synthetic ``__initial__`` agent rather than to any
        participant, so it does not contaminate anyone's PnL. Without seeding,
        the first hundred events are spent on price discovery from an empty book,
        which is a different (and less interesting) experiment.
        """
        centre = self.config.initial_price_ticks
        for level in range(1, self.config.seed_book_levels + 1):
            for side, price in (
                (Side.BUY, centre - level),
                (Side.SELL, centre + level),
            ):
                order_id = OrderId(self._next_order_id)
                self._next_order_id += 1
                self._order_owner[int(order_id)] = "__initial__"
                self.engine.submit(
                    Order(
                        order_id=order_id,
                        agent_id=AgentId("__initial__"),
                        side=side,
                        quantity=Quantity(self.config.seed_book_size),
                        price=Ticks(price),
                        order_type=OrderType.LIMIT,
                        time_in_force=TimeInForce.GTC,
                        timestamp=Nanos(0),
                    )
                )

    # -- event plumbing ---------------------------------------------------- #
    def _dispatch(self, agent: Agent, actions: list, timestamp: Nanos) -> None:
        """Send an agent's actions to the venue, after its latency."""
        for action in actions:
            delay = self.latency.draw(self._latency_rng)
            if isinstance(action, SubmitOrder):
                order_id = OrderId(self._next_order_id)
                self._next_order_id += 1
                self._order_owner[int(order_id)] = str(agent.agent_id)
                agent.state.open_orders.add(int(order_id))
                if isinstance(agent, MarketMaker):
                    agent.register_quote(order_id, action.side)
                self.clock.schedule_after(
                    delay,
                    self._make_submit_callback(agent, action, order_id),
                    priority=EventPriority.ORDER_ARRIVAL,
                    label=f"submit:{agent.agent_id}",
                )
            elif isinstance(action, CancelOrder):
                self.clock.schedule_after(
                    delay,
                    self._make_cancel_callback(agent, action.order_id),
                    priority=EventPriority.ORDER_CANCEL,
                    label=f"cancel:{agent.agent_id}",
                )

    def _make_submit_callback(
        self, agent: Agent, action: SubmitOrder, order_id: OrderId
    ) -> Callable[[Nanos], None]:
        def callback(timestamp: Nanos) -> None:
            order = Order(
                order_id=order_id,
                agent_id=agent.agent_id,
                side=action.side,
                quantity=action.quantity,
                price=action.price,
                order_type=action.order_type,
                time_in_force=action.time_in_force,
                timestamp=timestamp,
            )
            report = self.engine.submit(order)
            if not report.accepted or report.resting_quantity == 0:
                agent.state.open_orders.discard(int(order_id))

            # Taker fills go to the submitting agent.
            for fill in report.fills:
                agent.state.apply_fill(fill, self.engine.fees.charge(fill), self.config.tick_value)
                self._notify_fill(agent, fill)
            # Maker fills go to whoever owned the resting order.
            self._distribute_maker_fills()
            # Institutional POV agents need public volume.
            for filled in report.fills:
                for other in self.agents.values():
                    if isinstance(other, InstitutionalTrader):
                        other.observe_volume(int(filled.quantity))

        return callback

    def _make_cancel_callback(self, agent: Agent, order_id: OrderId) -> Callable[[Nanos], None]:
        def callback(timestamp: Nanos) -> None:
            self.engine.cancel(order_id)
            agent.state.open_orders.discard(int(order_id))

        return callback

    def _distribute_maker_fills(self) -> None:
        """Route the engine's asynchronous maker fills to their owners."""
        for fill in self.engine.drain_maker_fills():
            owner = self._order_owner.get(int(fill.order_id))
            if owner is None or owner == "__initial__":
                continue
            agent = self.agents.get(owner)
            if agent is None:
                continue
            agent.state.apply_fill(fill, self.engine.fees.charge(fill), self.config.tick_value)
            if fill.quantity > 0:
                agent.state.open_orders.discard(int(fill.order_id))
            self._notify_fill(agent, fill)

    def _notify_fill(self, agent: Agent, fill: Fill) -> None:
        """Deliver a fill notification after the return-trip latency."""
        delay = self.latency.draw(self._latency_rng)

        def callback(timestamp: Nanos) -> None:
            self._dispatch(agent, agent.on_fill(fill, timestamp), timestamp)

        self.clock.schedule_after(
            delay,
            callback,
            priority=EventPriority.FILL_NOTIFICATION,
            label=f"fill:{agent.agent_id}",
        )

    def _make_wakeup(self, agent: Agent) -> Callable[[Nanos], None]:
        def callback(timestamp: Nanos) -> None:
            book = self.engine.top_of_book(timestamp)
            self._dispatch(agent, agent.on_wakeup(book, timestamp), timestamp)

        return callback

    def _snapshot(self, timestamp: Nanos) -> None:
        # Advance the fundamental on the world's own clock. It must NOT be left
        # to advance lazily when an agent happens to query it: if informed
        # traders return early (one-sided book, edge below threshold), the value
        # would freeze, and the latent process would then depend on agent
        # behaviour rather than driving it. That bug held the stressed scenario's
        # fundamental to a 3-tick range when it should have moved ~100.
        self.fundamental.value_at(timestamp)
        book = self.engine.top_of_book(timestamp)
        self._snapshots.append(book)
        mid = book.mid
        for agent_id, agent in self.agents.items():
            self._equity[agent_id].append(agent.state.mark_to_market(mid, self.config.tick_value))
        # Market-data delivery is a separate, higher-priority event class, so
        # agents observe the book *before* any order that reacts to it arrives.
        for agent in self.agents.values():
            self._dispatch(agent, agent.on_market_data(book, timestamp), timestamp)

    # -- run --------------------------------------------------------------- #
    def run(self) -> SimulationResult:
        """Execute the simulation and return everything it produced."""
        if not self.agents:
            raise ValueError("no agents registered; add at least one before running")

        self.clock.reset()
        self._snapshots.clear()
        for key in self._equity:
            self._equity[key] = []
        self._seed_book()

        # Stagger the agents' first wakeup so they do not all fire on the same
        # nanosecond forever, which would make the population act in lockstep
        # and produce artefacts in the tape.
        jitter_rng = self.seed_bank.child("wakeup_jitter").generator()
        for agent in self.agents.values():
            if agent.wakeup_interval is None:
                continue
            interval = int(agent.wakeup_interval)
            offset = int(jitter_rng.integers(0, max(1, interval)))
            self.clock.schedule_at(
                Nanos(offset),
                self._make_wakeup(agent),
                priority=EventPriority.AGENT_WAKEUP,
                label=f"wakeup:{agent.agent_id}",
            )
            self.clock.schedule_recurring(
                Nanos(interval),
                self._make_wakeup(agent),
                until=Nanos(self.config.duration_ns),
                priority=EventPriority.AGENT_WAKEUP,
                label=f"wakeup:{agent.agent_id}",
            )

        self.clock.schedule_recurring(
            Nanos(self.config.snapshot_interval_ns),
            self._snapshot,
            until=Nanos(self.config.duration_ns),
            priority=EventPriority.MARKET_DATA,
            label="snapshot",
        )

        processed = self.clock.run(until=Nanos(self.config.duration_ns))

        final_book = self.engine.top_of_book(self.clock.now)
        summary = {
            agent_id: {
                "pnl": agent.state.mark_to_market(final_book.mid, self.config.tick_value),
                "position": float(agent.state.position),
                "n_fills": float(agent.state.n_fills),
                "volume": float(agent.state.volume_traded),
                "fees_paid": agent.state.fees_paid,
            }
            for agent_id, agent in self.agents.items()
        }

        return SimulationResult(
            trades=list(self.engine.trades),
            snapshots=list(self._snapshots),
            equity_curves={k: list(v) for k, v in self._equity.items()},
            agent_summary=summary,
            events_processed=processed,
            config=self.config,
            fundamental_path=self.fundamental.path_arrays(),
        )
