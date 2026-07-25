r"""Trading agents: the population whose interaction *is* the market.

Design contract
---------------
An agent implements :class:`Agent`, whose entire interface is:

* :meth:`Agent.on_market_data` -- a quote update arrived.
* :meth:`Agent.on_fill` -- one of my orders traded.
* :meth:`Agent.on_wakeup` -- my scheduled timer fired.

and which returns :class:`Action` objects rather than calling the venue
directly. That indirection is the single most important design decision in this
module. Because an agent cannot reach the exchange, it cannot:

* observe the book at a time it should not have (no look-ahead),
* have its orders arrive with zero latency (the world injects the delay),
* or bypass the matching engine's risk and fee accounting.

The same agent class therefore runs unmodified in a simulation, a historical
replay, and -- in principle -- against a live venue. A backtester whose
strategies call into the engine cannot make that claim, and the gap between
backtest and live is where most of the money goes.

The population
--------------
Emergence is the point. None of these agents is told to produce fat-tailed
returns or clustered volatility; those arise from the interaction, and
:mod:`quantos.sim.stylized_facts` measures whether they did.

===========================  ===============================================
:class:`NoiseTrader`         Zero-intelligence random orders (Farmer et al.).
                             Supplies the baseline order flow. Remarkably,
                             ZI agents alone reproduce realistic *spreads*.
:class:`InformedTrader`      Knows a private value; trades when price deviates.
                             The source of adverse selection -- without this,
                             market making is riskless and the simulation
                             teaches nothing.
:class:`MarketMaker`         Avellaneda-Stoikov optimal quoting with inventory
                             skew. Provides liquidity, earns the spread, and
                             loses to the informed.
:class:`MomentumTrader`      Trend follower. Amplifies moves; the mechanism by
                             which volatility clusters.
:class:`MeanReversionTrader` Contrarian. Dampens moves; competes with the MM.
:class:`InstitutionalTrader` Executes a large parent order over time. The
                             source of persistent one-sided pressure that
                             generates measurable price impact.
===========================  ===============================================

References
----------
Avellaneda, M. & Stoikov, S. (2008), "High-frequency trading in a limit order
    book", *Quantitative Finance* 8(3), 217-224.
Farmer, J. D., Patelli, P. & Zovko, I. I. (2005), "The predictive power of zero
    intelligence in financial markets", *PNAS* 102(6), 2254-2259.
Glosten, L. R. & Milgrom, P. R. (1985), "Bid, ask and transaction prices in a
    specialist market with heterogeneously informed traders",
    *J. Financial Economics* 14(1), 71-100.
Kyle, A. S. (1985), "Continuous auctions and insider trading",
    *Econometrica* 53(6), 1315-1335.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from quantos.core.types import (
    AgentId,
    Fill,
    Nanos,
    OrderId,
    OrderType,
    Quantity,
    Side,
    Ticks,
    TimeInForce,
    TopOfBook,
)
from quantos.sim.fundamental import FundamentalValue

__all__ = [
    "Action",
    "Agent",
    "AgentState",
    "CancelOrder",
    "InformedTrader",
    "InstitutionalTrader",
    "MarketMaker",
    "MeanReversionTrader",
    "MomentumTrader",
    "NoiseTrader",
    "SubmitOrder",
]


# --------------------------------------------------------------------------- #
# Actions                                                                     #
# --------------------------------------------------------------------------- #
class Action:
    """Something an agent wants the world to do on its behalf.

    A closed tagged union rather than an interface: it carries no behaviour, and
    the world dispatches on concrete type. ``abc.ABC`` would add nothing, since
    there is no method for a subclass to implement.
    """


@dataclass(frozen=True, slots=True)
class SubmitOrder(Action):
    """Request to submit an order. ``order_id`` is assigned by the world."""

    side: Side
    quantity: Quantity
    price: Ticks | None = None
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC


@dataclass(frozen=True, slots=True)
class CancelOrder(Action):
    """Request to cancel a resting order."""

    order_id: OrderId


# --------------------------------------------------------------------------- #
# Agent state                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class AgentState:
    """Position, cash and fee accounting for one agent.

    Mark-to-market uses the *mid*, not the last trade. Last-trade marking makes
    an agent's PnL jump with every bid-ask bounce, which manufactures spurious
    volatility in the PnL series and corrupts every risk statistic computed from
    it. Marking to mid is what real risk systems do.
    """

    agent_id: AgentId
    position: int = 0
    cash: float = 0.0
    fees_paid: float = 0.0
    #: Resting order ids this agent believes are live.
    open_orders: set[int] = field(default_factory=set)
    n_fills: int = 0
    volume_traded: int = 0
    #: Volume-weighted average entry price of the current position, in ticks.
    _cost_basis: float = 0.0

    def apply_fill(self, fill: Fill, fee: float, tick_value: float = 0.01) -> None:
        """Update position, cash and cost basis from one fill."""
        signed = fill.signed_quantity
        notional = float(fill.price) * abs(signed) * tick_value

        # Cost basis: only reset when the position flips or opens.
        if self.position == 0 or (self.position > 0) == (signed > 0):
            total = abs(self.position) + abs(signed)
            if total > 0:
                self._cost_basis = (
                    self._cost_basis * abs(self.position) + float(fill.price) * abs(signed)
                ) / total
        elif abs(signed) > abs(self.position):
            self._cost_basis = float(fill.price)

        self.cash -= signed * notional / abs(signed) if signed else 0.0
        self.position += signed
        self.fees_paid += fee
        self.cash -= fee
        self.n_fills += 1
        self.volume_traded += abs(signed)

    def mark_to_market(self, mid: float | None, tick_value: float = 0.01) -> float:
        """Total equity: cash plus position marked at the mid."""
        if mid is None:
            return self.cash
        return self.cash + self.position * mid * tick_value

    @property
    def average_entry_price(self) -> float:
        return self._cost_basis


# --------------------------------------------------------------------------- #
# Base agent                                                                  #
# --------------------------------------------------------------------------- #
class Agent:
    """Base class for all trading agents.

    Subclasses override the three event handlers. Each returns a (possibly
    empty) sequence of :class:`Action`. Agents must not mutate anything outside
    their own state.

    Deliberately not an ``abc.ABC``: all three handlers have useful no-op
    defaults, so there is nothing to declare abstract. An agent that only reacts
    to market data should not be forced to implement ``on_fill``.
    """

    def __init__(self, agent_id: str, rng: np.random.Generator) -> None:
        self.agent_id = AgentId(agent_id)
        self.rng = rng
        self.state = AgentState(agent_id=self.agent_id)

    #: Nanoseconds between scheduled wakeups; ``None`` for purely reactive agents.
    wakeup_interval: Nanos | None = None

    def on_market_data(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        """React to a quote update. Default: do nothing."""
        return []

    def on_fill(self, fill: Fill, timestamp: Nanos) -> list[Action]:
        """React to one of my orders trading. Default: do nothing."""
        return []

    def on_wakeup(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        """React to my timer firing. Default: do nothing."""
        return []

    def __repr__(self) -> str:  # pragma: no cover - display
        return (
            f"{type(self).__name__}(id={self.agent_id!r}, "
            f"pos={self.state.position}, fills={self.state.n_fills})"
        )


# --------------------------------------------------------------------------- #
# Noise trader                                                                #
# --------------------------------------------------------------------------- #
class NoiseTrader(Agent):
    r"""Zero-intelligence trader (Farmer-Patelli-Zovko).

    Submits random buy/sell orders with random prices drawn around the current
    mid, and cancels at random. It has no view, no memory and no strategy.

    Why bother? Because Farmer et al. (2005) showed that a population of exactly
    this agent reproduces the *magnitude* of real bid-ask spreads and market
    impact from order-flow statistics alone -- no rationality required. It
    establishes how much of observed microstructure is mechanical consequence of
    the matching rules rather than strategic behaviour, which is the right
    null model for everything else in this module.

    Order prices are drawn from a power law around the mid rather than a
    Gaussian: empirically, limit-order placement depth follows a power law with
    exponent near 1.5, with far more orders placed deep in the book than a
    normal distribution allows.
    """

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        *,
        order_size_mean: int = 10,
        depth_scale: float = 3.0,
        depth_exponent: float = 1.5,
        market_order_probability: float = 0.15,
        cancel_probability: float = 0.25,
        max_open_orders: int = 20,
        wakeup_ns: int = 1_000_000,
    ) -> None:
        super().__init__(agent_id, rng)
        self.order_size_mean = order_size_mean
        self.depth_scale = depth_scale
        self.depth_exponent = depth_exponent
        self.market_order_probability = market_order_probability
        self.cancel_probability = cancel_probability
        self.max_open_orders = max(1, max_open_orders)
        self.wakeup_interval = Nanos(wakeup_ns)

    def _draw_size(self) -> Quantity:
        """Order sizes are geometric: many small, few large, integer-valued."""
        return Quantity(int(1 + self.rng.geometric(1.0 / max(self.order_size_mean, 1))))

    def _draw_depth(self) -> int:
        r"""Placement depth in ticks from the mid, Pareto-distributed.

        :math:`P(\text{depth} > d) \propto d^{-\alpha}` with
        :math:`\alpha \approx 1.5`, matching the empirical placement
        distribution (Zovko & Farmer 2002).
        """
        return int(np.ceil(self.depth_scale * (self.rng.pareto(self.depth_exponent) + 1.0)))

    def on_wakeup(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        actions: list[Action] = []

        # Bound the quote inventory. This is not cosmetic: placement happens on
        # ~78% of wakeups while random cancellation happens on ~25%, so without
        # a cap the resting order count grows without bound. Measured on a 5
        # second run, that produced 21,034 resting orders and 23,459 lots at a
        # single price -- a wall that no realistic order could move, which pinned
        # the price to a 7-tick range while the fundamental wandered 56 ticks.
        # Real participants maintain a bounded working set of quotes; so does this.
        over_limit = len(self.state.open_orders) >= self.max_open_orders
        if self.state.open_orders and (over_limit or self.rng.random() < self.cancel_probability):
            victim = int(self.rng.choice(sorted(self.state.open_orders)))
            actions.append(CancelOrder(OrderId(victim)))

        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        size = self._draw_size()

        if self.rng.random() < self.market_order_probability:
            actions.append(SubmitOrder(side, size, None, OrderType.MARKET, TimeInForce.IOC))
            return actions

        reference = book.mid
        if reference is None:
            return actions
        depth = self._draw_depth()
        price = Ticks(round(reference) - depth * side.sign)
        if price <= 0:
            return actions
        actions.append(SubmitOrder(side, size, price, OrderType.LIMIT, TimeInForce.GTC))
        return actions


# --------------------------------------------------------------------------- #
# Informed trader                                                             #
# --------------------------------------------------------------------------- #
class InformedTrader(Agent):
    r"""Trader who observes the shared fundamental value the market cannot see.

    Reads :class:`~quantos.sim.fundamental.FundamentalValue` -- owned by the
    world, not by this agent -- and trades whenever the quoted price diverges
    from its perceived value by more than an edge threshold. Trade size scales
    with the perceived mispricing, as in Kyle (1985), where the informed
    trader's optimal intensity is linear in the deviation.

    This agent is what makes the simulation non-trivial. Without adverse
    selection, market making is a risk-free harvest of the spread and every
    liquidity-provision result is meaningless. With it, the market maker faces
    the real trade-off: quote tight and get picked off, quote wide and never
    trade. The measurable consequence is a positive Kyle's lambda and a
    permanent component to price impact -- both of which
    :mod:`quantos.research.features.microstructure` recovers from the tape.

    Signal quality, and why the noise must persist
        ``signal_bias_scale`` gives each agent a *fixed* misperception of the
        shared value, drawn once at construction. It represents heterogeneous
        interpretation of the same news, and it is what makes a population of
        informed traders interesting: with identical perfect information they act
        in unison and price becomes a step function, while with dispersed views
        informed flow arrives gradually and price discovery is smooth and
        measurable.

        The bias is deliberately **persistent rather than redrawn each wakeup**.
        An earlier version resampled i.i.d. observation noise on every wakeup,
        and the consequence was severe: since the target position is proportional
        to perceived mispricing, a fresh draw each time made the target jitter by
        the noise scale times ``aggression``, so the agent traded on its own
        measurement error rather than on information. It churned 4,800 round
        trips, paid the spread on every one, ended flat, and lost money on a
        signal that was genuinely predictive. Informed traders that lose to noise
        traders are a sign the *simulator* is wrong, not the strategy.

        ``min_trade_size`` adds hysteresis for the same reason: without it, tiny
        target adjustments generate a constant stream of one-lot market orders.
    """

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        *,
        fundamental: FundamentalValue,
        signal_bias_scale: float = 1.0,
        edge_threshold_ticks: float = 2.0,
        aggression: float = 4.0,
        max_position: int = 5_000,
        min_trade_size: int = 5,
        wakeup_ns: int = 5_000_000,
    ) -> None:
        super().__init__(agent_id, rng)
        self.fundamental = fundamental
        self.edge_threshold_ticks = edge_threshold_ticks
        self.aggression = aggression
        self.max_position = max_position
        self.min_trade_size = max(1, min_trade_size)
        self.wakeup_interval = Nanos(wakeup_ns)
        #: Fixed, agent-specific misreading of the shared value. Drawn once.
        self.signal_bias = float(signal_bias_scale * rng.standard_normal())

    def perceived_value(self, timestamp: Nanos) -> float:
        """The shared fundamental as this agent reads it, plus its fixed bias."""
        return self.fundamental.value_at(timestamp) + self.signal_bias

    def target_position(self, value: float, mid: float) -> int:
        r"""Kyle's linear demand: :math:`q^* = \beta(V - P)`, clipped to the limit.

        Expressing the strategy as a *target position* rather than as a stream of
        one-directional trades is what makes informed flow self-limiting, and it
        matters more than it sounds. An agent that buys whenever price is below
        value accumulates without bound, hits its position limit, and then goes
        silent -- at which point nothing is pushing price toward value any more.
        An earlier version of this class did exactly that, and the consequence
        was a market whose price moved seven ticks while the fundamental moved
        forty-nine: informed traders saturated in the first second and price
        discovery simply stopped.

        With a target position, the agent buys as the gap opens and *sells back
        as it closes*, so the demand curve is downward-sloping in price. That is
        both what Kyle (1985) actually derives and the mechanism that makes the
        simulated market efficient in the sense of tracking its own latent value.
        """
        raw = self.aggression * (value - mid)
        return int(np.clip(raw, -self.max_position, self.max_position))

    def on_wakeup(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        if book.bid_price is None or book.ask_price is None:
            return []
        mid = book.mid
        if mid is None:
            return []

        value = self.perceived_value(timestamp)
        # Only act when the edge clears the threshold, so the agent is not
        # constantly paying the spread to make one-lot adjustments.
        if abs(value - mid) < self.edge_threshold_ticks:
            return []

        delta = self.target_position(value, mid) - self.state.position
        if abs(delta) < self.min_trade_size:
            return []

        side = Side.BUY if delta > 0 else Side.SELL
        return [SubmitOrder(side, Quantity(abs(delta)), None, OrderType.MARKET, TimeInForce.IOC)]


# --------------------------------------------------------------------------- #
# Market maker                                                                #
# --------------------------------------------------------------------------- #
class MarketMaker(Agent):
    r"""Avellaneda-Stoikov optimal market maker with inventory management.

    The model
        Maximise exponential utility of terminal wealth while quoting a
        two-sided market. Avellaneda-Stoikov derive a reservation price that
        skews away from inventory and an optimal total spread:

        .. math::
            r(s, q, t) = s - q\gamma\sigma^2(T-t)

        .. math::
            \delta^{a} + \delta^{b} = \gamma\sigma^2 (T-t)
                                      + \frac{2}{\gamma}\ln\!\Big(1+\frac{\gamma}{\kappa}\Big)

        where :math:`q` is inventory, :math:`\gamma` risk aversion,
        :math:`\sigma` volatility, and :math:`\kappa` the order-arrival decay
        rate.

    Why the inventory skew is the whole point
        A naive maker quotes symmetrically about the mid and accumulates
        inventory in whichever direction the market is trending -- precisely the
        wrong direction, since it is being adversely selected. The
        :math:`-q\gamma\sigma^2(T-t)` term shifts *both* quotes down when long,
        making the maker more eager to sell and less eager to buy. That single
        term converts a strategy that blows up on a trend into one that
        survives, and its magnitude is the price the maker charges for bearing
        inventory risk.

    Anchoring, and why the microprice alone is not enough
        Quoting around the **microprice** (see
        :attr:`quantos.core.types.TopOfBook.microprice`) beats quoting around the
        mid, because queue imbalance predicts the next mid move. But the
        microprice is derived from the book, and the book is made of the makers'
        own quotes -- so a population of makers anchored only on it forms a
        closed feedback loop with **no external reference**. Nothing pulls the
        price toward the latent fundamental value, and price discovery fails.

        Measured, before this was fixed: across 20 seeds the correlation between
        the market mid and the latent fundamental averaged near **zero**, with
        individual runs as low as -0.45. A single lucky seed gave 0.78, which is
        exactly the kind of number one should not quote.

    Learning from order flow (Glosten-Milgrom)
        The missing mechanism is *inference*. A maker repeatedly lifted on its
        ask should conclude that informed traders are buying and revise its fair
        value **up**. That is the whole content of Glosten-Milgrom (1985): the
        quote is a Bayesian posterior, and each trade is evidence.

        Implemented here as an exponential update on aggressor-signed fill flow:

        .. math:: \hat{V} \leftarrow \hat{V}
                  + \eta \cdot \text{sign(aggressor)} \cdot \sqrt{q}

        with :math:`\eta` = ``learning_rate``. The square root rather than the
        raw quantity is deliberate -- it matches the empirical square-root impact
        law (see :mod:`quantos.execution.almgren_chriss`) and stops one large
        print from dominating the estimate.

        Setting ``learning_rate=0`` recovers the pure feedback-loop behaviour,
        which is useful for demonstrating that this term is what makes the market
        efficient rather than a cosmetic addition.
    """

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        *,
        risk_aversion: float = 0.05,
        volatility: float = 1.0,
        order_arrival_decay: float = 1.5,
        quote_size: int = 20,
        max_position: int = 300,
        horizon_ns: int = 1_000_000_000,
        min_half_spread_ticks: float = 1.0,
        learning_rate: float = 0.15,
        initial_fair_value: float | None = None,
        wakeup_ns: int = 2_000_000,
    ) -> None:
        super().__init__(agent_id, rng)
        self.risk_aversion = risk_aversion
        self.volatility = volatility
        self.order_arrival_decay = order_arrival_decay
        self.quote_size = quote_size
        self.max_position = max_position
        self.horizon_ns = horizon_ns
        self.min_half_spread_ticks = min_half_spread_ticks
        self.learning_rate = learning_rate
        self.wakeup_interval = Nanos(wakeup_ns)
        self._quotes: dict[int, Side] = {}
        #: The maker's own estimate of fair value, updated from order flow.
        #: ``None`` until the first quote, then seeded from the observed book.
        self.fair_value: float | None = initial_fair_value
        #: Diagnostics for the research layer.
        self.reservation_prices: list[tuple[int, float]] = []

    def learn_from_fill(self, fill: Fill) -> None:
        r"""Revise fair value from one of this maker's own fills.

        A maker's fill tells it the *aggressor's* direction: if the maker sold,
        someone bought. ``fill.side`` is the maker's side, so the aggressor's
        sign is its negation.
        """
        if self.learning_rate <= 0.0 or self.fair_value is None:
            return
        aggressor_sign = -fill.side.sign
        self.fair_value += self.learning_rate * aggressor_sign * float(np.sqrt(int(fill.quantity)))

    def anchor_price(self, book: TopOfBook) -> float | None:
        r"""Blend the maker's learned fair value with the observed microprice.

        The microprice carries immediate, high-frequency information about which
        side of the book is about to be exhausted; the learned fair value carries
        the slow, accumulated inference from order flow. Using only the former
        loses the external anchor; using only the latter ignores the queue.
        """
        observed = book.microprice if book.microprice is not None else book.mid
        if observed is None:
            return None
        if self.fair_value is None:
            # Seed from the first book we see, so no external knowledge leaks in.
            self.fair_value = observed
        if self.learning_rate <= 0.0:
            return observed
        # Pull the estimate toward the observed book, but only very weakly. This
        # term runs on every wakeup (every 2 ms) while learning happens per fill,
        # so a coefficient large enough to feel reasonable in isolation erases the
        # accumulated inference entirely: at 0.05 the measured mean correlation
        # with the fundamental stayed at 0.09, indistinguishable from having no
        # learning at all. It exists only to stop unbounded drift away from a
        # market the maker must actually trade in.
        self.fair_value += 0.002 * (observed - self.fair_value)
        return 0.7 * self.fair_value + 0.3 * observed

    def reservation_price(self, anchor: float, time_remaining: float) -> float:
        r"""Inventory-adjusted fair value :math:`s - q\gamma\sigma^2(T-t)`."""
        return (
            anchor - self.state.position * self.risk_aversion * self.volatility**2 * time_remaining
        )

    def optimal_half_spread(self, time_remaining: float) -> float:
        r"""Half of the Avellaneda-Stoikov optimal total spread.

        The first term is compensation for inventory risk over the remaining
        horizon; the second is the monopolistic markup a maker can charge given
        how quickly order flow decays with distance from the mid
        (:math:`\kappa`). Floored at ``min_half_spread_ticks`` because a
        sub-tick spread is not quotable.
        """
        inventory_term = 0.5 * self.risk_aversion * self.volatility**2 * time_remaining
        markup = (1.0 / self.risk_aversion) * np.log1p(
            self.risk_aversion / self.order_arrival_decay
        )
        return float(max(inventory_term + markup, self.min_half_spread_ticks))

    def on_wakeup(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        anchor = self.anchor_price(book)
        if anchor is None:
            return []

        # Normalised time remaining in the quoting horizon, cycling so the maker
        # behaves stationarily rather than winding down to nothing.
        time_remaining = 1.0 - (int(timestamp) % self.horizon_ns) / self.horizon_ns

        actions: list[Action] = [CancelOrder(OrderId(oid)) for oid in sorted(self._quotes)]
        self._quotes.clear()

        reservation = self.reservation_price(anchor, time_remaining)
        half_spread = self.optimal_half_spread(time_remaining)
        self.reservation_prices.append((int(timestamp), reservation))

        bid = Ticks(int(np.floor(reservation - half_spread)))
        ask = Ticks(int(np.ceil(reservation + half_spread)))
        if int(ask) - int(bid) < 1:
            ask = Ticks(int(bid) + 1)

        # Suppress the side that would breach the position limit. Quoting a side
        # you cannot afford to be filled on is how makers discover their risk
        # limits the expensive way.
        if self.state.position < self.max_position and bid > 0:
            actions.append(
                SubmitOrder(
                    Side.BUY, Quantity(self.quote_size), bid, OrderType.POST_ONLY, TimeInForce.GTC
                )
            )
        if self.state.position > -self.max_position:
            actions.append(
                SubmitOrder(
                    Side.SELL, Quantity(self.quote_size), ask, OrderType.POST_ONLY, TimeInForce.GTC
                )
            )
        return actions

    def on_fill(self, fill: Fill, timestamp: Nanos) -> list[Action]:
        self._quotes.pop(int(fill.order_id), None)
        self.learn_from_fill(fill)
        return []

    def register_quote(self, order_id: OrderId, side: Side) -> None:
        """Called by the world once an order id has been assigned."""
        self._quotes[int(order_id)] = side


# --------------------------------------------------------------------------- #
# Momentum and mean reversion                                                 #
# --------------------------------------------------------------------------- #
class MomentumTrader(Agent):
    r"""Trend follower on an exponentially-weighted mid-price change.

    Buys when the fast EWMA of the mid exceeds the slow one. Its role in the
    ecosystem is destabilising by design: momentum demand is positively
    correlated with recent returns, which lengthens moves and -- combined with
    the market maker's inventory-driven withdrawal -- is the mechanism that
    produces **volatility clustering** in the simulated tape. Neither agent is
    told to cluster volatility; it falls out of the feedback loop.
    """

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        *,
        fast_halflife: float = 20.0,
        slow_halflife: float = 100.0,
        entry_threshold_ticks: float = 1.0,
        order_size: int = 15,
        max_position: int = 200,
        wakeup_ns: int = 10_000_000,
    ) -> None:
        super().__init__(agent_id, rng)
        if fast_halflife >= slow_halflife:
            raise ValueError("fast_halflife must be shorter than slow_halflife")
        self.fast_alpha = 1.0 - 2.0 ** (-1.0 / fast_halflife)
        self.slow_alpha = 1.0 - 2.0 ** (-1.0 / slow_halflife)
        self.entry_threshold_ticks = entry_threshold_ticks
        self.order_size = order_size
        self.max_position = max_position
        self.wakeup_interval = Nanos(wakeup_ns)
        self._fast: float | None = None
        self._slow: float | None = None

    def on_market_data(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        mid = book.mid
        if mid is None:
            return []
        if self._fast is None:
            self._fast = self._slow = mid
        else:
            self._fast += self.fast_alpha * (mid - self._fast)
            self._slow += self.slow_alpha * (mid - self._slow)
        return []

    def on_wakeup(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        if self._fast is None or self._slow is None:
            return []
        signal = self._fast - self._slow
        if abs(signal) < self.entry_threshold_ticks:
            return []
        side = Side.BUY if signal > 0 else Side.SELL
        if side is Side.BUY and self.state.position >= self.max_position:
            return []
        if side is Side.SELL and self.state.position <= -self.max_position:
            return []
        return [
            SubmitOrder(side, Quantity(self.order_size), None, OrderType.MARKET, TimeInForce.IOC)
        ]


class MeanReversionTrader(Agent):
    """Contrarian trading deviations from a rolling mean of the mid.

    The stabilising counterweight to :class:`MomentumTrader`. The *ratio* of
    these two populations is the most important single knob in the simulation:
    momentum-dominated markets produce trending, fat-tailed, unstable prices,
    reversion-dominated ones produce excessively well-behaved prices with
    negative return autocorrelation that no real market shows. Realistic
    stylised facts appear at a balance between them, which is itself a finding
    worth reproducing.
    """

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        *,
        lookback_halflife: float = 200.0,
        entry_threshold_sd: float = 2.0,
        order_size: int = 15,
        max_position: int = 200,
        wakeup_ns: int = 8_000_000,
    ) -> None:
        super().__init__(agent_id, rng)
        self.alpha = 1.0 - 2.0 ** (-1.0 / lookback_halflife)
        self.entry_threshold_sd = entry_threshold_sd
        self.order_size = order_size
        self.max_position = max_position
        self.wakeup_interval = Nanos(wakeup_ns)
        self._mean: float | None = None
        self._var: float = 1.0

    def on_market_data(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        mid = book.mid
        if mid is None:
            return []
        if self._mean is None:
            self._mean = mid
            return []
        deviation = mid - self._mean
        self._mean += self.alpha * deviation
        # EWMA variance, so the threshold adapts to the prevailing regime rather
        # than being a fixed number of ticks.
        self._var += self.alpha * (deviation * deviation - self._var)
        return []

    def on_wakeup(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        mid = book.mid
        if mid is None or self._mean is None:
            return []
        sd = float(np.sqrt(max(self._var, 1e-12)))
        z = (mid - self._mean) / sd
        if abs(z) < self.entry_threshold_sd:
            return []
        side = Side.SELL if z > 0 else Side.BUY
        if side is Side.BUY and self.state.position >= self.max_position:
            return []
        if side is Side.SELL and self.state.position <= -self.max_position:
            return []
        return [
            SubmitOrder(side, Quantity(self.order_size), None, OrderType.MARKET, TimeInForce.IOC)
        ]


# --------------------------------------------------------------------------- #
# Institutional execution                                                     #
# --------------------------------------------------------------------------- #
class ExecutionStyle(enum.Enum):
    """How an institutional parent order is sliced."""

    TWAP = "twap"
    #: Participate at a fixed fraction of observed volume.
    POV = "pov"
    #: Front-loaded, per the Almgren-Chriss risk-averse solution.
    FRONT_LOADED = "front_loaded"


class InstitutionalTrader(Agent):
    r"""Executes a large parent order over a horizon, generating price impact.

    Purpose in the ecosystem
        This is the agent that makes impact *measurable*. Noise traders' flow is
        sign-balanced, so it produces no persistent pressure; an institution
        working a 50,000-lot buy produces exactly the sustained one-sided flow
        that generates the square-root impact law. The simulation can therefore
        be used to *test* impact estimators against a known ground truth --
        we know the parent size, so we can check whether
        :func:`quantos.research.features.impact.fit_square_root_law` recovers it.

    Slicing
        ``TWAP`` slices uniformly in time. ``POV`` targets a fraction of
        realised volume. ``FRONT_LOADED`` follows the Almgren-Chriss
        risk-averse schedule, trading faster early to reduce exposure to price
        risk at the cost of higher impact -- see
        :mod:`quantos.execution.almgren_chriss` for the derivation.
    """

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        *,
        parent_side: Side,
        parent_quantity: int,
        horizon_ns: int,
        style: ExecutionStyle = ExecutionStyle.TWAP,
        n_slices: int = 50,
        participation_rate: float = 0.10,
        urgency: float = 2.0,
        start_ns: int = 0,
    ) -> None:
        super().__init__(agent_id, rng)
        if parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        self.parent_side = parent_side
        self.parent_quantity = parent_quantity
        self.horizon_ns = horizon_ns
        self.style = style
        self.n_slices = max(1, n_slices)
        self.participation_rate = participation_rate
        self.urgency = urgency
        self.start_ns = start_ns
        self.wakeup_interval = Nanos(max(1, horizon_ns // self.n_slices))
        self.executed = 0
        self._observed_volume = 0
        #: (timestamp, executed_so_far) for TCA in the research layer.
        self.execution_path: list[tuple[int, int]] = []

    @property
    def remaining(self) -> int:
        return max(0, self.parent_quantity - self.executed)

    def _target_executed(self, elapsed_fraction: float) -> int:
        r"""Cumulative target at a given fraction of the horizon.

        ``FRONT_LOADED`` uses the Almgren-Chriss trajectory
        :math:`x(t) = X\frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}`, which for
        urgency :math:`\kappa > 0` is convex-decreasing in remaining quantity --
        i.e. front-loaded.
        """
        f = min(max(elapsed_fraction, 0.0), 1.0)
        if self.style is ExecutionStyle.FRONT_LOADED:
            k = self.urgency
            remaining_fraction = np.sinh(k * (1.0 - f)) / np.sinh(k)
            return round(self.parent_quantity * (1.0 - remaining_fraction))
        return round(self.parent_quantity * f)

    def on_market_data(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        return []

    def observe_volume(self, quantity: int) -> None:
        """Called by the world with public tape volume, for POV slicing."""
        self._observed_volume += quantity

    def on_wakeup(self, book: TopOfBook, timestamp: Nanos) -> list[Action]:
        if self.remaining == 0 or int(timestamp) < self.start_ns:
            return []

        elapsed = (int(timestamp) - self.start_ns) / max(self.horizon_ns, 1)
        if self.style is ExecutionStyle.POV:
            target = round(self.participation_rate * self._observed_volume)
        else:
            target = self._target_executed(elapsed)

        slice_size = min(self.remaining, max(0, target - self.executed))
        if elapsed >= 1.0:
            slice_size = self.remaining  # sweep whatever is left at the horizon
        if slice_size <= 0:
            return []

        self.executed += slice_size
        self.execution_path.append((int(timestamp), self.executed))
        return [
            SubmitOrder(
                self.parent_side, Quantity(slice_size), None, OrderType.MARKET, TimeInForce.IOC
            )
        ]
