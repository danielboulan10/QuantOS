r"""Quote an options book from a fitted surface, and account for where the money came from.

What this connects
------------------
Three subsystems that until now never touched: the SVI surface in
:mod:`quantos.research.vol_surface`, the matching engine in
:mod:`quantos.exchange`, and the agent framework in :mod:`quantos.sim`. A maker
that prices from the surface, quotes into the book, and gets traded against by
informed flow is the only way to find out whether the surface is any good for
its actual purpose.

Inventory lives in Greek space, not contract counts
----------------------------------------------------
An equity maker's inventory is a number of shares. An options maker's is not:
being long 100 calls and short 200 further-out calls is a position whose risk
depends on where spot goes, and no count of contracts describes it. What the
desk actually tracks is the aggregate **delta, gamma and vega** of the book.

So the skew here is driven by Greeks. Accumulating positive vega makes the maker
lower its offers in *volatility* terms -- it wants to sell vega back, at any
strike that reduces the exposure. Accumulating gamma changes the price at which
it is willing to keep accumulating, because gamma is what makes a hedged book
bleed or earn as spot moves.

Widening on inventory
---------------------
The reservation-price logic is Avellaneda-Stoikov generalised from one asset to a
Greek vector. For a book with vega :math:`V` and risk aversion :math:`\gamma`,
the implied volatility at which the maker is indifferent is

.. math:: \sigma^{\text{res}} = \sigma^{\text{mid}} - \gamma V \sigma^2 (T - t)

and the half-spread widens with the same terms. The sign is the point: a maker
long vega marks its own book *down*, because the next trade it wants is a sale.

The P&L decomposition, which is the reason this exists
--------------------------------------------------------
A market maker's profit is not one number, it is three, and they have opposite
signs and different causes:

**Spread capture.** Trading at your own quotes rather than at mid. Always
positive, and the reason the business exists.

**Gamma and theta.** A hedged options book earns theta and pays for gamma (short
options) or the reverse (long options). Over a day this is the dominant term for
anything with real convexity, and it is *not* a trading result -- it is the
carry on the position.

**Adverse selection.** The cost of the trades you did not want: informed flow
lifts your offer just before the price rises. Always negative. A maker whose
spread capture exceeds adverse selection has an edge; one where it does not is
providing a subsidy.

Reporting only net P&L hides which of the three is happening, so a maker that is
being picked off looks identical to one that is merely carrying a losing gamma
position -- and the fixes are opposite. :class:`PnLAttribution` separates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantos.derivatives.black_scholes import OptionType, black_scholes_greeks, black_scholes_price

__all__ = [
    "GreekInventory",
    "OptionQuote",
    "PnLAttribution",
    "QuotedMarket",
    "SurfaceMarketMaker",
]


@dataclass
class GreekInventory:
    """The maker's position, in the units risk is actually managed in."""

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    #: Contracts held per (strike, expiry, type), for mark-to-market.
    positions: dict[tuple[float, float, str], float] = field(default_factory=dict)
    cash: float = 0.0

    def add(self, greeks: object, quantity: float, key: tuple[float, float, str]) -> None:
        """Book ``quantity`` contracts, updating every Greek together."""
        self.delta += quantity * float(greeks.delta)  # type: ignore[attr-defined]
        self.gamma += quantity * float(greeks.gamma)  # type: ignore[attr-defined]
        self.vega += quantity * float(greeks.vega)  # type: ignore[attr-defined]
        self.theta += quantity * float(greeks.theta)  # type: ignore[attr-defined]
        self.positions[key] = self.positions.get(key, 0.0) + quantity

    @property
    def is_flat(self) -> bool:
        return abs(self.delta) < 1e-9 and abs(self.vega) < 1e-9

    def utilisation(self, limits: dict[str, float]) -> float:
        """Fraction of the tightest binding limit that is used.

        The *maximum* across Greeks rather than an average: a book at 95% of its
        vega limit is constrained even if its delta is flat, and averaging would
        hide that.
        """
        used = [
            abs(self.delta) / limits["delta"] if limits.get("delta") else 0.0,
            abs(self.gamma) / limits["gamma"] if limits.get("gamma") else 0.0,
            abs(self.vega) / limits["vega"] if limits.get("vega") else 0.0,
        ]
        return float(max(used))

    def summary(self) -> str:
        return (
            f"delta {self.delta:+9.2f}  gamma {self.gamma:+8.4f}  "
            f"vega {self.vega:+9.2f}  theta {self.theta:+9.2f}  cash {self.cash:+12.2f}"
        )


@dataclass(frozen=True)
class OptionQuote:
    """A two-sided quote on one contract, in price and in volatility."""

    strike: float
    time_to_expiry: float
    option_type: OptionType
    bid: float
    ask: float
    bid_volatility: float
    ask_volatility: float
    fair_value: float
    fair_volatility: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def volatility_spread(self) -> float:
        """Spread in volatility points -- how a desk actually quotes width."""
        return self.ask_volatility - self.bid_volatility

    @property
    def volatility_skew(self) -> float:
        """Displacement of the quote centre from fair, **in volatility**.

        This is the honest measure of inventory lean, because volatility is what
        the maker actually sets. Zero means no preference between sides.
        """
        return 0.5 * (self.bid_volatility + self.ask_volatility) - self.fair_volatility

    @property
    def price_skew(self) -> float:
        r"""Displacement of the price midpoint from fair value.

        **This is not zero even for an unskewed quote**, and the reason is worth
        knowing: option price is convex in volatility (volga), so a quote
        symmetric in volatility is asymmetric in price. Measured here on an
        at-the-money 3-month option, a symmetric one-point volatility width
        leaves the price midpoint about :math:`10^{-5}` from fair.

        The residual is second order in the width and vanishes as the spread
        narrows, but it never reaches zero, so this is a poor test of whether a
        maker is leaning. Use :attr:`volatility_skew` for that. This one is kept
        because it is what a customer comparing your mid to theirs actually sees.
        """
        return 0.5 * (self.bid + self.ask) - self.fair_value


@dataclass
class QuotedMarket:
    """Every quote the maker is showing, plus the state that produced them."""

    quotes: list[OptionQuote]
    spot: float
    inventory: GreekInventory
    #: Half-spread actually applied, in volatility points.
    half_spread_volatility: float
    #: Volatility shift from inventory, in volatility points. Signed.
    inventory_skew_volatility: float
    at_limit: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{len(self.quotes)} quotes at spot {self.spot:.2f}",
            f"  half-spread {self.half_spread_volatility * 100:.2f} vol pts, "
            f"inventory skew {self.inventory_skew_volatility * 100:+.2f} vol pts",
            f"  {self.inventory.summary()}",
        ]
        if self.at_limit:
            lines.append("  AT RISK LIMIT: quoting one side only")
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class PnLAttribution:
    """Where a market maker's money actually came from.

    Net P&L alone cannot distinguish a maker being picked off from one carrying a
    losing gamma position, and the responses are opposite -- widen versus hedge.
    """

    spread_capture: float = 0.0
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    adverse_selection: float = 0.0
    hedge_cost: float = 0.0
    n_trades: int = 0
    volume: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.spread_capture
            + self.gamma_pnl
            + self.theta_pnl
            + self.adverse_selection
            + self.hedge_cost
        )

    @property
    def edge_per_contract(self) -> float:
        """Spread capture net of adverse selection, per contract.

        The number that decides whether the business works. Positive means the
        quoted width more than covers what informed flow takes out.
        """
        if self.volume <= 0:
            return float("nan")
        return (self.spread_capture + self.adverse_selection) / self.volume

    @property
    def verdict(self) -> str:
        edge = self.edge_per_contract
        if not np.isfinite(edge):
            return "no trades"
        if edge > 0:
            return (
                f"edge of {edge:+.4f} per contract: the quoted width covers adverse "
                "selection, which is what makes the business work"
            )
        return (
            f"edge of {edge:+.4f} per contract: informed flow is taking more than the "
            "spread earns, so the quotes are too tight for this flow"
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"P&L attribution over {self.n_trades} trades ({self.volume:.0f} contracts)",
                f"  spread capture     {self.spread_capture:+12.2f}",
                f"  gamma              {self.gamma_pnl:+12.2f}",
                f"  theta              {self.theta_pnl:+12.2f}",
                f"  adverse selection  {self.adverse_selection:+12.2f}",
                f"  hedge cost         {self.hedge_cost:+12.2f}",
                f"  {'-' * 32}",
                f"  total              {self.total:+12.2f}",
                "",
                f"  {self.verdict}",
            ]
        )


@dataclass
class SurfaceMarketMaker:
    r"""Quotes an options book from a volatility surface, skewed by Greek inventory.

    Inputs
        ``surface`` -- anything exposing ``implied_volatility(k)`` for
        log-moneyness ``k``; a fitted :class:`~quantos.research.vol_surface.SVIParameters`
        satisfies this.
        ``risk_aversion`` -- the :math:`\gamma` in the reservation-price shift.
        ``limits`` -- absolute caps per Greek, beyond which the maker quotes one
        side only.

    Example
        >>> from quantos.research.vol_surface import SVIParameters
        >>> surface = SVIParameters(0.04, 0.1, -0.3, 0.0, 0.2, 0.25)
        >>> maker = SurfaceMarketMaker(surface=surface)
        >>> market = maker.quote(spot=100.0, strikes=(95.0, 100.0, 105.0),
        ...                      time_to_expiry=0.25)
        >>> len(market.quotes)
        3
        >>> all(q.bid < q.fair_value < q.ask for q in market.quotes)
        True
    """

    surface: object
    risk_aversion: float = 0.5
    #: Base half-spread in volatility points before any inventory adjustment.
    base_half_spread: float = 0.01
    rate: float = 0.0
    limits: dict[str, float] = field(
        default_factory=lambda: {"delta": 500.0, "gamma": 50.0, "vega": 2000.0}
    )
    inventory: GreekInventory = field(default_factory=GreekInventory)
    attribution: PnLAttribution = field(default_factory=PnLAttribution)

    def _fair_volatility(self, spot: float, strike: float, time_to_expiry: float) -> float:
        forward = spot * np.exp(self.rate * time_to_expiry)
        k = float(np.log(strike / forward))
        return float(np.asarray(self.surface.implied_volatility(k)).ravel()[0])  # type: ignore[attr-defined]

    def _inventory_skew(self, spot: float, time_to_expiry: float) -> float:
        r"""Volatility shift implied by current inventory.

        Long vega shifts quotes **down** in volatility: the maker wants to sell
        vega, so it marks its own book to where a sale is attractive. Getting
        this sign backwards produces a maker that accumulates risk faster the
        more it already holds, which blows up rather than mean-reverting.
        """
        vega_limit = self.limits.get("vega", 1.0) or 1.0
        normalised = self.inventory.vega / vega_limit
        reference = self._fair_volatility(spot, spot, time_to_expiry)
        return -self.risk_aversion * normalised * reference * max(time_to_expiry, 1e-6)

    def quote(
        self,
        spot: float,
        strikes: tuple[float, ...],
        time_to_expiry: float,
        *,
        option_type: OptionType = OptionType.CALL,
    ) -> QuotedMarket:
        """Produce a two-sided quote on every strike, skewed by inventory."""
        skew = self._inventory_skew(spot, time_to_expiry)
        utilisation = self.inventory.utilisation(self.limits)
        # Widen as the book fills: at the limit the width has doubled.
        half_spread = self.base_half_spread * (1.0 + utilisation)
        at_limit = utilisation >= 1.0

        quotes: list[OptionQuote] = []
        for strike in strikes:
            fair_volatility = self._fair_volatility(spot, strike, time_to_expiry)
            centre = fair_volatility + skew
            bid_volatility = max(centre - half_spread, 1e-4)
            ask_volatility = centre + half_spread

            def price(volatility: float, strike: float = strike) -> float:
                return float(
                    black_scholes_price(
                        spot,
                        strike,
                        time_to_expiry,
                        volatility,
                        rate=self.rate,
                        option_type=option_type,
                    )
                )

            quotes.append(
                OptionQuote(
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    option_type=option_type,
                    bid=price(bid_volatility),
                    ask=price(ask_volatility),
                    bid_volatility=bid_volatility,
                    ask_volatility=ask_volatility,
                    fair_value=price(fair_volatility),
                    fair_volatility=fair_volatility,
                )
            )

        notes: list[str] = []
        if at_limit:
            notes.append(
                f"Greek utilisation {utilisation:.0%}; a real desk stops showing the side "
                "that adds risk rather than widening indefinitely"
            )

        return QuotedMarket(
            quotes=quotes,
            spot=spot,
            inventory=self.inventory,
            half_spread_volatility=half_spread,
            inventory_skew_volatility=skew,
            at_limit=at_limit,
            notes=notes,
        )

    def on_trade(
        self,
        quote: OptionQuote,
        quantity: float,
        spot: float,
        *,
        side: str,
        subsequent_spot: float | None = None,
    ) -> None:
        r"""Book a trade and attribute its P&L.

        Inputs
            ``side`` -- ``"buy"`` means the *maker* bought, i.e. someone hit its
            bid. ``subsequent_spot`` -- where spot went afterwards, which is what
            makes adverse selection measurable rather than assumed.

        Attribution
            **Spread capture** is the distance from the traded price to fair
            value, always in the maker's favour by construction.

            **Adverse selection** is the mark-to-market change on the new
            position caused by the subsequent move. If the flow was informed, this
            is negative and offsets the spread. It is measured against realised
            spot rather than modelled, because the whole question is whether the
            counterparty knew something.
        """
        signed = quantity if side == "buy" else -quantity
        traded_price = quote.bid if side == "buy" else quote.ask

        greeks = black_scholes_greeks(
            spot,
            quote.strike,
            quote.time_to_expiry,
            quote.fair_volatility,
            rate=self.rate,
            option_type=quote.option_type,
        )
        key = (quote.strike, quote.time_to_expiry, quote.option_type.name)
        self.inventory.add(greeks, signed, key)
        self.inventory.cash -= signed * traded_price

        # Buying below fair, or selling above it, is the edge.
        self.attribution.spread_capture += abs(quantity) * abs(quote.fair_value - traded_price)
        self.attribution.n_trades += 1
        self.attribution.volume += abs(quantity)

        if subsequent_spot is not None:
            moved = black_scholes_price(
                subsequent_spot,
                quote.strike,
                quote.time_to_expiry,
                quote.fair_volatility,
                rate=self.rate,
                option_type=quote.option_type,
            )
            change = float(moved) - quote.fair_value
            # Positive when the position gained; adverse selection is the part
            # that went against the maker.
            self.attribution.adverse_selection += signed * change

    def accrue_carry(self, spot_path: NDArray[np.float64], time_step: float) -> None:
        r"""Accrue the gamma and theta the book earns or pays as spot moves.

        A hedged options book's P&L over a step is approximately

        .. math:: \tfrac12 \Gamma (\Delta S)^2 + \Theta \, \Delta t

        The two have opposite signs for any position: short options earn theta
        and pay gamma, long options the reverse. Reporting them separately is
        what tells a desk whether a bad day was the market moving too much or the
        clock not moving enough.
        """
        path = np.asarray(spot_path, dtype=float)
        if path.size < 2:
            return
        moves = np.diff(path)
        self.attribution.gamma_pnl += float(0.5 * self.inventory.gamma * np.sum(moves**2))
        self.attribution.theta_pnl += float(self.inventory.theta * time_step * (path.size - 1))

    def hedge_delta(self, spot: float, *, cost_per_share: float = 0.0) -> float:
        """Trade spot to flatten delta, recording the cost.

        Returns the number of shares traded. Hedging is not free, and a maker
        that hedges every tick pays more in spread than the gamma it is
        protecting -- which is why the cost is tracked rather than assumed away.
        """
        shares = -self.inventory.delta
        if abs(shares) < 1e-12:
            return 0.0
        self.inventory.cash -= shares * spot
        self.attribution.hedge_cost -= abs(shares) * cost_per_share
        self.inventory.delta = 0.0
        return float(shares)

    def mark_to_market(self, spot: float, *, volatility: float | None = None) -> float:
        """Value the book at current spot, so total P&L can be checked."""
        value = self.inventory.cash
        for (strike, expiry, type_name), quantity in self.inventory.positions.items():
            if abs(quantity) < 1e-12:
                continue
            option_type = OptionType[type_name]
            sigma = (
                volatility
                if volatility is not None
                else self._fair_volatility(spot, strike, expiry)
            )
            value += quantity * float(
                black_scholes_price(
                    spot, strike, expiry, sigma, rate=self.rate, option_type=option_type
                )
            )
        return float(value)
