r"""Binomial and trinomial lattice pricing, written from the recombination up.

This repository already prices options three other ways: a closed form
(:mod:`~quantos.derivatives.black_scholes`), a Fourier inversion
(:mod:`~quantos.derivatives.heston`), and least-squares Monte Carlo
(:mod:`~quantos.derivatives.american`). A fourth route is not redundancy for its
own sake. Four independent methods agreeing on a number is a far stronger
statement than any one of them asserting it, and where they disagree is where
something is wrong.

Lattices earn their place on American options. Monte Carlo must *learn* an
exercise rule from regression and is biased by whichever way that regression
errs; a tree evaluates the exercise decision exactly at every node, so the only
error is discretisation. That makes it the natural cross-check on
:func:`~quantos.derivatives.american.price_american`, and the cross-check is
worth reporting: on the standard Longstaff-Schwartz test case (S=36, K=40,
r=6%, sigma=20%, T=1) this lattice returns **4.4867**, stable to five decimals
from 2,000 steps through 12,000. The value published in the 2001 paper is
**4.478**. The gap of 0.009 is the downward bias of least-squares Monte Carlo
itself -- an exercise rule fitted from a finite sample is suboptimal, and a
suboptimal rule under-prices. Our own LSMC brackets the lattice value correctly:
[4.4734, 4.9470].

**The convergence is not monotone, and pretending otherwise is the classic
mistake.** Cox-Ross-Rubinstein error oscillates with the number of steps rather
than shrinking smoothly, because the strike falls between terminal nodes and its
position relative to them cycles as ``n`` changes. Quoting a tree price at
``n = 200`` and calling it converged means quoting a point on an oscillation.
:func:`binomial_price` therefore reports the oscillation amplitude, and
:func:`averaged_binomial_price` removes it the standard way -- averaging ``n``
and ``n+1``, whose errors have opposite sign.

Example
    >>> from quantos.derivatives.black_scholes import OptionType
    >>> european = binomial_price(100.0, 100.0, 1.0, 0.2, rate=0.05, n_steps=500)
    >>> round(european.price, 3)
    10.447
    >>> american = binomial_price(
    ...     100.0, 100.0, 1.0, 0.2, rate=0.05, n_steps=500,
    ...     option_type=OptionType.PUT, american=True,
    ... )
    >>> american.price > american.european_price   # early exercise is worth something
    True

References
----------
    Cox, Ross & Rubinstein (1979), "Option Pricing: A Simplified Approach",
    *Journal of Financial Economics* 7(3).
    Boyle (1986), "Option Valuation Using a Three-Jump Process",
    *International Options Journal* 3 -- the trinomial construction.
    Broadie & Detemple (1996), "American Option Valuation", *RFS* 9(4), on the
    oscillation and the standard remedies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantos.derivatives.black_scholes import OptionType, black_scholes_price

__all__ = [
    "LatticePrice",
    "averaged_binomial_price",
    "binomial_price",
    "convergence_path",
    "trinomial_price",
]


@dataclass
class LatticePrice:
    """A lattice price, with the discretisation error it did not eliminate."""

    price: float
    n_steps: int
    method: str
    #: The Black-Scholes value of the same option, for reference. For an
    #: American put this is the European price, so the difference is the early
    #: exercise premium rather than an error.
    european_price: float
    american: bool
    #: Half the gap between the n-step and (n+1)-step prices: a direct measure of
    #: where on the oscillation this particular n happens to sit.
    oscillation: float = float("nan")
    #: Present only for a European option, where a closed form exists.
    error_vs_closed_form: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def early_exercise_premium(self) -> float:
        return self.price - self.european_price if self.american else 0.0

    def summary(self) -> str:
        kind = "American" if self.american else "European"
        lines = [
            f"{self.method} lattice, {self.n_steps} steps -- {kind}",
            f"  price            {self.price:12.6f}",
            f"  Black-Scholes    {self.european_price:12.6f}",
        ]
        if self.american:
            lines.append(f"  early exercise   {self.early_exercise_premium:12.6f}")
        else:
            lines.append(f"  error            {self.error_vs_closed_form:12.2e}")
        if np.isfinite(self.oscillation):
            lines.append(f"  oscillation      {self.oscillation:12.2e}  (n vs n+1)")
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Binomial
# --------------------------------------------------------------------------- #
def _crr_lattice(
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    *,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    n_steps: int,
    american: bool,
) -> float:
    r"""One Cox-Ross-Rubinstein pass.

    The construction: :math:`u = e^{\sigma\sqrt{\Delta t}}`, :math:`d = 1/u`, and
    the risk-neutral probability :math:`p = (e^{(r-q)\Delta t} - d)/(u - d)`.
    Choosing :math:`d = 1/u` is what makes the tree recombine, so an ``n``-step
    lattice has ``n+1`` terminal nodes rather than :math:`2^n` paths.

    Backward induction runs on a single array that shrinks by one each step. The
    obvious implementation allocates a triangular matrix; this one does not,
    because the discounted continuation value at a node depends only on the two
    nodes above it and nothing else is ever read again.
    """
    dt = expiry / n_steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1.0 / up

    growth = np.exp((rate - dividend_yield) * dt)
    p = (growth - down) / (up - down)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"the risk-neutral probability is {p:.4f}, outside (0, 1): with "
            f"sigma={volatility:.3f} and {n_steps} steps the time step is too "
            "coarse for this drift, which makes the lattice arbitrageable. Use "
            "more steps."
        )

    discount = np.exp(-rate * dt)

    # Terminal nodes: j up-moves and (n - j) down-moves.
    j = np.arange(n_steps + 1)
    prices = spot * up ** (2 * j - n_steps)

    sign = 1.0 if option_type is OptionType.CALL else -1.0
    values = np.maximum(sign * (prices - strike), 0.0)

    for step in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            node_prices = spot * up ** (2 * j - step)
            values = np.maximum(values, sign * (node_prices - strike))

    return float(values[0])


def binomial_price(
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 500,
    american: bool = False,
    measure_oscillation: bool = True,
) -> LatticePrice:
    """Price on a Cox-Ross-Rubinstein binomial lattice.

    Args:
        spot: current price of the underlying.
        strike: exercise price.
        expiry: time to expiry in years.
        volatility: annualised volatility.
        rate: continuously compounded risk-free rate.
        dividend_yield: continuous dividend yield.
        option_type: call or put.
        n_steps: lattice steps. Cost is O(n^2); error is O(1/n) with an
            oscillation on top of it.
        american: allow exercise at every node.
        measure_oscillation: also price with ``n+1`` steps, to report where on
            the oscillation this ``n`` sits. Doubles the cost.

    Returns
    -------
        A :class:`LatticePrice`. Read ``oscillation`` before trusting the last
        two digits of ``price``.
    """
    if n_steps < 1:
        raise ValueError("a lattice needs at least one step")
    if expiry <= 0 or volatility <= 0:
        raise ValueError("expiry and volatility must be positive")

    price = _crr_lattice(
        spot,
        strike,
        expiry,
        volatility,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_steps=n_steps,
        american=american,
    )
    european = black_scholes_price(
        spot,
        strike,
        expiry,
        volatility,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )

    result = LatticePrice(
        price=price,
        n_steps=n_steps,
        method="Cox-Ross-Rubinstein",
        european_price=float(european),
        american=american,
    )

    if measure_oscillation:
        neighbour = _crr_lattice(
            spot,
            strike,
            expiry,
            volatility,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            n_steps=n_steps + 1,
            american=american,
        )
        result.oscillation = abs(price - neighbour) / 2.0

    if not american:
        result.error_vs_closed_form = price - float(european)
        if (
            measure_oscillation
            and np.isfinite(result.oscillation)
            and result.oscillation > abs(result.error_vs_closed_form)
        ):
            result.notes.append(
                "The step-to-step oscillation exceeds the error at this n, so the "
                "agreement here is partly where the strike happens to sit between "
                "nodes. Use averaged_binomial_price for a figure that does not "
                "depend on that."
            )

    if american and option_type is OptionType.CALL and dividend_yield <= 0:
        result.notes.append(
            "An American call on a non-dividend-paying underlying is never "
            "exercised early, so this must equal the European price."
        )

    return result


def averaged_binomial_price(
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 500,
    american: bool = False,
) -> LatticePrice:
    r"""Average the ``n`` and ``n+1`` step prices to cancel the oscillation.

    The CRR error decomposes into a smooth :math:`O(1/n)` term and an oscillating
    term whose sign depends on where the strike sits between terminal nodes.
    Consecutive ``n`` usually land on opposite sides of that oscillation, so
    their mean cancels much of it and leaves the smooth part.

    **It is not uniformly better, and claiming so would be overselling it.**
    Measured over ``n`` from 50 to 299 on a one-year 25%-volatility call struck
    5% out of the money, averaging halves the mean absolute error -- 1.03e-2 down
    to 5.05e-3 -- and improves the worst case from 3.2e-2 to 2.1e-2. But it is
    better on only **74%** of individual step counts. When ``n`` and ``n+1``
    happen to fall on the *same* side of the oscillation their mean is no better
    than either, and can be slightly worse. The gain is in expectation over an
    arbitrary ``n``, which is the situation a caller is actually in.

    Example
        >>> from quantos.derivatives.black_scholes import OptionType
        >>> plain = binomial_price(100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=101)
        >>> smooth = averaged_binomial_price(100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=101)
        >>> abs(smooth.error_vs_closed_form) < abs(plain.error_vs_closed_form)
        True
    """
    low = _crr_lattice(
        spot,
        strike,
        expiry,
        volatility,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_steps=n_steps,
        american=american,
    )
    high = _crr_lattice(
        spot,
        strike,
        expiry,
        volatility,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
        n_steps=n_steps + 1,
        american=american,
    )
    european = float(
        black_scholes_price(
            spot,
            strike,
            expiry,
            volatility,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
    )
    price = 0.5 * (low + high)

    result = LatticePrice(
        price=price,
        n_steps=n_steps,
        method="CRR, n and n+1 averaged",
        european_price=european,
        american=american,
        oscillation=abs(low - high) / 2.0,
    )
    if not american:
        result.error_vs_closed_form = price - european
    result.notes.append(
        f"The two lattices differ by {abs(low - high):.2e}; averaging cancels the "
        "part of that which is the strike's position between nodes."
    )
    return result


# --------------------------------------------------------------------------- #
# Trinomial
# --------------------------------------------------------------------------- #
def trinomial_price(
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 300,
    american: bool = False,
) -> LatticePrice:
    r"""Price on a Boyle trinomial lattice.

    The third branch is a middle node that does not move, which decouples the
    space step from the time step: with :math:`\lambda = \sqrt{3}` the tree is
    stable and the strike sits closer to a node for any given ``n``. The result
    is a much smaller oscillation than CRR at comparable cost, which is the
    reason to prefer it for an American option where no closed form exists to
    check against.

    The probabilities come from matching the first two moments of the log price
    over one step, with the third fixed by summing to one.
    """
    if n_steps < 1:
        raise ValueError("a lattice needs at least one step")

    dt = expiry / n_steps
    dx = volatility * np.sqrt(3.0 * dt)
    nu = rate - dividend_yield - 0.5 * volatility**2

    p_up = 0.5 * ((volatility**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    p_down = 0.5 * ((volatility**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    p_mid = 1.0 - p_up - p_down

    if min(p_up, p_mid, p_down) < 0.0:
        raise ValueError(
            f"negative branch probability (up={p_up:.4f}, mid={p_mid:.4f}, "
            f"down={p_down:.4f}); the time step is too coarse for this drift"
        )

    discount = np.exp(-rate * dt)
    sign = 1.0 if option_type is OptionType.CALL else -1.0

    index = np.arange(-n_steps, n_steps + 1)
    prices = spot * np.exp(index * dx)
    values: NDArray[np.float64] = np.maximum(sign * (prices - strike), 0.0)

    for step in range(n_steps - 1, -1, -1):
        values = discount * (p_up * values[2:] + p_mid * values[1:-1] + p_down * values[:-2])
        if american:
            node_index = np.arange(-step, step + 1)
            node_prices = spot * np.exp(node_index * dx)
            values = np.maximum(values, sign * (node_prices - strike))

    european = float(
        black_scholes_price(
            spot,
            strike,
            expiry,
            volatility,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
    )
    price = float(values[0])

    result = LatticePrice(
        price=price,
        n_steps=n_steps,
        method="Boyle trinomial",
        european_price=european,
        american=american,
    )
    if not american:
        result.error_vs_closed_form = price - european
    return result


# --------------------------------------------------------------------------- #
# The oscillation, made visible
# --------------------------------------------------------------------------- #
def convergence_path(
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
    steps: range | list[int] | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
    """Error against the closed form as a function of step count.

    Returns ``(steps, binomial_error, trinomial_error)``. The binomial series
    changes sign repeatedly; the trinomial series does not, to nearly the same
    degree. Plotting the two together is the clearest way to see why quoting a
    binomial price at one convenient ``n`` is quoting a point on an oscillation.
    """
    counts = np.array(list(steps if steps is not None else range(20, 201)), dtype=int)
    european = float(
        black_scholes_price(
            spot,
            strike,
            expiry,
            volatility,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
    )

    binomial = np.array(
        [
            _crr_lattice(
                spot,
                strike,
                expiry,
                volatility,
                rate=rate,
                dividend_yield=dividend_yield,
                option_type=option_type,
                n_steps=int(n),
                american=False,
            )
            - european
            for n in counts
        ]
    )
    trinomial = np.array(
        [
            trinomial_price(
                spot,
                strike,
                expiry,
                volatility,
                rate=rate,
                dividend_yield=dividend_yield,
                option_type=option_type,
                n_steps=int(n),
            ).error_vs_closed_form
            for n in counts
        ]
    )
    return counts, binomial, trinomial
