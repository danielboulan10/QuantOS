r"""The fundamental value process -- a property of the world, not of an agent.

Why this is its own object
--------------------------
An early version of this simulator gave each :class:`~quantos.sim.agents.
InformedTrader` its own private random walk. The result was a market whose price
barely moved: with three informed traders holding three *independent* views,
their order flow largely cancelled, net informed pressure was near zero, and the
market makers' quotes stayed anchored to their own microprice in a closed
feedback loop with no external reference. The price wandered eight ticks in sixty
seconds of market time while the "fundamental" wandered forty.

That was a modelling error, not a tuning problem. Informed traders are informed
about *the same thing*. In Kyle (1985) and Glosten-Milgrom (1985) there is one
liquidation value and the informed agents share it; their number affects the
*intensity* of informed flow, not its direction. So the value process belongs to
the world, and agents observe it.

What is imposed and what is emergent
------------------------------------
This distinction matters for reading :mod:`quantos.sim.stylized_facts` honestly.

**Imposed here.** The fundamental has i.i.d. Gaussian increments plus an optional
compound-Poisson jump component representing news arrival. The jumps put excess
kurtosis into the *value*, which then transmits to price. So if the simulated
market shows fat tails, some of that is inherited rather than emergent, and the
report should not be read as claiming otherwise.

**Not imposed, and therefore genuinely emergent.** The increments of this process
are independent and identically distributed. It has:

* no volatility clustering,
* no long memory,
* no leverage effect,
* no volume-volatility relationship.

If the simulated price series exhibits those, they were produced by the
interaction of order-book mechanics, inventory-averse liquidity provision and
trend-following -- because there is nowhere else for them to come from. Those are
the facts worth pointing at.

Set ``jump_intensity=0`` to make even the tails emergent, at the cost of a less
realistic news process.

References
----------
Kyle, A. S. (1985), *Econometrica* 53(6), 1315-1335.
Merton, R. C. (1976), "Option pricing when underlying stock returns are
    discontinuous", *J. Financial Economics* 3, 125-144.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantos.core.types import Nanos, Ticks

__all__ = ["FundamentalValue"]

_NANOS_PER_SECOND = 1_000_000_000.0


@dataclass
class FundamentalValue:
    r"""A shared jump-diffusion fundamental value, in ticks.

    .. math:: dV_t = \sigma\, dW_t + dJ_t

    where :math:`J` is a compound Poisson process with intensity ``jump_intensity``
    (per second) and Gaussian jump sizes of scale ``jump_scale``.

    Advancing is **lazy and monotone**: :meth:`value_at` brings the process
    forward to the requested timestamp and caches it, and refuses to go
    backwards. Combined with the deterministic event ordering of
    :class:`~quantos.sim.clock.SimulationClock`, that makes the value path a
    reproducible function of the seed -- every agent querying the same timestamp
    sees the same value, and querying twice does not advance it twice.

    Example
        >>> import numpy as np
        >>> fv = FundamentalValue(initial=10_000.0, volatility=2.0,
        ...                       rng=np.random.default_rng(0))
        >>> a = fv.value_at(Nanos(1_000_000_000))
        >>> a == fv.value_at(Nanos(1_000_000_000))    # idempotent
        True
        >>> b = fv.value_at(Nanos(2_000_000_000))
        >>> isinstance(b, float)
        True
    """

    initial: float
    #: Diffusion volatility, in ticks per second^(1/2).
    volatility: float = 2.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    #: News-arrival intensity, jumps per second. Zero makes tails fully emergent.
    jump_intensity: float = 0.5
    #: Standard deviation of a jump, in ticks.
    jump_scale: float = 8.0

    _value: float = field(init=False, default=0.0)
    _last_ns: int = field(init=False, default=0)
    #: (timestamp_ns, value) recorded whenever the process advances.
    path: list[tuple[int, float]] = field(init=False, default_factory=list)
    n_jumps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.volatility < 0 or self.jump_intensity < 0 or self.jump_scale < 0:
            raise ValueError("volatility, jump_intensity and jump_scale must be non-negative")
        self._value = float(self.initial)
        self.path.append((0, self._value))

    @property
    def value(self) -> float:
        """Current value without advancing the clock."""
        return self._value

    def value_at(self, timestamp: Nanos) -> float:
        """Advance to ``timestamp`` if needed, then return the value.

        Idempotent for repeated calls at the same timestamp, and a no-op for
        timestamps in the past -- so the order in which agents query it within
        one event cannot change the path.
        """
        now = int(timestamp)
        if now <= self._last_ns:
            return self._value

        elapsed_seconds = (now - self._last_ns) / _NANOS_PER_SECOND
        self._last_ns = now

        if self.volatility > 0.0:
            self._value += (
                self.volatility * np.sqrt(elapsed_seconds) * float(self.rng.standard_normal())
            )

        if self.jump_intensity > 0.0 and self.jump_scale > 0.0:
            n_jumps = int(self.rng.poisson(self.jump_intensity * elapsed_seconds))
            if n_jumps:
                self.n_jumps += n_jumps
                # Sum of n_jumps iid N(0, jump_scale^2) is N(0, n * scale^2).
                self._value += (
                    self.jump_scale * np.sqrt(n_jumps) * float(self.rng.standard_normal())
                )

        self._value = float(self._value)
        self.path.append((now, self._value))
        return self._value

    def as_ticks(self) -> Ticks:
        """The current value rounded to the nearest tick."""
        return Ticks(round(self._value))

    def path_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The recorded path as ``(timestamps_ns, values)``.

        Used by the research layer to measure how efficiently the simulated
        market discovers the value it was never told -- see
        :func:`quantos.research.features.microstructure.price_discovery_efficiency`.
        """
        if not self.path:
            return np.zeros(0), np.zeros(0)
        stamps, values = zip(*self.path, strict=False)
        return np.asarray(stamps, dtype=np.int64), np.asarray(values, dtype=float)
