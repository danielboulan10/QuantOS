"""Discrete-event simulation clock.

Why event-driven and not fixed-step
-----------------------------------
A fixed-timestep loop over a market forces a choice you cannot win: make the
step small and you burn almost all your compute on steps where nothing happens;
make it large and you serialise events that were genuinely simultaneous,
destroying the microstructure you are trying to study. Real markets are event
streams, and latency arbitrage -- the thing a market simulator exists to
study -- lives entirely in the ordering of events microseconds apart.

So time advances to the next scheduled event, always.

Deterministic tie-breaking
--------------------------
Two events can be scheduled for the identical nanosecond. Which fires first must
be decided *reproducibly*, or the simulation is not replayable. The heap is
therefore keyed on the triple

    ``(timestamp, priority, sequence)``

with ``sequence`` a monotonically increasing insertion counter. That makes the
order total: no two heap entries ever compare equal, so :mod:`heapq` never has
to fall back to comparing the payloads (which would raise, or worse, compare by
object identity and vary between runs).

``priority`` exists for the cases where simultaneity has real semantics -- market
data must be delivered before the agent actions that respond to it, and a
cancel submitted at the same nanosecond as a fill should resolve the same way
every time.
"""

from __future__ import annotations

import enum
import heapq
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from quantos.core.types import Nanos

__all__ = ["ClockError", "Event", "EventPriority", "SimulationClock"]


class ClockError(RuntimeError):
    """Raised on an attempt to schedule an event in the past."""


class EventPriority(enum.IntEnum):
    """Tie-breaking priority for events at the same timestamp. Lower fires first.

    The ordering encodes the causal structure of one market instant: data is
    published, then agents react, then their orders reach the venue, then the
    venue's own housekeeping runs. Without this, an agent could observe a quote
    and act on it within the same nanosecond, which is the simulator equivalent
    of look-ahead bias.
    """

    MARKET_DATA = 0
    AGENT_WAKEUP = 10
    ORDER_ARRIVAL = 20
    ORDER_CANCEL = 25
    FILL_NOTIFICATION = 30
    VENUE_MAINTENANCE = 40
    MEASUREMENT = 50


@dataclass(order=True, slots=True)
class Event:
    """A scheduled callback.

    Ordering is by ``(timestamp, priority, sequence)`` only -- ``compare=False``
    on the payload fields keeps them out of the comparison, so the total order
    is guaranteed and the payload never needs to be comparable.
    """

    timestamp: Nanos
    priority: int
    sequence: int
    callback: Callable[[Nanos], Any] = field(compare=False, default=lambda _: None)
    label: str = field(compare=False, default="")
    payload: Any = field(compare=False, default=None)


class SimulationClock:
    """Priority-queue event loop with a monotonically advancing clock.

    Example
        >>> clock = SimulationClock()
        >>> log = []
        >>> _ = clock.schedule_at(Nanos(100), lambda t: log.append(("b", int(t))))
        >>> _ = clock.schedule_at(Nanos(50), lambda t: log.append(("a", int(t))))
        >>> clock.run()
        2
        >>> log
        [('a', 50), ('b', 100)]
        >>> int(clock.now)
        100
    """

    __slots__ = ("_heap", "_now", "_processed", "_sequence", "_stop")

    def __init__(self, start: Nanos = Nanos(0)) -> None:
        self._heap: list[Event] = []
        self._sequence = 0
        self._now = start
        self._processed = 0
        self._stop = False

    @property
    def now(self) -> Nanos:
        """Current simulation time. Never decreases."""
        return self._now

    @property
    def pending(self) -> int:
        return len(self._heap)

    @property
    def processed(self) -> int:
        return self._processed

    def schedule_at(
        self,
        timestamp: Nanos,
        callback: Callable[[Nanos], Any],
        *,
        priority: int = EventPriority.AGENT_WAKEUP,
        label: str = "",
        payload: Any = None,
    ) -> Event:
        """Schedule ``callback`` for an absolute time.

        Raises
        ------
            :class:`ClockError` if ``timestamp`` is before :attr:`now`.
            Scheduling into the past is always a bug -- usually a latency model
            subtracting instead of adding -- and silently clamping it to the
            present would hide a causality violation.
        """
        if int(timestamp) < int(self._now):
            raise ClockError(
                f"cannot schedule at {timestamp} ns; the clock is already at "
                f"{self._now} ns. Scheduling into the past breaks causality."
            )
        self._sequence += 1
        event = Event(
            timestamp=timestamp,
            priority=int(priority),
            sequence=self._sequence,
            callback=callback,
            label=label,
            payload=payload,
        )
        heapq.heappush(self._heap, event)
        return event

    def schedule_after(
        self,
        delay: Nanos,
        callback: Callable[[Nanos], Any],
        *,
        priority: int = EventPriority.AGENT_WAKEUP,
        label: str = "",
        payload: Any = None,
    ) -> Event:
        """Schedule ``callback`` for ``delay`` nanoseconds from now."""
        if int(delay) < 0:
            raise ClockError(f"delay must be non-negative, got {delay}")
        return self.schedule_at(
            Nanos(int(self._now) + int(delay)),
            callback,
            priority=priority,
            label=label,
            payload=payload,
        )

    def schedule_recurring(
        self,
        interval: Nanos,
        callback: Callable[[Nanos], Any],
        *,
        until: Nanos | None = None,
        priority: int = EventPriority.MEASUREMENT,
        label: str = "",
    ) -> None:
        """Self-rescheduling periodic callback.

        Implemented by having the wrapper re-schedule itself rather than by
        pre-populating the heap with every occurrence: a 6.5-hour session
        sampled every millisecond is 23 million events, and materialising them
        up front would dominate the simulation's memory.
        """
        if int(interval) <= 0:
            raise ClockError("interval must be positive")

        def tick(timestamp: Nanos) -> None:
            callback(timestamp)
            nxt = Nanos(int(timestamp) + int(interval))
            if until is None or int(nxt) <= int(until):
                self.schedule_at(nxt, tick, priority=priority, label=label)

        self.schedule_after(interval, tick, priority=priority, label=label)

    def stop(self) -> None:
        """Request that :meth:`run` return after the current event completes."""
        self._stop = True

    def step(self) -> Event | None:
        """Fire the single next event. Returns it, or ``None`` if the heap is empty."""
        if not self._heap:
            return None
        event = heapq.heappop(self._heap)
        self._now = event.timestamp
        self._processed += 1
        event.callback(event.timestamp)
        return event

    def run(self, until: Nanos | None = None, *, max_events: int | None = None) -> int:
        """Process events until exhausted, ``until``, or ``max_events``.

        Returns the number of events processed. Events scheduled beyond
        ``until`` are **left on the heap**, so a run can be resumed -- which is
        what makes intraday session boundaries expressible without rebuilding
        the world.
        """
        self._stop = False
        count = 0
        while self._heap and not self._stop:
            if until is not None and int(self._heap[0].timestamp) > int(until):
                break
            if max_events is not None and count >= max_events:
                break
            self.step()
            count += 1
        if until is not None and not self._stop:
            # Advance the clock to the horizon even if nothing was scheduled
            # there, so measurements taken after the run see the right time.
            self._now = Nanos(max(int(self._now), int(until)))
        return count

    def drain(self) -> Iterator[Event]:
        """Yield remaining events in order without firing them (for inspection)."""
        while self._heap:
            yield heapq.heappop(self._heap)

    def reset(self, start: Nanos = Nanos(0)) -> None:
        """Clear the heap and reset the clock, for a fresh episode."""
        self._heap.clear()
        self._sequence = 0
        self._now = start
        self._processed = 0
        self._stop = False
