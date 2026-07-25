# DDR-003: Agents return actions; they never call the exchange

- **Status:** Accepted
- **Affects:** `sim/agents.py`, `sim/world.py`

## Context

An agent needs to submit and cancel orders. The direct approach gives each agent a
reference to the `MatchingEngine` and lets it call `submit()`.

## Decision

Agents implement three handlers (`on_market_data`, `on_fill`, `on_wakeup`) and
return a list of `Action` objects. `MarketSimulation` is the only thing that
touches the venue.

## Rationale

Three properties follow structurally rather than by discipline:

1. **No look-ahead bias.** An agent cannot read the book at a moment it should not
   have; market data is *pushed* to it at a scheduled time. With a direct engine
   reference, an agent could call `top_of_book()` mid-decision and observe state
   from after its own decision point. That bug is nearly invisible in review and
   it inflates every backtest.
2. **Latency is unavoidable.** The world draws a latency sample and schedules the
   order's arrival. An agent cannot opt out of it, so it cannot accidentally trade
   at a price it could not physically have reached.
3. **The same agent runs in simulation, replay, and live.** Because the agent
   expresses *intent* rather than performing *execution*, swapping the world for a
   historical replayer or a live gateway requires no change to the agent.

The cost is one level of indirection and a slightly awkward flow for order ids:
the agent does not know its order's id until the world assigns one, which is why
`MarketMaker.register_quote` exists. That awkwardness is real, and it is a fair
price for making look-ahead bias structurally impossible rather than merely
discouraged.

## Consequences

- **Positive:** the three properties above; agents are trivially unit-testable
  because their handlers are pure functions of their inputs and state.
- **Negative:** an agent cannot query the book on demand, so strategies that
  genuinely need on-demand depth must request it via a wakeup.
- **Negative:** the indirection costs a small amount of performance per action.

## Alternatives considered

**Give agents a read-only book view.** Tempting, and it would remove the
awkwardness. Rejected because "read-only" does not solve the problem: the issue is
*when* the read happens, not whether it mutates.

**Enforce the rule by convention and code review.** Rejected. This is precisely
the class of bug that survives review, because the incorrect code looks
reasonable.
