"""Simulation: reproducibility, economics, price discovery, and stylised facts."""

from __future__ import annotations

import numpy as np
import pytest

from quantos.core.types import Nanos, Side
from quantos.research.features.microstructure import price_discovery_efficiency
from quantos.sim.clock import ClockError, EventPriority, SimulationClock
from quantos.sim.fundamental import FundamentalValue
from quantos.sim.scenarios import build_impact_experiment, build_liquid_market
from quantos.sim.stylized_facts import analyse_stylized_facts, hurst_exponent
from quantos.sim.world import LatencyModel


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #
def test_clock_orders_by_timestamp_then_priority_then_sequence() -> None:
    """A total order is required, or the simulation is not replayable."""
    clock = SimulationClock()
    log: list[str] = []
    clock.schedule_at(Nanos(100), lambda _: log.append("later"))
    clock.schedule_at(
        Nanos(50), lambda _: log.append("action"), priority=EventPriority.AGENT_WAKEUP
    )
    clock.schedule_at(Nanos(50), lambda _: log.append("data"), priority=EventPriority.MARKET_DATA)
    clock.schedule_at(
        Nanos(50), lambda _: log.append("action2"), priority=EventPriority.AGENT_WAKEUP
    )
    assert clock.run() == 4
    # Market data precedes the agent actions that respond to it; equal-priority
    # events fire in insertion order.
    assert log == ["data", "action", "action2", "later"]


def test_clock_refuses_to_schedule_into_the_past() -> None:
    """Usually a latency model subtracting instead of adding."""
    clock = SimulationClock()
    clock.schedule_at(Nanos(100), lambda _: None)
    clock.run()
    with pytest.raises(ClockError, match="causality"):
        clock.schedule_at(Nanos(50), lambda _: None)


def test_clock_never_moves_backwards() -> None:
    clock = SimulationClock()
    for t in (500, 100, 300, 200):
        clock.schedule_at(Nanos(t), lambda _: None)
    seen = []
    while clock.step() is not None:
        seen.append(int(clock.now))
    assert seen == sorted(seen)


def test_recurring_events_do_not_materialise_the_whole_schedule() -> None:
    clock = SimulationClock()
    count = [0]
    clock.schedule_recurring(
        Nanos(10), lambda _: count.__setitem__(0, count[0] + 1), until=Nanos(100)
    )
    # Only one event is on the heap at a time, not ten.
    assert clock.pending == 1
    clock.run()
    assert count[0] == 10


def test_run_until_leaves_later_events_on_the_heap() -> None:
    clock = SimulationClock()
    for t in (10, 20, 500):
        clock.schedule_at(Nanos(t), lambda _: None)
    assert clock.run(until=Nanos(100)) == 2
    assert clock.pending == 1
    assert clock.run() == 1


# --------------------------------------------------------------------------- #
# Fundamental value
# --------------------------------------------------------------------------- #
def test_fundamental_is_idempotent_and_monotone_in_time() -> None:
    """Repeated queries at one timestamp must not advance the process.

    Otherwise the value path would depend on how many agents happened to ask.
    """
    fv = FundamentalValue(initial=10_000.0, volatility=2.0, rng=np.random.default_rng(0))
    first = fv.value_at(Nanos(1_000_000_000))
    assert fv.value_at(Nanos(1_000_000_000)) == first
    assert fv.value_at(Nanos(500_000_000)) == first  # no going backwards
    assert fv.value_at(Nanos(2_000_000_000)) != first


def test_fundamental_increments_are_uncorrelated() -> None:
    """The latent process must not itself contain the stylised facts we claim
    are emergent. Its increments are i.i.d. by construction; assert it."""
    fv = FundamentalValue(
        initial=0.0, volatility=2.0, rng=np.random.default_rng(1), jump_intensity=0.0
    )
    for i in range(1, 4_001):
        fv.value_at(Nanos(i * 1_000_000))
    _, values = fv.path_arrays()
    increments = np.diff(values)
    from quantos.core.stats.descriptive import autocorrelation

    acf = autocorrelation(increments, 20)[1:]
    band = 1.96 / np.sqrt(increments.size)
    assert np.mean(np.abs(acf) > band) < 0.25
    # And no volatility clustering either.
    acf_abs = autocorrelation(np.abs(increments), 20)[1:]
    assert np.mean(acf_abs) < band


def test_fundamental_validates_parameters() -> None:
    with pytest.raises(ValueError):
        FundamentalValue(initial=1.0, volatility=-1.0, rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def test_latency_is_positive_right_skewed_and_floored() -> None:
    model = LatencyModel(floor_ns=20_000, log_mean=10.0, log_sigma=0.7)
    rng = np.random.default_rng(0)
    draws = np.array([int(model.draw(rng)) for _ in range(20_000)])
    assert np.all(draws >= 20_000)
    # Log-normal, so the mean exceeds the median -- a symmetric model would not
    # reproduce the tail, and could produce negative latencies.
    assert np.mean(draws) > np.median(draws)
    assert model.mean_ns() == pytest.approx(float(np.mean(draws)), rel=0.05)


# --------------------------------------------------------------------------- #
# The simulation itself
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def liquid_result():
    """One 20-second liquid-market run, shared across tests (it is slow)."""
    return build_liquid_market(duration_ns=20_000_000_000, seed=42).run()


@pytest.mark.slow
def test_simulation_is_bit_reproducible() -> None:
    """Same seed, identical tape. A simulation without this is not an instrument."""
    first = build_liquid_market(duration_ns=5_000_000_000, seed=99).run()
    second = build_liquid_market(duration_ns=5_000_000_000, seed=99).run()
    assert len(first.trades) == len(second.trades)
    for a, b in zip(first.trades, second.trades, strict=False):
        assert (a.seq, a.price, a.quantity, a.aggressor_side) == (
            b.seq,
            b.price,
            b.quantity,
            b.aggressor_side,
        )
    assert first.agent_summary == second.agent_summary


@pytest.mark.slow
def test_different_seeds_produce_different_tapes() -> None:
    a = build_liquid_market(duration_ns=5_000_000_000, seed=1).run()
    b = build_liquid_market(duration_ns=5_000_000_000, seed=2).run()
    assert [int(t.price) for t in a.trades] != [int(t.price) for t in b.trades]


@pytest.mark.slow
def test_book_stays_uncrossed_and_bounded(liquid_result) -> None:
    """The order book cannot leak: quote inventory must stay bounded.

    An earlier version let noise traders accumulate orders without bound, reaching
    21,034 resting orders and pinning the price to a 7-tick range.
    """
    for snapshot in liquid_result.snapshots:
        if snapshot.bid_price is not None and snapshot.ask_price is not None:
            assert int(snapshot.bid_price) < int(snapshot.ask_price)
    spreads = liquid_result.spread_series()
    assert np.nanmean(spreads) < 20.0


@pytest.mark.slow
def test_market_makers_earn_and_noise_traders_pay(liquid_result) -> None:
    """The economics must come out the right way round.

    Market makers capture the spread and collect maker rebates; noise traders pay
    the spread and taker fees. If this inverts, the fee signs or the maker/taker
    flags are wrong.
    """
    makers = {k: v for k, v in liquid_result.agent_summary.items() if "maker" in k}
    noise = {k: v for k, v in liquid_result.agent_summary.items() if "noise" in k}
    assert makers and noise
    assert sum(v["pnl"] for v in makers.values()) > 0.0
    assert sum(v["pnl"] for v in noise.values()) < 0.0
    # Makers receive rebates, so their fees are negative.
    assert all(v["fees_paid"] < 0.0 for v in makers.values())
    assert all(v["fees_paid"] > 0.0 for v in noise.values())


@pytest.mark.slow
def test_price_moves_and_does_not_sit_pinned(liquid_result) -> None:
    """A necessary condition for discovery: the price must actually move.

    Much weaker than "the market is efficient", and unlike that claim it holds on
    every seed. An earlier version of the simulation pinned the mid to a 7-tick
    band while the fundamental wandered 49 ticks; this is the regression test.
    """
    mid = liquid_result.mid_series()
    _, values = liquid_result.fundamental_path
    assert np.nanmax(mid) - np.nanmin(mid) > 10.0
    assert float(values.max() - values.min()) > 10.0


def _discovery_correlation(seed: int, seconds: float, learning_rate: float | None) -> float:
    """Correlation between the market mid and the latent fundamental, one run."""
    from quantos.sim.agents import MarketMaker

    simulation = build_liquid_market(duration_ns=int(seconds * 1e9), seed=seed)
    if learning_rate is not None:
        for agent in simulation.agents.values():
            if isinstance(agent, MarketMaker):
                agent.learning_rate = learning_rate
    result = simulation.run()

    mid = result.mid_series()
    stamps, values = result.fundamental_path
    grid = np.arange(mid.size) * result.config.snapshot_interval_ns
    latent = np.interp(grid, stamps, values)
    finite = np.isfinite(mid)
    if int(np.sum(finite)) < 200:
        return float("nan")
    return price_discovery_efficiency(mid[finite], latent[finite]).correlation


@pytest.mark.slow
@pytest.mark.statistical
def test_order_flow_learning_improves_price_discovery() -> None:
    """Glosten-Milgrom learning beats no learning, averaged over seeds.

    This is the honest form of the claim. Discovery correlation on any single
    seed ranges from -0.69 to +0.88 (see scripts/measure_price_discovery.py), so
    asserting `correlation > 0.5` on one favourable run -- which this test used
    to do -- tests the seed rather than the mechanism.

    What survives averaging is the *ablation*: makers that infer fair value from
    aggressor-signed order flow track the latent fundamental measurably better
    than makers anchored only on the book's own microprice. Over 16 seeds the
    means are 0.29 against 0.09.
    """
    seeds = range(1, 7)
    with_learning = [
        c for c in (_discovery_correlation(s, 12.0, None) for s in seeds) if np.isfinite(c)
    ]
    without = [c for c in (_discovery_correlation(s, 12.0, 0.0) for s in seeds) if np.isfinite(c)]

    assert len(with_learning) >= 4 and len(without) >= 4
    assert float(np.mean(with_learning)) > float(np.mean(without))


@pytest.mark.slow
def test_stylised_facts_that_are_genuinely_emergent(liquid_result) -> None:
    """Volatility clustering and long memory are NOT in the latent process.

    The fundamental has i.i.d. increments (asserted separately), so if the tape
    shows clustering it came from agent interaction. Fat tails are only partly
    emergent -- the news process contributes some -- so they are not claimed here.

    `uncorrelated_returns` is deliberately NOT asserted. Enabling Glosten-Milgrom
    learning disperses the makers' fair values, which widens the spread and
    introduces negative first-order return autocorrelation (lag-1 ACF about
    -0.17) from bid-ask bounce. That is a real cost of the mechanism that buys
    better price discovery, quantified in build_liquid_market's docstring, and
    the honest response is to record the trade-off rather than assert past it.
    """
    report = analyse_stylized_facts(liquid_result.returns())
    assert report["volatility_clustering"].passed
    assert report["long_memory_volatility"].passed
    assert report.score >= 0.4


def test_stylised_facts_battery_discriminates() -> None:
    """A GARCH process passes the clustering tests; i.i.d. noise does not."""
    rng = np.random.default_rng(0)
    n = 20_000
    r = np.zeros(n)
    v = np.full(n, 1e-4)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        v[t] = 1e-6 + 0.10 * r[t - 1] ** 2 + 0.88 * v[t - 1]
        r[t] = np.sqrt(v[t]) * shocks[t]

    garch = analyse_stylized_facts(r)
    noise = analyse_stylized_facts(rng.standard_normal(n))
    assert garch["fat_tails"].passed and garch["volatility_clustering"].passed
    assert not noise["fat_tails"].passed
    assert not noise["volatility_clustering"].passed
    assert garch.score > noise.score


def test_leverage_criterion_rejects_pure_noise() -> None:
    """A sign test on a noisy correlation passes half the time; ours must not."""
    rng = np.random.default_rng(3)
    report = analyse_stylized_facts(rng.standard_normal(20_000))
    assert not report["leverage_effect"].passed


def test_hurst_exponent_on_known_processes() -> None:
    rng = np.random.default_rng(0)
    assert hurst_exponent(rng.standard_normal(20_000)) == pytest.approx(0.5, abs=0.08)
    assert hurst_exponent(np.cumsum(rng.standard_normal(20_000))) > 0.9


def test_stylised_facts_refuses_short_samples() -> None:
    with pytest.raises(ValueError, match="at least 500"):
        analyse_stylized_facts(np.random.default_rng(0).standard_normal(100))


@pytest.mark.slow
def test_institutional_order_is_fully_executed_and_moves_the_price() -> None:
    """Ground truth for impact: we know the parent size exactly."""
    simulation, institution = build_impact_experiment(
        parent_quantity=8_000,
        parent_side=Side.BUY,
        duration_ns=20_000_000_000,
        seed=5,
    )
    result = simulation.run()
    assert institution.executed == institution.parent_quantity
    assert institution.remaining == 0
    # A sustained one-sided buy programme must leave the agent long and paying.
    summary = result.agent_summary["institution"]
    assert summary["position"] > 0
    assert summary["fees_paid"] > 0.0
