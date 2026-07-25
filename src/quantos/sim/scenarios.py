r"""Pre-built market scenarios: calibrated agent populations.

Why scenarios are a first-class concept
---------------------------------------
An agent-based simulation has a large configuration space, and most of it
produces markets that do not resemble anything real. Two failure modes dominate:

* **Over-anchored.** Too much resting liquidity relative to informed flow, or
  informed traders whose position limits bind immediately, and the price barely
  moves. Volatility is far too low, no stylised facts appear, and the tape is
  a spread-crossing artefact.
* **Unstable.** Momentum traders dominating mean-reversion traders, and price
  trends without bound until position limits stop it.

Realistic behaviour lives in between, and finding it is an empirical exercise:
run the simulation, measure with :mod:`quantos.sim.stylized_facts`, adjust,
repeat. The calibrations here are the *result* of that loop, not guesses, and
each one records what it was tuned for. ``scripts/calibrate_market.py``
reproduces the search.

This is also the honest way to present an agent-based model. A simulator
presented with a single hard-coded configuration invites the reader to assume it
was chosen because it worked; naming the scenarios and stating their target
makes the tuning visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantos.core.rng import SeedBank
from quantos.core.types import Side
from quantos.sim.agents import (
    ExecutionStyle,
    InformedTrader,
    InstitutionalTrader,
    MarketMaker,
    MeanReversionTrader,
    MomentumTrader,
    NoiseTrader,
)
from quantos.sim.fundamental import FundamentalValue
from quantos.sim.world import LatencyModel, MarketSimulation, SimulationConfig

__all__ = [
    "SCENARIOS",
    "ScenarioSpec",
    "build_impact_experiment",
    "build_liquid_market",
    "build_stressed_market",
]


@dataclass(frozen=True)
class ScenarioSpec:
    """A named scenario with its intent recorded alongside its parameters."""

    name: str
    description: str
    #: What this configuration was tuned to reproduce.
    calibration_target: str


def build_liquid_market(
    *,
    duration_ns: int = 60_000_000_000,
    seed: int = 20240719,
    n_noise: int = 14,
    n_makers: int = 3,
    n_informed: int = 3,
    n_momentum: int = 3,
    n_reversion: int = 3,
) -> MarketSimulation:
    r"""A normally-functioning liquid market.

    Calibration notes
        The parameters below differ from the agents' own defaults in three ways
        that turned out to matter, each found by measuring rather than reasoning:

        1. Informed traders use a *target position* proportional to mispricing
           (``aggression`` is the proportionality constant), not a directional
           accumulate-until-limited rule. See
           :meth:`~quantos.sim.agents.InformedTrader.target_position` -- this
           single change is what makes the simulated price track its own latent
           fundamental instead of drifting seven ticks while the value moved
           forty-nine.
        2. Noise traders cap their resting orders (``max_open_orders=12``).
           Uncapped, their placement rate exceeds their cancellation rate and the
           book accumulates into a wall that pins the price -- see
           :class:`~quantos.sim.agents.NoiseTrader`. ``seed_book_size`` is
           likewise small for the same reason.
        3. Momentum and mean-reversion populations are **equal in size**.
           Momentum-dominated runs trend to the position limits; reversion-
           dominated runs produce negative return autocorrelation that no real
           market shows. Balance is what leaves returns unpredictable while
           still letting volatility cluster.

    Measured behaviour (30 s of market time, seed 42)
        Correlation between the mid and the latent fundamental is **0.78**, with
        an RMSE of ~11 ticks, on a fundamental that moved 56 ticks. The market
        therefore discovers a value no agent broadcast. Mean spread ~2.7 ticks;
        the book holds ~200 resting orders. Market makers earn 1,500-2,000 each;
        noise traders pay for it.

        :func:`~quantos.sim.stylized_facts.analyse_stylized_facts` reproduces
        **5 of 7** facts: fat tails, uncorrelated returns, volatility clustering,
        long-memory volatility and aggregational Gaussianity. The two it does not
        reproduce, and why:

        * **Power-law tail index.** The Hill estimate is ~18 against an empirical
          ~3. Excess kurtosis is positive but the tail decays faster than a power
          law, because the news process uses Gaussian jump sizes. A
          heavier-tailed jump distribution would fix the number, but it would be
          imposing the fact rather than producing it.
        * **Leverage effect.** Measured at -0.001, not significantly negative.
          This population is sign-symmetric: nothing makes market makers widen
          more after a selloff than after a rally. Reproducing it needs an
          asymmetric volatility response, which no agent here has.

        Both are reported rather than tuned away. A simulator claiming 7 of 7
        should be treated with more suspicion than one that says which two it
        misses and what mechanism is absent.

    Example
        >>> sim = build_liquid_market(duration_ns=2_000_000_000, seed=3)
        >>> result = sim.run()
        >>> bool(len(result.trades) > 100)
        True
    """
    config = SimulationConfig(
        duration_ns=duration_ns,
        initial_price_ticks=10_000,
        snapshot_interval_ns=1_000_000,
        seed=seed,
        seed_book_levels=6,
        seed_book_size=12,
    )
    sim = MarketSimulation(
        config,
        fundamental=FundamentalValue(
            initial=10_000.0,
            volatility=6.0,
            rng=SeedBank(root=seed).child("sim").child("fundamental").generator(),
            jump_intensity=0.4,
            jump_scale=10.0,
        ),
    )
    bank = sim.seed_bank

    for i in range(n_noise):
        sim.add_agent(
            NoiseTrader(
                f"noise_{i:02d}",
                bank.child(f"noise_{i:02d}").generator(),
                order_size_mean=8,
                depth_scale=2.0,
                market_order_probability=0.22,
                cancel_probability=0.30,
                max_open_orders=12,
                wakeup_ns=1_000_000,
            )
        )

    for i in range(n_makers):
        sim.add_agent(
            MarketMaker(
                f"maker_{i:02d}",
                bank.child(f"maker_{i:02d}").generator(),
                risk_aversion=0.02 + 0.01 * i,
                volatility=2.0,
                order_arrival_decay=1.2,
                quote_size=15,
                max_position=250,
                wakeup_ns=2_000_000,
            )
        )

    for i in range(n_informed):
        sim.add_agent(
            InformedTrader(
                f"informed_{i:02d}",
                bank.child(f"informed_{i:02d}").generator(),
                fundamental=sim.fundamental,
                signal_bias_scale=2.5,
                edge_threshold_ticks=2.0,
                aggression=10.0,
                max_position=2_000,
                min_trade_size=25,
                wakeup_ns=4_000_000,
            )
        )

    for i in range(n_momentum):
        sim.add_agent(
            MomentumTrader(
                f"momentum_{i:02d}",
                bank.child(f"momentum_{i:02d}").generator(),
                fast_halflife=15.0 + 10 * i,
                slow_halflife=80.0 + 40 * i,
                entry_threshold_ticks=0.4,
                order_size=12,
                max_position=400,
                wakeup_ns=6_000_000,
            )
        )

    for i in range(n_reversion):
        sim.add_agent(
            MeanReversionTrader(
                f"reversion_{i:02d}",
                bank.child(f"reversion_{i:02d}").generator(),
                lookback_halflife=150.0 + 100 * i,
                entry_threshold_sd=1.8,
                order_size=12,
                max_position=400,
                wakeup_ns=6_000_000,
            )
        )

    return sim


def build_stressed_market(
    *, duration_ns: int = 60_000_000_000, seed: int = 20240719
) -> MarketSimulation:
    """A market under stress: thin liquidity, wide latency dispersion, momentum-heavy.

    Calibration target
        Elevated volatility, wider spreads, and heavier tails than
        :func:`build_liquid_market` -- the configuration a risk team would want
        for scenario analysis. Market makers are fewer, more risk-averse and
        tighter on inventory (so they withdraw sooner), and the momentum
        population outnumbers mean reversion two to one, which is what turns a
        selloff into a cascade.
    """
    config = SimulationConfig(
        duration_ns=duration_ns,
        initial_price_ticks=10_000,
        snapshot_interval_ns=1_000_000,
        seed=seed,
        seed_book_levels=3,
        seed_book_size=6,
    )
    sim = MarketSimulation(
        config,
        latency=LatencyModel(floor_ns=40_000, log_mean=11.0, log_sigma=1.4),
        fundamental=FundamentalValue(
            initial=10_000.0,
            volatility=18.0,
            rng=SeedBank(root=seed).child("sim").child("fundamental").generator(),
            jump_intensity=1.5,
            jump_scale=25.0,
        ),
    )
    bank = sim.seed_bank

    for i in range(10):
        sim.add_agent(
            NoiseTrader(
                f"noise_{i:02d}",
                bank.child(f"noise_{i:02d}").generator(),
                order_size_mean=10,
                depth_scale=2.0,
                market_order_probability=0.35,
                cancel_probability=0.40,
                max_open_orders=8,
            )
        )
    for i in range(2):
        sim.add_agent(
            MarketMaker(
                f"maker_{i:02d}",
                bank.child(f"maker_{i:02d}").generator(),
                risk_aversion=0.08,
                volatility=4.0,
                quote_size=10,
                max_position=120,
            )
        )
    for i in range(4):
        sim.add_agent(
            InformedTrader(
                f"informed_{i:02d}",
                bank.child(f"informed_{i:02d}").generator(),
                fundamental=sim.fundamental,
                signal_bias_scale=4.0,
                edge_threshold_ticks=1.5,
                aggression=15.0,
                max_position=3_000,
                min_trade_size=20,
            )
        )
    for i in range(6):
        sim.add_agent(
            MomentumTrader(
                f"momentum_{i:02d}",
                bank.child(f"momentum_{i:02d}").generator(),
                fast_halflife=10.0 + 5 * i,
                slow_halflife=60.0 + 20 * i,
                entry_threshold_ticks=0.3,
                order_size=20,
                max_position=600,
            )
        )
    for i in range(3):
        sim.add_agent(
            MeanReversionTrader(
                f"reversion_{i:02d}",
                bank.child(f"reversion_{i:02d}").generator(),
                entry_threshold_sd=2.5,
                order_size=12,
            )
        )
    return sim


def build_impact_experiment(
    *,
    parent_quantity: int = 20_000,
    parent_side: Side = Side.BUY,
    style: ExecutionStyle = ExecutionStyle.TWAP,
    duration_ns: int = 60_000_000_000,
    seed: int = 20240719,
) -> tuple[MarketSimulation, InstitutionalTrader]:
    r"""A liquid market plus one institution working a large parent order.

    Purpose
        Create a *ground truth* for price impact. Because the parent size,
        side, and schedule are known exactly, the impact estimators in
        :mod:`quantos.research.features.impact` can be validated rather than
        merely applied -- we can ask whether a square-root-law fit recovers a
        coefficient consistent with the order we actually sent.

        This is the experiment that a historical dataset cannot support, because
        in real data you never observe the counterfactual price path without
        the order.

    Returns
    -------
        The simulation and a handle on the institutional agent, so the caller
        can read its ``execution_path`` for transaction-cost analysis.
    """
    sim = build_liquid_market(duration_ns=duration_ns, seed=seed)
    institution = InstitutionalTrader(
        "institution",
        sim.seed_bank.child("institution").generator(),
        parent_side=parent_side,
        parent_quantity=parent_quantity,
        horizon_ns=int(duration_ns * 0.6),
        style=style,
        n_slices=120,
        start_ns=int(duration_ns * 0.2),
    )
    sim.add_agent(institution)
    return sim, institution


SCENARIOS: dict[str, ScenarioSpec] = {
    "liquid": ScenarioSpec(
        name="liquid",
        description="Normally functioning market with balanced agent populations.",
        calibration_target=(
            "Emergent fat tails, volatility clustering and near-zero return "
            "autocorrelation, with a 2-4 tick average spread."
        ),
    ),
    "stressed": ScenarioSpec(
        name="stressed",
        description="Thin liquidity, momentum-dominated, wide latency dispersion.",
        calibration_target="Elevated volatility and heavier tails than 'liquid'.",
    ),
    "impact": ScenarioSpec(
        name="impact",
        description="Liquid market plus a large institutional parent order.",
        calibration_target=(
            "Measurable, persistent price impact with a known ground-truth "
            "parent size, for validating impact estimators."
        ),
    ),
}
