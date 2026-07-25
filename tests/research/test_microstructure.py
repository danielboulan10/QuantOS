"""Microstructure estimators, validated against simulator ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from quantos.core.rng import SeedBank
from quantos.research.features.microstructure import (
    amihud_illiquidity,
    kyle_lambda,
    lee_ready_classification,
    order_flow_imbalance,
    price_discovery_efficiency,
    price_impact_decomposition,
    queue_imbalance,
    roll_implied_spread,
    vpin,
)


def test_ofi_distinguishes_the_three_book_event_types() -> None:
    """Price improvement, size change at an unchanged price, and level depletion
    are different events. Naively differencing sizes conflates all three."""
    ofi = order_flow_imbalance(
        [10.0, 10.0, 11.0], [100.0, 150.0, 120.0], [11.0, 11.0, 12.0], [100.0, 100.0, 100.0]
    )
    assert ofi[0] == 0.0  # no previous state
    assert ofi[1] == 50.0  # bid size +50 at an unchanged price
    assert ofi[2] == 220.0  # +120 new bid level, +100 as the ask lifts


def test_ofi_is_signed_correctly() -> None:
    """Rising bids are buying pressure; falling asks are selling pressure."""
    buying = order_flow_imbalance([10.0, 11.0], [100.0, 100.0], [12.0, 12.0], [100.0, 100.0])
    selling = order_flow_imbalance([10.0, 10.0], [100.0, 100.0], [12.0, 11.0], [100.0, 100.0])
    assert buying[1] > 0
    assert selling[1] < 0


def test_queue_imbalance_is_bounded_and_signed() -> None:
    assert queue_imbalance([100.0], [100.0])[0] == pytest.approx(0.0)
    assert queue_imbalance([300.0], [100.0])[0] == pytest.approx(0.5)
    assert queue_imbalance([0.0], [0.0])[0] == 0.0
    values = queue_imbalance([1.0, 50.0, 99.0], [99.0, 50.0, 1.0])
    assert np.all(np.abs(values) <= 1.0)


def test_kyle_lambda_recovers_a_known_impact_coefficient() -> None:
    rng = SeedBank(root=1).child("kyle").generator()
    volume = rng.standard_normal(4_000) * 100
    truth = 0.002
    changes = truth * volume + rng.standard_normal(4_000) * 0.05
    result = kyle_lambda(changes, volume)
    assert result.lambda_ == pytest.approx(truth, rel=0.05)
    assert result.is_significant
    assert result.market_depth == pytest.approx(1.0 / truth, rel=0.05)


def test_kyle_lambda_uses_hac_errors_because_flow_is_autocorrelated() -> None:
    """Order splitting guarantees autocorrelated flow; classical errors overstate."""
    rng = SeedBank(root=2).child("kyle_hac").generator()
    n = 3_000
    volume = np.zeros(n)
    shocks = rng.standard_normal(n) * 100
    for t in range(1, n):
        volume[t] = 0.8 * volume[t - 1] + shocks[t]  # persistent flow
    changes = 0.002 * volume + rng.standard_normal(n) * 0.05
    assert kyle_lambda(changes, volume).lambda_ == pytest.approx(0.002, rel=0.1)


def test_kyle_lambda_requires_enough_data() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        kyle_lambda(np.zeros(10), np.zeros(10))


def test_vpin_separates_toxic_from_balanced_flow() -> None:
    rng = SeedBank(root=3).child("vpin").generator()
    balanced = rng.choice([-1.0, 1.0], 40_000) * 10
    one_sided = np.full(40_000, 10.0)
    assert float(np.mean(vpin(balanced, bucket_volume=500, n_buckets=20))) < 0.2
    assert float(np.mean(vpin(one_sided, bucket_volume=500, n_buckets=20))) > 0.9


def test_vpin_is_bounded_to_the_unit_interval() -> None:
    rng = SeedBank(root=4).child("vpin2").generator()
    values = vpin(rng.standard_normal(20_000) * 10, bucket_volume=300, n_buckets=15)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_roll_recovers_the_spread_from_trade_prices_alone() -> None:
    """No quotes, no signs -- just bid-ask bounce in the autocovariance."""
    rng = SeedBank(root=5).child("roll").generator()
    spread = 0.10
    mid = np.cumsum(rng.standard_normal(8_000) * 0.001)
    trades = mid + np.where(rng.random(8_000) < 0.5, -spread / 2, spread / 2)
    assert roll_implied_spread(trades) == pytest.approx(spread, rel=0.05)


def test_roll_returns_nan_when_momentum_outweighs_the_bounce() -> None:
    """Its central limitation, surfaced rather than hidden behind a zero.

    Roll's estimator identifies the spread from *negative* first-order
    autocovariance in trade-price changes. When genuine price momentum makes that
    autocovariance positive, no spread is implied, and returning 0.0 would be a
    silent lie. Constructed here with an AR(1) in returns and no bid-ask bounce
    at all, so the autocovariance is unambiguously positive.
    """
    rng = SeedBank(root=99).child("momentum").generator()
    n = 4_000
    returns = np.zeros(n)
    shocks = rng.standard_normal(n) * 0.01
    for t in range(1, n):
        returns[t] = 0.6 * returns[t - 1] + shocks[t]
    trending = 100.0 + np.cumsum(returns)
    assert np.isnan(roll_implied_spread(trending))


def test_impact_decomposition_satisfies_its_identity() -> None:
    """effective spread = realised spread + permanent impact, exactly."""
    rng = SeedBank(root=6).child("impact").generator()
    n = 2_000
    signs = rng.choice([-1.0, 1.0], n)
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    trades = mid + signs * 0.05
    future = mid + signs * 0.02
    result = price_impact_decomposition(trades, mid, future, signs)
    assert result.effective_spread == pytest.approx(
        result.realised_spread + result.permanent_impact, rel=1e-12
    )
    assert 0.0 < result.adverse_selection_share < 1.0


def test_negative_realised_spread_flags_adverse_selection() -> None:
    """A maker earning the quoted spread but losing to information."""
    signs = np.array([1.0, 1.0, -1.0, -1.0])
    mid = np.full(4, 100.0)
    trades = mid + signs * 0.05
    future = mid + signs * 0.20  # price runs past the trade every time
    result = price_impact_decomposition(trades, mid, future, signs)
    assert result.maker_is_adversely_selected
    assert result.realised_spread < 0.0


def test_lee_ready_is_exact_when_trades_are_away_from_the_mid() -> None:
    result = lee_ready_classification([101.0, 99.0, 102.0], [100.0] * 3, true_signs=[1, -1, 1])
    assert result.accuracy == 1.0
    assert result.quote_rule_share == 1.0


def test_lee_ready_falls_back_to_the_tick_test_at_the_mid() -> None:
    """The quote rule is silent at the mid; this is where the errors concentrate."""
    result = lee_ready_classification([100.0, 100.0, 100.0], [100.0] * 3)
    assert result.quote_rule_share == 0.0
    assert result.signs.shape == (3,)


def test_lee_ready_error_rate_is_measurable_against_ground_truth() -> None:
    """The point of having a simulator: turn an assumption into a number."""
    rng = SeedBank(root=7).child("lr").generator()
    n = 5_000
    signs = rng.choice([-1.0, 1.0], n)
    mid = np.full(n, 100.0)
    # Half the trades occur exactly at the mid, where inference must guess.
    at_mid = rng.random(n) < 0.5
    trades = np.where(at_mid, mid, mid + signs * 0.05)
    result = lee_ready_classification(trades, mid, true_signs=signs)
    assert result.accuracy is not None
    assert result.error_rate is not None
    assert 0.5 < result.accuracy < 1.0
    assert result.tick_rule_accuracy is not None
    # The tick fallback is much worse than the quote rule -- the whole point.
    assert result.tick_rule_accuracy < result.accuracy


def test_price_discovery_detects_an_efficient_market() -> None:
    rng = SeedBank(root=8).child("pd").generator()
    value = np.cumsum(rng.standard_normal(3_000))
    efficient = value + rng.standard_normal(3_000) * 0.4
    result = price_discovery_efficiency(efficient, value)
    assert result.is_efficient
    assert result.correlation > 0.95
    assert result.beta == pytest.approx(1.0, abs=0.05)


def test_price_discovery_detects_a_lagging_price() -> None:
    """A price that trails the value shows up as a positive optimal lag."""
    rng = SeedBank(root=9).child("pd_lag").generator()
    value = np.cumsum(rng.standard_normal(4_000))
    lagged = np.concatenate([np.zeros(8), value[:-8]])
    result = price_discovery_efficiency(lagged, value, max_lag=30)
    assert result.optimal_lag > 0


def test_price_discovery_rejects_a_constant_series() -> None:
    with pytest.raises(ValueError, match="constant"):
        price_discovery_efficiency(np.ones(500), np.arange(500.0))


def test_amihud_illiquidity_rises_as_volume_falls() -> None:
    returns = np.full(500, 0.01)
    assert amihud_illiquidity(returns, np.full(500, 100.0)) > amihud_illiquidity(
        returns, np.full(500, 1000.0)
    )
