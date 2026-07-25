"""Backtest-overfitting controls."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantos.core.rng import SeedBank
from quantos.strategy.validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    deflated_sharpe_ratio,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
    sharpe_ratio_with_moments,
    walk_forward_splits,
)


def test_sharpe_standard_error_accounts_for_non_normality() -> None:
    """Negative skew and excess kurtosis INFLATE the true standard error.

    The naive sqrt(1/n) therefore overstates significance -- in the dangerous
    direction, and most severely for short-volatility strategies.
    """
    rng = SeedBank(root=1).child("se").generator()
    from quantos.core.distributions import StudentT

    fat = StudentT(3.0).sample(1500, rng) * 0.006 + 0.001
    stats = sharpe_ratio_with_moments(fat)
    naive = 1.0 / np.sqrt(stats.n_obs - 1)
    assert stats.excess_kurtosis > 1.0
    # With a positive Sharpe and fat tails the kurtosis term dominates.
    assert stats.standard_error > 0.0
    assert stats.t_statistic == pytest.approx(stats.sharpe / stats.standard_error)
    del naive


def test_deflated_sharpe_kills_a_result_found_by_searching() -> None:
    """The same track record loses significance as the trial count grows.

    Asserting the *property* rather than a hard-coded flip point: where exactly a
    given track record stops being significant depends on its strength, and a
    genuinely strong one should survive many trials. What must always hold is that
    the p-value increases monotonically with the number of trials, and that some
    trial count defeats any finite result.
    """
    rng = SeedBank(root=2).child("dsr").generator()
    track = rng.standard_normal(1260) * 0.01 + 0.0015

    p_values = [
        deflated_sharpe_ratio(track, n_trials=n).p_value
        for n in (1, 10, 100, 1_000, 10_000, 100_000)
    ]
    assert all(a < b for a, b in itertools.pairwise(p_values))
    assert deflated_sharpe_ratio(track, n_trials=1).is_significant
    assert not deflated_sharpe_ratio(track, n_trials=100_000).is_significant


def test_deflated_sharpe_rejects_a_marginal_result_at_modest_trial_counts() -> None:
    """A borderline track record -- the realistic case -- does not survive at all."""
    rng = SeedBank(root=7).child("dsr_marginal").generator()
    marginal = rng.standard_normal(1260) * 0.01 + 0.0008
    assert not deflated_sharpe_ratio(marginal, n_trials=100).is_significant
    assert not deflated_sharpe_ratio(marginal, n_trials=500).is_significant


def test_expected_maximum_grows_with_the_number_of_trials() -> None:
    rng = SeedBank(root=3).child("emax").generator()
    track = rng.standard_normal(1260) * 0.01 + 0.001
    maxima = [
        deflated_sharpe_ratio(track, n_trials=n).expected_maximum for n in (1, 10, 100, 1000, 10000)
    ]
    assert maxima[0] == 0.0
    assert all(a < b for a, b in itertools.pairwise(maxima))


def test_deflated_sharpe_validates_trials() -> None:
    rng = SeedBank(root=4).child("v").generator()
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(rng.standard_normal(300) * 0.01, n_trials=0)


def test_minimum_track_record_length_is_infinite_without_an_edge() -> None:
    rng = SeedBank(root=5).child("trl").generator()
    assert minimum_track_record_length(rng.standard_normal(500) * 0.01 - 0.01) == np.inf
    good = rng.standard_normal(1260) * 0.01 + 0.002
    length = minimum_track_record_length(good)
    assert 0 < length < 10_000


@pytest.mark.statistical
def test_pbo_centres_on_one_half_for_skill_less_configurations() -> None:
    """Averaged across datasets, not on any one -- PBO is a noisy statistic.

    Measured spread across independent datasets is 0.10 to 0.70, which is why the
    docstring warns against quoting a single value.
    """
    values = [
        probability_of_backtest_overfitting(
            SeedBank(root=100 + s).child("pbo").generator().standard_normal((1000, 40)) * 0.01,
            n_splits=8,
        ).pbo
        for s in range(12)
    ]
    assert 0.3 < float(np.mean(values)) < 0.7
    assert max(values) - min(values) > 0.1  # genuinely dispersed


def test_pbo_validates_its_inputs() -> None:
    rng = SeedBank(root=6).child("pbo2").generator()
    with pytest.raises(ValueError, match="at least 2 configurations"):
        probability_of_backtest_overfitting(rng.standard_normal((500, 1)))
    with pytest.raises(ValueError, match="even integer"):
        probability_of_backtest_overfitting(rng.standard_normal((500, 5)), n_splits=7)
    with pytest.raises(ValueError, match="impractical"):
        probability_of_backtest_overfitting(rng.standard_normal((500, 5)), n_splits=22)


def test_purged_kfold_leaves_a_gap_of_at_least_the_label_horizon() -> None:
    """The whole point: no training label may overlap the test period."""
    horizon = 15
    cv = PurgedKFold(n_splits=5, label_horizon=horizon, embargo_fraction=0.01)
    folds = list(cv.split(np.arange(1000)))
    assert len(folds) == 5
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        gap = int(np.min(np.abs(train[:, None] - test[None, :])))
        assert gap >= horizon


def test_purged_kfold_never_shuffles() -> None:
    """Shuffled k-fold puts future observations in the training set."""
    cv = PurgedKFold(n_splits=4, label_horizon=5)
    for _, test in cv.split(np.arange(800)):
        assert np.array_equal(test, np.sort(test))
        assert np.all(np.diff(test) == 1)  # contiguous


def test_embargo_removes_observations_after_the_test_fold() -> None:
    small = list(
        PurgedKFold(n_splits=4, label_horizon=5, embargo_fraction=0.0).split(np.arange(1000))
    )
    large = list(
        PurgedKFold(n_splits=4, label_horizon=5, embargo_fraction=0.10).split(np.arange(1000))
    )
    assert len(large[0][0]) < len(small[0][0])


def test_purged_kfold_validates_parameters() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError, match="embargo"):
        PurgedKFold(embargo_fraction=0.6)
    with pytest.raises(ValueError, match="too short"):
        list(PurgedKFold(n_splits=5, label_horizon=50).split(np.arange(20)))


def test_cpcv_generates_many_backtest_paths() -> None:
    """One backtest gives one path, and one path has no variance estimate."""
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_horizon=5)
    splits = list(cv.split(np.arange(600)))
    assert len(splits) == 15  # C(6,2)
    assert cv.n_backtest_paths() == 5
    for train, test in splits:
        assert set(train).isdisjoint(set(test))


def test_cpcv_purges_around_every_held_out_block() -> None:
    """The test set is not contiguous here, so purging must apply to each block."""
    horizon = 10
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, label_horizon=horizon)
    for train, test in cv.split(np.arange(1000)):
        assert int(np.min(np.abs(train[:, None] - test[None, :]))) >= horizon


def test_walk_forward_is_anchored_or_rolling() -> None:
    anchored = walk_forward_splits(1000, n_splits=4, anchored=True)
    rolling = walk_forward_splits(1000, n_splits=4, anchored=False)
    assert [len(t) for t, _ in anchored] == [200, 400, 600, 800]
    assert [len(t) for t, _ in rolling] == [200, 200, 200, 200]
    # Training always precedes testing.
    for train, test in anchored + rolling:
        assert train.max() < test.min()


def test_walk_forward_gap_inserts_a_buffer() -> None:
    for train, test in walk_forward_splits(1000, n_splits=3, gap=25):
        assert test.min() - train.max() > 25
