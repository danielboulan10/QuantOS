"""Validation for the calibration machinery.

Calibration is the claim on which every forward probability rests, so the test of
the tester matters. These check that it detects a forecaster it should reject, and
does not reject one it should accept.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.forecast.calibration import brier_decomposition, calibration_test


# --------------------------------------------------------------------------- #
# Brier decomposition
# --------------------------------------------------------------------------- #
def test_the_decomposition_adds_up_exactly():
    """BS = reliability - resolution + uncertainty + binning residual.

    The three-term textbook identity is exact only for discrete forecasts. With a
    continuous forecaster the within-bin variance is a real term, and without it
    the identity misses by about 4e-4 here.
    """
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0, 1, 4000)
    outcomes = rng.uniform(0, 1, 4000) < predicted

    brier, reliability, resolution, uncertainty, residual = brier_decomposition(predicted, outcomes)
    assert brier == pytest.approx(reliability - resolution + uncertainty + residual, abs=1e-12)
    # And the term most implementations drop is not negligible at this bin width.
    assert abs(residual) > 1e-4


def test_the_binning_residual_vanishes_for_a_discrete_forecaster():
    """With one value per bin the classical three-term identity is exact."""
    rng = np.random.default_rng(10)
    predicted = rng.choice([0.05, 0.25, 0.45, 0.65, 0.85], size=4000)
    outcomes = rng.uniform(0, 1, 4000) < predicted

    brier, reliability, resolution, uncertainty, residual = brier_decomposition(predicted, outcomes)
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert brier == pytest.approx(reliability - resolution + uncertainty, abs=1e-12)


def test_a_perfect_forecaster_scores_zero():
    predicted = np.array([0.0, 1.0] * 500)
    outcomes = predicted.astype(bool)
    brier, reliability, _, _, _ = brier_decomposition(predicted, outcomes)
    assert brier == pytest.approx(0.0, abs=1e-12)
    assert reliability == pytest.approx(0.0, abs=1e-12)


def test_always_predicting_the_base_rate_has_zero_resolution():
    """The failure the decomposition exists to expose: calibrated but useless."""
    rng = np.random.default_rng(1)
    outcomes = rng.uniform(0, 1, 5000) < 0.3
    predicted = np.full(5000, float(np.mean(outcomes)))

    brier, reliability, resolution, uncertainty, _ = brier_decomposition(predicted, outcomes)
    assert reliability == pytest.approx(0.0, abs=1e-6)
    assert resolution == pytest.approx(0.0, abs=1e-6)
    assert brier == pytest.approx(uncertainty, abs=1e-6)


def test_an_informative_forecaster_has_positive_resolution():
    rng = np.random.default_rng(2)
    truth = rng.uniform(0, 1, 5000) < 0.5
    predicted = np.where(truth, 0.9, 0.1)
    _, _, resolution, uncertainty, _ = brier_decomposition(predicted, truth)
    assert resolution > 0.5 * uncertainty


def test_an_empty_input_returns_nan_rather_than_raising():
    assert all(np.isnan(v) for v in brier_decomposition(np.zeros(0), np.zeros(0, dtype=bool)))


# --------------------------------------------------------------------------- #
# The walk-forward test itself
# --------------------------------------------------------------------------- #
def test_calibration_runs_and_reports_its_overlap():
    rng = np.random.default_rng(3)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 1600)))
    result = calibration_test(
        prices,
        horizon=21,
        train_window=700,
        step=40,
        n_paths=400,
        event="touch_down",
        threshold=0.05,
    )
    assert result.n_forecasts > 5
    assert result.n_independent <= result.n_forecasts
    assert any("overlap" in note or "independent" in note for note in result.notes)


def test_the_verdict_reports_insufficient_evidence_rather_than_success():
    """The bug this replaced: a bare boolean said 'calibrated' on a no-skill run.

    With very few forecasts no bucket can fail, and an earlier version read that
    as a pass. It must now report the absence of evidence as its own outcome.
    """
    rng = np.random.default_rng(4)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 1000)))
    result = calibration_test(
        prices,
        horizon=21,
        train_window=700,
        step=60,
        n_paths=300,
        event="touch_down",
        threshold=0.05,
    )
    assert "INSUFFICIENT EVIDENCE" in result.verdict
    assert result.is_calibrated is False


def test_a_short_series_is_refused_with_the_requirement_stated():
    with pytest.raises(ValueError, match="need at least"):
        calibration_test(np.full(200, 100.0), horizon=21, train_window=750)


def test_an_unknown_event_is_refused():
    rng = np.random.default_rng(5)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 1200)))
    with pytest.raises(ValueError, match="unknown event"):
        calibration_test(prices, horizon=21, train_window=700, step=80, event="nonsense")


def test_skill_is_negative_for_a_forecaster_worse_than_the_base_rate():
    rng = np.random.default_rng(6)
    outcomes = rng.uniform(0, 1, 3000) < 0.2
    # Deliberately anti-correlated with the truth.
    predicted = np.where(outcomes, 0.05, 0.6)
    brier, _, _, uncertainty, _ = brier_decomposition(predicted, outcomes)
    assert 1.0 - brier / uncertainty < 0


def test_buckets_partition_the_forecasts_without_double_counting():
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 1600)))
    result = calibration_test(
        prices,
        horizon=21,
        train_window=700,
        step=25,
        n_paths=400,
        event="touch_down",
        threshold=0.05,
    )
    assert sum(b.n_forecasts for b in result.buckets) == result.n_forecasts
