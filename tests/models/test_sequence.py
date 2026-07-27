"""Validation for the hand-written sequence model.

The gradient check is the test that matters. Backpropagation written by hand is
wrong until proven otherwise, and a wrong gradient does not crash -- it trains to
a worse optimum and looks like a modelling result. Every parameter is checked
against central finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.models.sequence import AttentionVolatilityModel, make_windows


def clustered(n: int = 1200, *, seed: int = 0) -> np.ndarray:
    """GARCH-like returns, so there is real clustering to learn."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.12, 0.84
    variance = omega / (1 - alpha - beta)
    out = np.empty(n)
    for i in range(n):
        variance = omega + alpha * (out[i - 1] ** 2 if i else 0.0) + beta * variance
        out[i] = np.sqrt(variance) * rng.standard_normal()
    return out


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def test_windows_are_strictly_causal():
    """No window may contain its own target. This is the leak that flatters models."""
    series = np.arange(20.0)
    inputs, targets = make_windows(series, window=5)
    for row, target in zip(inputs, targets, strict=True):
        assert target not in row
        assert target == row[-1] + 1  # the very next observation


def test_window_shapes():
    inputs, targets = make_windows(np.arange(100.0), window=10)
    assert inputs.shape == (90, 10)
    assert targets.shape == (90,)


def test_too_short_a_series_is_refused():
    with pytest.raises(ValueError, match="need more than"):
        make_windows(np.arange(5.0), window=10)


# --------------------------------------------------------------------------- #
# THE gradient check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "parameter",
    ["projection", "query", "key", "value", "hidden_weight", "hidden_bias", "output_weight"],
)
def test_analytic_gradients_match_finite_differences(parameter):
    """Every hand-derived gradient, against a central difference.

    A wrong gradient does not raise. It quietly trains to a worse optimum, and the
    result is then reported as a property of the architecture. This is the only
    honest way to ship hand-written backpropagation.
    """
    rng = np.random.default_rng(1)
    model = AttentionVolatilityModel(window=6, d_model=4, seed=2)
    model.input_scale = 1.0
    model.initialise()

    x = rng.normal(0, 1, (12, 6))
    y = rng.normal(0, 1, 12)

    _, gradients = model._loss_and_gradients(x, y)
    analytic = np.asarray(gradients[parameter], dtype=float)

    current = np.asarray(getattr(model, parameter), dtype=float)
    numeric = np.zeros_like(current)
    step = 1e-6
    for index in np.ndindex(current.shape):
        original = current[index]

        current[index] = original + step
        up, _ = model._loss_and_gradients(x, y)
        current[index] = original - step
        down, _ = model._loss_and_gradients(x, y)
        current[index] = original

        numeric[index] = (up - down) / (2 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-8)


def test_the_output_bias_gradient_matches_too():
    rng = np.random.default_rng(3)
    model = AttentionVolatilityModel(window=6, d_model=4, seed=4)
    model.input_scale = 1.0
    model.initialise()
    x, y = rng.normal(0, 1, (12, 6)), rng.normal(0, 1, 12)

    _, gradients = model._loss_and_gradients(x, y)
    step = 1e-6
    original = model.output_bias
    model.output_bias = original + step
    up, _ = model._loss_and_gradients(x, y)
    model.output_bias = original - step
    down, _ = model._loss_and_gradients(x, y)
    model.output_bias = original

    assert gradients["output_bias"] == pytest.approx((up - down) / (2 * step), rel=1e-4)


# --------------------------------------------------------------------------- #
# Numerical safety
# --------------------------------------------------------------------------- #
def test_the_predicted_scale_is_always_positive():
    """Softplus, not a clamp: a clamp would zero the gradient where it is needed."""
    model = AttentionVolatilityModel(window=8, d_model=4).initialise()
    extreme = np.concatenate([np.full((3, 8), -50.0), np.full((3, 8), 50.0), np.zeros((3, 8))])
    scale = model.predict_scaled(extreme)
    assert np.all(scale > 0)
    assert np.all(np.isfinite(scale))


def test_softplus_and_its_derivative_are_stable_at_both_extremes():
    from quantos.models.sequence import _sigmoid, _softplus

    x = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    assert np.all(np.isfinite(_softplus(x)))
    assert np.all(np.isfinite(_sigmoid(x)))
    assert _softplus(np.array([800.0]))[0] == pytest.approx(800.0)
    assert _sigmoid(np.array([-800.0]))[0] == pytest.approx(0.0)


def test_attention_weights_form_a_distribution():
    model = AttentionVolatilityModel(window=10, d_model=4).initialise()
    _, cache = model._forward(np.random.default_rng(5).normal(0, 1, (7, 10)))
    weights = cache["weights"]
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-12)
    assert np.all(weights >= 0)


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #
def test_training_reduces_the_loss():
    model = AttentionVolatilityModel(window=15, d_model=6, seed=6)
    model.fit(clustered(1000, seed=7), epochs=60, patience=60)
    assert model.history.train_loss[-1] < model.history.train_loss[0]


def test_the_validation_split_is_chronological_not_shuffled():
    """A shuffled split leaks the future backwards, which is the classic error."""
    returns = clustered(800, seed=8)
    model = AttentionVolatilityModel(window=10, d_model=4, seed=9)
    model.fit(returns, epochs=5, patience=5, validation_fraction=0.25)

    inputs, targets = make_windows(returns, 10)
    split = int(len(inputs) * 0.75)
    # input_scale must come from the training portion only.
    assert model.input_scale == pytest.approx(float(np.std(targets[:split], ddof=1)))
    assert model.input_scale != pytest.approx(float(np.std(targets, ddof=1)))


def test_a_learned_model_tracks_volatility_rather_than_emitting_a_constant():
    """The minimum bar: predictions must respond to the input.

    A model that has learned nothing outputs the unconditional volatility for
    every window, which would still lower the loss versus a bad initialisation.
    """
    returns = clustered(1500, seed=10)
    model = AttentionVolatilityModel(window=20, d_model=8, seed=11)
    model.fit(returns, epochs=120, patience=25)

    inputs, _ = make_windows(returns, 20)
    predictions = model.predict(inputs)
    realised = np.sqrt(np.mean(inputs**2, axis=1))

    assert float(np.std(predictions)) > 0.1 * float(np.mean(predictions))
    assert float(np.corrcoef(predictions, realised)[0, 1]) > 0.5


def test_training_refuses_a_series_too_short_to_split():
    model = AttentionVolatilityModel(window=20, d_model=4)
    with pytest.raises(ValueError, match="need more data"):
        model.fit(clustered(60, seed=12), epochs=5)


def test_the_parameter_count_is_small_by_design():
    """Small on purpose: a large model would fit the sample, not the process."""
    model = AttentionVolatilityModel(window=20, d_model=8).initialise()
    assert model.n_parameters() < 400
