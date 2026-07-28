"""Validation for the forecasting baselines and their scoring rules.

Written because mutation testing said they needed it. The baselines module scored
**0 out of 14** mutants killed: `scripts/mutation_test.py` had been pointed at
`test_sequence.py`, which tests the neural model, and no file tested the
baselines at all. Every number in `docs/MODEL_LEADERBOARD.md` rests on these, so
that was the weakest link in the repository's most quoted result.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.models.baselines import (
    ForecastScore,
    ewma_volatility_forecast,
    garch_volatility_forecast,
    historical_volatility_forecast,
    pinball_loss,
    qlike,
    random_walk_volatility_forecast,
    score_forecast,
)


def clustered(n: int = 1500, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.10, 0.85
    variance = omega / (1 - alpha - beta)
    out = np.empty(n)
    for i in range(n):
        variance = omega + alpha * (out[i - 1] ** 2 if i else 0.0) + beta * variance
        out[i] = np.sqrt(variance) * rng.standard_normal()
    return out


# --------------------------------------------------------------------------- #
# Pinball loss
# --------------------------------------------------------------------------- #
def test_pinball_loss_is_zero_for_a_perfect_forecast():
    values = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(values, values, 0.5) == pytest.approx(0.0)


def test_pinball_loss_is_asymmetric_and_in_the_right_direction():
    r"""At :math:`\tau = 0.9` under-forecasting must cost more than over-forecasting.

    The quantile loss weights the two sides by tau and (1 - tau). Getting this
    backwards makes the scoring rule improper and rewards hedging, which is the
    entire reason it was chosen.
    """
    actual = np.array([10.0])
    under = pinball_loss(actual, np.array([8.0]), 0.9)  # forecast too low
    over = pinball_loss(actual, np.array([12.0]), 0.9)  # forecast too high
    assert under > over

    # And the weights are exactly tau and 1 - tau.
    assert under == pytest.approx(0.9 * 2.0)
    assert over == pytest.approx(0.1 * 2.0)


def test_pinball_loss_is_symmetric_at_the_median():
    actual = np.array([10.0])
    assert pinball_loss(actual, np.array([8.0]), 0.5) == pytest.approx(
        pinball_loss(actual, np.array([12.0]), 0.5)
    )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_an_out_of_range_quantile_is_refused(bad):
    with pytest.raises(ValueError, match="quantile must lie"):
        pinball_loss(np.array([1.0]), np.array([1.0]), bad)


def test_pinball_loss_is_minimised_at_the_true_quantile():
    """The defining property of a proper scoring rule."""
    rng = np.random.default_rng(1)
    sample = rng.normal(0, 1, 20_000)
    tau = 0.25
    truth = float(np.quantile(sample, tau))

    at_truth = pinball_loss(sample, np.full(sample.size, truth), tau)
    for offset in (-0.5, -0.2, 0.2, 0.5):
        assert pinball_loss(sample, np.full(sample.size, truth + offset), tau) > at_truth


# --------------------------------------------------------------------------- #
# QLIKE
# --------------------------------------------------------------------------- #
def test_qlike_is_zero_for_a_perfect_variance_forecast():
    variance = np.array([0.01, 0.04, 0.09])
    assert qlike(variance, variance) == pytest.approx(0.0)


def test_qlike_is_positive_for_any_error():
    truth = np.full(100, 0.04)
    assert qlike(truth, np.full(100, 0.02)) > 0
    assert qlike(truth, np.full(100, 0.08)) > 0


def test_qlike_punishes_underprediction_harder_than_overprediction():
    r"""The asymmetry that makes QLIKE the right loss for variance.

    Underforecasting variance by half is worse than overforecasting by double,
    which is what stops a model quietly minimising the loss by shading its risk
    estimates downward. MSE on variance does the opposite.
    """
    truth = np.full(500, 0.04)
    too_low = qlike(truth, np.full(500, 0.02))
    too_high = qlike(truth, np.full(500, 0.08))
    assert too_low > too_high


def test_qlike_ignores_nonpositive_and_missing_values():
    truth = np.array([0.04, 0.04, np.nan, 0.04])
    forecast = np.array([0.04, -1.0, 0.04, 0.04])
    assert qlike(truth, forecast) == pytest.approx(0.0)


def test_qlike_returns_nan_when_nothing_is_usable():
    assert np.isnan(qlike(np.array([-1.0]), np.array([-1.0])))


# --------------------------------------------------------------------------- #
# The forecasters
# --------------------------------------------------------------------------- #
def test_every_forecaster_recovers_a_constant_volatility():
    r"""A series with fixed volatility must be forecast at that volatility.

    The tolerances differ per forecaster and that is not arbitrary. EWMA with
    :math:`\lambda = 0.94` has an effective sample of about
    :math:`1/(1 - \lambda) \approx 17` observations, so its relative standard
    error is roughly :math:`1/\sqrt{2 \times 17} \approx 17\%`. Holding it to the
    same band as an estimator that averages 252 days would be demanding precision
    it deliberately trades away for responsiveness.
    """
    sigma = 0.02
    returns = np.random.default_rng(2).normal(0, sigma, 3000)

    for forecaster, tolerance in (
        (historical_volatility_forecast, 0.10),
        (garch_volatility_forecast, 0.10),
        (ewma_volatility_forecast, 0.40),
    ):
        assert forecaster(returns) == pytest.approx(sigma, rel=tolerance), forecaster.__name__


def test_ewma_weights_recent_observations_more():
    """The defining property. A calm history then a shock must move the forecast.

    What is compared is *relative* responsiveness. Two earlier attempts got this
    wrong: the first used a constant calm series, whose dispersion is 2e-19 so any
    ratio against it is meaningless; the second assumed a 500-day standard
    deviation "barely notices" one observation, which is false when that
    observation is a hundred sigma -- its squared contribution alone dwarfs the
    other five hundred combined, and the trailing estimate jumps sixfold.

    Both estimators move. EWMA moves several times further, which is the property
    that matters.
    """
    rng = np.random.default_rng(11)
    calm = rng.normal(0, 0.001, 500)
    shocked = np.concatenate([calm, np.array([0.10])])

    ewma_ratio = ewma_volatility_forecast(shocked) / ewma_volatility_forecast(calm)
    trailing_ratio = historical_volatility_forecast(shocked) / historical_volatility_forecast(calm)

    assert ewma_ratio > 3 * trailing_ratio, (
        f"EWMA moved {ewma_ratio:.1f}x against the trailing estimate's {trailing_ratio:.1f}x"
    )


def test_ewma_is_a_second_moment_and_the_trailing_estimate_is_a_dispersion():
    r"""They agree on mean-zero returns and disagree otherwise, which is by design.

    EWMA computes :math:`\sqrt{\sum w_i r_i^2}` -- root mean square about *zero*,
    which is the right object for a return series whose mean is not estimable.
    The trailing estimator computes a standard deviation about the sample mean.
    On a constant series the first returns the level and the second returns zero.
    """
    constant = np.full(500, 0.001)
    assert ewma_volatility_forecast(constant) == pytest.approx(0.001, rel=1e-6)
    assert historical_volatility_forecast(constant) == pytest.approx(0.0, abs=1e-12)

    # On mean-zero returns the distinction vanishes.
    centred = np.random.default_rng(12).normal(0, 0.02, 4000)
    centred -= centred.mean()
    assert ewma_volatility_forecast(centred) == pytest.approx(
        historical_volatility_forecast(centred), rel=0.5
    )


def test_a_lower_lambda_reacts_faster():
    returns = np.concatenate([np.full(300, 0.001), np.array([0.05])])
    fast = ewma_volatility_forecast(returns, lam=0.80)
    slow = ewma_volatility_forecast(returns, lam=0.99)
    assert fast > slow


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_an_invalid_lambda_is_refused(bad):
    with pytest.raises(ValueError, match="lambda must lie"):
        ewma_volatility_forecast(np.random.default_rng(0).normal(0, 0.01, 100), lam=bad)


def test_the_random_walk_uses_only_the_most_recent_return():
    returns = np.array([0.05, 0.01, 0.002])
    assert random_walk_volatility_forecast(returns) == pytest.approx(0.002)


def test_garch_falls_back_when_there_are_no_arch_effects():
    """Fitting GARCH to a series without clustering is worse than not fitting it."""
    plain = np.random.default_rng(3).normal(0, 0.02, 2000)
    assert garch_volatility_forecast(plain) == pytest.approx(
        historical_volatility_forecast(plain), rel=1e-9
    )


def test_garch_responds_to_clustering_where_the_trailing_estimate_does_not():
    """On a clustered series after a quiet stretch, GARCH must forecast lower."""
    returns = clustered(2000, seed=4)
    # Append a calm run: GARCH should follow it down, a 252-day window should not.
    calm = np.concatenate([returns, np.full(30, 1e-4)])
    assert garch_volatility_forecast(calm) < historical_volatility_forecast(calm)


def test_forecasters_return_nan_rather_than_crashing_on_a_short_series():
    tiny = np.array([0.01, -0.01])
    assert np.isnan(historical_volatility_forecast(tiny))
    assert np.isnan(ewma_volatility_forecast(tiny))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_a_better_forecaster_scores_better_on_both_rules():
    """The property the leaderboard depends on."""
    rng = np.random.default_rng(5)
    sigma = 0.02
    actual = rng.normal(0, sigma, 4000)

    good = score_forecast("good", actual, np.full(4000, sigma))
    bad = score_forecast("bad", actual, np.full(4000, sigma * 4))

    assert good.qlike < bad.qlike
    assert good.pinball_mean < bad.pinball_mean
    assert good.beats(bad)
    assert not bad.beats(good)


def test_beating_requires_winning_on_both_rules():
    """Winning one and losing the other is not winning."""
    mixed = ForecastScore(name="a", n_forecasts=10, qlike=1.0, pinball_mean=2.0)
    other = ForecastScore(name="b", n_forecasts=10, qlike=2.0, pinball_mean=1.0)
    assert not mixed.beats(other)
    assert not other.beats(mixed)


def test_bias_has_the_right_sign():
    actual = np.random.default_rng(6).normal(0, 0.02, 2000)
    over = score_forecast("over", actual, np.full(2000, 0.10))
    under = score_forecast("under", actual, np.full(2000, 0.001))
    assert over.bias > 0
    assert under.bias < 0


def test_scoring_drops_unusable_forecasts_rather_than_failing():
    actual = np.array([0.01, 0.02, 0.03, 0.04])
    forecast = np.array([0.02, np.nan, -1.0, 0.02])
    score = score_forecast("partial", actual, forecast)
    assert score.n_forecasts == 2


def test_scoring_an_empty_series_reports_nan_rather_than_raising():
    score = score_forecast("empty", np.zeros(0), np.zeros(0))
    assert score.n_forecasts == 0
    assert np.isnan(score.qlike)


def test_every_requested_quantile_is_scored():
    actual = np.random.default_rng(7).normal(0, 0.02, 500)
    score = score_forecast("q", actual, np.full(500, 0.02), quantiles=(0.1, 0.5, 0.9))
    assert set(score.pinball_by_quantile) == {0.1, 0.5, 0.9}
    assert score.pinball_mean == pytest.approx(
        float(np.mean(list(score.pinball_by_quantile.values())))
    )
