"""Validation for the scenario engine.

The arithmetic is a regression and a multiplication. What is being tested is
whether the module refuses to sound more certain than it is -- and, in the
`confidently_wrong` group, whether it catches the specific case where a narrow
interval and a wrong answer coexist. That case is the reason the module exists,
so it gets the most attention.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.risk.scenario import (
    SCENARIOS,
    apply_shock,
    estimate_response,
)


# --------------------------------------------------------------------------- #
# Recovering a known sensitivity
# --------------------------------------------------------------------------- #
def test_a_planted_beta_is_recovered():
    """Nothing below means anything if the estimator cannot find a beta that is there."""
    rng = np.random.default_rng(0)
    rates = rng.standard_normal(1500) * 0.0005
    asset = -7.5 * rates + rng.standard_normal(1500) * 0.0006

    response = estimate_response(asset, {"rates": rates})
    assert response.betas["rates"] == pytest.approx(-7.5, rel=0.1)
    assert response.t_statistics["rates"] < -5


def test_two_factors_are_disentangled_when_they_are_independent():
    rng = np.random.default_rng(1)
    rates = rng.standard_normal(1500) * 0.0005
    oil = rng.standard_normal(1500) * 0.02
    asset = -4.0 * rates + 0.3 * oil + rng.standard_normal(1500) * 0.001

    response = estimate_response(asset, {"rates": rates, "oil": oil})
    assert response.betas["rates"] == pytest.approx(-4.0, rel=0.15)
    assert response.betas["oil"] == pytest.approx(0.3, rel=0.15)


def test_the_shock_response_is_the_beta_times_the_shock():
    rng = np.random.default_rng(2)
    rates = rng.standard_normal(1500) * 0.0005
    asset = -6.0 * rates + rng.standard_normal(1500) * 0.0005

    response = estimate_response(asset, {"rates": rates})
    shock = apply_shock(response, {"rates": 0.01})

    assert shock.point == pytest.approx(response.betas["rates"] * 0.01)
    assert shock.contributions["rates"] == pytest.approx(shock.point)


# --------------------------------------------------------------------------- #
# The interval
# --------------------------------------------------------------------------- #
def test_the_interval_widens_with_the_confidence_level():
    rng = np.random.default_rng(3)
    rates = rng.standard_normal(800) * 0.0005
    asset = -3.0 * rates + rng.standard_normal(800) * 0.004

    response = estimate_response(asset, {"rates": rates})
    narrow = apply_shock(response, {"rates": 0.01}, confidence=0.80)
    wide = apply_shock(response, {"rates": 0.01}, confidence=0.99)

    assert wide.high - wide.low > narrow.high - narrow.low
    assert narrow.point == pytest.approx(wide.point)


def test_a_beta_indistinguishable_from_zero_produces_an_interval_spanning_zero():
    """The honest output when there is no measurable relationship.

    A point estimate alone would still print a signed percentage here, which a
    reader would take as a direction.
    """
    rng = np.random.default_rng(4)
    rates = rng.standard_normal(1000) * 0.0005
    asset = rng.standard_normal(1000) * 0.01  # no relationship at all

    response = estimate_response(asset, {"rates": rates})
    shock = apply_shock(response, {"rates": 0.01})

    assert not shock.direction_is_certain
    assert shock.low < 0 < shock.high
    assert "not even the" in shock.summary()
    assert any("not distinguishable from zero" in note for note in response.notes)


def test_the_interval_is_flagged_as_too_narrow_when_several_factors_are_shocked():
    """Ignoring the covariance between betas errs toward overconfidence.

    That is the dangerous direction, so it is stated rather than silently
    accepted.
    """
    rng = np.random.default_rng(5)
    rates = rng.standard_normal(900) * 0.0005
    oil = rng.standard_normal(900) * 0.02
    asset = -4.0 * rates + 0.3 * oil + rng.standard_normal(900) * 0.002

    response = estimate_response(asset, {"rates": rates, "oil": oil})
    shock = apply_shock(response, {"rates": 0.01, "oil": 0.5})

    assert any("WIDER than the one shown" in note for note in shock.notes)


def test_a_large_shock_is_flagged_as_an_extrapolation():
    """Nobody asks about 5bps. The sizes people ask about are outside the sample."""
    rng = np.random.default_rng(6)
    rates = rng.standard_normal(900) * 0.0005
    asset = -4.0 * rates + rng.standard_normal(900) * 0.001

    response = estimate_response(asset, {"rates": rates})
    assert any(
        "extrapolated linearly" in note for note in apply_shock(response, {"rates": 0.01}).notes
    )
    assert not any(
        "extrapolated linearly" in note for note in apply_shock(response, {"rates": 0.001}).notes
    )


# --------------------------------------------------------------------------- #
# Confidently wrong -- the reason the module exists
# --------------------------------------------------------------------------- #
def _regime_switching_data(n: int = 2000, *, seed: int = 7):
    """A beta of +9 for four fifths of the sample and -3 for the last fifth.

    This is the real shape: QQQ's beta to the 10-year yield was +8.5, +9.2, +7.9
    and +9.4 across 2006-2021, then -2.8 through the 2022 hiking cycle.
    """
    rng = np.random.default_rng(seed)
    rates = rng.standard_normal(n) * 0.0005
    beta = np.where(np.arange(n) < 0.8 * n, 9.0, -3.0)
    asset = beta * rates + rng.standard_normal(n) * 0.004
    return rates, asset


def test_a_beta_that_changes_sign_across_subsamples_is_called_unstable():
    rates, asset = _regime_switching_data()
    response = estimate_response(asset, {"rates": rates}, n_subsamples=5)

    assert response.is_unstable("rates")
    assert "UNSTABLE" in response.summary()
    assert any("average of two regimes" in note for note in response.notes)


def test_a_narrow_interval_on_an_unstable_beta_is_flagged_as_confidently_wrong():
    """The most dangerous output a scenario tool can produce.

    The full-sample estimate is strongly significant and its interval excludes
    zero, so every conventional check passes -- while the relationship it
    describes has already reversed. The interval measures sampling error inside
    the window. It cannot measure the risk that the window stops applying.
    """
    rates, asset = _regime_switching_data()
    response = estimate_response(asset, {"rates": rates}, n_subsamples=5)
    shock = apply_shock(response, {"rates": 0.01})

    assert shock.direction_is_certain, "the interval excludes zero, which is the trap"
    assert shock.confidently_wrong
    assert min(shock.subsample_points) < 0 < max(shock.subsample_points)
    assert "READ THIS BEFORE THE NUMBER" in shock.summary()


def test_a_stable_beta_is_not_flagged_as_confidently_wrong():
    """The guard: the flag must key on instability, not merely on significance."""
    rng = np.random.default_rng(8)
    rates = rng.standard_normal(2000) * 0.0005
    asset = -6.0 * rates + rng.standard_normal(2000) * 0.001

    response = estimate_response(asset, {"rates": rates}, n_subsamples=5)
    shock = apply_shock(response, {"rates": 0.01})

    assert shock.direction_is_certain
    assert not shock.confidently_wrong
    assert not response.is_unstable("rates")
    assert "READ THIS BEFORE THE NUMBER" not in shock.summary()


def test_an_insignificant_beta_is_not_flagged_as_confidently_wrong_either():
    """The flag is about false precision, and there is no precision to falsify here."""
    rng = np.random.default_rng(9)
    rates = rng.standard_normal(1200) * 0.0005
    asset = rng.standard_normal(1200) * 0.01

    response = estimate_response(asset, {"rates": rates}, n_subsamples=5)
    shock = apply_shock(response, {"rates": 0.01})

    assert not shock.direction_is_certain
    assert not shock.confidently_wrong


# --------------------------------------------------------------------------- #
# Model quality
# --------------------------------------------------------------------------- #
def test_a_model_explaining_almost_nothing_says_so():
    """A scenario built on an R-squared of 5% describes 5% of the outcome."""
    rng = np.random.default_rng(10)
    rates = rng.standard_normal(1500) * 0.0005
    asset = -1.0 * rates + rng.standard_normal(1500) * 0.01

    response = estimate_response(asset, {"rates": rates})
    assert response.r_squared < 0.10
    assert any("not in the model" in note for note in response.notes)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_misaligned_factors_are_refused_with_the_offending_name():
    rng = np.random.default_rng(11)
    asset = rng.standard_normal(500)
    with pytest.raises(ValueError, match="'oil' has 400 observations"):
        estimate_response(asset, {"oil": rng.standard_normal(400)})


def test_too_few_observations_for_a_meaningful_interval_are_refused():
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="at least 60"):
        estimate_response(rng.standard_normal(50), {"rates": rng.standard_normal(50)})


def test_no_factors_is_refused():
    with pytest.raises(ValueError, match="at least one factor"):
        estimate_response(np.zeros(200), {})


def test_shocking_a_factor_that_was_never_estimated_is_refused():
    """Silently returning zero would read as 'no effect' rather than 'not modelled'."""
    rng = np.random.default_rng(13)
    rates = rng.standard_normal(500) * 0.0005
    asset = -2.0 * rates + rng.standard_normal(500) * 0.001

    response = estimate_response(asset, {"rates": rates})
    with pytest.raises(ValueError, match=r"no beta was estimated for \['oil'\]"):
        apply_shock(response, {"oil": 0.5})


def test_every_named_scenario_has_a_shock_and_a_description():
    assert len(SCENARIOS) >= 5
    for scenario in SCENARIOS:
        assert scenario.shocks, f"{scenario.name} shocks nothing"
        assert len(scenario.description) > 40, f"{scenario.name} is not explained"
        assert all(abs(size) > 0 for size in scenario.shocks.values())
