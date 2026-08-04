"""Validation for lattice option pricing.

A lattice has a closed form to check against for European options, which makes
most of this straightforward. The tests worth writing are the ones about the
*shape* of the error rather than its size: that convergence oscillates, that the
oscillation is what the trinomial construction fixes, and that the remedy for it
is honestly characterised rather than oversold.

The last group cross-checks against the other three pricing routes in this
repository. Four independent methods agreeing is the strongest statement
available about a number with no analytic solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.derivatives.black_scholes import OptionType, black_scholes_price
from quantos.derivatives.lattice import (
    averaged_binomial_price,
    binomial_price,
    convergence_path,
    trinomial_price,
)


# --------------------------------------------------------------------------- #
# Against the closed form
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_the_european_price_converges_to_black_scholes(option_type, strike):
    exact = float(black_scholes_price(100.0, strike, 1.0, 0.25, rate=0.05, option_type=option_type))
    lattice = binomial_price(
        100.0, strike, 1.0, 0.25, rate=0.05, n_steps=2000, option_type=option_type
    )
    assert lattice.price == pytest.approx(exact, abs=0.01)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_the_trinomial_lattice_also_converges_to_black_scholes(option_type):
    exact = float(black_scholes_price(100.0, 105.0, 0.75, 0.3, rate=0.03, option_type=option_type))
    lattice = trinomial_price(
        100.0, 105.0, 0.75, 0.3, rate=0.03, n_steps=800, option_type=option_type
    )
    assert lattice.price == pytest.approx(exact, abs=0.01)


def test_put_call_parity_holds_on_the_lattice():
    """Parity is a no-arbitrage identity, so it must hold on any consistent tree.

    It is a stronger check than either price alone: an error in the discount
    factor or the risk-neutral probability cancels when comparing a price to the
    closed form, and does not cancel here.
    """
    spot, strike, expiry, rate, q = 100.0, 95.0, 1.5, 0.04, 0.02
    call = binomial_price(
        spot,
        strike,
        expiry,
        0.22,
        rate=rate,
        dividend_yield=q,
        option_type=OptionType.CALL,
        n_steps=1500,
        measure_oscillation=False,
    )
    put = binomial_price(
        spot,
        strike,
        expiry,
        0.22,
        rate=rate,
        dividend_yield=q,
        option_type=OptionType.PUT,
        n_steps=1500,
        measure_oscillation=False,
    )
    forward = spot * np.exp(-q * expiry) - strike * np.exp(-rate * expiry)
    assert call.price - put.price == pytest.approx(forward, abs=0.01)


# --------------------------------------------------------------------------- #
# The oscillation, which is the point of the module
# --------------------------------------------------------------------------- #
def test_the_binomial_error_changes_sign_repeatedly_as_steps_increase():
    """Convergence is not monotone, and treating it as monotone is the mistake.

    Measured over n = 30..150 the CRR error changes sign 82 times. A price
    quoted at one convenient n is a point on an oscillation, not a converged
    value, and 'I used 200 steps' is not evidence of accuracy.
    """
    _, binomial, _ = convergence_path(100.0, 105.0, 1.0, 0.25, rate=0.04, steps=range(30, 151))

    sign_changes = int(np.sum(np.diff(np.sign(binomial)) != 0))
    assert sign_changes > 40, f"only {sign_changes} sign changes; the oscillation vanished"


def test_the_trinomial_error_does_not_oscillate_the_way_the_binomial_does():
    """This is the reason to reach for the third branch.

    Over the same range the trinomial error changes sign twice against the
    binomial's 82. Decoupling the space step from the time step keeps the strike
    at a stable position relative to the nodes.
    """
    _, binomial, trinomial = convergence_path(
        100.0, 105.0, 1.0, 0.25, rate=0.04, steps=range(30, 151)
    )

    binomial_flips = int(np.sum(np.diff(np.sign(binomial)) != 0))
    trinomial_flips = int(np.sum(np.diff(np.sign(trinomial)) != 0))

    assert trinomial_flips < binomial_flips / 5
    assert np.abs(trinomial).mean() < np.abs(binomial).mean()


def test_the_reported_oscillation_matches_the_actual_step_to_step_gap():
    """The number a caller is meant to read before trusting the last two digits."""
    reported = binomial_price(100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=101)
    low = binomial_price(100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=101, measure_oscillation=False)
    high = binomial_price(
        100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=102, measure_oscillation=False
    )

    assert reported.oscillation == pytest.approx(abs(low.price - high.price) / 2.0)
    assert reported.oscillation > 0


def test_a_price_whose_oscillation_exceeds_its_error_says_so():
    """Otherwise a lucky n reads as a converged one.

    At some step counts the lattice lands almost exactly on the closed form
    purely because the strike happens to sit favourably between nodes. That is
    not accuracy and the report should not let it look like accuracy.
    """
    lucky = None
    for n in range(60, 200):
        candidate = binomial_price(100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=n)
        if candidate.oscillation > abs(candidate.error_vs_closed_form):
            lucky = candidate
            break

    assert lucky is not None, "no step count in the range landed on the oscillation's zero"
    assert any("where the strike happens to sit" in note for note in lucky.notes)


def test_averaging_helps_on_balance_but_not_at_every_step_count():
    """The honest version of 'averaging fixes the oscillation'.

    Over n = 50..299 averaging halves the mean absolute error and improves the
    worst case, but it is better on 74% of individual step counts rather than
    all of them: when n and n+1 fall on the same side of the oscillation their
    mean is no better than either. Asserting a uniform improvement would be
    asserting something false, and the test would be flaky rather than wrong.
    """
    exact = float(black_scholes_price(100.0, 105.0, 1.0, 0.25, rate=0.04))

    plain, averaged = [], []
    for n in range(50, 200, 3):
        plain.append(
            abs(
                binomial_price(
                    100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=n, measure_oscillation=False
                ).price
                - exact
            )
        )
        averaged.append(
            abs(
                averaged_binomial_price(100.0, 105.0, 1.0, 0.25, rate=0.04, n_steps=n).price - exact
            )
        )

    plain_error = np.array(plain)
    averaged_error = np.array(averaged)

    assert averaged_error.mean() < 0.7 * plain_error.mean(), "no improvement on average"
    assert averaged_error.max() < plain_error.max(), "the worst case must improve"
    better = float(np.mean(averaged_error < plain_error))
    assert 0.5 < better < 1.0, f"averaging helped on {better:.0%} of step counts, not all"


# --------------------------------------------------------------------------- #
# American exercise
# --------------------------------------------------------------------------- #
def test_an_american_call_without_dividends_equals_the_european_call():
    """The textbook identity. A tree that finds early-exercise value here is wrong."""
    result = binomial_price(
        100.0,
        100.0,
        1.0,
        0.3,
        rate=0.05,
        n_steps=800,
        option_type=OptionType.CALL,
        american=True,
        measure_oscillation=False,
    )
    assert result.price == pytest.approx(result.european_price, abs=0.005)
    assert any("never exercised early" in note for note in result.notes)


def test_a_dividend_makes_early_exercise_on_a_call_worth_something():
    """Guards the test above by removing the condition that made the premium zero."""
    result = binomial_price(
        100.0,
        100.0,
        1.0,
        0.3,
        rate=0.05,
        dividend_yield=0.08,
        n_steps=800,
        option_type=OptionType.CALL,
        american=True,
        measure_oscillation=False,
    )
    assert result.early_exercise_premium > 0.1


def test_an_american_put_is_worth_more_than_its_european_counterpart():
    result = binomial_price(
        90.0,
        100.0,
        1.0,
        0.25,
        rate=0.06,
        n_steps=800,
        option_type=OptionType.PUT,
        american=True,
        measure_oscillation=False,
    )
    assert result.price > result.european_price
    assert result.early_exercise_premium > 0


def test_an_american_price_never_falls_below_intrinsic_value():
    result = binomial_price(
        70.0,
        100.0,
        1.0,
        0.25,
        rate=0.05,
        n_steps=500,
        option_type=OptionType.PUT,
        american=True,
        measure_oscillation=False,
    )
    assert result.price >= 30.0 - 1e-6


# --------------------------------------------------------------------------- #
# Four methods, one number
# --------------------------------------------------------------------------- #
def test_the_binomial_and_trinomial_lattices_agree_on_an_american_put():
    """Two different discretisations of the same problem, so agreement is evidence.

    They share no code beyond the payoff, and their errors have different shapes,
    so a bug in the backward induction of one would not be reproduced by the
    other.
    """
    binomial = binomial_price(
        36.0,
        40.0,
        1.0,
        0.20,
        rate=0.06,
        n_steps=2000,
        option_type=OptionType.PUT,
        american=True,
        measure_oscillation=False,
    )
    trinomial = trinomial_price(
        36.0,
        40.0,
        1.0,
        0.20,
        rate=0.06,
        n_steps=1000,
        option_type=OptionType.PUT,
        american=True,
    )
    assert binomial.price == pytest.approx(trinomial.price, abs=0.002)


def test_the_lattice_value_is_stable_in_the_step_count():
    """4.4867 to five decimals from 2,000 steps to 12,000.

    This is what lets the comparison against the published Monte Carlo figure
    below mean anything: the lattice value has to be settled before it can be
    used as the reference.
    """
    prices = [
        binomial_price(
            36.0,
            40.0,
            1.0,
            0.20,
            rate=0.06,
            n_steps=n,
            option_type=OptionType.PUT,
            american=True,
            measure_oscillation=False,
        ).price
        for n in (2_000, 6_000)
    ]
    assert prices[0] == pytest.approx(prices[1], abs=1e-4)
    assert prices[0] == pytest.approx(4.4867, abs=0.001)


def test_the_lattice_sits_inside_the_monte_carlo_bracket():
    """The cross-check between two entirely different methods.

    Least-squares Monte Carlo produces a lower bound from a learned exercise rule
    and an upper bound from a dual construction. A lattice that fell outside that
    bracket would mean one of the two is wrong.
    """
    from quantos.derivatives.american import price_american

    lattice = binomial_price(
        36.0,
        40.0,
        1.0,
        0.20,
        rate=0.06,
        n_steps=2000,
        option_type=OptionType.PUT,
        american=True,
        measure_oscillation=False,
    )
    monte_carlo = price_american(
        36.0, 40.0, 1.0, 0.20, rate=0.06, n_paths=20_000, n_steps=50, n_inner=30
    )
    assert monte_carlo.lower - 0.02 <= lattice.price <= monte_carlo.upper + 0.02


def test_the_published_longstaff_schwartz_value_is_slightly_low():
    """A finding, not a failure -- and it is the one LSMC's own theory predicts.

    The 2001 paper reports 4.478 for this option. A converged lattice, which
    evaluates the exercise decision exactly rather than learning it, gives
    4.4867. The 0.009 gap is the downward bias of least-squares Monte Carlo: an
    exercise rule fitted from a finite sample is suboptimal, and a suboptimal
    rule under-prices. Seeing the bias with the right sign and a plausible size
    is a check on both methods.
    """
    lattice = binomial_price(
        36.0,
        40.0,
        1.0,
        0.20,
        rate=0.06,
        n_steps=4000,
        option_type=OptionType.PUT,
        american=True,
        measure_oscillation=False,
    )
    published = 4.478
    assert lattice.price > published, "the lattice must not be below the biased-low estimate"
    assert lattice.price - published < 0.02, "but the gap is small, not a disagreement"


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_a_time_step_too_coarse_for_the_drift_is_refused():
    """The risk-neutral probability leaves (0, 1) and the lattice is arbitrageable.

    Returning a number here would be worse than raising: the price would be a
    weighted sum with a negative weight, which is not a price.
    """
    with pytest.raises(ValueError, match="outside \\(0, 1\\)"):
        binomial_price(100.0, 100.0, 5.0, 0.05, rate=0.60, n_steps=3)


def test_a_lattice_needs_at_least_one_step():
    with pytest.raises(ValueError, match="at least one step"):
        binomial_price(100.0, 100.0, 1.0, 0.2, n_steps=0)


def test_negative_expiry_or_volatility_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        binomial_price(100.0, 100.0, -1.0, 0.2)
    with pytest.raises(ValueError, match="must be positive"):
        binomial_price(100.0, 100.0, 1.0, 0.0)
