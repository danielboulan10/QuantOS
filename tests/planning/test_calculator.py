"""Validation for the investment projection.

The deterministic half is checked against Calculator.net's own published worked
example, year by year. An independently published schedule is the strongest
available check on a convention -- there are several plausible ways to compound a
monthly contribution at an annual rate, and they disagree by real money.

The stochastic half is checked against the analytic volatility drag.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.planning.calculator import (
    FREQUENCY,
    investment_schedule,
    required_contribution,
    required_return,
    required_years,
    simulate_plan,
)

# Calculator.net, Investment Calculator: $20,000 start, 10 years, 6%,
# $1,000 additional contribution at the end of each month.
PUBLISHED = {
    1: 33_526.53,
    2: 47_864.65,
    3: 63_063.06,
    4: 79_173.37,
    5: 96_250.30,
    6: 114_351.84,
    7: 133_539.48,
    8: 153_878.38,
    9: 175_437.61,
    10: 198_290.40,
}


# --------------------------------------------------------------------------- #
# Against the published schedule
# --------------------------------------------------------------------------- #
def test_matches_the_published_end_balance_to_the_cent():
    plan = investment_schedule(20_000, 10, 0.06, contribution=1_000)
    assert plan.end_balance == pytest.approx(198_290.40, abs=0.01)
    assert plan.total_contributions == pytest.approx(120_000.00, abs=0.01)
    assert plan.total_interest == pytest.approx(58_290.40, abs=0.01)


def test_matches_every_year_of_the_published_schedule():
    """Not just the endpoint: the whole accumulation path."""
    plan = investment_schedule(20_000, 10, 0.06, contribution=1_000)
    assert len(plan.rows) == 10
    for row in plan.rows:
        assert row.ending_balance == pytest.approx(PUBLISHED[row.year], abs=0.01), (
            f"year {row.year}"
        )


def test_the_periodic_rate_convention_is_the_one_that_matches():
    r"""(1+r)^(1/12)-1, not r/12. The alternatives disagree by real money.

    Reproducing the published first-year interest of 1,526.53 requires the
    effective rate. The nominal convention r/12 gives 1,535.56 and a simple
    pro-rata split gives 1,530.00 -- both plausible, both wrong against a
    published source.
    """
    plan = investment_schedule(20_000, 1, 0.06, contribution=1_000)
    assert plan.total_interest == pytest.approx(1_526.53, abs=0.01)

    nominal = 20_000 * 0.06 + 1_000 * ((1 + 0.06 / 12) ** 12 - 1) / (0.06 / 12) - 12_000
    assert nominal == pytest.approx(1_535.56, abs=0.01)
    assert abs(nominal - plan.total_interest) > 9.0


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def test_contributing_at_the_start_of_the_period_is_worth_more():
    """One extra period of growth on every deposit."""
    end = investment_schedule(20_000, 30, 0.07, contribution=500)
    start = investment_schedule(20_000, 30, 0.07, contribution=500, contribute_at_start=True)
    assert start.end_balance > end.end_balance
    # Small, but real: about half a percent over 30 years.
    assert 1.001 < start.end_balance / end.end_balance < 1.02


def test_a_zero_return_returns_exactly_what_was_paid_in():
    plan = investment_schedule(10_000, 5, 0.0, contribution=100)
    assert plan.end_balance == pytest.approx(10_000 + 5 * 12 * 100)
    assert plan.total_interest == pytest.approx(0.0)


def test_the_balance_grows_with_the_return_and_the_horizon():
    base = investment_schedule(10_000, 10, 0.05, contribution=200).end_balance
    assert investment_schedule(10_000, 10, 0.08, contribution=200).end_balance > base
    assert investment_schedule(10_000, 20, 0.05, contribution=200).end_balance > base


@pytest.mark.parametrize("frequency", sorted(FREQUENCY))
def test_every_frequency_totals_the_contributions_correctly(frequency):
    per_year = FREQUENCY[frequency]
    plan = investment_schedule(0, 3, 0.05, contribution=10, frequency=frequency)
    assert plan.total_contributions == pytest.approx(3 * per_year * 10)


def test_more_frequent_contributions_of_the_same_annual_total_end_higher():
    """Money in earlier compounds longer."""
    annual = investment_schedule(0, 20, 0.06, contribution=12_000, frequency="annually")
    monthly = investment_schedule(0, 20, 0.06, contribution=1_000, frequency="monthly")
    assert monthly.total_contributions == pytest.approx(annual.total_contributions)
    assert monthly.end_balance > annual.end_balance


def test_inflation_is_reported_in_todays_money():
    plan = investment_schedule(20_000, 20, 0.07, contribution=500, inflation=0.025)
    assert any("today's money" in note for note in plan.notes)


@pytest.mark.parametrize("bad", [0, -5])
def test_a_nonpositive_horizon_is_refused(bad):
    with pytest.raises(ValueError, match="years must be positive"):
        investment_schedule(1000, bad, 0.05)


def test_an_unknown_frequency_is_refused():
    with pytest.raises(ValueError, match="frequency must be one of"):
        investment_schedule(1000, 10, 0.05, frequency="fortnightly")


# --------------------------------------------------------------------------- #
# Solving for the other unknowns
# --------------------------------------------------------------------------- #
def test_the_required_return_round_trips():
    target = investment_schedule(20_000, 10, 0.06, contribution=1_000).end_balance
    assert required_return(20_000, 10, target, contribution=1_000) == pytest.approx(0.06, abs=1e-6)


def test_the_required_contribution_round_trips():
    target = investment_schedule(20_000, 10, 0.06, contribution=1_000).end_balance
    assert required_contribution(20_000, 10, 0.06, target) == pytest.approx(1_000.0, abs=0.01)


def test_the_required_horizon_round_trips():
    target = investment_schedule(20_000, 10, 0.06, contribution=1_000).end_balance
    assert required_years(20_000, 0.06, target, contribution=1_000) == pytest.approx(
        10.0, abs=1 / 12
    )


def test_an_unreachable_target_reports_nan_rather_than_a_wrong_answer():
    assert np.isnan(required_return(1_000, 1, 1e12))
    assert np.isnan(required_years(1_000, 0.01, 1e12, max_years=5))


def test_a_target_already_met_needs_no_time():
    assert required_years(50_000, 0.05, 40_000) == 0.0


# --------------------------------------------------------------------------- #
# What the fixed-rate figure leaves out
# --------------------------------------------------------------------------- #
def test_volatility_pushes_the_median_below_the_fixed_rate_projection():
    """The whole reason the stochastic version exists.

    An earlier version of `simulate_plan` computed its drift as
    ``- 0.5*sigma**2 + 0.5*sigma**2``, which cancels to zero and removed the
    effect entirely -- the median came out equal to the deterministic figure.
    """
    outcome = simulate_plan(20_000, 10, 0.06, 0.15, contribution=1_000, n_paths=40_000)
    assert outcome.median < outcome.deterministic
    assert outcome.median_shortfall > 0


def test_the_lump_sum_drag_matches_the_analytic_factor():
    r"""With no contributions the median is exactly :math:`e^{-\sigma^2 T/2}` of the mean."""
    sigma, years = 0.20, 15.0
    outcome = simulate_plan(100_000, years, 0.07, sigma, contribution=0.0, n_paths=120_000)
    expected = np.exp(-0.5 * sigma**2 * years)
    assert outcome.median / outcome.deterministic == pytest.approx(expected, rel=0.02)


def test_contributions_dilute_the_drag():
    """Money added late compounds for less time, so it carries less drag."""
    lump = simulate_plan(200_000, 10, 0.06, 0.18, contribution=0.0, n_paths=40_000)
    with_contributions = simulate_plan(20_000, 10, 0.06, 0.18, contribution=1_500, n_paths=40_000)
    assert (
        with_contributions.median / with_contributions.deterministic
        > lump.median / lump.deterministic
    )


def test_more_volatility_widens_the_spread_and_lowers_the_median():
    calm = simulate_plan(20_000, 20, 0.06, 0.08, contribution=500, n_paths=40_000)
    wild = simulate_plan(20_000, 20, 0.06, 0.30, contribution=500, n_paths=40_000)

    assert wild.median < calm.median
    calm_spread = calm.percentiles[95] / calm.percentiles[5]
    wild_spread = wild.percentiles[95] / wild.percentiles[5]
    assert wild_spread > 2 * calm_spread


def test_the_probability_of_hitting_the_projected_number_is_well_below_certain():
    """The number the calculator prints is not the likely outcome."""
    outcome = simulate_plan(20_000, 10, 0.06, 0.15, contribution=1_000, n_paths=40_000)
    assert 0.30 < outcome.probability_of_target < 0.50
    assert "not the expected outcome" in outcome.verdict


def test_zero_volatility_recovers_the_deterministic_answer():
    """Guards every claim above: with no randomness the two must agree."""
    outcome = simulate_plan(20_000, 10, 0.06, 1e-9, contribution=1_000, n_paths=2_000)
    assert outcome.median == pytest.approx(outcome.deterministic, rel=1e-6)


def test_percentiles_are_ordered():
    outcome = simulate_plan(20_000, 10, 0.06, 0.15, contribution=1_000, n_paths=20_000)
    levels = sorted(outcome.percentiles)
    values = [outcome.percentiles[level] for level in levels]
    assert values == sorted(values)


# --------------------------------------------------------------------------- #
# Gaps found by mutation testing
#
# Every test below exists because `scripts/mutation_test.py --module calculator`
# changed the corresponding line and nothing failed. They are not padding: each
# one covers an output the module actually prints and a user would actually read.
# --------------------------------------------------------------------------- #
def test_the_two_shares_split_the_final_balance_between_growth_and_deposits():
    """The headline sentence in the summary: 'X% is growth, Y% is what you put in'.

    Mutation testing flipped the division in `interest_share` to a multiplication
    and no test noticed, which meant the one figure the summary leads with was
    unchecked.
    """
    plan = investment_schedule(20_000, 10, 0.06, contribution=1_000)

    # The two shares deliberately do NOT sum to one: the starting amount is a
    # third slice. An earlier version of this test asserted they summed to one
    # and failed, which is the right outcome -- money you began with is neither
    # growth nor a deposit you made along the way, and folding it into either
    # would overstate that one. The three do sum to the balance.
    assert plan.contribution_share == pytest.approx(120_000 / plan.end_balance)
    assert plan.interest_share == pytest.approx(plan.total_interest / plan.end_balance)

    starting_share = plan.starting_amount / plan.end_balance
    assert plan.interest_share + plan.contribution_share + starting_share == pytest.approx(1.0)
    assert 0.25 < plan.interest_share < 0.35


def test_an_empty_plan_reports_no_split_rather_than_dividing_by_zero():
    plan = investment_schedule(0.0, 5, 0.06, contribution=0.0)
    assert plan.end_balance == 0.0
    assert np.isnan(plan.interest_share)
    assert np.isnan(plan.contribution_share)


def test_the_first_year_deposit_column_includes_the_starting_amount():
    """Calculator.net counts the opening balance as a year-one deposit.

    This is a presentational choice, not arithmetic -- the ending balances are
    unaffected -- but the deposit column is printed, and copying their layout
    without copying this makes year one look $20,000 short.
    """
    plan = investment_schedule(20_000, 10, 0.06, contribution=1_000)

    assert plan.rows[0].deposits == pytest.approx(20_000 + 12_000)
    for row in plan.rows[1:]:
        assert row.deposits == pytest.approx(12_000)

    total = sum(row.deposits for row in plan.rows)
    assert total == pytest.approx(plan.starting_amount + plan.total_contributions)


def test_annual_compounding_still_attributes_the_starting_amount_to_year_one():
    """The `per_year == 1` branch takes a different path to the same column."""
    plan = investment_schedule(5_000, 3, 0.06, contribution=500, frequency="annually")
    assert plan.rows[0].deposits == pytest.approx(5_500)
    assert sum(r.deposits for r in plan.rows) == pytest.approx(6_500)


def test_daily_compounding_uses_a_365_day_year_and_beats_monthly():
    """More frequent compounding of the same effective annual rate must not change it.

    The rate convention is `(1+r)^(1/m)-1`, so compounding *more often* leaves the
    growth on the starting amount identical -- the extra balance comes only from
    contributions arriving sooner. That is the property worth pinning, because it
    is what distinguishes this convention from the nominal `r/m` one, where daily
    compounding would inflate the return itself.
    """
    lump_daily = investment_schedule(10_000, 5, 0.06, contribution=0.0, frequency="daily")
    lump_annual = investment_schedule(10_000, 5, 0.06, contribution=0.0, frequency="annually")
    assert lump_daily.end_balance == pytest.approx(lump_annual.end_balance, rel=1e-9)
    assert lump_daily.end_balance == pytest.approx(10_000 * 1.06**5, rel=1e-9)

    # With contributions, daily beats annual purely on timing.
    daily = investment_schedule(10_000, 5, 0.06, contribution=10.0, frequency="daily")
    assert daily.rows[-1].year == 5
    assert daily.total_contributions == pytest.approx(10.0 * 365 * 5)


def test_the_inflation_note_deflates_the_balance_and_is_omitted_when_not_asked_for():
    """A 30-year projection in nominal dollars answers the wrong question."""
    plan = investment_schedule(20_000, 30, 0.06, contribution=1_000, inflation=0.03)
    real = plan.end_balance / 1.03**30

    note = [n for n in plan.notes if "today's money" in n]
    assert len(note) == 1
    assert f"{real:,.2f}" in note[0]
    assert real < 0.45 * plan.end_balance, "three decades of 3% more than halves it"

    without = investment_schedule(20_000, 30, 0.06, contribution=1_000)
    assert not any("today's money" in n for n in without.notes)


def test_an_unreachable_target_returns_nan_rather_than_a_silly_rate():
    """No return under 1000% reaches it, so the honest answer is 'no rate does'.

    Returning the bracket endpoint instead would report a confident 1000% and let
    a caller act on it.
    """
    assert np.isnan(required_return(1_000, 5, 1e12, contribution=0.0))


def test_a_target_already_met_at_the_lowest_rate_returns_that_bound():
    """Contributions alone clear it, so no positive return is needed.

    The target has to be below a *single* contribution for this branch to fire.
    A $5,000 target does not qualify, even against a 99% annual loss: the balance
    decays about 32% a month and $1,000 arrives each month, so it settles near
    $3,158 and a real rate of -93.1% solves it exactly. That is the bisection
    working, not the guard, and asserting -0.99 there was simply wrong.
    """
    rate = required_return(10_000, 10, 500, contribution=1_000)
    assert rate == pytest.approx(-0.99)

    solved = required_return(10_000, 10, 5_000, contribution=1_000)
    assert -0.99 < solved < 0.0


def test_a_partial_final_year_counts_only_the_contributions_that_happened():
    """2.5 years is thirty monthly periods, and the last row covers six of them.

    The deposit column for a stub year is computed from the period index rather
    than assumed to be a full year. Mutation testing changed that arithmetic and
    nothing failed, because every other test used whole years.
    """
    plan = investment_schedule(1_000, 2.5, 0.06, contribution=100)

    assert [row.year for row in plan.rows] == [1, 2, 3]
    assert plan.rows[0].deposits == pytest.approx(1_000 + 1_200)
    assert plan.rows[1].deposits == pytest.approx(1_200)
    assert plan.rows[2].deposits == pytest.approx(600), "six months, not twelve"
    assert plan.total_contributions == pytest.approx(3_000)
