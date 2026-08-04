"""Validation for date alignment and factor differencing.

This module exists because two of this repository's worst bugs lived in
alignment, and both were silent. The tests are therefore mostly about the two
things that go wrong quietly: aligning in the direction that truncates, and
differencing in the units that are off by 100x.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.data.align import (
    MACRO_FACTORS,
    FactorKind,
    align_to_grid,
    coverage,
    factor_changes,
    simple_returns,
)


def dates(*days: str) -> np.ndarray:
    return np.array(days, dtype="datetime64[D]")


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def test_simple_returns_pads_so_the_result_lines_up_with_its_dates():
    """Seven call sites computed this inline and two disagreed on the length.

    A returns array one shorter than its date array is an off-by-one waiting for
    the first caller who zips them.
    """
    prices = [100.0, 110.0, 99.0]
    padded = simple_returns(prices)

    assert padded.size == len(prices)
    assert padded[0] == 0.0
    assert padded[1] == pytest.approx(0.10)
    assert padded[2] == pytest.approx(-0.10)

    assert simple_returns(prices, pad=False).size == len(prices) - 1


def test_simple_returns_handles_series_too_short_to_have_any():
    assert simple_returns([100.0]).tolist() == [0.0]
    assert simple_returns([100.0], pad=False).size == 0
    assert simple_returns([]).size == 0


def test_simple_returns_does_not_raise_on_a_zero_price():
    """A zero in a price series is bad data, not a crash.

    Raising here would take down a whole report over one malformed row; a NaN
    propagates and is visible to the finite-value masks every caller applies.
    """
    result = simple_returns([100.0, 0.0, 50.0])
    assert result[1] == pytest.approx(-1.0)
    assert not np.isfinite(result[2])


# --------------------------------------------------------------------------- #
# Alignment direction, which is the bug that cost 1,400 observations
# --------------------------------------------------------------------------- #
def test_the_grid_is_never_shortened_to_match_a_shorter_source():
    """The rule. Aligning the other way cut 2,151 observations to 747 once.

    The instrument's dates are the grid. A macro series missing on some of them
    contributes NaN there and nothing else changes.
    """
    grid = dates("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")
    source = dates("2024-01-02", "2024-01-03")

    aligned = align_to_grid(grid, source, [4.0, 4.5])

    assert aligned.size == grid.size, "the grid must survive intact"
    assert np.isnan(aligned[0]) and np.isnan(aligned[3])
    assert aligned[1] == 4.0
    assert aligned[2] == 4.5


def test_a_source_extending_beyond_the_grid_contributes_only_its_overlap():
    grid = dates("2024-01-02", "2024-01-03")
    source = dates("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")

    aligned = align_to_grid(grid, source, [1.0, 2.0, 3.0, 4.0])
    assert aligned.tolist() == [2.0, 3.0]


def test_dates_are_matched_exactly_rather_than_interpolated():
    """A nearest-date or interpolating join invents observations on days the
    source did not trade, which is how a market holiday becomes a data point."""
    grid = dates("2024-01-01", "2024-01-02", "2024-01-03")
    source = dates("2024-01-01", "2024-01-03")

    aligned = align_to_grid(grid, source, [10.0, 20.0])
    assert np.isnan(aligned[1]), "the gap must stay a gap, not become 15.0"


def test_misaligned_source_dates_and_values_are_refused():
    grid = dates("2024-01-01")
    with pytest.raises(ValueError, match="cannot be aligned, only guessed at"):
        align_to_grid(grid, dates("2024-01-01", "2024-01-02"), [1.0])


def test_coverage_reports_the_fraction_of_the_grid_that_is_present():
    """A factor present on 40% of an instrument's days is a subsample, not a
    factor, and a regression on it answers a question about different dates."""
    grid = dates("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")

    assert coverage(grid, grid) == 1.0
    assert coverage(grid, dates("2024-01-01", "2024-01-03")) == 0.5
    assert coverage(grid, dates("2030-01-01")) == 0.0
    assert coverage(dates(), grid) == 0.0


# --------------------------------------------------------------------------- #
# Units, which is the 100x error
# --------------------------------------------------------------------------- #
def test_a_yield_differences_absolutely_and_lands_in_shock_units():
    """4.25% to 4.35% is a 10bp move, which must come back as 0.001.

    A 100bp shock is 0.01 in these units, so a beta multiplied by 0.01 is the
    response to 100bp. Differencing a yield *relatively* instead would give
    0.0235 for the same move -- roughly 20x, and small enough to look fine.
    """
    grid = dates("2024-01-01", "2024-01-02")
    change = factor_changes(grid, grid, [4.25, 4.35], FactorKind.YIELD)

    assert np.isnan(change[0]), "the first change is undefined, not zero"
    assert change[1] == pytest.approx(0.001)


def test_a_level_differences_relatively():
    grid = dates("2024-01-01", "2024-01-02")
    change = factor_changes(grid, grid, [80.0, 88.0], FactorKind.LEVEL)
    assert change[1] == pytest.approx(0.10)


def test_the_two_kinds_disagree_by_the_factor_that_makes_this_worth_encoding():
    """The same numbers under the two conventions, side by side.

    This is the comparison that justifies FactorKind existing at all rather than
    leaving each call site to remember which series is which.
    """
    grid = dates("2024-01-01", "2024-01-02")
    values = [4.25, 4.35]

    as_yield = factor_changes(grid, grid, values, FactorKind.YIELD)[1]
    as_level = factor_changes(grid, grid, values, FactorKind.LEVEL)[1]

    assert as_yield == pytest.approx(0.001)
    assert as_level == pytest.approx(0.0235, abs=1e-4)
    assert as_level / as_yield > 20


def test_a_gap_in_the_source_produces_a_gap_in_the_changes():
    """Not a change computed across the gap, which would be a two-day move
    reported as a one-day move."""
    grid = dates("2024-01-01", "2024-01-02", "2024-01-03")
    change = factor_changes(grid, dates("2024-01-01", "2024-01-03"), [4.0, 4.5], FactorKind.YIELD)

    assert np.isnan(change[0])
    assert np.isnan(change[1]), "the missing day has no change"
    assert np.isnan(change[2]), "and neither does the day after it"


def test_a_single_observation_cannot_produce_a_change():
    grid = dates("2024-01-01")
    change = factor_changes(grid, grid, [4.0], FactorKind.YIELD)
    assert change.size == 1
    assert np.isnan(change[0])


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
def test_every_registered_factor_declares_its_kind_and_explains_itself():
    """The registry is the single place the yield/level choice is made.

    A factor added without a kind would fall back to whatever the caller
    assumed, which is the situation this module was built to remove.
    """
    assert set(MACRO_FACTORS) == {"rates", "credit", "oil", "dollar"}

    for name, factor in MACRO_FACTORS.items():
        assert factor.name == name, "the key must match the factor's own name"
        assert factor.series_id, f"{name} has no series identifier"
        assert isinstance(factor.kind, FactorKind)
        assert len(factor.description) > 40, f"{name} is not explained"


def test_rates_and_credit_are_yields_and_oil_and_the_dollar_are_levels():
    """Regression test for the specific assignments, since getting one wrong is
    a silent 100x error in whichever scenario uses it."""
    assert MACRO_FACTORS["rates"].kind is FactorKind.YIELD
    assert MACRO_FACTORS["credit"].kind is FactorKind.YIELD
    assert MACRO_FACTORS["oil"].kind is FactorKind.LEVEL
    assert MACRO_FACTORS["dollar"].kind is FactorKind.LEVEL
