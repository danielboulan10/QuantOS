"""Validation for historical stress testing.

The arithmetic here is easy -- drawdowns and correlations are undergraduate
material. What is not easy is refusing to answer, and that is what most of these
tests check. A stress tester that returns a confident number for a crisis the
instrument was not alive for is worse than one that returns nothing, because the
number will be used.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.risk.stress import (
    CRISES,
    correlation_breakdown,
    stress_test,
)

GFC = next(crisis for crisis in CRISES if "financial" in crisis.name)
COVID = next(crisis for crisis in CRISES if "COVID" in crisis.name)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def series(start: str, end: str, *, seed: int = 0, drift: float = 0.0002):
    dates = np.arange(np.datetime64(start), np.datetime64(end), dtype="datetime64[D]")
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(dates.size) * 0.01 + drift
    return dates, 100.0 * np.exp(np.cumsum(steps))


def crash_at(dates, crisis, *, depth: float = -0.5):
    """A price path that falls `depth` across a crisis window and recovers after."""
    prices = np.full(dates.size, 100.0)
    inside = (dates >= crisis.start_date) & (dates <= crisis.end_date)
    n = int(inside.sum())
    prices[inside] = 100.0 * np.linspace(1.0, 1.0 + depth, n)
    after = dates > crisis.end_date
    if after.any():
        prices[after] = np.linspace(prices[inside][-1], 130.0, int(after.sum()))
    return prices


# --------------------------------------------------------------------------- #
# Coverage, which is the whole point
# --------------------------------------------------------------------------- #
def test_a_window_the_history_does_not_reach_is_reported_not_dropped():
    """An omitted window reads as a window that was survived.

    Silently returning four results instead of five is the failure mode: the
    reader counts four green rows and concludes the portfolio is robust.
    """
    dates, prices = series("2010-01-01", "2024-01-01")
    report = stress_test(dates, prices)

    assert len(report.results) == len(CRISES)
    dotcom = next(r for r in report.results if "dot-com" in r.crisis.name)
    gfc = next(r for r in report.results if r.crisis is GFC)

    assert not dotcom.covered and not gfc.covered
    assert "does not reach" in dotcom.reason
    assert "NOT TESTABLE" in report.summary()


def test_an_instrument_listed_midway_through_a_crisis_is_refused():
    """Partial coverage is more dangerous than none, because it looks like a result.

    An instrument that began trading in November 2008 has data inside the GFC
    window. Testing it would report a modest drawdown -- it missed the collapse
    -- and that number would be indistinguishable from a genuine one.
    """
    dates, prices = series("2008-11-01", "2015-01-01")
    report = stress_test(dates, prices)

    gfc = next(r for r in report.results if r.crisis is GFC)
    assert not gfc.covered
    assert gfc.coverage > 0.1, "there IS data in the window, which is the trap"
    assert "after the crisis started" in gfc.reason
    assert "would miss the decline" in gfc.reason
    assert "Partial coverage is worse than none" in report.summary()


def test_a_history_starting_just_inside_the_window_is_allowed_by_the_grace_period():
    """A listing days before the peak is a boundary case, not a disqualification."""
    dates, prices = series("2007-10-15", "2012-01-01")  # 6 days after the peak
    report = stress_test(dates, prices)
    assert next(r for r in report.results if r.crisis is GFC).covered


def test_asking_for_the_worst_case_with_nothing_tested_raises():
    """A worst case derived from no data is the most misleading output available."""
    dates, prices = series("2024-01-01", "2026-01-01")
    report = stress_test(dates, prices)

    assert report.tested == []
    with pytest.raises(ValueError, match="no worst case"):
        _ = report.worst
    assert "Nothing testable" in report.summary()


# --------------------------------------------------------------------------- #
# The measurements
# --------------------------------------------------------------------------- #
def test_a_known_drawdown_is_recovered_to_the_percent():
    dates = np.arange(
        np.datetime64("2006-01-01"), np.datetime64("2012-01-01"), dtype="datetime64[D]"
    )
    prices = crash_at(dates, GFC, depth=-0.40)

    result = next(r for r in stress_test(dates, prices).results if r.crisis is GFC)
    assert result.covered
    assert result.max_drawdown == pytest.approx(-0.40, abs=0.01)
    assert result.total_return == pytest.approx(-0.40, abs=0.01)


def test_recovery_is_none_when_the_price_never_regains_its_high():
    """Reporting a recovery time for a position still under water would be a lie."""
    dates = np.arange(
        np.datetime64("2006-01-01"), np.datetime64("2012-01-01"), dtype="datetime64[D]"
    )
    prices = crash_at(dates, GFC, depth=-0.60)
    prices[dates > GFC.end_date] = 45.0  # flat, well below the pre-crisis 100

    result = next(r for r in stress_test(dates, prices).results if r.crisis is GFC)
    assert result.days_to_recover is None
    assert "never recovered" in result.summary()


def test_recovery_is_counted_from_the_trough_and_may_land_outside_the_window():
    """Cutting the search at the window edge would report 'never' for every crisis.

    Every crisis in the list ends at its trough, so a recovery by construction
    happens afterwards.
    """
    dates = np.arange(
        np.datetime64("2006-01-01"), np.datetime64("2014-01-01"), dtype="datetime64[D]"
    )
    prices = crash_at(dates, GFC, depth=-0.50)

    result = next(r for r in stress_test(dates, prices).results if r.crisis is GFC)
    assert result.days_to_recover is not None
    assert result.days_to_recover > 0


def test_the_volatility_multiple_compares_the_crisis_against_the_calm_before_it():
    """The number that says 'this was not a normal period', measured not assumed."""
    dates = np.arange(
        np.datetime64("2006-01-01"), np.datetime64("2010-01-01"), dtype="datetime64[D]"
    )
    rng = np.random.default_rng(1)
    steps = rng.standard_normal(dates.size) * 0.004
    inside = (dates >= GFC.start_date) & (dates <= GFC.end_date)
    steps[inside] *= 5.0

    prices = 100.0 * np.exp(np.cumsum(steps))
    result = next(r for r in stress_test(dates, prices).results if r.crisis is GFC)

    assert result.volatility_multiple == pytest.approx(5.0, rel=0.25)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_misaligned_dates_and_prices_are_refused():
    dates, prices = series("2010-01-01", "2015-01-01")
    with pytest.raises(ValueError, match="must align"):
        stress_test(dates, prices[:-5])


def test_unsorted_dates_are_refused():
    dates, prices = series("2010-01-01", "2015-01-01")
    scrambled = dates.copy()
    scrambled[10], scrambled[900] = dates[900], dates[10]
    with pytest.raises(ValueError, match="ascending"):
        stress_test(scrambled, prices)


def test_a_history_too_short_to_stress_test_is_refused():
    dates, prices = series("2010-01-01", "2010-01-20")
    with pytest.raises(ValueError, match="fewer than 30"):
        stress_test(dates, prices)


# --------------------------------------------------------------------------- #
# Correlation, and the statistic that hid the answer
# --------------------------------------------------------------------------- #
def _two_group_returns(n: int, *, crisis_mask, seed: int = 4):
    """Two equities that converge in the crisis, and a bond that hedges harder.

    Built to reproduce the real pattern: within-group correlation rises while
    cross-group correlation falls, so their average barely moves.
    """
    rng = np.random.default_rng(seed)
    market = rng.standard_normal(n) * 0.01
    idio_a = rng.standard_normal(n) * 0.01
    idio_b = rng.standard_normal(n) * 0.01

    # In the crisis the common factor dominates both equities.
    weight = np.where(crisis_mask, 3.0, 0.6)
    equity_a = weight * market + idio_a
    equity_b = weight * market + idio_b
    # The bond moves against the market, and more strongly in the crisis.
    bond = np.where(crisis_mask, -2.0, -0.2) * market + rng.standard_normal(n) * 0.01
    return {"EQ_A": equity_a, "EQ_B": equity_b, "BOND": bond}


def test_the_decomposition_separates_effects_that_a_single_average_nets_out():
    """The defect this function was rewritten to fix.

    The first version reported one mean pairwise correlation. On real data
    through the GFC that number FELL, and the function concluded the assets had
    diversified -- the opposite of the risk being looked for. Equity-equity
    correlation rose from +0.84 to +0.90 while equity-bond fell from +0.05 to
    -0.20, and averaging them produced a summary of nothing.
    """
    dates = np.arange(
        np.datetime64("2005-01-01"), np.datetime64("2010-01-01"), dtype="datetime64[D]"
    )
    inside = (dates >= GFC.start_date) & (dates <= GFC.end_date)
    returns = _two_group_returns(dates.size, crisis_mask=inside)

    result = correlation_breakdown(dates, returns, GFC, risk_assets={"EQ_A", "EQ_B"})

    assert result.within_risk_stressed > result.within_risk_calm, (
        "risk assets must converge, which is where the concentration is"
    )
    assert result.cross_stressed < result.cross_calm, "the hedge held harder"

    # The naive statistic the first version used, recomputed here to show it
    # would have missed the finding entirely.
    naive_calm = float(np.mean([pair.calm for pair in result.pairs]))
    naive_crisis = float(np.mean([pair.stressed for pair in result.pairs]))
    assert abs(naive_crisis - naive_calm) < abs(
        result.within_risk_stressed - result.within_risk_calm
    ), "the average understates the move it is meant to reveal"


def test_a_hedge_that_inverts_is_flagged_as_a_sign_flip():
    """2022: SPY-TLT went from -0.40 to +0.03 while TLT fell more than SPY.

    A hedge that weakens is a smaller offset. A hedge that inverts is a position
    sized as though it reduced the risk it was in fact adding to.
    """
    dates = np.arange(
        np.datetime64("2018-01-01"), np.datetime64("2023-06-01"), dtype="datetime64[D]"
    )
    shock = next(c for c in CRISES if "2022" in c.name)
    inside = (dates >= shock.start_date) & (dates <= shock.end_date)

    rng = np.random.default_rng(9)
    market = rng.standard_normal(dates.size) * 0.01
    # The bond hedges in calm and moves WITH equities during the shock.
    bond = np.where(inside, 0.9, -0.6) * market + rng.standard_normal(dates.size) * 0.003

    result = correlation_breakdown(dates, {"EQUITY": market, "BOND": bond}, shock)

    assert len(result.sign_flips) == 1
    assert "SIGN FLIP" in result.summary()
    assert "worse than one that weakens" in result.summary()


def test_no_group_tagging_means_no_invented_decomposition():
    """Guessing which asset is meant to be the hedge would be inventing the answer."""
    dates = np.arange(
        np.datetime64("2005-01-01"), np.datetime64("2010-01-01"), dtype="datetime64[D]"
    )
    inside = (dates >= GFC.start_date) & (dates <= GFC.end_date)
    result = correlation_breakdown(dates, _two_group_returns(dates.size, crisis_mask=inside), GFC)

    assert result.pairs, "the per-pair table is always available"
    assert np.isnan(result.within_risk_calm)
    assert np.isnan(result.cross_calm)


def test_correlation_needs_data_on_both_sides_of_the_window():
    dates = np.arange(
        np.datetime64("2020-03-01"), np.datetime64("2020-04-01"), dtype="datetime64[D]"
    )
    rng = np.random.default_rng(0)
    returns = {"A": rng.standard_normal(dates.size), "B": rng.standard_normal(dates.size)}

    result = correlation_breakdown(dates, returns, COVID)
    assert result.pairs == []
    assert "not enough data" in result.summary()


def test_a_single_asset_cannot_have_a_correlation():
    dates = np.arange(
        np.datetime64("2005-01-01"), np.datetime64("2010-01-01"), dtype="datetime64[D]"
    )
    with pytest.raises(ValueError, match="at least two"):
        correlation_breakdown(dates, {"ONLY": np.zeros(dates.size)}, GFC)


# --------------------------------------------------------------------------- #
# Gaps found by mutation testing
# --------------------------------------------------------------------------- #
def test_the_worst_day_is_dated_to_the_day_the_loss_happened():
    """An off-by-one here attributes the crash to the wrong day.

    Returns are differences, so return[i] spans dates[i] to dates[i+1] and the
    loss belongs to the later date. Mutation testing removed the +1 and nothing
    failed, which meant the one field a reader would check against a news
    archive was unverified.
    """
    dates = np.arange(
        np.datetime64("2006-01-01"), np.datetime64("2012-01-01"), dtype="datetime64[D]"
    )
    prices = np.full(dates.size, 100.0)
    crash_day = np.datetime64("2008-10-15")
    prices[dates >= crash_day] = 70.0  # a single -30% day, then flat

    result = next(r for r in stress_test(dates, prices).results if r.crisis is GFC)

    assert result.worst_day == pytest.approx(-0.30, abs=0.001)
    assert result.worst_day_date == str(crash_day)


def test_a_pair_correlation_reports_the_change_in_the_direction_it_moved():
    from quantos.risk.stress import PairCorrelation

    rose = PairCorrelation("A", "B", calm=0.2, stressed=0.7)
    fell = PairCorrelation("A", "B", calm=0.7, stressed=0.2)

    assert rose.change == pytest.approx(+0.5)
    assert fell.change == pytest.approx(-0.5)


def test_a_sign_flip_is_a_reversal_and_not_merely_a_move_toward_zero():
    """A hedge that weakens is a smaller offset; one that inverts adds to the risk.

    The distinction has to be exact, because the flag drives the loudest warning
    the module produces. A correlation reaching zero has not yet reversed.
    """
    from quantos.risk.stress import PairCorrelation

    assert PairCorrelation("A", "B", calm=-0.4, stressed=0.1).flipped_sign
    assert PairCorrelation("A", "B", calm=0.3, stressed=-0.2).flipped_sign

    assert not PairCorrelation("A", "B", calm=-0.4, stressed=-0.05).flipped_sign
    assert not PairCorrelation("A", "B", calm=0.6, stressed=0.1).flipped_sign
    assert not PairCorrelation("A", "B", calm=-0.4, stressed=0.0).flipped_sign, (
        "reaching zero is not yet a reversal"
    )


def test_the_grace_period_is_applied_at_its_stated_width():
    """Fourteen days of grace, checked on both sides of the boundary.

    Too narrow and a listing days before the peak is wrongly refused; too wide
    and an instrument that missed weeks of the decline is wrongly accepted.
    """

    def covered_when_starting(start: str) -> bool:
        dates, prices = series(start, "2012-01-01")
        return next(r for r in stress_test(dates, prices).results if r.crisis is GFC).covered

    assert covered_when_starting("2007-10-20"), "11 days in, within grace"
    assert not covered_when_starting("2007-11-01"), "23 days in, beyond grace"


def test_the_minimum_history_guard_is_checked_at_its_boundary():
    dates = np.arange(
        np.datetime64("2010-01-01"), np.datetime64("2010-01-30"), dtype="datetime64[D]"
    )
    prices = np.full(dates.size, 100.0)
    assert dates.size == 29

    with pytest.raises(ValueError, match="fewer than 30"):
        stress_test(dates, prices)

    # One more observation is enough to proceed, even though nothing is testable.
    longer = np.append(dates, np.datetime64("2010-01-30"))
    report = stress_test(longer, np.append(prices, 100.0))
    assert len(report.results) == len(CRISES)


def test_the_convergence_note_needs_a_material_move_not_a_rounding_difference():
    """A note that fires on +0.001 would appear on every report and mean nothing."""
    dates = np.arange(
        np.datetime64("2005-01-01"), np.datetime64("2010-01-01"), dtype="datetime64[D]"
    )
    inside = (dates >= GFC.start_date) & (dates <= GFC.end_date)

    rng = np.random.default_rng(2)
    market = rng.standard_normal(dates.size) * 0.01
    # Identical loadings in both regimes: correlation should not move materially.
    stable = {
        "EQ_A": 0.6 * market + rng.standard_normal(dates.size) * 0.01,
        "EQ_B": 0.6 * market + rng.standard_normal(dates.size) * 0.01,
    }
    result = correlation_breakdown(dates, stable, GFC, risk_assets={"EQ_A", "EQ_B"})
    assert abs(result.within_risk_stressed - result.within_risk_calm) < 0.10
    assert not any("ROSE into the crisis" in note for note in result.notes)

    # And it does fire when the convergence is real.
    converging = _two_group_returns(dates.size, crisis_mask=inside)
    loud = correlation_breakdown(dates, converging, GFC, risk_assets={"EQ_A", "EQ_B"})
    assert any("ROSE into the crisis" in note for note in loud.notes)
