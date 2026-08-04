"""Validation for yield curve construction and interest rate risk.

Fixed income is unusually well supplied with things that must be exactly true,
so most of these tests are identities rather than tolerances: a bootstrapped
curve reprices its inputs to par, a zero-coupon bond's duration equals its
maturity, compounding the forwards reproduces the spot, and key rate durations
sum to the total. Each of those fails loudly for a whole class of bug that a
plausible-looking number would hide.

The two tests worth reading are the ones that measure a *model's* error rather
than asserting it away: duration against a real reprice, and the Svensson fit
against its own interpretability.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.fixed_income import (
    NelsonSiegel,
    YieldCurve,
    bootstrap_zero_curve,
    convexity,
    duration,
    fit_nelson_siegel,
    fit_svensson,
    key_rate_durations,
    price_bond,
)

#: A realistic upward-sloping curve, close to the US Treasury curve in 2026.
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
PAR_YIELDS = np.array(
    [0.0383, 0.0398, 0.0408, 0.0428, 0.0434, 0.0445, 0.0459, 0.0475, 0.0528, 0.0527]
)


@pytest.fixture(scope="module")
def curve() -> YieldCurve:
    return bootstrap_zero_curve(MATURITIES, PAR_YIELDS)


# --------------------------------------------------------------------------- #
# The bootstrap identity
# --------------------------------------------------------------------------- #
def test_the_bootstrapped_curve_reprices_every_input_bond_to_par(curve):
    """The defining property, and the one the first implementation failed.

    A par yield is by definition the coupon that makes the bond worth 100. If
    the stripped curve does not reproduce that, it is not the curve implied by
    the quotes.

    The naive algebraic bootstrap gets this wrong: a two-year bond pays at 1.5
    years, which has to be interpolated between the one-year point and the
    two-year point still being solved. That circularity repriced the 2-year at
    99.9970 -- a 3bp error, small enough to read as rounding, and it compounds
    along the curve.
    """
    for maturity, par in zip(MATURITIES, PAR_YIELDS, strict=True):
        assert price_bond(curve, maturity, par) == pytest.approx(100.0, abs=1e-9)


def test_a_flat_par_curve_bootstraps_to_a_flat_zero_curve():
    """The one case with a closed form: if every par yield is the same, every
    zero rate is the same continuously compounded equivalent."""
    flat = 0.05
    maturities = np.array([1.0, 2.0, 5.0, 10.0])
    built = bootstrap_zero_curve(maturities, np.full(maturities.size, flat))

    spread = built.zero_rates.max() - built.zero_rates.min()
    assert spread < 1e-6, "a flat par curve cannot imply a sloping zero curve"
    # Semiannual par 5% is a continuously compounded 2*ln(1.025) = 4.939%.
    assert built.zero_rates[0] == pytest.approx(2.0 * np.log(1.025), abs=1e-4)


def test_a_bill_shorter_than_a_coupon_period_pays_at_its_maturity():
    """A one-month bill's cash flow falls in one month, not at the first coupon.

    Putting it at six months made the implied one-month zero rate 22% on real
    data -- wrong enough to notice at one month, and quietly plausible at three.
    """
    built = bootstrap_zero_curve([1 / 12, 0.25, 1.0], [0.0378, 0.0383, 0.0408])

    assert built.zero_rates[0] == pytest.approx(0.0378, abs=5e-4)
    assert price_bond(built, 1 / 12, 0.0378) == pytest.approx(100.0, abs=1e-9)


def test_the_zero_curve_is_not_the_par_curve():
    """The mistake the module exists to prevent -- compared in the same units.

    With an upward-sloping curve the zero rate must sit ABOVE the par yield at
    the long end: a par yield is a weighted average of the zero rates along the
    way, and those are lower early on.

    The first version of this test compared the two raw numbers and failed, and
    the failure was the test's, not the code's. The curve stores *continuously
    compounded* rates while the quotes are *semiannually compounded* par yields,
    and a 4.75% semiannual yield is 4.694% continuous. Comparing them directly
    made the zero look 0.5bp too low when it is in fact 5bp too high -- which is
    precisely the compounding-convention confusion that motivated storing this
    curve continuously in the first place.
    """
    built = bootstrap_zero_curve(MATURITIES, PAR_YIELDS)
    long_zero = float(np.asarray(built.zero_rate(10.0)))
    long_par = float(PAR_YIELDS[MATURITIES == 10.0][0])
    par_continuous = 2.0 * np.log1p(long_par / 2.0)

    assert long_zero > par_continuous, "an upward-sloping curve lifts the zero above the par"
    assert long_zero - par_continuous > 1e-4, "and materially, not by rounding"
    # And the raw comparison, which is the one that misleads.
    assert long_zero < long_par, "the very comparison this docstring warns against"


@pytest.mark.parametrize(
    ("maturities", "yields", "match"),
    [
        ([1.0, 2.0], [0.04], "2 maturities against 1 yields"),
        ([2.0, 1.0], [0.04, 0.05], "strictly increasing"),
        ([0.0, 1.0], [0.04, 0.05], "must be positive"),
        ([], [], "nothing to bootstrap"),
    ],
)
def test_malformed_quotes_are_refused(maturities, yields, match):
    with pytest.raises(ValueError, match=match):
        bootstrap_zero_curve(maturities, yields)


# --------------------------------------------------------------------------- #
# Forwards
# --------------------------------------------------------------------------- #
def test_compounding_the_forwards_reproduces_the_spot_rate(curve):
    r"""The no-arbitrage identity, in continuous time.

    Investing to 2 years must equal investing to 1 and rolling at the 1y1y
    forward: :math:`r_2 \cdot 2 = r_1 \cdot 1 + f_{1,2} \cdot 1`. Any sign or
    ordering slip in the forward formula breaks this immediately.
    """
    for short, long in [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (1.0, 30.0)]:
        forward = curve.forward_rate(short, long)
        r_short = float(np.asarray(curve.zero_rate(short)))
        r_long = float(np.asarray(curve.zero_rate(long)))
        assert r_long * long == pytest.approx(r_short * short + forward * (long - short))


def test_an_upward_sloping_curve_implies_forwards_above_the_spot(curve):
    """Which is why 'the market expects rates to rise', read off a spot curve,
    usually understates what is already priced."""
    assert curve.forward_rate(5.0, 10.0) > float(np.asarray(curve.zero_rate(10.0)))


def test_a_backwards_forward_period_is_refused(curve):
    with pytest.raises(ValueError, match="must be positive"):
        curve.forward_rate(10.0, 5.0)


# --------------------------------------------------------------------------- #
# Inversion, reported by location rather than as one flag
# --------------------------------------------------------------------------- #
def test_an_inversion_is_located_rather_than_merely_flagged():
    """A curve can be inverted at the long end while 2s10s is comfortably
    positive -- the 20s/30s kink is a pension-demand and convexity story, not
    the recession signal. Calling both 'inverted' invites the wrong reading.
    """
    built = YieldCurve(np.array([2.0, 10.0, 20.0, 30.0]), np.array([0.042, 0.047, 0.055, 0.053]))

    assert built.is_inverted
    assert built.slope() > 0, "2s10s is positive here"
    assert built.inversions() == [(20.0, 30.0)]
    assert "downward-sloping between 20y and 30y" in built.summary()


def test_a_genuinely_inverted_front_end_is_reported_as_a_negative_slope():
    built = YieldCurve(np.array([2.0, 5.0, 10.0]), np.array([0.050, 0.046, 0.044]))
    assert built.slope() < 0
    assert "INVERTED" in built.summary()


# --------------------------------------------------------------------------- #
# Risk: duration measured against a real reprice
# --------------------------------------------------------------------------- #
def test_a_zero_coupon_bond_has_duration_equal_to_its_maturity(curve):
    """Exactly, by definition -- there is only one cash flow to weight.

    Any weighting error shows up here immediately and nowhere else so cleanly.
    """
    for maturity in (2.0, 10.0, 30.0):
        macaulay, _ = duration(curve, maturity, coupon_rate=0.0)
        assert macaulay == pytest.approx(maturity, abs=1e-9)


def test_a_coupon_bond_has_duration_below_its_maturity(curve):
    """Coupons return capital early, so the weighted average time is shorter."""
    for maturity in (5.0, 10.0, 30.0):
        macaulay, _ = duration(curve, maturity, coupon_rate=0.05)
        assert 0 < macaulay < maturity


def test_duration_predicts_the_price_move_for_a_small_parallel_shift(curve):
    """The claim duration actually makes, checked against a genuine reprice.

    For a 1bp shift the first-order approximation should be accurate to a
    fraction of a percent of the move itself.
    """
    maturity, coupon, shift = 10.0, 0.045, 0.0001
    base = price_bond(curve, maturity, coupon)
    _, modified = duration(curve, maturity, coupon)

    shifted = YieldCurve(curve.maturities, curve.zero_rates + shift)
    actual = (price_bond(shifted, maturity, coupon) - base) / base
    predicted = -modified * shift

    assert actual == pytest.approx(predicted, rel=0.01)


def test_duration_alone_is_wrong_on_a_large_move_and_convexity_is_the_gap(curve):
    """The test that measures the model's error rather than asserting it away.

    On a 300bp shift in a 30-year bond, duration alone misses by several percent
    of notional. Adding the second-order term recovers most of it. The error is
    also *signed*: duration over-predicts the loss on a rise, which is the
    asymmetry that makes a long-duration position not a symmetric bet.
    """
    maturity, coupon, shift = 30.0, 0.05, 0.03
    base = price_bond(curve, maturity, coupon)
    _, modified = duration(curve, maturity, coupon)
    cx = convexity(curve, maturity, coupon)

    shifted = YieldCurve(curve.maturities, curve.zero_rates + shift)
    actual = (price_bond(shifted, maturity, coupon) - base) / base

    first_order = -modified * shift
    second_order = first_order + 0.5 * cx * shift**2

    assert abs(actual - first_order) > 0.02, "a 300bp move must expose the linearisation"
    assert abs(actual - second_order) < abs(actual - first_order) / 3
    assert actual > first_order, "duration over-predicts the loss; convexity is positive"


def test_convexity_is_positive_and_grows_with_maturity(curve):
    values = [convexity(curve, m, 0.045) for m in (2.0, 10.0, 30.0)]
    assert all(value > 0 for value in values), "an option-free bond is positively convex"
    assert values == sorted(values)


def test_key_rate_durations_sum_to_approximately_the_total(curve):
    """Bumping every point in turn and bumping them all at once must agree.

    This is what makes key rate durations a decomposition rather than a set of
    unrelated numbers -- and it is the check that catches a bump applied as a
    step instead of a tent.
    """
    maturity, coupon = 10.0, 0.045
    _, modified = duration(curve, maturity, coupon)
    partial = key_rate_durations(curve, maturity, coupon)

    assert sum(partial.values()) == pytest.approx(modified, rel=0.02)


def test_key_rate_exposure_concentrates_at_the_bond_s_own_maturity(curve):
    """A ten-year bond is mostly exposed to the ten-year point.

    Parallel duration cannot say this, and the 2022 selloff was a flattening
    rather than a parallel shift -- a book hedged on parallel duration alone was
    hedged against a move that did not happen.
    """
    partial = key_rate_durations(curve, 10.0, 0.045)
    dominant = max(partial, key=lambda point: abs(partial[point]))

    assert dominant == 10.0
    assert partial[10.0] > 0.5 * sum(v for v in partial.values() if v > 0)


# --------------------------------------------------------------------------- #
# Parametric fits
# --------------------------------------------------------------------------- #
def test_nelson_siegel_recovers_parameters_it_generated():
    """Fit a curve the model produced exactly, and the fit must find it back."""
    truth = NelsonSiegel(level=0.055, slope=-0.02, curvature=0.015, tau=2.5)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)

    fitted = fit_nelson_siegel(maturities, np.asarray(truth(maturities)))

    # Not exact: tau is found on a 220-point grid, so it lands within grid
    # resolution of the truth (2.5198 against 2.5) and leaves a residual. That
    # residual is 0.06bp, which is the price of a deterministic fit with no
    # starting value to tune -- a worthwhile trade, and stated rather than
    # hidden behind a loose tolerance.
    assert fitted.rmse < 1e-5
    assert fitted.level == pytest.approx(truth.level, abs=1e-3)
    assert fitted.slope == pytest.approx(truth.slope, abs=1e-3)
    assert fitted.tau == pytest.approx(truth.tau, rel=0.1)


def test_the_three_factors_read_as_level_slope_and_curvature():
    """Diebold and Li's interpretation, which is what makes the model useful
    rather than merely well-fitting."""
    fitted = fit_nelson_siegel(MATURITIES, PAR_YIELDS)

    # Level is the asymptote; short rate is level + slope.
    assert fitted.long_rate == pytest.approx(fitted.level)
    assert fitted.short_rate == pytest.approx(fitted.level + fitted.slope)
    # An upward-sloping curve has a negative slope coefficient by this sign
    # convention, because the loading decays from 1 at the short end to 0.
    assert fitted.slope < 0
    assert fitted.short_rate < fitted.long_rate


def test_the_fit_is_deterministic_because_tau_is_gridded_not_optimised():
    """Throwing all four parameters at a non-linear optimiser gives an answer
    that depends on the starting point. Profiling out the linear betas removes
    the starting value entirely."""
    first = fit_nelson_siegel(MATURITIES, PAR_YIELDS)
    second = fit_nelson_siegel(MATURITIES, PAR_YIELDS)

    assert first == second


def test_svensson_fits_at_least_as_well_as_nelson_siegel():
    """It nests Nelson-Siegel, so a worse in-sample fit would mean the optimiser
    failed rather than that the model is worse."""
    ns = fit_nelson_siegel(MATURITIES, PAR_YIELDS)
    sv = fit_svensson(MATURITIES, PAR_YIELDS)

    assert sv.rmse <= ns.rmse + 1e-9


def test_the_second_hump_earns_its_place_on_a_real_curve():
    """On the 2026 Treasury curve the extra term takes the fit from ~8bp to
    ~2bp. One hump has to choose between matching the belly and matching the
    20-30 year sector, which is why central banks publish Svensson."""
    ns = fit_nelson_siegel(MATURITIES, PAR_YIELDS)
    sv = fit_svensson(MATURITIES, PAR_YIELDS)

    assert ns.rmse > 4e-4, "the single hump should struggle here"
    assert sv.rmse < ns.rmse / 2


def test_svensson_fits_better_but_its_level_stops_being_interpretable():
    """A finding, and the reason both models are kept rather than only the best.

    The Svensson level parameter is the asymptotic long rate. On a real curve
    the fit drives it to a value nowhere near any observed yield -- the extra
    loadings absorb the shape and the betas trade off against each other -- so
    the model that fits better is the model whose parameters mean less.

    Nelson-Siegel's level stays inside the observed range and can be read as
    "the long rate". Choose on which property you need: fit, or interpretation.
    """
    ns = fit_nelson_siegel(MATURITIES, PAR_YIELDS)
    sv = fit_svensson(MATURITIES, PAR_YIELDS)

    low, high = PAR_YIELDS.min(), PAR_YIELDS.max()
    assert low - 0.01 <= ns.level <= high + 0.01, "NS level stays interpretable"
    assert sv.rmse < ns.rmse, "and Svensson still fits better"
    assert not (low <= sv.level <= high), (
        "the documented pathology: the better fit has an uninterpretable level"
    )


def test_a_fit_with_too_few_points_to_be_meaningful_is_refused():
    with pytest.raises(ValueError, match="exact and meaningless"):
        fit_nelson_siegel([1.0, 5.0], [0.04, 0.045])
    with pytest.raises(ValueError, match="four linear parameters"):
        fit_svensson([1.0, 5.0, 10.0], [0.04, 0.045, 0.047])


def test_a_fitted_curve_can_be_evaluated_where_nothing_was_quoted():
    """The reason to fit at all: a bootstrap cannot answer for 4.3 years."""
    fitted = fit_nelson_siegel(MATURITIES, PAR_YIELDS)
    value = float(np.asarray(fitted(4.3)))

    assert 0.03 < value < 0.06
    neighbours = [float(np.asarray(fitted(m))) for m in (3.0, 5.0)]
    assert min(neighbours) - 0.002 <= value <= max(neighbours) + 0.002
