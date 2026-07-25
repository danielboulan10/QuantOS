"""Validate ``quantos.core.special`` against two independent oracles.

Oracles
-------
1. :mod:`scipy.special` -- vectorised, the primary comparison.
2. :mod:`math` -- CPython's own C implementations, scalar. An independent second
   opinion for the functions it provides, which guards against the possibility
   that SciPy and QuantOS share a mistaken convention.

Metric
------
Relative error, computed only where it is meaningful:

* **Root neighbourhoods are excluded.** ``erf``, ``ndtri``, ``digamma`` and
  ``log_ndtr`` all cross zero, and relative error diverges there by construction
  while absolute error is at machine epsilon. Tests near roots assert on absolute
  error instead.
* **Subnormal results are excluded.** SciPy underflows ``erfc`` to ``0.0`` around
  ``x = 26``, where the continued fraction here returns the correct
  ``1.46e-311``. Comparing against the oracle there would penalise the more
  accurate implementation.

Both exclusions are stated because a hidden tolerance is how a numerical library
quietly stops being accurate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special as sp

from quantos.core import special as qs

# Sampling grids reused across tests.
FINE = np.concatenate(
    [
        np.linspace(-6.0, 6.0, 4001),
        np.linspace(-40.0, -6.0, 1001),
        np.linspace(6.0, 40.0, 1001),
        np.linspace(0.5, 3.5, 2001),  # dense across the erfc branch point
    ]
)
PROBABILITIES = np.concatenate(
    [
        np.linspace(1e-15, 1.0 - 1e-15, 5001),
        10.0 ** np.linspace(-300, -16, 200),
        1.0 - 10.0 ** np.linspace(-16, -1, 200),
    ]
)


def max_relative_error(
    actual: np.ndarray, expected: np.ndarray, *, root_guard: float = 0.0
) -> float:
    """Worst relative error, excluding subnormals and root neighbourhoods."""
    a = np.asarray(actual, dtype=float).ravel()
    b = np.asarray(expected, dtype=float).ravel()
    usable = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-290) & (np.abs(b) > root_guard)
    if not np.any(usable):
        return 0.0
    return float(np.max(np.abs(a[usable] - b[usable]) / np.abs(b[usable])))


# --------------------------------------------------------------------------- #
# Error function family
# --------------------------------------------------------------------------- #
def test_erf_matches_scipy() -> None:
    assert max_relative_error(qs.erf(FINE), sp.erf(FINE), root_guard=1e-8) < 1e-13


def test_erfc_matches_scipy_outside_subnormals() -> None:
    grid = FINE[FINE < 26.0]
    assert max_relative_error(qs.erfc(grid), sp.erfc(grid)) < 1e-13


def test_erfc_beats_scipy_in_the_far_tail() -> None:
    """QuantOS returns a correct subnormal where SciPy underflows to zero.

    Not a curiosity: this is the regime VaR and p-values live in, and it is the
    payoff from selecting erfc's branch points by cancellation analysis.
    """
    x = 26.68
    ours = float(qs.erfc(x))
    assert sp.erfc(x) == 0.0, "SciPy's behaviour changed; revisit this test"
    assert 0.0 < ours < 1e-300
    # Cross-check the magnitude against the asymptotic form exp(-x^2)/(x*sqrt(pi)).
    asymptotic = math.exp(-x * x) / (x * math.sqrt(math.pi))
    assert abs(ours - asymptotic) / asymptotic < 1e-3


def test_erf_matches_stdlib_math() -> None:
    """Second oracle: CPython's own erf, scalar."""
    for x in (-3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5, 5.0):
        assert abs(float(qs.erf(x)) - math.erf(x)) < 1e-14
        assert abs(float(qs.erfc(x)) - math.erfc(x)) <= 1e-14 * max(math.erfc(x), 1e-16)


def test_erfc_reflection_identity() -> None:
    """``erfc(-x) == 2 - erfc(x)`` must hold exactly in structure."""
    x = np.linspace(0.01, 8.0, 500)
    assert np.allclose(qs.erfc(-x), 2.0 - qs.erfc(x), rtol=1e-13, atol=0.0)


def test_erf_special_values() -> None:
    assert float(qs.erf(0.0)) == 0.0
    assert float(qs.erf(np.inf)) == 1.0
    assert float(qs.erf(-np.inf)) == -1.0
    assert float(qs.erfc(np.inf)) == 0.0
    assert float(qs.erfc(-np.inf)) == 2.0
    assert math.isnan(float(qs.erfc(np.nan)))


# --------------------------------------------------------------------------- #
# Normal distribution
# --------------------------------------------------------------------------- #
def test_ndtr_matches_scipy() -> None:
    assert max_relative_error(qs.ndtr(FINE), sp.ndtr(FINE)) < 1e-12


def test_log_ndtr_matches_scipy() -> None:
    assert max_relative_error(qs.log_ndtr(FINE), sp.log_ndtr(FINE), root_guard=1e-8) < 1e-12


def test_log_ndtr_is_accurate_where_ndtr_saturates() -> None:
    """``log(ndtr(x))`` loses all relative precision for x > 6; log_ndtr does not.

    At x = 8, Phi(x) = 1 - 6.7e-16, whose float64 neighbours are 1.1e-16 apart.
    The naive route therefore carries ~17% relative error in the log, even though
    it does not round to exactly zero. The log1p form is exact.
    """
    x = 8.0
    naive = math.log(float(qs.ndtr(x)))
    exact = sp.log_ndtr(x)
    naive_error = abs(naive - exact) / abs(exact)
    ours_error = abs(float(qs.log_ndtr(x)) - exact) / abs(exact)
    assert naive_error > 1e-2, "ndtr precision improved; revisit this test"
    assert ours_error < 1e-13
    assert ours_error < naive_error / 1e10


def test_ndtri_matches_scipy_away_from_its_root() -> None:
    """Full machine precision wherever the Halley refinement applies (|z| < 30)."""
    refined = PROBABILITIES[np.abs(sp.ndtri(PROBABILITIES)) < 30.0]
    assert max_relative_error(qs.ndtri(refined), sp.ndtri(refined), root_guard=1e-6) < 1e-11


def test_ndtri_extreme_tail_degrades_to_documented_accuracy() -> None:
    """Beyond |z| = 30 the Halley step is skipped (exp(z^2/2) overflows).

    Accuracy there falls back to Acklam's seed, ~1.15e-9 relative -- which the
    module docstring states. This test pins that promise so the degradation
    cannot silently get worse, and only affects p < 1e-197.
    """
    deep = 10.0 ** np.linspace(-300, -200, 100)
    error = max_relative_error(qs.ndtri(deep), sp.ndtri(deep))
    assert error < 2e-9
    assert error > 1e-12, "refinement now reaches the far tail; tighten this test"


def test_ndtri_is_accurate_near_p_equals_half() -> None:
    """Phi^-1 has a root at p=1/2; assert on absolute error there."""
    p = np.linspace(0.5 - 1e-9, 0.5 + 1e-9, 101)
    assert np.max(np.abs(qs.ndtri(p) - sp.ndtri(p))) < 1e-15


def test_ndtri_upper_tail_avoids_cancellation() -> None:
    """The Halley residual must be formed from the upper tail for p > 1/2.

    Computing ``ndtr(x) - p`` when both are within 1e-15 of 1.0 destroys every
    significant digit; the measured error before this was fixed was 9.6e-10.
    """
    p = 1.0 - 10.0 ** np.linspace(-16, -2, 300)
    assert max_relative_error(qs.ndtri(p), sp.ndtri(p)) < 1e-12


def test_ndtri_boundaries() -> None:
    assert float(qs.ndtri(0.0)) == -np.inf
    assert float(qs.ndtri(1.0)) == np.inf
    # Outside [0, 1] there is no quantile, so NaN -- not the boundary's infinity.
    assert math.isnan(float(qs.ndtri(-0.1)))
    assert math.isnan(float(qs.ndtri(1.1)))
    assert math.isnan(float(qs.ndtri(np.nan)))


def test_ndtr_and_ndtri_round_trip() -> None:
    """Round-trip accuracy is limited by float64 spacing near Phi = 1, not by us.

    At x = 6, Phi(x) = 1 - 9.9e-10 and neighbouring float64 values are 1.1e-16
    apart, so the *input* to ndtri carries only ~7 significant digits of tail
    information. A tighter tolerance here would be asserting something arithmetic
    cannot deliver; the left tail, where no saturation occurs, is exact.
    """
    x = np.linspace(-6.0, 6.0, 2001)
    assert np.max(np.abs(qs.ndtri(qs.ndtr(x)) - x)) < 1e-7
    left = np.linspace(-30.0, -1.0, 2001)
    assert np.max(np.abs(qs.ndtri(qs.ndtr(left)) - left)) < 1e-9


def test_norm_pdf_matches_scipy() -> None:
    from scipy import stats

    assert max_relative_error(qs.norm_pdf(FINE), stats.norm.pdf(FINE)) < 1e-14


def test_erfinv_matches_scipy() -> None:
    y = np.linspace(-0.99999, 0.99999, 5001)
    assert max_relative_error(qs.erfinv(y), sp.erfinv(y), root_guard=1e-6) < 1e-11


# --------------------------------------------------------------------------- #
# Gamma family
# --------------------------------------------------------------------------- #
GAMMA_GRID = np.concatenate([10.0 ** np.linspace(-300, -1, 400), np.linspace(1e-6, 60.0, 3000)])


def test_lgamma_matches_scipy() -> None:
    assert (
        max_relative_error(qs.lgamma(GAMMA_GRID), sp.gammaln(GAMMA_GRID), root_guard=1e-8) < 1e-10
    )


def test_lgamma_matches_stdlib() -> None:
    for x in (1e-8, 0.5, 1.0, 2.0, 7.5, 100.0, 1e6):
        expected = math.lgamma(x)
        # lgamma(1) and lgamma(2) are exactly zero, so an absolute floor is
        # required; a pure relative tolerance is vacuous against zero.
        assert abs(float(qs.lgamma(x)) - expected) <= 1e-11 * abs(expected) + 1e-14


def test_lgamma_handles_tiny_arguments() -> None:
    """The Lanczos series has a pole at x=0; small arguments must be shifted up.

    Without the recurrence shift, ``x = 1e-300`` divides by zero (because
    ``x - 1`` rounds to exactly ``-1``) and returns NaN with a warning.
    """
    assert abs(float(qs.lgamma(1e-300)) - sp.gammaln(1e-300)) < 1e-9
    assert np.all(np.isfinite(qs.lgamma(10.0 ** np.linspace(-300, -1, 100))))


def test_lgamma_rejects_non_positive() -> None:
    assert math.isnan(float(qs.lgamma(0.0)))
    assert math.isnan(float(qs.lgamma(-1.5)))


def test_digamma_matches_scipy() -> None:
    grid = np.linspace(1e-6, 60.0, 5000)
    assert max_relative_error(qs.digamma(grid), sp.digamma(grid), root_guard=1e-4) < 1e-10


def test_digamma_near_its_root() -> None:
    """psi has a zero at x ~ 1.4616; relative error is meaningless there."""
    grid = np.linspace(1.40, 1.52, 201)
    assert np.max(np.abs(qs.digamma(grid) - sp.digamma(grid))) < 1e-12


def test_digamma_recurrence_identity() -> None:
    r"""``psi(x+1) - psi(x) == 1/x`` exactly, by the functional equation."""
    x = np.linspace(0.1, 20.0, 500)
    assert np.allclose(qs.digamma(x + 1.0) - qs.digamma(x), 1.0 / x, rtol=1e-11)


@pytest.mark.parametrize("a_max,x_max", [(60.0, 120.0), (5.0, 5.0)])
def test_gammainc_matches_scipy(a_max: float, x_max: float) -> None:
    a, x = np.meshgrid(np.linspace(0.1, a_max, 60), np.linspace(0.01, x_max, 60))
    assert max_relative_error(qs.gammainc(a, x), sp.gammainc(a, x)) < 1e-12
    assert max_relative_error(qs.gammaincc(a, x), sp.gammaincc(a, x)) < 1e-12


def test_gammainc_complement_sums_to_one() -> None:
    a, x = np.meshgrid(np.linspace(0.2, 40.0, 40), np.linspace(0.01, 90.0, 40))
    assert np.allclose(qs.gammainc(a, x) + qs.gammaincc(a, x), 1.0, atol=1e-13)


def test_gammaincc_preserves_upper_tail_precision() -> None:
    """The upper tail must come from the continued fraction, not ``1 - P``."""
    a, x = 2.0, 80.0
    direct = float(qs.gammaincc(a, x))
    subtracted = 1.0 - float(qs.gammainc(a, x))
    assert direct > 0.0
    assert subtracted == 0.0, "gammainc no longer saturates; test is obsolete"
    assert abs(direct - sp.gammaincc(a, x)) / sp.gammaincc(a, x) < 1e-12


def test_gammainc_boundaries() -> None:
    assert float(qs.gammainc(2.0, 0.0)) == 0.0
    assert float(qs.gammaincc(2.0, 0.0)) == 1.0
    assert math.isnan(float(qs.gammainc(-1.0, 1.0)))


# --------------------------------------------------------------------------- #
# Beta family
# --------------------------------------------------------------------------- #
def test_betainc_matches_scipy() -> None:
    a, b, x = np.meshgrid(
        np.linspace(0.2, 30.0, 22), np.linspace(0.2, 30.0, 22), np.linspace(0.005, 0.995, 22)
    )
    assert max_relative_error(qs.betainc(a, b, x), sp.betainc(a, b, x)) < 1e-12


def test_betainc_symmetry_identity() -> None:
    r"""``I_x(a,b) == 1 - I_{1-x}(b,a)``."""
    a, b, x = np.meshgrid(
        np.linspace(0.5, 12.0, 12), np.linspace(0.5, 12.0, 12), np.linspace(0.02, 0.98, 12)
    )
    assert np.allclose(qs.betainc(a, b, x), 1.0 - qs.betainc(b, a, 1.0 - x), atol=1e-12)


def test_betainc_boundaries() -> None:
    assert float(qs.betainc(2.0, 3.0, 0.0)) == 0.0
    assert float(qs.betainc(2.0, 3.0, 1.0)) == 1.0
    assert math.isnan(float(qs.betainc(2.0, 3.0, 1.5)))


def test_log_beta_matches_scipy() -> None:
    a, b = np.meshgrid(np.linspace(0.3, 40.0, 50), np.linspace(0.3, 40.0, 50))
    assert max_relative_error(qs.log_beta(a, b), sp.betaln(a, b), root_guard=1e-8) < 1e-11


# --------------------------------------------------------------------------- #
# Shape and dtype contracts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "func",
    [qs.erf, qs.erfc, qs.ndtr, qs.log_ndtr, qs.norm_pdf, qs.lgamma, qs.digamma],
)
def test_preserves_shape(func) -> None:
    for shape in [(), (5,), (3, 4), (2, 3, 4)]:
        out = func(np.full(shape, 1.5))
        assert np.shape(out) == shape
        assert np.asarray(out).dtype == np.float64


def test_accepts_python_scalars_and_lists() -> None:
    assert np.isclose(float(qs.ndtr(0.0)), 0.5)
    assert qs.ndtr([0.0, 1.0]).shape == (2,)
    assert qs.gammainc(2.0, [1.0, 2.0]).shape == (2,)


def test_broadcasting_across_arguments() -> None:
    a = np.array([[1.0], [2.0], [3.0]])
    x = np.array([0.5, 1.0, 1.5, 2.0])
    assert qs.gammainc(a, x).shape == (3, 4)
    assert qs.betainc(a, a, np.full(4, 0.5)).shape == (3, 4)
