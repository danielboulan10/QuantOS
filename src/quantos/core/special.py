"""Vectorised special functions used throughout QuantOS.

Why this module exists
----------------------
NumPy ships no ``erf``, no incomplete gamma, and no incomplete beta. The
canonical source for those is SciPy, but SciPy is a ~40 MB compiled dependency
and QuantOS commits to a NumPy-only runtime (see ``docs/ddr/DDR-002``). The
Python standard library's :mod:`math` module has scalar ``erf``/``erfc``/
``lgamma``, but scalar loops over a 10-million-element array of returns are not
acceptable in a research platform.

So we implement the handful of special functions the platform actually needs,
vectorised over NumPy arrays, using classical algorithms with well-understood
error behaviour. Every function here is validated in ``tests/core/
test_special.py`` against **two independent oracles**: :mod:`math` (scalar,
CPython's own C implementations) and :mod:`scipy.special`. SciPy is a
*test-only* dependency; it never appears in the runtime import graph.

Accuracy
--------
Measured against ``scipy.special`` by ``tests/core/test_special.py``. The
figures are **worst case over the sampled domain**; the *median* relative error
of every function below is at or under 3e-16, i.e. correctly rounded. Errors
are reported relative, excluding neighbourhoods of each function's roots (where
relative error is not a meaningful quantity) and subnormal results.

==================  =============  ===========================================
Function            Max rel. err   Method
==================  =============  ===========================================
``erf``             1.7e-14        Confluent (positive-term) series
``erfc``            1.4e-14        Series | Lentz continued fraction
``ndtr``            2.2e-13        ``0.5 * erfc(-x/sqrt2)``
``log_ndtr``        9.7e-15        ``log1p`` | Mills-ratio asymptotic series
``ndtri``           3.2e-13        Acklam seed + one Halley step
``erfinv``          1.3e-12        Reduction to ``ndtri``
``lgamma``          2.1e-12        Lanczos, g=7, n=9
``digamma``         3.2e-11        Recurrence to z>=12 + asymptotic series
``gammainc(c)``     1.1e-13        Series | Legendre continued fraction
``betainc``         3.2e-13        Lentz CF + symmetry reflection
==================  =============  ===========================================

One caveat worth stating plainly: in the far right tail SciPy is *not* the more
accurate of the two. ``erfc(26.68)`` underflows to ``0.0`` in SciPy, while the
continued fraction here returns the correct subnormal ``1.46e-311``. The test
suite therefore excludes ``|result| < 1e-290`` from the comparison rather than
pretending the oracle is right there.

The branch points in :func:`erfc` are chosen by *cancellation* analysis, not by
convergence rate -- see the comment in its body. Getting that backwards is the
single most common way a hand-rolled ``erfc`` silently loses four digits in the
tail, which in turn is where every VaR number and every p-value lives.

References
----------
Abramowitz & Stegun (1964), *Handbook of Mathematical Functions*, ch. 6, 7, 26.
Press et al. (2007), *Numerical Recipes* (3rd ed.), ch. 6.
Lentz (1976), "Generating Bessel functions in Mie scattering calculations using
    continued fractions", *Applied Optics* 15(3), 668-671.
Acklam, P. J. (2003), "An algorithm for computing the inverse normal cumulative
    distribution function", https://web.archive.org/web/20151110174102/
    http://home.online.no/~pjacklam/notes/invnorm/
Lanczos, C. (1964), "A precision approximation of the gamma function",
    *SIAM J. Numer. Anal.* 1, 86-96.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "betainc",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "gammainc",
    "gammaincc",
    "lgamma",
    "log_beta",
    "log_ndtr",
    "ndtr",
    "ndtri",
    "norm_pdf",
]

_SQRT2 = float(np.sqrt(2.0))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_INV_SQRT_PI = float(1.0 / np.sqrt(np.pi))

# Series/continued-fraction changeover for erfc. Below this the positive-term
# confluent series is used (no cancellation); above it the Lentz continued
# fraction converges in <40 iterations. 2.5 is where the two are equally cheap.
_ERF_SERIES_CUTOFF = 2.5
# Where the *positive* axis of erfc leaves the series for the continued
# fraction. Set by cancellation error (eps * erf/erfc), not by convergence.
_ERFC_CF_CUTOFF = 1.2
_MAX_ITER = 300
_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)


def _asarray(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Error function                                                              #
# --------------------------------------------------------------------------- #
def _erf_series(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""``erf(x)`` via the confluent hypergeometric series.

    Uses the Kummer transformation of :math:`M(1/2, 3/2, -x^2)`:

    .. math::
        \operatorname{erf}(x) = \frac{2x}{\sqrt{\pi}} e^{-x^2}
            \sum_{n=0}^{\infty} \frac{(2x^2)^n}{1 \cdot 3 \cdot 5 \cdots (2n+1)}

    Crucially **every term is positive**, so there is no subtractive
    cancellation -- unlike the naive alternating Maclaurin series, which loses
    ~7 significant digits by :math:`x = 3`. Term ratio is
    :math:`t_n / t_{n-1} = 2x^2 / (2n+1)`, so convergence is geometric once
    :math:`n > x^2`.

    Valid and accurate for :math:`|x| \lesssim 4`; we only call it for
    :math:`|x| \le 2.5`.
    """
    x2 = x * x
    term = np.ones_like(x)
    total = np.ones_like(x)
    for n in range(1, _MAX_ITER):
        term = term * (2.0 * x2) / (2.0 * n + 1.0)
        total = total + term
        if np.all(term <= _EPS * total):
            break
    return 2.0 * x * _INV_SQRT_PI * np.exp(-x2) * total


def _erfc_continued_fraction(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""``erfc(x)`` for :math:`x > 0` via a continued fraction in Lentz form.

    .. math::
        \operatorname{erfc}(x) = \frac{e^{-x^2}}{\sqrt{\pi}}
            \cfrac{1}{x + \cfrac{1/2}{x + \cfrac{1}{x + \cfrac{3/2}{x + \cdots}}}}

    with partial numerators :math:`a_n = n/2`. Evaluated with the modified
    Lentz algorithm, which avoids the catastrophic failure mode of forward
    recurrence when an intermediate denominator passes near zero.

    Converges rapidly for :math:`x \gtrsim 2`; this is the branch that gives
    QuantOS accurate *tail* probabilities, which matter enormously for VaR and
    for p-values of the statistical tests in :mod:`quantos.core.stats`.
    """
    # Modified Lentz: b0 = 0, so seed f with a tiny value rather than zero.
    f = np.full_like(x, _TINY)
    c = f.copy()
    d = np.zeros_like(x)

    for n in range(0, _MAX_ITER):
        a = 1.0 if n == 0 else 0.5 * n
        b = x
        d = b + a * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = b + a / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        delta = c * d
        f = f * delta
        if np.all(np.abs(delta - 1.0) <= _EPS):
            break
    return np.exp(-x * x) * _INV_SQRT_PI * f


def erf(x: ArrayLike) -> NDArray[np.float64]:
    r"""Gauss error function :math:`\operatorname{erf}(x)`, vectorised.

    Purpose
        Underpins the normal CDF, and hence Black-Scholes, every z-test, and
        every Gaussian sampler diagnostic in the platform.
    Inputs
        ``x`` -- real array-like of any shape.
    Outputs
        ``float64`` array of the same shape, values in :math:`(-1, 1)`.
    Assumptions
        Real input. NaN propagates; ``+-inf`` maps to ``+-1``.
    Complexity
        :math:`O(n)` with a small constant (< 40 series/CF iterations).
    Failure modes
        None: both branches are unconditionally convergent. Complex input
        raises :class:`TypeError` at the ``asarray`` cast.
    Example
        >>> float(np.round(erf(1.0), 11))
        0.84270079295
    """
    return 1.0 - erfc(x)


def erfc(x: ArrayLike) -> NDArray[np.float64]:
    r"""Complementary error function :math:`1 - \operatorname{erf}(x)`.

    Computed *directly* rather than as ``1 - erf(x)`` so that tail values below
    machine epsilon retain full relative precision: ``erfc(6.0)`` is
    ``2.15e-17``, which the subtraction route would round to exactly zero.

    Complexity
        :math:`O(n)`; failure modes and assumptions as :func:`erf`.

    Example
        >>> float(erfc(6.0))  # doctest: +ELLIPSIS
        2.1519736712...e-17
    """
    x = _asarray(x)
    out = np.empty_like(x)
    ax = np.abs(x)
    fin = np.isfinite(x)

    # Branch structure is dictated entirely by cancellation, not by which
    # algorithm converges fastest. Writing erfc(x) = 1 - erf(x) is only safe
    # while erf(x) is not close to 1: the relative error of the subtraction is
    # eps * erf(x)/erfc(x), which passes 1e-15 around x = 1.2 and reaches 1e-12
    # by x = 2.5. So the positive axis switches to the continued fraction --
    # which computes the small result *directly* -- at 1.2, well before the
    # series stops converging.
    #
    # The negative axis has no such problem: erfc(-y) = 1 + erf(y) adds two
    # positive quantities. It is therefore handled by the series all the way
    # out to the series' own accuracy limit, and by 2 - CF(y) beyond that,
    # where the result is ~2 and relative precision is trivially preserved.
    neg_ser = fin & (x < 0.0) & (ax <= _ERF_SERIES_CUTOFF)
    neg_cf = fin & (x < 0.0) & (ax > _ERF_SERIES_CUTOFF)
    pos_ser = fin & (x >= 0.0) & (x < _ERFC_CF_CUTOFF)
    pos_cf = fin & (x >= _ERFC_CF_CUTOFF)

    if np.any(neg_ser):
        out[neg_ser] = 1.0 + _erf_series(ax[neg_ser])
    if np.any(neg_cf):
        out[neg_cf] = 2.0 - _erfc_continued_fraction(ax[neg_cf])
    if np.any(pos_ser):
        out[pos_ser] = 1.0 - _erf_series(x[pos_ser])
    if np.any(pos_cf):
        out[pos_cf] = _erfc_continued_fraction(x[pos_cf])

    out[np.isposinf(x)] = 0.0
    out[np.isneginf(x)] = 2.0
    out[np.isnan(x)] = np.nan
    return out


def erfinv(y: ArrayLike) -> NDArray[np.float64]:
    r"""Inverse error function, :math:`\operatorname{erf}^{-1}(y)`.

    Implemented by reduction to :func:`ndtri`, since
    :math:`\operatorname{erf}^{-1}(y) = \Phi^{-1}((y+1)/2)/\sqrt{2}`.
    Returns ``+-inf`` at ``y = +-1`` and NaN outside :math:`[-1, 1]`.
    """
    y = _asarray(y)
    return ndtri(0.5 * (y + 1.0)) / _SQRT2


# --------------------------------------------------------------------------- #
# Standard normal                                                             #
# --------------------------------------------------------------------------- #
def norm_pdf(x: ArrayLike) -> NDArray[np.float64]:
    r"""Standard normal density :math:`\phi(x) = e^{-x^2/2}/\sqrt{2\pi}`."""
    x = _asarray(x)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def ndtr(x: ArrayLike) -> NDArray[np.float64]:
    r"""Standard normal CDF :math:`\Phi(x)`.

    Uses ``0.5 * erfc(-x / sqrt(2))``, which is the numerically correct
    formulation: it routes the *small* result through :func:`erfc`'s continued
    fraction, preserving relative accuracy deep into the left tail where
    ``0.5 * (1 + erf(x/sqrt2))`` would return zero.

    Example
        >>> float(ndtr(-8.0))  # doctest: +ELLIPSIS
        6.22096...e-16
    """
    x = _asarray(x)
    return 0.5 * erfc(-x / _SQRT2)


def log_ndtr(x: ArrayLike) -> NDArray[np.float64]:
    r""":math:`\log \Phi(x)`, accurate for very negative ``x``.

    For ``x < -20`` the CDF underflows to zero in ``float64``, so we switch to
    the asymptotic expansion

    .. math::
        \log\Phi(x) = -\tfrac{x^2}{2} - \log(-x) - \tfrac12\log(2\pi)
                      + \log\!\Big(1 - \tfrac{1}{x^2} + \tfrac{3}{x^4}
                        - \tfrac{15}{x^6} + \cdots\Big)

    which is what makes Gaussian log-likelihoods (see
    :mod:`quantos.core.timeseries.garch`) stable under extreme standardised
    residuals.

    Symmetrically, for ``x > 6`` the CDF rounds to exactly ``1.0`` and a naive
    ``log(ndtr(x))`` returns ``0.0``, discarding the true value of order
    :math:`-10^{-16}`. There we use :math:`\log\Phi(x) = \log(1 - Q)` with
    :math:`Q = \tfrac12\operatorname{erfc}(x/\sqrt2)` from the continued
    fraction, and :func:`numpy.log1p` to keep the small result exact.
    """
    x = _asarray(x)
    out = np.empty_like(x)
    # The log1p form is at least as accurate as log(ndtr(x)) for *every*
    # x > 0, not merely for large x: at x = 6, Phi(x) = 1 - 9.87e-10, whose
    # float64 neighbours are 1.1e-16 apart, so log(ndtr(x)) carries ~1e-7
    # relative error. Branching at zero rather than at some large threshold
    # also removes a discontinuity in the error profile.
    deep = x < -30.0
    high = x > 0.0
    mid = ~deep & ~high
    if np.any(mid):
        out[mid] = np.log(ndtr(x[mid]))
    if np.any(high):
        out[high] = np.log1p(-0.5 * erfc(x[high] / _SQRT2))
    if np.any(deep):
        z = x[deep]
        z2 = z * z
        # Asymptotic series for the Mills ratio: coefficients are the double
        # factorials (-1)^n (2n-1)!!. Truncating at the z^-6 term leaves ~2e-11
        # relative error at the z = -20 branch point; carrying two more terms
        # (105, 945) drops that to ~3e-15 at z = -30, which is where we now cut
        # over. The series is asymptotic, not convergent -- more terms are only
        # an improvement because |z| is large here.
        u = 1.0 / z2
        corr = 1.0 + u * (-1.0 + u * (3.0 + u * (-15.0 + u * (105.0 - 945.0 * u))))
        out[deep] = -0.5 * z2 - np.log(-z) - 0.5 * np.log(2.0 * np.pi) + np.log(corr)
    return out


# Acklam's rational approximation coefficients (|rel err| < 1.15e-9).
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def _ndtri_acklam(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Acklam's seed value for the normal quantile: ~9 correct digits."""
    out = np.empty_like(p)

    lo = p < _P_LOW
    hi = p > _P_HIGH
    mid = ~lo & ~hi

    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(p[lo]))
        out[lo] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log1p(-p[hi]))
        out[hi] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    return out


def ndtri(p: ArrayLike) -> NDArray[np.float64]:
    r"""Standard normal quantile function :math:`\Phi^{-1}(p)`.

    Two-stage design, chosen deliberately over a single high-order rational
    approximation such as Wichura's AS 241:

    1. **Seed** with Acklam's rational approximation (relative error 1.15e-9).
    2. **Refine** with one Halley step on :math:`\Phi(x) - p = 0`. Halley is
       cubically convergent, so 9 digits become ~27 -- capped by ``float64`` at
       full machine precision.

    The refinement means the *result's* accuracy is inherited from :func:`ndtr`
    rather than from a table of memorised constants, which makes the whole
    routine auditable: if :func:`ndtr` is correct, :func:`ndtri` is correct.
    That is a property AS 241 does not have.

    The Halley update, with :math:`u = (\Phi(x)-p)\sqrt{2\pi}e^{x^2/2}`:

    .. math:: x \leftarrow x - \frac{u}{1 + xu/2}

    Failure modes
        ``p <= 0`` returns ``-inf``, ``p >= 1`` returns ``+inf``, ``p`` outside
        :math:`[0,1]` or NaN returns NaN. For :math:`|x| > 30` the Halley step
        is skipped (``exp(x^2/2)`` overflows) and Acklam's 1.15e-9 accuracy
        stands; this only affects ``p < 1e-197``.

    Example
        >>> float(np.round(ndtri(0.975), 10))
        1.9599639845
    """
    p = _asarray(p)
    out = np.full(p.shape, np.nan, dtype=np.float64)

    interior = (p > 0.0) & (p < 1.0)
    # Distinguish the boundary from the invalid: p exactly 0 or 1 has a genuine
    # infinite quantile, whereas p outside [0, 1] has no quantile at all and must
    # stay NaN. Collapsing the two (returning -inf for p = -0.1) hides a caller
    # bug behind a plausible-looking value.
    out[p == 0.0] = -np.inf
    out[p == 1.0] = np.inf
    if not np.any(interior):
        return out

    q = p[interior]
    x = _ndtri_acklam(q)

    # Halley refinement, guarded against overflow in exp(x^2/2).
    refinable = np.abs(x) < 30.0
    if np.any(refinable):
        xr = x[refinable]
        qr = q[refinable]
        # The residual Phi(x) - p must be evaluated without cancellation. For
        # p > 1/2 both Phi(x) and p are close to 1, and their difference loses
        # every significant digit. Rewriting it in terms of the upper tail,
        #     Phi(x) - p = (1 - Q(x)) - (1 - r) = r - Q(x),   r = 1 - p,
        # keeps both operands small so the subtraction is benign. This is the
        # difference between 9.6e-10 and 3e-16 relative accuracy near p = 1.
        upper = qr > 0.5
        e = np.empty_like(xr)
        if np.any(~upper):
            e[~upper] = ndtr(xr[~upper]) - qr[~upper]
        if np.any(upper):
            e[upper] = (1.0 - qr[upper]) - 0.5 * erfc(xr[upper] / _SQRT2)
        u = e * _SQRT_2PI * np.exp(0.5 * xr * xr)
        x[refinable] = xr - u / (1.0 + 0.5 * xr * u)

    out[interior] = x
    return out


# --------------------------------------------------------------------------- #
# Gamma family                                                                #
# --------------------------------------------------------------------------- #
# Lanczos g = 7, n = 9 -- the standard coefficient set (|rel err| < 2e-10 before
# the reflection, and we only use it for the *log*, so absolute error in the
# log is ~1e-15 for the arguments we care about).
_LANCZOS_G = 7.0
_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def lgamma(x: ArrayLike) -> NDArray[np.float64]:
    r"""Vectorised :math:`\log \Gamma(x)` for :math:`x > 0` (Lanczos, g=7).

    ``math.lgamma`` is scalar-only; this is the array version used by the
    Student-t and chi-square densities. Validated against ``math.lgamma`` to
    < 1e-14 relative error over :math:`(10^{-8}, 10^{8})`.
    """
    x = _asarray(x)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    pos = x > 0.0
    if not np.any(pos):
        return out

    # The Lanczos series is written in terms of z = x - 1 and evaluates
    # 1/(z + i) for i = 1..8. As x -> 0 the i=1 term divides by zero, and for
    # x below ~1e-16 the subtraction z = x - 1 rounds to exactly -1, which is a
    # hard pole rather than a gradual loss. Shift such arguments up by one
    # using the exact recurrence lgamma(x) = lgamma(x+1) - log(x); the
    # correction -log(x) carries the entire singular behaviour analytically.
    xw = x[pos]
    shifted = xw < 1.0
    correction = np.where(shifted, -np.log(np.where(shifted, xw, 1.0)), 0.0)
    xw = np.where(shifted, xw + 1.0, xw)

    z = xw - 1.0
    series = np.full(z.shape, _LANCZOS[0])
    for i in range(1, len(_LANCZOS)):
        series = series + _LANCZOS[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    out[pos] = 0.5 * np.log(2.0 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(series) + correction
    return out


def log_beta(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    r""":math:`\log B(a,b) = \log\Gamma(a) + \log\Gamma(b) - \log\Gamma(a+b)`."""
    a = _asarray(a)
    b = _asarray(b)
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def digamma(x: ArrayLike) -> NDArray[np.float64]:
    r"""Digamma :math:`\psi(x) = \frac{d}{dx}\log\Gamma(x)` for :math:`x > 0`.

    Recurrence :math:`\psi(x) = \psi(x+1) - 1/x` shifts the argument above 6,
    then the Stirling-type asymptotic series is applied. Used by the
    Newton solver for Student-t degrees-of-freedom MLE.
    """
    x = _asarray(x)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    pos = x > 0.0
    if not np.any(pos):
        return out

    # Shift to z >= 12 rather than the textbook 6: the asymptotic series is
    # truncated at the z^-8 term, whose first neglected successor is
    # 1/(132 z^10). At z=6 that is ~1e-10 absolute; at z=12 it is ~1e-13,
    # which buys three digits for the cost of six extra reciprocals.
    z = x[pos].copy()
    acc = np.zeros_like(z)
    shift = z < 12.0
    while np.any(shift):
        acc[shift] -= 1.0 / z[shift]
        z[shift] += 1.0
        shift = z < 12.0

    inv = 1.0 / z
    inv2 = inv * inv
    # psi(z) ~ ln z - 1/(2z) - 1/(12z^2) + 1/(120z^4) - 1/(252z^6) + 1/(240 z^8)
    series = (
        np.log(z)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
    )
    out[pos] = series + acc
    return out


def _gammainc_series(a: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Regularised lower incomplete gamma :math:`P(a,x)` by series.

    .. math:: P(a,x) = \frac{x^a e^{-x}}{\Gamma(a)}
                       \sum_{n=0}^{\infty} \frac{x^n}{a(a+1)\cdots(a+n)}

    All terms positive; used for :math:`x < a + 1` where it converges fastest.
    """
    ap = a.copy()
    term = 1.0 / a
    total = term.copy()
    for _ in range(_MAX_ITER):
        ap = ap + 1.0
        term = term * x / ap
        total = total + term
        if np.all(np.abs(term) <= np.abs(total) * _EPS):
            break
    return total * np.exp(-x + a * np.log(x) - lgamma(a))


def _gammaincc_cf(a: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Regularised upper incomplete gamma :math:`Q(a,x)` by continued fraction.

    Legendre's continued fraction in modified Lentz form; used for
    :math:`x \ge a + 1`.
    """
    b = x + 1.0 - a
    c = np.full_like(x, 1.0 / _TINY)
    d = 1.0 / b
    h = d.copy()
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = b + an / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if np.all(np.abs(delta - 1.0) <= _EPS):
            break
    return np.exp(-x + a * np.log(x) - lgamma(a)) * h


def gammainc(a: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    r"""Regularised lower incomplete gamma :math:`P(a,x) = \gamma(a,x)/\Gamma(a)`.

    This *is* the chi-square CDF: :math:`F_{\chi^2_k}(x) = P(k/2, x/2)`, which
    is how QuantOS produces p-values for Ljung-Box, Jarque-Bera, Engle's ARCH
    test, and the Johansen trace statistic's asymptotic tail.

    Branch selection follows Numerical Recipes: series for ``x < a+1``,
    continued fraction otherwise, which keeps both branches in their
    rapidly-convergent regime.
    """
    a, x = np.broadcast_arrays(_asarray(a), _asarray(x))
    out = np.full(a.shape, np.nan, dtype=np.float64)

    valid = (a > 0.0) & (x >= 0.0)
    zero = valid & (x == 0.0)
    out[zero] = 0.0

    work = valid & (x > 0.0)
    if not np.any(work):
        return out

    aw, xw = a[work], x[work]
    use_series = xw < aw + 1.0
    res = np.empty_like(xw)
    if np.any(use_series):
        res[use_series] = _gammainc_series(aw[use_series], xw[use_series])
    if np.any(~use_series):
        res[~use_series] = 1.0 - _gammaincc_cf(aw[~use_series], xw[~use_series])
    out[work] = np.clip(res, 0.0, 1.0)
    return out


def gammaincc(a: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    r"""Regularised upper incomplete gamma :math:`Q(a,x) = 1 - P(a,x)`.

    Computed directly through the continued-fraction branch wherever that
    branch applies, so upper-tail p-values keep full relative precision instead
    of being destroyed by ``1 - 0.9999999999999999``.
    """
    a, x = np.broadcast_arrays(_asarray(a), _asarray(x))
    out = np.full(a.shape, np.nan, dtype=np.float64)

    valid = (a > 0.0) & (x >= 0.0)
    zero = valid & (x == 0.0)
    out[zero] = 1.0

    work = valid & (x > 0.0)
    if not np.any(work):
        return out

    aw, xw = a[work], x[work]
    use_cf = xw >= aw + 1.0
    res = np.empty_like(xw)
    if np.any(use_cf):
        res[use_cf] = _gammaincc_cf(aw[use_cf], xw[use_cf])
    if np.any(~use_cf):
        res[~use_cf] = 1.0 - _gammainc_series(aw[~use_cf], xw[~use_cf])
    out[work] = np.clip(res, 0.0, 1.0)
    return out


def _betacf(
    a: NDArray[np.float64], b: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < _TINY, _TINY, d)
    d = 1.0 / d
    h = d.copy()

    for m in range(1, _MAX_ITER):
        m2 = 2 * m
        # Even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        h = h * d * c
        # Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if np.all(np.abs(delta - 1.0) <= _EPS):
            break
    return h


def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    r"""Regularised incomplete beta :math:`I_x(a,b) = B(x;a,b)/B(a,b)`.

    Supplies the Student-t and Fisher-F CDFs, hence every t-statistic p-value
    in :mod:`quantos.core.stats` -- including the HAC-corrected regression
    t-stats that the cointegration and factor-attribution code depends on.

    The reflection :math:`I_x(a,b) = 1 - I_{1-x}(b,a)` is applied whenever
    :math:`x > (a+1)/(a+b+2)`, which is exactly the condition that keeps the
    continued fraction in its fast-converging half.
    """
    a, b, x = np.broadcast_arrays(_asarray(a), _asarray(b), _asarray(x))
    out = np.full(a.shape, np.nan, dtype=np.float64)

    valid = (a > 0.0) & (b > 0.0) & (x >= 0.0) & (x <= 1.0)
    out[valid & (x == 0.0)] = 0.0
    out[valid & (x == 1.0)] = 1.0

    work = valid & (x > 0.0) & (x < 1.0)
    if not np.any(work):
        return out

    aw, bw, xw = a[work], b[work], x[work]
    front = np.exp(aw * np.log(xw) + bw * np.log1p(-xw) - log_beta(aw, bw))
    flip = xw > (aw + 1.0) / (aw + bw + 2.0)

    res = np.empty_like(xw)
    direct = ~flip
    if np.any(direct):
        res[direct] = front[direct] * _betacf(aw[direct], bw[direct], xw[direct]) / aw[direct]
    if np.any(flip):
        res[flip] = 1.0 - front[flip] * _betacf(bw[flip], aw[flip], 1.0 - xw[flip]) / bw[flip]

    out[work] = np.clip(res, 0.0, 1.0)
    return out
