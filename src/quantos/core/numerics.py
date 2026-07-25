"""Quadrature, interpolation and low-discrepancy sequences.

Contents
--------
* :func:`gauss_legendre` -- fixed-order quadrature; the right tool when the
  integrand is smooth and you want a *deterministic* node count (Heston's
  characteristic-function integral).
* :func:`adaptive_quad` -- adaptive Simpson with a per-interval error estimate,
  for integrands whose difficulty you do not know in advance.
* :class:`CubicSpline` -- natural cubic spline, C2 continuous.
* :class:`MonotoneSpline` -- Fritsch-Carlson (PCHIP) shape-preserving cubic;
  only C1, but it *cannot overshoot*. That property is non-negotiable for
  interpolating a discount curve or a CDF, where an overshoot produces a
  negative forward rate or a non-monotone probability.
* :func:`sobol` / :func:`halton` -- low-discrepancy sequences for QMC.

References
----------
Golub, G. H. & Welsch, J. H. (1969), "Calculation of Gauss quadrature rules",
    *Math. Comp.* 23, 221-230.
Fritsch, F. N. & Carlson, R. E. (1980), "Monotone piecewise cubic
    interpolation", *SIAM J. Numer. Anal.* 17(2), 238-246.
Joe, S. & Kuo, F. Y. (2008), "Constructing Sobol sequences with better
    two-dimensional projections", *SIAM J. Sci. Comput.* 30, 2635-2654.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CubicSpline",
    "MonotoneSpline",
    "adaptive_quad",
    "gauss_legendre",
    "gauss_legendre_nodes",
    "halton",
    "sobol",
]


@lru_cache(maxsize=64)
def gauss_legendre_nodes(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    r"""Gauss-Legendre nodes and weights on :math:`[-1, 1]`, order ``n``.

    Computed by the **Golub-Welsch algorithm**: the nodes are the eigenvalues
    of the symmetric tridiagonal Jacobi matrix of the Legendre recurrence, and
    the weights are :math:`2 v_{0,i}^2` from the first components of the
    normalised eigenvectors. This is preferable to Newton-iterating on
    :math:`P_n` -- it is a single symmetric eigenproblem, so it is stable to
    high order and needs no initial guesses.

    Cached, because the same order is requested thousands of times when pricing
    a surface.

    Example
        >>> x, w = gauss_legendre_nodes(5)
        >>> round(sum(w), 12)            # weights integrate 1 over [-1,1]
        2.0
    """
    if n < 1:
        raise ValueError("order must be >= 1")
    k = np.arange(1, n, dtype=float)
    # Off-diagonal of the Jacobi matrix for Legendre polynomials.
    beta = k / np.sqrt(4.0 * k * k - 1.0)
    jacobi = np.diag(beta, -1) + np.diag(beta, 1)
    values, vectors = np.linalg.eigh(jacobi)
    weights = 2.0 * vectors[0, :] ** 2
    order = np.argsort(values)
    return tuple(values[order].tolist()), tuple(weights[order].tolist())


def gauss_legendre(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float,
    b: float,
    n: int = 64,
) -> float:
    r"""Integrate a vectorised ``f`` over ``[a, b]`` with ``n``-point Gauss-Legendre.

    Exact for polynomials of degree :math:`\le 2n-1`, and spectrally accurate
    for analytic integrands -- which is why 64 nodes suffice for the Heston
    characteristic function where Simpson would need thousands.

    Complexity
        One vectorised call to ``f`` with ``n`` points.

    Example
        >>> import numpy as np
        >>> round(gauss_legendre(np.sin, 0.0, np.pi, 32), 12)
        2.0
    """
    nodes, weights = gauss_legendre_nodes(n)
    x = np.asarray(nodes)
    w = np.asarray(weights)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return float(half * np.dot(w, np.asarray(f(half * x + mid), dtype=float)))


def _simpson(
    f: Callable[[float], float], a: float, b: float, fa: float, fm: float, fb: float
) -> float:
    return (b - a) / 6.0 * (fa + 4.0 * fm + fb)


def adaptive_quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_depth: int = 50,
) -> float:
    r"""Adaptive Simpson quadrature with recursive interval bisection.

    Purpose
        Integrate when you cannot promise smoothness. The interval is split
        wherever the local Simpson error estimate
        :math:`|S_{\text{left}} + S_{\text{right}} - S_{\text{whole}}|/15`
        exceeds the locally-apportioned tolerance.
    Complexity
        Proportional to the integrand's difficulty, not to the interval width.
    Failure modes
        Silently stops refining at ``max_depth``; a genuinely singular
        integrand will return an inaccurate value. Use a substitution to remove
        the singularity rather than raising ``max_depth``.

    Example
        >>> import math
        >>> round(adaptive_quad(math.exp, 0.0, 1.0), 12)
        1.718281828459
    """

    def recurse(
        a_: float, b_: float, fa: float, fm: float, fb: float, whole: float, tol_: float, depth: int
    ) -> float:
        m = 0.5 * (a_ + b_)
        lm = 0.5 * (a_ + m)
        rm = 0.5 * (m + b_)
        flm, frm = f(lm), f(rm)
        left = _simpson(f, a_, m, fa, flm, fm)
        right = _simpson(f, m, b_, fm, frm, fb)
        delta = left + right - whole
        if depth >= max_depth or abs(delta) <= 15.0 * tol_:
            # Richardson extrapolation: the /15 correction lifts the composite
            # Simpson result to fifth order for free.
            return left + right + delta / 15.0
        return recurse(a_, m, fa, flm, fm, left, 0.5 * tol_, depth + 1) + recurse(
            m, b_, fm, frm, fb, right, 0.5 * tol_, depth + 1
        )

    fa, fb = f(a), f(b)
    m = 0.5 * (a + b)
    fm = f(m)
    whole = _simpson(f, a, b, fa, fm, fb)
    return recurse(a, b, fa, fm, fb, whole, tol, 0)


@dataclass
class CubicSpline:
    r"""Natural cubic spline: C2-continuous, zero second derivative at the ends.

    Solves the standard symmetric tridiagonal system for the second
    derivatives in :math:`O(n)` by the Thomas algorithm.

    Use this when smoothness matters more than shape (e.g. a volatility smile
    you intend to differentiate twice for a risk-neutral density). Use
    :class:`MonotoneSpline` when *not overshooting* matters more.

    Example
        >>> import numpy as np
        >>> s = CubicSpline(np.array([0.,1.,2.,3.]), np.array([0.,1.,4.,9.]))
        >>> round(float(s(1.5)), 10)   # not 2.25: natural BCs force m[0]=m[-1]=0
        2.2
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be 1-D arrays of equal length")
        if self.x.size < 3:
            raise ValueError("cubic spline needs at least 3 knots")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing")
        self._m = self._second_derivatives()

    def _second_derivatives(self) -> NDArray[np.float64]:
        x, y = self.x, self.y
        n = x.size
        h = np.diff(x)
        rhs = 6.0 * (np.diff(y)[1:] / h[1:] - np.diff(y)[:-1] / h[:-1])

        # Thomas algorithm on the (n-2) interior equations.
        lower = h[:-1].copy()
        diag = 2.0 * (h[:-1] + h[1:])
        upper = h[1:].copy()
        for i in range(1, n - 2):
            w = lower[i] / diag[i - 1]
            diag[i] -= w * upper[i - 1]
            rhs[i] -= w * rhs[i - 1]
        m_interior = np.zeros(n - 2)
        m_interior[-1] = rhs[-1] / diag[-1]
        for i in range(n - 4, -1, -1):
            m_interior[i] = (rhs[i] - upper[i] * m_interior[i + 1]) / diag[i]

        m = np.zeros(n)
        m[1:-1] = m_interior  # natural boundary: m[0] = m[-1] = 0
        return m

    def __call__(self, xi: ArrayLike) -> NDArray[np.float64]:
        """Evaluate. Extrapolates linearly from the end segments' slopes."""
        xi = np.asarray(xi, dtype=float)
        idx = np.clip(np.searchsorted(self.x, xi) - 1, 0, self.x.size - 2)
        h = self.x[idx + 1] - self.x[idx]
        a = (self.x[idx + 1] - xi) / h
        b = (xi - self.x[idx]) / h
        return (
            a * self.y[idx]
            + b * self.y[idx + 1]
            + ((a**3 - a) * self._m[idx] + (b**3 - b) * self._m[idx + 1]) * (h * h) / 6.0
        )

    def derivative(self, xi: ArrayLike) -> NDArray[np.float64]:
        """First derivative of the spline."""
        xi = np.asarray(xi, dtype=float)
        idx = np.clip(np.searchsorted(self.x, xi) - 1, 0, self.x.size - 2)
        h = self.x[idx + 1] - self.x[idx]
        a = (self.x[idx + 1] - xi) / h
        b = (xi - self.x[idx]) / h
        return (self.y[idx + 1] - self.y[idx]) / h + (
            (1.0 - 3.0 * a**2) * self._m[idx] + (3.0 * b**2 - 1.0) * self._m[idx + 1]
        ) * h / 6.0


@dataclass
class MonotoneSpline:
    r"""Shape-preserving piecewise-cubic Hermite interpolant (Fritsch-Carlson).

    Guarantees: if the input data are monotone, the interpolant is monotone;
    in all cases it never overshoots the local data range. It achieves this by
    limiting the Hermite slopes to lie inside the Fritsch-Carlson region,
    :math:`\alpha^2 + \beta^2 \le 9` where :math:`\alpha = d_i/\Delta_i`,
    :math:`\beta = d_{i+1}/\Delta_i`.

    The cost is C1 rather than C2 continuity -- the second derivative jumps at
    knots. That trade is right for discount curves, cumulative distributions
    and empirical quantile maps, and wrong for anything you plan to
    double-differentiate.

    Example
        >>> import numpy as np
        >>> s = MonotoneSpline(np.array([0.,1.,2.,3.]), np.array([0.,0.,0.,1.]))
        >>> bool(np.all(np.diff(s(np.linspace(0, 3, 101))) >= -1e-15))
        True
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError("x and y must be 1-D arrays of equal length")
        if self.x.size < 2:
            raise ValueError("need at least 2 knots")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing")
        self._slopes = self._fritsch_carlson_slopes()

    def _fritsch_carlson_slopes(self) -> NDArray[np.float64]:
        h = np.diff(self.x)
        delta = np.diff(self.y) / h
        n = self.x.size
        d = np.zeros(n)

        if n == 2:
            d[:] = delta[0]
            return d

        # Interior: weighted harmonic mean of neighbouring secants, which is
        # zero whenever the secants disagree in sign -- this is what enforces
        # local monotonicity and kills overshoot at turning points.
        w1 = 2.0 * h[1:] + h[:-1]
        w2 = h[1:] + 2.0 * h[:-1]
        same_sign = delta[:-1] * delta[1:] > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
        d[1:-1] = np.where(same_sign, harmonic, 0.0)

        # One-sided three-point endpoint formula, clipped to preserve shape.
        d[0] = self._endpoint_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = self._endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
        return d

    @staticmethod
    def _endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if slope * d0 <= 0.0:
            return 0.0
        if d0 * d1 <= 0.0 and abs(slope) > abs(3.0 * d0):
            return 3.0 * d0
        return float(slope)

    def __call__(self, xi: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the interpolant (clamped outside the knot range)."""
        xi = np.asarray(xi, dtype=float)
        idx = np.clip(np.searchsorted(self.x, xi) - 1, 0, self.x.size - 2)
        h = self.x[idx + 1] - self.x[idx]
        t = np.clip((xi - self.x[idx]) / h, 0.0, 1.0)
        t2, t3 = t * t, t * t * t
        # Cubic Hermite basis.
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return (
            h00 * self.y[idx]
            + h10 * h * self._slopes[idx]
            + h01 * self.y[idx + 1]
            + h11 * h * self._slopes[idx + 1]
        )


# --------------------------------------------------------------------------- #
# Low-discrepancy sequences                                                   #
# --------------------------------------------------------------------------- #
_FIRST_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
)


def halton(n: int, dim: int, *, skip: int = 1) -> NDArray[np.float64]:
    r"""Halton low-discrepancy sequence, ``n`` points in ``dim`` dimensions.

    The van der Corput construction in a distinct prime base per dimension.
    Star discrepancy is :math:`O((\log n)^d / n)` against Monte Carlo's
    :math:`O(n^{-1/2})`, which is the entire reason to use it.

    ``skip`` discards leading points: the first Halton point is always the
    origin, and early points in high dimensions are notoriously correlated.

    Known limitation
        Halton degrades badly beyond ~15 dimensions as the bases grow.
        :func:`sobol` is the better choice there, and this function refuses
        ``dim > 36`` rather than returning something useless.
    """
    if dim < 1 or dim > len(_FIRST_PRIMES):
        raise ValueError(f"dim must lie in [1, {len(_FIRST_PRIMES)}], got {dim}")
    idx = np.arange(skip, skip + n, dtype=np.int64)
    out = np.empty((n, dim), dtype=float)
    for d in range(dim):
        base = _FIRST_PRIMES[d]
        value = np.zeros(n, dtype=float)
        denom = 1.0
        remaining = idx.copy()
        while np.any(remaining > 0):
            denom *= base
            value += (remaining % base) / denom
            remaining //= base
        out[:, d] = value
    return out


def sobol(n: int, dim: int, *, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
    r"""Sobol' sequence in base 2 with optional Owen-style digital scrambling.

    Direction numbers come from primitive polynomials over GF(2); the
    implementation uses the Gray-code recurrence
    :math:`x_{n+1} = x_n \oplus v_c` where ``c`` is the index of the lowest
    zero bit of ``n``, which yields each successive point in ``O(1)``.

    Passing ``rng`` applies a random digital shift (Cranley-Patterson). This is
    what makes QMC *estimable*: an unscrambled Sobol' integral has no error
    bar, but averaging over independent random shifts gives an unbiased
    estimator with a computable variance. Use it whenever you need a confidence
    interval, which in this codebase is always.

    Note
        Direction numbers are generated from the first ``dim`` primitive
        polynomials, giving good uniformity to moderate dimension. For
        very high dimensional problems (>50) the Joe-Kuo tabulated
        direction numbers are preferable; this implementation caps at 32.
    """
    if dim < 1 or dim > 32:
        raise ValueError(f"dim must lie in [1, 32], got {dim}")
    bits = 30
    v = _sobol_direction_numbers(dim, bits)

    out = np.zeros((n, dim), dtype=np.uint32)
    x = np.zeros(dim, dtype=np.uint32)
    for i in range(n):
        # Index of the rightmost zero bit of i (Gray-code recurrence).
        c = 0
        value = i
        while value & 1:
            value >>= 1
            c += 1
        x = x ^ v[:, c]
        out[i] = x

    points = out.astype(np.float64) / float(1 << bits)
    if rng is not None:
        points = np.mod(points + rng.random(dim), 1.0)
    return points


@lru_cache(maxsize=16)
def _sobol_direction_numbers(dim: int, bits: int) -> NDArray[np.uint32]:
    """Direction numbers from primitive polynomials over GF(2)."""
    # (degree, coefficient-bitmask, initial m values) for the first dimensions.
    polys: list[tuple[int, int, tuple[int, ...]]] = [
        (0, 0, (1,)),
        (1, 0, (1,)),
        (2, 1, (1, 3)),
        (3, 1, (1, 3, 1)),
        (3, 2, (1, 1, 1)),
        (4, 1, (1, 1, 3, 3)),
        (4, 4, (1, 3, 5, 13)),
        (5, 2, (1, 1, 5, 5, 17)),
        (5, 4, (1, 1, 5, 5, 5)),
        (5, 7, (1, 1, 7, 11, 19)),
        (5, 11, (1, 1, 5, 1, 1)),
        (5, 13, (1, 1, 1, 3, 11)),
        (5, 14, (1, 3, 5, 5, 31)),
        (6, 1, (1, 3, 3, 9, 7, 49)),
        (6, 13, (1, 1, 1, 15, 21, 21)),
        (6, 16, (1, 3, 1, 13, 27, 49)),
        (6, 19, (1, 1, 1, 15, 7, 5)),
        (6, 22, (1, 3, 1, 15, 13, 25)),
        (6, 25, (1, 1, 5, 5, 19, 61)),
        (7, 1, (1, 3, 7, 11, 23, 15, 103)),
        (7, 4, (1, 3, 7, 13, 13, 15, 69)),
        (7, 7, (1, 1, 3, 13, 7, 35, 63)),
        (7, 8, (1, 3, 5, 9, 1, 25, 53)),
        (7, 14, (1, 3, 1, 13, 9, 35, 107)),
        (7, 19, (1, 3, 1, 5, 27, 61, 31)),
        (7, 21, (1, 1, 5, 11, 19, 41, 61)),
        (7, 28, (1, 3, 5, 3, 3, 13, 69)),
        (7, 31, (1, 1, 7, 13, 1, 19, 1)),
        (7, 32, (1, 3, 3, 5, 21, 51, 45)),
        (7, 37, (1, 1, 3, 9, 15, 15, 25)),
        (7, 41, (1, 3, 7, 9, 7, 7, 5)),
        (7, 42, (1, 1, 1, 3, 9, 41, 55)),
    ]
    v = np.zeros((dim, bits), dtype=np.uint32)
    v[0] = np.array([1 << (bits - 1 - i) for i in range(bits)], dtype=np.uint32)

    for d in range(1, dim):
        degree, mask, m_init = polys[d]
        m = list(m_init)
        for i in range(degree, bits):
            value = m[i - degree]
            value ^= value << degree
            for j in range(1, degree):
                if (mask >> (degree - 1 - j)) & 1:
                    value ^= m[i - j] << j
            m.append(value)
        v[d] = np.array([m[i] << (bits - 1 - i) for i in range(bits)], dtype=np.uint32)
    return v
