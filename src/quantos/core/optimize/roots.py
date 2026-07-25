"""Root finding: bracketing, Brent, safeguarded Newton, and a vectorised bisector.

Selection guide
---------------
==========================  ==================================================
Routine                     Use when
==========================  ==================================================
:func:`bisect_vectorised`   You need the *same* monotone scalar root solved for
                            a whole array of targets (every ``ppf`` in
                            :mod:`quantos.core.distributions`). Slow per-root
                            but the array is amortised.
:func:`brentq`              One root, function is cheap, you have a bracket.
                            Superlinear and cannot diverge. The default.
:func:`newton_safeguarded`  One root, derivative available and cheap, you have
                            a bracket, and you want quadratic convergence
                            without giving up guaranteed containment
                            (implied volatility).
==========================  ==================================================

The safeguarding theme is deliberate. Pure Newton is what most implied-vol
implementations use, and it is exactly what blows up on deep out-of-the-money
options where vega underflows: the step ``f/f'`` becomes enormous and the
iterate leaves the domain. Every routine here either cannot leave its bracket
or refuses to return an unconverged answer silently.

References
----------
Brent, R. P. (1973), *Algorithms for Minimization without Derivatives*, ch. 4.
Press et al. (2007), *Numerical Recipes* (3rd ed.), ch. 9.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ConvergenceError",
    "RootResult",
    "bisect_vectorised",
    "bracket_root",
    "brentq",
    "newton_safeguarded",
]


class ConvergenceError(RuntimeError):
    """Raised when an iterative solver fails to reach its tolerance.

    QuantOS never returns an unconverged value with a quiet warning: a
    mispriced option or a wrong quantile that *looks* fine is far more damaging
    than a loud failure.
    """


@dataclass(frozen=True, slots=True)
class RootResult:
    """Outcome of a scalar root solve, including the evidence it converged."""

    root: float
    iterations: int
    function_calls: int
    residual: float
    converged: bool

    def __float__(self) -> float:
        return self.root


def bisect_vectorised(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> NDArray[np.float64]:
    r"""Solve ``f(x) = 0`` elementwise for a monotone increasing ``f``.

    Purpose
        Provide quantile functions for distributions with no closed-form
        inverse, at array granularity. Every iteration evaluates ``f`` once on
        the *entire* array, so the cost is ``max_iter`` vectorised calls rather
        than ``n`` scalar solves -- typically 100x faster than looping.
    Inputs
        ``f`` -- vectorised, monotone non-decreasing in ``x``.
        ``lo``, ``hi`` -- arrays bracketing the root, same shape.
    Outputs
        Array of roots, same shape.
    Assumptions
        ``f`` is monotone non-decreasing. **This is not checked**: verifying it
        would cost more than the solve. Callers in this codebase are all CDFs,
        which are monotone by definition.
    Complexity
        :math:`O(\text{max\_iter})` vectorised evaluations. Bisection halves
        the interval each step, so the iteration count needed for tolerance
        ``t`` on a bracket of width ``w`` is :math:`\log_2(w/t)` -- about 90
        for the ``1e-300`` to ``1e4`` brackets used by the gamma quantiles.
    Failure modes
        If the root is not bracketed, returns the nearer endpoint rather than
        raising: the array setting makes per-element diagnostics impractical,
        and the callers clamp ``p`` to ``[0,1]`` beforehand so a missing
        bracket means a saturated quantile, which is the correct answer.

    Example
        >>> import numpy as np
        >>> r = bisect_vectorised(lambda x: x**2 - np.array([4.0, 9.0]),
        ...                       np.zeros(2), np.full(2, 10.0))
        >>> np.round(r, 9).tolist()
        [2.0, 3.0]
    """
    lo = np.array(lo, dtype=float, copy=True)
    hi = np.array(hi, dtype=float, copy=True)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = np.asarray(f(mid), dtype=float)
        go_right = val < 0.0
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
        if np.all(hi - lo <= tol * np.maximum(1.0, np.abs(mid))):
            break
    return 0.5 * (lo + hi)


def bracket_root(
    f: Callable[[float], float],
    x0: float,
    *,
    factor: float = 1.6,
    max_iter: int = 60,
) -> tuple[float, float]:
    """Expand outward from ``x0`` until a sign change is bracketed.

    Geometric expansion (factor 1.6, the Numerical Recipes default) balances
    the risk of overshooting a nearby root against the cost of many function
    calls when the root is far away.

    Raises
    ------
        :class:`ConvergenceError` if no sign change is found.
    """
    a, b = x0 - 1.0, x0 + 1.0
    fa, fb = f(a), f(b)
    for _ in range(max_iter):
        if fa * fb < 0.0:
            return a, b
        if abs(fa) < abs(fb):
            a += factor * (a - b)
            fa = f(a)
        else:
            b += factor * (b - a)
            fb = f(b)
    raise ConvergenceError(f"could not bracket a root near x0={x0}")


def brentq(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float = 1e-14,
    rtol: float = 4.0 * np.finfo(float).eps,
    max_iter: int = 200,
) -> RootResult:
    r"""Brent's method: inverse quadratic interpolation with bisection fallback.

    Purpose
        The general-purpose scalar root finder. Combines the superlinear
        convergence of the secant/IQI family with bisection's guarantee, by
        rejecting any interpolated step that fails a set of acceptance tests
        and falling back to a bisection step instead.
    Inputs
        ``f`` -- continuous scalar function; ``a``, ``b`` -- a bracket with
        ``f(a) * f(b) <= 0``.
    Outputs
        :class:`RootResult`.
    Complexity
        Superlinear (order ~1.84 on smooth functions); worst case is
        bisection's :math:`\log_2` behaviour, never worse.
    Failure modes
        :class:`ValueError` if the interval is not bracketed;
        :class:`ConvergenceError` if ``max_iter`` is exhausted.

    Example
        >>> import math
        >>> res = brentq(lambda x: math.cos(x) - x, 0.0, 1.0)
        >>> round(res.root, 10)
        0.7390851332
    """
    fa, fb = f(a), f(b)
    calls = 2
    if fa == 0.0:
        return RootResult(a, 0, calls, 0.0, True)
    if fb == 0.0:
        return RootResult(b, 0, calls, 0.0, True)
    if fa * fb > 0.0:
        raise ValueError(f"root not bracketed: f({a})={fa:g} and f({b})={fb:g} share a sign")

    c, fc = a, fa
    d = e = b - a

    for iteration in range(1, max_iter + 1):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            # Keep b as the best estimate so far.
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol = 2.0 * rtol * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return RootResult(float(b), iteration, calls, float(abs(fb)), True)

        if abs(e) < tol or abs(fa) <= abs(fb):
            # Interpolation is unreliable here; bisect.
            d = e = m
        else:
            s = fb / fa
            if a == c:
                # Two points only: linear (secant) interpolation.
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # Three points: inverse quadratic interpolation.
                q_ = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q_ * (q_ - r) - (b - a) * (r - 1.0))
                q = (q_ - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            # Accept the interpolated step only if it stays comfortably inside
            # the bracket and is shrinking; otherwise fall back to bisection.
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m

        a, fa = b, fb
        # float() keeps the promise made by RootResult's annotation: a numpy
        # scalar leaking out here propagates np.float64 through every caller and
        # turns `x == y` into np.True_, which is a different type from bool.
        b = float(b + (d if abs(d) > tol else (tol if m > 0 else -tol)))
        fb = f(b)
        calls += 1

    raise ConvergenceError(f"brentq failed to converge in {max_iter} iterations")


def newton_safeguarded(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    x0: float,
    lo: float,
    hi: float,
    *,
    xtol: float = 1e-12,
    ftol: float = 1e-14,
    max_iter: int = 100,
) -> RootResult:
    r"""Newton's method that can never leave ``[lo, hi]``.

    Purpose
        Quadratic convergence where a derivative is available, without pure
        Newton's failure modes. This is the engine behind
        :func:`quantos.derivatives.implied_vol.implied_volatility`.
    Method
        Take the Newton step. If it lands outside the bracket, or the
        derivative is negligible, or the step fails to reduce the interval,
        substitute a bisection step. The bracket is contracted using the sign
        of the residual at every accepted iterate, so containment is an
        invariant, not a hope.
    Inputs
        ``f``, ``fprime`` -- scalar function and derivative; ``x0`` -- initial
        guess; ``lo``, ``hi`` -- a valid bracket.
    Outputs
        :class:`RootResult`.
    Complexity
        Quadratic near a simple root; bisection-bounded in the worst case.
    Failure modes
        :class:`ValueError` for an invalid bracket, :class:`ConvergenceError`
        on iteration exhaustion.

    Example
        >>> res = newton_safeguarded(lambda x: x*x - 2, lambda x: 2*x,
        ...                          1.0, 0.0, 2.0)
        >>> round(res.root, 12)
        1.414213562373
    """
    flo, fhi = f(lo), f(hi)
    calls = 2
    if flo == 0.0:
        return RootResult(lo, 0, calls, 0.0, True)
    if fhi == 0.0:
        return RootResult(hi, 0, calls, 0.0, True)
    if flo * fhi > 0.0:
        raise ValueError(f"root not bracketed in [{lo}, {hi}]")

    # Orient so that f(lo) < 0 < f(hi); simplifies the containment update.
    if flo > 0.0:
        lo, hi = hi, lo

    x = min(max(x0, min(lo, hi)), max(lo, hi))
    step = abs(hi - lo)

    for iteration in range(1, max_iter + 1):
        fx = f(x)
        calls += 1
        if abs(fx) <= ftol:
            return RootResult(x, iteration, calls, abs(fx), True)

        # Contract the bracket using the sign of the residual.
        if fx < 0.0:
            lo = x
        else:
            hi = x

        dfx = fprime(x)
        calls += 1
        newton_ok = dfx != 0.0
        if newton_ok:
            candidate = x - fx / dfx
            inside = min(lo, hi) < candidate < max(lo, hi)
            # Require the step to at least halve the previous one; otherwise
            # Newton is stalling and bisection will do better.
            newton_ok = inside and abs(fx / dfx) < 0.5 * step

        if newton_ok:
            step = abs(fx / dfx)
            x_new = x - fx / dfx
        else:
            step = 0.5 * abs(hi - lo)
            x_new = 0.5 * (lo + hi)

        if abs(x_new - x) <= xtol * max(1.0, abs(x_new)):
            return RootResult(x_new, iteration, calls, abs(f(x_new)), True)
        x = x_new

    raise ConvergenceError(
        f"newton_safeguarded failed to converge in {max_iter} iterations "
        f"(last iterate {x!r}, residual {f(x)!r})"
    )
