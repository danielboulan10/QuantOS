r"""Unconstrained and simply-constrained minimisation.

Contents
--------
* :func:`nelder_mead` -- derivative-free simplex search. The default for
  likelihood surfaces whose gradients are recursive and error-prone (GARCH, OU,
  SABR calibration).
* :func:`bfgs` -- quasi-Newton with a strong-Wolfe line search, for smooth
  objectives where a gradient is available or a finite-difference one is
  affordable.
* :func:`projected_gradient` -- for problems with a simple projection
  (simplex, box), used by the long-only portfolio optimisers.
* :func:`project_to_simplex` -- exact Euclidean projection onto
  :math:`\\{w : w \\ge 0, \\sum w = 1\\}` in :math:`O(n \\log n)`.

On not writing a general-purpose optimiser
------------------------------------------
These are deliberately modest, well-understood algorithms rather than an
interior-point solver. Every optimisation problem in QuantOS is either
low-dimensional (calibration: 2-5 parameters) or has exploitable structure
(portfolio weights: a simplex projection, or a closed form). Reaching for a
generic constrained solver would add a heavy dependency and obscure the fact
that the structure was there to be used.

References
----------
Nelder, J. A. & Mead, R. (1965), *Computer Journal* 7(4), 308-313.
Nocedal, J. & Wright, S. (2006), *Numerical Optimization* (2nd ed.), ch. 3, 6.
Duchi, J. et al. (2008), "Efficient projections onto the l1-ball", ICML.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "OptimizeResult",
    "bfgs",
    "nelder_mead",
    "numerical_gradient",
    "project_to_box",
    "project_to_simplex",
    "projected_gradient",
]


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of a minimisation."""

    x: NDArray[np.float64]
    fun: float
    iterations: int
    function_calls: int
    converged: bool
    message: str = ""

    def __repr__(self) -> str:  # pragma: no cover - display
        return (
            f"OptimizeResult(fun={self.fun:.6g}, iterations={self.iterations}, "
            f"converged={self.converged})"
        )


def numerical_gradient(
    f: Callable[[NDArray[np.float64]], float], x: NDArray[np.float64], *, step: float | None = None
) -> NDArray[np.float64]:
    r"""Central-difference gradient.

    Step defaults to :math:`\varepsilon^{1/3}\max(|x_i|, 1)`. The cube root is
    not arbitrary: for central differences the truncation error is
    :math:`O(h^2)` and the round-off error :math:`O(\varepsilon/h)`, and
    balancing them gives :math:`h \propto \varepsilon^{1/3} \approx 6\times
    10^{-6}`. Using the more familiar :math:`\sqrt{\varepsilon}` (correct for
    *forward* differences) leaves about three digits of accuracy on the table.
    """
    x = np.asarray(x, dtype=float)
    eps = float(np.finfo(float).eps) ** (1.0 / 3.0)
    grad = np.empty_like(x)
    for i in range(x.size):
        h = step if step is not None else eps * max(abs(float(x[i])), 1.0)
        forward = x.copy()
        backward = x.copy()
        forward[i] += h
        backward[i] -= h
        grad[i] = (f(forward) - f(backward)) / (2.0 * h)
    return grad


def nelder_mead(
    f: Callable[[NDArray[np.float64]], float],
    x0: ArrayLike,
    *,
    max_iter: int = 2000,
    xtol: float = 1e-9,
    ftol: float = 1e-11,
    initial_step: float = 0.1,
) -> OptimizeResult:
    r"""Nelder-Mead downhill simplex minimisation.

    Purpose
        Minimise a possibly-noisy, non-differentiable, low-dimensional
        objective. QuantOS uses it for every MLE where the analytic gradient
        would involve differentiating a recursion.
    Method
        Standard reflection / expansion / contraction / shrink with the
        conventional coefficients :math:`(\rho,\chi,\gamma,\sigma) =
        (1, 2, 1/2, 1/2)`.
    Inputs
        ``f`` -- objective; ``x0`` -- starting point (its length sets the
        dimension). ``initial_step`` -- relative size of the initial simplex.
    Outputs
        :class:`OptimizeResult`.
    Complexity
        1-2 function evaluations per iteration; :math:`n+1` to build the simplex.
    Failure modes
        Not guaranteed to converge to a stationary point, and can stagnate in
        dimensions above ~10. It reports ``converged=False`` rather than raising:
        for a likelihood the best point found is still useful information.
        For smooth problems above a few dimensions prefer :func:`bfgs`.

    Example
        >>> import numpy as np
        >>> rosen = lambda v: (1 - v[0])**2 + 100 * (v[1] - v[0]**2)**2
        >>> res = nelder_mead(rosen, np.array([-1.0, 1.0]), max_iter=5000)
        >>> bool(np.allclose(res.x, [1.0, 1.0], atol=1e-4))
        True
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    if n == 0:
        raise ValueError("x0 must be non-empty")

    rho, chi, gamma, sigma = 1.0, 2.0, 0.5, 0.5

    # Build the initial simplex: perturb each coordinate in turn.
    simplex = np.empty((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        point = x0.copy()
        point[i] = point[i] + initial_step * (abs(point[i]) if point[i] != 0.0 else 1.0)
        simplex[i + 1] = point

    values = np.array([f(p) for p in simplex])
    calls = n + 1

    for iteration in range(1, max_iter + 1):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        if (
            np.max(np.abs(simplex[1:] - simplex[0])) <= xtol
            and np.max(np.abs(values[1:] - values[0])) <= ftol
        ):
            return OptimizeResult(simplex[0], float(values[0]), iteration, calls, True, "converged")

        centroid = simplex[:-1].mean(axis=0)
        worst = simplex[-1]

        reflected = centroid + rho * (centroid - worst)
        f_reflected = f(reflected)
        calls += 1

        if values[0] <= f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
            continue

        if f_reflected < values[0]:
            expanded = centroid + chi * (reflected - centroid)
            f_expanded = f(expanded)
            calls += 1
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
            continue

        # Contraction: outside if the reflection improved on the worst point.
        if f_reflected < values[-1]:
            contracted = centroid + gamma * (reflected - centroid)
            f_contracted = f(contracted)
            calls += 1
            if f_contracted <= f_reflected:
                simplex[-1], values[-1] = contracted, f_contracted
                continue
        else:
            contracted = centroid + gamma * (worst - centroid)
            f_contracted = f(contracted)
            calls += 1
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
                continue

        # Shrink everything toward the best vertex.
        simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
        values[1:] = np.array([f(p) for p in simplex[1:]])
        calls += n

    order = np.argsort(values)
    return OptimizeResult(
        simplex[order][0],
        float(values[order][0]),
        max_iter,
        calls,
        False,
        f"iteration limit {max_iter} reached",
    )


def _strong_wolfe_line_search(
    f: Callable[[NDArray[np.float64]], float],
    grad: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    direction: NDArray[np.float64],
    f0: float,
    g0: NDArray[np.float64],
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_step: float = 50.0,
) -> tuple[float, int]:
    r"""Backtracking line search satisfying the strong Wolfe conditions.

    The curvature condition :math:`|d^\top \nabla f(x+\alpha d)| \le c_2
    |d^\top \nabla f(x)|` is what keeps BFGS's Hessian approximation positive
    definite. Enforcing only sufficient decrease (Armijo) lets the update
    produce an indefinite approximation, after which the "descent" direction
    may point uphill.
    """
    slope0 = float(direction @ g0)
    if slope0 >= 0.0:
        return 0.0, 0

    alpha = 1.0
    calls = 0
    for _ in range(60):
        candidate = x + alpha * direction
        f_new = f(candidate)
        calls += 1
        if f_new > f0 + c1 * alpha * slope0:
            alpha *= 0.5
            continue
        slope_new = float(direction @ grad(candidate))
        calls += 1
        if abs(slope_new) <= -c2 * slope0:
            return alpha, calls
        if slope_new >= 0.0:
            alpha *= 0.5
        else:
            alpha = min(2.0 * alpha, max_step)
    return alpha, calls


def bfgs(
    f: Callable[[NDArray[np.float64]], float],
    x0: ArrayLike,
    *,
    grad: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    max_iter: int = 500,
    gtol: float = 1e-8,
) -> OptimizeResult:
    r"""BFGS quasi-Newton minimisation.

    Maintains an approximation :math:`H_k \approx (\nabla^2 f)^{-1}` updated by
    the BFGS formula, so no Hessian is ever formed or inverted. Superlinear
    convergence near a minimum.

    The update is **skipped** when :math:`s^\top y \le 0`, which would otherwise
    destroy positive definiteness of :math:`H`. That guard is why this
    implementation does not blow up on the mildly non-convex likelihoods it is
    applied to.

    ``grad`` defaults to :func:`numerical_gradient`, which costs :math:`2n`
    function evaluations per gradient -- acceptable below ~50 dimensions.

    Example
        >>> import numpy as np
        >>> res = bfgs(lambda v: float((v[0]-3)**2 + (v[1]+1)**2), np.zeros(2))
        >>> np.round(res.x, 6).tolist()
        [3.0, -1.0]
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    n = x.size
    gradient = grad if grad is not None else (lambda v: numerical_gradient(f, v))

    h = np.eye(n)
    fx = f(x)
    g = gradient(x)
    calls = 1 + (2 * n if grad is None else 1)

    for iteration in range(1, max_iter + 1):
        if float(np.max(np.abs(g))) <= gtol:
            return OptimizeResult(x, float(fx), iteration, calls, True, "gradient below tolerance")

        direction = -h @ g
        alpha, ls_calls = _strong_wolfe_line_search(f, gradient, x, direction, float(fx), g)
        calls += ls_calls
        if alpha == 0.0:
            return OptimizeResult(x, float(fx), iteration, calls, False, "line search failed")

        s = alpha * direction
        x_new = x + s
        f_new = f(x_new)
        g_new = gradient(x_new)
        calls += 1 + (2 * n if grad is None else 1)
        y = g_new - g

        sy = float(s @ y)
        if sy > 1e-12:
            # Sherman-Morrison form of the BFGS inverse-Hessian update.
            rho = 1.0 / sy
            identity = np.eye(n)
            left = identity - rho * np.outer(s, y)
            h = left @ h @ left.T + rho * np.outer(s, s)

        if abs(f_new - float(fx)) <= 1e-14 * (1.0 + abs(float(fx))):
            return OptimizeResult(x_new, float(f_new), iteration, calls, True, "objective stalled")

        x, fx, g = x_new, f_new, g_new

    return OptimizeResult(x, float(fx), max_iter, calls, False, "iteration limit reached")


def project_to_simplex(v: ArrayLike, total: float = 1.0) -> NDArray[np.float64]:
    r"""Euclidean projection onto :math:`\{w \ge 0,\ \sum w = \text{total}\}`.

    Purpose
        Enforce long-only, fully-invested portfolio weights inside an iterative
        optimiser, exactly rather than by clipping and renormalising (which is
        *not* the projection and can move the point much further than
        necessary).
    Method
        Duchi et al. (2008): sort descending, find the largest :math:`\rho`
        with :math:`u_\rho - (\text{cumsum}_\rho - \text{total})/\rho > 0`,
        then shift and clip.
    Complexity
        :math:`O(n \log n)`, dominated by the sort.

    Example
        >>> import numpy as np
        >>> w = project_to_simplex([0.5, 0.4, -0.2, 0.9])
        >>> bool(np.isclose(w.sum(), 1.0)), bool(np.all(w >= 0))
        (True, True)
    """
    v = np.asarray(v, dtype=float).ravel()
    if total <= 0:
        raise ValueError("total must be positive")
    n = v.size
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u)
    indices = np.arange(1, n + 1)
    condition = u - (cumulative - total) / indices > 0
    rho = int(indices[condition][-1])
    theta = (cumulative[rho - 1] - total) / rho
    return np.maximum(v - theta, 0.0)


def project_to_box(
    v: ArrayLike, lower: ArrayLike | float = 0.0, upper: ArrayLike | float = 1.0
) -> NDArray[np.float64]:
    """Projection onto a box, i.e. elementwise clipping."""
    return np.clip(np.asarray(v, dtype=float), lower, upper)


def projected_gradient(
    f: Callable[[NDArray[np.float64]], float],
    projection: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: ArrayLike,
    *,
    grad: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    max_iter: int = 2000,
    step: float = 0.01,
    tol: float = 1e-10,
) -> OptimizeResult:
    r"""Projected gradient descent with backtracking.

    Each iteration takes a gradient step then projects back onto the feasible
    set. Converges for convex ``f`` and convex feasible sets; used here for
    long-only minimum-variance and risk-parity problems where the projection is
    exact and cheap.

    The step size backtracks whenever the objective fails to decrease, which
    removes the need to know a Lipschitz constant in advance.
    """
    x = projection(np.asarray(x0, dtype=float).ravel())
    gradient = grad if grad is not None else (lambda v: numerical_gradient(f, v))
    fx = f(x)
    calls = 1
    current_step = step

    for iteration in range(1, max_iter + 1):
        g = gradient(x)
        calls += 2 * x.size if grad is None else 1

        improved = False
        for _ in range(40):
            candidate = projection(x - current_step * g)
            f_candidate = f(candidate)
            calls += 1
            if f_candidate < fx:
                shift = float(np.max(np.abs(candidate - x)))
                x, fx = candidate, f_candidate
                improved = True
                current_step *= 1.1
                if shift <= tol:
                    return OptimizeResult(
                        x, float(fx), iteration, calls, True, "step below tolerance"
                    )
                break
            current_step *= 0.5
        if not improved:
            return OptimizeResult(x, float(fx), iteration, calls, True, "no further descent")

    return OptimizeResult(x, float(fx), max_iter, calls, False, "iteration limit reached")
