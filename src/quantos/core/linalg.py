"""Linear algebra for covariance matrices that do not behave.

The recurring problem
---------------------
Every covariance matrix in finance is estimated, and estimated covariance
matrices are ill-conditioned or outright indefinite far more often than the
textbooks suggest:

* With ``n`` assets and ``T < n`` observations the sample covariance is
  *singular by construction* -- rank at most ``T-1``.
* Even for ``T > n``, the smallest eigenvalues are dominated by noise. Markowitz
  optimisation inverts the covariance, so it loads maximally on exactly the
  directions that are least reliable. This is why unconstrained mean-variance
  portfolios are famously unusable.
* Correlation matrices assembled from pairwise estimates, or repaired by hand,
  routinely fail to be positive semi-definite.

This module supplies the repairs and decompositions that make the rest of the
platform safe: :func:`nearest_correlation` (Higham's alternating projections),
:func:`safe_cholesky` (diagonal loading with a reported jitter), and
:func:`condition_number` so callers can *know* when they are on thin ice.

References
----------
Higham, N. J. (2002), "Computing the nearest correlation matrix -- a problem
    from finance", *IMA J. Numer. Anal.* 22(3), 329-343.
Golub, G. H. & Van Loan, C. F. (2013), *Matrix Computations* (4th ed.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PCAResult",
    "condition_number",
    "correlation_from_covariance",
    "covariance_from_correlation",
    "effective_rank",
    "is_positive_definite",
    "marchenko_pastur_edge",
    "nearest_correlation",
    "nearest_positive_definite",
    "pca",
    "ridge_solve",
    "safe_cholesky",
]


def _symmetrise(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average with the transpose. Cheap insurance against accumulated asymmetry."""
    return 0.5 * (a + a.T)


def is_positive_definite(a: ArrayLike, *, tol: float = 0.0) -> bool:
    """Whether ``a`` is symmetric positive definite.

    Tested by attempting a Cholesky factorisation rather than by inspecting
    eigenvalues: Cholesky is twice as fast and is the *operative* definition,
    since PD-ness matters precisely because something downstream wants to
    factorise.
    """
    m = np.asarray(a, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    if not np.allclose(m, m.T, atol=1e-10):
        return False
    try:
        np.linalg.cholesky(m - tol * np.eye(m.shape[0]))
    except np.linalg.LinAlgError:
        return False
    return True


@dataclass(frozen=True)
class CholeskyResult:
    """Cholesky factor with the jitter that was needed to obtain it."""

    factor: NDArray[np.float64]
    jitter: float
    attempts: int

    @property
    def was_repaired(self) -> bool:
        return self.jitter > 0.0


def safe_cholesky(a: ArrayLike, *, max_attempts: int = 12) -> CholeskyResult:
    r"""Cholesky factorisation, adding the smallest diagonal jitter that works.

    Purpose
        Correlated random-variate generation and quadratic forms need a
        factorisation. Estimated covariances often fail to admit one by a
        hair -- eigenvalues of order ``-1e-18`` from rounding. Failing outright
        would be unhelpful; silently repairing would be dishonest. So we repair
        *and report*.
    Method
        Try plain Cholesky. On failure add :math:`\varepsilon I` with
        :math:`\varepsilon` starting at ``1e-12 * trace/n`` and multiplying by
        10 each attempt.
    Outputs
        :class:`CholeskyResult`; inspect :attr:`~CholeskyResult.jitter`. A
        large jitter means the matrix was *substantially* indefinite and the
        result should be treated with suspicion -- prefer
        :func:`nearest_correlation`, which finds the genuinely closest valid
        matrix rather than the nearest one along the diagonal.
    Failure modes
        :class:`numpy.linalg.LinAlgError` if even ``1e-12 * 10^12`` scaled
        jitter fails, which means the input is badly wrong rather than slightly
        broken.

    Example
        >>> import numpy as np
        >>> a = np.array([[1.0, 1.0], [1.0, 1.0]])       # singular
        >>> res = safe_cholesky(a)
        >>> res.was_repaired
        True
    """
    m = _symmetrise(np.asarray(a, dtype=float))
    n = m.shape[0]
    scale = float(np.trace(m)) / n if n else 1.0
    try:
        return CholeskyResult(np.linalg.cholesky(m), 0.0, 1)
    except np.linalg.LinAlgError:
        pass

    jitter = 1e-12 * max(scale, 1e-300)
    for attempt in range(2, max_attempts + 2):
        try:
            factor = np.linalg.cholesky(m + jitter * np.eye(n))
            return CholeskyResult(factor, jitter, attempt)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(
        f"matrix is not repairable by diagonal loading up to jitter {jitter:g}; "
        "use nearest_positive_definite or nearest_correlation instead"
    )


def nearest_positive_definite(a: ArrayLike, *, epsilon: float = 1e-10) -> NDArray[np.float64]:
    r"""Nearest positive-definite matrix in the Frobenius norm, by eigenvalue clipping.

    Symmetrise, then clip eigenvalues at ``epsilon``. This is the exact
    Frobenius-nearest *symmetric PSD* matrix (Higham 1988), but note it does
    **not** preserve a unit diagonal -- so for a correlation matrix it produces
    something that is no longer a correlation matrix. Use
    :func:`nearest_correlation` in that case.
    """
    m = _symmetrise(np.asarray(a, dtype=float))
    values, vectors = np.linalg.eigh(m)
    return _symmetrise(vectors @ np.diag(np.maximum(values, epsilon)) @ vectors.T)


def nearest_correlation(
    a: ArrayLike, *, max_iter: int = 200, tol: float = 1e-10, epsilon: float = 1e-10
) -> NDArray[np.float64]:
    r"""Nearest valid correlation matrix by Higham's alternating projections.

    Purpose
        Repair a correlation matrix that has been perturbed -- by pairwise
        estimation over unequal samples, by expert overrides, or by stress
        scenarios -- so that it is positive semi-definite *and* still has a unit
        diagonal.
    Method
        Alternately project onto the set of PSD matrices (eigenvalue clipping)
        and onto the set of unit-diagonal matrices, with Dykstra's correction
        so the iteration converges to the true projection onto the
        intersection rather than to some point that merely satisfies both
        loosely.
    Complexity
        One symmetric eigendecomposition per iteration: :math:`O(k n^3)`.
    Failure modes
        Converges for any symmetric input. Non-symmetric input is symmetrised
        first, silently -- an asymmetric "correlation matrix" is a caller bug,
        but averaging is the only sensible interpretation.

    Example
        >>> import numpy as np
        >>> bad = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.9], [0.7, 0.9, 1.0]])
        >>> bad[0, 2] = bad[2, 0] = -0.9        # now indefinite
        >>> fixed = nearest_correlation(bad)
        >>> bool(np.all(np.linalg.eigvalsh(fixed) > -1e-9)), np.allclose(np.diag(fixed), 1)
        (True, True)
    """
    m = _symmetrise(np.asarray(a, dtype=float))
    m.shape[0]
    y = m.copy()
    delta_s = np.zeros_like(m)

    for _ in range(max_iter):
        r = y - delta_s
        # Projection onto the PSD cone.
        values, vectors = np.linalg.eigh(_symmetrise(r))
        x = vectors @ np.diag(np.maximum(values, epsilon)) @ vectors.T
        delta_s = x - r
        # Projection onto unit diagonal.
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        if np.linalg.norm(y_new - y, ord="fro") <= tol * max(1.0, np.linalg.norm(y, ord="fro")):
            y = y_new
            break
        y = y_new

    y = _symmetrise(y)
    np.fill_diagonal(y, 1.0)
    return y


def condition_number(a: ArrayLike) -> float:
    r"""2-norm condition number :math:`\lambda_{\max}/\lambda_{\min}`.

    A practical reading for covariance matrices: above about ``1e8`` the
    inverse retains fewer than 8 significant digits, and a mean-variance
    optimiser built on it is reporting noise. Above ``1/eps`` (~4.5e15) the
    matrix is numerically singular.
    """
    m = _symmetrise(np.asarray(a, dtype=float))
    values = np.linalg.eigvalsh(m)
    smallest = float(np.min(np.abs(values)))
    return float("inf") if smallest == 0.0 else float(np.max(np.abs(values)) / smallest)


def correlation_from_covariance(cov: ArrayLike) -> NDArray[np.float64]:
    """Convert a covariance matrix to a correlation matrix."""
    c = np.asarray(cov, dtype=float)
    d = np.sqrt(np.diag(c))
    if np.any(d <= 0):
        raise ValueError("covariance has a non-positive variance on the diagonal")
    out = c / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return _symmetrise(out)


def covariance_from_correlation(corr: ArrayLike, volatilities: ArrayLike) -> NDArray[np.float64]:
    """Rebuild a covariance from a correlation matrix and a volatility vector."""
    r = np.asarray(corr, dtype=float)
    v = np.asarray(volatilities, dtype=float).ravel()
    if r.shape[0] != v.size:
        raise ValueError("correlation dimension does not match the volatility vector")
    return _symmetrise(r * np.outer(v, v))


@dataclass(frozen=True)
class PCAResult:
    """Principal component decomposition of a covariance matrix."""

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]

    @property
    def explained_variance_ratio(self) -> NDArray[np.float64]:
        total = float(np.sum(self.eigenvalues))
        return self.eigenvalues / total if total > 0 else np.zeros_like(self.eigenvalues)

    @property
    def cumulative_variance_ratio(self) -> NDArray[np.float64]:
        return np.cumsum(self.explained_variance_ratio)

    def components_for_variance(self, threshold: float = 0.95) -> int:
        """Number of components needed to reach ``threshold`` explained variance."""
        return int(np.searchsorted(self.cumulative_variance_ratio, threshold) + 1)

    def project(self, x: ArrayLike, n_components: int | None = None) -> NDArray[np.float64]:
        """Project observations onto the leading components."""
        k = n_components or self.eigenvalues.size
        return np.asarray(x, dtype=float) @ self.eigenvectors[:, :k]


def pca(cov: ArrayLike) -> PCAResult:
    """Eigendecomposition of a covariance matrix, sorted by descending eigenvalue.

    Uses ``eigh`` (symmetric) rather than ``eig``: it is faster, guarantees real
    eigenvalues, and returns an orthonormal basis. Applying general ``eig`` to a
    symmetric matrix can return complex values from rounding, which then
    silently propagate.

    Example
        >>> import numpy as np
        >>> c = np.array([[4.0, 2.0], [2.0, 3.0]])
        >>> res = pca(c)
        >>> bool(res.eigenvalues[0] > res.eigenvalues[1])
        True
    """
    m = _symmetrise(np.asarray(cov, dtype=float))
    values, vectors = np.linalg.eigh(m)
    order = np.argsort(values)[::-1]
    return PCAResult(eigenvalues=values[order], eigenvectors=vectors[:, order])


def ridge_solve(a: ArrayLike, b: ArrayLike, penalty: float) -> NDArray[np.float64]:
    r"""Solve :math:`(A + \lambda I)x = b` with a Cholesky factorisation.

    The workhorse for regularised portfolio and regression problems. The
    penalty is what makes an ill-conditioned system solvable at all: it shifts
    every eigenvalue up by ``penalty``, capping the condition number at
    :math:`(\lambda_{\max}+\lambda)/\lambda`.
    """
    m = _symmetrise(np.asarray(a, dtype=float))
    rhs = np.asarray(b, dtype=float)
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    factor = safe_cholesky(m + penalty * np.eye(m.shape[0])).factor
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


def effective_rank(a: ArrayLike) -> float:
    r"""Effective rank of a matrix, via its spectral entropy.

    Defined as :math:`\exp\big(-\sum_i p_i \log p_i\big)` with
    :math:`p_i = \lambda_i/\sum\lambda`.

    A *continuous* measure of how many directions the matrix genuinely spans,
    far more informative than :func:`numpy.linalg.matrix_rank`'s integer answer.
    A 500-asset equity covariance typically has full numerical rank but an
    effective rank in the low tens -- which is the quantitative statement of
    "everything is one market factor plus noise".
    """
    m = _symmetrise(np.asarray(a, dtype=float))
    values = np.abs(np.linalg.eigvalsh(m))
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    p = values[values > 0] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def marchenko_pastur_edge(
    n_assets: int, n_observations: int, variance: float = 1.0
) -> tuple[float, float]:
    r"""Support of the Marchenko-Pastur law, :math:`[\lambda_-, \lambda_+]`.

    .. math:: \lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{q}\right)^2,
              \qquad q = n/T

    Eigenvalues of a sample correlation matrix falling *inside* this band are
    statistically indistinguishable from pure noise -- this is the theoretical
    basis for random-matrix eigenvalue clipping, and the principled answer to
    "how many principal components are real?". Only the eigenvalues above
    :math:`\lambda_+` carry signal.

    Example
        >>> lo, hi = marchenko_pastur_edge(100, 500)
        >>> round(hi, 4)                      # q = 0.2, so (1 + sqrt(0.2))^2
        2.0944
    """
    if n_assets < 1 or n_observations < 1:
        raise ValueError("dimensions must be positive")
    q = n_assets / n_observations
    root = np.sqrt(q)
    return float(variance * (1.0 - root) ** 2), float(variance * (1.0 + root) ** 2)
