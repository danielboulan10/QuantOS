r"""Portfolio construction.

Why the sample covariance matrix is not good enough
---------------------------------------------------
Mean-variance optimisation inverts the covariance matrix, which means it loads
most heavily on the *smallest* eigenvalues -- exactly the directions estimated
with least precision. With :math:`N` assets and :math:`T` observations the noise
in the eigenvalue spectrum scales with :math:`N/T`, so a 500-asset portfolio on
two years of daily data (:math:`N/T \approx 1`) produces an "optimal" portfolio
that is mostly an artefact of estimation error. Michaud called it an error
maximiser, and the name has stuck because it is accurate.

Three responses, all implemented here:

1. **Shrink the covariance** (:func:`ledoit_wolf_shrinkage`). Pull the sample
   estimate toward a structured target with an analytically optimal intensity.
2. **Avoid inverting it** (:func:`hierarchical_risk_parity`). HRP uses only
   correlation *distances* and a recursive bisection, never an inverse, so it is
   defined even when the covariance is singular.
3. **Constrain the solution** (``long_only=True`` throughout). Constraints act as
   implicit regularisation -- Jagannathan & Ma showed that a no-short constraint
   is equivalent to shrinking the covariance matrix.

References
----------
Markowitz, H. (1952), *J. Finance* 7(1), 77-91.
Ledoit, O. & Wolf, M. (2004), "A well-conditioned estimator for large-dimensional
    covariance matrices", *J. Multivariate Analysis* 88(2), 365-411.
Lopez de Prado, M. (2016), "Building diversified portfolios that outperform out
    of sample", *J. Portfolio Management* 42(4), 59-69.
Michaud, R. O. (1989), "The Markowitz optimization enigma", *Financial Analysts
    Journal* 45(1), 31-42.
Jagannathan, R. & Ma, T. (2003), *J. Finance* 58(4), 1651-1683.
Kelly, J. L. (1956), *Bell System Technical Journal* 35(4), 917-926.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.linalg import (
    condition_number,
    correlation_from_covariance,
    effective_rank,
    nearest_positive_definite,
)

__all__ = [
    "PortfolioSolution",
    "diversification_ratio",
    "hierarchical_risk_parity",
    "kelly_weights",
    "ledoit_wolf_shrinkage",
    "maximum_sharpe",
    "mean_variance",
    "minimum_variance",
    "risk_contributions",
    "risk_parity",
]


@dataclass(frozen=True)
class PortfolioSolution:
    """Optimal weights with the diagnostics needed to judge them."""

    weights: NDArray[np.float64]
    expected_return: float
    volatility: float
    objective: str
    #: Condition number of the covariance actually used.
    covariance_condition: float = float("nan")
    #: Ledoit-Wolf shrinkage intensity applied, if any.
    shrinkage: float = 0.0
    converged: bool = True
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def sharpe(self) -> float:
        return (
            float(self.expected_return / self.volatility) if self.volatility > 0 else float("nan")
        )

    @property
    def effective_positions(self) -> float:
        r"""Inverse Herfindahl index, :math:`1/\sum w_i^2`.

        The number of *genuinely independent* bets. A nominally 500-name
        portfolio with an effective count of 4 is a concentrated position wearing
        a diversified label, and this number says so immediately.
        """
        squared = float(np.sum(self.weights**2))
        return float(1.0 / squared) if squared > 0 else 0.0

    @property
    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.weights)))

    @property
    def net_exposure(self) -> float:
        return float(np.sum(self.weights))


def ledoit_wolf_shrinkage(returns: ArrayLike) -> tuple[NDArray[np.float64], float]:
    r"""Ledoit-Wolf shrinkage of the sample covariance toward a scaled identity.

    Purpose
        Produce a covariance matrix that is well-conditioned and invertible even
        when :math:`T < N`, with **no tuning parameter**.
    Method
        Shrink toward :math:`F = \bar{v} I` where :math:`\bar{v}` is the average
        sample variance:

        .. math:: \hat\Sigma = \delta F + (1-\delta) S

        The optimal intensity :math:`\delta^{*}` minimises expected squared
        Frobenius error and is available in closed form as
        :math:`\delta^{*} = \frac{\pi - \rho}{\gamma T}` clipped to
        :math:`[0,1]`, where :math:`\pi` estimates the sum of asymptotic
        variances of the sample covariance entries and :math:`\gamma` the
        misspecification of the target.

        That the intensity is *derived* rather than chosen is the whole point: a
        shrinkage parameter picked by cross-validation on the same data
        reintroduces exactly the selection bias shrinkage was meant to remove.
    Inputs
        ``returns`` -- ``(T, N)`` matrix.
    Outputs
        ``(shrunk_covariance, delta)``. Inspect ``delta``: values near 1 mean the
        sample covariance carried almost no usable information, which is itself
        the finding.
    Complexity
        :math:`O(TN^2)`.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # 60 assets, only 80 observations: the sample estimate is near-singular.
        >>> r = rng.standard_normal((80, 60)) * 0.01
        >>> cov, delta = ledoit_wolf_shrinkage(r)
        >>> bool(0.0 < delta <= 1.0)
        True
        >>> from quantos.core.linalg import condition_number
        >>> bool(condition_number(cov) < condition_number(np.cov(r, rowvar=False)))
        True
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a 2-D (T, N) array")
    t, n = x.shape
    if t < 2:
        raise ValueError("need at least 2 observations")

    centred = x - x.mean(axis=0, keepdims=True)
    sample = centred.T @ centred / t

    mean_variance_ = float(np.trace(sample) / n)
    target = mean_variance_ * np.eye(n)

    # gamma: squared Frobenius distance between sample and target.
    gamma = float(np.sum((sample - target) ** 2))
    # pi: sum of asymptotic variances of the sample covariance entries.
    squared = (centred**2).T @ (centred**2) / t
    pi = float(np.sum(squared - sample**2))
    # rho: the target's own estimation error. For a scaled-identity target only
    # the diagonal contributes, since the off-diagonal target entries are exactly
    # zero and carry no estimation error.
    rho = float(np.sum(np.diag(squared) - np.diag(sample) ** 2))

    if gamma <= 0:
        return sample, 0.0
    delta = float(np.clip((pi - rho) / (gamma * t), 0.0, 1.0))
    return delta * target + (1.0 - delta) * sample, delta


def _prepare(
    covariance: ArrayLike | None,
    returns: ArrayLike | None,
    shrink: bool,
) -> tuple[NDArray[np.float64], float]:
    """Resolve a usable covariance matrix and report the shrinkage applied."""
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("covariance must be square")
        return nearest_positive_definite(cov), 0.0
    if returns is None:
        raise ValueError("supply either covariance or returns")
    if shrink:
        cov, delta = ledoit_wolf_shrinkage(returns)
        return nearest_positive_definite(cov), delta
    return nearest_positive_definite(np.cov(np.asarray(returns, dtype=float), rowvar=False)), 0.0


def minimum_variance(
    covariance: ArrayLike | None = None,
    *,
    returns: ArrayLike | None = None,
    long_only: bool = True,
    shrink: bool = True,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    r"""Global minimum-variance portfolio.

    Unconstrained the solution is closed-form,
    :math:`w = \Sigma^{-1}\mathbf{1} / (\mathbf{1}'\Sigma^{-1}\mathbf{1})`.
    With ``long_only`` the problem becomes a quadratic program over the simplex
    and is solved by projected gradient
    (:func:`~quantos.core.optimize.minimize.projected_gradient`), whose exact
    simplex projection keeps every iterate feasible.

    Example
        >>> import numpy as np
        >>> cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        >>> sol = minimum_variance(cov)
        >>> bool(sol.weights[0] > sol.weights[1])   # tilt to the lower-vol asset
        True
        >>> round(float(sol.weights.sum()), 10)
        1.0
    """
    from quantos.core.optimize.minimize import project_to_simplex, projected_gradient

    cov, delta = _prepare(covariance, returns, shrink)
    n = cov.shape[0]
    ones = np.ones(n)

    if not long_only:
        inverse_ones = np.linalg.solve(cov, ones)
        weights = inverse_ones / float(ones @ inverse_ones)
    else:

        def objective(w: NDArray[np.float64]) -> float:
            return float(w @ cov @ w)

        def gradient(w: NDArray[np.float64]) -> NDArray[np.float64]:
            return 2.0 * cov @ w

        def projection(w: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.minimum(project_to_simplex(w), max_weight)

        result = projected_gradient(
            objective, projection, ones / n, grad=gradient, max_iter=3000, step=1.0
        )
        weights = project_to_simplex(result.x)

    volatility = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    expected = (
        float(weights @ np.asarray(returns, dtype=float).mean(axis=0))
        if returns is not None
        else 0.0
    )
    return PortfolioSolution(
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        objective="minimum_variance",
        covariance_condition=condition_number(cov),
        shrinkage=delta,
        detail={"effective_rank": effective_rank(cov)},
    )


def mean_variance(
    expected_returns: ArrayLike,
    covariance: ArrayLike | None = None,
    *,
    returns: ArrayLike | None = None,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    shrink: bool = True,
) -> PortfolioSolution:
    r"""Mean-variance optimal weights.

    Maximises :math:`\mu'w - \frac{\lambda}{2} w'\Sigma w` subject to
    :math:`\mathbf{1}'w = 1` (and :math:`w \ge 0` if ``long_only``).

    A warning the closed form invites people to ignore: this objective is far
    more sensitive to ``expected_returns`` than to the covariance. Errors in
    :math:`\mu` of a magnitude that is completely unavoidable in practice swamp
    any refinement of :math:`\Sigma`, which is why practitioners so often end up
    at :func:`minimum_variance` or :func:`hierarchical_risk_parity` -- both of
    which need no return forecast at all.
    """
    from quantos.core.optimize.minimize import project_to_simplex, projected_gradient

    mu = np.asarray(expected_returns, dtype=float).ravel()
    cov, delta = _prepare(covariance, returns, shrink)
    n = cov.shape[0]
    if mu.size != n:
        raise ValueError("expected_returns length must match the covariance dimension")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")

    if not long_only:
        # Analytic solution of the equality-constrained problem via Lagrange.
        inv_mu = np.linalg.solve(cov, mu)
        inv_ones = np.linalg.solve(cov, np.ones(n))
        a = float(np.ones(n) @ inv_ones)
        weights = (
            inv_mu / risk_aversion
            + inv_ones * (1.0 - float(np.ones(n) @ inv_mu) / risk_aversion) / a
        )
    else:

        def objective(w: NDArray[np.float64]) -> float:
            return float(-(mu @ w) + 0.5 * risk_aversion * (w @ cov @ w))

        def gradient(w: NDArray[np.float64]) -> NDArray[np.float64]:
            return -mu + risk_aversion * cov @ w

        result = projected_gradient(
            objective,
            project_to_simplex,
            np.ones(n) / n,
            grad=gradient,
            max_iter=3000,
            step=0.5,
        )
        weights = project_to_simplex(result.x)

    return PortfolioSolution(
        weights=weights,
        expected_return=float(mu @ weights),
        volatility=float(np.sqrt(max(weights @ cov @ weights, 0.0))),
        objective="mean_variance",
        covariance_condition=condition_number(cov),
        shrinkage=delta,
        detail={"risk_aversion": risk_aversion},
    )


def maximum_sharpe(
    expected_returns: ArrayLike,
    covariance: ArrayLike | None = None,
    *,
    returns: ArrayLike | None = None,
    risk_free: float = 0.0,
    long_only: bool = True,
    shrink: bool = True,
) -> PortfolioSolution:
    r"""Maximum-Sharpe (tangency) portfolio.

    Unconstrained, :math:`w \propto \Sigma^{-1}(\mu - r_f\mathbf{1})`. Long-only,
    the Sharpe ratio is maximised directly on the simplex -- the objective is not
    concave there, so the projected-gradient result is a local optimum; starting
    from equal weights is a deliberate choice for reproducibility.
    """
    from quantos.core.optimize.minimize import project_to_simplex, projected_gradient

    mu = np.asarray(expected_returns, dtype=float).ravel()
    cov, delta = _prepare(covariance, returns, shrink)
    n = cov.shape[0]
    excess = mu - risk_free

    if not long_only:
        raw = np.linalg.solve(cov, excess)
        total = float(np.sum(raw))
        weights = raw / total if total != 0 else np.ones(n) / n
    else:

        def negative_sharpe(w: NDArray[np.float64]) -> float:
            vol = float(np.sqrt(max(w @ cov @ w, 1e-300)))
            return float(-(excess @ w) / vol)

        result = projected_gradient(
            negative_sharpe, project_to_simplex, np.ones(n) / n, max_iter=2000, step=0.1
        )
        weights = project_to_simplex(result.x)

    return PortfolioSolution(
        weights=weights,
        expected_return=float(mu @ weights),
        volatility=float(np.sqrt(max(weights @ cov @ weights, 0.0))),
        objective="maximum_sharpe",
        covariance_condition=condition_number(cov),
        shrinkage=delta,
    )


def risk_contributions(weights: ArrayLike, covariance: ArrayLike) -> NDArray[np.float64]:
    r"""Each asset's share of total portfolio volatility.

    .. math:: RC_i = \frac{w_i (\Sigma w)_i}{\sqrt{w'\Sigma w}}

    These sum exactly to the portfolio volatility (Euler's theorem for the
    homogeneous-of-degree-one risk measure), which is what makes them a valid
    *decomposition* rather than a heuristic attribution.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    volatility = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if volatility == 0:
        return np.zeros_like(w)
    return w * (cov @ w) / volatility


def risk_parity(
    covariance: ArrayLike | None = None,
    *,
    returns: ArrayLike | None = None,
    shrink: bool = True,
    max_iter: int = 2000,
) -> PortfolioSolution:
    r"""Equal-risk-contribution portfolio.

    Finds :math:`w > 0` with :math:`\mathbf{1}'w = 1` such that every asset
    contributes equally to portfolio volatility, i.e.
    :math:`w_i(\Sigma w)_i` is constant in :math:`i`.

    Iteration
        The obvious update :math:`w_i \leftarrow (\Sigma w)_i^{-1}` has the right
        fixed point but **oscillates** rather than converging. On
        :math:`\Sigma = \mathrm{diag}(0.04, 0.16)` it flips between
        :math:`(0.8, 0.2)` and :math:`(0.5, 0.5)` forever and returns whichever
        iterate the loop happened to stop on -- a 1:1 weighting where the correct
        answer is the 2:1 inverse-volatility split. Correlated inputs damp the
        oscillation enough to hide the problem, which is why a diagonal test case
        is the one worth having.

        The damped update used instead,

        .. math:: w_i \leftarrow \text{normalise}\!\left(
                  \sqrt{w_i / (\Sigma w)_i}\right),

        takes a geometric mean of the current iterate and the target. It has the
        same fixed point -- squaring gives :math:`w_i(\Sigma w)_i = c^2` -- and it
        converges monotonically. On the diagonal case it lands exactly on the
        solution in a single step. No matrix inverse is required either way.

    Requires no return forecast, which is its practical appeal.

    Example
        >>> import numpy as np
        >>> cov = np.diag([0.04, 0.16])         # vols of 20% and 40%
        >>> sol = risk_parity(cov)
        >>> # With zero correlation, weights go as the inverse of volatility: 2:1
        >>> round(float(sol.weights[0] / sol.weights[1]), 4)
        2.0
    """
    cov, delta = _prepare(covariance, returns, shrink)
    n = cov.shape[0]
    w = np.ones(n) / n

    iteration = 0
    for iteration in range(max_iter):  # noqa: B007
        marginal = np.maximum(cov @ w, 1e-300)
        raw = np.sqrt(w / marginal)
        updated = raw / float(np.sum(raw))
        if float(np.max(np.abs(updated - w))) < 1e-14:
            w = updated
            break
        w = updated

    contributions = risk_contributions(w, cov)
    total = float(np.sum(contributions))
    dispersion = float(np.std(contributions / total)) if total > 0 else float("nan")
    return PortfolioSolution(
        weights=w,
        expected_return=(
            float(w @ np.asarray(returns, dtype=float).mean(axis=0)) if returns is not None else 0.0
        ),
        volatility=float(np.sqrt(max(w @ cov @ w, 0.0))),
        objective="risk_parity",
        covariance_condition=condition_number(cov),
        shrinkage=delta,
        detail={"risk_contribution_dispersion": dispersion, "iterations": float(iteration)},
    )


def hierarchical_risk_parity(
    covariance: ArrayLike | None = None,
    *,
    returns: ArrayLike | None = None,
    shrink: bool = False,
) -> PortfolioSolution:
    r"""Hierarchical Risk Parity (Lopez de Prado 2016).

    Purpose
        Allocate without inverting -- or even requiring the invertibility of --
        the covariance matrix.
    Method
        Three stages:

        1. **Tree clustering.** Convert correlations to the distance
           :math:`d_{ij} = \sqrt{\tfrac12(1-\rho_{ij})}`, which is a proper
           metric, then build a hierarchy by single linkage.
        2. **Quasi-diagonalisation.** Reorder the covariance so that similar
           assets are adjacent, concentrating mass near the diagonal.
        3. **Recursive bisection.** Split the ordered list in two and allocate
           between the halves in inverse proportion to their cluster variances;
           recurse.

    Why it holds up out of sample
        HRP never inverts the covariance, so it is immune to the small-eigenvalue
        amplification that makes Markowitz portfolios unstable, and it is defined
        for :math:`T < N`. It gives up any claim to in-sample optimality in
        exchange, which is the correct trade when the inputs are estimates.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # 40 assets, 30 observations: the sample covariance is singular.
        >>> r = rng.standard_normal((30, 40)) * 0.01
        >>> sol = hierarchical_risk_parity(returns=r)
        >>> bool(np.all(sol.weights > 0)), round(float(sol.weights.sum()), 10)
        (True, 1.0)
    """
    cov, delta = _prepare(covariance, returns, shrink)
    n = cov.shape[0]
    if n == 1:
        return PortfolioSolution(
            weights=np.ones(1),
            expected_return=0.0,
            volatility=float(np.sqrt(cov[0, 0])),
            objective="hrp",
        )

    corr = correlation_from_covariance(cov)
    distance = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    order = _single_linkage_order(distance)
    weights = _recursive_bisection(cov, order)

    return PortfolioSolution(
        weights=weights,
        expected_return=(
            float(weights @ np.asarray(returns, dtype=float).mean(axis=0))
            if returns is not None
            else 0.0
        ),
        volatility=float(np.sqrt(max(weights @ cov @ weights, 0.0))),
        objective="hierarchical_risk_parity",
        covariance_condition=condition_number(cov),
        shrinkage=delta,
        detail={"effective_rank": effective_rank(cov)},
    )


def _single_linkage_order(distance: NDArray[np.float64]) -> list[int]:
    """Single-linkage clustering, returning the quasi-diagonalising leaf order.

    Implemented directly rather than via SciPy's ``linkage``/``dendrogram``: the
    algorithm is a dozen lines, and it keeps the NumPy-only runtime promise.
    """
    n = distance.shape[0]
    # Each cluster is a list of original indices; merge the closest pair until one.
    clusters: list[list[int]] = [[i] for i in range(n)]
    work = distance.copy()
    np.fill_diagonal(work, np.inf)

    while len(clusters) > 1:
        flat = int(np.argmin(work))
        i, j = divmod(flat, work.shape[1])
        if i > j:
            i, j = j, i
        # Merge j into i, keeping the concatenated leaf order.
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
        # Single linkage: the distance to a merged cluster is the minimum.
        merged = np.minimum(work[i], work[j])
        work[i] = merged
        work[:, i] = merged
        work = np.delete(np.delete(work, j, axis=0), j, axis=1)
        np.fill_diagonal(work, np.inf)

    return clusters[0]


def _cluster_variance(cov: NDArray[np.float64], indices: list[int]) -> float:
    """Variance of the inverse-variance-weighted sub-portfolio.

    Inverse-variance weighting *within* a cluster is HRP's bottom-level
    allocation; it is the analytic minimum-variance solution when correlations
    are ignored, and ignoring them here is deliberate -- the hierarchy has
    already grouped correlated assets together.
    """
    block = cov[np.ix_(indices, indices)]
    diagonal = np.diag(block)
    inverse = 1.0 / np.where(diagonal > 0, diagonal, 1e-300)
    w = inverse / float(np.sum(inverse))
    return float(w @ block @ w)


def _recursive_bisection(cov: NDArray[np.float64], order: list[int]) -> NDArray[np.float64]:
    """Allocate top-down, splitting inversely to cluster variance."""
    weights = np.ones(cov.shape[0])
    clusters: list[list[int]] = [order]

    while clusters:
        next_level: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            total = var_left + var_right
            # Inverse-variance split between the two halves.
            alpha = 1.0 - var_left / total if total > 0 else 0.5
            for index in left:
                weights[index] *= alpha
            for index in right:
                weights[index] *= 1.0 - alpha
            next_level.extend([left, right])
        clusters = next_level

    total = float(np.sum(weights))
    return weights / total if total > 0 else weights


def kelly_weights(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    *,
    fraction: float = 1.0,
    max_leverage: float | None = None,
) -> NDArray[np.float64]:
    r"""Continuous-time Kelly (growth-optimal) weights.

    .. math:: w^{*} = \frac{1}{\text{fraction}^{-1}} \Sigma^{-1}\mu

    Purpose
        Maximise the expected logarithm of wealth, which maximises the long-run
        growth rate.
    Why ``fraction`` defaults to 1 but should usually not be
        Full Kelly maximises growth but has brutal properties on the way: the
        expected maximum drawdown of a full-Kelly strategy approaches 100%, and
        the time spent below any prior peak is enormous. It is also acutely
        sensitive to errors in :math:`\mu` -- overestimating the edge by a factor
        of two produces *double* Kelly, which has **negative** expected log
        growth. Practitioners use a half or quarter, giving up 25% of growth for a
        large reduction in drawdown. ``fraction=0.5`` is the defensible default
        for real capital; 1.0 is here because it is the mathematical object.
    Failure modes
        Raises if the covariance is singular. ``max_leverage`` rescales the
        result if the gross exposure exceeds it.
    """
    mu = np.asarray(expected_returns, dtype=float).ravel()
    cov = nearest_positive_definite(np.asarray(covariance, dtype=float))
    if fraction <= 0:
        raise ValueError("fraction must be positive")
    weights = fraction * np.linalg.solve(cov, mu)
    if max_leverage is not None:
        gross = float(np.sum(np.abs(weights)))
        if gross > max_leverage > 0:
            weights = weights * (max_leverage / gross)
    return weights


def diversification_ratio(weights: ArrayLike, covariance: ArrayLike) -> float:
    r"""Weighted average volatility divided by portfolio volatility.

    .. math:: DR = \frac{\sum_i w_i \sigma_i}{\sqrt{w'\Sigma w}}

    Equals 1 for a single asset (or perfectly correlated ones) and grows with
    genuine diversification. A cleaner summary than counting positions.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    individual = float(np.sum(np.abs(w) * np.sqrt(np.diag(cov))))
    portfolio = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return float(individual / portfolio) if portfolio > 0 else float("nan")
