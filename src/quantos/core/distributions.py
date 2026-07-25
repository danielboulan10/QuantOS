"""Probability distributions built on :mod:`quantos.core.special`.

Every distribution implements the :class:`Distribution` protocol, so any code
that needs "a distribution" -- a Monte Carlo engine, a hypothesis test's null,
a GARCH innovation model, a VaR calculator -- can accept any of them. Adding a
new distribution requires implementing one class and nothing else changes.
That is the single extensibility promise this module makes.

Design choices worth defending
------------------------------
*Sampling is by inverse transform where a closed-form quantile exists, and by a
purpose-built algorithm otherwise.* Inverse transform is not always fastest,
but it is *monotone in the underlying uniform*, which means antithetic and
quasi-Monte-Carlo variance reduction (:mod:`quantos.core.montecarlo`) compose
with it correctly. A Box-Muller or ziggurat sampler silently breaks QMC.

*Log-densities are primary.* :meth:`Distribution.logpdf` is the method that
must be numerically careful; :meth:`Distribution.pdf` is defined as its
exponential. Maximum-likelihood estimation lives entirely in log space, and a
distribution that only exposes ``pdf`` forces every MLE to compute
``log(exp(...))``.

References
----------
Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer.
Marsaglia, G. & Tsang, W. (2000), "A simple method for generating gamma
    variables", *ACM Trans. Math. Softw.* 26(3), 363-372.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core import special as sf

__all__ = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "ChiSquare",
    "Distribution",
    "Exponential",
    "FisherF",
    "Gamma",
    "Laplace",
    "Normal",
    "Poisson",
    "SkewNormal",
    "StudentT",
    "Uniform",
]


class Distribution(abc.ABC):
    """Abstract base for univariate distributions.

    Subclasses must supply :meth:`logpdf`, :meth:`cdf`, :meth:`ppf`,
    :meth:`sample`, :attr:`mean` and :attr:`variance`. Everything else --
    ``pdf``, ``sf``, ``std``, ``entropy`` estimates, moment checks -- follows.
    """

    @abc.abstractmethod
    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Log probability density (or mass) at ``x``."""

    @abc.abstractmethod
    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Cumulative distribution :math:`F(x) = P(X \le x)`."""

    @abc.abstractmethod
    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        r"""Quantile function :math:`F^{-1}(p)`."""

    @abc.abstractmethod
    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw variates. ``rng`` is required -- there is no global default."""

    @property
    @abc.abstractmethod
    def mean(self) -> float:
        """First moment, or ``nan`` where undefined."""

    @property
    @abc.abstractmethod
    def variance(self) -> float:
        """Second central moment, or ``nan``/``inf`` where undefined."""

    # -- derived ---------------------------------------------------------- #
    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Density, as ``exp(logpdf)``."""
        return np.exp(self.logpdf(x))

    def sf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Survival function :math:`1 - F(x)`. Override for tail accuracy."""
        return 1.0 - self.cdf(x)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def loglikelihood(self, x: ArrayLike) -> float:
        """Total log-likelihood of an i.i.d. sample."""
        return float(np.sum(self.logpdf(x)))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({fields})"


@dataclass(frozen=True)
class Normal(Distribution):
    r"""Gaussian :math:`\mathcal{N}(\mu, \sigma^2)`.

    Example
        >>> import numpy as np
        >>> float(np.round(Normal().cdf(1.96), 4))
        0.975
    """

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        z = (np.asarray(x, dtype=float) - self.mu) / self.sigma
        return np.asarray(
            -0.5 * z * z - np.log(self.sigma) - 0.5 * np.log(2.0 * np.pi), dtype=np.float64
        )

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return sf.ndtr((np.asarray(x, dtype=float) - self.mu) / self.sigma)

    def sf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Upper tail via ``ndtr(-z)``, preserving relative precision."""
        return sf.ndtr(-(np.asarray(x, dtype=float) - self.mu) / self.sigma)

    def logcdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return sf.log_ndtr((np.asarray(x, dtype=float) - self.mu) / self.sigma)

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        return self.mu + self.sigma * sf.ndtri(p)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.normal(self.mu, self.sigma, size=size)

    @property
    def mean(self) -> float:
        return float(self.mu)

    @property
    def variance(self) -> float:
        return float(self.sigma**2)


@dataclass(frozen=True)
class StudentT(Distribution):
    r"""Student-t with ``df`` degrees of freedom, location ``loc``, scale ``scale``.

    The workhorse for financial return modelling: it reproduces the fat tails
    that a Gaussian cannot, with a single interpretable parameter. Empirical
    equity return ``df`` typically lands between 3 and 6, which is exactly the
    range where the variance exists but the kurtosis does not.
    """

    df: float
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.df <= 0:
            raise ValueError(f"df must be positive, got {self.df}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        v = self.df
        const = (
            sf.lgamma(np.array(0.5 * (v + 1.0)))
            - sf.lgamma(np.array(0.5 * v))
            - 0.5 * np.log(v * np.pi)
        )
        return np.asarray(
            const - 0.5 * (v + 1.0) * np.log1p(z * z / v) - np.log(self.scale), dtype=np.float64
        )

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""CDF via the incomplete beta identity.

        .. math::
            F(t) = 1 - \tfrac12 I_{\nu/(\nu+t^2)}(\tfrac{\nu}{2}, \tfrac12)
            \quad (t > 0)

        with the reflection :math:`F(-t) = 1 - F(t)` for the left tail. Routing
        both tails through the *small* branch of ``betainc`` keeps tail
        p-values relatively accurate.
        """
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        v = self.df
        half_ib = 0.5 * sf.betainc(0.5 * v, 0.5, v / (v + z * z))
        return np.where(z > 0.0, 1.0 - half_ib, half_ib)

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        """Quantile by bisection on :meth:`cdf` (monotone, so unconditionally safe)."""
        from quantos.core.optimize.roots import bisect_vectorised

        p = np.asarray(p, dtype=float)
        lo = np.full(p.shape, -1e4)
        hi = np.full(p.shape, 1e4)
        z = bisect_vectorised(lambda t: StudentT(self.df).cdf(t) - p, lo, hi, tol=1e-13)
        out = self.loc + self.scale * z
        return np.where(p <= 0.0, -np.inf, np.where(p >= 1.0, np.inf, out))

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return self.loc + self.scale * rng.standard_t(self.df, size=size)

    @property
    def mean(self) -> float:
        return float(self.loc) if self.df > 1 else float("nan")

    @property
    def variance(self) -> float:
        if self.df > 2:
            return float(self.scale**2 * self.df / (self.df - 2.0))
        return float("inf") if self.df > 1 else float("nan")

    @property
    def excess_kurtosis(self) -> float:
        """Excess kurtosis; infinite for ``df <= 4``, undefined for ``df <= 2``."""
        if self.df > 4:
            return float(6.0 / (self.df - 4.0))
        return float("inf") if self.df > 2 else float("nan")


@dataclass(frozen=True)
class ChiSquare(Distribution):
    r""":math:`\chi^2_k`. The null distribution of most portmanteau tests here."""

    df: float

    def __post_init__(self) -> None:
        if self.df <= 0:
            raise ValueError(f"df must be positive, got {self.df}")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        k = self.df
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                (0.5 * k - 1.0) * np.log(x)
                - 0.5 * x
                - 0.5 * k * np.log(2.0)
                - sf.lgamma(np.array(0.5 * k))
            )
        return np.where(x > 0.0, out, -np.inf)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return sf.gammainc(0.5 * self.df, 0.5 * np.asarray(x, dtype=float))

    def sf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Upper tail directly from ``gammaincc`` -- this is the p-value path."""
        return sf.gammaincc(0.5 * self.df, 0.5 * np.asarray(x, dtype=float))

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        from quantos.core.optimize.roots import bisect_vectorised

        p = np.asarray(p, dtype=float)
        lo = np.full(p.shape, 1e-300)
        hi = np.full(p.shape, max(1e4, 100.0 * self.df))
        return bisect_vectorised(lambda t: self.cdf(t) - p, lo, hi, tol=1e-12)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.chisquare(self.df, size=size)

    @property
    def mean(self) -> float:
        return float(self.df)

    @property
    def variance(self) -> float:
        return float(2.0 * self.df)


@dataclass(frozen=True)
class FisherF(Distribution):
    r"""Fisher-Snedecor :math:`F(d_1, d_2)`; null for regression F-tests."""

    dfn: float
    dfd: float

    def __post_init__(self) -> None:
        if self.dfn <= 0 or self.dfd <= 0:
            raise ValueError("both degrees of freedom must be positive")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        d1, d2 = self.dfn, self.dfd
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                0.5 * d1 * np.log(d1 * x)
                + 0.5 * d2 * np.log(d2)
                - 0.5 * (d1 + d2) * np.log(d1 * x + d2)
                - np.log(x)
                - sf.log_beta(np.array(0.5 * d1), np.array(0.5 * d2))
            )
        return np.where(x > 0.0, out, -np.inf)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        d1, d2 = self.dfn, self.dfd
        z = np.where(x > 0.0, d1 * x / (d1 * x + d2), 0.0)
        return np.where(x > 0.0, sf.betainc(0.5 * d1, 0.5 * d2, z), 0.0)

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        from quantos.core.optimize.roots import bisect_vectorised

        p = np.asarray(p, dtype=float)
        lo = np.full(p.shape, 1e-300)
        hi = np.full(p.shape, 1e6)
        return bisect_vectorised(lambda t: self.cdf(t) - p, lo, hi, tol=1e-12)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.f(self.dfn, self.dfd, size=size)

    @property
    def mean(self) -> float:
        return float(self.dfd / (self.dfd - 2.0)) if self.dfd > 2 else float("nan")

    @property
    def variance(self) -> float:
        d1, d2 = self.dfn, self.dfd
        if d2 <= 4:
            return float("nan")
        return float(2.0 * d2**2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) ** 2 * (d2 - 4.0)))


@dataclass(frozen=True)
class Exponential(Distribution):
    r"""Exponential with *rate* :math:`\lambda`. Inter-arrival times of a Poisson process."""

    rate: float = 1.0

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0.0, np.log(self.rate) - self.rate * x, -np.inf)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        return np.where(x > 0.0, -np.expm1(-self.rate * x), 0.0)

    def sf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        return np.where(x > 0.0, np.exp(-self.rate * x), 1.0)

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=float)
        return -np.log1p(-p) / self.rate

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.exponential(1.0 / self.rate, size=size)

    @property
    def mean(self) -> float:
        return 1.0 / self.rate

    @property
    def variance(self) -> float:
        return 1.0 / self.rate**2


@dataclass(frozen=True)
class Uniform(Distribution):
    """Continuous uniform on ``[low, high]``."""

    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError("require high > low")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        inside = (x >= self.low) & (x <= self.high)
        return np.where(inside, -np.log(self.high - self.low), -np.inf)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        return np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        return self.low + np.asarray(p, dtype=float) * (self.high - self.low)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(self.low, self.high, size=size)

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def variance(self) -> float:
        return (self.high - self.low) ** 2 / 12.0


@dataclass(frozen=True)
class Laplace(Distribution):
    """Laplace, also called the double-exponential distribution.

    Heavier tails than a Gaussian, but still light enough for a closed-form MLE
    (the location estimate is simply the sample median).
    """

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        return np.asarray(
            -np.abs(x - self.loc) / self.scale - np.log(2.0 * self.scale), dtype=np.float64
        )

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        return np.where(z < 0.0, 0.5 * np.exp(z), 1.0 - 0.5 * np.exp(-z))

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=float)
        return np.where(
            p < 0.5,
            self.loc + self.scale * np.log(2.0 * p),
            self.loc - self.scale * np.log(2.0 * (1.0 - p)),
        )

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.laplace(self.loc, self.scale, size=size)

    @property
    def mean(self) -> float:
        return float(self.loc)

    @property
    def variance(self) -> float:
        return 2.0 * self.scale**2


@dataclass(frozen=True)
class Gamma(Distribution):
    r"""Gamma with shape :math:`k` and *scale* :math:`\theta`."""

    shape: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("shape and scale must be positive")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        k, th = self.shape, self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (k - 1.0) * np.log(x) - x / th - k * np.log(th) - sf.lgamma(np.array(k))
        return np.where(x > 0.0, out, -np.inf)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return sf.gammainc(self.shape, np.asarray(x, dtype=float) / self.scale)

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        from quantos.core.optimize.roots import bisect_vectorised

        p = np.asarray(p, dtype=float)
        lo = np.full(p.shape, 1e-300)
        hi = np.full(p.shape, self.scale * max(1e4, 100.0 * self.shape))
        return bisect_vectorised(lambda t: self.cdf(t) - p, lo, hi, tol=1e-12)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.gamma(self.shape, self.scale, size=size)

    @property
    def mean(self) -> float:
        return self.shape * self.scale

    @property
    def variance(self) -> float:
        return self.shape * self.scale**2


@dataclass(frozen=True)
class Beta(Distribution):
    r"""Beta on :math:`(0,1)`. Conjugate prior for a Bernoulli rate."""

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("a and b must be positive")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                (self.a - 1.0) * np.log(x)
                + (self.b - 1.0) * np.log1p(-x)
                - sf.log_beta(np.array(self.a), np.array(self.b))
            )
        return np.where((x > 0.0) & (x < 1.0), out, -np.inf)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return sf.betainc(self.a, self.b, np.clip(np.asarray(x, dtype=float), 0.0, 1.0))

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        from quantos.core.optimize.roots import bisect_vectorised

        p = np.asarray(p, dtype=float)
        lo = np.zeros(p.shape)
        hi = np.ones(p.shape)
        return bisect_vectorised(lambda t: self.cdf(t) - p, lo, hi, tol=1e-13)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.beta(self.a, self.b, size=size)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def variance(self) -> float:
        s = self.a + self.b
        return self.a * self.b / (s * s * (s + 1.0))


@dataclass(frozen=True)
class SkewNormal(Distribution):
    r"""Azzalini skew-normal, density :math:`2\phi(z)\Phi(\alpha z)/\omega`.

    Included because equity index returns are visibly left-skewed and the
    symmetric Student-t cannot express that. ``alpha = 0`` recovers the normal.
    """

    loc: float = 0.0
    scale: float = 1.0
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        return np.asarray(
            np.log(2.0)
            - np.log(self.scale)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * z * z
            + sf.log_ndtr(self.alpha * z),
            dtype=np.float64,
        )

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """CDF by adaptive quadrature of the density (no closed form exists)."""
        from quantos.core.numerics import adaptive_quad

        x = np.asarray(x, dtype=float)
        flat = np.atleast_1d(x).ravel()
        out = np.array(
            [
                adaptive_quad(lambda t: float(self.pdf(t)), -40.0 * self.scale + self.loc, float(v))
                for v in flat
            ]
        )
        return np.asarray(np.clip(out, 0.0, 1.0).reshape(np.shape(x)), dtype=np.float64)

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        from quantos.core.optimize.roots import bisect_vectorised

        p = np.asarray(p, dtype=float)
        lo = np.full(p.shape, self.loc - 40.0 * self.scale)
        hi = np.full(p.shape, self.loc + 40.0 * self.scale)
        return bisect_vectorised(lambda t: self.cdf(t) - p, lo, hi, tol=1e-10)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        r"""Draw variates using Azzalini's construction.

        If :math:`(U,V)` are bivariate normal with correlation
        :math:`\delta = \alpha/\sqrt{1+\alpha^2}`, then
        :math:`\delta|U| + \sqrt{1-\delta^2}\,V` is skew-normal.
        """
        delta = self.alpha / np.sqrt(1.0 + self.alpha**2)
        u = rng.standard_normal(size)
        v = rng.standard_normal(size)
        z = delta * np.abs(u) + np.sqrt(1.0 - delta**2) * v
        return np.asarray(self.loc + self.scale * z, dtype=np.float64)

    @property
    def mean(self) -> float:
        delta = self.alpha / np.sqrt(1.0 + self.alpha**2)
        return float(self.loc + self.scale * delta * np.sqrt(2.0 / np.pi))

    @property
    def variance(self) -> float:
        delta = self.alpha / np.sqrt(1.0 + self.alpha**2)
        return float(self.scale**2 * (1.0 - 2.0 * delta**2 / np.pi))


# --------------------------------------------------------------------------- #
# Discrete                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bernoulli(Distribution):
    """Bernoulli(p). ``logpdf`` is a log *mass*."""

    p: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must lie in [0,1], got {self.p}")

    def logpdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore"):
            out = np.where(x == 1.0, np.log(self.p), np.log1p(-self.p))
        return np.where((x == 0.0) | (x == 1.0), out, -np.inf)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        return np.where(x < 0.0, 0.0, np.where(x < 1.0, 1.0 - self.p, 1.0))

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        return (np.asarray(p, dtype=float) > (1.0 - self.p)).astype(float)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return (rng.random(size) < self.p).astype(float)

    @property
    def mean(self) -> float:
        return self.p

    @property
    def variance(self) -> float:
        return self.p * (1.0 - self.p)


@dataclass(frozen=True)
class Binomial(Distribution):
    """Binomial(n, p)."""

    n: int
    p: float = 0.5

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must be non-negative")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must lie in [0,1]")

    def logpdf(self, k: ArrayLike) -> NDArray[np.float64]:
        k = np.asarray(k, dtype=float)
        valid = (k >= 0) & (k <= self.n) & (k == np.floor(k))
        log_choose = (
            sf.lgamma(np.array(self.n + 1.0))
            - sf.lgamma(np.clip(k, 0, None) + 1.0)
            - sf.lgamma(np.clip(self.n - k, 0, None) + 1.0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = log_choose + k * np.log(self.p) + (self.n - k) * np.log1p(-self.p)
        return np.where(valid, out, -np.inf)

    def cdf(self, k: ArrayLike) -> NDArray[np.float64]:
        r"""Exact CDF via the beta identity, with no summation.

        :math:`F(k) = I_{1-p}(n-k,\, k+1)`, so evaluation stays fast and
        accurate for ``n`` in the millions.
        """
        k = np.floor(np.asarray(k, dtype=float))
        out = sf.betainc(np.clip(self.n - k, 1e-300, None), k + 1.0, 1.0 - self.p)
        return np.where(k < 0, 0.0, np.where(k >= self.n, 1.0, out))

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=float)
        support = np.arange(self.n + 1, dtype=float)
        cdf = self.cdf(support)
        idx = np.searchsorted(cdf, np.clip(p, 0.0, 1.0), side="left")
        return np.asarray(support[np.clip(idx, 0, self.n)], dtype=np.float64)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.binomial(self.n, self.p, size=size).astype(float)

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def variance(self) -> float:
        return self.n * self.p * (1.0 - self.p)


@dataclass(frozen=True)
class Poisson(Distribution):
    r"""Poisson(:math:`\lambda`). Counts of order arrivals in a fixed window."""

    lam: float = 1.0

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError("lam must be non-negative")

    def logpdf(self, k: ArrayLike) -> NDArray[np.float64]:
        k = np.asarray(k, dtype=float)
        valid = (k >= 0) & (k == np.floor(k))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = k * np.log(self.lam) - self.lam - sf.lgamma(np.clip(k, 0, None) + 1.0)
        return np.where(valid, out, -np.inf)

    def cdf(self, k: ArrayLike) -> NDArray[np.float64]:
        r"""Exact CDF via the regularised upper incomplete gamma.

        :math:`F(k) = Q(k+1, \lambda)`, again avoiding an explicit sum.
        """
        k = np.floor(np.asarray(k, dtype=float))
        return np.where(k < 0, 0.0, sf.gammaincc(k + 1.0, self.lam))

    def ppf(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=float)
        upper = int(max(20.0, self.lam + 12.0 * np.sqrt(self.lam + 1.0)))
        support = np.arange(upper + 1, dtype=float)
        cdf = self.cdf(support)
        idx = np.searchsorted(cdf, np.clip(p, 0.0, 1.0), side="left")
        return np.asarray(support[np.clip(idx, 0, upper)], dtype=np.float64)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.poisson(self.lam, size=size).astype(float)

    @property
    def mean(self) -> float:
        return self.lam

    @property
    def variance(self) -> float:
        return self.lam
