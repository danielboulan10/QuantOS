r"""Heston stochastic volatility, priced by Fourier inversion — and the branch cut.

Why a dynamic model
-------------------
:mod:`quantos.research.vol_surface` fits SVI, which describes the surface at one
instant and has no dynamics: it cannot say what the surface will look like
tomorrow, cannot price a path-dependent payoff, and cannot be tested for
consistency over time. Heston is a *process*,

.. math::
   dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW^1_t \\
   dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^2_t,
   \quad d\langle W^1, W^2\rangle = \rho\,dt

so the same five parameters price every strike and maturity at once, and the
correlation :math:`\rho` *generates* the skew rather than describing it.

There is no closed-form price, but there is a closed-form **characteristic
function**, and a European option is a Fourier integral against it. That is
different mathematics from anything else in this repository: complex analysis on
the real line rather than real-valued quadrature.

The branch cut, which is the point of this module
--------------------------------------------------
The characteristic function contains a complex logarithm. Written the obvious way
-- the form in Heston's 1993 paper -- that logarithm is evaluated on its
**principal branch**, and the principal branch has a discontinuity along the
negative real axis.

For short maturities the integrand never approaches it and everything works. For
longer maturities, or large :math:`\kappa`, or strong correlation, the argument
crosses the cut and the integrand **jumps**. The integral then converges happily
to the wrong number: prices that look plausible, violate put-call parity by a few
percent, and produce a calibration that reports success while sitting on nonsense.

The failure is silent, and it is why this module exists rather than a shorter one.

The fix is one sign
-------------------
Albrecher, Mayer, Schoutens and Tistaert (2007) showed the two algebraically
equivalent forms of the same function behave completely differently in floating
point. Writing

.. math:: d = \sqrt{(\rho\xi i u - \kappa)^2 + \xi^2(iu + u^2)}, \qquad
          g_2 = \frac{\kappa - \rho\xi i u - d}{\kappa - \rho\xi i u + d}

instead of :math:`g_1 = 1/g_2` keeps :math:`|g_2| \le 1` for every :math:`u`,
which keeps the logarithm's argument off the cut. Same mathematics, stable
evaluation. :func:`characteristic_function` implements both so the difference can
be *shown* -- see ``tests/derivatives/test_heston.py``, which demonstrates the
discontinuity rather than asserting it.

Integration
-----------
The Lewis/Lipton form is used: a single integral for the option price rather than
Heston's original two probabilities. It needs one characteristic-function
evaluation per node instead of two, and the integrand decays smoothly, so
Gauss-Legendre on a truncated domain converges to machine precision in a few
hundred nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantos.derivatives.black_scholes import OptionType

__all__ = [
    "HestonParameters",
    "characteristic_function",
    "heston_implied_volatility",
    "heston_price",
]


@dataclass(frozen=True)
class HestonParameters:
    r"""The five parameters of the Heston process.

    ``kappa``
        Mean-reversion speed of variance.
    ``theta``
        Long-run variance. The surface flattens toward :math:`\sqrt{\theta}`.
    ``xi``
        Volatility of variance. Generates the *smile* -- curvature.
    ``rho``
        Correlation between the price and variance shocks. Generates the
        *skew*; negative in equities, which is why crash protection is dearer.
    ``v0``
        Initial variance, so :math:`\sqrt{v_0}` is roughly today's short-dated
        implied volatility.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0 or self.v0 <= 0:
            raise ValueError(
                f"kappa, theta, xi and v0 must be positive, got kappa={self.kappa}, "
                f"theta={self.theta}, xi={self.xi}, v0={self.v0}"
            )
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1], got {self.rho}")

    @property
    def feller_ratio(self) -> float:
        r"""The Feller quantity :math:`2\kappa\theta / \xi^2`.

        Above 1 the variance process cannot reach zero. Below it, variance
        touches zero with positive probability -- which is not an error and is
        common in fitted parameters, but it breaks naive Euler simulation and is
        worth knowing before trusting a Monte Carlo built on it.
        """
        return float(2.0 * self.kappa * self.theta / self.xi**2)

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_ratio > 1.0

    def summary(self) -> str:
        return (
            f"Heston(kappa={self.kappa:.4f}, theta={self.theta:.4f}, xi={self.xi:.4f}, "
            f"rho={self.rho:+.4f}, v0={self.v0:.4f})\n"
            f"  long-run vol {np.sqrt(self.theta):.2%}, spot vol {np.sqrt(self.v0):.2%}, "
            f"Feller {self.feller_ratio:.3f}"
            f"{'' if self.feller_satisfied else '  (variance can reach zero)'}"
        )


def characteristic_function(
    u: NDArray[np.complex128],
    parameters: HestonParameters,
    time_to_expiry: float,
    *,
    rate: float = 0.0,
    formulation: str = "stable",
) -> NDArray[np.complex128]:
    r"""The Heston characteristic function of the log forward price.

    Inputs
        ``formulation`` -- ``"stable"`` uses the Albrecher form with
        :math:`|g_2| \le 1`; ``"original"`` uses Heston's 1993 form, which is
        algebraically identical and numerically unstable. The unstable one is
        kept deliberately so the failure can be demonstrated.
    Outputs
        :math:`\mathbb{E}[e^{iu\log S_T}]`, evaluated elementwise.

    Example
        >>> import numpy as np
        >>> p = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        >>> phi = characteristic_function(np.array([0.0 + 0j]), p, 1.0)
        >>> bool(np.isclose(phi[0].real, 1.0) and abs(phi[0].imag) < 1e-12)
        True
    """
    u = np.asarray(u, dtype=np.complex128)
    kappa, theta, xi, rho, v0 = (
        parameters.kappa,
        parameters.theta,
        parameters.xi,
        parameters.rho,
        parameters.v0,
    )
    tau = float(time_to_expiry)

    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))

    if formulation == "stable":
        # |g| <= 1 for all u, so log(1 - g e^{-d tau}) never approaches the cut.
        g = (beta - d) / (beta + d)
        exponent = np.exp(-d * tau)
        first = (kappa * theta / xi**2) * (
            (beta - d) * tau - 2.0 * np.log((1.0 - g * exponent) / (1.0 - g))
        )
        second = (v0 / xi**2) * (beta - d) * (1.0 - exponent) / (1.0 - g * exponent)
    elif formulation == "original":
        # Heston (1993). Algebraically the same; |g| can exceed 1, and then the
        # principal-branch logarithm jumps.
        g = (beta + d) / (beta - d)
        exponent = np.exp(d * tau)
        first = (kappa * theta / xi**2) * (
            (beta + d) * tau - 2.0 * np.log((1.0 - g * exponent) / (1.0 - g))
        )
        second = (v0 / xi**2) * (beta + d) * (1.0 - exponent) / (1.0 - g * exponent)
    else:
        raise ValueError(f"formulation must be 'stable' or 'original', got {formulation!r}")

    return np.exp(first + second + 1j * u * rate * tau)


def _gauss_legendre(
    n: int, lower: float, upper: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodes and weights on ``[lower, upper]``.

    Golub-Welsch: the nodes are eigenvalues of the symmetric tridiagonal Jacobi
    matrix of the Legendre recurrence, and the weights come from the first
    component of each eigenvector. Cheaper to write correctly than Newton
    iteration on the polynomials, and exact to machine precision.
    """
    k = np.arange(1, n, dtype=float)
    off_diagonal = k / np.sqrt(4.0 * k**2 - 1.0)
    jacobi = np.diag(off_diagonal, -1) + np.diag(off_diagonal, 1)
    nodes, vectors = np.linalg.eigh(jacobi)
    weights = 2.0 * vectors[0, :] ** 2

    half = 0.5 * (upper - lower)
    centre = 0.5 * (upper + lower)
    return centre + half * nodes, half * weights


def heston_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    parameters: HestonParameters,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
    n_nodes: int = 256,
    upper_limit: float = 200.0,
    formulation: str = "stable",
) -> float:
    r"""Price a European option under Heston by Fourier inversion.

    Method
        The Lewis form,

        .. math::
           C = S e^{-qT} - \frac{\sqrt{SK}e^{-rT}}{\pi}
               \int_0^\infty \Re\left[e^{iuk}\,
               \phi\!\left(u - \tfrac{i}{2}\right)\right]
               \frac{du}{u^2 + \tfrac14}

        one integral rather than Heston's two probabilities, so one
        characteristic-function evaluation per node instead of two. The
        integrand decays like :math:`u^{-2}` times an exponential, so a truncated
        Gauss-Legendre rule converges quickly.
    Failure modes
        Returns the intrinsic value at zero maturity. Puts come from put-call
        parity rather than a second integral, which also makes parity exact by
        construction rather than approximately satisfied.

    Example
        >>> p = HestonParameters(kappa=2.0, theta=0.04, xi=0.001, rho=0.0, v0=0.04)
        >>> # With almost no vol-of-vol this must reproduce Black-Scholes at 20%.
        >>> from quantos.derivatives.black_scholes import black_scholes_price
        >>> heston = heston_price(100.0, 100.0, 1.0, p, rate=0.03)
        >>> bs = float(black_scholes_price(100.0, 100.0, 1.0, 0.2, rate=0.03))
        >>> bool(abs(heston - bs) < 0.01)
        True
    """
    if time_to_expiry <= 0:
        intrinsic = (
            max(spot - strike, 0.0) if option_type is OptionType.CALL else max(strike - spot, 0.0)
        )
        return float(intrinsic)

    forward = spot * np.exp((rate - dividend_yield) * time_to_expiry)
    log_moneyness = np.log(forward / strike)

    nodes, weights = _gauss_legendre(n_nodes, 1e-10, upper_limit)
    shifted = np.asarray(nodes - 0.5j, dtype=np.complex128)
    phi = characteristic_function(
        shifted, parameters, time_to_expiry, rate=0.0, formulation=formulation
    )
    integrand = np.real(np.exp(1j * nodes * log_moneyness) * phi) / (nodes**2 + 0.25)
    integral = float(np.sum(weights * integrand))

    discount = np.exp(-rate * time_to_expiry)
    call = (
        spot * np.exp(-dividend_yield * time_to_expiry)
        - (np.sqrt(forward * strike) * discount / np.pi) * integral
    )
    call = float(max(call, 0.0))

    if option_type is OptionType.CALL:
        return call
    # Put-call parity, so parity holds exactly rather than to integration error.
    return float(call - spot * np.exp(-dividend_yield * time_to_expiry) + strike * discount)


def heston_implied_volatility(
    spot: float,
    strike: float,
    time_to_expiry: float,
    parameters: HestonParameters,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    **kwargs: object,
) -> float:
    """The Black-Scholes volatility that reproduces the Heston price.

    This is how a stochastic-volatility model is compared with a fitted surface:
    price under Heston, invert through Black-Scholes, and the result is directly
    comparable with an SVI smile.
    """
    from quantos.derivatives.black_scholes import implied_volatility

    price = heston_price(
        spot,
        strike,
        time_to_expiry,
        parameters,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL,
        **kwargs,  # type: ignore[arg-type]
    )
    try:
        return float(
            implied_volatility(
                price,
                spot,
                strike,
                time_to_expiry,
                rate=rate,
                dividend_yield=dividend_yield,
                option_type=OptionType.CALL,
            )
        )
    except (ValueError, RuntimeError):
        return float("nan")


@dataclass
class SmileComparison:
    """A Heston smile beside the SVI fit it is being checked against."""

    log_moneyness: NDArray[np.float64]
    heston_volatility: NDArray[np.float64]
    reference_volatility: NDArray[np.float64]
    time_to_expiry: float
    notes: list[str] = field(default_factory=list)

    @property
    def rmse_vol_points(self) -> float:
        residual = self.heston_volatility - self.reference_volatility
        finite = residual[np.isfinite(residual)]
        return float(np.sqrt(np.mean(finite**2)) * 100) if finite.size else float("nan")

    def summary(self) -> str:
        return (
            f"Heston vs reference at T={self.time_to_expiry:.3f}: "
            f"{self.rmse_vol_points:.2f} vol points RMSE over {self.log_moneyness.size} strikes"
        )
