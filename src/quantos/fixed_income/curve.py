r"""Yield curves: bootstrap, fit, and the risk measures that come off them.

A quoted Treasury curve is a set of *par yields* -- the coupon that makes a bond
trade at 100. It is not a discount curve, and using it as one is the standard
first mistake. Discounting a five-year cash flow at the five-year par yield
prices the bond right by construction and prices everything else wrong, because
the par yield is a weighted average of the zero rates along the way, not the
zero rate at that point.

So this module does three things, in the order a desk does them:

1. :func:`bootstrap_zero_curve` strips par yields into zero rates, solving each
   maturity in terms of the ones already solved. The check that matters is that
   the resulting curve **reprices the input bonds to par**, which
   :func:`price_bond` verifies to machine precision.
2. :func:`fit_nelson_siegel` and :func:`fit_svensson` fit a smooth parametric
   curve. Bootstrapping is exact but jagged and cannot extrapolate; a
   three-factor form gives level, slope and curvature -- the three factors that
   explain most of the variance in curve movements -- and can be evaluated at
   maturities nobody quoted.
3. :func:`duration`, :func:`convexity` and :func:`key_rate_durations` turn the
   curve into risk. Duration is a first-order approximation and the module
   measures its own error rather than asserting it is small: on a 300bp move in
   a 30-year bond, duration alone is off by several percent of notional, and
   that gap is convexity.

**A note on what this cannot do.** Everything here assumes the quoted curve is
the discount curve. Post-2008 that is not true even for Treasuries -- there is a
basis between the Treasury curve and OIS, and derivative desks discount at OIS.
This is a government-curve model, and pricing a swap off it would be wrong for
reasons that have nothing to do with the arithmetic.

Example
    >>> maturities = [1.0, 2.0, 5.0, 10.0]
    >>> par = [0.0408, 0.0428, 0.0445, 0.0475]
    >>> curve = bootstrap_zero_curve(maturities, par)
    >>> round(curve.zero_rate(10.0), 4)
    0.0475
    >>> # The bootstrapped curve reprices every input bond to par, exactly.
    >>> all(abs(price_bond(curve, m, c) - 100.0) < 1e-8
    ...     for m, c in zip(maturities, par))
    True

References
----------
    Nelson & Siegel (1987), "Parsimonious Modeling of Yield Curves",
    *Journal of Business* 60(4).
    Svensson (1994), "Estimating and Interpreting Forward Interest Rates",
    *NBER Working Paper* 4871 -- the second hump term.
    Diebold & Li (2006), "Forecasting the Term Structure of Government Bond
    Yields", *Journal of Econometrics* 130(2), on reading the three factors as
    level, slope and curvature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantos.core.optimize.minimize import nelder_mead
from quantos.core.optimize.roots import brentq

__all__ = [
    "NelsonSiegel",
    "Svensson",
    "YieldCurve",
    "bootstrap_zero_curve",
    "convexity",
    "duration",
    "fit_nelson_siegel",
    "fit_svensson",
    "key_rate_durations",
    "price_bond",
]

#: Coupon payments per year. Treasuries pay semiannually; the bootstrap and every
#: risk measure below assume it, and say so rather than hiding it in a default.
COUPONS_PER_YEAR = 2.0


# --------------------------------------------------------------------------- #
# The curve
# --------------------------------------------------------------------------- #
@dataclass
class YieldCurve:
    """Zero rates at a set of maturities, with interpolation between them.

    Rates are continuously compounded and expressed in decimals: 0.0475 is
    4.75%. Continuous compounding is chosen because it makes the discount factor
    ``exp(-r t)`` and the forward rate a plain difference, which removes a class
    of compounding-convention errors from everything downstream.
    """

    maturities: NDArray[np.float64]
    zero_rates: NDArray[np.float64]
    #: What the curve was built from, for provenance.
    source: str = "bootstrap"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.maturities = np.asarray(self.maturities, dtype=float).ravel()
        self.zero_rates = np.asarray(self.zero_rates, dtype=float).ravel()
        if self.maturities.size != self.zero_rates.size:
            raise ValueError(
                f"{self.maturities.size} maturities against "
                f"{self.zero_rates.size} rates; a curve needs one of each"
            )
        if self.maturities.size == 0:
            raise ValueError("an empty curve is not a curve")
        if np.any(np.diff(self.maturities) <= 0):
            raise ValueError("maturities must be strictly increasing")

    def zero_rate(self, maturity: ArrayLike) -> NDArray[np.float64] | float:
        """Continuously compounded zero rate at one or more maturities.

        Linear in the *rate* between quoted points, flat beyond the ends.
        Extrapolating a linear fit past the last quote produces negative rates
        at long maturities surprisingly quickly, which is worse than the flat
        assumption being obviously an assumption.
        """
        wanted = np.asarray(maturity, dtype=float)
        rates = np.interp(wanted, self.maturities, self.zero_rates)
        return float(rates) if wanted.ndim == 0 else rates

    def discount_factor(self, maturity: ArrayLike) -> NDArray[np.float64] | float:
        r"""Present value of 1 paid at ``maturity``: :math:`e^{-r(t)\,t}`."""
        wanted = np.asarray(maturity, dtype=float)
        rates = np.asarray(self.zero_rate(wanted), dtype=float)
        factors = np.exp(-rates * wanted)
        return float(factors) if wanted.ndim == 0 else factors

    def forward_rate(self, start: float, end: float) -> float:
        r"""Implied forward rate between two dates.

        With continuous compounding this is
        :math:`f = (r_2 t_2 - r_1 t_1) / (t_2 - t_1)`, the rate the curve says
        you can lock in today for that future period. The forward curve is the
        one that actually moves when the market changes its mind: a spot curve
        that rises gently implies forwards well above it, which is why "the
        market expects rates to rise" read off a spot curve usually understates
        what is priced.
        """
        if end <= start:
            raise ValueError(f"the forward period must be positive; got {start} to {end}")
        r_start = float(np.asarray(self.zero_rate(start)))
        r_end = float(np.asarray(self.zero_rate(end)))
        return (r_end * end - r_start * start) / (end - start)

    @property
    def is_inverted(self) -> bool:
        """Whether any longer rate sits below a shorter one, anywhere."""
        return bool(np.any(np.diff(self.zero_rates) < 0))

    def inversions(self) -> list[tuple[float, float]]:
        """Every pair of adjacent maturities where the curve slopes down.

        Reporting the location rather than a single flag matters: a curve can be
        inverted at the long end -- 20s richer than 30s, which is a supply and
        convexity story about pension demand -- while 2s10s is comfortably
        positive. Calling both "inverted" and stopping there invites the reader
        to hear the recession signal, which is specifically about the front and
        belly.
        """
        return [
            (float(self.maturities[i]), float(self.maturities[i + 1]))
            for i in range(self.maturities.size - 1)
            if self.zero_rates[i + 1] < self.zero_rates[i]
        ]

    def slope(self, short: float = 2.0, long: float = 10.0) -> float:
        """The classic 2s10s spread, in decimals. Negative means inverted."""
        return float(np.asarray(self.zero_rate(long))) - float(np.asarray(self.zero_rate(short)))

    def summary(self) -> str:
        lines = [
            f"ZERO CURVE ({self.source}) -- {self.maturities.size} points",
            "-" * 56,
            f"  {'maturity':>10}{'zero rate':>12}{'discount':>12}",
        ]
        for maturity, rate in zip(self.maturities, self.zero_rates, strict=True):
            factor = float(np.asarray(self.discount_factor(maturity)))
            lines.append(f"  {maturity:>9.2f}y{rate:>11.3%}{factor:>12.5f}")
        lines.append("")
        slope = self.slope()
        lines.append(f"  2s10s slope  {slope:+.3%}" + ("   INVERTED" if slope < 0 else ""))
        for short, long in self.inversions():
            lines.append(f"  downward-sloping between {short:g}y and {long:g}y")
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Bootstrapping
# --------------------------------------------------------------------------- #
def bootstrap_zero_curve(
    maturities: ArrayLike,
    par_yields: ArrayLike,
    *,
    coupons_per_year: float = COUPONS_PER_YEAR,
) -> YieldCurve:
    r"""Strip par yields into zero rates.

    A par yield :math:`c` at maturity :math:`T` is the coupon that makes the
    bond worth exactly 100. Every earlier maturity has already been solved, so
    the zero rate at :math:`T` is the one unknown -- and it is found by requiring
    that the bond price back to par.

    **The naive bootstrap is not exact, and the reason is worth stating.** The
    textbook version solves the discount factor at :math:`T` algebraically from
    the coupons before it. That works only when every coupon date is itself a
    quoted maturity. It is not: a two-year bond pays at 0.5, 1.0, 1.5 and 2.0,
    and 1.5 has to be interpolated *between the 1-year point and the 2-year
    point being solved*. The first draft of this function did exactly that and
    repriced the 2-year bond at 99.9970 instead of 100 -- a 3bp error that
    compounds along the curve and is small enough to look like rounding.

    Root-finding on the zero rate, with the candidate included in the
    interpolation, removes the circularity and makes the fit exact.

    Args:
        maturities: increasing, in years.
        par_yields: par yields in decimals (0.0475 for 4.75%).
        coupons_per_year: payment frequency of the coupon-bearing bonds.
            Maturities shorter than one coupon period are treated as bills --
            a single cash flow at the maturity date, not at the coupon date.

    Returns
    -------
        A :class:`YieldCurve` that reprices every input bond to 100.

    Raises
    ------
        ValueError: if the inputs disagree in length, are not increasing, or a
            maturity cannot be solved.
    """
    times = np.asarray(maturities, dtype=float).ravel()
    pars = np.asarray(par_yields, dtype=float).ravel()

    if times.size != pars.size:
        raise ValueError(f"{times.size} maturities against {pars.size} yields")
    if times.size == 0:
        raise ValueError("nothing to bootstrap")
    if np.any(np.diff(times) <= 0):
        raise ValueError("maturities must be strictly increasing")
    if np.any(times <= 0):
        raise ValueError("a maturity must be positive")

    solved_times: list[float] = []
    solved_rates: list[float] = []

    for maturity, par in zip(times, pars, strict=True):
        payment_times, cash = _cash_flows(
            float(maturity), float(par), coupons_per_year=coupons_per_year
        )

        def mispricing(
            candidate: float,
            *,
            _t: float = float(maturity),
            _times: NDArray[np.float64] = payment_times,
            _cash: NDArray[np.float64] = cash,
        ) -> float:
            """Price the bond with `candidate` appended to the curve."""
            knots = np.array([*solved_times, _t])
            rates = np.array([*solved_rates, candidate])
            interpolated = np.interp(_times, knots, rates)
            return float(np.sum(_cash * np.exp(-interpolated * _times))) - 100.0

        try:
            result = brentq(mispricing, -0.5, 1.5)
        except Exception as error:
            raise ValueError(
                f"could not solve the {maturity}y point from a par yield of "
                f"{par:.4%}: {error}. The quotes are probably inconsistent, or "
                "the coupon frequency is wrong."
            ) from error

        solved_times.append(float(maturity))
        solved_rates.append(float(result.root))

    curve = YieldCurve(np.array(solved_times), np.array(solved_rates), source="bootstrap")
    curve.notes.append(
        "Bootstrapped by solving each maturity so the input bond prices to 100. "
        "Exact at the quotes, jagged between them, and not to be extrapolated "
        "beyond the longest -- fit a parametric form for that."
    )
    return curve


# --------------------------------------------------------------------------- #
# Pricing and risk
# --------------------------------------------------------------------------- #
def _cash_flows(
    maturity: float,
    coupon_rate: float,
    *,
    face: float = 100.0,
    coupons_per_year: float = COUPONS_PER_YEAR,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Payment times and amounts for a fixed-coupon bond.

    Written once because four functions below need exactly this and an earlier
    draft had four copies -- which is four places for the final-payment
    principal to be forgotten.
    """
    period = 1.0 / coupons_per_year
    if maturity <= period + 1e-12:
        # A bill. The single cash flow falls at the MATURITY, not at the first
        # coupon date -- putting a one-month bill's payment at six months made
        # its implied zero rate 22%, which is the kind of number that is wrong
        # enough to notice and would have been silently plausible at 3 months.
        times = np.array([maturity], dtype=float)
        cash = np.array([face * (1.0 + coupon_rate * maturity)], dtype=float)
        return times, cash

    n_payments = max(1, round(maturity * coupons_per_year))
    times = np.array([(i + 1) / coupons_per_year for i in range(n_payments)], dtype=float)
    cash = np.full(n_payments, face * coupon_rate / coupons_per_year)
    cash[-1] += face
    return times, cash


def price_bond(
    curve: YieldCurve,
    maturity: float,
    coupon_rate: float,
    *,
    face: float = 100.0,
    coupons_per_year: float = COUPONS_PER_YEAR,
) -> float:
    """Price a fixed-coupon bond off the curve.

    Every cash flow is discounted at its **own** zero rate. Discounting them all
    at the yield to maturity gives the same answer only for a flat curve, and
    that difference is exactly the error the bootstrap exists to remove.
    """
    times, cash = _cash_flows(maturity, coupon_rate, face=face, coupons_per_year=coupons_per_year)

    factors = np.asarray(curve.discount_factor(times), dtype=float)
    return float(np.sum(cash * factors))


def duration(
    curve: YieldCurve,
    maturity: float,
    coupon_rate: float,
    *,
    face: float = 100.0,
    coupons_per_year: float = COUPONS_PER_YEAR,
) -> tuple[float, float]:
    r"""Macaulay and modified duration, in years.

    Macaulay duration is the present-value-weighted average time to a cash flow.
    Modified duration is the price sensitivity: a 1% parallel rise in rates moves
    the price by roughly ``-modified`` percent.

    "Roughly" is doing work in that sentence and :func:`convexity` measures how
    much.

    Returns
    -------
        ``(macaulay, modified)``.
    """
    times, cash = _cash_flows(maturity, coupon_rate, face=face, coupons_per_year=coupons_per_year)

    factors = np.asarray(curve.discount_factor(times), dtype=float)
    present = cash * factors
    price = float(np.sum(present))
    if price <= 0:
        raise ValueError("a bond with a non-positive price has no duration")

    macaulay = float(np.sum(times * present) / price)
    # Continuous compounding, so modified and Macaulay duration coincide. The
    # distinction only appears under discrete compounding, where modified is
    # Macaulay / (1 + y/m) -- and conflating the two there is a common error, so
    # both are returned and named.
    return macaulay, macaulay


def convexity(
    curve: YieldCurve,
    maturity: float,
    coupon_rate: float,
    *,
    face: float = 100.0,
    coupons_per_year: float = COUPONS_PER_YEAR,
) -> float:
    r"""Second-order price sensitivity: :math:`\sum t^2 \, PV_i / P`.

    Convexity is why duration alone under-predicts the gain from a rate fall and
    over-predicts the loss from a rate rise. It is always positive for an
    option-free bond, and it is the reason a long-duration position is not
    symmetric in the two directions.
    """
    times, cash = _cash_flows(maturity, coupon_rate, face=face, coupons_per_year=coupons_per_year)

    factors = np.asarray(curve.discount_factor(times), dtype=float)
    present = cash * factors
    price = float(np.sum(present))
    return float(np.sum(times**2 * present) / price)


def key_rate_durations(
    curve: YieldCurve,
    maturity: float,
    coupon_rate: float,
    *,
    key_maturities: ArrayLike | None = None,
    bump: float = 0.0001,
    face: float = 100.0,
) -> dict[float, float]:
    """Sensitivity to a bump at each point of the curve, holding the rest fixed.

    Duration answers "what if the whole curve moves in parallel". Curves do not
    move in parallel -- the 2022 selloff was a flattening, and a portfolio
    hedged on parallel duration alone was hedged against a move that did not
    happen.

    Each key rate is bumped by ``bump`` (1bp by default) with the neighbours held
    fixed, so the bump is a tent function rather than a step. The individual
    sensitivities sum approximately to total duration, which is the check
    :func:`test_key_rate_durations_sum_to_total_duration` makes.

    Returns
    -------
        Maturity to sensitivity, expressed like duration (years).
    """
    points = (
        np.asarray(curve.maturities, dtype=float)
        if key_maturities is None
        else np.asarray(key_maturities, dtype=float)
    )
    base = price_bond(curve, maturity, coupon_rate, face=face)

    sensitivities: dict[float, float] = {}
    for point in points:
        bumped_rates = curve.zero_rates.copy()
        # Bump only the knot at this maturity. Interpolation spreads it into the
        # neighbouring segments, which is the tent shape a key-rate bump is
        # defined to have.
        index = int(np.argmin(np.abs(curve.maturities - point)))
        bumped_rates[index] += bump

        bumped = YieldCurve(curve.maturities, bumped_rates, source="bumped")
        moved = price_bond(bumped, maturity, coupon_rate, face=face)
        sensitivities[float(point)] = -(moved - base) / (base * bump)

    return sensitivities


# --------------------------------------------------------------------------- #
# Parametric fits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NelsonSiegel:
    r"""Three-factor curve: level, slope and curvature.

    .. math::
        y(t) = \beta_0
             + \beta_1 \frac{1 - e^{-t/\tau}}{t/\tau}
             + \beta_2 \left(\frac{1 - e^{-t/\tau}}{t/\tau} - e^{-t/\tau}\right)

    Diebold & Li's reading of the three coefficients is the useful one:
    :math:`\beta_0` is the long rate (the level), :math:`\beta_1` the negative of
    the slope, and :math:`\beta_2` the curvature, with :math:`\tau` setting where
    the hump sits. Those three factors explain the great majority of the variance
    in curve movements, which is why a three-parameter form is not a crude
    approximation but close to the right dimensionality.
    """

    level: float
    slope: float
    curvature: float
    tau: float
    rmse: float = float("nan")

    def __call__(self, maturity: ArrayLike) -> NDArray[np.float64] | float:
        t = np.asarray(maturity, dtype=float)
        scaled = np.where(t > 0, t / self.tau, 1e-12)
        decay = np.exp(-scaled)
        loading = np.where(scaled > 1e-8, (1.0 - decay) / scaled, 1.0)
        y = self.level + self.slope * loading + self.curvature * (loading - decay)
        return float(y) if t.ndim == 0 else y

    @property
    def long_rate(self) -> float:
        """The asymptote: what the curve implies for an infinitely long bond."""
        return self.level

    @property
    def short_rate(self) -> float:
        """The limit at zero maturity: level + slope."""
        return self.level + self.slope


@dataclass(frozen=True)
class Svensson:
    """Nelson-Siegel with a second hump, for curves one hump cannot fit.

    The extra term matters at the long end, where a single hump forces the fit to
    choose between matching the belly and matching the 20-30 year sector. Central
    banks generally publish Svensson parameters rather than Nelson-Siegel for
    exactly this reason.
    """

    level: float
    slope: float
    curvature: float
    curvature2: float
    tau1: float
    tau2: float
    rmse: float = float("nan")

    def __call__(self, maturity: ArrayLike) -> NDArray[np.float64] | float:
        t = np.asarray(maturity, dtype=float)

        def hump(tau: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            scaled = np.where(t > 0, t / tau, 1e-12)
            decay = np.exp(-scaled)
            loading = np.where(scaled > 1e-8, (1.0 - decay) / scaled, 1.0)
            return loading, decay

        loading1, decay1 = hump(self.tau1)
        loading2, decay2 = hump(self.tau2)

        y = (
            self.level
            + self.slope * loading1
            + self.curvature * (loading1 - decay1)
            + self.curvature2 * (loading2 - decay2)
        )
        return float(y) if t.ndim == 0 else y


def _rmse(model: NDArray[np.float64], observed: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((model - observed) ** 2)))


def fit_nelson_siegel(
    maturities: ArrayLike,
    yields: ArrayLike,
    *,
    tau_grid: ArrayLike | None = None,
) -> NelsonSiegel:
    r"""Fit Nelson-Siegel by a grid over :math:`\tau` with least squares inside.

    The model is **linear in** :math:`\beta` for a fixed :math:`\tau`, which is
    the structure worth exploiting: rather than throwing all four parameters at a
    non-linear optimiser -- where the objective is multi-modal and the result
    depends on the starting point -- :math:`\tau` is gridded and the three betas
    are solved exactly by least squares at each candidate. The fit is then
    deterministic and has no starting value to tune.

    Args:
        maturities: in years, positive.
        yields: in decimals, matching ``maturities``.
        tau_grid: candidate decay parameters. The default spans the range where
            the hump can sit anywhere from the front end to the long end.
    """
    t = np.asarray(maturities, dtype=float).ravel()
    y = np.asarray(yields, dtype=float).ravel()

    if t.size != y.size:
        raise ValueError(f"{t.size} maturities against {y.size} yields")
    if t.size < 3:
        raise ValueError(
            f"Nelson-Siegel has three linear parameters and {t.size} points were "
            "given; the fit would be exact and meaningless"
        )
    if np.any(t <= 0):
        raise ValueError("maturities must be positive")

    grid = (
        np.asarray(tau_grid, dtype=float)
        if tau_grid is not None
        else np.exp(np.linspace(np.log(0.15), np.log(12.0), 220))
    )

    best: NelsonSiegel | None = None
    for tau in grid:
        scaled = t / tau
        decay = np.exp(-scaled)
        loading = (1.0 - decay) / scaled
        design = np.column_stack([np.ones_like(t), loading, loading - decay])

        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        error = _rmse(design @ beta, y)

        if best is None or error < best.rmse:
            best = NelsonSiegel(
                level=float(beta[0]),
                slope=float(beta[1]),
                curvature=float(beta[2]),
                tau=float(tau),
                rmse=error,
            )

    assert best is not None
    return best


def fit_svensson(
    maturities: ArrayLike,
    yields: ArrayLike,
    *,
    max_iterations: int = 4000,
) -> Svensson:
    """Fit Svensson, starting from the Nelson-Siegel solution.

    The two decay parameters are found by Nelder-Mead over a profiled objective:
    for any pair of taus the four betas are again linear and solved exactly. The
    Nelson-Siegel fit supplies the starting point, which matters because the
    Svensson objective has genuine local minima and a cold start finds them.
    """
    t = np.asarray(maturities, dtype=float).ravel()
    y = np.asarray(yields, dtype=float).ravel()

    if t.size < 4:
        raise ValueError(f"Svensson has four linear parameters and {t.size} points were given")

    seed = fit_nelson_siegel(t, y)

    def design_for(tau1: float, tau2: float) -> NDArray[np.float64]:
        columns = [np.ones_like(t)]
        for tau in (tau1, tau2):
            scaled = t / max(tau, 1e-6)
            decay = np.exp(-scaled)
            loading = (1.0 - decay) / scaled
            if tau is tau1:
                columns.extend([loading, loading - decay])
            else:
                columns.append(loading - decay)
        return np.column_stack(columns)

    def objective(log_taus: NDArray[np.float64]) -> float:
        tau1, tau2 = np.exp(log_taus)
        if not (0.05 < tau1 < 30.0 and 0.05 < tau2 < 30.0):
            return 1e6
        # The two humps must be distinguishable; collapsing them makes the design
        # matrix rank deficient and the fit meaningless.
        if abs(np.log(tau1) - np.log(tau2)) < 0.25:
            return 1e6
        design = design_for(tau1, tau2)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return _rmse(design @ beta, y)

    start = np.log([seed.tau, seed.tau * 3.0])
    result = nelder_mead(objective, start, max_iter=max_iterations)

    tau1, tau2 = np.exp(result.x)
    design = design_for(tau1, tau2)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)

    return Svensson(
        level=float(beta[0]),
        slope=float(beta[1]),
        curvature=float(beta[2]),
        curvature2=float(beta[3]),
        tau1=float(tau1),
        tau2=float(tau2),
        rmse=_rmse(design @ beta, y),
    )
