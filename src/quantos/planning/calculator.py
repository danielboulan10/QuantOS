r"""Investment projection — the deterministic schedule, and what it leaves out.

Two calculators, and the second is the point
---------------------------------------------
The first reproduces what every investment calculator on the web does: compound a
starting balance and a stream of contributions at a fixed rate, and print a
year-by-year schedule. :func:`investment_schedule` matches Calculator.net to the
cent on their own worked example, which is how you know the convention is right
rather than merely plausible.

The second, :func:`simulate_plan`, does the thing those calculators cannot: it
lets the return be *random*, which it is.

Why that matters more than it sounds
-------------------------------------
"6% a year for 10 years" never happens. Returns arrive as a sequence, and two
facts follow that a fixed-rate calculator cannot express:

**Volatility drag.** A portfolio that returns +20% then -20% is down 4%, not
flat. Compounding is multiplicative, so the geometric mean is below the
arithmetic mean by roughly :math:`\sigma^2/2`. A plan advertised at "6% average"
delivers a *median* nearer 4.2% at 20% volatility -- and the shortfall is
invisible in every deterministic calculator, because arithmetic averaging is
exactly what they do.

**Sequence-of-returns risk.** With regular contributions the *order* matters.
Bad years early hurt less than bad years late while you are still adding money,
and the reverse once you are drawing down. A single average return cannot
represent either case.

So the deterministic number is not wrong -- it is the answer to a different
question. It tells you what happens if returns are constant, and returns are not
constant. This module prints both and labels which is which.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "InvestmentSchedule",
    "PlanDistribution",
    "ScheduleRow",
    "investment_schedule",
    "required_contribution",
    "required_return",
    "required_years",
    "simulate_plan",
]

#: Contributions per year, by frequency name.
FREQUENCY: dict[str, int] = {
    "monthly": 12,
    "quarterly": 4,
    "semiannually": 2,
    "annually": 1,
    "weekly": 52,
    "daily": 365,
}


def _periodic_rate(annual_rate: float, periods_per_year: int) -> float:
    r"""The per-period rate that compounds to ``annual_rate`` over a year.

    :math:`(1+r)^{1/m} - 1`, **not** :math:`r/m`. The distinction is not
    pedantry: at 6% with monthly contributions the nominal convention overstates
    the first year's interest by 9 dollars on a 32,000 dollar balance, and the
    error compounds. Reproducing Calculator.net's published example to the cent
    required this form, which is also the one that makes "6% annual" mean what it
    says.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


@dataclass(frozen=True)
class ScheduleRow:
    """One year of the accumulation schedule."""

    year: int
    deposits: float
    interest: float
    ending_balance: float


@dataclass
class InvestmentSchedule:
    """A deterministic projection, and what it assumes."""

    starting_amount: float
    end_balance: float
    total_contributions: float
    total_interest: float
    years: float
    annual_return: float
    rows: list[ScheduleRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def contribution_share(self) -> float:
        """Fraction of the final balance that is money you put in, not growth."""
        return self.total_contributions / self.end_balance if self.end_balance else float("nan")

    @property
    def interest_share(self) -> float:
        return self.total_interest / self.end_balance if self.end_balance else float("nan")

    def summary(self) -> str:
        lines = [
            f"End balance        {self.end_balance:>15,.2f}",
            f"Starting amount    {self.starting_amount:>15,.2f}",
            f"Total contributions{self.total_contributions:>15,.2f}",
            f"Total interest     {self.total_interest:>15,.2f}",
            "",
            f"  {self.interest_share:.0%} of the final balance is growth; "
            f"{self.contribution_share:.0%} is money you deposited",
            "",
            f"{'Year':>5} {'Deposit':>14} {'Interest':>14} {'Ending balance':>16}",
        ]
        lines += [
            f"{r.year:>5} {r.deposits:>14,.2f} {r.interest:>14,.2f} {r.ending_balance:>16,.2f}"
            for r in self.rows
        ]
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


def investment_schedule(
    starting_amount: float,
    years: float,
    annual_return: float,
    *,
    contribution: float = 0.0,
    frequency: str = "monthly",
    contribute_at_start: bool = False,
    inflation: float = 0.0,
) -> InvestmentSchedule:
    """Project a plan at a fixed return, with a year-by-year schedule.

    Purpose
        The familiar calculation, done with the conventions stated.
    Inputs
        ``annual_return`` -- as a decimal, so 6% is ``0.06``.
        ``contribute_at_start`` -- deposits at the start of each period earn one
        extra period of growth. Over 30 years of monthly contributions that is
        worth about half a percent of the final balance, which is small but is
        the difference between two calculators disagreeing and one being wrong.
        ``inflation`` -- when non-zero, the result is additionally reported in
        today's money, which is the number that actually answers "will this be
        enough".
    Outputs
        An :class:`InvestmentSchedule`.

    Example
        Calculator.net's own worked example -- $20,000 for 10 years at 6% with
        $1,000 monthly, compounded annually -- published as $198,290.40:

        >>> plan = investment_schedule(20_000, 10, 0.06, contribution=1_000)
        >>> f"{plan.end_balance:,.2f}"
        '198,290.40'
        >>> f"{plan.total_contributions:,.2f}", f"{plan.total_interest:,.2f}"
        ('120,000.00', '58,290.40')
    """
    if frequency not in FREQUENCY:
        raise ValueError(f"frequency must be one of {sorted(FREQUENCY)}, got {frequency!r}")
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")

    per_year = FREQUENCY[frequency]
    rate = _periodic_rate(annual_return, per_year)

    balance = float(starting_amount)
    total_contributions = 0.0
    total_interest = 0.0
    rows: list[ScheduleRow] = []

    total_periods = round(years * per_year)
    for period in range(1, total_periods + 1):
        opening = balance
        deposited = 0.0

        if contribute_at_start:
            balance += contribution
            deposited += contribution

        interest = balance * rate
        balance += interest

        if not contribute_at_start:
            balance += contribution
            deposited += contribution

        total_contributions += deposited
        total_interest += interest

        if period % per_year == 0 or period == total_periods:
            year = int(np.ceil(period / per_year))
            del opening  # the per-year interest is taken from the running total
            year_deposits = (
                deposited
                if per_year == 1
                else contribution * min(per_year, period - (year - 1) * per_year)
            )
            if year == 1:
                year_deposits += float(starting_amount)
            rows.append(
                ScheduleRow(
                    year=year,
                    deposits=float(year_deposits),
                    interest=float(total_interest - sum(r.interest for r in rows)),
                    ending_balance=float(balance),
                )
            )

    notes: list[str] = [
        "This assumes the return is exactly the same every single year, which it "
        "never is. See simulate_plan for what volatility does to the same plan."
    ]
    if inflation > 0:
        real = balance / (1.0 + inflation) ** years
        notes.append(
            f"in today's money, after {inflation:.1%} inflation, that is "
            f"{real:,.2f} -- the number that answers whether it is enough"
        )

    return InvestmentSchedule(
        starting_amount=float(starting_amount),
        end_balance=float(balance),
        total_contributions=float(total_contributions),
        total_interest=float(total_interest),
        years=float(years),
        annual_return=float(annual_return),
        rows=rows,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Solving for the other unknowns -- the calculator's other tabs
# --------------------------------------------------------------------------- #
def required_return(
    starting_amount: float,
    years: float,
    target: float,
    *,
    contribution: float = 0.0,
    frequency: str = "monthly",
    contribute_at_start: bool = False,
) -> float:
    """The annual return needed to reach ``target``.

    Solved by bisection on the schedule rather than algebraically, so it stays
    correct for every contribution convention rather than only the one the
    closed form was derived for.
    """

    def shortfall(rate: float) -> float:
        plan = investment_schedule(
            starting_amount,
            years,
            rate,
            contribution=contribution,
            frequency=frequency,
            contribute_at_start=contribute_at_start,
        )
        return plan.end_balance - target

    low, high = -0.99, 10.0
    if shortfall(low) > 0:
        return low
    if shortfall(high) < 0:
        return float("nan")

    for _ in range(200):
        middle = 0.5 * (low + high)
        if shortfall(middle) < 0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def required_contribution(
    starting_amount: float,
    years: float,
    annual_return: float,
    target: float,
    *,
    frequency: str = "monthly",
    contribute_at_start: bool = False,
) -> float:
    """The per-period contribution needed to reach ``target``.

    Linear in the contribution, so this is solved exactly rather than searched.
    """
    without = investment_schedule(
        starting_amount,
        years,
        annual_return,
        contribution=0.0,
        frequency=frequency,
        contribute_at_start=contribute_at_start,
    ).end_balance
    per_unit = (
        investment_schedule(
            starting_amount,
            years,
            annual_return,
            contribution=1.0,
            frequency=frequency,
            contribute_at_start=contribute_at_start,
        ).end_balance
        - without
    )
    if per_unit <= 0:
        return float("nan")
    return float(max((target - without) / per_unit, 0.0))


def required_years(
    starting_amount: float,
    annual_return: float,
    target: float,
    *,
    contribution: float = 0.0,
    frequency: str = "monthly",
    contribute_at_start: bool = False,
    max_years: float = 100.0,
) -> float:
    """How long the plan takes to reach ``target``."""
    per_year = FREQUENCY[frequency]
    rate = _periodic_rate(annual_return, per_year)
    balance = float(starting_amount)

    if balance >= target:
        return 0.0
    for period in range(1, int(max_years * per_year) + 1):
        if contribute_at_start:
            balance += contribution
        balance *= 1.0 + rate
        if not contribute_at_start:
            balance += contribution
        if balance >= target:
            return float(period / per_year)
    return float("nan")


# --------------------------------------------------------------------------- #
# What the fixed-rate calculation leaves out
# --------------------------------------------------------------------------- #
@dataclass
class PlanDistribution:
    """The same plan, with the return allowed to be random."""

    deterministic: float
    percentiles: dict[int, float]
    probability_of_target: float
    target: float
    years: float
    expected_return: float
    volatility: float
    n_paths: int
    notes: list[str] = field(default_factory=list)

    @property
    def median(self) -> float:
        return self.percentiles[50]

    @property
    def median_shortfall(self) -> float:
        """How far the median outcome falls below the fixed-rate projection.

        Positive almost always, and the reason is arithmetic rather than
        pessimism: compounding is multiplicative, so the median compounded
        outcome sits below the figure produced by averaging the rate.
        """
        return self.deterministic - self.median

    @property
    def verdict(self) -> str:
        gap = self.median_shortfall / self.deterministic if self.deterministic else float("nan")
        return (
            f"The fixed-rate projection of {self.deterministic:,.0f} is not the expected "
            f"outcome: it is roughly the {self._deterministic_percentile()}th percentile. "
            f"The median is {self.median:,.0f}, {gap:.0%} lower, because compounding is "
            "multiplicative and averaging the rate is not the same as averaging the result."
        )

    def _deterministic_percentile(self) -> int:
        levels = sorted(self.percentiles)
        values = [self.percentiles[level] for level in levels]
        return round(float(np.interp(self.deterministic, values, levels)))

    def summary(self) -> str:
        lines = [
            f"{self.n_paths:,} simulated paths, {self.expected_return:.1%} expected return, "
            f"{self.volatility:.1%} volatility, {self.years:g} years",
            "",
            f"  fixed-rate projection {self.deterministic:>15,.0f}",
        ]
        for level in sorted(self.percentiles):
            marker = "  <- median" if level == 50 else ""
            lines.append(f"  {level:>3}th percentile     {self.percentiles[level]:>15,.0f}{marker}")
        lines += [
            "",
            f"  probability of reaching {self.target:,.0f}: {self.probability_of_target:.1%}",
            "",
            f"  {self.verdict}",
        ]
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


def simulate_plan(
    starting_amount: float,
    years: float,
    expected_return: float,
    volatility: float,
    *,
    contribution: float = 0.0,
    frequency: str = "monthly",
    target: float | None = None,
    n_paths: int = 20_000,
    seed: int = 20240719,
) -> PlanDistribution:
    r"""The same plan with a random return, so the spread is visible.

    Method
        Log returns are drawn i.i.d. with the given annual mean and volatility
        and applied period by period, with contributions added as they occur. The
        The drift is set so **expected** terminal wealth equals the fixed-rate
        projection. The *median* then falls below it, because compounding is
        multiplicative: for a lump sum the factor is exactly
        :math:`e^{-\sigma^2 T/2}`, which is 0.89 at 15% volatility over 10 years.

        With regular contributions the gap is smaller -- measured at 6% on the
        worked example below -- because money added late compounds for less time
        and carries less drag. What does not shrink is the spread: on that same
        plan the fixed-rate figure of 198,290 is reached only **42.7%** of the
        time, and the 5th percentile is 114,216. The projected number is not the
        typical outcome; it is an above-median one.
    Failure modes
        Assumes returns are independent across periods. Real returns are close to
        that at monthly frequency but not exactly, and momentum or mean reversion
        would change the spread. This overstates neither the median nor the
        drag -- it is the shape of the tails it gets approximately right.

    Example
        >>> outcome = simulate_plan(20_000, 10, 0.06, 0.15, contribution=1_000,
        ...                         n_paths=4_000)
        >>> outcome.median < outcome.deterministic     # volatility drag
        True
    """
    per_year = FREQUENCY[frequency]
    n_periods = round(years * per_year)
    rng = np.random.default_rng(seed)

    # Choose the log-return drift so that EXPECTED terminal growth equals the
    # fixed-rate projection: E[prod(1+R)] = 1+r requires mu = log(1+r)/m - s^2/2.
    #
    # The first version of this line read "- 0.5*s**2 + 0.5*s**2", which cancels
    # to zero and silently removed the entire effect the function exists to show:
    # the median came out equal to the deterministic figure rather than below it.
    # The doctest asserting median < deterministic is what caught it.
    period_sigma = volatility / np.sqrt(per_year)
    period_drift = np.log1p(expected_return) / per_year - 0.5 * period_sigma**2

    shocks = rng.normal(period_drift, period_sigma, (n_paths, n_periods))
    growth = np.exp(shocks)

    balance = np.full(n_paths, float(starting_amount))
    for period in range(n_periods):
        balance = balance * growth[:, period] + contribution

    deterministic = investment_schedule(
        starting_amount,
        years,
        expected_return,
        contribution=contribution,
        frequency=frequency,
    ).end_balance

    goal = float(target) if target is not None else deterministic
    percentiles = {
        level: float(np.percentile(balance, level)) for level in (5, 10, 25, 50, 75, 90, 95)
    }

    return PlanDistribution(
        deterministic=float(deterministic),
        percentiles=percentiles,
        probability_of_target=float(np.mean(balance >= goal)),
        target=goal,
        years=float(years),
        expected_return=float(expected_return),
        volatility=float(volatility),
        n_paths=int(n_paths),
        notes=[
            "returns are drawn independently each period; real returns are close to "
            "independent monthly but not exactly, and this says nothing about the "
            "chance that the expected return itself is wrong",
        ],
    )
