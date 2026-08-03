"""Investment projection: the familiar calculation, and what it leaves out.

A fixed-rate calculator answers "what if the return is the same every year".
Returns are not the same every year, and the difference is not a rounding error
-- see :func:`quantos.planning.calculator.simulate_plan`.
"""

from quantos.planning.calculator import (
    InvestmentSchedule,
    PlanDistribution,
    ScheduleRow,
    investment_schedule,
    required_contribution,
    required_return,
    required_years,
    simulate_plan,
)

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
