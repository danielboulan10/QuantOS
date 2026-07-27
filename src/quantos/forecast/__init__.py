"""Forward-looking distributions: simulate paths, summarise them, check them.

Three modules, in the order they matter:

:mod:`quantos.forecast.paths`
    Simulate many forward paths rather than predicting one. Two engines that fail
    differently -- GARCH with fat tails, and a model-free block bootstrap.
:mod:`quantos.forecast.probabilities`
    Turn an ensemble into numbers that bear on a decision: what it might touch,
    how deep a drawdown might get, what a loss looks like when it happens, and how
    differently the long and short sides are exposed.
:mod:`quantos.forecast.calibration`
    Test whether those probabilities are true. When it says 10%, does it happen
    10% of the time? Everything above is worth nothing until this passes.
"""

from quantos.forecast.calibration import CalibrationResult, calibration_test
from quantos.forecast.paths import (
    PathEnsemble,
    compare_engines,
    simulate_bootstrap_paths,
    simulate_garch_paths,
)
from quantos.forecast.probabilities import (
    LongShortComparison,
    ProbabilityReport,
    drawdown_probability,
    first_passage_probability,
    long_short_comparison,
    probability_report,
)

__all__ = [
    "CalibrationResult",
    "LongShortComparison",
    "PathEnsemble",
    "ProbabilityReport",
    "calibration_test",
    "compare_engines",
    "drawdown_probability",
    "first_passage_probability",
    "long_short_comparison",
    "probability_report",
    "simulate_bootstrap_paths",
    "simulate_garch_paths",
]
