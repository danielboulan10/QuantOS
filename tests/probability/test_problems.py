"""Every probability problem's closed form must survive its own simulation."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantos.core.rng import SeedBank
from quantos.probability.problems import (
    ALL_PROBLEMS,
    BallotProblem,
    BirthdayProblem,
    CouponCollector,
    GamblersRuin,
    MontyHall,
    OptimalCardStopping,
    Problem,
    SecretaryProblem,
    StickBreaking,
)

# Problems whose simulator is a Python loop rather than vectorised get fewer
# samples; the confidence interval widens accordingly, so the test stays valid.
SAMPLE_SIZES = {
    "expected_dice_rolls": 20_000,
    "coupon_collector": 20_000,
    "secretary_problem": 20_000,
    "ballot_problem": 20_000,
    "optimal_card_stopping": 20_000,
    "monty_hall": 20_000,
}


@pytest.mark.statistical
@pytest.mark.parametrize("problem", ALL_PROBLEMS, ids=lambda p: p.name)
def test_analytic_agrees_with_monte_carlo(problem: Problem) -> None:
    """The core contract of this module.

    The simulator is written from the problem statement and never consults
    ``analytic()``, so agreement is genuine evidence rather than self-consistency.
    """
    n = SAMPLE_SIZES.get(problem.name, 200_000)
    rng = SeedBank(root=7).child(problem.name).generator()
    result = problem.verify(n, rng, level=0.995)
    assert result.agrees, (
        f"{problem.name}: analytic {result.analytic:.6f} outside the Monte Carlo "
        f"interval {result.simulated:.6f} +/- {result.standard_error:.6f} "
        f"(z = {result.z_score:+.2f})"
    )
    assert abs(result.z_score) < 4.0


def test_gamblers_ruin_matches_closed_form_special_cases() -> None:
    # Fair coin: P = a/N exactly.
    assert GamblersRuin(start=10, target=100, win_probability=0.5).analytic() == pytest.approx(0.1)
    assert GamblersRuin(start=50, target=100, win_probability=0.5).analytic() == pytest.approx(0.5)
    # An edge helps, but capital matters more than the edge.
    small_stack = GamblersRuin(start=10, target=100, win_probability=0.51).analytic()
    big_stack = GamblersRuin(start=100, target=200, win_probability=0.51).analytic()
    assert small_stack < 0.4
    assert big_stack > 0.95


def test_gamblers_ruin_is_monotone_in_edge_and_capital() -> None:
    edges = [GamblersRuin(10, 100, p).analytic() for p in (0.48, 0.50, 0.52, 0.55)]
    assert all(a < b for a, b in itertools.pairwise(edges))
    stacks = [GamblersRuin(a, 100, 0.51).analytic() for a in (5, 10, 25, 50)]
    assert all(a < b for a, b in itertools.pairwise(stacks))


def test_secretary_problem_converges_to_one_over_e() -> None:
    """The success probability does not vanish as n grows -- that is the point."""
    for n, tolerance in [(100, 0.01), (1_000, 2e-3), (100_000, 1e-4)]:
        problem = SecretaryProblem(n_candidates=n)
        assert problem.analytic() == pytest.approx(1 / np.e, abs=tolerance)
        assert problem.optimal_cutoff() / n == pytest.approx(1 / np.e, abs=tolerance * 3)


def test_secretary_cutoff_is_the_argmax_not_an_approximation() -> None:
    """n/e rounded is off by one for small n, which shifts the third decimal."""
    problem = SecretaryProblem(n_candidates=10)
    best = problem.optimal_cutoff()
    values = [problem._success_probability(r) for r in range(1, 11)]
    assert values[best - 1] == max(values)


def test_ballot_problem_depends_only_on_the_margin() -> None:
    assert BallotProblem(60, 40).analytic() == pytest.approx(0.2)
    assert BallotProblem(600, 400).analytic() == pytest.approx(0.2)
    assert BallotProblem(3, 1).analytic() == pytest.approx(0.5)


def test_birthday_problem_classic_values() -> None:
    assert BirthdayProblem(23).analytic() == pytest.approx(0.5072972343, abs=1e-9)
    assert BirthdayProblem(50).analytic() == pytest.approx(0.9703735796, abs=1e-9)
    assert BirthdayProblem(366).analytic() == 1.0


def test_birthday_problem_is_stable_for_large_groups() -> None:
    """The log-space product must not underflow where a naive product would."""
    assert BirthdayProblem(200).analytic() == pytest.approx(1.0, abs=1e-9)
    assert np.isfinite(BirthdayProblem(365).analytic())


def test_coupon_collector_mean_and_variance() -> None:
    problem = CouponCollector(n_types=50)
    harmonic = float(np.sum(1.0 / np.arange(1, 51)))
    assert problem.analytic() == pytest.approx(50 * harmonic)
    # Variance approaches n^2 * pi^2 / 6 for large n.
    assert problem.analytic_variance() == pytest.approx(50**2 * np.pi**2 / 6, rel=0.15)


def test_stick_breaking_is_a_quarter_not_a_half() -> None:
    """The triangle inequality binds on all three pieces, not one."""
    assert StickBreaking().analytic() == 0.25


def test_monty_hall_generalises() -> None:
    assert MontyHall(3, 1).analytic() == pytest.approx(2 / 3)
    # 100 doors, 98 opened: switching wins 99% of the time.
    assert MontyHall(100, 98).analytic() == pytest.approx(0.99)
    # Switching always beats staying (which wins 1/n).
    for n in (3, 5, 10, 50):
        assert MontyHall(n, n - 2).analytic() > 1.0 / n


def test_optimal_stopping_value_exceeds_immediate_stopping() -> None:
    """The option to wait has value even when stopping now is worth zero.

    Same reasoning that makes an American option worth more than its intrinsic
    value: with 10 red and 10 black, stopping immediately has expected payoff
    exactly 0, yet the game is worth 1/2.
    """
    problem = OptimalCardStopping(n_red=10, n_black=10)
    stop_now = (problem.n_red - problem.n_black) / (problem.n_red + problem.n_black)
    assert stop_now == 0.0
    assert problem.analytic() == pytest.approx(0.5)


@pytest.mark.parametrize("n_red", [1, 2, 3, 7, 13, 26, 30])
@pytest.mark.parametrize("n_black", [1, 2, 5, 11, 26, 30])
def test_optimal_stopping_dp_matches_its_closed_form(n_red: int, n_black: int) -> None:
    """The backward induction equals r/(r+b) exactly.

    Two independent routes to the same number: the DP validates the formula, and
    the formula validates the DP's boundary conditions -- which is where backward
    induction almost always goes wrong.
    """
    problem = OptimalCardStopping(n_red=n_red, n_black=n_black)
    assert problem.analytic() == pytest.approx(problem.analytic_closed_form(), abs=1e-12)


def test_optimal_stopping_value_rises_with_the_red_share() -> None:
    values = [OptimalCardStopping(r, 10).analytic() for r in (2, 5, 10, 20, 40)]
    assert all(a < b for a, b in itertools.pairwise(values))
    # A deck with no red cards is worthless; you simply never stop.
    assert OptimalCardStopping(n_red=0, n_black=10).analytic() == 0.0


def test_problem_parameters_are_validated() -> None:
    with pytest.raises(ValueError):
        GamblersRuin(start=100, target=10)
    with pytest.raises(ValueError):
        BallotProblem(votes_a=40, votes_b=60)
    with pytest.raises(ValueError):
        MontyHall(n_doors=3, n_opened=2)  # leaves nothing to switch to


def test_verification_is_reproducible() -> None:
    problem = StickBreaking()
    a = problem.verify(50_000, SeedBank(root=1).child("x").generator())
    b = problem.verify(50_000, SeedBank(root=1).child("x").generator())
    assert a.simulated == b.simulated
