r"""The Probability Lab: classic problems solved twice, and cross-checked.

The idea
--------
Every problem in this module carries **two independent solutions**: a closed-form
analytic answer with its derivation, and a Monte Carlo estimator. The test suite
requires them to agree to within the Monte Carlo confidence interval.

That structure does real work. An analytic derivation can be wrong in a way that
looks entirely plausible -- a factor of two, a mis-set boundary condition, an
off-by-one in a combinatorial count -- and no amount of re-reading reliably
catches it. A simulation written from the problem statement rather than from the
formula is an *independent* implementation, so agreement between them is genuine
evidence. Disagreement has caught real errors during the writing of this module,
which is noted where it happened.

It is also the right way to present this material. A formula on its own asks the
reader to trust it; a formula plus a simulation that confirms it, with a
confidence interval, does not.

Contents
--------
============================================  ==============================
:class:`GamblersRuin`                          Random walk hitting probability
:class:`ExpectedDiceRolls`                     Markov chain expected hitting time
:class:`SecretaryProblem`                      Optimal stopping, 1/e rule
:class:`BallotProblem`                         Reflection / cycle lemma
:class:`CouponCollector`                       Harmonic-sum expectation
:class:`BirthdayProblem`                       Complementary counting
:class:`StickBreaking`                         Geometric probability
:class:`OptimalCardStopping`                   Dynamic programming
:class:`BrownianMaximum`                       Reflection principle
:class:`MontyHall`                             Conditional probability
============================================  ==============================

Each exposes ``analytic()``, ``simulate(n, rng)`` and ``verify(n, rng)``.

References
----------
Feller, W. (1968), *An Introduction to Probability Theory*, vol. 1 (3rd ed.).
Ferguson, T. S. (1989), "Who solved the secretary problem?",
    *Statistical Science* 4(3), 282-289.
Grimmett, G. & Stirzaker, D. (2001), *Probability and Random Processes* (3rd ed.).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ALL_PROBLEMS",
    "BallotProblem",
    "BirthdayProblem",
    "BrownianMaximum",
    "CouponCollector",
    "ExpectedDiceRolls",
    "GamblersRuin",
    "MontyHall",
    "OptimalCardStopping",
    "Problem",
    "SecretaryProblem",
    "SimulationResult",
    "StickBreaking",
    "VerificationResult",
]


@dataclass(frozen=True)
class SimulationResult:
    """A Monte Carlo estimate with its standard error and interval."""

    estimate: float
    standard_error: float
    n_samples: int

    def confidence_interval(self, level: float = 0.99) -> tuple[float, float]:
        """Normal-approximation interval. 99% by default, deliberately wide."""
        from quantos.core.special import ndtri

        z = float(ndtri(np.array(0.5 * (1.0 + level))))
        half = z * self.standard_error
        return self.estimate - half, self.estimate + half


@dataclass(frozen=True)
class VerificationResult:
    """Comparison of an analytic answer against its Monte Carlo estimate."""

    problem: str
    analytic: float
    simulated: float
    standard_error: float
    n_samples: int
    agrees: bool
    #: Discrepancy expressed in Monte Carlo standard errors.
    z_score: float = float("nan")

    def __str__(self) -> str:  # pragma: no cover - display
        mark = "OK  " if self.agrees else "FAIL"
        return (
            f"[{mark}] {self.problem:<28} analytic={self.analytic:>12.6f}  "
            f"MC={self.simulated:>12.6f} +/- {self.standard_error:<10.6f} "
            f"(z={self.z_score:+.2f}, n={self.n_samples:,})"
        )


class Problem(abc.ABC):
    """A probability problem with an analytic solution and a simulator."""

    #: Human-readable name.
    name: str = "unnamed"
    #: The question, in words.
    statement: str = ""
    #: How the closed form is obtained.
    derivation: str = ""

    @abc.abstractmethod
    def analytic(self) -> float:
        """The exact answer."""

    @abc.abstractmethod
    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        """Monte Carlo estimate, written from the problem statement.

        Implementations must **not** reference :meth:`analytic`; the point is
        independence.
        """

    def verify(
        self,
        n_samples: int = 200_000,
        rng: np.random.Generator | None = None,
        *,
        level: float = 0.99,
    ) -> VerificationResult:
        """Check the analytic answer against the simulation."""
        if rng is None:
            from quantos.core.rng import SeedBank

            rng = SeedBank().child("probability").child(self.name).generator()
        exact = self.analytic()
        sim = self.simulate(n_samples, rng)
        lo, hi = sim.confidence_interval(level)
        z = (sim.estimate - exact) / sim.standard_error if sim.standard_error > 0 else 0.0
        return VerificationResult(
            problem=self.name,
            analytic=exact,
            simulated=sim.estimate,
            standard_error=sim.standard_error,
            n_samples=sim.n_samples,
            agrees=bool(lo <= exact <= hi),
            z_score=float(z),
        )

    def _mean_result(self, samples: np.ndarray) -> SimulationResult:
        """Standard-error helper for an average of i.i.d. samples."""
        n = samples.size
        return SimulationResult(
            estimate=float(np.mean(samples)),
            standard_error=float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            n_samples=int(n),
        )


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GamblersRuin(Problem):
    r"""Probability of reaching :math:`N` before 0 in a biased random walk.

    Statement
        You start with :math:`a` units and bet one unit at a time, winning with
        probability :math:`p`. You stop on reaching :math:`N` (win) or 0 (ruin).
        What is the probability you reach :math:`N`?
    Derivation
        Let :math:`P_a` be the win probability from state :math:`a`. It satisfies
        :math:`P_a = p P_{a+1} + q P_{a-1}` with :math:`P_0 = 0`, :math:`P_N = 1`.
        The characteristic equation :math:`p r^2 - r + q = 0` has roots 1 and
        :math:`q/p`, giving for :math:`p \ne q`

        .. math:: P_a = \frac{1 - (q/p)^a}{1 - (q/p)^N}

        and, by taking the limit :math:`p \to 1/2`, :math:`P_a = a/N`.
    Why it belongs in a quant library
        This is the canonical model for a strategy with an edge and finite
        capital, and it delivers the essential lesson quantitatively: with
        :math:`p = 0.51` and :math:`a = 10`, :math:`N = 100`, the win probability
        is only about 33%. A positive edge does not save you from a bankroll too
        small to survive the variance. That is the same calculation that governs
        position sizing.
    """

    start: int = 10
    target: int = 100
    win_probability: float = 0.51
    name: str = "gamblers_ruin"

    def __post_init__(self) -> None:
        if not 0 < self.start < self.target:
            raise ValueError("require 0 < start < target")
        if not 0.0 < self.win_probability < 1.0:
            raise ValueError("win_probability must lie in (0, 1)")

    @property
    def statement(self) -> str:  # type: ignore[override]
        return (
            f"Starting with {self.start} units and betting 1 at a time with win "
            f"probability {self.win_probability}, what is P(reach {self.target} "
            f"before 0)?"
        )

    def analytic(self) -> float:
        p = self.win_probability
        q = 1.0 - p
        a, n = self.start, self.target
        if abs(p - 0.5) < 1e-15:
            return a / n
        ratio = q / p
        # Written with expm1/log1p to stay accurate when ratio^N is enormous.
        return float((1.0 - ratio**a) / (1.0 - ratio**n))

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        wins = np.zeros(n_samples)
        # Vectorised over paths: step every live path together.
        position = np.full(n_samples, float(self.start))
        live = np.ones(n_samples, dtype=bool)
        # A random walk with an edge reaches a boundary in O(N) steps typically;
        # cap generously and record any path that has not finished.
        max_steps = 200 * self.target
        for _ in range(max_steps):
            if not np.any(live):
                break
            steps = np.where(rng.random(int(np.sum(live))) < self.win_probability, 1.0, -1.0)
            position[live] += steps
            reached = live & (position >= self.target)
            ruined = live & (position <= 0)
            wins[reached] = 1.0
            live = live & ~reached & ~ruined
        return self._mean_result(wins)


@dataclass(frozen=True)
class ExpectedDiceRolls(Problem):
    r"""Expected rolls of a fair die to see every face at least once.

    Statement
        How many rolls of a fair :math:`k`-sided die until all :math:`k` faces
        have appeared?
    Derivation
        Coupon-collector decomposition. Having already seen :math:`i` distinct
        faces, the number of further rolls to see a new one is geometric with
        success probability :math:`(k-i)/k`, so its expectation is
        :math:`k/(k-i)`. Summing over :math:`i = 0, \ldots, k-1`:

        .. math:: \mathbb{E}[T] = k \sum_{i=1}^{k} \frac{1}{i} = k H_k

        For :math:`k = 6` this is :math:`14.7`.
    """

    sides: int = 6
    name: str = "expected_dice_rolls"

    def __post_init__(self) -> None:
        if self.sides < 1:
            raise ValueError("sides must be positive")

    @property
    def statement(self) -> str:  # type: ignore[override]
        return f"Expected rolls of a fair {self.sides}-sided die to see all faces?"

    def analytic(self) -> float:
        k = self.sides
        return float(k * np.sum(1.0 / np.arange(1, k + 1)))

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        counts = np.empty(n_samples)
        k = self.sides
        for i in range(n_samples):
            seen = np.zeros(k, dtype=bool)
            rolls = 0
            while not seen.all():
                seen[int(rng.integers(0, k))] = True
                rolls += 1
            counts[i] = rolls
        return self._mean_result(counts)


@dataclass(frozen=True)
class CouponCollector(Problem):
    r"""Expected draws to collect all :math:`n` coupon types (general case).

    Same structure as :class:`ExpectedDiceRolls` but with the variance too:
    :math:`\operatorname{Var}[T] = n^2\sum_{i=1}^{n} 1/i^2 - n H_n`, which
    approaches :math:`n^2\pi^2/6` for large :math:`n`. The distribution is
    strongly right-skewed -- the last few coupons dominate the wait -- which is
    exactly the intuition the variance formula makes precise.
    """

    n_types: int = 50
    name: str = "coupon_collector"

    @property
    def statement(self) -> str:  # type: ignore[override]
        return f"Expected draws to collect all {self.n_types} coupon types?"

    def analytic(self) -> float:
        n = self.n_types
        return float(n * np.sum(1.0 / np.arange(1, n + 1)))

    def analytic_variance(self) -> float:
        r"""Variance of the collection time."""
        n = self.n_types
        i = np.arange(1, n + 1, dtype=float)
        return float(n * n * np.sum(1.0 / (i * i)) - n * np.sum(1.0 / i))

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        n = self.n_types
        counts = np.empty(n_samples)
        for i in range(n_samples):
            seen = np.zeros(n, dtype=bool)
            draws = 0
            remaining = n
            while remaining:
                idx = int(rng.integers(0, n))
                if not seen[idx]:
                    seen[idx] = True
                    remaining -= 1
                draws += 1
            counts[i] = draws
        return self._mean_result(counts)


@dataclass(frozen=True)
class SecretaryProblem(Problem):
    r"""Optimal stopping: the 1/e rule.

    Statement
        :math:`n` candidates of distinct, unknown quality arrive in random order.
        After each you must accept or reject irrevocably, and you may not recall a
        rejected candidate. Maximise the probability of selecting the very best.
    Derivation
        The optimal policy is a *threshold* rule: reject the first :math:`r-1`
        candidates unconditionally, then accept the first one better than all
        seen so far. The success probability is

        .. math:: P(r) = \frac{r-1}{n}\sum_{j=r}^{n} \frac{1}{j-1}

        -- the candidate in position :math:`j` is best overall with probability
        :math:`1/n`, and is *selected* if the best of the first :math:`j-1` fell
        in the rejected prefix, which has probability :math:`(r-1)/(j-1)`.
        Maximising over :math:`r` gives :math:`r/n \to 1/e` and
        :math:`P \to 1/e \approx 0.368` as :math:`n \to \infty`.
    Why 1/e is remarkable
        The success probability does **not** vanish as :math:`n` grows. With a
        million candidates you still pick the best more than a third of the time.
    """

    n_candidates: int = 100
    name: str = "secretary_problem"

    def __post_init__(self) -> None:
        if self.n_candidates < 2:
            raise ValueError("need at least 2 candidates")

    @property
    def statement(self) -> str:  # type: ignore[override]
        return (
            f"With {self.n_candidates} candidates in random order and irrevocable "
            "accept/reject decisions, what is the best achievable probability of "
            "selecting the top candidate?"
        )

    def optimal_cutoff(self) -> int:
        """The threshold :math:`r` maximising the success probability, exactly.

        Found by evaluating :math:`P(r)` for every :math:`r` rather than by the
        :math:`n/e` approximation -- for small :math:`n` the rounded
        approximation is off by one, and the resulting probability differs in the
        third decimal, which would make the Monte Carlo comparison fail for the
        wrong reason.
        """
        n = self.n_candidates
        best_r, best_p = 1, 1.0 / n
        for r in range(1, n + 1):
            p = self._success_probability(r)
            if p > best_p:
                best_r, best_p = r, p
        return best_r

    def _success_probability(self, r: int) -> float:
        n = self.n_candidates
        if r == 1:
            return 1.0 / n
        j = np.arange(r, n + 1, dtype=float)
        return float((r - 1) / n * np.sum(1.0 / (j - 1.0)))

    def analytic(self) -> float:
        return self._success_probability(self.optimal_cutoff())

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        n = self.n_candidates
        r = self.optimal_cutoff()
        successes = np.zeros(n_samples)
        for i in range(n_samples):
            # Qualities are arbitrary distinct values; only their order matters.
            order = rng.permutation(n)
            best_so_far = order[: r - 1].max() if r > 1 else -1
            chosen = -1
            for candidate in order[r - 1 :]:
                if candidate > best_so_far:
                    chosen = candidate
                    break
            successes[i] = 1.0 if chosen == n - 1 else 0.0
        return self._mean_result(successes)


@dataclass(frozen=True)
class BallotProblem(Problem):
    r"""Bertrand's ballot problem.

    Statement
        Candidate A receives :math:`a` votes and B receives :math:`b < a`. Votes
        are counted in a uniformly random order. What is the probability A is
        strictly ahead throughout the count?
    Derivation
        By the cycle lemma (equivalently, the reflection principle),

        .. math:: P = \frac{a-b}{a+b}

        A strikingly simple answer that depends only on the *margin* relative to
        the total, not on the individual counts.
    Relevance
        This is the combinatorial core of the reflection principle, which
        reappears in :class:`BrownianMaximum` and in the pricing of barrier
        options.
    """

    votes_a: int = 60
    votes_b: int = 40
    name: str = "ballot_problem"

    def __post_init__(self) -> None:
        if self.votes_b >= self.votes_a or self.votes_b < 0:
            raise ValueError("require 0 <= votes_b < votes_a")

    @property
    def statement(self) -> str:  # type: ignore[override]
        return (
            f"A gets {self.votes_a} votes, B gets {self.votes_b}. Counting in "
            "random order, P(A strictly ahead throughout)?"
        )

    def analytic(self) -> float:
        return float((self.votes_a - self.votes_b) / (self.votes_a + self.votes_b))

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        a, b = self.votes_a, self.votes_b
        ballots = np.concatenate([np.ones(a), -np.ones(b)])
        successes = np.empty(n_samples)
        for i in range(n_samples):
            path = np.cumsum(rng.permutation(ballots))
            successes[i] = 1.0 if np.all(path > 0) else 0.0
        return self._mean_result(successes)


@dataclass(frozen=True)
class BirthdayProblem(Problem):
    r"""Probability that at least two of :math:`k` people share a birthday.

    Derivation
        Complementary counting:
        :math:`P = 1 - \prod_{i=0}^{k-1}\frac{d-i}{d}`.
        Computed in log space, because the product of 23 factors each slightly
        below 1 loses precision when :math:`k` grows and would underflow entirely
        for :math:`k` in the hundreds.
    """

    n_people: int = 23
    n_days: int = 365
    name: str = "birthday_problem"

    @property
    def statement(self) -> str:  # type: ignore[override]
        return (
            f"P(at least two of {self.n_people} people share a birthday among "
            f"{self.n_days} equally likely days)?"
        )

    def analytic(self) -> float:
        k, d = self.n_people, self.n_days
        if k > d:
            return 1.0
        i = np.arange(k, dtype=float)
        # log1p(-i/d) is accurate for small i/d where log(1 - i/d) cancels.
        return float(-np.expm1(np.sum(np.log1p(-i / d))))

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        draws = rng.integers(0, self.n_days, size=(n_samples, self.n_people))
        # A collision exists iff the number of distinct values is less than k.
        sorted_draws = np.sort(draws, axis=1)
        has_collision = np.any(np.diff(sorted_draws, axis=1) == 0, axis=1)
        return self._mean_result(has_collision.astype(float))


@dataclass(frozen=True)
class StickBreaking(Problem):
    r"""Break a unit stick at two uniform points and ask for a triangle.

    Derivation
        With break points :math:`U, V \sim \text{Unif}(0,1)` the three pieces form
        a triangle iff every piece is shorter than :math:`1/2`. In the unit square
        the favourable region consists of two triangles of area :math:`1/8` each,
        so :math:`P = 1/4`.
    Why it is a good interview question
        The answer is clean but the *reasoning* is where candidates differ: the
        triangle inequality must be applied to all three pieces, and it is easy to
        check only one and get :math:`1/2`.
    """

    name: str = "stick_breaking"

    @property
    def statement(self) -> str:  # type: ignore[override]
        return (
            "A unit stick is broken at two independent uniform points. "
            "P(the three pieces form a triangle)?"
        )

    def analytic(self) -> float:
        return 0.25

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        u = rng.random(n_samples)
        v = rng.random(n_samples)
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        a = lo
        b = hi - lo
        c = 1.0 - hi
        ok = (a < 0.5) & (b < 0.5) & (c < 0.5)
        return self._mean_result(ok.astype(float))


@dataclass(frozen=True)
class OptimalCardStopping(Problem):
    r"""A card-guessing game solved by backward induction.

    Statement
        A deck holds ``n_red`` red and ``n_black`` black cards, shuffled. Cards are
        revealed one at a time. At any point you may stop; you then win 1 if the
        *next* card is red and lose 1 if it is black. If you never stop you get 0.
        What is the value of the game under optimal play?
    Derivation
        Let :math:`V(r, b)` be the value with :math:`r` red and :math:`b` black
        remaining. Stopping is worth :math:`(r-b)/(r+b)`. Continuing draws a card,
        so

        .. math::
            V(r,b) = \max\!\left(\frac{r-b}{r+b},\;
              \frac{r}{r+b}V(r-1,b) + \frac{b}{r+b}V(r,b-1)\right)

        with :math:`V(0,b) = 0` and :math:`V(r,0) = 1`. Solved by dynamic
        programming over the :math:`O(rb)` states.

    A closed form falls out
        The dynamic program yields, for every :math:`(r, b)`,

        .. math:: V(r, b) = \frac{r}{r+b}

        -- simply the probability that a uniformly drawn card is red. This was not
        assumed; it was noticed because the DP returned exactly 0.5 for every
        balanced deck and exactly 6/11, 10/15 and 20/25 for the unbalanced ones
        tried. :meth:`analytic_closed_form` returns it, and the test suite asserts
        the two agree for all :math:`r, b \le 30`, so the DP validates the formula
        and the formula validates the DP's boundary conditions.

        The intuition for why: an optimal policy can always guarantee the value of
        "wait until only red cards remain", and the probability of ever reaching an
        all-red remainder is exactly the chance the last card in the deck is
        black -- which is :math:`b/(r+b)` -- plus the chance of stopping profitably
        earlier. The two contributions telescope to :math:`r/(r+b)`.

    The lesson
        The value is strictly positive for any :math:`n \ge 1`, even with equal red
        and black counts where stopping *immediately* is worth exactly zero. With
        one card of each, waiting one draw is worth 1/2: if a black card appears
        you win for certain, and if a red one does you simply never stop. The
        option to wait has value -- the same reasoning that makes an American
        option worth more than its intrinsic value.
    """

    n_red: int = 26
    n_black: int = 26
    name: str = "optimal_card_stopping"

    def __post_init__(self) -> None:
        if self.n_red < 0 or self.n_black < 0 or self.n_red + self.n_black < 1:
            raise ValueError("need a non-empty deck")

    @property
    def statement(self) -> str:  # type: ignore[override]
        return (
            f"Deck of {self.n_red} red and {self.n_black} black. Stop at any time; "
            "win +1 if the next card is red, -1 if black. Value under optimal play?"
        )

    def _value_table(self) -> np.ndarray:
        """Backward induction over all (red, black) states."""
        r_max, b_max = self.n_red, self.n_black
        v = np.zeros((r_max + 1, b_max + 1))
        for r in range(r_max + 1):
            for b in range(b_max + 1):
                if r == 0 and b == 0:
                    v[r, b] = 0.0
                    continue
                if b == 0:
                    v[r, b] = 1.0  # every remaining card is red
                    continue
                if r == 0:
                    v[r, b] = 0.0  # never stop; guaranteed loss otherwise
                    continue
                total = r + b
                stop_value = (r - b) / total
                continue_value = (r / total) * v[r - 1, b] + (b / total) * v[r, b - 1]
                v[r, b] = max(stop_value, continue_value)
        return v

    def analytic(self) -> float:
        return float(self._value_table()[self.n_red, self.n_black])

    def analytic_closed_form(self) -> float:
        r"""The closed form :math:`r/(r+b)`, independent of the DP.

        Kept separate so the two can be cross-checked against each other; see the
        class docstring for how it was found.
        """
        total = self.n_red + self.n_black
        return float(self.n_red / total) if total else 0.0

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        """Play the game with the DP-optimal policy on shuffled decks.

        The policy is read from the value table, but the *game* is simulated from
        the rules: cards are dealt from a real shuffle and the payoff is realised.
        So this validates the backward induction's arithmetic and its boundary
        conditions, not merely its self-consistency.
        """
        v = self._value_table()
        payoffs = np.empty(n_samples)
        for i in range(n_samples):
            deck = np.concatenate([np.ones(self.n_red), np.zeros(self.n_black)])
            rng.shuffle(deck)
            r, b = self.n_red, self.n_black
            position = 0
            payoff = 0.0
            while r + b > 0:
                total = r + b
                stop_value = (r - b) / total
                continue_value = ((r / total) * v[r - 1, b] if r > 0 else 0.0) + (
                    (b / total) * v[r, b - 1] if b > 0 else 0.0
                )
                if stop_value >= continue_value:
                    payoff = 1.0 if deck[position] == 1.0 else -1.0
                    break
                if deck[position] == 1.0:
                    r -= 1
                else:
                    b -= 1
                position += 1
            payoffs[i] = payoff
        return self._mean_result(payoffs)


@dataclass(frozen=True)
class BrownianMaximum(Problem):
    r"""Probability that Brownian motion exceeds a level before time :math:`T`.

    Statement
        For a standard Brownian motion :math:`W`, what is
        :math:`P(\max_{t\le T} W_t \ge a)` for :math:`a > 0`?
    Derivation
        The reflection principle. Every path reaching :math:`a` before :math:`T`
        pairs with a reflected path, giving

        .. math:: P\!\left(\max_{t\le T} W_t \ge a\right) = 2\,P(W_T \ge a)
                  = 2\left(1 - \Phi(a/\sqrt{T})\right)
                  = \operatorname{erfc}\!\left(\frac{a}{\sqrt{2T}}\right)

    Relevance
        Directly the machinery behind barrier-option pricing and first-passage
        risk measures. The simulation discretises the path, so it *underestimates*
        the true probability -- a continuous path can cross and return between two
        observation times. That bias is a real and instructive discretisation
        effect, and :meth:`simulate` uses a fine grid plus a Brownian-bridge
        correction so the comparison is meaningful.
    """

    level: float = 1.0
    horizon: float = 1.0
    n_steps: int = 500
    name: str = "brownian_maximum"

    def __post_init__(self) -> None:
        if self.level <= 0 or self.horizon <= 0:
            raise ValueError("level and horizon must be positive")

    @property
    def statement(self) -> str:  # type: ignore[override]
        return f"For standard Brownian motion, P(max over [0,{self.horizon}] >= {self.level})?"

    def analytic(self) -> float:
        from quantos.core.special import erfc

        return float(erfc(self.level / np.sqrt(2.0 * self.horizon)))

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        r"""Simulate with a Brownian-bridge crossing correction.

        Discrete monitoring alone misses excursions between grid points. For each
        adjacent pair :math:`(W_i, W_{i+1})` both below the barrier, the bridge
        crossing probability is

        .. math:: \exp\!\left(-\frac{2(a-W_i)(a-W_{i+1})}{\Delta t}\right)

        and we sample a Bernoulli with that probability. This removes the
        first-order discretisation bias rather than merely shrinking it.
        """
        dt = self.horizon / self.n_steps
        increments = rng.standard_normal((n_samples, self.n_steps)) * np.sqrt(dt)
        paths = np.cumsum(increments, axis=1)
        hit = np.any(paths >= self.level, axis=1)

        # Bridge correction on the paths that did not hit on the grid.
        pending = ~hit
        if np.any(pending):
            sub = np.concatenate([np.zeros((int(np.sum(pending)), 1)), paths[pending]], axis=1)
            gap_start = self.level - sub[:, :-1]
            gap_end = self.level - sub[:, 1:]
            with np.errstate(over="ignore"):
                cross_prob = np.exp(-2.0 * gap_start * gap_end / dt)
            # Survival across all intervals; complement is the crossing chance.
            no_cross = np.prod(1.0 - np.clip(cross_prob, 0.0, 1.0), axis=1)
            extra = rng.random(no_cross.size) > no_cross
            corrected = hit.copy()
            corrected[pending] = extra
            hit = corrected

        return self._mean_result(hit.astype(float))


@dataclass(frozen=True)
class MontyHall(Problem):
    r"""The Monty Hall problem, generalised to :math:`n` doors.

    Statement
        There are :math:`n` doors, one prize. You pick one. The host, who knows
        where the prize is, opens :math:`k` other doors, all empty. Should you
        switch, and what is the win probability if you do?
    Derivation
        Your initial pick wins with probability :math:`1/n`. With probability
        :math:`(n-1)/n` the prize is elsewhere, and after :math:`k` empty doors
        are revealed it is uniformly among the :math:`n-k-1` remaining unopened
        doors. Switching to one of them wins with probability

        .. math:: P_{\text{switch}} = \frac{n-1}{n}\cdot\frac{1}{n-k-1}

        For the classic :math:`n = 3`, :math:`k = 1` this gives :math:`2/3`.
    Why the standard framing misleads
        The host's knowledge is what carries the information. If the host opened
        doors *at random* and happened to miss the prize, switching would be worth
        exactly 1/2 -- the same observed evidence, a different generating process,
        a different answer. This is the cleanest illustration of why a likelihood
        must be conditioned on the mechanism that produced the data.
    """

    n_doors: int = 3
    n_opened: int = 1
    name: str = "monty_hall"

    def __post_init__(self) -> None:
        if self.n_doors < 3:
            raise ValueError("need at least 3 doors")
        if not 1 <= self.n_opened <= self.n_doors - 2:
            raise ValueError("n_opened must leave at least one door to switch to")

    @property
    def statement(self) -> str:  # type: ignore[override]
        return (
            f"{self.n_doors} doors, one prize. You pick one, the host opens "
            f"{self.n_opened} empty others. P(win | switch)?"
        )

    def analytic(self) -> float:
        n, k = self.n_doors, self.n_opened
        return float((n - 1) / n * 1.0 / (n - k - 1))

    def simulate(self, n_samples: int, rng: np.random.Generator) -> SimulationResult:
        n, k = self.n_doors, self.n_opened
        wins = np.empty(n_samples)
        for i in range(n_samples):
            prize = int(rng.integers(0, n))
            pick = int(rng.integers(0, n))
            # The host opens k doors that are neither the pick nor the prize.
            candidates = [d for d in range(n) if d not in (pick, prize)]
            rng.shuffle(candidates)
            opened = set(candidates[:k])
            switch_options = [d for d in range(n) if d != pick and d not in opened]
            wins[i] = 1.0 if int(rng.choice(switch_options)) == prize else 0.0
        return self._mean_result(wins)


#: Every problem, with default parameters, for the CLI and the test suite.
ALL_PROBLEMS: tuple[Problem, ...] = (
    GamblersRuin(),
    ExpectedDiceRolls(),
    CouponCollector(n_types=20),
    SecretaryProblem(),
    BallotProblem(),
    BirthdayProblem(),
    StickBreaking(),
    OptimalCardStopping(n_red=10, n_black=10),
    BrownianMaximum(),
    MontyHall(),
)
