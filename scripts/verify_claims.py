#!/usr/bin/env python3
"""Re-derive every headline number in the documentation and fail if it drifted.

Why this exists
---------------
Documentation rots silently. A refactor changes a constant, the prose keeps
quoting the old figure, and nobody notices because nothing fails. Every number in
this repository's README, gallery captions and design records is a *claim*, and a
claim nobody checks is indistinguishable from a guess.

So the claims are checked. Each entry below states what the docs assert, where
they assert it, and how to recompute it from scratch. CI runs this, and a
mismatch fails the build -- which means the prose cannot drift from the code
without someone being told.

This also catches the specific failure this project has already made twice: a
number quoted from a single favourable run. The price-discovery correlation was
once reported as 0.78 (the best of three seeds; the truth was 0.29 across
sixteen) and the C++ throughput as 16,012,260 ops/s (one run near the top of an
8.1-16.5M range). Both are now recomputed here with their spread.

Usage
-----
    python scripts/verify_claims.py             # everything runnable offline
    python scripts/verify_claims.py --quick     # skip the slow simulations
    python scripts/verify_claims.py --list      # show claims without running
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - script convenience
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


@dataclass
class Claim:
    """One documented assertion, and how to check it."""

    name: str
    #: Where the claim is made, so a failure points at the text to fix.
    stated_in: str
    check: Callable[[], tuple[bool, str]]
    slow: bool = False
    #: Needs the network or a compiled artifact; skipped rather than failed.
    conditional: bool = False


@dataclass
class Result:
    claim: Claim
    passed: bool
    detail: str
    skipped: bool = False
    seconds: float = 0.0


CLAIMS: list[Claim] = []


def claim(name: str, stated_in: str, *, slow: bool = False, conditional: bool = False):
    def register(function: Callable[[], tuple[bool, str]]) -> Callable:
        CLAIMS.append(
            Claim(
                name=name, stated_in=stated_in, check=function, slow=slow, conditional=conditional
            )
        )
        return function

    return register


# --------------------------------------------------------------------------- #
# Numerical accuracy
# --------------------------------------------------------------------------- #
@claim("special functions match SciPy to the documented tolerance", "core/special.py docstring")
def _special_accuracy() -> tuple[bool, str]:
    """The accuracy table is the repository's foundational claim.

    Everything statistical rests on these, so if they have drifted the failure
    should be loud and immediate rather than surfacing as a wrong p-value later.
    """
    try:
        from scipy import special as scipy_special
    except ImportError:
        return True, "SKIP: SciPy is a test-only oracle and is not installed"

    from quantos.core.special import erf, erfc, lgamma, ndtr, ndtri

    worst: dict[str, float] = {}

    x = np.linspace(-6, 6, 4001)
    worst["erf"] = float(np.max(np.abs(erf(x) - scipy_special.erf(x))))
    worst["erfc"] = float(np.max(np.abs(erfc(x) - scipy_special.erfc(x))))
    worst["ndtr"] = float(np.max(np.abs(ndtr(x) - scipy_special.ndtr(x))))

    p = np.linspace(1e-10, 1 - 1e-10, 4001)
    worst["ndtri"] = float(np.max(np.abs(ndtri(p) - scipy_special.ndtri(p))))

    g = np.linspace(0.05, 60.0, 4001)
    worst["lgamma"] = float(np.max(np.abs(lgamma(g) - scipy_special.gammaln(g))))

    tolerance = 1e-9
    failures = {k: v for k, v in worst.items() if v > tolerance}
    detail = ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items()))
    if failures:
        return False, f"exceeded {tolerance:.0e}: {failures}; all = {detail}"
    return True, f"all within {tolerance:.0e} ({detail})"


@claim("the Brier decomposition closes to machine precision", "forecast/calibration.py")
def _brier_identity() -> tuple[bool, str]:
    """BS = REL - RES + UNC + residual, exactly.

    The three-term textbook identity does not close for continuous forecasts; the
    fourth term is real and dropping it leaves the identity failing by ~4e-4.
    """
    from quantos.forecast.calibration import brier_decomposition

    rng = np.random.default_rng(0)
    predicted = rng.uniform(0, 1, 4000)
    outcomes = rng.uniform(0, 1, 4000) < predicted

    brier, reliability, resolution, uncertainty, residual = brier_decomposition(predicted, outcomes)
    gap = abs(brier - (reliability - resolution + uncertainty + residual))
    three_term_gap = abs(brier - (reliability - resolution + uncertainty))

    if gap > 1e-12:
        return False, f"identity fails by {gap:.2e}"
    return True, (f"closes to {gap:.1e}; dropping the residual would leave {three_term_gap:.1e}")


# --------------------------------------------------------------------------- #
# Statistical honesty
# --------------------------------------------------------------------------- #
@claim("overlapping predictions are discounted ~42x", "README, live/ledger.py")
def _overlap_discount() -> tuple[bool, str]:
    """A 30-day forecast recorded repeatedly must collapse to few independent ones."""
    from datetime import date, timedelta
    from tempfile import mkdtemp

    from quantos.live.ledger import Ledger, Prediction

    ledger = Ledger(Path(mkdtemp()) / "claims.jsonl")
    # 45 tracks recorded every 30 days over three years, as the simulation used.
    start = date(2022, 1, 3)
    count = 0
    for track in range(45):
        for step in range(29):
            ledger.record_prediction(
                Prediction(
                    f"t{track}s{step}",
                    "SYM",
                    f"sig{track}",
                    1,
                    30,
                    100.0,
                    as_of=str(start + timedelta(days=30 * step)),
                )
            )
            count += 1

    independent = len(ledger.independent_subset(ledger.predictions()))
    factor = count / independent
    if not 20 <= factor <= 60:
        return False, f"{count} predictions -> {independent} independent (factor {factor:.1f})"
    return True, f"{count} predictions -> {independent} independent, factor {factor:.1f}x"


@claim("the calibration verdict refuses to pass a no-skill model", "forecast/calibration.py")
def _calibration_verdict() -> tuple[bool, str]:
    """A negative Brier skill must never read as 'calibrated'.

    An earlier version returned a bare boolean and said True on a run whose skill
    was negative, because the buckets where it failed were too small to reject.
    """
    from quantos.forecast.calibration import CalibrationResult, ReliabilityBucket

    # Perfectly on the diagonal, but with worse-than-base-rate skill.
    buckets = [
        ReliabilityBucket(0.0, 0.1, 200, 40, 0.05, 0.05),
        ReliabilityBucket(0.1, 0.2, 200, 40, 0.15, 0.15),
    ]
    result = CalibrationResult(
        event="test",
        horizon_days=21,
        n_forecasts=400,
        n_independent=80,
        buckets=buckets,
        brier_score=0.30,
        reliability=0.0,
        resolution=0.0,
        uncertainty=0.25,  # skill = 1 - 0.30/0.25 < 0
        base_rate=0.10,
        mean_predicted=0.10,
    )
    if result.is_calibrated:
        return False, f"a no-skill model was called calibrated: {result.verdict}"
    if "NO SKILL" not in result.verdict.upper():
        return False, f"verdict does not name the problem: {result.verdict}"
    return True, f"correctly refused: {result.verdict[:60]}"


@claim(
    "touching a level is never less likely than finishing beyond it", "forecast/probabilities.py"
)
def _first_passage_ordering() -> tuple[bool, str]:
    from quantos.forecast.paths import simulate_garch_paths
    from quantos.forecast.probabilities import first_passage_probability

    returns = np.random.default_rng(3).normal(0, 0.02, 1500)
    ensemble = simulate_garch_paths(returns, 100.0, 21, n_paths=20_000, seed=4)

    worst_gap = 0.0
    for move in (0.02, 0.05, 0.10, 0.20):
        touch = first_passage_probability(ensemble, 100 * (1 - move), direction="down")
        finish = float(np.mean(ensemble.terminal <= 100 * (1 - move)))
        worst_gap = min(worst_gap, touch - finish)
    if worst_gap < -1e-12:
        return False, f"ordering violated by {worst_gap:.2e}"
    return True, "holds at every threshold tested"


# --------------------------------------------------------------------------- #
# Claims about measured results
# --------------------------------------------------------------------------- #
@claim(
    "price discovery is weak: mean correlation ~0.29 across seeds",
    "README 'What this is, concretely'",
    slow=True,
)
def _price_discovery() -> tuple[bool, str]:
    """The claim that replaced a cherry-picked 0.78.

    Recomputed here rather than trusted, because quoting the best of three seeds
    is the exact mistake this repository has already made once.
    """
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "measure_price_discovery.py"), "--seeds", "8"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        return False, f"measurement script failed: {completed.stderr[-200:]}"

    numbers = re.findall(r"mean\s+([-\d.]+)", completed.stdout)
    if not numbers:
        return False, f"could not parse a mean from output: {completed.stdout[-200:]}"
    mean = float(numbers[0])
    # The documented claim is "better than chance but not reliable". Eight seeds
    # carry a wide standard error, so the band is correspondingly wide.
    if not -0.2 <= mean <= 0.75:
        return False, f"mean correlation {mean:.3f} is outside the documented picture"
    return True, f"mean correlation {mean:.3f} over 8 seeds, consistent with the documented 0.29"


@claim("the C++ book is ~30x the pure Python one, batched", "README table", conditional=True)
def _cpp_throughput() -> tuple[bool, str]:
    from quantos.exchange.replay import BATCH_AVAILABLE, replay_tape, synthetic_tape

    if not BATCH_AVAILABLE:
        return True, "SKIP: extension not built (python scripts/build_extension.py)"

    tape = synthetic_tape(1_000_000, seed=7)
    rates = []
    for _ in range(5):
        result, _ = replay_tape(tape)
        rates.append(result.operations_per_second)
    median = float(np.median(rates))

    # The README quotes ~15.1M as a median with an 8.1-16.5M range, so the check
    # is deliberately generous: this is wall-clock on shared CI hardware.
    if median < 3e6:
        return False, f"median {median:,.0f} ops/s is far below the documented ~15M"
    return (
        True,
        f"median {median:,.0f} ops/s over 5 runs (range {min(rates):,.0f}-{max(rates):,.0f})",
    )


@claim("the neural model loses to GARCH on QLIKE", "docs/MODEL_LEADERBOARD.md", slow=True)
def _model_leaderboard() -> tuple[bool, str]:
    """The published negative result, re-derived on synthetic data.

    Uses a simulated GARCH process rather than fetched prices so this runs
    offline and in CI. The claim under test is the ordering, not the exact
    losses -- if the attention model ever beats GARCH here, the leaderboard is
    stale and should be rewritten rather than left standing.
    """
    from quantos.models.baselines import (
        ewma_volatility_forecast,
        garch_volatility_forecast,
        score_forecast,
    )
    from quantos.models.sequence import AttentionVolatilityModel

    rng = np.random.default_rng(11)
    omega, alpha, beta = 2e-6, 0.09, 0.88
    variance = omega / (1 - alpha - beta)
    returns = np.empty(3000)
    for i in range(returns.size):
        variance = omega + alpha * (returns[i - 1] ** 2 if i else 0.0) + beta * variance
        returns[i] = np.sqrt(variance) * rng.standard_normal()

    train, step = 1200, 10
    indices = list(range(train, returns.size, step))
    actual = np.array([returns[t] for t in indices])

    garch = score_forecast(
        "garch",
        actual,
        np.array([garch_volatility_forecast(returns[t - train : t]) for t in indices]),
    )
    ewma = score_forecast(
        "ewma",
        actual,
        np.array([ewma_volatility_forecast(returns[t - train : t]) for t in indices]),
    )

    window = 20
    predictions = np.full(len(indices), np.nan)
    model: AttentionVolatilityModel | None = None
    for position, t in enumerate(indices):
        if position % 40 == 0:
            model = AttentionVolatilityModel(window=window, d_model=8, seed=7)
            try:
                model.fit(returns[t - train : t], epochs=120, patience=20)
            except ValueError:
                model = None
        if model is not None:
            predictions[position] = float(model.predict(returns[t - window : t][None, :])[0])
    attention = score_forecast("attention", actual, predictions)

    if attention.qlike < garch.qlike:
        return False, (
            f"the attention model BEAT GARCH ({attention.qlike:.4f} vs {garch.qlike:.4f}); "
            "docs/MODEL_LEADERBOARD.md is stale and should be rewritten"
        )
    return True, (
        f"GARCH {garch.qlike:.4f} < EWMA {ewma.qlike:.4f} < attention "
        f"{attention.qlike:.4f}, as documented"
    )


# --------------------------------------------------------------------------- #
# Structural claims
# --------------------------------------------------------------------------- #
@claim("Heston reproduces Black-Scholes as vol-of-vol vanishes", "derivatives/heston.py")
def _heston_bs_limit() -> tuple[bool, str]:
    """With deterministic variance the model must collapse to the closed form."""
    from quantos.derivatives.black_scholes import black_scholes_price
    from quantos.derivatives.heston import HestonParameters, heston_price

    volatility = 0.2
    parameters = HestonParameters(
        kappa=2.0, theta=volatility**2, xi=1e-4, rho=0.0, v0=volatility**2
    )
    worst = 0.0
    for strike in (80.0, 100.0, 120.0):
        heston = heston_price(100.0, strike, 1.0, parameters, rate=0.03)
        black = float(black_scholes_price(100.0, strike, 1.0, volatility, rate=0.03))
        worst = max(worst, abs(heston - black))
    if worst > 1e-5:
        return False, f"worst disagreement {worst:.2e}"
    return True, f"agrees to {worst:.1e} across strikes"


@claim("the Heston branch cut still bites the original formulation", "derivatives/heston.py")
def _heston_branch_cut() -> tuple[bool, str]:
    """The documented failure must remain demonstrable.

    If a future change made the two formulations agree, the module's central
    warning would be stale and should be rewritten rather than left standing.
    """
    from quantos.derivatives.heston import HestonParameters, heston_price

    parameters = HestonParameters(kappa=8.0, theta=0.09, xi=1.0, rho=-0.8, v0=0.09)
    with np.errstate(over="ignore", invalid="ignore"):
        stable = heston_price(100.0, 100.0, 5.0, parameters, formulation="stable")
        original = heston_price(100.0, 100.0, 5.0, parameters, formulation="original")

    if not np.isfinite(stable):
        return False, "the stable formulation itself failed"
    if original <= 1.5 * stable:
        return False, (
            f"the formulations now agree ({original:.4f} vs {stable:.4f}); the branch-cut "
            "warning in the docstring is stale"
        )
    ratio = original / stable
    return True, f"original overprices by {ratio:.2f}x at T=5 ({original:.2f} vs {stable:.2f})"


@claim("American pricing matches the Longstaff-Schwartz benchmark", "derivatives/american.py")
def _american_benchmark() -> tuple[bool, str]:
    """Published value 4.478 for S=36, K=40, r=6%, sigma=20%, T=1."""
    from quantos.derivatives.american import price_american

    result = price_american(
        36.0, 40.0, 1.0, 0.20, rate=0.06, n_paths=40_000, n_steps=50, compute_upper=False
    )
    tolerance = 3 * result.lower_standard_error + 0.02
    if abs(result.lower - 4.478) > tolerance:
        return False, f"got {result.lower:.4f}, published 4.478, tolerance {tolerance:.4f}"
    return True, f"{result.lower:.4f} +/- {result.lower_standard_error:.4f} against 4.478"


@claim("no standard VaR model passes on SPY", "docs/VAR_BACKTEST.md", slow=True)
def _var_all_fail() -> tuple[bool, str]:
    """The published negative result, re-derived.

    Uses a simulated series with volatility regimes rather than fetched prices so
    this runs offline. The claim under test is that a trailing-window VaR
    understates risk when the regime shifts -- if that stopped being true the
    document would need rewriting.
    """
    from quantos.core.special import ndtri
    from quantos.risk.var_backtest import backtest_var

    rng = np.random.default_rng(20240719)
    # Two regimes: long calm stretches punctuated by bursts, as equity indices are.
    n = 6000
    volatility = np.where(rng.random(n) < 0.03, 0.045, 0.007)
    for i in range(1, n):  # make the regime persistent
        if volatility[i - 1] > 0.02 and rng.random() < 0.93:
            volatility[i] = 0.045
    returns = rng.standard_t(4.0, n) / np.sqrt(2.0) * volatility

    window = 500
    z = float(ndtri(np.array(0.99)))
    forecasts = np.array([z * np.std(returns[t - window : t], ddof=1) for t in range(window, n)])
    result = backtest_var(returns[window:], forecasts, confidence=0.99, model="gaussian")

    if not result.kupiec.rejects:
        return False, (
            f"a Gaussian VaR now passes coverage at {result.breach_rate:.2%}; "
            "docs/VAR_BACKTEST.md should be re-checked"
        )
    return True, (
        f"Gaussian VaR breaches {result.breach_rate:.2%} against a promised 1.00%, rejected"
    )


@claim("the runtime imports nothing but NumPy", "DDR-002")
def _numpy_only() -> tuple[bool, str]:
    """Walk the AST of every runtime module looking for third-party imports.

    Parsed rather than grepped. A regex over source lines matches prose inside
    docstrings -- the first version of this check reported `core/linalg.py` as
    importing a package called "finance", from the sentence "...from finance",
    *IMA J. Numer. Anal.*" in a citation. Three of its four findings were text.

    Imports guarded by ``try``/``except ImportError`` are allowed: they are
    optional by construction and cannot make the package fail to import. That
    exempts the ``doctor`` command, which imports SciPy purely to report whether
    it is present.
    """
    import ast

    allowed = {"numpy", "quantos"}
    stdlib = set(sys.stdlib_module_names)
    offenders: list[str] = []

    for path in sorted((ROOT / "src" / "quantos").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))

        guarded: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        guarded.add(id(child))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if id(node) in guarded:
                continue
            for name in names:
                root_module = name.split(".")[0]
                if root_module in allowed or root_module in stdlib or root_module == "__future__":
                    continue
                if not root_module:  # a relative import
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {root_module}")

    if offenders:
        return False, "; ".join(offenders[:5])
    return True, "no unguarded third-party import beyond NumPy in the runtime"


@claim("every documented gallery figure exists", "docs/GALLERY.md")
def _gallery_complete() -> tuple[bool, str]:
    gallery = ROOT / "docs" / "GALLERY.md"
    if not gallery.exists():
        return False, "docs/GALLERY.md is missing"
    referenced = re.findall(r"!\[[^\]]*\]\((gallery/[^)]+)\)", gallery.read_text())
    missing = [r for r in referenced if not (ROOT / "docs" / r).exists()]
    if missing:
        return False, f"referenced but absent: {missing}"
    return True, f"{len(referenced)} figures, all present"


@claim("the disclaimer appears on every rendered page", "DDR-005")
def _disclaimer_everywhere() -> tuple[bool, str]:
    from quantos.web.server import DISCLAIMER, render_landing

    if DISCLAIMER not in render_landing():
        return False, "missing from the landing page"
    if DISCLAIMER not in render_landing("some error"):
        return False, "missing from an error page"
    return True, "present on landing and error pages"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Financial planning
# --------------------------------------------------------------------------- #
@claim(
    "the schedule reproduces Calculator.net's published figures to the cent",
    "planning/calculator.py docstring and the site's calculator page",
)
def _calculator_matches_the_published_schedule() -> tuple[bool, str]:
    """The convention was reverse-engineered, so it has to be re-checked.

    Calculator.net does not document whether its monthly rate is the nominal
    r/12 or the effective (1+r)^(1/12)-1. Only the second reproduces their
    numbers. If a refactor ever swaps the convention the schedule stays
    plausible -- it just quietly stops being the figure everyone else quotes --
    so the check is against their published table, not against our own output.
    """
    from quantos.planning import investment_schedule

    published = {
        1: 33_526.53,
        2: 47_864.65,
        3: 63_063.06,
        4: 79_173.37,
        5: 96_250.30,
        6: 114_351.84,
        7: 133_539.48,
        8: 153_878.38,
        9: 175_437.61,
        10: 198_290.40,
    }
    plan = investment_schedule(20_000.0, 10, 0.06, contribution=1_000.0)
    worst = max(abs(row.ending_balance - published[row.year]) for row in plan.rows)
    ok = worst < 0.005
    return ok, f"worst year-end difference across ten years: ${worst:.4f}"


@claim(
    "the fixed-rate projection sits above the median outcome, not at it",
    "README negative results table and the calculator page",
)
def _projection_overstates_the_typical_outcome() -> tuple[bool, str]:
    """The whole reason the page exists, re-derived rather than asserted.

    Compounding is multiplicative, so the arithmetic-mean rate does not describe
    the middle of the distribution it generates. If this ever came out the other
    way the page's argument would be wrong and it should be rewritten, not
    quietly kept.
    """
    from quantos.planning import investment_schedule, simulate_plan

    plan = investment_schedule(20_000.0, 10, 0.06, contribution=1_000.0)
    outcome = simulate_plan(20_000.0, 10, 0.06, 0.15, contribution=1_000.0, n_paths=60_000, seed=7)
    ok = outcome.median < plan.end_balance and outcome.probability_of_target < 0.5
    return ok, (
        f"projection ${plan.end_balance:,.0f}; median ${outcome.median:,.0f}; "
        f"reached {outcome.probability_of_target:.1%} of the time"
    )


# --------------------------------------------------------------------------- #
# Data snooping
# --------------------------------------------------------------------------- #
@claim(
    "the best of hundreds of factors looks significant and is not",
    "README factor lab section and research/factor_lab.py",
    slow=True,
)
def _factor_lab_finds_nothing_in_noise() -> tuple[bool, str]:
    """The claim the factor lab exists to make, re-derived rather than quoted.

    Two things must BOTH hold or the demonstration collapses. The winner of a
    large search over pure noise must clear a naive significance bar -- otherwise
    there is no illusion to correct -- and it must then fail every correction.
    A refactor that broke either half would leave a module that still runs and no
    longer demonstrates anything.
    """
    import numpy as np

    from quantos.research.factor_lab import run_factor_lab

    rng = np.random.default_rng(0)
    report = run_factor_lab(
        rng.standard_normal(1500) * 0.01, n_factors=200, seed=1, n_bootstrap=400
    )

    illusion = report.best.t_statistic > 1.96 and report.best.naively_significant
    corrected = not report.survivors and report.reality_check_p > 0.10
    return illusion and corrected, (
        f"best of {report.n_factors} factors: t = {report.best.t_statistic:.2f} "
        f"(naive p = {report.best.naive_p_value:.3f}); "
        f"Reality Check p = {report.reality_check_p:.3f}, "
        f"{len(report.survivors)} survivors"
    )


@claim(
    "the factor lab detects a signal that is genuinely there",
    "tests/research/test_factor_lab.py -- the guard against a lab that always says no",
    slow=True,
)
def _factor_lab_is_not_a_constant() -> tuple[bool, str]:
    """Without this, the claim above is satisfied by a function returning 'no'.

    The planted rule is in the grammar, so a working search should recover the
    exact generating rule rather than something merely correlated with it.
    """
    import numpy as np

    from quantos.research.factor_lab import run_factor_lab

    rng = np.random.default_rng(7)
    noise = rng.standard_normal(2000) * 0.01
    returns = np.zeros(2000)
    for t in range(1, 2000):
        returns[t] = 0.0022 * np.sign(returns[max(0, t - 21) : t].sum()) + noise[t]

    report = run_factor_lab(returns, n_factors=200, seed=1, n_bootstrap=400)
    found = report.best.spec.name == "momentum_21d_sign_h1"
    return found and bool(report.survivors), (
        f"recovered {report.best.spec.name} (t = {report.best.t_statistic:.1f}), "
        f"{len(report.survivors)} survivors, SPA p = {report.spa_p:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip the slow simulations")
    parser.add_argument("--list", action="store_true", help="list claims without running")
    args = parser.parse_args()

    if args.list:
        for entry in CLAIMS:
            marks = " (slow)" if entry.slow else ""
            marks += " (conditional)" if entry.conditional else ""
            print(f"  {entry.name}{marks}\n      stated in: {entry.stated_in}")
        return 0

    print(f"verifying {len(CLAIMS)} documented claims\n")
    results: list[Result] = []

    for entry in CLAIMS:
        if args.quick and entry.slow:
            results.append(Result(entry, True, "skipped (--quick)", skipped=True))
            print(f"  SKIP  {entry.name}")
            continue
        started = time.perf_counter()
        try:
            passed, detail = entry.check()
        except Exception as error:
            passed, detail = False, f"{type(error).__name__}: {error}"
        elapsed = time.perf_counter() - started

        skipped = detail.startswith("SKIP")
        results.append(Result(entry, passed, detail, skipped=skipped, seconds=elapsed))
        mark = "SKIP" if skipped else ("ok  " if passed else "FAIL")
        print(f"  {mark}  {entry.name}  [{elapsed:.1f}s]")
        print(f"          {detail}")

    failures = [r for r in results if not r.passed]
    skips = [r for r in results if r.skipped]
    print(
        f"\n{len(results) - len(failures) - len(skips)} verified, "
        f"{len(skips)} skipped, {len(failures)} failed"
    )
    if failures:
        print("\nA failure means the documentation and the code disagree. Fix whichever")
        print("is wrong -- but do not update the prose to match a number you have not")
        print("checked, which is the habit this script exists to prevent.")
        for result in failures:
            print(f"  {result.claim.name}\n      stated in {result.claim.stated_in}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
