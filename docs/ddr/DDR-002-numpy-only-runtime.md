# DDR-002: NumPy is the only runtime dependency

- **Status:** Accepted
- **Date:** 2026-07
- **Supersedes:** none
- **Affects:** `quantos.core.special`, `quantos.core.stats`, `quantos.core.optimize`,
  `quantos.viz`, `quantos.cli`, `pyproject.toml`, CI

## Context

QuantOS needs the error function, the incomplete gamma and beta functions, a
dozen statistical tests, several optimisers, root finders, quadrature, splines,
low-discrepancy sequences, and charts. The obvious way to obtain all of that is
`scipy`, `statsmodels`, `pandas`, `matplotlib`, and `click`. That is what most
comparable projects do, and it is not an unreasonable choice.

We chose differently, and the reasoning is worth recording because the decision
is expensive and easy to reverse by accident.

## Decision

The runtime dependency set is exactly `numpy>=1.24`.

Everything else is implemented inside `quantos.core`:

| Need | Where it lives | Would have been |
|---|---|---|
| `erf`, `erfc`, `ndtr`, `ndtri`, incomplete gamma/beta, `lgamma`, `digamma` | `core/special.py` | `scipy.special` |
| Distributions with pdf/cdf/ppf/sample | `core/distributions.py` | `scipy.stats` |
| ADF, KPSS, Ljung-Box, ARCH-LM, variance ratio, Jarque-Bera, KS | `core/stats/hypothesis.py` | `statsmodels` |
| OLS with HAC/White errors, GARCH MLE, cointegration | `core/timeseries/` | `statsmodels` |
| Brent, safeguarded Newton, Nelder-Mead, BFGS, projected gradient | `core/optimize/` | `scipy.optimize` |
| Gauss-Legendre, adaptive Simpson, cubic and monotone splines, Sobol, Halton | `core/numerics.py` | `scipy.integrate`, `scipy.interpolate`, `scipy.stats.qmc` |
| Charts | `viz/svg.py` | `matplotlib` |
| CLI | `cli/main.py` (argparse) | `click`, `typer` |

**SciPy is a test-only dependency.** It appears in
`[project.optional-dependencies].test` and nowhere else, where it serves as an
*independent oracle* for the implementations above.

## Consequences

### What this buys

1. **The numerics are auditable.** A reader who wants to know how a p-value was
   computed can read the code that computed it. With SciPy the answer is "a
   Fortran routine from 1987, probably correct" — which it is, but the reader
   learns nothing and cannot check.
2. **Correctness becomes demonstrable rather than assumed.** Because we do not
   share an implementation with the oracle, agreement to 1e-13 across 12 special
   functions and 11 distributions is real evidence. If we called SciPy, the tests
   would be tautologies.
3. **Install is trivial and universal.** No compiler, no BLAS variant, no wheel
   availability question on a new Python release. `pip install quantos` works the
   day 3.14 ships.
4. **The dependency surface cannot rot.** There is exactly one upstream project
   whose API changes can break us.

### What this costs

1. **We own the bugs.** Five real defects were found and fixed in this code
   during development, each documented at its site: catastrophic cancellation in
   `erfc` near its branch point, a Halley residual that cancelled for `p → 1` in
   `ndtri`, a Lanczos pole at tiny arguments in `lgamma`, an oscillating
   risk-parity iteration, and a duplicate-entry leak in the order book's price
   heap. SciPy would have had none of these.
2. **Less coverage than SciPy.** We implement what QuantOS needs, not the
   universe. There is no Bessel function here.
3. **Slower in places.** Pure-Python continued fractions lose to compiled code.
   Measured impact is small because the loops are vectorised over arrays.
4. **The temptation to `import scipy` is constant.** Hence the enforcement below.

### Enforcement

Good intentions do not survive contact with a deadline, so the rule is
mechanical. CI has a `runtime-dependency-audit` job that:

1. installs with `pip install -e .` (no `[test]`, so SciPy is absent),
2. asserts `import scipy` **fails**,
3. runs `quantos doctor`, importing every module,
4. runs three CLI subcommands end to end.

Any runtime module that grows a SciPy import fails this job. This is the only
part of the decision that actually holds it in place.

## Alternatives considered

**Depend on SciPy.** Rejected on the grounds above — principally that it turns
the test suite from evidence into tautology, and that the numerics are a
substantial part of what this repository is for.

**Vendor a subset of SciPy.** Rejected: inherits the licence and maintenance
burden without the auditability benefit, since vendored Fortran translations are
no more readable than the originals.

**Make SciPy an optional accelerator, used when present.** Seriously considered
and rejected: two code paths means two behaviours, and the one that runs in CI is
not necessarily the one that runs for a user. Numerical results that depend on
which optional packages are installed are worse than slightly slower results.

**Depend on pandas for data handling.** Rejected separately. Structured NumPy
arrays and explicit dataclasses cover what we need, and pandas' index alignment
semantics are a common source of silent look-ahead bias in backtests — a
misaligned join that quietly shifts a signal forward by one period is very hard
to see and very easy to write.

## What would change this decision

- A need for functions whose correct implementation is genuinely research-grade
  (multivariate special functions, sparse eigensolvers, stiff ODE integration).
- Evidence that the pure-Python numerics are a material bottleneck in realistic
  use — the benchmarks exist to detect this.
- A decision to add a compiled core, at which point linking against SciPy's
  underlying libraries becomes free.

## References

- `docs/ddr/DDR-004` — the Johansen critical-value incident, which is the
  clearest illustration of both the cost and the benefit of owning the numerics.
- `tests/core/test_special.py` — the oracle comparison, including the case where
  QuantOS is *more* accurate than SciPy in the far tail.
