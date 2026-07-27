# Contributing

Thanks for looking. This repository has an unusual standard for what counts as a
finished change, so it is worth stating up front.

## The standard

**A claim in this repository is expected to be checked.** Not asserted, not cited
from a textbook — checked, against a case where the answer is known, with the
result recorded at the site of the code.

That cuts both ways. Several things here are documented as *not working*: the
market simulation does not reliably discover its own fundamental (mean
correlation 0.29 across 16 seeds), the neural volatility model loses to GARCH,
and forecast probabilities for rare events carry no skill over the base rate.
Those results were published rather than buried, and a change that quietly
removes an inconvenient measurement will not be merged.

## What a good change looks like

1. **A test that would fail without it.** Ideally one that recovers a known
   answer — inject a parameter, then require the code to return it. Tests that
   only assert the code runs pass equally well on code that is wrong.
2. **Statistical tests get a size check.** If you add a hypothesis test, measure
   its false-positive rate under the null. This repository found a published
   critical-value table that over-rejected at 28% against a nominal 5%; the
   check is not a formality.
3. **A comment explaining *why*, at the point of the decision.** The branch
   points in `core/special.py` are chosen by cancellation analysis, and the
   reasoning lives beside them. "What" is readable from the code; "why" is not.
4. **Measurements, not adjectives.** "Faster" is not a claim. "16M ops/s, median
   of seven runs, range 8.1–16.5M" is.

## Before opening a pull request

```bash
pip install -e ".[dev]"
pytest                                   # 750 tests, including every docstring
ruff check src tests benchmarks scripts
ruff format --check src tests
mypy
python scripts/validate_workflows.py     # CI YAML has silently broken before
```

The optional C++ order book is built with `python scripts/build_extension.py`.
It is not required — the pure Python book is the specification, and if the two
ever disagree the Python one is right.

## Design decisions

Significant choices live in [`docs/ddr/`](docs/ddr) as Design Decision Records:
the choice, the alternatives, the trade-off accepted, and what would change the
decision. If your change contradicts one, that is fine — but write the DDR that
supersedes it rather than leaving the old one to mislead.

The one most likely to affect you is
[DDR-002](docs/ddr/DDR-002-numpy-only-runtime.md): **NumPy is the only runtime
dependency.** SciPy appears solely as a test-time oracle, enforced by a CI job
that installs without it. Adding a runtime dependency needs a very good argument.

## Reporting something wrong

Open an issue. A reproduction beats a description, and a numerical counterexample
beats both — if a function returns the wrong answer for particular inputs, those
inputs are the most useful thing you can send.
