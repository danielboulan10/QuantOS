# How good are the tests?

786 tests and 78% line coverage are counts, not evidence. Coverage says lines were
*executed*; a test that calls a function and asserts nothing gives full coverage
and zero protection. The question that matters is whether a **bug** would be
caught.

[`scripts/mutation_test.py`](../scripts/mutation_test.py) answers it directly:
inject a deliberate fault — flip `<` to `<=`, change a sign, perturb a constant —
and re-run the tests. If they still pass, the suite cannot distinguish working
code from broken code.

## Results

14 mutants per module, sampled deterministically. **Higher is better.**

| Module | Score | Reading |
|---|---:|---|
| `exchange/book.py` | **93%** | property-based tests plus C++ equivalence |
| `core/special.py` | **93%** | the SciPy oracle catches nearly everything |
| `derivatives/black_scholes.py` | 86% | strong |
| `forecast/probabilities.py` | 57% | mediocre |
| `models/baselines.py` | 50% | **was 0%** — see below |
| `forecast/paths.py` | 50% | mediocre |
| `research/intraday.py` | 36% | weak |
| `research/vol_surface.py` | 36% | weak |
| `forecast/calibration.py` | 29% | weak |
| `live/ledger.py` | 21% | weak |

**Overall: about 50%.** That is a mediocre number and it is published because it
is the number.

## What it found

**`models/baselines.py` scored 0 out of 14.** Every mutant survived. The cause
was not subtle: the module had no test file at all. `pinball_loss`, `qlike`, and
all four baseline forecasters underpin every figure in
[the model leaderboard](MODEL_LEADERBOARD.md), and none of them was directly
tested — the most-quoted result in the repository rested on the least-tested code.

Writing [`tests/models/test_baselines.py`](../tests/models/test_baselines.py) —
31 tests — took the score to **50%**. Two of those tests failed on their first
run for reasons worth keeping:

- A "calm" series built with `np.full` has zero dispersion, so the trailing
  estimator returns `2e-19` and any ratio against it is meaningless.
- A 500-day standard deviation does *not* "barely notice" one observation when
  that observation is a hundred sigma: its squared contribution alone dwarfs the
  other five hundred, and the estimate jumps sixfold. The correct claim is that
  EWMA moves *several times further*, not that the trailing estimate stays put.

Both are now documented in the tests rather than quietly corrected.

## Why the weak modules are weak

The low scorers share a shape: much of their surface is **presentation** —
`summary()` methods, verdict strings, formatted tables. Mutating a constant
inside an f-string changes output that no test asserts on. That is a real gap,
but a less alarming one than a mutated *tolerance* surviving.

The high scorers share the opposite shape. `book.py` is tested against a second
implementation in C++ that must agree byte for byte, and `special.py` against
SciPy as an oracle. **An independent implementation is the strongest test there
is**, and the mutation scores say so quantitatively.

## An operational note

The first version of this tool mutated files **in place** and restored them
afterwards. That is fine until something goes wrong. During development an
unrelated script crashed inside `LimitOrderBook.match` on a `None` maker — which
looked exactly like a serious order-book bug, and was in fact this tool holding
`and` mutated to `or` in a concurrently-running background job.

Mutants are now applied inside a temporary copy of the repository. Copying costs
about a second per module and removes the entire failure class.

## Reproducing

```bash
python scripts/mutation_test.py                    # every module
python scripts/mutation_test.py --module ledger    # one
python scripts/mutation_test.py --limit 40         # more mutants, slower
```

## What would improve the score

Asserting on `summary()` output would raise it quickly and mean little. The
honest improvements are narrower: property-based tests for `paths.py` and
`vol_surface.py`, and assertions on the *numeric* fields the ledger and
calibration reports compute rather than the strings they print.
