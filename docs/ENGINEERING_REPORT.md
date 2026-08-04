# Engineering audit

**August 2026 · QuantOS v1.5.0 → v1.6.0**

An audit of the whole repository against production standards, the fixes that
followed, and the things deliberately left alone. Every number below was
measured, not estimated; the commands that produce them are given so the audit
can be repeated rather than believed.

---

## Summary

| | Before | After |
|---|---|---|
| Structured logging | **none** | package logger, `--log-level`, quiet by default |
| Broad `except` clauses that logged nothing | 3 | 0 |
| Tracebacks printed to stdout | 2 | 0 |
| Duplicated return-computation sites | 7 | 1 |
| Business logic inside CLI command bodies | ~90 lines | 0 |
| `data/analysis.py` coverage | **29.8%** | **91.6%** |
| Modules with no test file | 1 | 0 |
| One-command setup | no | `make setup` |
| Container image | none | multi-stage, non-root, verified in CI |
| Issue / PR templates | none | 3 issue forms + PR checklist |
| `.editorconfig`, `.dockerignore`, devcontainer | none | present |

Unchanged and deliberately so: 0 TODO/FIXME markers, 0 functions missing return
annotations, mypy `--strict` clean across 92 files.

---

## What was found

### 1. No logging anywhere — fixed

The package had no logging at all. Defensible while it was pure functions over
arrays; not defensible once it had network fetches, an on-disk cache, a
hash-chained ledger and a CLI. Three call sites caught a broad `Exception` and
continued, and **a reader of the output could not distinguish a clean run from a
degraded one.**

The concrete case: `data/market.py` serves a stale cached copy when the network
fetch fails. That is the right behaviour — a backtest should not die because a
public endpoint rate-limited — but it happened silently. A run against
three-week-old prices looked exactly like a run against today's.

**Fixed.** [`core/logging.py`](../src/quantos/core/logging.py) with two rules
stated in its docstring:

- *The library never configures logging.* It attaches a `NullHandler` and emits
  records. A library that calls `basicConfig` at import steals the root logger
  from whatever imported it. Nothing in `src/quantos` outside the CLI configures
  anything.
- *Numbers go in the record, not the message.* `logger.info("fetched %s bars", n)`
  is greppable, and the interpolation never runs when the level is disabled.

Stale-cache service is now a `WARNING`. Cache hits, fetch timings and skipped
factors are `DEBUG`. Logs go to **stderr**, so a report on stdout stays pipeable.
Default level is `warning` — a research tool that chatters at INFO buries the
report it was asked to produce.

```bash
quantos --log-level debug stress --ticker SPY
```

### 2. Business logic inside a CLI command — fixed

`cmd_scenario` contained the macro-factor registry, date alignment, and the
yield-versus-level differencing rule. Roughly ninety lines of it. Two
consequences: the only way to test it was to run the CLI, and the only way to
reuse it was to copy it.

Worse, one of those rules is a **100× error waiting to happen**. A yield quoted
at 4.25 means 4.25 percent, so its difference is a move in percentage points and
a 100bp shock is 1.0 of them. A price level differences relatively. Both appear
as "change in the factor", and mixing them produces numbers still small enough to
look reasonable.

**Fixed.** [`data/align.py`](../src/quantos/data/align.py) holds the registry,
`align_to_grid`, `coverage`, and `factor_changes`. `FactorKind` makes the
yield/level choice explicit **where the series is declared, once**, rather than
at every call site. A test asserts the two conventions differ by more than 20×
on the same input, which is the whole reason the enum exists.

The scenario command produces byte-identical output after the refactor.

### 3. Seven copies of the same three lines — fixed

`np.diff(prices) / prices[:-1]` appeared at seven call sites. Two of them
disagreed about whether the result was `n` or `n-1` long — an off-by-one waiting
for the first caller who zips it against dates.

**Fixed.** One `simple_returns(prices, *, pad=True)`, with the padding
convention documented and tested rather than re-decided per site.

### 4. A module with no test file — fixed

`data/analysis.py` sat at **29.8%** line coverage, reached only incidentally
through the CLI. No test file existed for it. This is the same shape as the
finding already published in [TEST_QUALITY.md](TEST_QUALITY.md), where mutation
testing found a module at 0%.

**Fixed.** 16 tests, coverage **29.8% → 91.6%**. They target the *judgements*
rather than the arithmetic, because the statistics are covered where they are
implemented but the decisions are not:

- a yield series must **not** be given a Sharpe ratio (the "return" of a yield is
  not a return, and printing one invites comparison against an equity Sharpe that
  means something entirely different)
- CVaR can never be smaller than the VaR it conditions on
- ADF and KPSS must be read **jointly** — they have opposite nulls, and "neither
  rejects" means *not enough data*, not *stationary*
- cointegration is tested on levels, never on returns, which is a category error
  that rejects every time

### 5. Tracebacks on stdout — fixed

`web/server.py` called `traceback.print_exc()` twice. In the request handler that
interleaved a Python traceback with the HTML being served. Both are now
`_LOG.exception(...)`, and the broad `except` clauses carry a comment saying why
they are broad — one failing report section must not take down a report that is
otherwise complete.

### 6. No path from clean checkout to working — fixed

There was no `Makefile`, no container, no devcontainer, no `.editorconfig`. The
README's install instructions worked, but "does the CI pipeline agree with my
machine" had no answer short of pushing.

**Fixed.**

```bash
make setup    # clean checkout to working environment
make check    # exactly what CI runs — green here means green there
make demo
```

Plus a multi-stage `Dockerfile` (non-root, pinned base, healthcheck) and a
devcontainer. **I could not build the image on this machine — Docker is not
installed — so rather than claim it works, CI builds it**, runs the suite inside
it, and asserts the runtime image contains nothing beyond NumPy. An unverified
Dockerfile is a claim that rots on the first dependency change.

### 7. Open-source scaffolding — added

Three issue forms, including a **methodology challenge** template. That one is
deliberate: several published results here are negative and CI fails if any of
them changes, so a correction is a real contribution and should have a front
door. The PR template asks how a change was *validated* against something
independent, and has a section for results that came out badly.

---

## What was deliberately not changed

**`cli/main.py` is 1,646 lines.** A monolith, and I considered splitting it into
`cli/commands/*.py`. Not done, and the reason is a tradeoff rather than laziness:
the file is a flat list of `cmd_*` functions and their parsers with no shared
state, so the cost of finding anything is low and a split would trade one large
navigable file for twenty small files plus a registry. Now that the *business*
logic has moved out — item 2 above — what remains is genuinely presentation. It
is on the roadmap to split when it next grows, which is the right trigger.

**Coverage was not driven to 95%.** This repository already published the finding
that coverage lies: mutation testing found a module with tests that verified
nothing. The measure optimised here is the mutation score, and the coverage that
was raised (`data/analysis.py`) was raised because the module had *no tests at
all*, not to move a number. Chasing 95% would mean writing assertions against
code paths that do not matter, which is the exact behaviour
[TEST_QUALITY.md](TEST_QUALITY.md) argues against.

**No dependency was added.** Not for logging (stdlib), not for the container, not
for the Makefile. DDR-002 holds.

---

## Reproducing this audit

```bash
grep -rn "TODO\|FIXME\|HACK\|XXX" src/ scripts/          # 0
grep -rn "except:" src/                                   # 0 bare
grep -rn "traceback.print_exc" src/                       # 0
find src -name "*.py" -exec wc -l {} + | sort -rn | head  # module sizes
make coverage                                             # per-module coverage
make mutation                                             # are the tests load-bearing
make check                                                # everything CI runs
```

The typing and dead-code checks are worth stating as negatives, since finding
nothing is the useful outcome: **0 TODO markers**, **0 bare excepts**, **0
functions missing return annotations**, **0 unreachable modules**, and mypy
`--strict` clean across 92 source files.
