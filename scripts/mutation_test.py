#!/usr/bin/env python3
"""Inject faults into the numerical core and measure how many the tests catch.

The question coverage cannot answer
------------------------------------
This repository reports 78% line coverage. That number says lines were
*executed*. It says nothing about whether a bug in them would be *caught* --
a test that calls a function and asserts nothing gives full coverage and zero
protection.

Mutation testing answers the real question directly. Change ``<`` to ``<=``,
flip a sign, perturb a constant, and re-run the tests. If they still pass, the
suite could not tell a working function from a broken one, and every claim
resting on that function is weaker than it looks.

The score
---------
**Mutation score = mutants killed / mutants injected.** A survivor is not
automatically a bug -- some mutations are semantically equivalent, and some
touch code paths that genuinely do not matter. But each survivor is a place
where a real change would go unnoticed, and the list is worth reading rather
than the number alone.

Why this is written here rather than installed
-----------------------------------------------
``mutmut`` and ``cosmic-ray`` exist and are good. This is ~250 lines, adds no
dependency, and targets the mutations that matter for numerical code
specifically: comparison boundaries, sign errors, off-by-one in slice indices,
and perturbed constants. Those are the failure modes this codebase has actually
had -- an inverted sign in ``dual_delta``, a spurious ``* n`` in the variance
ratio, an off-by-one in ``_strategy_returns``.

Isolation
---------
Mutants are applied inside a **temporary copy of the repository**, never the
working tree. The first version mutated files in place and restored them
afterwards, which is fine until something goes wrong: an interrupted run leaves
the source broken, and any concurrent work sees a half-mutated repository. That
happened during development -- an unrelated script crashed inside
``LimitOrderBook.match`` on a ``None`` maker, which looked exactly like a real
order-book bug and was in fact this tool holding ``and`` mutated to ``or``.

Copying costs a second per module and removes the entire failure class.

Usage
-----
    python scripts/mutation_test.py                     # the numerical core
    python scripts/mutation_test.py --module special    # one module
    python scripts/mutation_test.py --limit 20          # sample, for a quick look
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]

#: Module -> the tests that exercise it. Running the whole suite per mutant would
#: take days; running the relevant file takes seconds and is what decides whether
#: *those* tests catch the fault.
TARGETS: dict[str, tuple[str, str]] = {
    "special": ("src/quantos/core/special.py", "tests/core/test_special.py"),
    "black_scholes": (
        "src/quantos/derivatives/black_scholes.py",
        "tests/derivatives/test_black_scholes.py",
    ),
    "paths": ("src/quantos/forecast/paths.py", "tests/forecast/test_paths.py"),
    "probabilities": (
        "src/quantos/forecast/probabilities.py",
        "tests/forecast/test_probabilities.py",
    ),
    "calibration": (
        "src/quantos/forecast/calibration.py",
        "tests/forecast/test_calibration.py",
    ),
    "ledger": ("src/quantos/live/ledger.py", "tests/live/test_ledger.py"),
    "baselines": ("src/quantos/models/baselines.py", "tests/models/test_baselines.py"),
    "intraday": ("src/quantos/research/intraday.py", "tests/research/test_intraday.py"),
    "vol_surface": (
        "src/quantos/research/vol_surface.py",
        "tests/research/test_vol_surface.py",
    ),
    "book": ("src/quantos/exchange/book.py", "tests/exchange"),
    "heston": ("src/quantos/derivatives/heston.py", "tests/derivatives/test_heston.py"),
    "american": ("src/quantos/derivatives/american.py", "tests/derivatives/test_american.py"),
    "var_backtest": ("src/quantos/risk/var_backtest.py", "tests/risk/test_var_backtest.py"),
    "market_making": (
        "src/quantos/derivatives/market_making.py",
        "tests/derivatives/test_market_making.py",
    ),
}


@dataclass
class Mutant:
    """One injected fault."""

    module: str
    line: int
    original: str
    mutated: str
    kind: str
    source: str = field(repr=False, default="")


@dataclass
class Outcome:
    mutant: Mutant
    killed: bool
    seconds: float
    note: str = ""


class Mutator(ast.NodeTransformer):
    """Apply exactly one mutation, chosen by index, and record what it did."""

    COMPARISONS: ClassVar[dict[type, type]] = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }
    ARITHMETIC: ClassVar[dict[type, type]] = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
    }

    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.seen = 0
        self.applied: tuple[str, str, str, int] | None = None

    def _should_apply(self) -> bool:
        hit = self.seen == self.target_index
        self.seen += 1
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        mutable = len(node.ops) == 1 and type(node.ops[0]) in self.COMPARISONS
        if mutable and self._should_apply():
            before = type(node.ops[0]).__name__
            after = self.COMPARISONS[type(node.ops[0])]
            node.ops[0] = after()
            self.applied = ("comparison", before, after.__name__, node.lineno)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if type(node.op) in self.ARITHMETIC and self._should_apply():
            before = type(node.op).__name__
            after = self.ARITHMETIC[type(node.op)]
            node.op = after()
            self.applied = ("arithmetic", before, after.__name__, node.lineno)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # Numeric constants only, and not the booleans that `bool` subclasses int.
        numeric = isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        if numeric and self._should_apply():
            before = repr(node.value)
            # Perturb rather than zero: a tolerance changed from 1e-9 to 0 often
            # raises, which is a crash rather than a silent wrong answer.
            after_value = node.value * 2 if node.value not in (0, 1) else node.value + 1
            node.value = after_value
            self.applied = ("constant", before, repr(after_value), node.lineno)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.USub) and self._should_apply():
            self.applied = ("sign", "-x", "x", node.lineno)
            return node.operand
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if self._should_apply():
            before = type(node.op).__name__
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.applied = ("boolean", before, type(node.op).__name__, node.lineno)
        return node


def count_sites(tree: ast.AST) -> int:
    counter = Mutator(-1)
    counter.visit(ast.parse(ast.unparse(tree)))
    return counter.seen


def make_mutant(source: str, index: int, module: str) -> Mutant | None:
    tree = ast.parse(source)
    mutator = Mutator(index)
    mutated_tree = mutator.visit(tree)
    if mutator.applied is None:
        return None
    kind, before, after, line = mutator.applied
    ast.fix_missing_locations(mutated_tree)
    return Mutant(
        module=module,
        line=line,
        original=before,
        mutated=after,
        kind=kind,
        source=ast.unparse(mutated_tree),
    )


def run_tests(test_path: str, timeout: float, *, cwd: Path) -> tuple[bool, str]:
    """Run the tests and report whether they passed.

    A timeout counts as a kill: an infinite loop is a detected fault, not a
    mutant that slipped past the suite.
    """
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                test_path,
                "-x",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(cwd / "src")},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return completed.returncode == 0, ""


def test_module(name: str, limit: int | None, timeout: float, seed: int) -> list[Outcome]:
    """Mutate one module inside a throwaway copy of the repository."""
    source_path, test_path = TARGETS[name]
    original = (ROOT / source_path).read_text()

    total_sites = count_sites(ast.parse(original))
    indices = list(range(total_sites))
    random.Random(seed).shuffle(indices)
    if limit:
        indices = indices[:limit]

    print(f"\n{name}: {total_sites} mutation sites, testing {len(indices)}")

    outcomes: list[Outcome] = []
    with tempfile.TemporaryDirectory(prefix="quantos-mutation-") as workspace:
        sandbox = Path(workspace) / "repo"
        shutil.copytree(
            ROOT,
            sandbox,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc", "site", "forward", ".pytest_cache"
            ),
        )
        target = sandbox / source_path

        for position, index in enumerate(indices, 1):
            mutant = make_mutant(original, index, name)
            if mutant is None:
                continue
            target.write_text(mutant.source)
            started = time.perf_counter()
            passed, note = run_tests(test_path, timeout, cwd=sandbox)
            elapsed = time.perf_counter() - started
            # The mutant is KILLED when the tests FAIL.
            outcomes.append(Outcome(mutant, killed=not passed, seconds=elapsed, note=note))
            mark = "killed  " if not passed else "SURVIVED"
            print(
                f"  [{position:3d}/{len(indices)}] {mark} line {mutant.line:4d} "
                f"{mutant.kind:10s} {mutant.original} -> {mutant.mutated}  ({elapsed:.1f}s)"
            )

    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default=None, choices=sorted(TARGETS))
    parser.add_argument("--limit", type=int, default=25, help="mutants per module")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20240719)
    args = parser.parse_args()

    modules = [args.module] if args.module else sorted(TARGETS)
    all_outcomes: dict[str, list[Outcome]] = {}

    started = time.perf_counter()
    for name in modules:
        all_outcomes[name] = test_module(name, args.limit, args.timeout, args.seed)

    print("\n" + "=" * 74)
    print(f"{'module':16s} {'tested':>7s} {'killed':>7s} {'survived':>9s} {'score':>7s}")
    print("-" * 74)
    total_killed = total_tested = 0
    for name, outcomes in all_outcomes.items():
        if not outcomes:
            continue
        killed = sum(o.killed for o in outcomes)
        total_killed += killed
        total_tested += len(outcomes)
        score = killed / len(outcomes)
        print(f"{name:16s} {len(outcomes):7d} {killed:7d} {len(outcomes) - killed:9d} {score:6.0%}")
    print("-" * 74)
    overall = total_killed / total_tested if total_tested else 0.0
    print(
        f"{'OVERALL':16s} {total_tested:7d} {total_killed:7d} "
        f"{total_tested - total_killed:9d} {overall:6.0%}"
    )
    print(f"\n{time.perf_counter() - started:.0f}s elapsed")

    survivors = [o for outcomes in all_outcomes.values() for o in outcomes if not o.killed]
    if survivors:
        print(f"\n{len(survivors)} survivors -- places a real change would go unnoticed:")
        for outcome in survivors[:20]:
            m = outcome.mutant
            print(f"  {m.module}:{m.line}  {m.kind}: {m.original} -> {m.mutated}")
        print(
            "\nNot every survivor is a bug: some mutations are semantically equivalent\n"
            "(a tolerance doubled from 1e-12 to 2e-12 changes nothing observable). Read\n"
            "them rather than chasing the number."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
