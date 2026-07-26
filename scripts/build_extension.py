#!/usr/bin/env python3
"""Build the optional C++ order-book accelerator.

The extension is genuinely optional. If it is absent, ``quantos.exchange.book``
uses the pure Python implementation and every result is identical -- matching is
exact integer arithmetic, so there is nothing for a second implementation to
disagree about. See ``src/quantos/exchange/_book.cpp`` for why this is the one
place DDR-002's "no optional accelerators" rule does not apply.

Usage
-----
    python scripts/build_extension.py            # build in place
    python scripts/build_extension.py --check    # report status only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "quantos" / "exchange" / "_book.cpp"
OUTPUT_DIR = ROOT / "src" / "quantos" / "exchange"


def extension_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def is_built() -> Path | None:
    candidate = OUTPUT_DIR / f"_book{extension_suffix()}"
    return candidate if candidate.exists() else None


def build() -> int:
    include = sysconfig.get_paths()["include"]
    output = OUTPUT_DIR / f"_book{extension_suffix()}"

    if sys.platform == "win32":
        print("Windows build is not wired up; the pure Python path is used there.")
        return 0

    compiler = "clang++" if sys.platform == "darwin" else "g++"
    command = [compiler, "-std=c++17", "-O3", "-fPIC", "-shared"]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    command += [f"-I{include}", str(SOURCE), "-o", str(output)]

    print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stderr[:4000])
        print("\nbuild failed; the pure Python implementation still works")
        return 1
    print(f"built {output.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        built = is_built()
        print(f"extension: {'built at ' + built.name if built else 'not built'}")
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from quantos.exchange import book as book_module

            print(f"backend in use: {book_module.BACKEND}")
        except ImportError as error:
            print(f"could not import quantos: {error}")
        return 0
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
