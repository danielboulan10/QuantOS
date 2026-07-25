#!/usr/bin/env python3
"""Validate every GitHub Actions workflow parses as YAML.

Why this exists
---------------
A workflow with a YAML syntax error does not fail loudly -- GitHub creates the
run, creates *zero jobs*, and marks it failed. There is no log to read and no
job to inspect, so the cause is invisible from the Actions tab.

That happened here: an unquoted `run:` value containing `echo "text: more"` was
read as a nested mapping, because a colon-space inside an unquoted YAML scalar
starts one. Every job silently vanished.

This script catches that locally in under a second. It is deliberately
dependency-light: it uses PyYAML if available, and otherwise reports that it
could not check rather than pretending it passed.

Usage
-----
    python scripts/validate_workflows.py
"""

from __future__ import annotations

from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def main() -> int:
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed; cannot validate. `pip install pyyaml`")
        return 0

    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    if not files:
        print(f"no workflows found under {WORKFLOWS}")
        return 0

    failures = 0
    for path in files:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            print(f"FAIL  {path.name}\n      {error}")
            failures += 1
            continue

        jobs = (document or {}).get("jobs")
        if not jobs:
            print(f"FAIL  {path.name}: parsed, but defines no jobs")
            failures += 1
            continue

        print(f"ok    {path.name}: {len(jobs)} jobs")
        for name, job in jobs.items():
            steps = job.get("steps", [])
            if not steps:
                print(f"      warning: job {name!r} has no steps")
            for step in steps:
                if "uses" not in step and "run" not in step:
                    print(f"      warning: a step in {name!r} has neither `uses` nor `run`")

    if failures:
        print(f"\n{failures} workflow(s) invalid")
        return 1
    print(f"\nall {len(files)} workflow(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
