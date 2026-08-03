#!/usr/bin/env python3
"""Check that every local file the documentation links to actually exists.

A README is the first thing anyone reads and a dead link in it is a small,
avoidable, entirely visible failure. GitHub renders a broken relative link as
ordinary blue text, so nothing about the page looks wrong until someone clicks.

Only local paths are checked. External URLs are deliberately not fetched: a
network call would make CI fail for reasons that have nothing to do with this
repository, which is exactly the kind of flaky gate that gets disabled and then
protects nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Markdown files whose links are checked.
DOCUMENTS = ["README.md", "ROADMAP.md", "CHANGELOG.md", "CONTRIBUTING.md"]

LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

#: Anchors and protocols that are not local files.
SKIP = ("http://", "https://", "mailto:", "#")


def check(document: Path) -> list[str]:
    text = document.read_text(encoding="utf-8")
    broken: list[str] = []

    for target in LINK.findall(text):
        if target.startswith(SKIP):
            continue
        # Strip an anchor: docs/X.md#section points at the file.
        path = target.split("#", 1)[0].strip()
        if not path:
            continue
        resolved = (document.parent / path).resolve()
        if not resolved.exists():
            broken.append(f"{document.relative_to(REPO)} -> {target}")

    return broken


def main() -> int:
    broken: list[str] = []
    checked = 0

    for name in DOCUMENTS:
        document = REPO / name
        if not document.exists():
            print(f"  missing document: {name}")
            broken.append(name)
            continue
        found = check(document)
        checked += 1
        mark = "FAIL" if found else "ok  "
        print(f"  {mark}  {name}")
        broken.extend(found)

    for document in sorted((REPO / "docs").rglob("*.md")):
        found = check(document)
        checked += 1
        if found:
            print(f"  FAIL  {document.relative_to(REPO)}")
        broken.extend(found)

    print(f"\n{checked} documents checked, {len(broken)} broken local links")
    if broken:
        print("\nBroken links render as ordinary text on GitHub, so nothing looks")
        print("wrong until a reader clicks. Fix the path or remove the link:")
        for entry in broken:
            print(f"  {entry}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
