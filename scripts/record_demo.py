#!/usr/bin/env python3
"""Record a real terminal session as an animated SVG.

There is no video here and no screen capture. This runs the commands, captures
what they actually printed, measures how long they actually took, and replays
that as an SVG animation -- so the demo cannot drift away from the software the
way a recorded video does. Re-run it after a change and the demo is current, or
it fails.

The SVG is written by hand for the same reason the charts are: DDR-002 fixes the
runtime dependency at NumPy, and a demo GIF is not a good enough reason to add a
recording toolchain. CSS keyframes drive the animation, which GitHub renders in a
README.

    python scripts/record_demo.py --out docs/demo.svg

Pass --fast to skip the commands and render from a cached transcript, which is
what CI does when it only needs to check the file still builds.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# What the demo shows
#
# Chosen to answer, in order: what is it, does it run, and is it honest? The
# third is the one that matters and it goes last, because a viewer who stops
# early should still have seen the tool work.
# --------------------------------------------------------------------------- #
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

#: (command, lines to keep, caption). A *negative* count keeps the head instead
#: of the tail -- right for a banner like `doctor`, wrong for a report whose
#: conclusion is the last thing printed.
Step = tuple[str, int, str]

SCRIPT: list[Step] = [
    ("quantos doctor", -7, "86 modules, one runtime dependency"),
    ("quantos research --ticker NVDA", 22, "any listed symbol, no API key"),
    ("quantos stress --ticker SPY", 12, "replayed through crises that happened"),
    ("quantos factors --n-factors 200", 12, "840 factors, corrected for the search"),
    ("python scripts/verify_claims.py", 11, "every documented number, re-derived"),
]


@dataclass
class Recorded:
    command: str
    caption: str
    lines: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _run(command: str, keep: int, caption: str, *, repo: Path) -> Recorded:
    """Run one command and keep the most informative slice of its output.

    Truncation is from the *end* for long reports, because the summary a reader
    needs is the last thing printed, not the banner.
    """
    started = time.perf_counter()
    argv = command.split()
    if argv[0] == "quantos":
        argv = [sys.executable, "-m", "quantos.cli.main", *argv[1:]]
    elif argv[0] == "python":
        argv = [sys.executable, *argv[1:]]

    # check=False on purpose: a command that exits non-zero is still worth
    # showing. A demo that silently drops the failing step is the kind of demo
    # this whole script exists not to be.
    process = subprocess.run(
        argv, cwd=repo, capture_output=True, text=True, timeout=600, check=False
    )
    elapsed = time.perf_counter() - started

    raw = [ANSI.sub("", line).rstrip() for line in process.stdout.splitlines()]
    return Recorded(
        command=command,
        caption=caption,
        lines=_slice([line for line in raw if line.strip()], keep),
        seconds=elapsed,
    )


def _slice(raw: list[str], keep: int) -> list[str]:
    """Keep `keep` lines, from the head if negative, and say what was dropped.

    Cutting output is fine; cutting it without saying so is not, because a
    reader cannot tell a short command from a truncated one.
    """
    want = abs(keep)
    if len(raw) <= want:
        return list(raw)
    if keep < 0:
        return [*raw[:want], f"... {len(raw) - want} more lines ..."]
    return [f"... {len(raw) - want} lines omitted ...", *raw[-want:]]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
COLUMNS = 96
CHAR_W = 8.4
LINE_H = 19.0
PAD = 22.0
CHROME = 38.0

#: Seconds of animation per output line. Not the real elapsed time -- a viewer
#: will not sit through an eleven-second GARCH fit -- but the real time is
#: printed beside each command so the difference is stated rather than implied.
LINE_DELAY = 0.09
COMMAND_PAUSE = 0.9


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _fit(line: str) -> str:
    """Trim to the terminal width at a word boundary, marking the cut.

    Slicing mid-word looked like corrupted output rather than a deliberate
    truncation, which is worse than losing the words.
    """
    if len(line) <= COLUMNS:
        return line
    cut = line[: COLUMNS - 2]
    space = cut.rfind(" ")
    if space > COLUMNS - 24:
        cut = cut[:space]
    return cut + " \u2026"


def _classify(line: str) -> str:
    """Colour a line by what it is, without parsing it as data."""
    stripped = line.strip()
    if stripped.startswith(("ok ", "OK", "PASS", "verified")):
        return "ok"
    if stripped.startswith(("FAIL", "ERROR", "WARN")):
        return "bad"
    if "%" in stripped and any(c.isdigit() for c in stripped[:24]):
        return "num"
    if stripped.startswith(("---", "===", "===")):
        return "dim"
    return "out"


def render(recordings: list[Recorded], *, title: str = "QuantOS") -> str:
    """Lay the transcript out as one scrolling terminal."""
    rows: list[tuple[str, str, float]] = []  # (css class, text, appear-at seconds)
    clock = 0.4

    for record in recordings:
        rows.append(("cmd", f"$ {record.command}", clock))
        clock += COMMAND_PAUSE
        for line in record.lines:
            rows.append((_classify(line), _fit(line), clock))
            clock += LINE_DELAY
        rows.append(("time", f"  ({record.seconds:.1f}s real){'':4}{record.caption}", clock))
        clock += COMMAND_PAUSE

    total = clock + 2.2
    width = PAD * 2 + COLUMNS * CHAR_W
    height = CHROME + PAD + len(rows) * LINE_H + PAD

    # Each line is hidden until its moment, then stays. Expressing that as a
    # per-line keyframe percentage keeps the whole animation on one timeline, so
    # a viewer scrubbing or a renderer that ignores delays still sees a coherent
    # end state rather than a half-drawn screen.
    keyframes: list[str] = []
    body: list[str] = []
    for index, (kind, text, at) in enumerate(rows):
        start = min(99.0, 100.0 * at / total)
        appear = min(99.5, start + 0.35)
        keyframes.append(
            f"@keyframes r{index}{{0%,{start:.2f}%{{opacity:0}}{appear:.2f}%,100%{{opacity:1}}}}"
        )
        y = CHROME + PAD + (index + 1) * LINE_H
        body.append(
            f'<text class="{kind} r{index}" x="{PAD:.0f}" y="{y:.0f}">{_escape(text)}</text>'
        )

    styles = "".join(
        f".r{index}{{animation:r{index} {total:.1f}s linear infinite}}"
        for index in range(len(rows))
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" \
height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" role="img" \
aria-label="Recorded QuantOS terminal session">
<title>QuantOS - recorded terminal session</title>
<style>
  text {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
          font-size: 13px; white-space: pre; }}
  .cmd  {{ fill:#7dd3fc; font-weight:600 }}
  .out  {{ fill:#d4d4d8 }}
  .num  {{ fill:#fcd34d }}
  .ok   {{ fill:#86efac }}
  .bad  {{ fill:#fca5a5 }}
  .dim  {{ fill:#71717a }}
  .time {{ fill:#a1a1aa; font-style:italic }}
  {styles}
  {"".join(keyframes)}
</style>
<rect width="100%" height="100%" rx="10" fill="#18181b"/>
<rect width="100%" height="{CHROME:.0f}" rx="10" fill="#27272a"/>
<rect y="{CHROME - 10:.0f}" width="100%" height="10" fill="#27272a"/>
<circle cx="22" cy="19" r="6" fill="#ef4444"/>
<circle cx="43" cy="19" r="6" fill="#f59e0b"/>
<circle cx="64" cy="19" r="6" fill="#22c55e"/>
<text class="dim" x="86" y="24" font-size="12">{_escape(title)} &#8212; recorded, not staged</text>
{chr(10).join(body)}
</svg>
"""


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/demo.svg"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("docs/.demo_transcript.json"),
        help="where the captured transcript is stored, so --fast can re-render it",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="re-render from the cached transcript instead of running the commands",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent

    if args.fast:
        if not args.cache.exists():
            print(f"no cached transcript at {args.cache}; run without --fast first")
            return 1
        data = json.loads(args.cache.read_text())
        recordings = [Recorded(**entry) for entry in data]
    else:
        if shutil.which(sys.executable) is None:  # pragma: no cover - defensive
            print("no interpreter")
            return 1
        recordings = []
        for command, keep, caption in SCRIPT:
            print(f"  running {command}")
            recordings.append(_run(command, keep, caption, repo=repo))
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            json.dumps([record.__dict__ for record in recordings], indent=1) + "\n"
        )

    svg = render(recordings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")

    lines = sum(len(record.lines) for record in recordings)
    real = sum(record.seconds for record in recordings)
    print(
        f"\nwrote {args.out} -- {len(recordings)} commands, {lines} lines, "
        f"{real:.1f}s of real runtime, {len(svg) / 1024:.0f} kB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
