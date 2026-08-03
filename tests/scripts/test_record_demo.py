"""The demo is a claim about the software, so it gets checked like one.

A README demo that no longer matches the tool is worse than no demo: it is a
confident, specific, wrong description. These tests do not check that the
animation is pretty. They check the properties that make it honest -- that it
came from real output, that nothing is silently dropped, and that a viewer who
sees only the final frame sees the whole transcript.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("record_demo", REPO / "scripts" / "record_demo.py")
assert _spec and _spec.loader
record_demo = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through sys.modules,
# and a module absent from it fails at class-creation time rather than at use.
sys.modules["record_demo"] = record_demo
_spec.loader.exec_module(record_demo)


@pytest.fixture
def transcript() -> list[record_demo.Recorded]:
    return [
        record_demo.Recorded("quantos doctor", "caption one", ["alpha", "beta"], 0.4),
        record_demo.Recorded("quantos plan", "caption two", ["gamma 12.5%"], 1.25),
    ]


def test_every_line_reaches_full_opacity_by_the_end_of_the_cycle():
    """A still render must show the complete transcript, not a half-drawn screen.

    GitHub renders the SVG, but so do previews and readers that ignore CSS
    animation entirely. Each line therefore animates to opacity 1 and stays
    there, so the last frame is the whole session.
    """
    svg = (REPO / "docs" / "demo.svg").read_text()
    frames = re.findall(r"@keyframes r\d+\{([^}]*\}[^}]*)\}", svg)
    assert frames, "no keyframes found"
    for frame in frames:
        assert frame.rstrip().endswith("100%{opacity:1"), frame


def test_the_number_of_animated_lines_matches_the_number_of_rendered_lines():
    """An orphaned class or a line with no animation would show at the wrong time."""
    svg = (REPO / "docs" / "demo.svg").read_text()
    rendered = set(re.findall(r'class="\w+ (r\d+)"', svg))
    animated = set(re.findall(r"\.(r\d+)\{animation", svg))
    keyframed = set(re.findall(r"@keyframes (r\d+)\{", svg))
    assert rendered == animated == keyframed


def test_the_committed_demo_still_shows_the_current_version():
    """`quantos doctor` prints the version, so a stale demo is detectable.

    This is the check that makes the demo self-invalidating: bump the version and
    forget to re-record, and this fails rather than the README quietly showing a
    release that no longer exists.
    """
    from quantos import __version__

    svg = (REPO / "docs" / "demo.svg").read_text()
    assert "quantos doctor" in svg
    assert __version__ in svg, "re-run scripts/record_demo.py after a version bump"


def test_truncation_is_announced_rather_than_silent():
    """Cutting output is fine. Cutting it without saying so is not.

    An earlier version of this test built the expected lists itself and then
    asserted things about them, which exercised no code at all. It now calls the
    function.
    """
    raw = [f"line {i}" for i in range(60)]

    tail = record_demo._slice(raw, 7)
    assert tail[0] == "... 53 lines omitted ..."
    assert tail[1:] == raw[-7:], "the tail keeps the conclusion, which is the point"

    head = record_demo._slice(raw, -7)
    assert head[:7] == raw[:7]
    assert head[-1] == "... 53 more lines ..."

    # Short output is passed through untouched, with no misleading marker.
    assert record_demo._slice(raw[:3], 7) == raw[:3]
    assert record_demo._slice(raw[:7], -7) == raw[:7]


def test_long_lines_are_cut_at_a_word_boundary_and_marked():
    """Slicing mid-word reads as corrupted output rather than deliberate trimming."""
    line = "the historical Sharpe ratio of 1.01 is statistically " + "distinguishable " * 6
    fitted = record_demo._fit(line)

    assert len(fitted) <= record_demo.COLUMNS
    assert fitted.endswith("…")
    assert not fitted[:-2].rstrip().endswith("disting"), "cut landed inside a word"
    # A short line is returned untouched, ellipsis and all.
    assert record_demo._fit("short") == "short"


def test_real_elapsed_time_is_shown_rather_than_the_animation_time(transcript):
    """The animation is faster than reality; the difference is stated, not hidden."""
    svg = record_demo.render(transcript)
    assert "(0.4s real)" in svg
    assert "(1.2s real)" in svg or "(1.3s real)" in svg


def test_markup_in_captured_output_cannot_break_the_svg():
    """Command output is untrusted text as far as the renderer is concerned."""
    hostile = record_demo.Recorded(
        "quantos research", "cap", ['</text><script>alert("x")</script>', "a < b & c > d"], 0.1
    )
    svg = record_demo.render([hostile])

    assert "<script>" not in svg
    assert "&lt;/text&gt;" in svg
    assert "a &lt; b &amp; c &gt; d" in svg


def test_lines_are_classified_by_shape_not_by_guessing_at_meaning():
    assert record_demo._classify("  ok    something passed") == "ok"
    assert record_demo._classify("FAIL  something broke") == "bad"
    assert record_demo._classify("  49.7% annualised") == "num"
    assert record_demo._classify("plain output") == "out"
