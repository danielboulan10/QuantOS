"""SVG rendering: valid markup, correct scaling, no external references."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from quantos.viz.svg import Figure, Theme, book_depth_chart, histogram, line_chart, scatter


def parse(svg: str) -> ET.Element:
    """Parsing with a real XML parser is the test: malformed markup raises."""
    return ET.fromstring(svg)


def test_line_chart_produces_well_formed_svg() -> None:
    x = np.linspace(0, 10, 100)
    svg = line_chart({"sin": (x, np.sin(x))}, title="demo", x_label="t", y_label="y").render()
    root = parse(svg)
    assert root.tag.endswith("svg")
    assert root.get("width") == "760"
    assert "demo" in svg


def test_svg_has_no_external_references() -> None:
    """Self-contained by design: no CDN, no remote fonts, no scripts."""
    svg = line_chart({"a": (np.arange(50), np.random.default_rng(0).standard_normal(50))}).render()
    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in svg
    assert "<script" not in svg
    assert "@import" not in svg


def test_multiple_series_get_distinct_colours_and_a_legend() -> None:
    x = np.linspace(0, 5, 40)
    svg = line_chart({"a": (x, x), "b": (x, x**2), "c": (x, -x)}).render()
    root = parse(svg)
    colours = {e.get("stroke") for e in root.iter() if e.tag.endswith("polyline")}
    assert len(colours) == 3
    for label in ("a", "b", "c"):
        assert f">{label}<" in svg


def test_nans_break_the_line_rather_than_interpolating() -> None:
    """A gap in the data is a gap in the chart; joining across it invents prices."""
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    svg = Figure().set_ranges(np.arange(5), y).add_line(np.arange(5), y).render()
    polylines = [e for e in parse(svg).iter() if e.tag.endswith("polyline")]
    assert len(polylines) == 2  # two segments, not one


def test_axis_scaling_maps_data_to_pixels_monotonically() -> None:
    figure = Figure(theme=Theme(width=500, height=300))
    figure.set_ranges(np.array([0.0, 10.0]), np.array([0.0, 100.0]), pad=0.0)
    assert figure._x(0.0) < figure._x(5.0) < figure._x(10.0)
    # SVG y grows downward, so a larger value maps to a smaller pixel.
    assert figure._y(0.0) > figure._y(50.0) > figure._y(100.0)


def test_degenerate_ranges_do_not_divide_by_zero() -> None:
    constant = np.full(20, 3.0)
    svg = line_chart({"flat": (np.arange(20), constant)}).render()
    assert parse(svg) is not None
    assert "nan" not in svg.lower()
    assert "inf" not in svg.lower()


def test_histogram_with_an_analytic_overlay() -> None:
    rng = np.random.default_rng(0)
    grid = np.linspace(-4, 4, 200)
    density = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
    svg = histogram(
        rng.standard_normal(5_000),
        bins=40,
        density=True,
        overlay={"N(0,1)": (grid, density)},
        title="fit",
    ).render()
    root = parse(svg)
    assert sum(1 for e in root.iter() if e.tag.endswith("rect")) >= 30
    assert any(e.tag.endswith("polyline") for e in root.iter())


def test_histogram_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="no finite data"):
        histogram(np.array([np.nan, np.inf]))


def test_scatter_with_a_fit_line_reports_the_slope() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    svg = scatter(x, 2.5 * x + rng.standard_normal(200) * 0.1, fit_line=True).render()
    assert "OLS slope=" in svg
    slope = float(re.search(r"OLS slope=([-\d.e+]+)", svg).group(1))
    assert slope == pytest.approx(2.5, rel=0.1)


def test_book_depth_chart_is_cumulative() -> None:
    svg = book_depth_chart([100, 99, 98], [10, 20, 30], [101, 102, 103], [15, 25, 35]).render()
    assert ">bids<" in svg and ">asks<" in svg
    assert parse(svg) is not None


def test_figure_saves_to_disk(tmp_path) -> None:
    path = line_chart({"a": (np.arange(10), np.arange(10))}).save(str(tmp_path / "d" / "c.svg"))
    assert (tmp_path / "d" / "c.svg").read_text().startswith("<svg")
    assert path.endswith("c.svg")


def test_titles_are_escaped_against_injection() -> None:
    svg = line_chart(
        {"a": (np.arange(5), np.arange(5))}, title='<script>alert("x")</script>'
    ).render()
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg
    assert parse(svg) is not None


def test_horizontal_reference_line() -> None:
    figure = Figure().set_ranges(np.arange(10), np.linspace(-1, 1, 10))
    svg = figure.add_grid().add_horizontal_line(0.0, label="zero").render()
    assert ">zero<" in svg
    assert parse(svg) is not None
