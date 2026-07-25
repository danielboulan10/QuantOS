"""Dependency-free SVG plotting.

Why not matplotlib
------------------
QuantOS commits to a NumPy-only runtime (``docs/ddr/DDR-002``). Charts are needed
in three places -- the research journal, the CLI, and the README -- and all three
want *text* output that a reader can open without installing anything and that
version-controls as a readable diff. SVG delivers that; a PNG does not.

The generated markup is deliberately plain: no CSS, no JavaScript, no external
fonts. It renders in any browser, embeds directly in Markdown and GitHub
comments, and stays legible when opened in an editor.

A matplotlib backend is available via ``quantos[viz]`` for anyone who wants
publication figures; this module is what runs by default.
"""

from __future__ import annotations

import html
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Figure", "Theme", "book_depth_chart", "histogram", "line_chart", "scatter"]


@dataclass(frozen=True)
class Theme:
    """Colours and geometry. Defaults are chosen for legibility on white or dark."""

    width: int = 760
    height: int = 420
    margin_left: int = 64
    margin_right: int = 20
    margin_top: int = 36
    margin_bottom: int = 48
    background: str = "none"
    axis: str = "#8892a0"
    text: str = "#3c4450"
    grid: str = "#e2e6ec"
    palette: tuple[str, ...] = (
        "#2b6cb0",
        "#c05621",
        "#2f855a",
        "#805ad5",
        "#b7791f",
        "#c53030",
        "#2c7a7b",
        "#4a5568",
    )
    font_family: str = "ui-monospace, SFMono-Regular, Menlo, monospace"
    font_size: int = 12

    @property
    def plot_width(self) -> int:
        return self.width - self.margin_left - self.margin_right

    @property
    def plot_height(self) -> int:
        return self.height - self.margin_top - self.margin_bottom


@dataclass
class Figure:
    """An SVG figure under construction.

    Coordinates are supplied in *data* space; :meth:`_x` and :meth:`_y` map them
    to pixels. The y-axis is inverted, as SVG measures downward.
    """

    title: str = ""
    x_label: str = ""
    y_label: str = ""
    theme: Theme = field(default_factory=Theme)
    _elements: list[str] = field(default_factory=list)
    _x_range: tuple[float, float] = (0.0, 1.0)
    _y_range: tuple[float, float] = (0.0, 1.0)
    _legend: list[tuple[str, str]] = field(default_factory=list)

    def set_ranges(
        self, x: ArrayLike, y: ArrayLike, *, pad: float = 0.05, y_from_zero: bool = False
    ) -> Figure:
        """Fit the axes to the data with a small margin."""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        xa = xa[np.isfinite(xa)]
        ya = ya[np.isfinite(ya)]
        if xa.size == 0 or ya.size == 0:
            return self

        x0, x1 = float(xa.min()), float(xa.max())
        y0, y1 = float(ya.min()), float(ya.max())
        if y_from_zero:
            y0 = min(0.0, y0)
        # Degenerate ranges would divide by zero in the scale maps.
        if x1 - x0 <= 0:
            x0, x1 = x0 - 0.5, x1 + 0.5
        if y1 - y0 <= 0:
            y0, y1 = y0 - 0.5, y1 + 0.5
        span_y = (y1 - y0) * pad
        self._x_range = (x0, x1)
        self._y_range = (y0 - span_y, y1 + span_y)
        return self

    def _x(self, value: float) -> float:
        x0, x1 = self._x_range
        return self.theme.margin_left + (value - x0) / (x1 - x0) * self.theme.plot_width

    def _y(self, value: float) -> float:
        y0, y1 = self._y_range
        return (
            self.theme.margin_top
            + self.theme.plot_height
            - (value - y0) / (y1 - y0) * self.theme.plot_height
        )

    # -- primitives -------------------------------------------------------- #
    def add_grid(self, n_x: int = 6, n_y: int = 5) -> Figure:
        """Grid lines with tick labels."""
        t = self.theme
        for value in np.linspace(*self._y_range, n_y):
            py = self._y(float(value))
            self._elements.append(
                f'<line x1="{t.margin_left:.1f}" y1="{py:.1f}" '
                f'x2="{t.margin_left + t.plot_width:.1f}" y2="{py:.1f}" '
                f'stroke="{t.grid}" stroke-width="1"/>'
            )
            self._elements.append(
                f'<text x="{t.margin_left - 8:.1f}" y="{py + 4:.1f}" '
                f'text-anchor="end" fill="{t.text}" font-family="{t.font_family}" '
                f'font-size="{t.font_size - 1}">{_format_tick(float(value))}</text>'
            )
        for value in np.linspace(*self._x_range, n_x):
            px = self._x(float(value))
            self._elements.append(
                f'<line x1="{px:.1f}" y1="{t.margin_top:.1f}" x2="{px:.1f}" '
                f'y2="{t.margin_top + t.plot_height:.1f}" stroke="{t.grid}" '
                f'stroke-width="1"/>'
            )
            self._elements.append(
                f'<text x="{px:.1f}" y="{t.margin_top + t.plot_height + 18:.1f}" '
                f'text-anchor="middle" fill="{t.text}" font-family="{t.font_family}" '
                f'font-size="{t.font_size - 1}">{_format_tick(float(value))}</text>'
            )
        return self

    def add_line(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        label: str = "",
        colour: str | None = None,
        width: float = 1.6,
        dashed: bool = False,
        max_points: int | None = 3000,
    ) -> Figure:
        """A polyline. NaNs break the line rather than being interpolated across.

        ``max_points`` decimates long series before rendering. This is not a
        shortcut: a 760-pixel-wide chart cannot display more than about 1,500
        distinct x positions, so emitting 20,000 coordinate pairs produces a
        546 KB file that looks identical to a 20 KB one. Decimation uses
        **min/max per bucket** rather than simple striding, which preserves the
        visual envelope -- a stride can step straight over a single-sample spike
        and silently erase the most interesting feature of a price series.
        """
        t = self.theme
        colour = colour or t.palette[len(self._legend) % len(t.palette)]
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if max_points is not None and xa.size > max_points:
            xa, ya = _decimate_min_max(xa, ya, max_points)

        segments: list[list[str]] = [[]]
        for xi, yi in zip(xa, ya, strict=False):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                if segments[-1]:
                    segments.append([])
                continue
            segments[-1].append(f"{self._x(float(xi)):.2f},{self._y(float(yi)):.2f}")

        dash = ' stroke-dasharray="5,4"' if dashed else ""
        for points in segments:
            if len(points) < 2:
                continue
            self._elements.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{colour}" stroke-width="{width}" '
                f'stroke-linejoin="round"{dash}/>'
            )
        if label:
            self._legend.append((label, colour))
        return self

    def add_bars(
        self,
        edges: ArrayLike,
        heights: ArrayLike,
        *,
        label: str = "",
        colour: str | None = None,
        opacity: float = 0.75,
    ) -> Figure:
        """Bars spanning consecutive ``edges`` (length ``len(heights) + 1``)."""
        t = self.theme
        colour = colour or t.palette[len(self._legend) % len(t.palette)]
        e = np.asarray(edges, dtype=float)
        h = np.asarray(heights, dtype=float)
        baseline = self._y(max(self._y_range[0], 0.0))
        for i, value in enumerate(h):
            if not np.isfinite(value):
                continue
            x0 = self._x(float(e[i]))
            x1 = self._x(float(e[i + 1]))
            top = self._y(float(value))
            self._elements.append(
                f'<rect x="{min(x0, x1):.2f}" y="{min(top, baseline):.2f}" '
                f'width="{max(abs(x1 - x0) - 0.6, 0.4):.2f}" '
                f'height="{abs(baseline - top):.2f}" fill="{colour}" '
                f'opacity="{opacity}"/>'
            )
        if label:
            self._legend.append((label, colour))
        return self

    def add_points(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        label: str = "",
        colour: str | None = None,
        radius: float = 2.2,
        opacity: float = 0.7,
    ) -> Figure:
        """Scatter markers."""
        t = self.theme
        colour = colour or t.palette[len(self._legend) % len(t.palette)]
        for xi, yi in zip(np.asarray(x, dtype=float), np.asarray(y, dtype=float), strict=False):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            self._elements.append(
                f'<circle cx="{self._x(float(xi)):.2f}" cy="{self._y(float(yi)):.2f}" '
                f'r="{radius}" fill="{colour}" opacity="{opacity}"/>'
            )
        if label:
            self._legend.append((label, colour))
        return self

    def add_horizontal_line(
        self, value: float, *, colour: str = "#a0aec0", label: str = ""
    ) -> Figure:
        """A reference line, e.g. zero or a threshold."""
        t = self.theme
        py = self._y(value)
        self._elements.append(
            f'<line x1="{t.margin_left:.1f}" y1="{py:.1f}" '
            f'x2="{t.margin_left + t.plot_width:.1f}" y2="{py:.1f}" '
            f'stroke="{colour}" stroke-width="1.2" stroke-dasharray="4,3"/>'
        )
        if label:
            self._elements.append(
                f'<text x="{t.margin_left + t.plot_width - 4:.1f}" y="{py - 5:.1f}" '
                f'text-anchor="end" fill="{colour}" font-family="{t.font_family}" '
                f'font-size="{t.font_size - 2}">{html.escape(label)}</text>'
            )
        return self

    def render(self) -> str:
        """Serialise to a standalone SVG document."""
        t = self.theme
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{t.width}" '
            f'height="{t.height}" viewBox="0 0 {t.width} {t.height}" '
            f'role="img" aria-label="{html.escape(self.title or "chart")}">',
        ]
        if t.background != "none":
            parts.append(f'<rect width="{t.width}" height="{t.height}" fill="{t.background}"/>')
        parts.extend(self._elements)

        # Axes drawn last so they sit above the grid and series.
        parts.append(
            f'<line x1="{t.margin_left}" y1="{t.margin_top}" x2="{t.margin_left}" '
            f'y2="{t.margin_top + t.plot_height}" stroke="{t.axis}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<line x1="{t.margin_left}" y1="{t.margin_top + t.plot_height}" '
            f'x2="{t.margin_left + t.plot_width}" y2="{t.margin_top + t.plot_height}" '
            f'stroke="{t.axis}" stroke-width="1.2"/>'
        )
        if self.title:
            parts.append(
                f'<text x="{t.margin_left}" y="{t.margin_top - 14}" fill="{t.text}" '
                f'font-family="{t.font_family}" font-size="{t.font_size + 2}" '
                f'font-weight="600">{html.escape(self.title)}</text>'
            )
        if self.x_label:
            parts.append(
                f'<text x="{t.margin_left + t.plot_width / 2:.0f}" '
                f'y="{t.height - 8}" text-anchor="middle" fill="{t.text}" '
                f'font-family="{t.font_family}" font-size="{t.font_size}">'
                f"{html.escape(self.x_label)}</text>"
            )
        if self.y_label:
            cy = t.margin_top + t.plot_height / 2
            parts.append(
                f'<text x="14" y="{cy:.0f}" text-anchor="middle" fill="{t.text}" '
                f'font-family="{t.font_family}" font-size="{t.font_size}" '
                f'transform="rotate(-90 14 {cy:.0f})">{html.escape(self.y_label)}</text>'
            )
        for i, (label, colour) in enumerate(self._legend):
            ly = t.margin_top + 6 + i * 17
            lx = t.margin_left + t.plot_width - 12
            parts.append(
                f'<line x1="{lx - 22:.0f}" y1="{ly:.0f}" x2="{lx - 6:.0f}" '
                f'y2="{ly:.0f}" stroke="{colour}" stroke-width="2.4"/>'
            )
            parts.append(
                f'<text x="{lx - 28:.0f}" y="{ly + 4:.0f}" text-anchor="end" '
                f'fill="{t.text}" font-family="{t.font_family}" '
                f'font-size="{t.font_size - 1}">{html.escape(label)}</text>'
            )
        parts.append("</svg>")
        return "\n".join(parts)

    def save(self, path: str) -> str:
        """Write the SVG and return the path."""
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
        return str(target)


def _decimate_min_max(
    x: NDArray[np.float64], y: NDArray[np.float64], max_points: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reduce a series to ~``max_points`` while preserving its envelope.

    The series is split into buckets and each contributes its minimum and its
    maximum, in the order they occurred. Extremes therefore survive, which is
    exactly what plain striding fails to guarantee.
    """
    n = x.size
    buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    keep: list[int] = [0]
    for start, stop in itertools.pairwise(edges):
        if stop <= start:
            continue
        segment = y[start:stop]
        finite = np.isfinite(segment)
        if not np.any(finite):
            keep.append(start)
            continue
        offsets = np.nonzero(finite)[0]
        lo = start + int(offsets[np.argmin(segment[offsets])])
        hi = start + int(offsets[np.argmax(segment[offsets])])
        keep.extend(sorted((lo, hi)))
    keep.append(n - 1)
    index = np.unique(np.asarray(keep, dtype=int))
    return x[index], y[index]


def _format_tick(value: float) -> str:
    """Compact axis label: avoids both scientific noise and long decimals."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e6 or magnitude < 1e-4:
        return f"{value:.1e}"
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}".rstrip("0")


# --------------------------------------------------------------------------- #
# Convenience constructors                                                    #
# --------------------------------------------------------------------------- #
def line_chart(
    series: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    theme: Theme | None = None,
    zero_line: bool = False,
) -> Figure:
    """One or more labelled line series on shared axes.

    Example
        >>> import numpy as np
        >>> x = np.linspace(0, 10, 100)
        >>> fig = line_chart({"sin": (x, np.sin(x))}, title="demo")
        >>> svg = fig.render()
        >>> svg.startswith("<svg") and svg.endswith("</svg>")
        True
    """
    figure = Figure(title=title, x_label=x_label, y_label=y_label, theme=theme or Theme())
    all_x = np.concatenate([np.asarray(x, dtype=float).ravel() for x, _ in series.values()])
    all_y = np.concatenate([np.asarray(y, dtype=float).ravel() for _, y in series.values()])
    figure.set_ranges(all_x, all_y)
    figure.add_grid()
    if zero_line and figure._y_range[0] < 0 < figure._y_range[1]:
        figure.add_horizontal_line(0.0)
    for label, (x, y) in series.items():
        figure.add_line(x, y, label=label if len(series) > 1 else "")
    return figure


def histogram(
    data: ArrayLike,
    *,
    bins: int = 50,
    title: str = "",
    x_label: str = "",
    density: bool = False,
    overlay: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Histogram, optionally with analytic density curves overlaid.

    The overlay is the reason this exists: comparing an empirical distribution
    against a fitted density is how every distributional claim in this repository
    is checked visually as well as numerically.
    """
    values = np.asarray(data, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite data to plot")
    # np.histogram's stubs overload on `density` being a literal, so the call
    # is split rather than passing a bool variable through.
    if density:
        heights, edges = np.histogram(values, bins=bins, density=True)
    else:
        counts, edges = np.histogram(values, bins=bins)
        heights = counts.astype(np.float64)

    figure = Figure(
        title=title,
        x_label=x_label,
        y_label="density" if density else "count",
        theme=theme or Theme(),
    )
    y_all = [heights]
    if overlay:
        y_all.extend(np.asarray(y, dtype=float).ravel() for _, y in overlay.values())
    figure.set_ranges(edges, np.concatenate(y_all), y_from_zero=True)
    figure.add_grid()
    figure.add_bars(edges, heights, label="observed" if overlay else "")
    if overlay:
        for label, (x, y) in overlay.items():
            figure.add_line(x, y, label=label, width=2.0)
    return figure


def scatter(
    x: ArrayLike,
    y: ArrayLike,
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    fit_line: bool = False,
    theme: Theme | None = None,
) -> Figure:
    """Scatter plot, optionally with an OLS fit line and its slope annotated."""
    figure = Figure(title=title, x_label=x_label, y_label=y_label, theme=theme or Theme())
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    figure.set_ranges(xa, ya)
    figure.add_grid()
    figure.add_points(xa, ya)
    if fit_line:
        mask = np.isfinite(xa) & np.isfinite(ya)
        if int(np.sum(mask)) >= 3:
            slope, intercept = np.polyfit(xa[mask], ya[mask], 1)
            grid = np.array(figure._x_range)
            figure.add_line(
                grid,
                slope * grid + intercept,
                label=f"OLS slope={slope:.4g}",
                colour=figure.theme.palette[1],
                width=2.0,
            )
    return figure


def book_depth_chart(
    bid_prices: ArrayLike,
    bid_sizes: ArrayLike,
    ask_prices: ArrayLike,
    ask_sizes: ArrayLike,
    *,
    title: str = "Order book depth",
    theme: Theme | None = None,
) -> Figure:
    """Cumulative depth on both sides of the book.

    Cumulative rather than per-level, because the cumulative curve is what
    determines the cost of a sweep -- it *is* the market-impact function of a
    marketable order, read off directly.
    """
    bp = np.asarray(bid_prices, dtype=float).ravel()
    bs = np.cumsum(np.asarray(bid_sizes, dtype=float).ravel())
    ap = np.asarray(ask_prices, dtype=float).ravel()
    asz = np.cumsum(np.asarray(ask_sizes, dtype=float).ravel())

    figure = Figure(
        title=title,
        x_label="price (ticks)",
        y_label="cumulative size",
        theme=theme or Theme(),
    )
    figure.set_ranges(np.concatenate([bp, ap]), np.concatenate([bs, asz]), y_from_zero=True)
    figure.add_grid()
    figure.add_line(bp, bs, label="bids", colour="#2f855a", width=2.2)
    figure.add_line(ap, asz, label="asks", colour="#c53030", width=2.2)
    return figure
