"""Load intraday bars and ticks, and split them into trading sessions.

Why this is not the daily loader with a different date format
-------------------------------------------------------------
:mod:`quantos.data.loader` reads one price per day, and every estimator built on
it treats the series as a single continuous sequence. Intraday data breaks that
assumption in a way that changes the arithmetic, not just the parsing:

**The overnight gap is not a return like the others.** Between one session's close
and the next session's open the market is shut, news accumulates, and the price
jumps. That gap is real, but it is not an intraday return, and including it in a
sum of squared five-second returns adds a single enormous term to an estimator
built on the assumption that no term dominates. On a typical name the overnight
move is comparable to the entire intraday range, so one gap can contribute more to
realised variance than the whole session it precedes.

So this loader **groups by calendar date** and every downstream estimator works
within a session. :meth:`IntradayBars.sessions` returns one array per day, and the
gaps between them are never differenced.

**Sessions are not the same length.** Holidays are shortened, data feeds drop
ticks, and a symbol may not trade in the first minutes. Anything averaging across
days therefore has to map each session onto a common axis rather than assume a
fixed count -- which is what :func:`quantos.research.intraday.intraday_seasonality`
does.

Timestamp formats
-----------------
ISO 8601 with or without the ``T``, with or without seconds or a timezone offset,
plus epoch seconds, milliseconds and nanoseconds. Epoch units are distinguished by
magnitude, which is unambiguous for any date this century.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["IntradayBars", "load_intraday_csv"]

_TIMESTAMP_COLUMNS = (
    "timestamp",
    "datetime",
    "date_time",
    "time",
    "date",
    "t",
    "bar_time",
    "window_start",
)
_PRICE_PREFERENCE = ("close", "price", "last", "mid", "trade_price", "vwap", "c")
_VOLUME_COLUMNS = ("volume", "vol", "size", "quantity", "v")

#: Epoch magnitude thresholds and the multiplier that takes each to nanoseconds.
#: Checked most-precise first. The magnitudes are unambiguous for any date this
#: century: seconds since 1970 is ten digits, milliseconds thirteen, and so on.
_EPOCH_BOUNDS: tuple[tuple[float, int], ...] = (
    (1e17, 1),  # nanoseconds
    (1e14, 1_000),  # microseconds
    (1e11, 1_000_000),  # milliseconds
    (0.0, 1_000_000_000),  # seconds
)


@dataclass
class IntradayBars:
    """Intraday observations for one symbol, grouped into sessions."""

    symbol: str
    timestamps: NDArray[np.datetime64]
    prices: NDArray[np.float64]
    volume: NDArray[np.float64] | None = None
    price_column: str = "close"
    source: str = "csv"
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.prices.size)

    @property
    def start(self) -> str:
        return str(self.timestamps[0]) if len(self) else ""

    @property
    def end(self) -> str:
        return str(self.timestamps[-1]) if len(self) else ""

    def session_dates(self) -> list[np.datetime64]:
        """Every distinct calendar date present, in order."""
        days = self.timestamps.astype("datetime64[D]")
        return list(np.unique(days))

    def sessions(self, *, min_observations: int = 2) -> list[NDArray[np.float64]]:
        """One price array per trading day.

        Sessions shorter than ``min_observations`` are dropped: a day with three
        ticks cannot support a variance estimate, and including it would put a
        wild outlier into any per-session average.
        """
        days = self.timestamps.astype("datetime64[D]")
        out: list[NDArray[np.float64]] = []
        for day in np.unique(days):
            prices = self.prices[days == day]
            if prices.size >= min_observations:
                out.append(prices)
        return out

    def session(self, day: str | np.datetime64) -> NDArray[np.float64]:
        """One named session's prices."""
        wanted = np.datetime64(str(day), "D")
        selected = self.prices[self.timestamps.astype("datetime64[D]") == wanted]
        return np.asarray(selected, dtype=float)

    def median_session_length(self) -> int:
        lengths = [s.size for s in self.sessions()]
        return int(np.median(lengths)) if lengths else 0

    def overnight_returns(self) -> NDArray[np.float64]:
        r"""Close-to-open log returns, which the session estimators exclude.

        Returned separately rather than discarded, because the overnight move is
        genuine risk -- it is simply not *intraday* variance, and a realised
        variance that silently includes it is measuring two different things at
        once. Anyone wanting total daily risk should add this back explicitly.
        """
        days = self.timestamps.astype("datetime64[D]")
        unique = np.unique(days)
        if unique.size < 2:
            return np.zeros(0)
        closes, opens = [], []
        for day in unique:
            prices = self.prices[days == day]
            if prices.size:
                closes.append(float(prices[-1]))
                opens.append(float(prices[0]))
        gaps = np.log(np.asarray(opens[1:]) / np.asarray(closes[:-1]))
        return gaps.astype(float)

    def summary(self) -> str:
        sessions = self.sessions()
        lines = [
            f"{self.symbol}: {len(self)} observations from '{self.price_column}'",
            f"  {self.start} .. {self.end}",
            f"  {len(sessions)} sessions, median {self.median_session_length()} observations",
        ]
        if sessions:
            lengths = [s.size for s in sessions]
            lines.append(f"  session length range {min(lengths)}..{max(lengths)}")
        gaps = self.overnight_returns()
        if gaps.size:
            lines.append(
                f"  overnight gaps: {gaps.size}, sd {np.std(gaps):.4%} "
                "(excluded from intraday estimators)"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def _parse_timestamp(text: str) -> np.datetime64 | None:
    """Parse one timestamp, accepting ISO variants and epoch numbers."""
    text = text.strip().strip('"')
    if not text:
        return None

    # Epoch numbers, distinguished by magnitude.
    if text.replace(".", "", 1).replace("-", "", 1).isdigit() and len(text.split(".")[0]) >= 9:
        try:
            value = float(text)
        except ValueError:
            return None
        for bound, to_nanoseconds in _EPOCH_BOUNDS:
            if abs(value) >= bound:
                try:
                    return np.datetime64(int(value * to_nanoseconds), "ns")
                except (ValueError, OverflowError):
                    return None

    cleaned = text.replace("/", "-")
    if " " in cleaned and "T" not in cleaned:
        cleaned = cleaned.replace(" ", "T", 1)
    # numpy rejects an explicit UTC designator in some versions; drop the offset.
    for marker in ("Z", "+"):
        if marker in cleaned[10:]:
            cleaned = cleaned[:10] + cleaned[10:].split(marker)[0]
            break

    try:
        return np.datetime64(cleaned)
    except ValueError:
        return None


def load_intraday_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
    price_column: str | None = None,
    max_rows: int | None = None,
) -> IntradayBars:
    r"""Read intraday bars or ticks from a CSV.

    Purpose
        Turn a vendor intraday file into session-grouped prices that the
        estimators in :mod:`quantos.research.intraday` can consume.
    Inputs
        ``path`` -- CSV with a timestamp column and a price column. Common vendor
        names are recognised; ``price_column`` overrides the choice.
    Outputs
        An :class:`IntradayBars`. Rows are sorted by timestamp, and duplicate
        timestamps are kept (two trades in the same second are two observations,
        not an error).
    Failure modes
        Missing timestamp or price columns raise :class:`ValueError` naming what
        was found. Unparseable rows are counted in ``notes`` rather than raised;
        a single malformed line should not discard a day of ticks.

    Example
        >>> import tempfile, pathlib
        >>> text = "timestamp,close\n"
        >>> for day, base in (("2024-06-03", 100.0), ("2024-06-04", 101.0)):
        ...     for i in range(120):
        ...         minute = 30 + i
        ...         text += (f"{day}T{9 + minute // 60:02d}:{minute % 60:02d}:00,"
        ...                  f"{base + i * 0.01:.4f}\n")
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "ticks.csv"
        >>> _ = p.write_text(text)
        >>> bars = load_intraday_csv(p, symbol="TEST")
        >>> len(bars), len(bars.sessions())
        (240, 2)
        >>> bars.overnight_returns().size          # one gap between two sessions
        1
    """
    path = Path(path)
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise ValueError(f"{path} is empty")

    header = [column.strip().lower().replace(" ", "_") for column in rows[0]]

    time_index = next((header.index(name) for name in _TIMESTAMP_COLUMNS if name in header), None)
    if time_index is None:
        raise ValueError(
            f"{path}: no timestamp column. Looked for {list(_TIMESTAMP_COLUMNS)}, found {rows[0]}."
        )

    if price_column:
        wanted = price_column.strip().lower().replace(" ", "_")
        if wanted not in header:
            raise ValueError(f"{path}: no column named {price_column!r}; found {rows[0]}")
        price_index, chosen = header.index(wanted), wanted
    else:
        found = next((name for name in _PRICE_PREFERENCE if name in header), None)
        if found is None:
            raise ValueError(
                f"{path}: no price column. Looked for {list(_PRICE_PREFERENCE)}, found {rows[0]}."
            )
        price_index, chosen = header.index(found), found

    volume_index = next((header.index(name) for name in _VOLUME_COLUMNS if name in header), None)

    timestamps: list[np.datetime64] = []
    prices: list[float] = []
    volumes: list[float] = []
    bad_timestamp = bad_price = 0

    body = rows[1 : (max_rows + 1) if max_rows else None]
    for row in body:
        if not row or len(row) <= max(time_index, price_index):
            continue
        stamp = _parse_timestamp(row[time_index])
        if stamp is None:
            bad_timestamp += 1
            continue
        try:
            price = float(row[price_index].strip().replace(",", "").replace("$", ""))
        except ValueError:
            bad_price += 1
            continue
        if not np.isfinite(price) or price <= 0:
            bad_price += 1
            continue

        timestamps.append(stamp)
        prices.append(price)
        if volume_index is not None and volume_index < len(row):
            try:
                volumes.append(float(row[volume_index] or "nan"))
            except ValueError:
                volumes.append(float("nan"))

    if not prices:
        raise ValueError(
            f"{path}: no usable rows. {bad_timestamp} unparseable timestamps, "
            f"{bad_price} unparseable prices. Header was {rows[0]}."
        )

    stamp_array = np.asarray(timestamps, dtype="datetime64[ns]")
    price_array = np.asarray(prices, dtype=float)
    order = np.argsort(stamp_array, kind="stable")

    notes: list[str] = []
    if bad_timestamp:
        notes.append(f"{bad_timestamp} rows had unparseable timestamps and were skipped")
    if bad_price:
        notes.append(f"{bad_price} rows had unparseable prices and were skipped")
    if not np.all(order == np.arange(order.size)):
        notes.append("rows were not in chronological order and have been sorted")

    volume_array: NDArray[np.float64] | None = None
    if volumes and len(volumes) == len(prices):
        volume_array = np.asarray(volumes, dtype=float)[order]

    return IntradayBars(
        symbol=symbol or path.stem.upper(),
        timestamps=stamp_array[order],
        prices=price_array[order],
        volume=volume_array,
        price_column=chosen,
        notes=notes,
    )
