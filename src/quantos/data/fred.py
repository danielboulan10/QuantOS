"""Real market and macroeconomic data from FRED, with no API key and no new dependency.

Why FRED, and why not the obvious alternatives
-----------------------------------------------
QuantOS is otherwise built on simulated data, deliberately: ground truth is what
lets an estimator be *validated* rather than merely applied (see
``docs/ers/ERS-001``). But an estimator that has only ever seen synthetic data is
an estimator nobody should trust, so the platform needs a path to real series.

The constraint is ``docs/ddr/DDR-002``: NumPy is the only runtime dependency.
That rules out ``yfinance``, ``pandas-datareader`` and friends. It leaves
anything reachable with :mod:`urllib.request` from the standard library.

Sources evaluated:

============  ==========================================================
Stooq         Blocked by a JavaScript bot-check; not usable from a script.
Yahoo         Requires a session cookie and crumb, and rate-limits harshly.
Alpha Vantage Needs an API key.
**FRED**      **Plain CSV over HTTPS, no key, no rate limit in practice.**
============  ==========================================================

FRED is the Federal Reserve Bank of St. Louis's economic database. It carries far
more than macro aggregates: equity indices (S&P 500, Nasdaq, Wilshire), implied
volatility (VIX), the whole Treasury curve, credit spreads, FX, and commodities —
all daily, all free, all keyless.

What FRED does not have is *individual equities and ETFs*. For those, use
:func:`quantos.data.loader.load_ohlcv_csv`, which reads any CSV you already have —
a Yahoo Finance download, a broker export, a vendor extract. That keeps the
dependency promise while letting you analyse anything you can obtain.

Caching
-------
Every response is cached on disk under ``~/.cache/quantos/fred``. Re-running an
analysis does not re-download, which matters both for politeness to a free public
service and for reproducibility: a cached run is deterministic.

Attribution
-----------
Data is provided by the Federal Reserve Bank of St. Louis. Series are subject to
their original sources' terms; FRED aggregates and redistributes them. This
module retrieves public CSV endpoints exactly as a browser would.
"""

from __future__ import annotations

import csv
import io
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["FredClient", "FredError", "FredSeries", "default_cache_dir"]

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_USER_AGENT = "quantos/1.0 (+https://github.com/danielboulan10/QuantOS)"


class FredError(RuntimeError):
    """Raised when a series cannot be retrieved or parsed."""


def default_cache_dir() -> Path:
    """Cache location, honouring ``QUANTOS_CACHE_DIR`` if set."""
    override = os.environ.get("QUANTOS_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "quantos"
    return base / "fred"


@dataclass(frozen=True)
class FredSeries:
    """One time series: aligned dates and values, missing observations removed.

    FRED encodes missing values as ``"."`` — holidays in a daily series, or
    periods before a series began. Those rows are dropped rather than
    interpolated, because inventing an observation is a decision the caller
    should make explicitly, not something a loader should do silently.
    """

    series_id: str
    dates: NDArray[np.datetime64]
    values: NDArray[np.float64]
    #: Where it came from, for the research journal's provenance record.
    source: str = "FRED"
    retrieved_at: str = ""
    detail: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.values.size)

    @property
    def start(self) -> str:
        return str(self.dates[0])[:10] if len(self) else ""

    @property
    def end(self) -> str:
        return str(self.dates[-1])[:10] if len(self) else ""

    @property
    def latest(self) -> float:
        return float(self.values[-1]) if len(self) else float("nan")

    def since(self, start: str) -> FredSeries:
        """Restrict to observations on or after ``start`` (``"YYYY-MM-DD"``)."""
        cutoff = np.datetime64(start)
        mask = self.dates >= cutoff
        return FredSeries(
            series_id=self.series_id,
            dates=self.dates[mask],
            values=self.values[mask],
            source=self.source,
            retrieved_at=self.retrieved_at,
            detail=dict(self.detail),
        )

    def log_returns(self) -> NDArray[np.float64]:
        r"""Log returns, :math:`\log(P_t/P_{t-1})`.

        Only meaningful for a *level* series that is strictly positive — an index
        or a price. Applying it to a series that can be zero or negative (a
        yield, a spread, a growth rate) raises rather than silently producing
        NaNs, because that mistake is easy to make and hard to see afterwards.
        """
        if len(self) < 2:
            return np.zeros(0)
        if np.any(self.values <= 0):
            raise ValueError(
                f"{self.series_id} contains non-positive values, so log returns are "
                "undefined. This series is probably a rate, spread or growth rate "
                "rather than a price level -- use `differences()` instead."
            )
        return np.diff(np.log(self.values))

    def differences(self) -> NDArray[np.float64]:
        """First differences. The right transform for yields and spreads."""
        return np.diff(self.values) if len(self) >= 2 else np.zeros(0)

    def __repr__(self) -> str:  # pragma: no cover - display
        return (
            f"FredSeries({self.series_id!r}, n={len(self)}, "
            f"{self.start}..{self.end}, latest={self.latest:g})"
        )


@dataclass
class FredClient:
    """Retrieves FRED series over plain HTTPS, with an on-disk cache.

    Example
        >>> client = FredClient()                       # doctest: +SKIP
        >>> spx = client.get("SP500").since("2020-01-01")   # doctest: +SKIP
        >>> spx.log_returns().std() * (252 ** 0.5)      # doctest: +SKIP
    """

    cache_dir: Path = field(default_factory=default_cache_dir)
    #: Seconds before a cached file is considered stale. One day by default.
    max_age_seconds: float = 86_400.0
    timeout: float = 30.0
    offline: bool = False

    def _cache_path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.csv"

    def _fetch(self, series_id: str) -> str:
        """Download the raw CSV. Raises :class:`FredError` on any failure."""
        url = _FRED_CSV.format(series_id=series_id)
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = str(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as error:
            raise FredError(
                f"FRED returned HTTP {error.code} for series {series_id!r}. "
                "Check the series ID at https://fred.stlouisfed.org"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise FredError(
                f"could not reach FRED for series {series_id!r}: {error}. "
                "If you are offline, previously-cached series still work."
            ) from error

        if payload.lstrip().startswith("<"):
            raise FredError(
                f"FRED returned HTML rather than CSV for {series_id!r}; the series "
                "ID is probably wrong."
            )
        return payload

    def get(self, series_id: str, *, refresh: bool = False) -> FredSeries:
        """Fetch a series, using the cache when it is fresh enough.

        Inputs
            ``series_id`` -- a FRED code such as ``"SP500"`` or ``"DGS10"``.
            ``refresh`` -- ignore the cache and re-download.
        Failure modes
            :class:`FredError` if the series cannot be retrieved *and* no cached
            copy exists. If the network fails but a stale cache is present, the
            stale copy is used and the staleness is recorded in ``detail``.
        """
        path = self._cache_path(series_id)
        cached_text: str | None = None
        cache_age = float("inf")

        if path.exists():
            cache_age = time.time() - path.stat().st_mtime
            cached_text = path.read_text(encoding="utf-8")

        use_cache = (
            cached_text is not None
            and not refresh
            and (self.offline or cache_age < self.max_age_seconds)
        )

        if use_cache:
            assert cached_text is not None
            return self._parse(series_id, cached_text, from_cache=True, age=cache_age)

        if self.offline:
            raise FredError(f"offline and no cached copy of {series_id!r} at {path}")

        try:
            payload = self._fetch(series_id)
        except FredError:
            if cached_text is not None:
                # Network is down but we have something. Better a stale number
                # that says it is stale than a hard failure mid-analysis.
                return self._parse(series_id, cached_text, from_cache=True, age=cache_age)
            raise

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return self._parse(series_id, payload, from_cache=False, age=0.0)

    def get_many(self, series_ids: list[str], *, refresh: bool = False) -> dict[str, FredSeries]:
        """Fetch several series. Failures are skipped with a recorded reason."""
        out: dict[str, FredSeries] = {}
        for series_id in series_ids:
            try:
                out[series_id] = self.get(series_id, refresh=refresh)
            except FredError:
                continue
        return out

    @staticmethod
    def _parse(series_id: str, payload: str, *, from_cache: bool, age: float) -> FredSeries:
        """Parse FRED's CSV into aligned date and value arrays."""
        reader = csv.reader(io.StringIO(payload))
        try:
            header = next(reader)
        except StopIteration as error:
            raise FredError(f"empty response for {series_id!r}") from error
        if len(header) < 2:
            raise FredError(f"unexpected CSV header for {series_id!r}: {header!r}")

        dates: list[np.datetime64] = []
        values: list[float] = []
        skipped = 0
        for row in reader:
            if len(row) < 2:
                continue
            raw = row[1].strip()
            # FRED uses "." for a missing observation.
            if raw in {".", "", "NA", "NaN"}:
                skipped += 1
                continue
            try:
                value = float(raw)
                stamp = np.datetime64(datetime.strptime(row[0].strip(), "%Y-%m-%d").date())
            except ValueError:
                skipped += 1
                continue
            dates.append(stamp)
            values.append(value)

        if not values:
            raise FredError(f"no usable observations parsed for {series_id!r}")

        return FredSeries(
            series_id=series_id,
            dates=np.array(dates, dtype="datetime64[D]"),
            values=np.asarray(values, dtype=np.float64),
            retrieved_at=date.today().isoformat(),
            detail={
                "column": header[1],
                "missing_observations": str(skipped),
                "from_cache": str(from_cache),
                "cache_age_seconds": f"{age:.0f}" if np.isfinite(age) else "n/a",
            },
        )
