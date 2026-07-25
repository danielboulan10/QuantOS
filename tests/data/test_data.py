"""Data layer: CSV loading, alignment, catalogue, and the FRED client.

Network tests are marked and skipped by default, so the suite runs offline and
in CI without depending on a third party's availability. Everything that can be
tested without a network is tested without one.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.data.catalog import BUNDLES, CATALOG, Kind, resolve
from quantos.data.fred import FredClient, FredError, FredSeries
from quantos.data.loader import align, load_ohlcv_csv, to_returns


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_prefers_adjusted_close_over_raw_close(tmp_path) -> None:
    """The single most common data error in retail backtests.

    Using the raw close for a dividend payer injects a fake negative return on
    every ex-dividend date, biasing volatility and drawdown downward.
    """
    path = write(
        tmp_path,
        "x.csv",
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,10,11,9,10.9,10.5,1000\n"
        "2024-01-03,11,12,10,11.9,11.5,1200\n",
    )
    series = load_ohlcv_csv(path, symbol="TEST")
    assert series.price_column == "adj close"
    assert series.prices.tolist() == [10.5, 11.5]
    assert series.detail["dividend_adjusted"] == "True"


def test_falls_back_to_close_and_says_so(tmp_path) -> None:
    path = write(tmp_path, "x.csv", "Date,Close\n2024-01-02,10\n2024-01-03,11\n")
    series = load_ohlcv_csv(path)
    assert series.price_column == "close"
    assert series.detail["dividend_adjusted"] == "False"


@pytest.mark.parametrize(
    "date_text,layout",
    [
        ("2024-01-02", "iso"),
        ("02/01/2024", "dmy"),
        ("20240102", "compact"),
    ],
)
def test_accepts_common_date_formats(tmp_path, date_text: str, layout: str) -> None:
    path = write(tmp_path, f"{layout}.csv", f"Date,Close\n{date_text},10\n")
    assert len(load_ohlcv_csv(path)) == 1


def test_sorts_by_date_regardless_of_file_order(tmp_path) -> None:
    """Stooq ships newest-first; Yahoo ships oldest-first. Both must work."""
    path = write(tmp_path, "x.csv", "Date,Close\n2024-01-05,12\n2024-01-02,10\n2024-01-03,11\n")
    series = load_ohlcv_csv(path)
    assert series.prices.tolist() == [10.0, 11.0, 12.0]
    assert np.all(np.diff(series.dates.astype("datetime64[D]").astype(int)) > 0)


def test_skips_unparseable_rows_rather_than_failing(tmp_path) -> None:
    path = write(
        tmp_path,
        "x.csv",
        "Date,Close\n2024-01-02,10\nnot-a-date,99\n2024-01-03,null\n2024-01-04,12\n",
    )
    series = load_ohlcv_csv(path)
    assert series.prices.tolist() == [10.0, 12.0]
    assert int(series.detail["rows_skipped"]) == 2


def test_handles_thousands_separators(tmp_path) -> None:
    path = write(tmp_path, "x.csv", 'Date,Close\n2024-01-02,"1,234.50"\n')
    assert load_ohlcv_csv(path).prices.tolist() == [1234.5]


def test_reports_a_useful_error_for_an_unrecognisable_file(tmp_path) -> None:
    path = write(tmp_path, "x.csv", "Foo,Bar\n1,2\n")
    with pytest.raises(ValueError, match=r"no date column|no recognisable price"):
        load_ohlcv_csv(path)


def test_missing_file_says_so(tmp_path) -> None:
    with pytest.raises(ValueError, match="no such file"):
        load_ohlcv_csv(tmp_path / "absent.csv")


def test_since_filters_by_date(tmp_path) -> None:
    path = write(tmp_path, "x.csv", "Date,Close\n2023-01-02,1\n2024-01-02,2\n2025-01-02,3\n")
    assert len(load_ohlcv_csv(path).since("2024-01-01")) == 2


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_align_takes_the_date_intersection() -> None:
    """Mis-paired dates make a correlation meaningless; this prevents it."""
    d1 = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    d2 = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    dates, out = align({"a": (d1, np.array([1.0, 2.0, 3.0])), "b": (d2, np.array([9.0, 8.0, 7.0]))})
    assert len(dates) == 2
    assert out["a"].tolist() == [2.0, 3.0]
    assert out["b"].tolist() == [9.0, 8.0]


def test_align_with_no_overlap_returns_empty_rather_than_guessing() -> None:
    """A daily index and a quarterly series genuinely share few dates."""
    d1 = np.array(["2024-01-01"], dtype="datetime64[D]")
    d2 = np.array(["2025-06-01"], dtype="datetime64[D]")
    dates, _ = align({"a": (d1, np.array([1.0])), "b": (d2, np.array([2.0]))})
    assert len(dates) == 0


def test_align_of_a_single_series_is_the_identity() -> None:
    d = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    _, out = align({"a": (d, np.array([1.0, 2.0]))})
    assert out["a"].tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def test_log_and_simple_returns_differ_where_it_matters() -> None:
    prices = np.array([100.0, 150.0])
    assert to_returns(prices, method="simple")[0] == pytest.approx(0.5)
    assert to_returns(prices, method="log")[0] == pytest.approx(np.log(1.5))


def test_log_returns_reject_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        to_returns(np.array([1.0, 0.0, 2.0]), method="log")


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
def test_every_catalogued_series_is_well_formed() -> None:
    for key, series in CATALOG.items():
        assert series.key == key
        assert series.fred_id and series.fred_id.upper() == series.fred_id
        assert series.name and series.reads_as
        assert isinstance(series.kind, Kind)


def test_every_bundle_references_real_series() -> None:
    for name, (description, members) in BUNDLES.items():
        assert description
        for member in members:
            assert member in CATALOG, f"bundle {name} references unknown {member}"


def test_resolve_by_key_and_by_fred_id() -> None:
    assert resolve("spx").fred_id == "SP500"
    assert resolve("SP500").key == "spx"
    assert resolve("VIXCLS").key == "vix"


def test_resolve_passes_through_uncatalogued_fred_ids() -> None:
    """Any FRED series should be usable without editing the catalogue."""
    series = resolve("GDPC1MEASURE")
    assert series.fred_id == "GDPC1MEASURE"
    assert series.kind is Kind.LEVEL


def test_rate_series_use_differences_and_levels_use_log_returns() -> None:
    """The distinction that stops 'VIX annualised volatility: 128%'."""
    values = np.array([100.0, 110.0, 121.0])
    level = resolve("spx").transform(values)
    rate = resolve("ust10y").transform(values)
    assert level == pytest.approx([np.log(1.1), np.log(1.1)])
    assert rate.tolist() == [10.0, 11.0]


def test_level_transform_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        resolve("spx").transform(np.array([1.0, -1.0]))


def test_rate_transform_accepts_negative_values() -> None:
    """Yields and spreads legitimately go negative; that must not raise."""
    assert resolve("curve10y2y").transform(np.array([0.5, -0.2])).tolist() == [-0.7]


# --------------------------------------------------------------------------- #
# FRED client (parsing offline, network marked)
# --------------------------------------------------------------------------- #
def test_parses_fred_csv_and_drops_missing_observations() -> None:
    """FRED encodes missing values as '.'; they are dropped, not interpolated."""
    payload = (
        "observation_date,DGS10\n"
        "2024-01-02,4.06\n"
        "2024-01-03,.\n"  # a market holiday
        "2024-01-04,4.01\n"
    )
    series = FredClient._parse("DGS10", payload, from_cache=False, age=0.0)
    assert len(series) == 2
    assert series.values.tolist() == [4.06, 4.01]
    assert series.detail["missing_observations"] == "1"


def test_parse_rejects_html_with_a_useful_message() -> None:
    with pytest.raises(FredError, match="no usable observations"):
        FredClient._parse("BOGUS", "observation_date,BOGUS\n", from_cache=False, age=0.0)


def test_series_since_and_transforms() -> None:
    dates = np.array(["2023-01-01", "2024-01-01", "2025-01-01"], dtype="datetime64[D]")
    series = FredSeries("X", dates, np.array([100.0, 110.0, 121.0]))
    assert len(series.since("2024-01-01")) == 2
    assert series.log_returns() == pytest.approx([np.log(1.1), np.log(1.1)])
    assert series.differences().tolist() == [10.0, 11.0]
    assert series.latest == 121.0


def test_log_returns_refuse_a_rate_series() -> None:
    """Yields go negative; the error must say what to use instead."""
    dates = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    series = FredSeries("T10Y2Y", dates, np.array([0.5, -0.2]))
    with pytest.raises(ValueError, match="differences"):
        series.log_returns()


def test_offline_client_without_cache_fails_clearly(tmp_path) -> None:
    client = FredClient(cache_dir=tmp_path, offline=True)
    with pytest.raises(FredError, match="offline and no cached copy"):
        client.get("SP500")


def test_cached_series_is_used_without_network(tmp_path) -> None:
    """A cached run must be reproducible and require no network."""
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "TEST.csv").write_text(
        "observation_date,TEST\n2024-01-02,1.0\n2024-01-03,2.0\n", encoding="utf-8"
    )
    client = FredClient(cache_dir=tmp_path, offline=True)
    series = client.get("TEST")
    assert series.values.tolist() == [1.0, 2.0]
    assert series.detail["from_cache"] == "True"


# --------------------------------------------------------------------------- #
# Live network (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_live_fred_fetch(tmp_path) -> None:
    """Opt-in: run with `pytest -m network`."""
    client = FredClient(cache_dir=tmp_path)
    series = client.get("DGS10")
    assert len(series) > 10_000
    assert series.detail["from_cache"] == "False"
    assert -5.0 < series.latest < 25.0
