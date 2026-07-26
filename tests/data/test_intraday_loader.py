"""Tests for the intraday loader.

The behaviour worth testing is not the parsing -- it is the session splitting.
An intraday estimator that differences across the overnight gap gets a wrong
answer with no error message, so the tests below pin the boundary handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.data.intraday import IntradayBars, load_intraday_csv


def write(tmp_path, text: str, name: str = "ticks.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def two_sessions(tmp_path, n: int = 100):
    lines = ["timestamp,close,volume"]
    for day, base in (("2024-06-03", 100.0), ("2024-06-04", 108.0)):
        for i in range(n):
            minute = 30 + i  # 09:30 onward, rolling into later hours correctly
            stamp = f"{day}T{9 + minute // 60:02d}:{minute % 60:02d}:00"
            lines.append(f"{stamp},{base + i * 0.01:.4f},{100 + i}")
    return write(tmp_path, "\n".join(lines))


# --------------------------------------------------------------------------- #
# Session splitting -- the reason this module exists
# --------------------------------------------------------------------------- #
def test_sessions_are_split_by_calendar_date(tmp_path):
    bars = load_intraday_csv(two_sessions(tmp_path), symbol="TEST")
    assert len(bars) == 200
    sessions = bars.sessions()
    assert len(sessions) == 2
    assert all(s.size == 100 for s in sessions)


def test_the_overnight_gap_is_never_differenced_inside_a_session(tmp_path):
    """The central guarantee.

    The file jumps 8% overnight. Summing squared returns across the whole series
    would capture that jump; summing within each session must not. The two must
    therefore differ by approximately the squared gap.
    """
    from quantos.research.intraday import realized_variance

    bars = load_intraday_csv(two_sessions(tmp_path), symbol="TEST")

    within = sum(realized_variance(s, annualise=False) for s in bars.sessions())
    across = realized_variance(bars.prices, annualise=False)

    gaps = bars.overnight_returns()
    assert gaps.size == 1
    assert across == pytest.approx(within + float(gaps[0] ** 2), rel=1e-9)
    # And the gap is the dominant term, which is why it must be excluded.
    assert float(gaps[0] ** 2) > 10 * within


def test_overnight_returns_are_close_to_open(tmp_path):
    bars = load_intraday_csv(two_sessions(tmp_path), symbol="TEST")
    sessions = bars.sessions()
    expected = np.log(sessions[1][0] / sessions[0][-1])
    assert float(bars.overnight_returns()[0]) == pytest.approx(expected, rel=1e-12)


def test_short_sessions_are_excluded_on_request(tmp_path):
    lines = ["timestamp,close"]
    for i in range(50):
        lines.append(f"2024-06-03T09:30:{i:02d},{100 + i * 0.01:.4f}")
    for i in range(3):  # a stub session, e.g. a half day with a broken feed
        lines.append(f"2024-06-04T09:3{i}:00,101.0")
    bars = load_intraday_csv(write(tmp_path, "\n".join(lines)))

    assert len(bars.sessions(min_observations=2)) == 2
    assert len(bars.sessions(min_observations=40)) == 1


def test_named_session_lookup(tmp_path):
    bars = load_intraday_csv(two_sessions(tmp_path))
    assert bars.session("2024-06-04").size == 100
    assert bars.session("2024-06-05").size == 0


# --------------------------------------------------------------------------- #
# Timestamp formats
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stamp",
    [
        "2024-06-03T09:30:00",
        "2024-06-03 09:30:00",
        "2024-06-03T09:30",
        "2024-06-03T09:30:00Z",
        "2024-06-03T09:30:00+00:00",
        "2024/06/03 09:30:00",
    ],
)
def test_timestamp_variants_all_parse_to_the_same_instant(tmp_path, stamp):
    bars = load_intraday_csv(write(tmp_path, f"timestamp,close\n{stamp},100.0\n"))
    assert len(bars) == 1
    assert bars.timestamps[0].astype("datetime64[m]") == np.datetime64("2024-06-03T09:30")


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        ("1717406100", "s"),
        ("1717406100000", "ms"),
        ("1717406100000000000", "ns"),
    ],
)
def test_epoch_timestamps_are_distinguished_by_magnitude(tmp_path, value, unit):
    """Seconds, milliseconds and nanoseconds must all land on the same instant."""
    bars = load_intraday_csv(write(tmp_path, f"timestamp,close\n{value},100.0\n"))
    assert bars.timestamps[0].astype("datetime64[s]") == np.datetime64("2024-06-03T09:15:00")


# --------------------------------------------------------------------------- #
# Column selection and robustness
# --------------------------------------------------------------------------- #
def test_price_column_is_chosen_by_preference_and_recorded(tmp_path):
    text = "timestamp,open,high,low,close,vwap\n2024-06-03T09:30:00,1,2,0.5,1.5,1.4\n"
    bars = load_intraday_csv(write(tmp_path, text))
    assert bars.price_column == "close"
    assert bars.prices[0] == 1.5

    override = load_intraday_csv(write(tmp_path, text), price_column="vwap")
    assert override.price_column == "vwap"
    assert override.prices[0] == 1.4


def test_missing_timestamp_column_names_what_it_looked_for(tmp_path):
    with pytest.raises(ValueError, match="no timestamp column"):
        load_intraday_csv(write(tmp_path, "when,close\n2024-06-03,100\n"))


def test_missing_price_column_names_what_it_looked_for(tmp_path):
    with pytest.raises(ValueError, match="no price column"):
        load_intraday_csv(write(tmp_path, "timestamp,bid,ask\n2024-06-03T09:30:00,1,2\n"))


def test_unknown_named_price_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no column named"):
        load_intraday_csv(
            write(tmp_path, "timestamp,close\n2024-06-03T09:30:00,100\n"),
            price_column="nope",
        )


def test_bad_rows_are_counted_not_raised(tmp_path):
    """One malformed line must not discard a day of ticks."""
    lines = ["timestamp,close"]
    for i in range(50):
        lines.append(f"2024-06-03T09:30:{i:02d},{100 + i * 0.01:.4f}")
    lines.append("not-a-date,100.0")
    lines.append("2024-06-03T10:00:00,not-a-price")
    lines.append("2024-06-03T10:01:00,-5.0")  # non-positive prices are unusable

    bars = load_intraday_csv(write(tmp_path, "\n".join(lines)))
    assert len(bars) == 50
    assert any("unparseable timestamps" in note for note in bars.notes)
    assert any("unparseable prices" in note for note in bars.notes)


def test_a_file_with_nothing_usable_raises_with_the_counts(tmp_path):
    text = "timestamp,close\nbad,1.0\nworse,2.0\n"
    with pytest.raises(ValueError, match="no usable rows"):
        load_intraday_csv(write(tmp_path, text))


def test_out_of_order_rows_are_sorted_and_flagged(tmp_path):
    text = (
        "timestamp,close\n"
        "2024-06-03T10:00:00,101.0\n"
        "2024-06-03T09:30:00,100.0\n"
        "2024-06-03T11:00:00,102.0\n"
    )
    bars = load_intraday_csv(write(tmp_path, text))
    assert list(bars.prices) == [100.0, 101.0, 102.0]
    assert any("chronological" in note for note in bars.notes)


def test_duplicate_timestamps_are_kept(tmp_path):
    """Two trades in the same second are two observations, not an error."""
    text = (
        "timestamp,close\n"
        "2024-06-03T09:30:00,100.0\n"
        "2024-06-03T09:30:00,100.5\n"
        "2024-06-03T09:30:01,100.2\n"
    )
    assert len(load_intraday_csv(write(tmp_path, text))) == 3


def test_max_rows_limits_the_read(tmp_path):
    bars = load_intraday_csv(two_sessions(tmp_path), max_rows=30)
    assert len(bars) == 30


def test_volume_is_read_when_present(tmp_path):
    bars = load_intraday_csv(two_sessions(tmp_path))
    assert bars.volume is not None
    assert bars.volume.size == len(bars)


def test_summary_reports_sessions_and_gaps(tmp_path):
    summary = load_intraday_csv(two_sessions(tmp_path), symbol="TEST").summary()
    assert "TEST" in summary
    assert "2 sessions" in summary
    assert "overnight gaps" in summary


def test_empty_bars_do_not_crash_the_accessors():
    empty = IntradayBars(
        symbol="EMPTY",
        timestamps=np.zeros(0, dtype="datetime64[ns]"),
        prices=np.zeros(0),
    )
    assert len(empty) == 0
    assert empty.start == "" and empty.end == ""
    assert empty.sessions() == []
    assert empty.overnight_returns().size == 0
    assert empty.median_session_length() == 0
