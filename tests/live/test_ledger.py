"""Tests for the append-only forward-testing ledger.

The tests that matter here are not the happy paths. They are the ones that check
the ledger *refuses* things: revising a forecast, re-settling an outcome,
settling early, and silently editing history. Each of those is a way a forward
record could be quietly turned back into an in-sample one, and each is the
failure this module exists to make impossible.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from quantos.data.loader import PriceSeries
from quantos.live.ledger import Ledger, LedgerError, Prediction, Settlement
from quantos.live.runner import DEFAULT_HORIZON_DAYS, prediction_id_for, propose, run_daily, settle


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


def make_series(n: int = 400, *, seed: int = 0, symbol: str = "TEST") -> PriceSeries:
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    dates = np.datetime64("2023-01-01") + np.arange(n)
    return PriceSeries(symbol=symbol, dates=dates, prices=prices)


# --------------------------------------------------------------------------- #
# What the ledger refuses
# --------------------------------------------------------------------------- #
def test_a_prediction_cannot_be_revised(ledger):
    """The whole point: a forecast is written once."""
    ledger.record_prediction(Prediction("p1", "SPY", "momentum_21d", 1, 30, 470.0))

    with pytest.raises(LedgerError, match="append-only"):
        ledger.record_prediction(Prediction("p1", "SPY", "momentum_21d", -1, 30, 470.0))

    # The original survives, with its original direction.
    assert ledger.find_prediction("p1").direction == 1
    assert len(ledger.predictions()) == 1


def test_an_outcome_cannot_be_replaced(ledger):
    """Re-settling would let a loss be swapped for a gain."""
    ledger.record_prediction(Prediction("p1", "SPY", "momentum_21d", 1, 30, 470.0))
    ledger.record_settlement(Settlement("p1", 450.0))  # a loss

    with pytest.raises(LedgerError, match="already settled"):
        ledger.record_settlement(Settlement("p1", 490.0))  # a "correction"

    assert ledger.find_settlement("p1").exit_price == 450.0
    assert ledger.score()["mean_return"] < 0


def test_settling_an_unknown_prediction_is_refused(ledger):
    with pytest.raises(LedgerError, match="no prediction"):
        ledger.record_settlement(Settlement("never-recorded", 100.0))


# --------------------------------------------------------------------------- #
# Tamper detection
# --------------------------------------------------------------------------- #
def test_chain_verifies_on_an_untouched_ledger(ledger):
    for i in range(10):
        ledger.record_prediction(Prediction(f"p{i}", "SPY", "momentum_21d", 1, 30, 100.0 + i))
    ledger.record_settlement(Settlement("p3", 120.0))
    assert ledger.verify_chain() is True


def test_editing_a_past_record_breaks_the_chain(ledger):
    """Rewriting history in place is detected, and located."""
    for i in range(5):
        ledger.record_prediction(Prediction(f"p{i}", "SPY", "momentum_21d", 1, 30, 100.0))
    assert ledger.verify_chain() is True

    # Flip an early prediction's direction, the way someone might "fix" a bad call.
    lines = ledger.path.read_text().splitlines()
    record = json.loads(lines[1])
    record["payload"]["direction"] = -1
    lines[1] = json.dumps(record, sort_keys=True)
    ledger.path.write_text("\n".join(lines) + "\n")

    with pytest.raises(LedgerError, match="modified after it was written"):
        ledger.verify_chain()


def test_deleting_a_record_breaks_the_chain(ledger):
    """Removing an embarrassing line is detected too."""
    for i in range(5):
        ledger.record_prediction(Prediction(f"p{i}", "SPY", "momentum_21d", 1, 30, 100.0))

    lines = ledger.path.read_text().splitlines()
    del lines[2]
    ledger.path.write_text("\n".join(lines) + "\n")

    with pytest.raises(LedgerError, match="chain broken"):
        ledger.verify_chain()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_direction_is_applied_to_the_return(ledger):
    """A short that falls is a win. Getting this backwards would flatter shorts."""
    ledger.record_prediction(Prediction("long", "A", "s", 1, 30, 100.0))
    ledger.record_prediction(Prediction("short", "B", "s", -1, 30, 100.0))
    ledger.record_settlement(Settlement("long", 110.0))
    ledger.record_settlement(Settlement("short", 90.0))

    scored = ledger.score()
    assert scored["n_settled"] == 2
    assert scored["hit_rate"] == 1.0
    assert scored["mean_return"] > 0

    losing = ledger.score(symbol="B")
    assert losing["n_settled"] == 1
    assert losing["mean_return"] == pytest.approx(-np.log(0.9), rel=1e-12)


def test_wilson_interval_is_honest_about_small_samples():
    """Three correct calls out of three is not evidence of skill."""
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "l.jsonl"
    small = Ledger(path)
    for i in range(3):
        small.record_prediction(Prediction(f"p{i}", "A", "s", 1, 30, 100.0))
        small.record_settlement(Settlement(f"p{i}", 110.0))

    scored = small.score()
    assert scored["hit_rate"] == 1.0
    # A naive interval would be [1, 1]. Wilson is not that confident.
    assert scored["hit_rate_95_low"] < 0.5
    assert scored["hit_rate_beats_coin_flip"] is False


def test_empty_ledger_says_so_rather_than_returning_nan_silently(ledger):
    scored = ledger.score()
    assert scored["n_settled"] == 0
    assert "forward testing takes as long as it takes" in scored["note"]


def test_open_and_due_predictions_are_distinguished(ledger):
    today = date.today()
    ledger.record_prediction(
        Prediction("old", "A", "s", 1, 30, 100.0, as_of=str(today - timedelta(days=60)))
    )
    ledger.record_prediction(
        Prediction("new", "A", "s", 1, 30, 100.0, as_of=str(today - timedelta(days=2)))
    )

    assert {p.prediction_id for p in ledger.open_predictions()} == {"old", "new"}
    assert [p.prediction_id for p in ledger.due_predictions()] == ["old"]


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def test_prediction_ids_are_deterministic():
    """Re-running the same day must not create a second record."""
    a = prediction_id_for("SPY", "momentum_21d", "2025-01-02", 30)
    b = prediction_id_for("SPY", "momentum_21d", "2025-01-02", 30)
    c = prediction_id_for("SPY", "momentum_21d", "2025-01-03", 30)
    assert a == b
    assert a != c


def test_running_twice_in_one_day_records_nothing_new(ledger):
    series = make_series()
    first = run_daily(ledger, series)
    second = run_daily(ledger, series)

    assert first["recorded"] == 9
    assert second["recorded"] == 0
    assert second["already_present"] == 9
    assert len(ledger.predictions()) == 9


def test_predictions_are_dated_from_the_data_not_from_today():
    """A stale file must not be dated as if it were current."""
    series = make_series(n=300)
    predictions = propose(series)
    assert all(p.as_of == series.end for p in predictions)
    assert predictions[0].as_of != date.today().isoformat()


def test_no_lookahead_appending_future_bars_cannot_change_a_past_signal():
    """The central guarantee: signals are prefix-stable.

    For every signal, the value at bar ``t`` computed from a 300-bar history must
    equal the value at bar ``t`` computed from a 400-bar history. A signal that
    normalised by the full-sample standard deviation, used a centred window, or
    otherwise peeked forward would fail this -- the extra hundred bars would
    silently rewrite its own past.

    This is the property the ledger depends on. Without it, a forward record
    would drift every time new data arrived.
    """
    from quantos.research.signals import SIGNALS

    rng = np.random.default_rng(11)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 400)))
    cut = 300

    for name, _, _, function in SIGNALS:
        short = np.asarray(function(prices[:cut]), dtype=float)
        long = np.asarray(function(prices), dtype=float)
        np.testing.assert_array_equal(
            short,
            long[:cut],
            err_msg=f"{name} changed its own history when future bars were appended",
        )


def test_predictions_actually_respond_to_the_data():
    """Guards the test above: prefix-stability is trivial if nothing ever fires."""
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 900)))
    dates = np.datetime64("2022-01-01") + np.arange(900)

    directions = set()
    for cut in range(300, 900, 40):
        window = PriceSeries(symbol="TEST", dates=dates[:cut], prices=prices[:cut])
        directions.add(tuple(p.direction for p in propose(window)))

    assert len(directions) > 1, "signals never changed their minds across 15 decision dates"
    assert any(-1 in d for d in directions) and any(1 in d for d in directions)


def test_settlement_never_uses_the_entry_bar(ledger):
    """A prediction made today cannot be settled by today's own price."""
    series = make_series()
    run_daily(ledger, series)
    assert ledger.settlements() == []
    assert len(ledger.open_predictions()) == 9


def test_a_full_forward_cycle_settles_only_when_due(tmp_path):
    """Walk a year of data one month at a time and check the arithmetic."""
    ledger = Ledger(tmp_path / "walk.jsonl")
    prices = make_series(n=600).prices
    dates = np.datetime64("2023-01-01") + np.arange(600)

    for cut in range(300, 600, 30):
        window = PriceSeries(symbol="TEST", dates=dates[:cut], prices=prices[:cut])
        run_daily(ledger, window, as_of=dates[cut - 1].astype(date))

    ledger.verify_chain()
    scored = ledger.score()

    # Every settled prediction's return must reconcile against the raw prices.
    for settlement in ledger.settlements():
        prediction = ledger.find_prediction(settlement.prediction_id)
        expected = prediction.direction * np.log(settlement.exit_price / prediction.entry_price)
        assert settlement.realised_return(
            prediction.entry_price, prediction.direction
        ) == pytest.approx(expected, rel=1e-12)

    assert scored["n_settled"] > 0
    assert scored["n_settled"] + scored["n_open"] == scored["n_predictions"]
    assert 0.0 <= scored["hit_rate"] <= 1.0
    assert scored["hit_rate_95_low"] <= scored["hit_rate"] <= scored["hit_rate_95_high"]


def test_settle_only_touches_the_matching_symbol(ledger):
    past = str(date.today() - timedelta(days=90))
    ledger.record_prediction(Prediction("a", "AAA", "s", 1, 30, 100.0, as_of=past))
    ledger.record_prediction(Prediction("b", "BBB", "s", 1, 30, 100.0, as_of=past))

    written = settle(ledger, make_series(n=50, symbol="AAA"))
    assert [s.prediction_id for s in written] == ["a"]
    assert [p.prediction_id for p in ledger.open_predictions()] == ["b"]


def test_horizon_default_is_recorded_not_assumed(ledger):
    series = make_series()
    predictions = propose(series)
    assert all(p.horizon_days == DEFAULT_HORIZON_DAYS for p in predictions)
    assert all(p.due_on() > date.fromisoformat(p.as_of) for p in predictions)
