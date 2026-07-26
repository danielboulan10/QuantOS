"""Turn today's prices into dated predictions, and settle yesterday's.

The daily cycle
---------------
``propose``  reads a price history, evaluates every signal *at the last bar*,
             and returns one :class:`~quantos.live.ledger.Prediction` per signal.
``settle``   finds predictions whose horizon has elapsed, looks up the price on
             the settlement date, and appends the outcome.

Running those two, in that order, once a day, is the entire loop. There is no
optimiser, no fitting step and no parameter to choose at run time -- the signals
are pre-registered in :data:`quantos.research.signals.SIGNALS` and their
parameters are fixed in source. That is deliberate: if the daily job could tune
anything, the ledger would stop being a clean out-of-sample record and become
just another backtest with extra steps.

Why every signal is recorded, not the best one
-----------------------------------------------
It is tempting to record only the signal that currently looks best. Doing that
would reintroduce exactly the selection bias the ledger exists to avoid -- the
choice of "best" is made on data, and the forward record would inherit it.

Recording all nine costs nothing and makes the multiple-comparison problem
*visible*: after a year there are nine independent forward track records, and if
the best of nine has a 55% hit rate, the reader can see that is roughly what nine
coin-flippers would produce.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from quantos.data.loader import PriceSeries
from quantos.live.ledger import Ledger, Prediction, Settlement
from quantos.research.signals import SIGNALS

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = ["prediction_id_for", "propose", "run_daily", "settle"]

#: Horizon in calendar days. 21 trading days is about 30 calendar days.
DEFAULT_HORIZON_DAYS = 30


def prediction_id_for(symbol: str, signal: str, as_of: str, horizon_days: int) -> str:
    """A deterministic id, so re-running the same day cannot double-record.

    Derived from the content of the decision rather than a counter or a
    timestamp: running the job twice on the same afternoon produces the same id,
    the second write is refused by the ledger, and nothing is duplicated. A
    random id would silently create a second record.
    """
    material = f"{symbol}|{signal}|{as_of}|{horizon_days}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def propose(
    series: PriceSeries,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    as_of: str | None = None,
    signals: list[str] | None = None,
) -> list[Prediction]:
    """Evaluate every signal at the final bar and produce dated forecasts.

    Purpose
        Convert a price history into predictions that can be scored later.
    Inputs
        ``series`` -- price history ending at the decision date. Nothing after
        the last bar is read, and the signals themselves only ever look
        backwards, so no future information can leak in.
        ``as_of`` -- override the decision date; defaults to the series' last
        date, *not* to today, so a stale data file cannot be silently dated as
        current.
    Outputs
        One prediction per signal that has enough history to fire.
    Failure modes
        Signals needing more history than is available are skipped with a
        recorded reason rather than firing on a partial window.

    Example
        >>> import numpy as np
        >>> from quantos.data.loader import PriceSeries
        >>> rng = np.random.default_rng(0)
        >>> prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 400)))
        >>> dates = np.datetime64("2024-01-01") + np.arange(400)
        >>> series = PriceSeries(symbol="TEST", dates=dates, prices=prices)
        >>> predictions = propose(series)
        >>> len(predictions), predictions[0].as_of
        (9, '2025-02-03')
        >>> all(p.direction in (-1, 0, 1) for p in predictions)
        True
    """
    prices = np.asarray(series.prices, dtype=float)
    if prices.size < 2:
        raise ValueError(f"{series.symbol}: need at least two prices to form a prediction")

    if as_of is None:
        as_of = _last_date_string(series)

    entry_price = float(prices[-1])
    wanted = set(signals) if signals else None
    predictions: list[Prediction] = []

    for name, description, reads_as, function in SIGNALS:
        if wanted is not None and name not in wanted:
            continue
        compute = cast("Callable[[NDArray[np.float64]], NDArray[np.float64]]", function)
        positions = np.asarray(compute(prices), dtype=float)
        latest = float(positions[-1])
        if not np.isfinite(latest):
            continue
        direction = int(np.sign(latest))

        predictions.append(
            Prediction(
                prediction_id=prediction_id_for(series.symbol, name, as_of, horizon_days),
                symbol=series.symbol,
                signal=name,
                direction=direction,
                horizon_days=horizon_days,
                entry_price=entry_price,
                as_of=as_of,
                confidence=abs(latest),
                rationale=f"{description}; positive means {reads_as}",
                data_source=getattr(series, "source", "") or "",
                quantos_version=_version(),
                metadata={
                    "raw_position": latest,
                    "n_bars_available": int(prices.size),
                },
            )
        )

    return predictions


def settle(
    ledger: Ledger,
    series: PriceSeries,
    *,
    as_of: date | None = None,
) -> list[Settlement]:
    """Settle every due prediction for this symbol at the latest price.

    Only predictions whose horizon has actually elapsed are settled. A position
    that is currently underwater but not yet due stays open -- closing it early
    because it looks bad, or late because it looks good, is the failure mode this
    guard exists to prevent.
    """
    prices = np.asarray(series.prices, dtype=float)
    if prices.size == 0:
        return []
    exit_price = float(prices[-1])
    settled_on = _last_date_string(series)

    written: list[Settlement] = []
    for prediction in ledger.due_predictions(as_of):
        if prediction.symbol != series.symbol:
            continue
        settlement = Settlement(
            prediction_id=prediction.prediction_id,
            exit_price=exit_price,
            settled_on=settled_on,
            notes=f"horizon of {prediction.horizon_days}d elapsed on {prediction.due_on()}",
        )
        ledger.record_settlement(settlement)
        written.append(settlement)
    return written


def run_daily(
    ledger: Ledger,
    series: PriceSeries,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    as_of: date | None = None,
) -> dict[str, Any]:
    """One full cycle: settle what is due, then record today's forecasts.

    Settlement runs *first* so that a prediction made today can never be settled
    by today's own price.
    """
    settlements = settle(ledger, series, as_of=as_of)

    recorded, skipped = [], []
    for prediction in propose(series, horizon_days=horizon_days):
        try:
            ledger.record_prediction(prediction)
            recorded.append(prediction)
        except Exception:  # already recorded for this (symbol, signal, date)
            skipped.append(prediction.signal)

    return {
        "symbol": series.symbol,
        "settled": len(settlements),
        "recorded": len(recorded),
        "already_present": len(skipped),
        "open_after": len(ledger.open_predictions()),
        "chain_valid": ledger.verify_chain(),
    }


def _last_date_string(series: PriceSeries) -> str:
    """The series' final date as ISO text, falling back to today."""
    dates = getattr(series, "dates", None)
    if dates is not None and len(dates) > 0:
        last = dates[-1]
        if isinstance(last, np.datetime64):
            return str(last.astype("datetime64[D]"))
        try:
            return str(np.datetime64(last, "D"))
        except (ValueError, TypeError):
            pass
    return date.today().isoformat()


def _version() -> str:
    try:
        from quantos import __version__

        return str(__version__)
    except Exception:
        return "unknown"
