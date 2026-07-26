"""Append-only forward-testing ledger: predictions recorded before outcomes exist.

Why this is the only backtest that cannot be overfit
----------------------------------------------------
Every other validation technique in this repository -- the deflated Sharpe ratio,
the probability of backtest overfitting, purged cross-validation, Hansen's SPA --
is a *correction*. Each one exists because the researcher saw the data before
choosing the strategy, and the corrections estimate how much that contaminated
the result. They are good corrections, and they are still corrections.

Forward testing needs none of them. A prediction written down today, for a
horizon that has not happened yet, cannot be tuned to its own outcome. There is
no lookback to adjust, no threshold to re-pick, no trial count to under-report.
Whatever the ledger says after a year is simply what happened.

The cost is that it is slow. You cannot forward-test a year of ideas in an
afternoon; that is precisely the property that makes it trustworthy.

The append-only guarantee
-------------------------
The ledger is a JSONL file, one record per line, and this module **never edits or
deletes a line**. Recording an outcome appends a *settlement* record that
references the prediction's id; it does not modify the prediction. So the file is
a complete history, and any attempt to quietly improve a past forecast leaves
both versions visible.

That is enforced structurally rather than by discipline: :meth:`Ledger.record`
opens the file in append mode and writes one line. There is no update path.

A hash chain makes tampering detectable
---------------------------------------
Each record carries the SHA-256 of the previous record's hash plus its own
content. Editing any earlier line breaks every subsequent hash, and
:meth:`Ledger.verify_chain` reports exactly where. This is not security -- anyone
with the file can rewrite the whole chain -- but it makes *accidental* corruption
and casual after-the-fact editing immediately visible, which is the realistic
failure mode for a research log.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Ledger",
    "LedgerError",
    "Prediction",
    "Settlement",
    "default_ledger_path",
]

GENESIS_HASH = "0" * 64


class LedgerError(RuntimeError):
    """Raised when the ledger is malformed or its hash chain is broken."""


def default_ledger_path() -> Path:
    """Where the ledger lives, honouring ``QUANTOS_LEDGER``."""
    override = os.environ.get("QUANTOS_LEDGER")
    if override:
        return Path(override)
    return Path.home() / ".quantos" / "forward_ledger.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Prediction:
    """A forecast, written before its outcome exists."""

    prediction_id: str
    symbol: str
    signal: str
    #: +1 long, -1 short, 0 flat.
    direction: int
    horizon_days: int
    #: The price at the moment of the prediction. Fixes the benchmark.
    entry_price: float
    #: Everything needed to reproduce the decision.
    as_of: str = field(default_factory=lambda: date.today().isoformat())
    created_at: str = field(default_factory=_utc_now)
    confidence: float = float("nan")
    rationale: str = ""
    data_source: str = ""
    quantos_version: str = ""
    #: Free-form, e.g. the signal's parameters.
    metadata: dict[str, Any] = field(default_factory=dict)

    def due_on(self) -> date:
        """Calendar date the horizon elapses. Trading days would need a calendar."""
        from datetime import timedelta

        return date.fromisoformat(self.as_of) + timedelta(days=self.horizon_days)

    def is_due(self, as_of: date | None = None) -> bool:
        return (as_of or date.today()) >= self.due_on()


@dataclass
class Settlement:
    """The realised outcome of a prediction. Appended, never merged."""

    prediction_id: str
    exit_price: float
    settled_on: str = field(default_factory=lambda: date.today().isoformat())
    created_at: str = field(default_factory=_utc_now)
    notes: str = ""

    def realised_return(self, entry_price: float, direction: int) -> float:
        """Signed log return over the horizon."""
        if entry_price <= 0 or self.exit_price <= 0:
            return float("nan")
        return float(direction * np.log(self.exit_price / entry_price))


@dataclass
class Ledger:
    """An append-only JSONL forward-testing ledger.

    Example
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "ledger.jsonl"
        >>> ledger = Ledger(path)
        >>> p = Prediction("p1", "SPY", "momentum_252d", 1, 21, 470.0)
        >>> _ = ledger.record_prediction(p)
        >>> _ = ledger.record_settlement(Settlement("p1", 480.0))
        >>> ledger.verify_chain()
        True
        >>> scored = ledger.score()
        >>> scored["n_settled"], round(scored["hit_rate"], 3)
        (1, 1.0)
    """

    path: Path = field(default_factory=default_ledger_path)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # -- writing ----------------------------------------------------------- #
    def _last_hash(self) -> str:
        records = self.read_all()
        return records[-1]["hash"] if records else GENESIS_HASH

    @staticmethod
    def _hash_record(previous_hash: str, payload: dict[str, Any]) -> str:
        material = previous_hash + json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def record(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one record. There is deliberately no update or delete path."""
        previous = self._last_hash()
        record = {
            "kind": kind,
            "previous_hash": previous,
            "payload": payload,
        }
        record["hash"] = self._hash_record(previous, payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def record_prediction(self, prediction: Prediction) -> dict[str, Any]:
        """Write a forecast. Fails if the id already exists.

        Duplicate ids are refused because the whole value of the ledger is that
        one prediction has one outcome. Allowing a second record under the same
        id would permit exactly the revision this design exists to prevent.
        """
        if self.find_prediction(prediction.prediction_id) is not None:
            raise LedgerError(
                f"prediction {prediction.prediction_id!r} already exists. The ledger "
                "is append-only: use a new id rather than revising a forecast."
            )
        return self.record("prediction", asdict(prediction))

    def record_settlement(self, settlement: Settlement) -> dict[str, Any]:
        """Write an outcome. Fails if unknown or already settled."""
        prediction = self.find_prediction(settlement.prediction_id)
        if prediction is None:
            raise LedgerError(f"no prediction with id {settlement.prediction_id!r}")
        if self.find_settlement(settlement.prediction_id) is not None:
            raise LedgerError(
                f"prediction {settlement.prediction_id!r} is already settled. Re-settling "
                "would let an unfavourable outcome be replaced by a favourable one."
            )
        return self.record("settlement", asdict(settlement))

    # -- reading ----------------------------------------------------------- #
    def read_all(self) -> list[dict[str, Any]]:
        """Every record, in order. Returns empty if the ledger does not exist."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise LedgerError(f"{self.path}:{number} is not valid JSON: {error}") from error
        return records

    def predictions(self) -> list[Prediction]:
        return [
            Prediction(**record["payload"])
            for record in self.read_all()
            if record.get("kind") == "prediction"
        ]

    def settlements(self) -> list[Settlement]:
        return [
            Settlement(**record["payload"])
            for record in self.read_all()
            if record.get("kind") == "settlement"
        ]

    def find_prediction(self, prediction_id: str) -> Prediction | None:
        for prediction in self.predictions():
            if prediction.prediction_id == prediction_id:
                return prediction
        return None

    def find_settlement(self, prediction_id: str) -> Settlement | None:
        for settlement in self.settlements():
            if settlement.prediction_id == prediction_id:
                return settlement
        return None

    def open_predictions(self, as_of: date | None = None) -> list[Prediction]:
        """Predictions with no settlement yet."""
        settled = {s.prediction_id for s in self.settlements()}
        return [p for p in self.predictions() if p.prediction_id not in settled]

    def due_predictions(self, as_of: date | None = None) -> list[Prediction]:
        """Open predictions whose horizon has elapsed and which can be settled."""
        return [p for p in self.open_predictions() if p.is_due(as_of)]

    # -- integrity --------------------------------------------------------- #
    def verify_chain(self) -> bool:
        """Recompute every hash. Raises :class:`LedgerError` at the first break."""
        previous = GENESIS_HASH
        for number, record in enumerate(self.read_all(), 1):
            if record.get("previous_hash") != previous:
                raise LedgerError(
                    f"{self.path}:{number} chain broken: previous_hash is "
                    f"{record.get('previous_hash')!r}, expected {previous!r}. A record "
                    "before this one was edited or removed."
                )
            expected = self._hash_record(previous, record["payload"])
            if record.get("hash") != expected:
                raise LedgerError(
                    f"{self.path}:{number} content hash mismatch: this record's payload "
                    "was modified after it was written."
                )
            previous = record["hash"]
        return True

    # -- scoring ----------------------------------------------------------- #
    def score(self, *, signal: str | None = None, symbol: str | None = None) -> dict[str, Any]:
        """Score every settled prediction.

        No corrections are applied and none are needed. Each prediction was
        recorded before its outcome existed, so this is out-of-sample by
        construction -- there is no selection bias to deflate.

        The one statistic that *is* reported with its uncertainty is the hit
        rate, because a run of six correct calls out of ten says very little and
        the binomial interval makes that concrete.
        """
        from quantos.core.special import ndtri

        predictions = {p.prediction_id: p for p in self.predictions()}
        if signal:
            predictions = {k: v for k, v in predictions.items() if v.signal == signal}
        if symbol:
            predictions = {k: v for k, v in predictions.items() if v.symbol == symbol}

        returns: list[float] = []
        for settlement in self.settlements():
            prediction = predictions.get(settlement.prediction_id)
            if prediction is None:
                continue
            realised = settlement.realised_return(prediction.entry_price, prediction.direction)
            if np.isfinite(realised):
                returns.append(realised)

        n_open = len([p for p in self.open_predictions() if p.prediction_id in predictions])
        if not returns:
            return {
                "n_predictions": len(predictions),
                "n_settled": 0,
                "n_open": n_open,
                "hit_rate": float("nan"),
                "mean_return": float("nan"),
                "note": "nothing settled yet; forward testing takes as long as it takes",
            }

        values = np.asarray(returns, dtype=float)
        hits = int(np.sum(values > 0))
        hit_rate = hits / values.size

        # Wilson interval: correct for small n, unlike the normal approximation,
        # which can produce a lower bound below zero on a handful of trials.
        z = float(ndtri(np.array(0.975)))
        denominator = 1.0 + z * z / values.size
        centre = (hit_rate + z * z / (2 * values.size)) / denominator
        spread = (
            z
            * np.sqrt(hit_rate * (1 - hit_rate) / values.size + z * z / (4 * values.size**2))
            / denominator
        )

        return {
            "n_predictions": len(predictions),
            "n_settled": int(values.size),
            "n_open": n_open,
            "hit_rate": hit_rate,
            "hit_rate_95_low": float(max(0.0, centre - spread)),
            "hit_rate_95_high": float(min(1.0, centre + spread)),
            "mean_return": float(np.mean(values)),
            "total_return": float(np.sum(values)),
            "std_return": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
            "best": float(np.max(values)),
            "worst": float(np.min(values)),
            "hit_rate_beats_coin_flip": bool(max(0.0, centre - spread) > 0.5),
        }
