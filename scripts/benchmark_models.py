#!/usr/bin/env python3
"""Score every volatility forecaster on the same walk-forward split.

The table this prints is the point of the exercise, whichever way it falls. A
sequence model reported without baselines is unfalsifiable, and in volatility
forecasting the baselines are strong -- an exponentially weighted moving average
with no fitted parameters is genuinely hard to beat at a one-day horizon.

Every forecaster sees exactly the same data in exactly the same order, and every
fit uses only observations preceding the point it forecasts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def run(ticker: str, *, train: int, step: int, refit_every: int) -> list:
    from quantos.data.market import fetch_prices
    from quantos.models.baselines import (
        ewma_volatility_forecast,
        garch_volatility_forecast,
        historical_volatility_forecast,
        random_walk_volatility_forecast,
        score_forecast,
    )
    from quantos.models.sequence import AttentionVolatilityModel

    price, _ = fetch_prices(ticker, start="2006-01-01", range_key="20y")
    returns = np.diff(np.log(price.prices))
    print(f"{ticker}: {returns.size} returns, {price.start}..{price.end}")

    indices = list(range(train, returns.size, step))
    actual = np.array([returns[t] for t in indices])

    forecasters = {
        "random walk (yesterday)": random_walk_volatility_forecast,
        "trailing 252d": historical_volatility_forecast,
        "EWMA (lambda=0.94)": ewma_volatility_forecast,
        "GARCH(1,1)-t": garch_volatility_forecast,
    }

    scores = []
    for name, forecaster in forecasters.items():
        started = time.perf_counter()
        predictions = np.array([forecaster(returns[max(0, t - train) : t]) for t in indices])
        scores.append(score_forecast(name, actual, predictions))
        print(f"  {name:26s} done in {time.perf_counter() - started:5.1f}s")

    # The sequence model is refit periodically rather than at every step: a full
    # retrain per forecast would be honest but take hours, and refitting on a
    # schedule is what a desk actually does.
    started = time.perf_counter()
    predictions = np.full(len(indices), np.nan)
    model: AttentionVolatilityModel | None = None
    window = 20
    for position, t in enumerate(indices):
        if position % refit_every == 0:
            model = AttentionVolatilityModel(window=window, d_model=8, seed=7)
            try:
                model.fit(returns[max(0, t - train) : t], epochs=150, patience=20)
            except ValueError:
                model = None
        if model is not None and t >= window:
            predictions[position] = float(model.predict(returns[t - window : t][None, :])[0])
    scores.append(score_forecast("attention (NumPy)", actual, predictions))
    print(f"  {'attention (NumPy)':26s} done in {time.perf_counter() - started:5.1f}s")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=["SPY", "AAPL", "TLT"])
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--refit-every", type=int, default=50)
    args = parser.parse_args()

    all_scores: dict[str, list] = {}
    for ticker in args.tickers:
        for score in run(ticker, train=args.train, step=args.step, refit_every=args.refit_every):
            all_scores.setdefault(score.name, []).append(score)
        print()

    print("=" * 78)
    print(f"{'forecaster':26s} {'QLIKE':>9s} {'pinball':>10s} {'RMSE(vol)':>10s} {'bias':>9s}")
    print("-" * 78)
    rows = []
    for name, scores in all_scores.items():
        qlike = float(np.mean([s.qlike for s in scores]))
        pinball = float(np.mean([s.pinball_mean for s in scores]))
        rmse = float(np.mean([s.rmse_volatility for s in scores]))
        bias = float(np.mean([s.bias for s in scores]))
        rows.append((qlike, name, pinball, rmse, bias))
    for qlike, name, pinball, rmse, bias in sorted(rows):
        print(f"{name:26s} {qlike:9.4f} {pinball:10.6f} {rmse:10.5f} {bias:+9.5f}")
    print("-" * 78)

    best = min(rows)[1]
    ewma = next((r for r in rows if "EWMA" in r[1]), None)
    attention = next((r for r in rows if "attention" in r[1]), None)
    print(f"best on QLIKE: {best}")
    if ewma and attention:
        verdict = "BEATS" if attention[0] < ewma[0] else "does NOT beat"
        print(
            f"the attention model {verdict} the EWMA baseline on QLIKE "
            f"({attention[0]:.4f} vs {ewma[0]:.4f})"
        )
        print(
            "Lower is better. A model that loses to a parameter-free baseline has not\n"
            "earned a claim, and that result is reported here rather than omitted."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
