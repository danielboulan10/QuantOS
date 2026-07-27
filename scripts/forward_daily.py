#!/usr/bin/env python3
"""Run one forward-testing cycle across the tracked universe, and publish it.

What this produces
------------------
``forward/ledger.jsonl``       the append-only record itself
``forward/RECORD.md``          a human-readable scoreboard

Run daily. It settles every prediction whose horizon has elapsed, records new
forecasts for signals whose previous view has matured, and rewrites the
scoreboard. Nothing is ever edited: see :mod:`quantos.live.ledger`.

Why the ledger is committed to the repository
---------------------------------------------
The hash chain makes tampering detectable to anyone holding the file, but it
cannot prove *when* a record was written -- a chain can be rebuilt from scratch.
Committing it supplies the missing half: every line's arrival is timestamped by
a commit that a third party (the forge) signed and that cannot be backdated
without rewriting public history.

So the two mechanisms cover each other. The chain shows nothing was edited in
place; the commit history shows when each prediction appeared, which is the claim
that actually matters -- that these forecasts were written before their outcomes
existed.

Usage
-----
    python scripts/forward_daily.py                  # the default universe
    python scripts/forward_daily.py --tickers SPY QQQ
    python scripts/forward_daily.py --score-only     # rewrite the scoreboard
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - script convenience
    sys.path.insert(0, str(ROOT / "src"))

#: Deliberately spread across asset classes rather than ten correlated megacaps.
#:
#: The point of tracking several is not to find the best one. It is that signals
#: which look identical on US equities often behave differently on bonds, gold or
#: crypto, and a universe of one cannot show that. Nine signals over eight
#: instruments also makes the multiple-comparison problem explicit rather than
#: hidden: 72 tracks means the best-looking one is expected to look good.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "SPY",  # US large cap
    "QQQ",  # US technology
    "IWM",  # US small cap
    "EFA",  # developed ex-US
    "TLT",  # long Treasuries
    "GLD",  # gold
    "USO",  # crude oil
    "BTC-USD",  # crypto
)

LEDGER_PATH = ROOT / "forward" / "ledger.jsonl"
RECORD_PATH = ROOT / "forward" / "RECORD.md"


def _utc_today() -> str:
    from datetime import datetime

    return datetime.now(timezone.utc).date().isoformat()


def run_cycle(tickers: tuple[str, ...], ledger_path: Path, *, offline: bool = False) -> list[dict]:
    """Settle what is due and record what is not already live, per ticker."""
    from quantos.data.market import MarketDataError, fetch_prices
    from quantos.live.ledger import Ledger
    from quantos.live.runner import run_daily

    ledger = Ledger(ledger_path)
    results: list[dict] = []

    for ticker in tickers:
        try:
            series, info = fetch_prices(ticker, start="2015-01-01", offline=offline)
        except MarketDataError as error:
            # One unavailable symbol must not abort the cycle: the others still
            # have predictions to settle, and a gap is recoverable while a
            # missed settlement is not.
            print(f"  {ticker:9s} SKIPPED — {error}")
            results.append({"symbol": ticker, "error": str(error)})
            continue

        outcome = run_daily(ledger, series)
        outcome["name"] = info.name
        results.append(outcome)
        print(
            f"  {ticker:9s} as of {outcome['as_of']}  "
            f"settled {outcome['settled']:2d}  recorded {outcome['recorded']:2d}  "
            f"live {outcome['still_live']:2d}  open {outcome['open_after']:3d}"
        )

    return results


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines += [
        "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |" for r in rows
    ]
    return "\n".join(lines)


def render_record(ledger_path: Path) -> str:
    """Render the scoreboard, including what it does not yet establish."""
    from quantos.live.ledger import Ledger, LedgerError

    ledger = Ledger(ledger_path)
    predictions = ledger.predictions()
    overall = ledger.score()

    try:
        ledger.verify_chain()
        integrity = "intact — no record has been edited or removed"
    except LedgerError as error:  # pragma: no cover - only on a damaged file
        integrity = f"**BROKEN**: {error}"

    started = min((p.as_of for p in predictions), default="--")
    symbols = sorted({p.symbol for p in predictions})
    signals = sorted({p.signal for p in predictions})

    out = [
        "# Forward-testing record",
        "",
        "Predictions recorded **before** their outcomes existed, and scored after.",
        "",
        "This is the only validation in this repository that needs no correction.",
        "Deflated Sharpe ratios, purged cross-validation and reality checks all exist",
        "because the researcher saw the data before choosing the strategy. Nothing here",
        "did. Whatever this table eventually says is simply what happened.",
        "",
        f"- **Started:** {started}",
        f"- **Last updated:** {_utc_today()}",
        f"- **Universe:** {', '.join(symbols) if symbols else '--'}",
        f"- **Signals:** {len(signals)} pre-registered",
        f"- **Records:** {len(ledger.read_all()):,} ({len(predictions):,} predictions)",
        f"- **Hash chain:** {integrity}",
        "",
        "## Overall",
        "",
    ]

    if overall.get("n_settled", 0) == 0:
        out += [
            f"**{overall['n_open']} predictions are open; none has matured yet.**",
            "",
            "That is the expected state early on, and it cannot be shortcut: a 30-day",
            "forecast takes 30 days. Come back later.",
            "",
        ]
    else:
        out += [
            _table(
                [
                    ["Settled predictions", f"{overall['n_settled']:,}"],
                    [
                        "Effective (independent) sample",
                        f"{overall['n_effective']:,}"
                        f"  (overlap factor {overall['overlap_factor']:.1f}x)",
                    ],
                    ["Still open", f"{overall['n_open']:,}"],
                    ["Hit rate", f"{overall['hit_rate']:.1%}"],
                    [
                        "95% interval",
                        f"{overall['hit_rate_95_low']:.1%} to {overall['hit_rate_95_high']:.1%}",
                    ],
                    ["Mean return per prediction", f"{overall['mean_return']:+.3%}"],
                    [
                        "Beats a coin flip at 95%?",
                        "**yes**" if overall["hit_rate_beats_coin_flip"] else "not established",
                    ],
                ],
                ["Measure", "Value"],
            ),
            "",
            "The hit rate uses every settled prediction; the interval uses the",
            "**independent** subset. A 30-day forecast recorded repeatedly overlaps",
            "itself, and counting overlapping outcomes as independent observations is",
            "how a forward record talks itself into significance it has not earned.",
            "",
        ]

    duplicates = ledger.duplicate_signal_groups()
    distinct = ledger.distinct_signal_count() or len(signals)

    by_signal = [r for r in ledger.score_by_signal() if r.get("n_settled", 0) > 0]
    if by_signal:
        threshold = 0.05 / max(1, distinct)
        out += [
            "## By signal",
            "",
            _table(
                [
                    [
                        r["signal"],
                        f"{r['n_settled']}",
                        f"{r['n_effective']}",
                        f"{r['hit_rate']:.0%}",
                        f"{r['hit_rate_95_low']:.0%}-{r['hit_rate_95_high']:.0%}",
                        f"{r['mean_return']:+.2%}",
                    ]
                    for r in by_signal
                ],
                ["Signal", "Settled", "Independent", "Hit rate", "95% interval", "Mean"],
            ),
            "",
            f"With {distinct} directionally distinct signals tracked, the best-looking one",
            f"is expected to look good by chance. A single-signal claim needs p < {threshold:.4f}",
            "after Bonferroni correction, not p < 0.05 — and the intervals above are not",
            "corrected for that, so read the best row with suspicion.",
            "",
        ]
        if duplicates:
            out += [
                "### Signals that are not actually distinct",
                "",
                "Only the *sign* of each signal's position is recorded, so signals that",
                "differ only in how large a bet they take collapse to the same prediction.",
                "Detected from the records themselves:",
                "",
            ]
            out += [f"- `{'` == `'.join(group)}`" for group in duplicates]
            out += [
                "",
                f"They are counted once for the correction above ({distinct} distinct, not",
                f"{len(signals)}). Volatility scaling changes position size, and size is not",
                "what a direction-only record scores — so the scaled variant is currently",
                "indistinguishable from its unscaled parent. Scoring size-weighted returns",
                "would separate them and is not yet done.",
                "",
            ]

    out += [
        "## What this does and does not establish",
        "",
        "**Does:** whether these nine pre-registered signals, applied mechanically to",
        "this universe, would have made money over the period covered — with no",
        "opportunity to revise the rules, the horizon, or the universe after seeing an",
        "outcome. Prediction ids are derived from the content of each decision, so a",
        "re-run cannot quietly add a second version.",
        "",
        "**Does not:** anything about a strategy that is not in the list; anything about",
        "sizing, costs beyond the entry-to-exit return, or capacity; and nothing at all",
        "until the effective sample is large enough for the interval to be narrow.",
        "Volatility forecasting is a separate matter and is not measured here.",
        "",
        "---",
        "",
        "Generated by `scripts/forward_daily.py`. The ledger is",
        "[`forward/ledger.jsonl`](ledger.jsonl); every line is hash-chained to the one",
        "before it, and the commit history timestamps when each arrived.",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=None, help="override the universe")
    parser.add_argument("--ledger", default=str(LEDGER_PATH))
    parser.add_argument("--record", default=str(RECORD_PATH))
    parser.add_argument("--score-only", action="store_true", help="rewrite the scoreboard only")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    universe = tuple(args.tickers) if args.tickers else DEFAULT_UNIVERSE

    if not args.score_only:
        print(f"forward cycle, {date.today()} — {len(universe)} instruments")
        results = run_cycle(universe, ledger_path, offline=args.offline)
        failures = [r for r in results if "error" in r]
        if len(failures) == len(universe):
            print("every instrument failed to fetch; not rewriting the record")
            return 1

    Path(args.record).write_text(render_record(ledger_path), encoding="utf-8")
    print(f"wrote {args.record}")

    from quantos.live.ledger import Ledger

    scored = Ledger(ledger_path).score()
    print(
        f"  {scored.get('n_predictions', 0)} predictions, "
        f"{scored.get('n_settled', 0)} settled, {scored.get('n_open', 0)} open"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
