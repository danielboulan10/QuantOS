"""Load real option chains, and turn quoted prices into a usable surface.

The gap between "an option chain CSV" and "data you can fit"
------------------------------------------------------------
A downloaded chain is mostly unusable, and the filtering is not a detail -- it
determines whether the resulting surface means anything. Four problems, in
descending order of how badly they corrupt a naive fit:

**1. In-the-money options carry almost no volatility information.**
A call 40% in the money is nearly all intrinsic value. Its vega is small relative
to its price, so a one-cent quoting error moves the implied volatility enormously
-- and its price is dominated by a quantity (``S - K``) that has nothing to do
with volatility. This module therefore takes **out-of-the-money options only**:
puts below the forward, calls above it. That is what every options desk does, and
it is why a chain of 200 contracts yields perhaps 60 usable points.

**2. Zero-bid options are not quotes.**
A contract bid at 0.00 and offered at 0.05 tells you the market will not pay
anything for it. Taking the 0.025 midpoint invents a price nobody traded, and
because these are deep wings, that invented price sets the tail of the surface.
They are dropped.

**3. The spot price is the wrong forward.**
Discounting and dividends move the at-the-money point. Using spot rather than the
forward tilts the entire smile, and the tilt masquerades as skew. Where both a
call and a put trade at the same strike, this module recovers the forward from
**put-call parity** rather than assuming a dividend yield -- the market's own
forward, implied by prices that must satisfy parity or be arbitraged.

**4. Wide spreads are noise dressed as data.**
A contract quoted 1.00 / 3.00 has a midpoint of 2.00 and an implied volatility
uncertainty of tens of points. The relative spread filter drops them.

What survives all four is a small, trustworthy set of points. That is the correct
outcome, and a pipeline that keeps 200 noisy points instead of 60 clean ones will
produce a smoother-looking surface that is more wrong.

Expected CSV columns
--------------------
Case-insensitive, with common aliases accepted:
``expiry``, ``strike``, ``type`` (call/put/C/P), and either ``bid``/``ask`` or
``last``/``price``. Optional: ``volume``, ``open_interest``, ``underlying``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quantos.derivatives.black_scholes import OptionType, implied_volatility

__all__ = [
    "ChainFilter",
    "OptionChain",
    "OptionQuote",
    "SmileSlice",
    "build_chain",
    "load_option_chain_csv",
]

_TRUE_CALL = {"c", "call", "calls"}
_TRUE_PUT = {"p", "put", "puts"}

_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d", "%d-%b-%Y", "%b %d %Y")

_ALIASES: dict[str, tuple[str, ...]] = {
    "expiry": (
        "expiry",
        "expiration",
        "expiration_date",
        "exp_date",
        "maturity",
        "expirationdate",
    ),
    "strike": ("strike", "strike_price", "k", "strikeprice"),
    "type": ("type", "option_type", "right", "cp", "call_put", "putcall", "optiontype"),
    "bid": ("bid", "bid_price", "bidprice"),
    "ask": ("ask", "ask_price", "askprice", "offer"),
    "last": ("last", "last_price", "price", "mark", "close", "lastprice"),
    "volume": ("volume", "vol", "trade_volume"),
    "open_interest": ("open_interest", "openinterest", "oi"),
    "underlying": ("underlying", "underlying_price", "spot", "stock_price", "underlyingprice"),
}


@dataclass(frozen=True)
class OptionQuote:
    """One contract, with everything needed to price it and to judge the quote."""

    expiry: date
    strike: float
    option_type: OptionType
    bid: float = float("nan")
    ask: float = float("nan")
    last: float = float("nan")
    volume: float = float("nan")
    open_interest: float = float("nan")

    @property
    def mid(self) -> float:
        """Midpoint where both sides are quoted, else the last trade."""
        if np.isfinite(self.bid) and np.isfinite(self.ask) and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.last

    @property
    def spread(self) -> float:
        if np.isfinite(self.bid) and np.isfinite(self.ask):
            return self.ask - self.bid
        return float("nan")

    @property
    def relative_spread(self) -> float:
        """Spread as a fraction of the midpoint -- the usable measure of quote quality."""
        mid = self.mid
        if not np.isfinite(mid) or mid <= 0:
            return float("inf")
        spread = self.spread
        return spread / mid if np.isfinite(spread) else float("inf")

    def is_quoted(self) -> bool:
        """A real two-sided quote, not a placeholder."""
        return bool(
            np.isfinite(self.bid) and self.bid > 0 and np.isfinite(self.ask) and self.ask > 0
        )


@dataclass
class ChainFilter:
    """Quality thresholds. Every rejection is counted and reported.

    The defaults are deliberately strict. A surface fitted to everything is
    smoother and less true; see the module docstring.
    """

    #: Drop contracts bid at zero -- the market will not pay for them.
    require_positive_bid: bool = True
    #: Drop quotes wider than this fraction of the midpoint.
    max_relative_spread: float = 0.50
    #: Keep only out-of-the-money contracts, measured against the forward.
    otm_only: bool = True
    #: Drop contracts expiring inside this many days: mostly pin risk and noise.
    min_days_to_expiry: int = 7
    #: Drop implied volatilities outside this range as fitting failures.
    min_implied_vol: float = 0.01
    max_implied_vol: float = 3.00
    #: Require at least this many usable strikes before an expiry is kept.
    min_strikes_per_expiry: int = 4


@dataclass(frozen=True)
class SmileSlice:
    """One expiry's worth of a chain, sorted by log-moneyness.

    A dataclass rather than a dict because the fields are genuinely of different
    types -- five arrays, a bool array, and two scalars. A ``dict[str, NDArray]``
    annotation covering all of them is simply false, and the falsehood propagates:
    every caller then reads ``slice["forward"]`` as an array and passes it where a
    float is expected. The type checker flagged exactly that in seven places.
    """

    strikes: NDArray[np.float64]
    log_moneyness: NDArray[np.float64]
    implied_vols: NDArray[np.float64]
    mids: NDArray[np.float64]
    is_call: NDArray[np.bool_]
    time_to_expiry: float
    forward: float

    def __len__(self) -> int:
        return int(self.strikes.size)


@dataclass
class OptionChain:
    """A filtered, IV-solved chain for one underlying on one date.

    All arrays are parallel and one entry long per surviving contract.
    """

    symbol: str
    as_of: date
    spot: float
    quotes: list[OptionQuote] = field(default_factory=list)

    #: Filled by :func:`build_chain`.
    strikes: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    expiries: NDArray[np.datetime64] = field(
        default_factory=lambda: np.zeros(0, dtype="datetime64[D]")
    )
    times_to_expiry: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    implied_vols: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    mids: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    is_call: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    forwards: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    #: log(K/F) -- the correct x-axis for a smile.
    log_moneyness: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    rate: float = 0.0
    rejections: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.implied_vols.size)

    def unique_expiries(self) -> list[np.datetime64]:
        return sorted(set(self.expiries.tolist()))

    def slice_at(self, expiry: np.datetime64) -> SmileSlice:
        """One expiry's smile, sorted by log-moneyness."""
        mask = self.expiries == expiry
        order = np.argsort(self.log_moneyness[mask])
        return SmileSlice(
            strikes=self.strikes[mask][order],
            log_moneyness=self.log_moneyness[mask][order],
            implied_vols=self.implied_vols[mask][order],
            mids=self.mids[mask][order],
            is_call=self.is_call[mask][order],
            time_to_expiry=float(self.times_to_expiry[mask][0]) if mask.any() else float("nan"),
            forward=float(self.forwards[mask][0]) if mask.any() else float("nan"),
        )

    def summary(self) -> str:
        lines = [
            f"{self.symbol} option chain as of {self.as_of}",
            f"  spot {self.spot:.2f}, {len(self)} usable contracts "
            f"across {len(self.unique_expiries())} expiries",
        ]
        if self.rejections:
            total = sum(self.rejections.values())
            lines.append(f"  {total} contracts rejected:")
            for reason, count in sorted(self.rejections.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {count:5d}  {reason}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def _parse_date(text: str) -> date | None:
    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _to_float(text: str) -> float:
    text = text.strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"na", "n/a", "nan", "-", "--", "null", "none"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _resolve_columns(header: list[str]) -> dict[str, int]:
    normalised = [column.strip().lower().replace(" ", "_") for column in header]
    found: dict[str, int] = {}
    for canonical, aliases in _ALIASES.items():
        for index, name in enumerate(normalised):
            if name in aliases:
                found[canonical] = index
                break
    return found


def load_option_chain_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
    as_of: date | None = None,
    spot: float | None = None,
    rate: float = 0.0,
    chain_filter: ChainFilter | None = None,
) -> OptionChain:
    r"""Read an option chain CSV and return a filtered, IV-solved chain.

    Purpose
        Get from a vendor file to something a surface can be fitted to, with
        every discarded contract accounted for.
    Inputs
        ``path`` -- CSV with expiry, strike, type and prices (see module docs).
        ``spot`` -- underlying price; read from the file if it carries one.
        ``rate`` -- continuously compounded discount rate.
    Outputs
        An :class:`OptionChain`. ``chain.rejections`` explains every drop.
    Failure modes
        Missing required columns raise :class:`ValueError` naming what was
        expected and what was found. Unparseable rows are counted, not raised --
        one malformed line should not discard a chain of thousands.

    Example
        A chain priced from a flat 20% volatility, so the recovered implied
        volatilities must come back at 20%:

        >>> import tempfile, pathlib, datetime
        >>> import numpy as np
        >>> from quantos.derivatives.black_scholes import black_scholes_price, OptionType
        >>> spot, T = 100.0, 0.5
        >>> rows = ["expiry,strike,type,bid,ask"]
        >>> for strike in range(80, 126, 5):
        ...     kind = OptionType.CALL if strike >= spot else OptionType.PUT
        ...     fair = float(black_scholes_price(spot, strike, T, 0.20, option_type=kind))
        ...     rows.append(f"2025-07-03,{strike},{kind.name.lower()},"
        ...                 f"{fair * 0.99:.4f},{fair * 1.01:.4f}")
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "chain.csv"
        >>> _ = p.write_text("\n".join(rows))
        >>> chain = load_option_chain_csv(p, symbol="TEST", spot=spot,
        ...                               as_of=datetime.date(2025, 1, 2))
        >>> len(chain)
        10
        >>> bool(np.all(np.abs(chain.implied_vols - 0.20) < 0.005))
        True

        Adding the in-the-money half of the grid changes nothing, because those
        contracts are filtered out rather than fitted:

        >>> for strike in range(80, 100, 5):                      # ITM calls
        ...     fair = float(black_scholes_price(spot, strike, T, 0.20,
        ...                  option_type=OptionType.CALL))
        ...     rows.append(f"2025-07-03,{strike},call,{fair * 0.99:.4f},{fair * 1.01:.4f}")
        >>> _ = p.write_text("\n".join(rows))
        >>> wider = load_option_chain_csv(p, symbol="TEST", spot=spot,
        ...                               as_of=datetime.date(2025, 1, 2))
        >>> len(wider), wider.rejections["in the money (little volatility information)"]
        (10, 4)
    """
    path = Path(path)
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise ValueError(f"{path} is empty")

    columns = _resolve_columns(rows[0])
    missing = [name for name in ("expiry", "strike", "type") if name not in columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}. Found headers "
            f"{rows[0]}. Accepted aliases: "
            + "; ".join(f"{k}={'/'.join(v)}" for k, v in _ALIASES.items() if k in missing)
        )
    if "bid" not in columns and "last" not in columns:
        raise ValueError(f"{path}: need either bid/ask or last/price columns; found {rows[0]}")

    quotes: list[OptionQuote] = []
    rejections: dict[str, int] = {}
    file_spot = float("nan")

    def cell(row: list[str], name: str) -> str:
        index = columns.get(name)
        return row[index] if index is not None and index < len(row) else ""

    for row in rows[1:]:
        if not row or not any(field.strip() for field in row):
            continue
        expiry = _parse_date(cell(row, "expiry"))
        if expiry is None:
            rejections["unparseable expiry"] = rejections.get("unparseable expiry", 0) + 1
            continue
        strike = _to_float(cell(row, "strike"))
        if not np.isfinite(strike) or strike <= 0:
            rejections["invalid strike"] = rejections.get("invalid strike", 0) + 1
            continue

        raw_type = cell(row, "type").strip().lower()
        if raw_type in _TRUE_CALL:
            option_type = OptionType.CALL
        elif raw_type in _TRUE_PUT:
            option_type = OptionType.PUT
        else:
            rejections["unrecognised option type"] = (
                rejections.get("unrecognised option type", 0) + 1
            )
            continue

        underlying = _to_float(cell(row, "underlying"))
        if np.isfinite(underlying) and underlying > 0:
            file_spot = underlying

        quotes.append(
            OptionQuote(
                expiry=expiry,
                strike=strike,
                option_type=option_type,
                bid=_to_float(cell(row, "bid")),
                ask=_to_float(cell(row, "ask")),
                last=_to_float(cell(row, "last")),
                volume=_to_float(cell(row, "volume")),
                open_interest=_to_float(cell(row, "open_interest")),
            )
        )

    if spot is None:
        spot = file_spot
    if spot is None or not np.isfinite(spot) or spot <= 0:
        raise ValueError(
            f"{path}: no underlying price. Pass spot=... or include an "
            "'underlying'/'spot' column -- the chain cannot be interpreted without it."
        )

    return build_chain(
        quotes,
        symbol=symbol or path.stem.upper(),
        as_of=as_of or date.today(),
        spot=float(spot),
        rate=rate,
        chain_filter=chain_filter,
        seed_rejections=rejections,
    )


def implied_forward(
    quotes: list[OptionQuote], spot: float, time_to_expiry: float, rate: float
) -> tuple[float, str]:
    r"""Recover the forward from put-call parity, falling back to the carry forward.

    Parity says :math:`C - P = e^{-rT}(F - K)`, so a call and put at the *same*
    strike pin the forward exactly:

    .. math:: F = K + e^{rT}(C - P)

    This is worth doing rather than assuming a dividend yield, because the market
    prices in discrete dividends, borrow costs and hard-to-borrow spreads that no
    single yield reproduces. Using the wrong forward tilts the whole smile and
    the tilt looks exactly like skew -- a mistake that survives every later step.

    The strike nearest the money is used, because that is where both legs are
    liquid and where parity binds most tightly.
    """
    calls = {q.strike: q for q in quotes if q.option_type is OptionType.CALL and q.is_quoted()}
    puts = {q.strike: q for q in quotes if q.option_type is OptionType.PUT and q.is_quoted()}
    shared = sorted(set(calls) & set(puts))

    if shared:
        atm_strike = min(shared, key=lambda k: abs(k - spot))
        call_mid = calls[atm_strike].mid
        put_mid = puts[atm_strike].mid
        if np.isfinite(call_mid) and np.isfinite(put_mid):
            forward = atm_strike + np.exp(rate * time_to_expiry) * (call_mid - put_mid)
            # Sanity bound: a forward far from spot means the parity pair was bad.
            if 0.5 * spot < forward < 2.0 * spot:
                return float(forward), f"put-call parity at K={atm_strike:g}"

    return float(spot * np.exp(rate * time_to_expiry)), "carry (no parity pair available)"


def build_chain(
    quotes: list[OptionQuote],
    *,
    symbol: str,
    as_of: date,
    spot: float,
    rate: float = 0.0,
    chain_filter: ChainFilter | None = None,
    seed_rejections: dict[str, int] | None = None,
) -> OptionChain:
    """Filter quotes, recover forwards, and solve for implied volatility.

    The order matters: the forward is recovered *before* the OTM test, because
    "out of the money" is defined against the forward, not against spot.
    """
    rules = chain_filter or ChainFilter()
    rejections = dict(seed_rejections or {})

    def reject(reason: str, count: int = 1) -> None:
        rejections[reason] = rejections.get(reason, 0) + count

    by_expiry: dict[date, list[OptionQuote]] = {}
    for quote in quotes:
        by_expiry.setdefault(quote.expiry, []).append(quote)

    strikes: list[float] = []
    expiry_stamps: list[np.datetime64] = []
    times: list[float] = []
    vols: list[float] = []
    mids: list[float] = []
    call_flags: list[bool] = []
    forwards: list[float] = []
    notes: list[str] = []

    for expiry in sorted(by_expiry):
        days = (expiry - as_of).days
        if days < 0:
            # Distinguished from the near-dated case because the cause is
            # different: this is a stale file or a wrong as_of, not a filter
            # choice, and reporting it as "within 7 days" sends the reader to
            # adjust a threshold that is not the problem.
            reject(
                f"expiry {expiry} is before the as-of date {as_of} (stale chain, or pass as_of=)",
                len(by_expiry[expiry]),
            )
            continue
        if days < rules.min_days_to_expiry:
            reject(f"expiry within {rules.min_days_to_expiry} days", len(by_expiry[expiry]))
            continue
        time_to_expiry = days / 365.25

        forward, how = implied_forward(by_expiry[expiry], spot, time_to_expiry, rate)
        if expiry == min(by_expiry):
            notes.append(f"forward for {expiry}: {forward:.4f} via {how}")

        accepted_this_expiry: list[tuple] = []
        for quote in by_expiry[expiry]:
            if rules.require_positive_bid and not quote.is_quoted():
                reject("zero or missing bid")
                continue
            mid = quote.mid
            if not np.isfinite(mid) or mid <= 0:
                reject("no usable price")
                continue
            if quote.relative_spread > rules.max_relative_spread:
                reject(f"spread wider than {rules.max_relative_spread:.0%} of mid")
                continue
            if rules.otm_only:
                is_otm = (quote.option_type is OptionType.CALL and quote.strike >= forward) or (
                    quote.option_type is OptionType.PUT and quote.strike <= forward
                )
                if not is_otm:
                    reject("in the money (little volatility information)")
                    continue

            # Price off the forward: discounted-forward Black, so the recovered
            # forward is actually used rather than silently ignored.
            try:
                vol = implied_volatility(
                    mid * np.exp(rate * time_to_expiry),
                    forward,
                    quote.strike,
                    time_to_expiry,
                    rate=0.0,
                    option_type=quote.option_type,
                )
            except (ValueError, RuntimeError) as error:
                reason = str(error).split(".")[0][:60]
                reject(f"no implied volatility: {reason}")
                continue

            if not (rules.min_implied_vol <= vol <= rules.max_implied_vol):
                reject(
                    f"implied volatility outside [{rules.min_implied_vol}, {rules.max_implied_vol}]"
                )
                continue

            accepted_this_expiry.append(
                (
                    quote.strike,
                    np.datetime64(expiry, "D"),
                    time_to_expiry,
                    vol,
                    mid,
                    quote.option_type is OptionType.CALL,
                    forward,
                )
            )

        if len(accepted_this_expiry) < rules.min_strikes_per_expiry:
            reject(
                f"expiry kept fewer than {rules.min_strikes_per_expiry} strikes",
                len(accepted_this_expiry),
            )
            continue

        for record in accepted_this_expiry:
            strikes.append(record[0])
            expiry_stamps.append(record[1])
            times.append(record[2])
            vols.append(record[3])
            mids.append(record[4])
            call_flags.append(record[5])
            forwards.append(record[6])

    strike_array = np.asarray(strikes, dtype=float)
    forward_array = np.asarray(forwards, dtype=float)
    log_moneyness = (
        np.log(strike_array / forward_array) if strike_array.size else np.zeros(0, dtype=float)
    )

    if strike_array.size == 0:
        notes.append(
            "no contracts survived filtering; loosen ChainFilter or check the "
            "spot price -- an OTM test against a wrong forward rejects everything"
        )

    return OptionChain(
        symbol=symbol,
        as_of=as_of,
        spot=float(spot),
        quotes=quotes,
        strikes=strike_array,
        expiries=np.asarray(expiry_stamps, dtype="datetime64[D]"),
        times_to_expiry=np.asarray(times, dtype=float),
        implied_vols=np.asarray(vols, dtype=float),
        mids=np.asarray(mids, dtype=float),
        is_call=np.asarray(call_flags, dtype=bool),
        forwards=forward_array,
        log_moneyness=log_moneyness,
        rate=rate,
        rejections=rejections,
        notes=notes,
    )
