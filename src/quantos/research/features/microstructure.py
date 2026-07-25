r"""Market-microstructure features and estimators.

What makes this module unusual
------------------------------
Every estimator here can be *validated* rather than merely applied, because
:mod:`quantos.sim` provides ground truth that no historical dataset can:

* The simulator knows the true aggressor side of every trade, so
  :func:`lee_ready_classification` can be scored on how often it is wrong --
  turning a universally-used heuristic into a measured error rate.
* The simulator knows the latent fundamental value, so
  :func:`price_discovery_efficiency` can ask how much of the information the
  market actually incorporated.
* The simulator knows the size of the institutional parent order, so
  :func:`kyle_lambda` and the impact fits can be checked against a known answer.

That inversion -- estimator as the object of study rather than the tool -- is the
main reason to build a simulator instead of downloading a dataset.

Contents
--------
==========================================  =================================
:func:`order_flow_imbalance`                Cont-Kukanov-Stoikov OFI
:func:`queue_imbalance`                     Depth-based imbalance
:func:`trade_flow_imbalance`                Signed volume over a window
:func:`kyle_lambda`                         Price impact per unit signed flow
:func:`amihud_illiquidity`                  |return| per unit volume
:func:`vpin`                                Volume-synchronised toxicity
:func:`effective_spread`                    Realised cost of a round trip
:func:`realised_spread`                     Maker revenue net of adverse selection
:func:`price_impact_decomposition`          Permanent vs transient impact
:func:`lee_ready_classification`            Trade-sign inference, and its error
:func:`price_discovery_efficiency`          Variance-ratio and tracking measures
:func:`roll_implied_spread`                 Spread from trade autocovariance
==========================================  =================================

References
----------
Kyle, A. S. (1985), *Econometrica* 53(6), 1315-1335.
Lee, C. M. C. & Ready, M. J. (1991), "Inferring trade direction from intraday
    data", *J. Finance* 46(2), 733-746.
Cont, R., Kukanov, A. & Stoikov, S. (2014), "The price impact of order book
    events", *J. Financial Econometrics* 12(1), 47-88.
Easley, D., Lopez de Prado, M. & O'Hara, M. (2012), "Flow toxicity and liquidity
    in a high-frequency world", *Rev. Financial Studies* 25(5), 1457-1493.
Amihud, Y. (2002), "Illiquidity and stock returns", *J. Financial Markets* 5(1).
Roll, R. (1984), "A simple implicit measure of the effective bid-ask spread",
    *J. Finance* 39(4), 1127-1139.
Hasbrouck, J. (1991), "Measuring the information content of stock trades",
    *J. Finance* 46(1), 179-207.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ImpactDecomposition",
    "KyleLambdaResult",
    "LeeReadyResult",
    "PriceDiscoveryResult",
    "amihud_illiquidity",
    "effective_spread",
    "kyle_lambda",
    "lee_ready_classification",
    "order_flow_imbalance",
    "price_discovery_efficiency",
    "price_impact_decomposition",
    "queue_imbalance",
    "realised_spread",
    "roll_implied_spread",
    "trade_flow_imbalance",
    "vpin",
]


def _finite(x: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


# --------------------------------------------------------------------------- #
# Imbalance measures                                                          #
# --------------------------------------------------------------------------- #
def order_flow_imbalance(
    bid_prices: ArrayLike,
    bid_sizes: ArrayLike,
    ask_prices: ArrayLike,
    ask_sizes: ArrayLike,
) -> NDArray[np.float64]:
    r"""Order Flow Imbalance (Cont-Kukanov-Stoikov).

    Purpose
        Measure net buying pressure from *book events* rather than from trades.
        OFI is the strongest known linear predictor of short-horizon price
        changes -- Cont et al. report R-squared around 0.65 at the event
        frequency, far above anything trade-based.
    Method
        For each update, the contribution of the bid side is

        .. math::
            e_n^{b} = \begin{cases}
                q_n^{b} & P_n^{b} > P_{n-1}^{b} \\
                q_n^{b} - q_{n-1}^{b} & P_n^{b} = P_{n-1}^{b} \\
                -q_{n-1}^{b} & P_n^{b} < P_{n-1}^{b}
            \end{cases}

        and symmetrically (with sign reversed) for the ask. ``OFI = e_b - e_a``.

        The three cases are the whole idea: a bid *price* improvement is
        unambiguously new demand at a better level, a size change at an unchanged
        price is incremental, and a bid price falling means the previous level was
        consumed or cancelled entirely. Simply differencing sizes -- the common
        shortcut -- conflates all three and destroys most of the signal.
    Inputs
        Four equal-length arrays of top-of-book prices and sizes.
    Outputs
        Array of length ``n``; the first element is 0 (no previous state).
    Failure modes
        NaNs in the input (one-sided book) propagate as 0 contributions for that
        step rather than poisoning the series.

    Example
        >>> import numpy as np
        >>> ofi = order_flow_imbalance([10., 10., 11.], [100., 150., 120.],
        ...                            [11., 11., 12.], [100., 100., 100.])
        >>> ofi.tolist()   # step 3: +120 new bid, +100 from the ask lifting away
        [0.0, 50.0, 220.0]
    """
    bp = np.asarray(bid_prices, dtype=float).ravel()
    bs = np.asarray(bid_sizes, dtype=float).ravel()
    ap = np.asarray(ask_prices, dtype=float).ravel()
    asz = np.asarray(ask_sizes, dtype=float).ravel()
    if not (bp.size == bs.size == ap.size == asz.size):
        raise ValueError("all four arrays must have the same length")
    n = bp.size
    out = np.zeros(n)
    if n < 2:
        return out

    # Bid side: price up -> all new size counts; equal -> the delta; down -> the
    # whole previous level left.
    bid_contrib = np.where(
        bp[1:] > bp[:-1], bs[1:], np.where(bp[1:] == bp[:-1], bs[1:] - bs[:-1], -bs[:-1])
    )
    # Ask side mirrors it: an ask price *falling* is new supply.
    ask_contrib = np.where(
        ap[1:] < ap[:-1], asz[1:], np.where(ap[1:] == ap[:-1], asz[1:] - asz[:-1], -asz[:-1])
    )
    step = bid_contrib - ask_contrib
    out[1:] = np.where(np.isfinite(step), step, 0.0)
    return out


def queue_imbalance(bid_sizes: ArrayLike, ask_sizes: ArrayLike) -> NDArray[np.float64]:
    r"""Normalised depth imbalance :math:`(Q_b - Q_a)/(Q_b + Q_a) \in [-1,1]`.

    A weaker predictor than :func:`order_flow_imbalance` (it uses levels, not
    changes) but it needs only a single snapshot, which makes it the practical
    choice when only sampled data is available.
    """
    bs = np.asarray(bid_sizes, dtype=float).ravel()
    asz = np.asarray(ask_sizes, dtype=float).ravel()
    total = bs + asz
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (bs - asz) / total
    return np.where(total > 0, out, 0.0)


def trade_flow_imbalance(signed_volumes: ArrayLike, window: int = 50) -> NDArray[np.float64]:
    """Rolling sum of aggressor-signed trade volume.

    Requires a trade *sign*, which real feeds do not provide -- see
    :func:`lee_ready_classification` for how it is normally inferred and how
    often that inference is wrong.
    """
    v = np.asarray(signed_volumes, dtype=float).ravel()
    if window < 1 or window > v.size:
        raise ValueError(f"window must lie in [1, {v.size}]")
    kernel = np.ones(window)
    out = np.full(v.size, np.nan)
    out[window - 1 :] = np.convolve(v, kernel, mode="valid")
    return out


# --------------------------------------------------------------------------- #
# Price impact                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KyleLambdaResult:
    r"""Estimated Kyle's lambda with inference."""

    #: Price impact per unit of signed volume, in price units per lot.
    lambda_: float
    standard_error: float
    t_statistic: float
    r_squared: float
    n_obs: int
    #: Implied depth: signed volume required to move price one unit.
    market_depth: float = field(default=float("nan"))

    @property
    def is_significant(self) -> bool:
        return bool(abs(self.t_statistic) > 1.96)


def kyle_lambda(
    price_changes: ArrayLike, signed_volumes: ArrayLike, *, hac_lags: int | None = None
) -> KyleLambdaResult:
    r"""Estimate Kyle's lambda by regressing price change on signed volume.

    .. math:: \Delta P_t = \lambda \cdot \text{SignedVolume}_t + \varepsilon_t

    Purpose
        :math:`\lambda` is *the* summary measure of illiquidity: it is the price
        concession per unit of net order flow, and its reciprocal is market depth.
        In Kyle's model it is exactly the rate at which the market maker learns
        from order flow, so a high lambda means either a thin book or a high
        proportion of informed flow -- the two are not separately identified from
        price and volume alone, which is worth remembering before interpreting it
        as pure liquidity.
    Inputs
        ``price_changes`` -- :math:`\Delta P`. ``signed_volumes`` -- aggressor-
        signed volume over the same intervals. **No intercept is fitted**: Kyle's
        model implies impact is zero at zero flow, and adding a free intercept
        lets a drift in the sample absorb impact that belongs to lambda.
    Outputs
        :class:`KyleLambdaResult`. Standard errors are HAC (Newey-West) by
        default, because signed order flow is strongly autocorrelated -- order
        splitting guarantees it -- and classical standard errors overstate
        significance substantially.
    Complexity
        :math:`O(n)`.
    Failure modes
        Raises if fewer than 30 usable observations remain after removing NaNs.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> v = rng.standard_normal(2000) * 100
        >>> dp = 0.002 * v + rng.standard_normal(2000) * 0.05
        >>> res = kyle_lambda(dp, v)
        >>> bool(abs(res.lambda_ - 0.002) < 2e-4), res.is_significant
        (True, True)
    """
    from quantos.core.timeseries.ols import ols

    dp = np.asarray(price_changes, dtype=float).ravel()
    v = np.asarray(signed_volumes, dtype=float).ravel()
    if dp.size != v.size:
        raise ValueError("price_changes and signed_volumes must be the same length")
    mask = np.isfinite(dp) & np.isfinite(v)
    dp, v = dp[mask], v[mask]
    if dp.size < 30:
        raise ValueError(f"need at least 30 usable observations, got {dp.size}")

    fit = ols(dp, v.reshape(-1, 1), cov_type="hac", hac_lags=hac_lags)
    lam = float(fit.coefficients[0])
    se = float(fit.standard_errors[0])
    return KyleLambdaResult(
        lambda_=lam,
        standard_error=se,
        t_statistic=float(fit.t_statistics[0]),
        r_squared=float(fit.r_squared),
        n_obs=int(dp.size),
        market_depth=float(1.0 / lam) if lam != 0 else float("inf"),
    )


def amihud_illiquidity(returns: ArrayLike, volumes: ArrayLike) -> float:
    r"""Amihud's ILLIQ: :math:`\text{mean}(|r_t| / \text{Volume}_t)`.

    A crude but famously robust illiquidity proxy that needs only daily data.
    Unlike :func:`kyle_lambda` it requires no trade signs, which is why it
    dominates the empirical asset-pricing literature. The cost is that it cannot
    distinguish impact from volatility.
    """
    r = np.asarray(returns, dtype=float).ravel()
    v = np.asarray(volumes, dtype=float).ravel()
    if r.size != v.size:
        raise ValueError("returns and volumes must be the same length")
    mask = np.isfinite(r) & np.isfinite(v) & (v > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(r[mask]) / v[mask]))


def vpin(
    signed_volumes: ArrayLike, *, bucket_volume: float, n_buckets: int = 50
) -> NDArray[np.float64]:
    r"""Volume-Synchronised Probability of Informed Trading.

    .. math:: \text{VPIN}_t = \frac{\sum_{\tau=t-n+1}^{t}
              |V^{B}_\tau - V^{S}_\tau|}{n \cdot V}

    Purpose
        Estimate order-flow *toxicity*: the fraction of volume attributable to
        informed traders, and hence the adverse-selection risk a market maker
        currently faces.
    Method
        The defining idea is **volume-time rather than clock-time bucketing**.
        Information arrives with volume, not with the wall clock, so sampling in
        equal-volume buckets makes the measure invariant to intraday activity
        patterns. That is what lets VPIN be compared across a quiet mid-morning
        and a frantic close without a seasonal adjustment.
    Inputs
        ``signed_volumes`` -- per-trade signed volume. ``bucket_volume`` -- total
        volume per bucket. ``n_buckets`` -- rolling window length.
    Outputs
        One VPIN value per bucket beyond the first ``n_buckets``, in ``[0, 1]``.
    Known limitation
        VPIN's usefulness as an early warning is genuinely contested (Andersen
        & Bondarenko 2014 argue its predictive power is an artefact of volume
        clustering). It is included because it is standard, not because the
        matter is settled.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> balanced = rng.choice([-1.0, 1.0], 20000) * 10
        >>> toxic = np.full(20000, 10.0)          # entirely one-sided
        >>> b = float(np.mean(vpin(balanced, bucket_volume=500, n_buckets=20)))
        >>> t = float(np.mean(vpin(toxic, bucket_volume=500, n_buckets=20)))
        >>> bool(t > 0.9 and b < 0.2)
        True
    """
    v = np.asarray(signed_volumes, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if bucket_volume <= 0:
        raise ValueError("bucket_volume must be positive")
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")

    # Accumulate trades into equal-volume buckets.
    buy_volume: list[float] = []
    sell_volume: list[float] = []
    current_buy = current_sell = current_total = 0.0
    for signed in v:
        magnitude = abs(signed)
        if signed >= 0:
            current_buy += magnitude
        else:
            current_sell += magnitude
        current_total += magnitude
        if current_total >= bucket_volume:
            buy_volume.append(current_buy)
            sell_volume.append(current_sell)
            current_buy = current_sell = current_total = 0.0

    if len(buy_volume) <= n_buckets:
        return np.zeros(0)

    buys = np.asarray(buy_volume)
    sells = np.asarray(sell_volume)
    imbalance = np.abs(buys - sells)
    totals = buys + sells
    kernel = np.ones(n_buckets)
    rolling_imbalance = np.convolve(imbalance, kernel, mode="valid")
    rolling_total = np.convolve(totals, kernel, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = rolling_imbalance / rolling_total
    return np.clip(np.nan_to_num(out), 0.0, 1.0)


def effective_spread(
    trade_prices: ArrayLike, mid_prices: ArrayLike, trade_signs: ArrayLike
) -> NDArray[np.float64]:
    r"""Effective (half-)spread paid per trade.

    .. math:: S^{\text{eff}}_t = 2 \, d_t (P_t - M_t)

    with :math:`d_t = +1` for a buy. This is the *actual* transaction cost, and
    it can differ substantially from the quoted spread: it is smaller when trades
    receive price improvement, and larger when they walk the book.
    """
    p = np.asarray(trade_prices, dtype=float).ravel()
    m = np.asarray(mid_prices, dtype=float).ravel()
    d = np.sign(np.asarray(trade_signs, dtype=float).ravel())
    if not (p.size == m.size == d.size):
        raise ValueError("all inputs must be the same length")
    return 2.0 * d * (p - m)


def realised_spread(
    trade_prices: ArrayLike,
    future_mid_prices: ArrayLike,
    trade_signs: ArrayLike,
) -> NDArray[np.float64]:
    r"""Realised spread: the maker's revenue *after* adverse selection.

    .. math:: S^{\text{real}}_t = 2 \, d_t (P_t - M_{t+\Delta})

    Purpose
        Split the effective spread into what the liquidity provider keeps and
        what it loses to information. The identity is

        .. math:: \underbrace{S^{\text{eff}}}_{\text{cost to taker}}
                  = \underbrace{S^{\text{real}}}_{\text{maker revenue}}
                  + \underbrace{2 d_t (M_{t+\Delta} - M_t)}_{\text{price impact}}

        and it is the single most useful decomposition in microstructure: a
        market maker with a healthy effective spread and a *negative* realised
        spread is being adversely selected and is losing money while appearing to
        earn the spread.
    Inputs
        ``future_mid_prices`` -- the mid at a fixed horizon after each trade
        (commonly 5 minutes in equities, milliseconds in HFT contexts).
    """
    p = np.asarray(trade_prices, dtype=float).ravel()
    m = np.asarray(future_mid_prices, dtype=float).ravel()
    d = np.sign(np.asarray(trade_signs, dtype=float).ravel())
    if not (p.size == m.size == d.size):
        raise ValueError("all inputs must be the same length")
    return 2.0 * d * (p - m)


@dataclass(frozen=True)
class ImpactDecomposition:
    """Effective spread split into permanent and transient components."""

    effective_spread: float
    realised_spread: float
    permanent_impact: float
    n_trades: int

    @property
    def adverse_selection_share(self) -> float:
        """Fraction of the effective spread lost to information."""
        if self.effective_spread == 0:
            return float("nan")
        return self.permanent_impact / self.effective_spread

    @property
    def maker_is_adversely_selected(self) -> bool:
        """Whether providing liquidity was unprofitable before fees."""
        return self.realised_spread < 0.0


def price_impact_decomposition(
    trade_prices: ArrayLike,
    mid_prices: ArrayLike,
    future_mid_prices: ArrayLike,
    trade_signs: ArrayLike,
) -> ImpactDecomposition:
    r"""Decompose the effective spread into permanent and transient impact.

    Applies the identity documented in :func:`realised_spread`. The permanent
    component :math:`2 d_t (M_{t+\Delta} - M_t)` is the market's revision of its
    fair-value estimate in response to the trade -- i.e. the information content.
    """
    eff = effective_spread(trade_prices, mid_prices, trade_signs)
    real = realised_spread(trade_prices, future_mid_prices, trade_signs)
    d = np.sign(np.asarray(trade_signs, dtype=float).ravel())
    m = np.asarray(mid_prices, dtype=float).ravel()
    fm = np.asarray(future_mid_prices, dtype=float).ravel()
    permanent = 2.0 * d * (fm - m)

    mask = np.isfinite(eff) & np.isfinite(real) & np.isfinite(permanent)
    if not np.any(mask):
        raise ValueError("no usable observations")
    return ImpactDecomposition(
        effective_spread=float(np.mean(eff[mask])),
        realised_spread=float(np.mean(real[mask])),
        permanent_impact=float(np.mean(permanent[mask])),
        n_trades=int(np.sum(mask)),
    )


# --------------------------------------------------------------------------- #
# Trade-sign inference                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeeReadyResult:
    """Inferred trade signs, plus accuracy when ground truth is available."""

    signs: NDArray[np.float64]
    #: Fraction classified by the quote rule (rather than the tick fallback).
    quote_rule_share: float
    accuracy: float | None = None
    #: Accuracy restricted to trades that needed the tick-test fallback.
    tick_rule_accuracy: float | None = None

    @property
    def error_rate(self) -> float | None:
        return None if self.accuracy is None else 1.0 - self.accuracy


def lee_ready_classification(
    trade_prices: ArrayLike,
    mid_prices: ArrayLike,
    *,
    true_signs: ArrayLike | None = None,
) -> LeeReadyResult:
    r"""Lee-Ready trade-sign inference, and its measured error rate.

    Purpose
        Real market-data feeds publish price and size but not who initiated the
        trade. Every signed-flow measure in this module -- OFI, VPIN, Kyle's
        lambda -- therefore rests on an *inferred* sign, and the standard
        inference is Lee-Ready.
    Method
        **Quote rule** first: a trade above the mid is a buy, below is a sell.
        For trades exactly at the mid the quote rule is silent, and the **tick
        test** is used instead: classify by the sign of the change from the
        previous different trade price.
    Validation
        Pass ``true_signs`` -- which :mod:`quantos.sim` can supply and a
        historical dataset cannot -- and the result reports the accuracy, broken
        out for the trades that needed the tick fallback. This converts an
        assumption every microstructure paper makes into a number. Published
        estimates put Lee-Ready accuracy at 85% or so in equities, with the
        errors concentrated exactly where it matters: at-the-mid trades in fast
        markets.
    Outputs
        :class:`LeeReadyResult`.

    Example
        >>> import numpy as np
        >>> res = lee_ready_classification([101., 99., 100.], [100., 100., 100.],
        ...                                true_signs=[1, -1, 1])
        >>> res.signs[:2].tolist()
        [1.0, -1.0]
    """
    p = np.asarray(trade_prices, dtype=float).ravel()
    m = np.asarray(mid_prices, dtype=float).ravel()
    if p.size != m.size:
        raise ValueError("trade_prices and mid_prices must be the same length")
    n = p.size

    signs = np.sign(p - m)
    needs_tick = signs == 0.0

    if np.any(needs_tick):
        # Tick test: compare against the last *different* trade price.
        last_different = np.full(n, np.nan)
        previous = np.nan
        for i in range(n):
            last_different[i] = previous
            if not np.isfinite(previous) or p[i] != previous:
                previous = p[i]
        tick = np.sign(p - last_different)
        # A zero tick (no prior different price) defaults to a buy, matching the
        # convention in the literature; it affects only the first few trades.
        tick = np.where(tick == 0.0, 1.0, tick)
        tick = np.where(np.isfinite(tick), tick, 1.0)
        signs = np.where(needs_tick, tick, signs)

    quote_share = float(np.mean(~needs_tick))

    accuracy: float | None = None
    tick_accuracy: float | None = None
    if true_signs is not None:
        truth = np.sign(np.asarray(true_signs, dtype=float).ravel())
        if truth.size != n:
            raise ValueError("true_signs must match the number of trades")
        accuracy = float(np.mean(signs == truth))
        if np.any(needs_tick):
            tick_accuracy = float(np.mean(signs[needs_tick] == truth[needs_tick]))

    return LeeReadyResult(
        signs=signs,
        quote_rule_share=quote_share,
        accuracy=accuracy,
        tick_rule_accuracy=tick_accuracy,
    )


# --------------------------------------------------------------------------- #
# Price discovery                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PriceDiscoveryResult:
    """How well an observed price tracked a latent fundamental value."""

    correlation: float
    rmse: float
    #: Regression slope of price on value; 1.0 means full incorporation.
    beta: float
    #: Fraction of fundamental variance reflected in price variance.
    variance_ratio: float
    #: Lag (in samples) that maximises correlation -- how far price trails value.
    optimal_lag: int
    n_obs: int

    @property
    def is_efficient(self) -> bool:
        """Loose efficiency criterion: high correlation and near-unit beta."""
        return bool(self.correlation > 0.7 and 0.5 < self.beta < 1.5)


def price_discovery_efficiency(
    observed_prices: ArrayLike,
    fundamental_values: ArrayLike,
    *,
    max_lag: int = 50,
) -> PriceDiscoveryResult:
    r"""Measure how efficiently a market discovered a latent value.

    Purpose
        The question an agent-based simulation exists to answer: given that no
        participant broadcasts the fundamental, how much of it does the price
        incorporate, and how fast?
    Method
        Correlation, RMSE, and the slope of price on value; plus a scan over lags
        to find where the cross-correlation peaks, which measures how far the
        price *trails* the value. A peak at a positive lag means information is
        being incorporated with a delay -- the size of that delay is the
        market's price-discovery latency.
    Inputs
        Two equal-length, contemporaneously-sampled series. Use
        :meth:`quantos.sim.world.SimulationResult.fundamental_path` interpolated
        onto the snapshot grid.
    Outputs
        :class:`PriceDiscoveryResult`.
    Failure modes
        Raises if either series is constant (correlation undefined) or shorter
        than ``max_lag * 2``.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> value = np.cumsum(rng.standard_normal(3000))
        >>> price = value + rng.standard_normal(3000) * 0.4    # efficient
        >>> price_discovery_efficiency(price, value).is_efficient
        True
    """
    from quantos.core.timeseries.ols import ols

    p = np.asarray(observed_prices, dtype=float).ravel()
    v = np.asarray(fundamental_values, dtype=float).ravel()
    n = min(p.size, v.size)
    p, v = p[:n], v[:n]
    mask = np.isfinite(p) & np.isfinite(v)
    p, v = p[mask], v[mask]
    if p.size < max(20, 2 * max_lag):
        raise ValueError(f"need at least {max(20, 2 * max_lag)} usable observations")
    if np.std(p) == 0 or np.std(v) == 0:
        raise ValueError("one of the series is constant; correlation is undefined")

    correlation = float(np.corrcoef(p, v)[0, 1])
    rmse = float(np.sqrt(np.mean((p - v) ** 2)))

    design = np.column_stack([np.ones(p.size), v])
    beta = float(ols(p, design).coefficients[1])
    variance_ratio = float(np.var(p) / np.var(v)) if np.var(v) > 0 else float("nan")

    # Lag scan on differences: levels are integrated, and cross-correlating two
    # integrated series is dominated by the common trend regardless of lag.
    dp = np.diff(p)
    dv = np.diff(v)
    best_lag, best = 0, -np.inf
    for lag in range(0, min(max_lag, dp.size // 4)):
        a = dp[lag:] if lag else dp
        b = dv[: dv.size - lag] if lag else dv
        m = min(a.size, b.size)
        if m < 10 or np.std(a[:m]) == 0 or np.std(b[:m]) == 0:
            continue
        c = float(np.corrcoef(a[:m], b[:m])[0, 1])
        if c > best:
            best, best_lag = c, lag

    return PriceDiscoveryResult(
        correlation=correlation,
        rmse=rmse,
        beta=beta,
        variance_ratio=variance_ratio,
        optimal_lag=best_lag,
        n_obs=int(p.size),
    )


def roll_implied_spread(trade_prices: ArrayLike) -> float:
    r"""Roll's (1984) effective spread from trade-price autocovariance.

    .. math:: S = 2\sqrt{-\operatorname{Cov}(\Delta P_t, \Delta P_{t-1})}

    Purpose
        Recover the effective spread from **trade prices alone** -- no quotes, no
        signs. Bid-ask bounce induces negative first-order autocovariance in
        trade-price changes, and its magnitude identifies the spread.
    Failure modes
        Returns ``nan`` when the autocovariance is *positive*, which happens
        whenever genuine price momentum outweighs the bounce. That is not a bug
        but the estimator's central limitation, and it occurs often enough in
        trending markets that Roll's measure should never be used without
        checking. Returning ``nan`` rather than zero forces the caller to notice.
    """
    p = _finite(np.asarray(trade_prices, dtype=float))
    if p.size < 30:
        raise ValueError("need at least 30 trades")
    dp = np.diff(p)
    if dp.size < 2:
        return float("nan")
    covariance = float(np.cov(dp[1:], dp[:-1], ddof=1)[0, 1])
    if covariance >= 0:
        return float("nan")
    return float(2.0 * np.sqrt(-covariance))
