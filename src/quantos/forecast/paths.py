r"""Simulate forward price paths, so every forward-looking number is a distribution.

The idea
--------
A single forecast path is close to useless: it is one draw from a distribution,
and the distribution is the thing that matters. So nothing here predicts *a*
price. It generates many thousands of paths consistent with the instrument's
estimated dynamics, and every forward-looking number the rest of the package
reports -- fan charts, threshold probabilities, stop-out odds, expected
shortfall -- is a summary of that ensemble.

That reframing is what makes the outputs defensible. "There is a 22% chance of a
10% drawdown within a month" is a statement about a simulated distribution whose
assumptions are written down and whose calibration is testable
(:mod:`quantos.forecast.calibration`). "The price will be 312 in six months" is
not a statement about anything.

Two engines, deliberately
-------------------------
**GARCH with fat-tailed innovations** (:func:`simulate_garch_paths`). Volatility
clusters, so tomorrow's variance depends on today's shock. This engine reproduces
that, and draws innovations from a Student-t rather than a normal because equity
returns have tails a normal cannot represent -- with 4 degrees of freedom a
five-sigma day is thousands of times more likely than under a Gaussian, which is
much closer to what markets actually do.

**Block bootstrap** (:func:`simulate_bootstrap_paths`). Resamples contiguous
blocks of the instrument's own historical returns. It assumes no model at all:
no distributional form, no volatility equation. What it cannot do is produce a
crash larger than any in the sample, which is exactly the limitation the
parametric engine does not have.

They are both here because they fail differently. When they agree, the answer is
insensitive to the model; when they disagree, the disagreement is the finding,
and :func:`compare_engines` reports it rather than averaging it away.

What is deliberately *not* modelled
-----------------------------------
**Drift.** Expected return over a month cannot be estimated from ten years of
daily data to any useful precision -- the standard error swamps the estimate. So
paths are simulated driftless by default. That is not a shortcut: putting a
noisy drift estimate in would make every long-horizon probability a function of
that noise, and directional probabilities would then be manufactured rather than
measured. ``drift`` exists as an argument for callers who want to impose a view,
and defaults to zero so no view is imposed accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PathEnsemble",
    "compare_engines",
    "simulate_bootstrap_paths",
    "simulate_garch_paths",
]

TRADING_DAYS = 252


@dataclass
class PathEnsemble:
    """Simulated forward paths, and the assumptions behind them."""

    #: ``(n_paths, horizon + 1)`` prices. Column 0 is the (shared) spot.
    paths: NDArray[np.float64]
    spot: float
    horizon: int
    engine: str
    #: Everything a reader needs to judge the simulation.
    assumptions: dict[str, float | str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n_paths(self) -> int:
        return int(self.paths.shape[0])

    @property
    def terminal(self) -> NDArray[np.float64]:
        """Price at the horizon, one per path."""
        return self.paths[:, -1]

    @property
    def terminal_returns(self) -> NDArray[np.float64]:
        """Total log return to the horizon, one per path."""
        return np.log(self.terminal / self.spot)

    def quantile_bands(
        self, levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    ) -> dict[float, NDArray[np.float64]]:
        """Cross-sectional quantiles at each step -- the fan chart.

        Note these are *pointwise* quantiles, not a simultaneous band: the 5%
        line is the 5th percentile at each step separately, so a single path is
        far more likely to breach it somewhere than 5%. That distinction is why
        the drawdown and first-passage probabilities are computed from the paths
        themselves rather than read off these bands.
        """
        return {level: np.quantile(self.paths, level, axis=0) for level in levels}

    def running_extremes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Running maximum and minimum along each path, for path-dependent stats."""
        return np.maximum.accumulate(self.paths, axis=1), np.minimum.accumulate(self.paths, axis=1)

    def summary(self) -> str:
        lines = [
            f"{self.n_paths:,} paths, {self.horizon} steps, engine '{self.engine}'",
            f"  spot {self.spot:,.4f}",
        ]
        bands = self.quantile_bands()
        lines.append(
            f"  terminal 5/50/95%: {bands[0.05][-1]:,.2f} / "
            f"{bands[0.50][-1]:,.2f} / {bands[0.95][-1]:,.2f}"
        )
        for key, value in self.assumptions.items():
            shown = f"{value:.4f}" if isinstance(value, float) else value
            lines.append(f"  {key}: {shown}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def _innovations(
    rng: np.random.Generator, distribution: str, df: float | None, size: tuple[int, ...]
) -> tuple[NDArray[np.float64], float]:
    """Unit-variance innovations for the requested distribution.

    Returns ``(draws, effective_df)`` where the df is ``inf`` for a Gaussian.

    This wrapper exists because the first version chose innovations with
    ``df = fitted.df if fitted.df else 5.0``, which inverted the two cases: a
    ``normal`` fit has no ``df``, so it fell through to ``t(5)`` and produced
    *fatter* tails than the ``t`` fit, whose maximum-likelihood df on
    near-Gaussian data is large. Measured on a clustered series, asking for
    ``normal`` gave kurtosis 5.11 and asking for ``t`` gave 3.68 -- exactly
    backwards, and entirely silent.
    """
    if distribution == "normal":
        return rng.standard_normal(size), float("inf")
    effective = float(df) if df and df > 2.0 else 5.0
    return _standardised_t(rng, effective, size), effective


def _standardised_t(
    rng: np.random.Generator, df: float, size: tuple[int, ...]
) -> NDArray[np.float64]:
    r"""Student-t innovations rescaled to unit variance.

    A raw :math:`t_\nu` has variance :math:`\nu/(\nu-2)`, so feeding it straight
    into a GARCH recursion inflates every simulated volatility by that factor --
    32% too high at four degrees of freedom. Dividing by
    :math:`\sqrt{\nu/(\nu-2)}` keeps the fat tails while leaving the variance
    equal to the one the model estimated, which is the whole point of fitting it.
    """
    if df <= 2.0:
        # Variance is infinite at or below 2; there is nothing to standardise to.
        raise ValueError(f"degrees of freedom must exceed 2 to have finite variance, got {df}")
    raw = rng.standard_t(df, size=size)
    return np.asarray(raw / np.sqrt(df / (df - 2.0)), dtype=float)


def simulate_garch_paths(
    returns: NDArray[np.float64],
    spot: float,
    horizon: int,
    *,
    n_paths: int = 10_000,
    drift: float = 0.0,
    distribution: str = "t",
    seed: int = 20240719,
) -> PathEnsemble:
    """Simulate paths from a GARCH(1,1) fitted to ``returns``.

    Purpose
        Produce a forward distribution that reproduces volatility clustering and
        fat tails, which a random walk with constant variance does not.
    Inputs
        ``returns`` -- historical log returns. ``horizon`` -- steps to simulate.
        ``drift`` -- per-step drift; zero by default, deliberately (see module
        docstring).
    Outputs
        A :class:`PathEnsemble`.
    Method
        Fit by maximum likelihood, then iterate the variance recursion forward,
        seeding it with the last conditional variance rather than the
        unconditional one -- starting from the unconditional variance would throw
        away the model's entire short-horizon information.
    Failure modes
        If the fit does not converge, or the series shows no ARCH effects at all,
        the ensemble falls back to constant volatility and records that in
        ``notes``. Simulating from an unconverged GARCH would look sophisticated
        and be worse than the simple thing.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> r = rng.normal(0, 0.02, 1500)
        >>> ensemble = simulate_garch_paths(r, 100.0, 21, n_paths=2000)
        >>> ensemble.paths.shape
        (2000, 22)
        >>> bool(np.all(ensemble.paths[:, 0] == 100.0))
        True
    """
    from quantos.core.stats.hypothesis import engle_arch
    from quantos.core.timeseries.garch import fit_garch

    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 100:
        raise ValueError(f"need at least 100 returns to fit a variance model, got {returns.size}")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")

    rng = np.random.default_rng(seed)
    notes: list[str] = []
    centred = returns - float(np.mean(returns))

    # Only fit GARCH if there is clustering to model. Fitting it to a series
    # without ARCH effects produces persistence near 1.0 and half-lives of
    # centuries -- the failure this gate exists to prevent.
    arch = engle_arch(centred, lags=5)
    use_garch = bool(arch.p_value < 0.05)
    if not use_garch:
        notes.append(
            f"no ARCH effects detected (Engle p={arch.p_value:.3f}); simulated at "
            "constant volatility rather than fitting a variance model to noise"
        )

    fitted = None
    if use_garch:
        try:
            fitted = fit_garch(centred, distribution=distribution)
            if not fitted.converged:
                notes.append("the GARCH fit did not converge; using constant volatility")
                fitted = None
        except (ValueError, np.linalg.LinAlgError) as error:
            notes.append(f"GARCH fit failed ({error}); using constant volatility")
            fitted = None

    shape = (n_paths, horizon)
    if fitted is None:
        sigma = float(np.std(centred, ddof=1))
        innovations, df = _innovations(rng, distribution, None, shape)
        steps = drift + sigma * innovations
        assumptions: dict[str, float | str] = {
            "engine": "constant volatility, Student-t innovations",
            "daily_volatility": sigma,
            "annualised_volatility": sigma * np.sqrt(TRADING_DAYS),
            "innovation_df": df,
            "drift_per_step": drift,
        }
    else:
        omega, alpha, beta = fitted.omega, fitted.alpha, fitted.beta
        innovations, df = _innovations(rng, distribution, fitted.df, shape)

        # Seed from the LAST conditional variance and the last shock, so the
        # simulation begins in the volatility state the market is actually in.
        variance = np.full(n_paths, float(fitted.conditional_variance[-1]))
        last_shock = float(centred[-1])
        previous_squared = np.full(n_paths, last_shock**2)

        steps = np.empty(shape, dtype=float)
        for step in range(horizon):
            variance = omega + alpha * previous_squared + beta * variance
            sigma_t = np.sqrt(variance)
            shock = sigma_t * innovations[:, step]
            steps[:, step] = drift + shock
            previous_squared = shock**2

        assumptions = {
            "engine": f"GARCH(1,1) with {distribution} innovations",
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": fitted.persistence,
            "innovation_df": df,
            "starting_annualised_volatility": float(
                np.sqrt(fitted.conditional_variance[-1] * TRADING_DAYS)
            ),
            "drift_per_step": drift,
        }
        if fitted.persistence > 0.99:
            notes.append(
                f"persistence is {fitted.persistence:.4f}, so shocks decay very slowly and "
                "long-horizon variance is sensitive to the fit; treat the far end of the "
                "fan with caution"
            )

    log_paths = np.cumsum(steps, axis=1)
    prices = spot * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1))

    return PathEnsemble(
        paths=prices,
        spot=float(spot),
        horizon=int(horizon),
        engine="garch" if fitted is not None else "constant-vol",
        assumptions=assumptions,
        notes=notes,
    )


def simulate_bootstrap_paths(
    returns: NDArray[np.float64],
    spot: float,
    horizon: int,
    *,
    n_paths: int = 10_000,
    block: int = 21,
    drift: float | None = None,
    seed: int = 20240719,
) -> PathEnsemble:
    r"""Simulate paths by resampling contiguous blocks of historical returns.

    Purpose
        A forward distribution that assumes **no model**: no distributional form,
        no variance equation, no parameters to misfit.
    Why blocks rather than single returns
        Resampling individual days destroys volatility clustering -- the
        resampled series is i.i.d. by construction, so it cannot produce the
        sustained turbulent stretches that dominate real drawdowns, and every
        path-dependent probability comes out too low. Blocks of about a month
        preserve clustering within a block.
    Inputs
        ``drift`` -- ``None`` (the default) removes the historical mean, so the
        simulation is driftless. Passing ``0.0`` keeps the sample mean, which
        embeds ten years of realised drift as a forecast; that is usually not
        what a caller wants and never what they want by accident.
    Failure modes
        Cannot generate a move larger than the largest in the sample. A ten-year
        window that happens to exclude a crash produces a forward distribution
        that thinks crashes do not happen.
    The block length is a real trade-off, not a free parameter
        Terminal variance is governed by the sample's *realised* autocorrelation
        over the block, not by its variance alone. Individually negligible
        autocorrelations accumulate: on 3,000 simulated Gaussian returns whose
        per-lag autocorrelations were all within :math:`\pm 0.03`, the sum
        :math:`\sum_k (1 - k/b)\rho_k` came to :math:`-0.082` at :math:`b = 21`,
        predicting a variance factor of :math:`1 + 2(-0.082) = 0.835` -- against a
        measured 0.838.

        So a 21-day block reproduced only 84% of the horizon variance that i.i.d.
        scaling implies, purely from sampling noise in the history. Longer blocks
        preserve volatility clustering better and track horizon variance worse.
        Neither end is correct in general, which is why
        :func:`~quantos.forecast.paths.compare_engines` exists: if the parametric
        and bootstrap answers agree, the block length did not matter.

    Example
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> r = rng.normal(0.0005, 0.015, 2000)
        >>> ensemble = simulate_bootstrap_paths(r, 50.0, 63, n_paths=1500)
        >>> ensemble.paths.shape
        (1500, 64)
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 100:
        raise ValueError(f"need at least 100 returns to bootstrap, got {returns.size}")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")

    block = int(np.clip(block, 1, max(1, returns.size // 4)))
    sample = returns.copy()
    notes: list[str] = []

    if drift is None:
        sample = sample - float(np.mean(sample))
        drift_note = "historical mean removed (driftless)"
        applied_drift = 0.0
    else:
        applied_drift = float(drift)
        drift_note = f"drift of {applied_drift:.6f} per step imposed by the caller"

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    # One start index per block per path, drawn with replacement.
    starts = rng.integers(0, sample.size - block + 1, size=(n_paths, n_blocks))
    offsets = np.arange(block)
    drawn = sample[starts[:, :, None] + offsets[None, None, :]]
    steps = drawn.reshape(n_paths, n_blocks * block)[:, :horizon] + applied_drift

    log_paths = np.cumsum(steps, axis=1)
    prices = spot * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1))

    notes.append(drift_note)
    notes.append(
        f"cannot produce a single-step move beyond the sample extremes "
        f"({np.min(sample):+.2%} to {np.max(sample):+.2%})"
    )

    return PathEnsemble(
        paths=prices,
        spot=float(spot),
        horizon=int(horizon),
        engine="block-bootstrap",
        assumptions={
            "engine": f"block bootstrap, block={block}",
            "block_length": float(block),
            "n_historical_returns": float(sample.size),
            "sample_annualised_volatility": float(np.std(sample, ddof=1) * np.sqrt(TRADING_DAYS)),
            "drift_per_step": applied_drift,
        },
        notes=notes,
    )


def compare_engines(
    returns: NDArray[np.float64],
    spot: float,
    horizon: int,
    *,
    n_paths: int = 10_000,
    seed: int = 20240719,
) -> dict[str, object]:
    """Run both engines and report where they disagree.

    Agreement means the answer does not depend on the model, which is the
    strongest thing that can be said about a simulated probability. Disagreement
    is not averaged away: it is reported, because it localises exactly which
    assumption a conclusion rests on.
    """
    garch = simulate_garch_paths(returns, spot, horizon, n_paths=n_paths, seed=seed)
    boot = simulate_bootstrap_paths(returns, spot, horizon, n_paths=n_paths, seed=seed)

    levels = (0.05, 0.25, 0.50, 0.75, 0.95)
    garch_terminal = np.quantile(garch.terminal, levels)
    boot_terminal = np.quantile(boot.terminal, levels)
    relative = np.abs(garch_terminal - boot_terminal) / spot

    verdict = (
        "the two engines agree closely; this result does not depend on the variance model"
        if float(np.max(relative)) < 0.03
        else "the engines disagree materially, so the result depends on the variance "
        "model — the GARCH tails are usually the wider of the two"
    )
    return {
        "garch": garch,
        "bootstrap": boot,
        "levels": levels,
        "garch_terminal_quantiles": garch_terminal,
        "bootstrap_terminal_quantiles": boot_terminal,
        "max_relative_gap": float(np.max(relative)),
        "verdict": verdict,
    }
