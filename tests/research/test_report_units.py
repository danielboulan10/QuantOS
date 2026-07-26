"""Unit consistency in the research report.

A dataclass field whose units differ from its sibling is a trap that costs
someone an afternoon and, worse, can pass review: the number looks plausible, it
is just 16x too small. ``sharpe_standard_error`` was exactly that -- stored
per-period beside an annualised ``sharpe`` -- until a caller forgot to convert
and reported a Sharpe of 0.51 as significant at 25 standard errors.

These tests pin the units so the trap cannot come back.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.research.instruments import AssetClass, Instrument
from quantos.research.report import generate_report

TRADING_DAYS = 252


def make_instrument(n: int = 2520, *, sigma: float = 0.02, mu: float = 0.0004, seed: int = 3):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    dates = np.datetime64("2015-01-02") + np.arange(n)
    return Instrument(
        symbol="TEST",
        asset_class=AssetClass.EQUITY,
        dates=dates,
        prices=prices,
        dividend_adjusted=True,
    )


def test_sharpe_standard_error_is_annualised_like_the_sharpe():
    """The two must be directly comparable without any conversion.

    For i.i.d. returns the annualised Sharpe standard error is approximately
    sqrt(periods_per_year / n). If the field were still per-period it would come
    back ~16x smaller, which this bound catches.
    """
    report = generate_report(make_instrument(), run_signals=False)
    expected = np.sqrt(TRADING_DAYS / report.n_returns)

    assert report.sharpe_standard_error == pytest.approx(expected, rel=0.35)
    # And decisively not the per-period value.
    assert report.sharpe_standard_error > 5 * (expected / np.sqrt(TRADING_DAYS))


def test_significance_uses_the_same_units_on_both_sides():
    report = generate_report(make_instrument(), run_signals=False)
    manual = abs(report.sharpe / report.sharpe_standard_error) > 1.96
    assert report.sharpe_is_significant is manual


def test_the_rendered_text_shows_the_field_unchanged():
    """The renderer must not re-apply a conversion the field already carries."""
    from quantos.research.render import render_text

    report = generate_report(make_instrument(), run_signals=False)
    text = render_text(report)
    assert f"{report.sharpe_standard_error:.3f}" in text


def test_standard_error_shrinks_with_the_square_root_of_sample_size():
    short = generate_report(make_instrument(n=630), run_signals=False)
    long = generate_report(make_instrument(n=2520), run_signals=False)
    ratio = short.sharpe_standard_error / long.sharpe_standard_error
    assert ratio == pytest.approx(2.0, rel=0.35)


@pytest.mark.parametrize("mu", [0.0, 0.00008, 0.00017, 0.0004, 0.0009])
def test_the_verdict_tracks_the_two_sigma_threshold(mu):
    """Across drifts, the verdict must be exactly the 1.96-sigma rule.

    Asserting a specific verdict for a specific drift would encode whatever the
    fixture happened to produce -- an earlier version of this test did that and
    failed because a Sharpe of 0.698 over ten years genuinely *is* significant at
    2.2 standard errors. What must hold is the relationship, not the outcome.
    """
    report = generate_report(make_instrument(mu=mu, sigma=0.02), run_signals=False)
    if not np.isfinite(report.sharpe_standard_error):
        pytest.skip("no standard error for this configuration")
    expected = abs(report.sharpe / report.sharpe_standard_error) > 1.96
    assert report.sharpe_is_significant is expected


def test_a_driftless_series_is_not_called_significant():
    """The case that must never come back: noise reported as skill.

    With the per-period standard error, a driftless series' Sharpe was divided by
    a number sqrt(252) too small, so pure noise cleared the threshold routinely.
    """
    verdicts = [
        generate_report(make_instrument(mu=0.0, sigma=0.02, seed=s), run_signals=False)
        for s in range(12)
    ]
    false_positives = sum(r.sharpe_is_significant for r in verdicts)
    assert false_positives <= 2, (
        f"{false_positives}/12 driftless series called significant; "
        "the standard error is too small again"
    )
