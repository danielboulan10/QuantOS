"""CLI smoke tests: every subcommand must run and exit zero."""

from __future__ import annotations

import pytest

from quantos.cli.main import build_parser, main


def test_parser_exposes_every_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # a subcommand is required


@pytest.mark.parametrize(
    "argv",
    [
        ["options"],
        ["options", "--spot", "95", "--strike", "105", "--volatility", "0.35"],
        ["book", "--operations", "20000"],
        ["probability", "--samples", "8000"],
        ["probability", "--samples", "8000", "--problem", "stick"],
        ["validate", "--configurations", "40"],
        ["portfolio", "--assets", "20", "--train", "60"],
        ["execution"],
        ["doctor"],
    ],
)
def test_subcommand_runs_and_exits_zero(argv: list[str], capsys) -> None:
    assert main(argv) == 0
    captured = capsys.readouterr().out
    assert len(captured) > 100
    # No unformatted numpy reprs or tracebacks leaking into user-facing output.
    assert "Traceback" not in captured
    assert "np.float64" not in captured


@pytest.mark.slow
def test_simulate_runs_and_writes_charts(tmp_path, capsys) -> None:
    assert main(["simulate", "--seconds", "2", "--seed", "3", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "price_discovery.svg").exists()
    assert (tmp_path / "return_distribution.svg").exists()
    out = capsys.readouterr().out
    assert "Price discovery" in out
    assert "Stylised facts" in out


def test_unknown_scenario_returns_an_error_code(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["simulate", "--scenario", "nonsense"])


def test_probability_filter_that_matches_nothing_returns_two(capsys) -> None:
    assert main(["probability", "--problem", "no-such-problem"]) == 2


def test_probability_applies_a_multiple_testing_correction(capsys) -> None:
    """Ten simultaneous 99% intervals would fail together ~10% of the time.

    The command Bonferroni-corrects its own confidence level, which is both the
    statistically correct thing to do and consistent with what the library
    preaches in quantos.core.stats.multipletest.
    """
    assert main(["probability", "--samples", "8000"]) == 0
    out = capsys.readouterr().out
    assert "Bonferroni-corrected" in out
    assert "family-wise alpha" in out


def test_doctor_reports_every_module(capsys) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "modules import cleanly" in out
    assert "quantos.core.special" in out
    assert "FAIL" not in out
