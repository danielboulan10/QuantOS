"""Validation for logging setup.

The properties that matter are about what the library does *not* do: it must not
print anything unless asked, must not touch the root logger, and must not send
records to stdout where they would contaminate a report someone is piping.
"""

from __future__ import annotations

import io
import logging

import pytest

from quantos.core import logging as quantos_logging
from quantos.core.logging import ROOT, configure_logging, get_logger, log_duration


@pytest.fixture(autouse=True)
def restore_logging():
    """Every test here mutates global logging state, so it is restored."""
    root = logging.getLogger(ROOT)
    saved_handlers = list(root.handlers)
    saved_level, saved_propagate = root.level, root.propagate
    saved_flag = quantos_logging._configured
    yield
    root.handlers = saved_handlers
    root.level, root.propagate = saved_level, saved_propagate
    quantos_logging._configured = saved_flag


def test_every_logger_hangs_off_the_one_package_root():
    """Configuring `quantos` must configure everything under it.

    A module that called getLogger(__name__) without the prefix would be
    unreachable from configure_logging and would silently ignore the level.
    """
    assert get_logger("quantos.data.market").name == "quantos.data.market"
    assert get_logger("data.market").name == "quantos.data.market"
    assert get_logger(ROOT).name == ROOT


def test_the_library_prints_nothing_until_an_application_configures_it():
    """A NullHandler is the difference between a library and a nuisance."""
    logger = get_logger("quantos.test.silent")
    root = logging.getLogger(ROOT)

    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)
    logger.warning("this must not reach anyone's terminal")


def test_configuring_does_not_touch_the_root_logger():
    """basicConfig at import steals the root logger from whatever imported us.

    That is why importing some packages silently changes another package's log
    format, and it is the reason nothing in src/quantos outside the CLI calls
    this function.
    """
    before = list(logging.getLogger().handlers)
    configure_logging("debug", stream=io.StringIO(), force=True)
    assert logging.getLogger().handlers == before


def test_records_go_to_the_given_stream_and_carry_their_level():
    stream = io.StringIO()
    configure_logging("info", stream=stream, force=True)

    get_logger("quantos.test.stream").warning("fetched %s bars", 2511)

    written = stream.getvalue()
    assert "WARNING" in written
    assert "quantos.test.stream" in written
    assert "fetched 2511 bars" in written, "the arguments must be interpolated"


def test_a_level_below_the_threshold_is_dropped():
    stream = io.StringIO()
    configure_logging("warning", stream=stream, force=True)

    logger = get_logger("quantos.test.threshold")
    logger.debug("invisible")
    logger.info("also invisible")
    logger.error("visible")

    written = stream.getvalue()
    assert "invisible" not in written
    assert "visible" in written


def test_configuring_twice_does_not_duplicate_every_record():
    """Without the idempotency flag, two calls attach two handlers and every
    line is logged twice -- which reads as a loop rather than a config bug."""
    stream = io.StringIO()
    configure_logging("info", stream=stream, force=True)
    configure_logging("info", stream=stream)  # no force: must be a no-op

    get_logger("quantos.test.once").info("single")
    assert stream.getvalue().count("single") == 1


def test_records_do_not_propagate_to_an_application_root_handler():
    """An application with its own root handler would otherwise see everything
    twice: once from ours, once from theirs."""
    ours, theirs = io.StringIO(), io.StringIO()
    configure_logging("info", stream=ours, force=True)

    root_handler = logging.StreamHandler(theirs)
    logging.getLogger().addHandler(root_handler)
    try:
        get_logger("quantos.test.propagate").info("mine")
    finally:
        logging.getLogger().removeHandler(root_handler)

    assert "mine" in ours.getvalue()
    assert "mine" not in theirs.getvalue()


def test_an_unknown_level_is_refused_with_the_valid_ones_named():
    """logging.getLevelName returns the string 'Level FOO' for an unknown name
    rather than raising, so a typo would otherwise set an unreachable level and
    silence everything."""
    with pytest.raises(ValueError, match="debug, info, warning"):
        configure_logging("verbose", stream=io.StringIO(), force=True)


def test_the_environment_variable_is_read_when_no_level_is_given(monkeypatch):
    monkeypatch.setenv("QUANTOS_LOG_LEVEL", "debug")
    stream = io.StringIO()
    configure_logging(stream=stream, force=True)

    get_logger("quantos.test.env").debug("from the environment")
    assert "from the environment" in stream.getvalue()


def test_the_default_is_quiet_because_a_report_should_not_be_buried(monkeypatch):
    monkeypatch.delenv("QUANTOS_LOG_LEVEL", raising=False)
    stream = io.StringIO()
    configure_logging(stream=stream, force=True)

    logger = get_logger("quantos.test.default")
    logger.info("chatter")
    assert stream.getvalue() == ""
    assert logging.getLogger(ROOT).level == logging.WARNING


def test_log_duration_reports_at_debug_and_survives_an_exception():
    """The timing must be emitted even when the block fails, or the slow path
    that failed is the one path with no timing."""
    stream = io.StringIO()
    configure_logging("debug", stream=stream, force=True)
    logger = get_logger("quantos.test.timing")

    with log_duration(logger, "fast thing", ticker="SPY"):
        pass
    assert "fast thing took" in stream.getvalue()
    assert "ticker=SPY" in stream.getvalue()

    with pytest.raises(RuntimeError), log_duration(logger, "failing thing"):
        raise RuntimeError("boom")
    assert "failing thing took" in stream.getvalue()
