r"""Logging that a research tool can actually use.

The repository had none. For a library that is mostly pure functions over arrays
that was defensible; it stopped being defensible once there were network fetches,
an on-disk cache, a hash-chained ledger and a CLI that silently swallowed
failures. Three call sites caught a broad ``Exception`` and continued, and a
reader of the output could not tell a clean run from a degraded one.

Two decisions worth stating.

**The library never configures logging.** It attaches a ``NullHandler`` and
emits records; whether anything is displayed is the application's choice.
A library that calls ``basicConfig`` at import steals the root logger from
whatever imported it, which is why importing some packages silently changes
another package's log format.

**Numbers go in the record, not the message.** ``logger.info("fetched %s bars",
n)`` is greppable and machine-readable in a way that an f-string is not, and it
costs nothing when the level is disabled because the interpolation never runs.

Example
    >>> logger = get_logger(__name__)
    >>> logger.name
    'quantos.core.logging'
    >>> configure_logging("warning")            # an application would call this
    >>> get_logger("quantos").level             # doctest: +SKIP
    30
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["configure_logging", "get_logger", "log_duration"]

#: The one logger the whole package hangs off. Configuring `quantos` configures
#: everything without touching the root logger.
ROOT = "quantos"

#: Environment variable an operator can set instead of calling configure_logging.
LEVEL_VARIABLE = "QUANTOS_LOG_LEVEL"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``quantos`` root, with a NullHandler attached.

    Args:
        name: usually ``__name__``.

    The NullHandler is what stops Python emitting "No handlers could be found"
    and, more importantly, what stops the library printing anything at all
    unless the application asked for it.
    """
    logger = logging.getLogger(name if name.startswith(ROOT) else f"{ROOT}.{name}")
    root = logging.getLogger(ROOT)
    if not any(isinstance(handler, logging.NullHandler) for handler in root.handlers):
        root.addHandler(logging.NullHandler())
    return logger


def configure_logging(
    level: str | int | None = None,
    *,
    stream: Any = None,
    force: bool = False,
) -> None:
    """Attach a human-readable handler to the ``quantos`` logger.

    Applications call this. Libraries must not, which is why nothing in
    ``src/quantos`` outside the CLI does.

    Args:
        level: ``"debug"``, ``"info"``, ``"warning"``, ``"error"``, or a numeric
            level. ``None`` reads ``QUANTOS_LOG_LEVEL`` and falls back to
            ``"warning"`` -- quiet by default, because a research tool that
            chatters at INFO buries the report it was asked to produce.
        stream: destination. Defaults to stderr, so logs never contaminate a
            report on stdout that a caller may be piping.
        force: replace existing handlers rather than returning early.
    """
    global _configured  # noqa: PLW0603 - module-level idempotency flag, by design
    if _configured and not force:
        return

    if level is None:
        level = os.environ.get(LEVEL_VARIABLE, "warning")
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(
                f"unknown log level {level!r}; use debug, info, warning, error or critical"
            )
        level = resolved

    logger = logging.getLogger(ROOT)
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s  %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    # Do not hand records to the root logger as well: an application that has
    # configured its own root handler would otherwise see everything twice.
    logger.propagate = False
    _configured = True


@contextmanager
def log_duration(logger: logging.Logger, what: str, **fields: Any) -> Iterator[None]:
    """Time a block and log how long it took, at DEBUG.

    Used around the operations that are slow enough for a user to wonder whether
    the process has hung -- a network fetch, a 20,000-path simulation, a
    12,000-step lattice.

    Example
        >>> import logging
        >>> logger = get_logger("quantos.example")
        >>> with log_duration(logger, "nothing", n=3):
        ...     pass
    """
    import time

    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        if fields:
            detail = " ".join(f"{key}={value}" for key, value in fields.items())
            logger.debug("%s took %.3fs (%s)", what, elapsed, detail)
        else:
            logger.debug("%s took %.3fs", what, elapsed)
