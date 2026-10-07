"""Shared fixtures for the scheduler tests. Plain helpers in a conftest are not
importable without __init__ machinery - consume them as pytest fixtures only."""
from __future__ import annotations

import contextlib
import logging

import pytest


class LogCapture(logging.Handler):
    """Formatted-message collector. The freetoken module loggers do not propagate
    (init_logger sets propagate=False), so caplog is blind and a handler must be
    attached to the module logger directly."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def log_capture_factory():
    """`with tier_log_capture() as cap:` - the injected per-module binding calls this
    factory once with the logger name and gets a zero-arg context-manager function; each
    call attaches a FRESH handler for one with-block, so repeated or nested captures
    inside one test never share messages."""
    def factory(logger_name: str):
        @contextlib.contextmanager
        def capture():
            lg = logging.getLogger(logger_name)
            cap = LogCapture()
            lg.addHandler(cap)
            try:
                yield cap
            finally:
                lg.removeHandler(cap)

        return capture

    return factory
