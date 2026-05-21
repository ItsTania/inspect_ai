import logging

import pytest

from inspect_ai._display.log.display import LogDisplay


def test_log_display_uses_named_logger() -> None:
    named_logger = logging.getLogger("test.custom.logger")
    display = LogDisplay(logger=named_logger)
    assert display.logger is named_logger


def test_log_display_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_LOG_DISPLAY_PY_LOGGER", "my.env.logger")
    display = LogDisplay()
    assert display.logger.name == "my.env.logger"


def test_log_display_defaults_to_root_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INSPECT_LOG_DISPLAY_PY_LOGGER", raising=False)
    display = LogDisplay()
    assert display.logger is logging.getLogger()
