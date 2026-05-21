import contextlib
from typing import AsyncIterator, Callable, Coroutine, Iterator
from unittest.mock import MagicMock

import pytest

from inspect_ai._display.composite.display import CompositeDisplay
from inspect_ai._display.core.active import _create_display, display
from inspect_ai._display.core.display import (
    TR,
    Display,
    Progress,
    TaskDisplay,
    TaskDisplayMetric,
    TaskProfile,
    TaskResult,
    TaskScreen,
    TaskSpec,
)
from inspect_ai._display.log.display import LogDisplay

# -- Stub display for testing delegation ---------------------------------


class _StubProgress(Progress):
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.completed = False

    def update(self, n: int = 1) -> None:
        self.updates.append(n)

    def complete(self) -> None:
        self.completed = True


class _StubTaskDisplay(TaskDisplay):
    def __init__(self) -> None:
        self.sample_calls: list[tuple[int, int]] = []
        self.metric_calls: list[list[TaskDisplayMetric]] = []
        self.complete_calls: list[TaskResult] = []
        self._progress = _StubProgress()

    @contextlib.contextmanager
    def progress(self) -> Iterator[Progress]:
        yield self._progress

    def sample_complete(self, complete: int, total: int) -> None:
        self.sample_calls.append((complete, total))

    def update_metrics(self, scores: list[TaskDisplayMetric]) -> None:
        self.metric_calls.append(scores)

    def complete(self, result: TaskResult) -> None:
        self.complete_calls.append(result)


class _StubDisplay(Display):
    def __init__(self) -> None:
        self.print_calls: list[str] = []
        self.counter_calls: list[tuple[str, str]] = []
        self.task_display = _StubTaskDisplay()
        self._progress = _StubProgress()

    def print(self, message: str) -> None:
        self.print_calls.append(message)

    @contextlib.contextmanager
    def progress(self, total: int) -> Iterator[Progress]:
        yield self._progress

    def run_task_app(self, main: Callable[[], Coroutine[None, None, TR]]) -> TR:
        raise NotImplementedError

    @contextlib.contextmanager
    def suspend_task_app(self) -> Iterator[None]:
        yield

    @contextlib.asynccontextmanager
    async def task_screen(
        self, tasks: list[TaskSpec], parallel: bool
    ) -> AsyncIterator[TaskScreen]:
        yield TaskScreen()

    @contextlib.contextmanager
    def task(self, profile: TaskProfile) -> Iterator[TaskDisplay]:
        yield self.task_display

    def display_counter(self, caption: str, value: str) -> None:
        self.counter_calls.append((caption, value))


# -- CompositeDisplay delegation tests -----------------------------------


def test_print_delegates_to_both() -> None:
    primary = _StubDisplay()
    secondary = _StubDisplay()
    composite = CompositeDisplay(primary, secondary)

    composite.print("hello")

    assert primary.print_calls == ["hello"]
    assert secondary.print_calls == ["hello"]


def test_display_counter_delegates_to_both() -> None:
    primary = _StubDisplay()
    secondary = _StubDisplay()
    composite = CompositeDisplay(primary, secondary)

    composite.display_counter("rate limits", "42")

    assert primary.counter_calls == [("rate limits", "42")]
    assert secondary.counter_calls == [("rate limits", "42")]


def test_progress_delegates_to_both() -> None:
    primary = _StubDisplay()
    secondary = _StubDisplay()
    composite = CompositeDisplay(primary, secondary)

    with composite.progress(100) as progress:
        progress.update(5)
        progress.complete()

    assert primary._progress.updates == [5]
    assert primary._progress.completed
    assert secondary._progress.updates == [5]
    assert secondary._progress.completed


def test_task_display_delegates_to_both() -> None:
    primary = _StubDisplay()
    secondary = _StubDisplay()
    composite = CompositeDisplay(primary, secondary)

    profile = MagicMock(spec=TaskProfile)
    with composite.task(profile) as task_display:
        task_display.sample_complete(1, 10)
        task_display.update_metrics([])

    assert primary.task_display.sample_calls == [(1, 10)]
    assert secondary.task_display.sample_calls == [(1, 10)]
    assert primary.task_display.metric_calls == [[]]
    assert secondary.task_display.metric_calls == [[]]


# -- CLI / env var integration tests -------------------------------------


def test_display_secondary_env_creates_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When INSPECT_DISPLAY_SECONDARY=log, display() returns a CompositeDisplay."""
    import inspect_ai._display.core.active as active_module

    monkeypatch.setattr(active_module, "_active_display", None)
    monkeypatch.setenv("INSPECT_DISPLAY_SECONDARY", "log")
    monkeypatch.setenv("INSPECT_DISPLAY", "plain")

    result = display()

    assert isinstance(result, CompositeDisplay)
    assert isinstance(result.secondary, LogDisplay)

    # cleanup
    monkeypatch.setattr(active_module, "_active_display", None)


def test_display_no_secondary_env_returns_plain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without INSPECT_DISPLAY_SECONDARY, display() returns a plain Display."""
    import inspect_ai._display.core.active as active_module

    monkeypatch.setattr(active_module, "_active_display", None)
    monkeypatch.delenv("INSPECT_DISPLAY_SECONDARY", raising=False)
    monkeypatch.setenv("INSPECT_DISPLAY", "plain")

    result = display()

    assert not isinstance(result, CompositeDisplay)

    # cleanup
    monkeypatch.setattr(active_module, "_active_display", None)


def test_display_secondary_none_returns_plain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INSPECT_DISPLAY_SECONDARY=none should not create a composite."""
    import inspect_ai._display.core.active as active_module

    monkeypatch.setattr(active_module, "_active_display", None)
    monkeypatch.setenv("INSPECT_DISPLAY_SECONDARY", "none")
    monkeypatch.setenv("INSPECT_DISPLAY", "plain")

    result = display()

    assert not isinstance(result, CompositeDisplay)

    # cleanup
    monkeypatch.setattr(active_module, "_active_display", None)


def test_create_display_log_returns_log_display() -> None:
    result = _create_display("log")
    assert isinstance(result, LogDisplay)


def test_create_display_plain_returns_plain_display() -> None:
    from inspect_ai._display.plain.display import PlainDisplay

    result = _create_display("plain")
    assert isinstance(result, PlainDisplay)


def test_composite_secondary_log_receives_logger_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The secondary LogDisplay picks up INSPECT_LOG_DISPLAY_PY_LOGGER."""
    import inspect_ai._display.core.active as active_module

    monkeypatch.setattr(active_module, "_active_display", None)
    monkeypatch.setenv("INSPECT_DISPLAY_SECONDARY", "log")
    monkeypatch.setenv("INSPECT_DISPLAY", "plain")
    monkeypatch.setenv("INSPECT_LOG_DISPLAY_PY_LOGGER", "my.test.logger")

    result = display()

    assert isinstance(result, CompositeDisplay)
    assert isinstance(result.secondary, LogDisplay)
    assert result.secondary.logger.name == "my.test.logger"

    # cleanup
    monkeypatch.setattr(active_module, "_active_display", None)
