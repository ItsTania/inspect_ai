import contextlib
from typing import Any, AsyncIterator, Callable, Coroutine, Iterator

from rich.console import Console

from ..core.display import (
    TR,
    Display,
    Progress,
    TaskDisplay,
    TaskDisplayMetric,
    TaskProfile,
    TaskResult,
    TaskScreen,
    TaskSpec,
    TaskWithResult,
)


class _CompositeProgress(Progress):
    def __init__(self, primary: Progress, secondary: Progress) -> None:
        self.primary = primary
        self.secondary = secondary

    def update(self, n: int = 1) -> None:
        self.primary.update(n)
        self.secondary.update(n)

    def complete(self) -> None:
        self.primary.complete()
        self.secondary.complete()


class _CompositeTaskDisplay(TaskDisplay):
    def __init__(self, primary: TaskDisplay, secondary: TaskDisplay) -> None:
        self.primary = primary
        self.secondary = secondary

    @contextlib.contextmanager
    def progress(self) -> Iterator[Progress]:
        with self.primary.progress() as primary_progress:
            with self.secondary.progress() as secondary_progress:
                yield _CompositeProgress(primary_progress, secondary_progress)

    def sample_complete(self, complete: int, total: int) -> None:
        self.primary.sample_complete(complete, total)
        self.secondary.sample_complete(complete, total)

    def update_metrics(self, scores: list[TaskDisplayMetric]) -> None:
        self.primary.update_metrics(scores)
        self.secondary.update_metrics(scores)

    def complete(self, result: TaskResult) -> None:
        self.primary.complete(result)
        self.secondary.complete(result)


class _CompositeTaskScreen(TaskScreen):
    def __init__(self, primary: TaskScreen, secondary: TaskScreen) -> None:
        self.primary = primary
        self.secondary = secondary

    @contextlib.contextmanager
    def input_screen(
        self,
        header: str | None = None,
        transient: bool | None = None,
        width: int | None = None,
    ) -> Iterator[Console]:
        with self.primary.input_screen(header, transient, width) as console:
            yield console


class CompositeDisplay(Display):
    def __init__(self, primary: Display, secondary: Display) -> None:
        self.primary = primary
        self.secondary = secondary

    def print(self, message: str) -> None:
        self.primary.print(message)
        self.secondary.print(message)

    @contextlib.contextmanager
    def progress(self, total: int) -> Iterator[Progress]:
        with self.primary.progress(total) as primary_progress:
            with self.secondary.progress(total) as secondary_progress:
                yield _CompositeProgress(primary_progress, secondary_progress)

    def run_task_app(self, main: Callable[[], Coroutine[None, None, TR]]) -> TR:
        return self.primary.run_task_app(main)

    @contextlib.contextmanager
    def suspend_task_app(self) -> Iterator[None]:
        with self.primary.suspend_task_app():
            yield

    @contextlib.asynccontextmanager
    async def task_screen(
        self, tasks: list[TaskSpec], parallel: bool
    ) -> AsyncIterator[TaskScreen]:
        async with self.primary.task_screen(tasks, parallel) as primary_screen:
            async with self.secondary.task_screen(tasks, parallel) as secondary_screen:
                yield _CompositeTaskScreen(primary_screen, secondary_screen)

    @contextlib.contextmanager
    def task(self, profile: TaskProfile) -> Iterator[TaskDisplay]:
        with self.primary.task(profile) as primary_task:
            with self.secondary.task(profile) as secondary_task:
                yield _CompositeTaskDisplay(primary_task, secondary_task)

    def display_counter(self, caption: str, value: str) -> None:
        self.primary.display_counter(caption, value)
        self.secondary.display_counter(caption, value)
