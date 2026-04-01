import contextlib
import logging
import os
import sys
from datetime import datetime, timezone
from typing import IO, AsyncIterator, Callable, Coroutine, Iterator

import rich
from rich.console import Console

from inspect_ai._util._async import configured_async_backend
from inspect_ai._util.thread import is_main_thread

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
)
from ..plain.display import PlainDisplay
from ..textual.display import TextualDisplay

# Cached so that FullLogDisplay and external callers (e.g. inspect_wandb hooks)
# always resolve to the same path within a single process invocation.
_resolved_log_path: str | None = None


def default_log_file_path(subdirectory: str = "display_tee") -> str:
    global _resolved_log_path
    if _resolved_log_path is None:
        """Generate a default log file path with timestamp."""
        base_log_dir = os.environ.get("INSPECT_LOG_DIR", "./logs")
        log_dir = os.path.join(base_log_dir, subdirectory)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        _resolved_log_path = os.path.join(log_dir, f"inspect_display_{timestamp}.log")
    return _resolved_log_path


class _TeeIO:
    """Write to both a primary stream and a secondary file."""

    def __init__(self, primary: IO[str], secondary: IO[str]) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._secondary.write(s)
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def __getattr__(self, name: str) -> object:
        return getattr(self._primary, name)


class _TeeProgress(Progress):
    """Forward progress updates to two Progress instances."""

    def __init__(self, primary: Progress, secondary: Progress) -> None:
        self._primary = primary
        self._secondary = secondary

    def update(self, n: int = 1) -> None:
        self._primary.update(n)
        self._secondary.update(n)

    def complete(self) -> None:
        self._primary.complete()
        self._secondary.complete()


class _TeeTaskDisplay(TaskDisplay):
    """Forward task display events to two TaskDisplay instances."""

    def __init__(self, primary: TaskDisplay, secondary: TaskDisplay) -> None:
        self._primary = primary
        self._secondary = secondary

    @contextlib.contextmanager
    def progress(self) -> Iterator[Progress]:
        with self._primary.progress() as primary_progress:
            with self._secondary.progress() as secondary_progress:
                yield _TeeProgress(primary_progress, secondary_progress)

    def sample_complete(self, complete: int, total: int) -> None:
        self._primary.sample_complete(complete, total)
        self._secondary.sample_complete(complete, total)

    def update_metrics(self, scores: list[TaskDisplayMetric]) -> None:
        self._primary.update_metrics(scores)
        self._secondary.update_metrics(scores)

    def complete(self, result: TaskResult) -> None:
        self._primary.complete(result)
        self._secondary.complete(result)


class FullLogDisplay(Display):
    """Display that shows the full UI on the terminal and tees log output to a file.

    Uses TextualDisplay (or PlainDisplay as fallback) for the terminal, and a
    LogDisplay with a file handler to write a plain-text log file simultaneously.
    """

    def __init__(self) -> None:
        # Primary display for the terminal
        if (
            sys.stdout.isatty()
            and not rich.get_console().is_jupyter
            and is_main_thread()
            and configured_async_backend() != "trio"
        ):
            self._primary: Display = TextualDisplay()
        else:
            self._primary = PlainDisplay()

        # Set up log file and secondary plain display writing to it
        self._log_file_path = default_log_file_path()
        self._log_file = open(self._log_file_path, "w", buffering=1, encoding="utf-8")
        file_console = Console(
            file=self._log_file, force_terminal=False, no_color=True, highlight=False
        )
        self._log_display = PlainDisplay(console=file_console)

        # Capture Python logging output to the same file
        self._log_handler = logging.StreamHandler(self._log_file)
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(self._log_handler)

        # Tee stderr to the log file for external library output
        self._original_stderr = sys.stderr
        sys.stderr = _TeeIO(self._original_stderr, self._log_file)  # type: ignore[assignment]

    def _cleanup(self) -> None:
        """Restore stderr, remove log handler, and close the log file."""
        sys.stderr = self._original_stderr
        logging.getLogger().removeHandler(self._log_handler)
        self._log_file.close()

    def print(self, message: str) -> None:
        self._primary.print(message)
        self._log_display.print(message)

    @contextlib.contextmanager
    def progress(self, total: int) -> Iterator[Progress]:
        with self._primary.progress(total) as primary_progress:
            with self._log_display.progress(total) as log_progress:
                yield _TeeProgress(primary_progress, log_progress)

    def run_task_app(self, main: Callable[[], Coroutine[None, None, TR]]) -> TR:
        return self._primary.run_task_app(main)

    @contextlib.contextmanager
    def suspend_task_app(self) -> Iterator[None]:
        with self._primary.suspend_task_app():
            yield

    @contextlib.asynccontextmanager
    async def task_screen(
        self, tasks: list[TaskSpec], parallel: bool
    ) -> AsyncIterator[TaskScreen]:
        try:
            async with self._primary.task_screen(tasks, parallel) as screen:
                async with self._log_display.task_screen(tasks, parallel):
                    yield screen
        finally:
            self._cleanup()

    @contextlib.contextmanager
    def task(self, profile: TaskProfile) -> Iterator[TaskDisplay]:
        with self._primary.task(profile) as primary_td:
            with self._log_display.task(profile) as log_td:
                yield _TeeTaskDisplay(primary_td, log_td)

    def display_counter(self, caption: str, value: str) -> None:
        self._primary.display_counter(caption, value)
        self._log_display.display_counter(caption, value)
