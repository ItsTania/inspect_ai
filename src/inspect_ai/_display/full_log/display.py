"""display='full_log': TextualDisplay (= display='full') + simultaneous plain-text log file.

Writes plain progress lines to a file so that external tools (e.g. wandb) can
stream or upload the log without seeing ANSI escape codes from the TUI renderer.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import IO, AsyncIterator, Callable, Coroutine, Iterator

import rich
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
)
from ..plain.display import PlainDisplay
from ..textual.display import TextualDisplay

# ---------------------------------------------------------------------------
# Public helpers (read by active.py and inspect_wandb)
# ---------------------------------------------------------------------------

# Cached so that FullLogDisplay and external callers (e.g. inspect_wandb hooks)
# always resolve to the same path within a single process invocation.
_resolved_log_path: str | None = None


def default_plain_log_path() -> str:
    global _resolved_log_path
    if _resolved_log_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        hash_suffix = uuid.uuid4().hex[:8]
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        _resolved_log_path = os.path.join(logs_dir, f"terminal_log_{timestamp}_{hash_suffix}.txt")
    return _resolved_log_path


# ---------------------------------------------------------------------------
# Tee helpers: forward every lifecycle event to two TaskDisplays simultaneously
# ---------------------------------------------------------------------------


class _TeeProgress(Progress):
    def __init__(self, a: Progress, b: Progress) -> None:
        self._a = a
        self._b = b

    def update(self, n: int = 1) -> None:
        self._a.update(n)
        self._b.update(n)

    def complete(self) -> None:
        self._a.complete()
        self._b.complete()


class _TeeTaskDisplay(TaskDisplay):
    def __init__(self, a: TaskDisplay, b: TaskDisplay) -> None:
        self._a = a
        self._b = b

    @contextlib.contextmanager
    def progress(self) -> Iterator[Progress]:
        with self._a.progress() as pa, self._b.progress() as pb:
            yield _TeeProgress(pa, pb)

    def sample_complete(self, complete: int, total: int) -> None:
        self._a.sample_complete(complete, total)
        self._b.sample_complete(complete, total)

    def update_metrics(self, metrics: list[TaskDisplayMetric]) -> None:
        self._a.update_metrics(metrics)
        self._b.update_metrics(metrics)

    def complete(self, result: TaskResult) -> None:
        self._a.complete(result)
        self._b.complete(result)


# ---------------------------------------------------------------------------
# FullLogDisplay: display='full' for the terminal, PlainDisplay for the file
# ---------------------------------------------------------------------------


class FullLogDisplay(Display):
    """display='full' TUI on the terminal + simultaneous plain-text log file."""

    def __init__(self) -> None:
        # Terminal side: Textual TUI when a real TTY is available, else plain text.
        if sys.stdout.isatty() and not rich.get_console().is_jupyter:
            self._full: Display = TextualDisplay()
        else:
            self._full = PlainDisplay()

        # File side: PlainDisplay writing to a line-buffered file with no ANSI codes.
        self.log_path: str = default_plain_log_path()
        self._log_file: IO[str] = open(self.log_path, "w", buffering=1)
        file_console = Console(file=self._log_file, force_terminal=False, no_color=True, highlight=False)
        self._sink = PlainDisplay(console=file_console)

        # Also capture Python logging output to the same file.
        self._log_handler = logging.FileHandler(self.log_path, mode="a")
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(self._log_handler)

    # ------------------------------------------------------------------
    # Display protocol — delegate terminal interactions to self._full
    # ------------------------------------------------------------------

    def print(self, message: str) -> None:
        self._full.print(message)

    @contextlib.contextmanager
    def progress(self, total: int) -> Iterator[Progress]:
        with self._full.progress(total) as p:
            yield p

    def run_task_app(self, main: Callable[[], Coroutine[None, None, TR]]) -> TR:
        return self._full.run_task_app(main)

    @contextlib.contextmanager
    def suspend_task_app(self) -> Iterator[None]:
        with self._full.suspend_task_app():
            yield

    def display_counter(self, caption: str, value: str) -> None:
        self._full.display_counter(caption, value)

    # ------------------------------------------------------------------
    # task_screen — run both displays; file sink writes header + final results
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def task_screen(
        self, tasks: list[TaskSpec], parallel: bool
    ) -> AsyncIterator[TaskScreen]:
        try:
            async with self._full.task_screen(tasks, parallel) as screen:
                async with self._sink.task_screen(tasks, parallel):
                    yield screen
        finally:
            self._log_file.flush()
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()

    # ------------------------------------------------------------------
    # task — tee task events to both the TUI and the file sink
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def task(self, profile: TaskProfile) -> Iterator[TaskDisplay]:
        with self._full.task(profile) as full_td, self._sink.task(profile) as sink_td:
            yield _TeeTaskDisplay(full_td, sink_td)
