import os
import sys
from contextvars import ContextVar

import rich

from inspect_ai.util._display import display_type

from ..composite.display import CompositeDisplay
from ..log.display import LogDisplay
from ..plain.display import PlainDisplay
from ..rich.display import RichDisplay
from ..textual.display import TextualDisplay
from .display import Display, TaskScreen

_active_display: Display | None = None


def active_display() -> Display | None:
    global _active_display
    return _active_display


def _create_display(dt: str) -> Display:
    if dt == "plain":
        return PlainDisplay()
    elif dt == "full" and sys.stdout.isatty() and not rich.get_console().is_jupyter:
        return TextualDisplay()
    elif dt == "log":
        return LogDisplay()
    else:
        return RichDisplay()


def display() -> Display:
    global _active_display
    if _active_display is None:
        _active_display = _create_display(display_type())

        # Use composite display option if INSPECT_DISPLAY_SECONDARY is set.
        secondary_type = os.environ.get("INSPECT_DISPLAY_SECONDARY")
        if secondary_type is not None:
            _active_display = CompositeDisplay(
                _active_display, _create_display(secondary_type)
            )

    return _active_display


def task_screen() -> TaskScreen:
    screen = _active_task_screen.get(None)
    if screen is None:
        raise RuntimeError(
            "console input function called outside of running evaluation."
        )
    return screen


def init_task_screen(screen: TaskScreen) -> None:
    _active_task_screen.set(screen)


def clear_task_screen() -> None:
    _active_task_screen.set(None)


_active_task_screen: ContextVar[TaskScreen | None] = ContextVar(
    "task_screen", default=None
)
