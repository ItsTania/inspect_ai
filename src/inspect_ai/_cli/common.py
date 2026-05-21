import functools
import os
from typing import Any, Callable, Literal, cast

import click
import rich
from typing_extensions import TypedDict

from inspect_ai._util.constants import (
    ALL_LOG_LEVELS,
    DEFAULT_DISPLAY,
    DEFAULT_LOG_LEVEL,
)
from inspect_ai._util.dotenv import init_cli_env
from inspect_ai.util._display import init_display_type

from .util import parse_cli_args


class CommonOptions(TypedDict):
    log_level: str
    log_dir: str
    display: Literal["full", "conversation", "rich", "plain", "none"]
    display_secondary: str | None
    log_display_py_logger: str | None
    no_ansi: bool | None
    traceback_locals: bool
    env: tuple[str, ...] | None
    debug: bool
    debug_port: int
    debug_errors: bool


def log_level_options(func: Callable[..., Any]) -> Callable[..., click.Context]:
    @click.option(
        "--log-level",
        type=click.Choice(
            [level.lower() for level in ALL_LOG_LEVELS],
            case_sensitive=False,
        ),
        default=DEFAULT_LOG_LEVEL,
        envvar="INSPECT_LOG_LEVEL",
        help=f"Set the log level (defaults to '{DEFAULT_LOG_LEVEL}')",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> click.Context:
        return cast(click.Context, func(*args, **kwargs))

    return wrapper


def common_options(func: Callable[..., Any]) -> Callable[..., click.Context]:
    @log_level_options
    @click.option(
        "--log-dir",
        type=str,
        default="./logs",
        callback=clean_log_dir,
        envvar="INSPECT_LOG_DIR",
        help="Directory for log files.",
    )
    @click.option(
        "--display",
        type=click.Choice(
            ["full", "conversation", "rich", "plain", "log", "none"],
            case_sensitive=False,
        ),
        default=DEFAULT_DISPLAY,
        envvar="INSPECT_DISPLAY",
        help="Set the display type (defaults to 'full')",
    )
    # Currently only "log" is supported as a secondary display as other display types (full, rich, conversation, plain) write to the terminal and would conflict with the primary display.
    # To add support of other secondary display types we would need to further understand usecase.
    @click.option(
        "--display-secondary",
        type=click.Choice(["log"], case_sensitive=False),
        default=None,
        envvar="INSPECT_DISPLAY_SECONDARY",
        help="Secondary display to run alongside the primary display. Only 'log' is supported.",
    )
    @click.option(
        "--log-display-py-logger",
        type=str,
        default=None,
        envvar="INSPECT_LOG_DISPLAY_PY_LOGGER",
        help="Name of the Python logger used by the log display (defaults to root logger).",
    )
    @click.option(
        "--no-ansi",
        type=bool,
        is_flag=True,
        hidden=True,
        help="Do not print ANSI control characters.",
        envvar="INSPECT_NO_ANSI",
    )
    @click.option(
        "--traceback-locals",
        type=bool,
        is_flag=True,
        envvar="INSPECT_TRACEBACK_LOCALS",
        help="Include values of local variables in tracebacks (note that this can leak private data e.g. API keys so should typically only be enabled for targeted debugging).",
    )
    @click.option(
        "--env",
        multiple=True,
        type=str,
        envvar="INSPECT_EVAL_ENV",
        help="Define an environment variable e.g. --env NAME=value (--env can be specified multiple times)",
    )
    @click.option(
        "--debug", is_flag=True, envvar="INSPECT_DEBUG", help="Wait to attach debugger"
    )
    @click.option(
        "--debug-port",
        default=5678,
        envvar="INSPECT_DEBUG_PORT",
        help="Port number for debugger",
    )
    @click.option(
        "--debug-errors",
        type=bool,
        is_flag=True,
        envvar="INSPECT_DEBUG_ERRORS",
        help="Raise task errors (rather than logging them) so they can be debugged.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> click.Context:
        return cast(click.Context, func(*args, **kwargs))

    return wrapper


def process_common_options(options: CommonOptions) -> None:
    # set environment variables
    env_args = parse_cli_args(options["env"])
    init_cli_env(env_args)

    # set traceback locals env var
    if options.get("traceback_locals", False):
        os.environ["INSPECT_TRACEBACK_LOCALS"] = "1"

    # set log display py logger name
    log_display_py_logger = options.get("log_display_py_logger")
    if log_display_py_logger:
        os.environ["INSPECT_LOG_DISPLAY_PY_LOGGER"] = log_display_py_logger

    # set secondary display type
    display_secondary = options.get("display_secondary")
    if display_secondary:
        os.environ["INSPECT_DISPLAY_SECONDARY"] = display_secondary

    # propagate display
    if options["no_ansi"]:
        display = "rich"
        rich.reconfigure(no_color=True)
    else:
        display = options["display"].lower().strip()
    init_display_type(display)

    # attach debugger if requested
    if options["debug"]:
        import debugpy  # type: ignore

        debugpy.listen(options["debug_port"])
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Debugger attached")


def clean_log_dir(
    ctx: click.Context, param: click.Option, value: str | None
) -> str | None:
    if value is not None:
        value = value.rstrip("/\\")
    return value
