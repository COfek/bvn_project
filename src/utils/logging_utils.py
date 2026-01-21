from __future__ import annotations

import logging
import os
import multiprocessing
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()

# The run folder will inject its preferred log file here.
RUN_LOG_FILE: Path | None = None


def init_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Initialize a logger with:
    - Rich console output
    - File logging (Main Process only)
    """
    logger = logging.getLogger("bvn_project")

    # Check if this is the Parent process
    is_main = multiprocessing.current_process().name == 'MainProcess'

    # If handlers are already attached, don't add them again
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 1. Rich console handler (All processes can have this for console output)
    rich_handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    logger.addHandler(rich_handler)

    # 2. File logging (ONLY for the MainProcess to prevent worker spam)
    if is_main:
        if RUN_LOG_FILE is not None:
            log_file = RUN_LOG_FILE
            os.makedirs(log_file.parent, exist_ok=True)
        else:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir,
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

        # Only print the setup message once in the main terminal
        console.print(
            f"[green]Logging initialized.[/green] "
            f"Log file: [cyan]{log_file}[/cyan]"
        )

    logger.propagate = False  # prevent double logs
    return logger


# Global shared logger instance.
# On Windows, this runs on every 'import', but our guard inside
# the function prevents it from printing or creating files in workers.
LOGGER = init_logger()


def print_banner(text: str) -> None:
    """
    Print a nice banner in the console to mark experiment phases.
    """
    # Only the main process should print banners
    if multiprocessing.current_process().name == 'MainProcess':
        console.rule(f"[bold cyan]{text}[/bold cyan]")


@contextmanager
def timed_section(name: str):
    """
    Measure execution time of a code section and log it.
    """
    # Only log start/finish in the main process to keep console clean
    is_main = multiprocessing.current_process().name == 'MainProcess'

    if is_main:
        LOGGER.info(f"[bold green]Starting:[/bold green] {name}")

    start = perf_counter()
    yield
    elapsed = perf_counter() - start

    if is_main:
        LOGGER.info(
            f"[bold green]Finished:[/bold green] {name} "
            f"in [cyan]{elapsed:.3f}[/cyan] seconds"
        )