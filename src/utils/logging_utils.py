from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


console = Console()

# The run folder will inject its preferred log file here.
# If None → we default to logs/ like your original version.
RUN_LOG_FILE: Path | None = None


def init_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Initialize a logger with:
    - Rich console output (colored, beautiful)
    - File logging:
         * If RUN_LOG_FILE is set → use run folder log file
         * Otherwise → fallback to logs/run_YYYYMMDD_HHMMSS.log  (your original behavior)
    """
    logger = logging.getLogger("bvn_project")
    logger.setLevel(level)

    # Avoid attaching multiple handlers
    if logger.handlers:
        return logger

    # ------------------------------------------------------------
    # 1. Rich console handler  (same as your original)
    # ------------------------------------------------------------
    rich_handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    logger.addHandler(rich_handler)

    # ------------------------------------------------------------
    # 2. File logging
    # ------------------------------------------------------------
    if RUN_LOG_FILE is not None:
        # Use run folder file
        log_file = RUN_LOG_FILE
        os.makedirs(log_file.parent, exist_ok=True)
    else:
        # Fallback to your original: logs/ directory
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

    logger.propagate = False  # prevent double logs

    console.print(
        f"[green]Logging initialized.[/green] "
        f"Log file: [cyan]{log_file}[/cyan]"
    )

    return logger


# Global shared logger instance (kept exactly like your version)
LOGGER = init_logger()


# ------------------------------------------------------------
# Utility: Pretty banners (SECTION HEADERS)
# ------------------------------------------------------------
def print_banner(text: str) -> None:
    """
    Print a nice banner in the console to mark experiment phases.
    """
    console.rule(f"[bold cyan]{text}[/bold cyan]")


# ------------------------------------------------------------
# Utility: timed section (unchanged from your version)
# ------------------------------------------------------------
@contextmanager
def timed_section(name: str):
    """
    Measure execution time of a code section and log it.
    Usage:
        with timed_section("Running BVN"):
            run_bvn()
    """
    LOGGER.info(f"[bold green]Starting:[/bold green] {name}")
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    LOGGER.info(
        f"[bold green]Finished:[/bold green] {name} "
        f"in [cyan]{elapsed:.3f}[/cyan] seconds"
    )
