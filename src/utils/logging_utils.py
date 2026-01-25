from __future__ import annotations

import logging
import multiprocessing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.logging import RichHandler

console = Console()

def configure_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """
    Configure the root logger with:
    - Rich console output (for all processes, though workers usually shouldn't log much)
    - File logging (Main Process only, if log_file is provided)
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates on re-init
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Rich console handler
    rich_handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # 2. File logging (Main Process Only)
    is_main = multiprocessing.current_process().name == 'MainProcess'
    
    if is_main:
        if log_file:
            # Ensure directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            target_file = log_file
        else:
            # Fallback default
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            target_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        file_handler = logging.FileHandler(target_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        
        # We can print to console that logging is set up
        console.print(f"[green]Logging configured.[/green] Log file: [cyan]{target_file}[/cyan]")


def print_banner(text: str) -> None:
    """
    Print a nice banner in the console to mark experiment phases.
    """
    if multiprocessing.current_process().name == 'MainProcess':
        console.rule(f"[bold cyan]{text}[/bold cyan]")


@contextmanager
def timed_section(name: str):
    """
    Measure execution time of a code section and log it.
    """
    logger = logging.getLogger("bvn.timer")
    is_main = multiprocessing.current_process().name == 'MainProcess'

    if is_main:
        logger.info(f"[bold green]Starting:[/bold green] {name}")

    start = perf_counter()
    yield
    elapsed = perf_counter() - start

    if is_main:
        logger.info(
            f"[bold green]Finished:[/bold green] {name} "
            f"in [cyan]{elapsed:.3f}[/cyan] seconds"
        )