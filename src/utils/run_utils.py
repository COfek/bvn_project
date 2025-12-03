from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def create_run_folder(base_dir: str = "runs") -> Path:
    """
    Create a new timestamped run folder:
        runs/YYYYMMDD_HHMMSS/

    Returns:
        Path object of the new directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(config, run_dir: Path):
    """
    Save the ExperimentConfig as a JSON file inside the run directory.
    """
    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=4)
    return config_path


def get_log_file_path(run_dir: Path) -> Path:
    """
    Return the path where the log file should be stored.
    """
    return run_dir / "log.txt"
