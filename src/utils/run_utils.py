from __future__ import annotations

import json
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
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    # two runs started in the same second must not share a folder
    run_dir, n = Path(base_dir) / timestamp, 1
    while True:
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            n += 1
            run_dir = Path(base_dir) / f"{timestamp}_{n}"


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
