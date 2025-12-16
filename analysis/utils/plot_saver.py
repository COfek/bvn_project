from __future__ import annotations

import os
from datetime import datetime
import matplotlib.pyplot as plt


def save_current_figures(
    plot_names: list[str],
    base_dir: str = "/plots",
    dpi: int = 200,
) -> str:
    """
    Save currently active matplotlib figures into a timestamped folder.

    Parameters
    ----------
    plot_names : list[str]
        Names for figures in the order they were created.
        Example: ["frontier_evolution", "split_tree"]

    base_dir : str
        Base directory where plots will be stored.

    dpi : int
        DPI for saved figures.

    Returns
    -------
    str
        Path to the created timestamped plot directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    fig_nums = plt.get_fignums()

    if len(fig_nums) < len(plot_names):
        raise RuntimeError(
            f"Expected at least {len(plot_names)} figures, "
            f"but found only {len(fig_nums)}."
        )

    for fig_num, name in zip(fig_nums[-len(plot_names):], plot_names):
        fig = plt.figure(fig_num)
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    print(f"[plots] Saved {len(plot_names)} figures to: {out_dir}")
    return out_dir
