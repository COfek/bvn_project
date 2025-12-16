from __future__ import annotations

import matplotlib.pyplot as plt


def plot_frontier_evolution(
    frontier_result,
    show_active_matrices: bool = True,
    figsize=(10, 5),
):
    """
    Plot frontier evolution across split steps (STEP-based).
    Returns the matplotlib Figure (does NOT call plt.show()).

    Expects frontier_result.steps with fields:
      - step
      - num_matrices
      - total_permutations
      - total_cycle_length
    """
    import matplotlib.pyplot as plt

    steps = [s.step for s in frontier_result.steps]
    cycles = [s.total_cycle_length for s in frontier_result.steps]
    perms = [s.total_permutations for s in frontier_result.steps]
    mats = [s.num_matrices for s in frontier_result.steps]

    fig, ax1 = plt.subplots(figsize=figsize)

    # --- Left Y-axis: cycle length ---
    ax1.plot(
        steps,
        cycles,
        marker="o",
        linewidth=2,
        label="Total cycle length",
        color="tab:blue",
    )
    ax1.set_xlabel("Split step")
    ax1.set_ylabel("Total cycle length", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    # --- Right Y-axis: permutations ---
    ax2 = ax1.twinx()
    ax2.plot(
        steps,
        perms,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Total permutations",
        color="tab:red",
    )
    ax2.set_ylabel("Total permutations", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # --- Optional: number of active matrices ---
    if show_active_matrices:
        ax2.plot(
            steps,
            mats,
            marker="^",
            linestyle=":",
            linewidth=2,
            label="Active matrices",
            color="tab:green",
        )

    # --- Legend ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
    )

    plt.title("Split-Tree Frontier Evolution (Step-Based)")
    plt.tight_layout()

    return fig

