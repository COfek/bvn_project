from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path
from scipy.stats import gaussian_kde  # New import for smoothing

from .utils.stats import DecompositionStats


# ============================================================
# Utilities
# ============================================================

def _prepare_plot_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def _moving_average(values, window):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _smooth(xs, ys, window):
    if len(ys) < window:
        return xs, ys
    ys_smooth = _moving_average(ys, window)
    xs_smooth = xs[window - 1:]
    return xs_smooth, ys_smooth


def _extract_metric(
        stats: List[DecompositionStats],
        field: str,
) -> List[float]:
    vals = []
    for s in stats:
        v = getattr(s, field, None)
        if v is not None:
            vals.append(float(v))
    return vals


# ============================================================
# Existing time-series plots (UNCHANGED)
# ============================================================

def plot_final_cycle_length(stats: List[DecompositionStats], out_dir: Path):
    _prepare_plot_dir(out_dir)

    xs = np.array([s.matrix_index for s in stats])

    cycle_bvn = [s.cycle_length_bvn for s in stats]
    cycle_max = [s.cycle_maximum for s in stats]
    cycle_wfa = [s.cycle_maximal for s in stats]
    cycle_split = [s.cycle_split for s in stats]

    window = max(5, len(stats) // 50)

    xs_bvn, cycle_bvn = _smooth(xs, cycle_bvn, window)
    xs_max, cycle_max = _smooth(xs, cycle_max, window)
    xs_wfa, cycle_wfa = _smooth(xs, cycle_wfa, window)
    xs_split, cycle_split = _smooth(xs, cycle_split, window)

    plt.figure(figsize=(12, 7))
    plt.plot(xs_bvn, cycle_bvn, label="BVN", linewidth=2)
    plt.plot(xs_max, cycle_max, label="Bitplane Maximum", linewidth=2)
    plt.plot(xs_wfa, cycle_wfa, label="Bitplane Maximal (WFA)", linewidth=2)
    plt.plot(xs_split, cycle_split, label="Split-Tree", linewidth=2)

    plt.axhline(1.0, color="black", linestyle="--", label="Upper Bound = 1")
    plt.xlabel("Matrix Index")
    plt.ylabel("Cycle Length")
    plt.title("Cycle Length (Smoothed)")
    plt.grid(True)
    plt.legend()

    plt.savefig(out_dir / "cycle_length_all_methods.png", dpi=220)
    plt.close()


def plot_final_num_permutations(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    _prepare_plot_dir(out_dir)

    xs = np.array([s.matrix_index for s in stats])

    perm_bvn = [s.num_permutations_bvn for s in stats]
    perm_max = [s.num_perm_maximum for s in stats]
    perm_wfa = [s.num_perm_maximal for s in stats]
    perm_split = [s.num_perm_split for s in stats]

    window = max(5, len(stats) // 50)

    xs_bvn, perm_bvn = _smooth(xs, perm_bvn, window)
    xs_max, perm_max = _smooth(xs, perm_max, window)
    xs_wfa, perm_wfa = _smooth(xs, perm_wfa, window)
    xs_split, perm_split = _smooth(xs, perm_split, window)

    plt.figure(figsize=(12, 7))
    plt.plot(xs_bvn, perm_bvn, label="BVN", linewidth=2)
    plt.plot(xs_max, perm_max, label="Bitplane Maximum", linewidth=2)
    plt.plot(xs_wfa, perm_wfa, label="Bitplane Maximal (WFA)", linewidth=2)
    plt.plot(xs_split, perm_split, label="Split-Tree", linewidth=2)

    plt.axhline(n * n - 2 * n + 2, color="orange", linestyle="--", label="BVN Upper Bound")
    plt.axhline(bits * n, color="purple", linestyle="--", label="Bitplane Upper Bound")

    plt.xlabel("Matrix Index")
    plt.ylabel("Number of Permutations")
    plt.title("Permutation Count (Smoothed)")
    plt.grid(True)
    plt.legend()

    plt.savefig(out_dir / "permutation_count_all_methods.png", dpi=220)
    plt.close()


def plot_runtime(stats: List[DecompositionStats], out_dir: Path):
    _prepare_plot_dir(out_dir)

    xs = np.array([s.matrix_index for s in stats])

    rt_bvn = [s.runtime_bvn for s in stats]
    rt_max = [s.runtime_maximum for s in stats]
    rt_wfa = [s.runtime_maximal for s in stats]
    rt_split = [s.runtime_split for s in stats]

    window = max(5, len(stats) // 50)

    xs_bvn, rt_bvn = _smooth(xs, rt_bvn, window)
    xs_max, rt_max = _smooth(xs, rt_max, window)
    xs_wfa, rt_wfa = _smooth(xs, rt_wfa, window)
    xs_split, rt_split = _smooth(xs, rt_split, window)

    plt.figure(figsize=(12, 7))
    plt.plot(xs_bvn, rt_bvn, label="BVN", linewidth=2)
    plt.plot(xs_max, rt_max, label="Bitplane Maximum", linewidth=2)
    plt.plot(xs_wfa, rt_wfa, label="Bitplane Maximal (WFA)", linewidth=2)
    plt.plot(xs_split, rt_split, label="Split-Tree", linewidth=2)

    plt.xlabel("Matrix Index")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime (Smoothed)")
    plt.grid(True)
    plt.legend()

    plt.savefig(out_dir / "runtime_comparison.png", dpi=220)
    plt.close()


# ============================================================
# SMOOTHED DISTRIBUTION PLOTS
# ============================================================

def _plot_pdf_cdf_on_ax(
        ax,
        values,
        label: str,
        bins: int = 50,
):
    """
    Draw SMOOTH PDF and CDF and return combined legend handles.
    """
    values = [v for v in values if v is not None]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return [], []

    # Handle constant data
    if np.all(values == values[0]):
        line = ax.axvline(values[0], color="tab:blue", lw=2, label=f"{label} (Constant)")
        return [line], [f"{label} (Constant)"]

    # Jittering for discrete data
    if np.all(np.mod(values, 1) == 0):
        values = values + np.random.normal(0, 0.3, size=values.shape)

    # --- 1. PDF ---
    pdf_handle = None
    try:
        kde = gaussian_kde(values)
        x_min, x_max = values.min(), values.max()
        margin = (x_max - x_min) * 0.2
        x_range = np.linspace(x_min - margin, x_max + margin, 500)
        pdf_values = kde(x_range)
        pdf_handle, = ax.plot(x_range, pdf_values, linewidth=2, label=f"{label} PDF")
        ax.fill_between(x_range, pdf_values, alpha=0.1)
    except:
        pass

    # --- 2. CDF ---
    ax_cdf = ax.twinx()
    sorted_vals = np.sort(values)
    cdf_y = np.linspace(0, 1, len(sorted_vals))
    cdf_handle, = ax_cdf.plot(sorted_vals, cdf_y, linestyle="--", linewidth=1.5,
                              color="tab:orange", label=f"{label} CDF")

    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_ylabel("Cumulative Probability", color="tab:orange", fontsize=8)
    ax_cdf.tick_params(axis='y', labelcolor="tab:orange", labelsize=7)

    # Return handles so the calling function can build a single legend
    handles = [pdf_handle, cdf_handle]
    labels = [f"{label} PDF", f"{label} CDF"]
    return handles, labels


def plot_distribution_runtime(stats: List[DecompositionStats], out_dir: Path):
    _prepare_plot_dir(out_dir)
    methods = {
        "BVN": [s.runtime_bvn for s in stats],
        "Bitplane Max": [s.runtime_maximum for s in stats],
        "Bitplane WFA": [s.runtime_maximal for s in stats],
        "Split-Tree": [s.runtime_split for s in stats],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, values) in zip(axes, methods.items()):
        # Get the handles from the helper
        h, l = _plot_pdf_cdf_on_ax(ax, values, label=name)

        ax.set_title(f"{name} Runtime Distribution")
        ax.set_xlabel("Runtime (seconds)")
        ax.set_ylabel("Density (PDF)")

        # Build one clean legend per subplot
        if h:
            ax.legend(h, l, loc="upper right", fontsize=9)

    plt.suptitle("Individual Runtime Distributions (Seconds)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / "runtime_pdf_cdf_subplots.png", dpi=220)
    plt.close()


def plot_cycle_length_distributions(stats: List[DecompositionStats], out_dir: Path):
    """
    Cycle length PDF + CDF in 2×2 subplots with clean combined legends.
    """
    _prepare_plot_dir(out_dir)

    methods = {
        "BVN": [s.cycle_length_bvn for s in stats],
        "Bitplane Max": [s.cycle_maximum for s in stats],
        "Bitplane WFA": [s.cycle_maximal for s in stats],
        "Split-Tree": [s.cycle_split for s in stats],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, values) in zip(axes, methods.items()):
        # Capture handles/labels from helper to avoid messed up legends
        h, l = _plot_pdf_cdf_on_ax(ax, values, label=name)
        ax.set_title(f"{name} Cycle Length")
        ax.set_xlabel("Cycle length")
        ax.set_ylabel("Density (PDF)")

        if h:
            # Combine handles into a single legend on the primary axis
            ax.legend(h, l, loc="upper left", fontsize=8)

    plt.suptitle("Cycle Length Distributions (PDF + CDF)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / "cycle_length_pdf_cdf.png", dpi=220)
    plt.close()


def plot_permutation_distributions(stats: List[DecompositionStats], out_dir: Path):
    """
    Permutation count PDF + CDF in 2×2 subplots with linear CDFs to remove staircases.
    """
    _prepare_plot_dir(out_dir)

    methods = {
        "BVN": [s.num_permutations_bvn for s in stats],
        "Bitplane Max": [s.num_perm_maximum for s in stats],
        "Bitplane WFA": [s.num_perm_maximal for s in stats],
        "Split-Tree": [s.num_perm_split for s in stats],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, values) in zip(axes, methods.items()):
        # Helper now uses jittering and linear CDF interpolation
        h, l = _plot_pdf_cdf_on_ax(ax, values, label=name)
        ax.set_title(f"{name} Permutations")
        ax.set_xlabel("Number of permutations")
        ax.set_ylabel("Density (PDF)")

        if h:
            ax.legend(h, l, loc="upper left", fontsize=8)

    plt.suptitle("Permutation Count Distributions (Smoothed)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / "permutation_pdf_cdf.png", dpi=220)
    plt.close()


def plot_runtime_vs_cycle_efficiency(stats: List[DecompositionStats], out_dir: Path):
    _prepare_plot_dir(out_dir)

    methods = {
        "BVN": ("runtime_bvn", "cycle_length_bvn"),
        "Bitplane Max": ("runtime_maximum", "cycle_maximum"),
        "Bitplane WFA": ("runtime_maximal", "cycle_maximal"),
        "Split-Tree": ("runtime_split", "cycle_split"),
    }

    plt.figure(figsize=(10, 7))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for (name, (rt_attr, cyc_attr)), color in zip(methods.items(), colors):
        runtimes = [getattr(s, rt_attr) for s in stats if getattr(s, rt_attr) is not None]
        cycles = [getattr(s, cyc_attr) for s in stats if getattr(s, cyc_attr) is not None]

        if not runtimes or not cycles:
            continue

        mean_rt = np.mean(runtimes)
        std_rt = np.std(runtimes)
        mean_cyc = np.mean(cycles)
        std_cyc = np.std(cycles)

        plt.errorbar(
            mean_cyc, mean_rt,
            xerr=std_cyc, yerr=std_rt,
            fmt='o', markersize=8, capsize=5,
            label=name, color=color, alpha=0.8
        )

    plt.xlabel("Average Cycle Length (Lower is better, ideally 1.0)")
    plt.ylabel("Average Runtime (Seconds)")
    plt.title("Algorithm Efficiency: Runtime vs. Cycle Length")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="Ideal Cycle")

    plt.savefig(out_dir / "runtime_vs_cycle_efficiency.png", dpi=220)
    plt.close()


def plot_results(stats_list: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    _prepare_plot_dir(out_dir)

    plot_final_cycle_length(stats_list, out_dir=out_dir)
    plot_final_num_permutations(stats_list, n=n, bits=bits, out_dir=out_dir)
    plot_runtime(stats_list, out_dir=out_dir)

    plot_distribution_runtime(stats_list, out_dir=out_dir)
    plot_cycle_length_distributions(stats_list, out_dir)
    plot_permutation_distributions(stats_list, out_dir)
    plot_runtime_vs_cycle_efficiency(stats_list, out_dir)