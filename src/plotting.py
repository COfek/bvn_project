from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path
from scipy.stats import gaussian_kde

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

# ============================================================
# Time-Series Plots
# ============================================================

def plot_final_cycle_length(stats: List[DecompositionStats], out_dir: Path):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)

    methods = {
        "BVN": [s.cycle_length_bvn for s in stats],
        "Bitplane Max": [s.cycle_maximum for s in stats],
        "Bitplane WFA": [s.cycle_maximal for s in stats],
        "Split-Tree": [s.cycle_split for s in stats],
        "Radix (Base-8)": [s.cycle_radix for s in stats],
    }

    plt.figure(figsize=(12, 7))
    for name, values in methods.items():
        if all(v is None for v in values): continue
        x_s, y_s = _smooth(xs, values, window)
        plt.plot(x_s, y_s, label=name, linewidth=2)

    plt.axhline(1.0, color="black", linestyle="--", label="Ideal = 1.0")
    plt.xlabel("Matrix Index")
    plt.ylabel("Cycle Length")
    plt.title("Cycle Length Trends (Smoothed)")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir / "cycle_length_all_methods.png", dpi=220)
    plt.close()

def plot_final_num_permutations(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)

    methods = {
        "BVN": [s.num_permutations_bvn for s in stats],
        "Bitplane Max": [s.num_perm_maximum for s in stats],
        "Bitplane WFA": [s.num_perm_maximal for s in stats],
        "Split-Tree": [s.num_perm_split for s in stats],
        "Radix (Base-8)": [s.num_perm_radix for s in stats],
    }

    plt.figure(figsize=(12, 7))
    for name, values in methods.items():
        if all(v is None for v in values): continue
        x_s, y_s = _smooth(xs, values, window)
        plt.plot(x_s, y_s, label=name, linewidth=2)

    plt.axhline(n * n - 2 * n + 2, color="orange", linestyle="--", label="BVN UB")
    plt.axhline(bits * n, color="purple", linestyle="--", label="Bitplane UB")
    plt.xlabel("Matrix Index")
    plt.ylabel("Number of Permutations")
    plt.title("Permutation Count Trends (Smoothed)")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir / "permutation_count_all_methods.png", dpi=220)
    plt.close()

def plot_runtime(stats: List[DecompositionStats], out_dir: Path):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)

    methods = {
        "BVN": [s.runtime_bvn for s in stats],
        "Bitplane Max": [s.runtime_maximum for s in stats],
        "Bitplane WFA": [s.runtime_maximal for s in stats],
        "Split-Tree": [s.runtime_split for s in stats],
        "Radix (Base-8)": [s.runtime_radix for s in stats],
    }

    plt.figure(figsize=(12, 7))
    for name, values in methods.items():
        if all(v is None for v in values): continue
        x_s, y_s = _smooth(xs, values, window)
        plt.plot(x_s, y_s, label=name, linewidth=2)

    plt.xlabel("Matrix Index")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Trends (Smoothed)")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir / "runtime_comparison.png", dpi=220)
    plt.close()

# ============================================================
# Smoothed Distribution Plots (Subplots)
# ============================================================

def _plot_pdf_cdf_on_ax(ax, values, label: str):
    values = [v for v in values if v is not None]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return [], []

    if np.all(values == values[0]):
        line = ax.axvline(values[0], color="tab:blue", lw=2, label=f"{label} (Const)")
        return [line], [f"{label} (Const)"]

    # Jittering for discrete data to allow KDE smoothing
    if np.all(np.mod(values, 1) == 0):
        values = values + np.random.normal(0, 0.3, size=values.shape)

    # PDF
    pdf_handle = None
    try:
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 500)
        pdf_values = kde(x_range)
        pdf_handle, = ax.plot(x_range, pdf_values, lw=2, label="PDF")
        ax.fill_between(x_range, pdf_values, alpha=0.1)
    except: pass

    # CDF on Twin Axis
    ax_cdf = ax.twinx()
    sorted_vals = np.sort(values)
    cdf_y = np.linspace(0, 1, len(sorted_vals))
    cdf_handle, = ax_cdf.plot(sorted_vals, cdf_y, "--", color="tab:orange", lw=1.5, label="CDF")
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.tick_params(axis='y', labelcolor="tab:orange", labelsize=7)

    return [pdf_handle, cdf_handle], ["PDF", "CDF"]

def _plot_grid(stats, out_dir, filename, title, field_map):
    _prepare_plot_dir(out_dir)
    # Using 2x3 grid to fit all 5 methods
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (name, field) in enumerate(field_map.items()):
        ax = axes[i]
        vals = [getattr(s, field) for s in stats]
        h, l = _plot_pdf_cdf_on_ax(ax, vals, name)
        ax.set_title(f"{name} {title}")
        if h: ax.legend(h, l, loc="upper right", fontsize=8)

    for j in range(len(field_map), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()

def plot_distribution_runtime(stats: List[DecompositionStats], out_dir: Path):
    f_map = {
        "BVN": "runtime_bvn", "Bitplane Max": "runtime_maximum",
        "Bitplane WFA": "runtime_maximal", "Split-Tree": "runtime_split",
        "Radix (Base-8)": "runtime_radix"
    }
    _plot_grid(stats, out_dir, "runtime_pdf_cdf_subplots.png", "Runtime (sec)", f_map)

def plot_cycle_length_distributions(stats: List[DecompositionStats], out_dir: Path):
    f_map = {
        "BVN": "cycle_length_bvn", "Bitplane Max": "cycle_maximum",
        "Bitplane WFA": "cycle_maximal", "Split-Tree": "cycle_split",
        "Radix (Base-8)": "cycle_radix"
    }
    _plot_grid(stats, out_dir, "cycle_length_pdf_cdf.png", "Cycle Length", f_map)

def plot_permutation_distributions(stats: List[DecompositionStats], out_dir: Path):
    f_map = {
        "BVN": "num_permutations_bvn", "Bitplane Max": "num_perm_maximum",
        "Bitplane WFA": "num_perm_maximal", "Split-Tree": "num_perm_split",
        "Radix (Base-8)": "num_perm_radix"
    }
    _plot_grid(stats, out_dir, "permutation_pdf_cdf.png", "Permutations", f_map)

# ============================================================
# Efficiency Plot
# ============================================================

def plot_runtime_vs_cycle_efficiency(stats: List[DecompositionStats], out_dir: Path):
    _prepare_plot_dir(out_dir)
    methods = {
        "BVN": ("runtime_bvn", "cycle_length_bvn", "tab:blue"),
        "Bitplane Max": ("runtime_maximum", "cycle_maximum", "tab:orange"),
        "Bitplane WFA": ("runtime_maximal", "cycle_maximal", "tab:green"),
        "Split-Tree": ("runtime_split", "cycle_split", "tab:red"),
        "Radix (Base-8)": ("runtime_radix", "cycle_radix", "tab:purple"),
    }

    plt.figure(figsize=(10, 7))
    for name, (rt_f, cyc_f, color) in methods.items():
        rts = [getattr(s, rt_f) for s in stats if getattr(s, rt_f) is not None]
        cycs = [getattr(s, cyc_f) for s in stats if getattr(s, cyc_f) is not None]
        if rts and cycs:
            plt.errorbar(np.mean(cycs), np.mean(rts), xerr=np.std(cycs), yerr=np.std(rts),
                         fmt='o', markersize=8, capsize=5, label=name, color=color, alpha=0.8)

    plt.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="Optimal")
    plt.xlabel("Average Cycle Length")
    plt.ylabel("Average Runtime (Seconds)")
    plt.title("Efficiency Comparison: All Methods")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.savefig(out_dir / "runtime_vs_cycle_efficiency.png", dpi=220)
    plt.close()

def plot_results(stats_list: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    plot_final_cycle_length(stats_list, out_dir)
    plot_final_num_permutations(stats_list, n, bits, out_dir)
    plot_runtime(stats_list, out_dir)
    plot_distribution_runtime(stats_list, out_dir)
    plot_cycle_length_distributions(stats_list, out_dir)
    plot_permutation_distributions(stats_list, out_dir)
    plot_runtime_vs_cycle_efficiency(stats_list, out_dir)