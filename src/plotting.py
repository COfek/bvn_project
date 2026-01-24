from __future__ import annotations

import matplotlib as mpl
mpl.use('Agg')
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
    if len(values) < window: return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _smooth(xs, ys, window):
    ys = [v for v in ys if v is not None]
    if len(ys) < window:
        return xs[:len(ys)], ys
    ys_smooth = _moving_average(ys, window)
    xs_smooth = xs[window - 1:window - 1 + len(ys_smooth)]
    return xs_smooth, ys_smooth


def _get_all_keys(stats: List[DecompositionStats]) -> List[str]:
    """Helper to find all unique result keys (e.g. 'wfa_2')."""
    keys = set()
    for s in stats:
        keys.update(s.radix_multi_results.keys())
    return sorted(list(keys))


# ============================================================
# Distribution Helpers (Restored Missing Function)
# ============================================================

def _plot_pdf_cdf_on_ax(ax, values, label: str):
    """Calculates and plots the PDF (KDE) and CDF on a given axis."""
    values = [v for v in values if v is not None]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return [], []

    # Handle constant data (no variance for KDE)
    if np.all(values == values[0]):
        line = ax.axvline(values[0], color="tab:blue", lw=2, label=f"{label} (Const)")
        return [line], [f"{label} (Const)"]

    # Jitter for integer-heavy data to help KDE converge
    if np.all(np.mod(values, 1) == 0):
        values = values + np.random.normal(0, 0.1, size=values.shape)

    pdf_handle = None
    try:
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 500)
        pdf_values = kde(x_range)
        pdf_handle, = ax.plot(x_range, pdf_values, lw=2, label="PDF")
        ax.fill_between(x_range, pdf_values, alpha=0.1)
    except Exception:
        pass

    # Secondary Y-axis for CDF
    ax_cdf = ax.twinx()
    sorted_vals = np.sort(values)
    cdf_y = np.linspace(0, 1, len(sorted_vals))
    cdf_handle, = ax_cdf.plot(sorted_vals, cdf_y, "--", color="tab:orange", lw=1.5, label="CDF")
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.tick_params(axis='y', labelcolor="tab:orange", labelsize=7)

    return [pdf_handle, cdf_handle], ["PDF", "CDF"]


# ============================================================
# Dynamic Grid Plotting
# ============================================================

def _plot_dynamic_grid(stats, out_dir, filename, title, baseline_map, radix_index):
    _prepare_plot_dir(out_dir)

    # Filter for fields that actually contain data
    active_baselines = {n: f for n, f in baseline_map.items() if any(getattr(s, f) is not None for s in stats)}
    keys = _get_all_keys(stats)

    total_plots = len(active_baselines) + len(keys)
    if total_plots == 0:
        return

    cols = 3
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    # Handle the case where subplots returns a single object instead of an array
    if total_plots == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()

    idx = 0
    # 1. Plot Core Baselines
    for name, field in active_baselines.items():
        vals = [getattr(s, field) for s in stats]
        h, l = _plot_pdf_cdf_on_ax(axes_list[idx], vals, name)
        axes_list[idx].set_title(f"{name} {title}")
        if h: axes_list[idx].legend(h, l, loc="upper right", fontsize=8)
        idx += 1

    # 2. Plot All Radix Bases
    for key in keys:
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()} (B-{b})"
        else:
            label = f"Result ({key})"
            
        vals = [s.radix_multi_results[key][radix_index] for s in stats if key in s.radix_multi_results]
        h, l = _plot_pdf_cdf_on_ax(axes_list[idx], vals, label)
        axes_list[idx].set_title(f"{label} {title}")
        if h: axes_list[idx].legend(h, l, loc="upper right", fontsize=8)
        idx += 1

    # Hide unused axes
    for j in range(idx, len(axes_list)):
        axes_list[j].axis('off')

    plt.suptitle(f"Distribution Comparison: {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()


# ============================================================
# Time-Series Plots
# ============================================================

def plot_final_cycle_length(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)

    plt.figure(figsize=(12, 7))
    baselines = {"BVN": [s.cycle_length_bvn for s in stats], "Split-Tree": [s.cycle_split for s in stats]}

    for name, values in baselines.items():
        if all(v is None for v in values): continue
        x_s, y_s = _smooth(xs, values, window)
        plt.plot(x_s, y_s, label=name, linewidth=2.5)

    keys = _get_all_keys(stats)
    for key in keys:
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()} (B-{b})"
        else:
            label = f"Result ({key})"

        vals = [s.radix_multi_results[key][1] for s in stats if key in s.radix_multi_results]
        x_s, y_s = _smooth(xs, vals, window)
        plt.plot(x_s, y_s, label=label, linestyle="--", alpha=0.8)

    plt.axhline(1.0, color="black", linestyle=":", label="Ideal = 1.0")
    plt.xlabel("Matrix Index")
    plt.ylabel("Cycle Length")
    plt.title(f"Cycle Length Trends (Smoothed) {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "cycle_length_all_methods.png", dpi=220)
    plt.close()


def plot_final_num_permutations(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path, title_suffix: str = ""):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)

    plt.figure(figsize=(12, 7))
    baselines = {"BVN": [s.num_permutations_bvn for s in stats], "Split-Tree": [s.num_perm_split for s in stats]}

    for name, values in baselines.items():
        if all(v is None for v in values): continue
        x_s, y_s = _smooth(xs, values, window)
        plt.plot(x_s, y_s, label=name, linewidth=2.5)

    keys = _get_all_keys(stats)
    for key in keys:
        # Key format expected: "engine_base" or just "base" (legacy)
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()} (B-{b})"
        else:
            label = f"Result ({key})"

        vals = [s.radix_multi_results[key][2] for s in stats if key in s.radix_multi_results]
        x_s, y_s = _smooth(xs, vals, window)
        plt.plot(x_s, y_s, label=label, linestyle="--", alpha=0.8)

    plt.axhline(n * n - 2 * n + 2, color="orange", linestyle="--", label="BVN UB", alpha=0.5)
    plt.xlabel("Matrix Index")
    plt.ylabel("Number of Permutations")
    plt.title(f"Permutation Count Trends (Smoothed) {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "permutation_count_all_methods.png", dpi=220)
    plt.close()


def plot_runtime(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)

    plt.figure(figsize=(12, 7))
    baselines = {"BVN": [s.runtime_bvn for s in stats], "Split-Tree": [s.runtime_split for s in stats]}

    for name, values in baselines.items():
        if all(v is None for v in values): continue
        x_s, y_s = _smooth(xs, values, window)
        plt.plot(x_s, y_s, label=name, linewidth=2.5)

    keys = _get_all_keys(stats)
    for key in keys:
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()} (B-{b})"
        else:
            label = f"Result ({key})"

        vals = [s.radix_multi_results[key][0] for s in stats if key in s.radix_multi_results]
        x_s, y_s = _smooth(xs, vals, window)
        plt.plot(x_s, y_s, label=label, linestyle="--", alpha=0.8)

    plt.xlabel("Matrix Index")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime Trends (Smoothed) {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_comparison.png", dpi=220)
    plt.close()


# ============================================================
# Plot Orchestration
# ============================================================

def plot_distribution_runtime(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    b_map = {"BVN": "runtime_bvn", "Split-Tree": "runtime_split"}
    _plot_dynamic_grid(stats, out_dir, "runtime_pdf_cdf_subplots.png", f"Runtime {title_suffix}", b_map, 0)


def plot_cycle_length_distributions(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    b_map = {"BVN": "cycle_length_bvn", "Split-Tree": "cycle_split"}
    _plot_dynamic_grid(stats, out_dir, "cycle_length_pdf_cdf.png", f"Cycle Length {title_suffix}", b_map, 1)


def plot_permutation_distributions(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    b_map = {"BVN": "num_permutations_bvn", "Split-Tree": "num_perm_split"}
    _plot_dynamic_grid(stats, out_dir, "permutation_pdf_cdf.png", f"Permutations {title_suffix}", b_map, 2)


def plot_runtime_vs_cycle_efficiency(stats: List[DecompositionStats], out_dir: Path, matching_name: str, metadata: str = ""):
    """
    Plots Average Runtime against the Average Cycle Length.
    Includes the matching method in the title for better experiment tracking.
    """
    _prepare_plot_dir(out_dir)

    baselines = {
        "BVN": ("runtime_bvn", "cycle_length_bvn", "tab:blue", "o"),
        "Split-Tree": ("runtime_split", "cycle_split", "tab:red", "^")
    }

    plt.figure(figsize=(12, 8))

    # 1. Plot Baselines (BVN / Split-Tree)
    for name, (rt_f, cyc_f, color, marker) in baselines.items():
        rts = [getattr(s, rt_f) for s in stats if getattr(s, rt_f) is not None]
        cycs = [getattr(s, cyc_f) for s in stats if getattr(s, cyc_f) is not None]

        if rts and cycs:
            med_rt = np.median(rts)
            med_cyc = np.median(cycs)
            
            # Plot Medians Only
            plt.plot(med_cyc, med_rt,
                     marker=marker, markersize=10,
                     label=name, color=color, alpha=0.7, linestyle='None')

    # 2. Plot Radix Bases
    keys = _get_all_keys(stats)
    if keys:
        cmap = mpl.colormaps['plasma']
        colors = cmap(np.linspace(0, 0.85, len(keys)))

        for i, key in enumerate(keys):
            parts = str(key).split('_')
            if len(parts) == 2:
                eng, b = parts
                label = f"{eng.upper()} (B-{b})"
            else:
                label = f"Result ({key})"

            # Index 0 is Runtime, Index 1 is Cycle Length
            base_data = [s.radix_multi_results[key] for s in stats if key in s.radix_multi_results]

            if base_data:
                base_rts = [d[0] for d in base_data]
                base_cycs = [d[1] for d in base_data]

                # Calculate asymmetric error bars (Percentiles)
                # Plot Medians Only (No Error Bars)
                med_rt = np.median(base_rts)
                med_cyc = np.median(base_cycs)
                
                rt_err = [[med_rt - np.percentile(base_rts, 5)], [np.percentile(base_rts, 95) - med_rt]]
                cyc_err = [[med_cyc - np.percentile(base_cycs, 5)], [np.percentile(base_cycs, 95) - med_cyc]]

                # Ensure non-negative (can happen if distribution is weirdly tight)
                rt_err = np.maximum(0, rt_err)
                cyc_err = np.maximum(0, cyc_err)

                plt.plot(med_cyc, med_rt,
                            marker='v', markersize=8,
                            label=label, color=colors[i], alpha=0.9, linestyle='None')

    # Reference line removed per user request
    # Allow autoscaling
    plt.autoscale(enable=True, axis='y')

    plt.xlabel("Average Cycle Length")
    plt.ylabel("Average Runtime (Seconds)")

    # Updated Title incorporating the matching method
    plt.title(f"Efficiency Pareto (Matching: {matching_name.upper()}) {metadata}\nRuntime vs. Cycle Length")

    plt.grid(True, linestyle=":", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Filename includes the matching method to prevent overwriting during multi-run batches
    plt.savefig(out_dir / f"runtime_vs_cycle_{matching_name}.png", dpi=250)
    plt.close()


def plot_runtime_vs_permutation_efficiency(
        stats: List[DecompositionStats],
        out_dir: Path,
        matching_name: str,
        metadata: str = ""
):
    """
    Plots Average Runtime against the Average Number of Permutations.
    The matching method is included in the title for better experiment tracking.
    """
    _prepare_plot_dir(out_dir)

    baselines = {
        "BVN": ("runtime_bvn", "num_permutations_bvn", "tab:blue", "o"),
        "Split-Tree": ("runtime_split", "num_perm_split", "tab:red", "^")
    }

    plt.figure(figsize=(12, 8))

    # 1. Plot Baselines
    for name, (rt_f, perm_f, color, marker) in baselines.items():
        rts = [getattr(s, rt_f) for s in stats if getattr(s, rt_f) is not None]
        perms = [getattr(s, perm_f) for s in stats if getattr(s, perm_f) is not None]

        if rts and perms:
            med_rt = np.median(rts)
            med_perm = np.median(perms)
            
            # Plot Medians Only
            plt.plot(med_perm, med_rt,
                     marker=marker, markersize=10,
                     label=name, color=color, alpha=0.7, linestyle='None')

    # 2. Plot Radix Bases
    keys = _get_all_keys(stats)
    if keys:
        cmap = mpl.colormaps['plasma']
        colors = cmap(np.linspace(0, 0.85, len(keys)))

        for i, key in enumerate(keys):
            parts = str(key).split('_')
            if len(parts) == 2:
                eng, b = parts
                label = f"{eng.upper()} (B-{b})"
            else:
                label = f"Result ({key})"

            base_data = [s.radix_multi_results[key] for s in stats if key in s.radix_multi_results]

            if base_data:
                base_rts = [d[0] for d in base_data]  # Index 0: Runtime
                base_perms = [d[2] for d in base_data]  # Index 2: Num Perms

                # Calculate asymmetric error bars (Percentiles)
                med_rt = np.median(base_rts)
                med_perms = np.median(base_perms)
                
                # Plot Medians Only (No Error Bars)
                plt.plot(med_perms, med_rt,
                         marker='v', markersize=8,
                         label=label, color=colors[i], alpha=0.9, linestyle='None')

    # Allow autoscaling to zoom in on the data
    plt.autoscale(enable=True, axis='y')

    plt.xlabel("Average Number of Permutations")
    plt.ylabel("Average Runtime (Seconds)")

    # Updated Title with Matching Method
    plt.title(f"Efficiency Pareto (Matching: {matching_name.upper()}) {metadata}\nRuntime vs. Permutation Count")

    plt.grid(True, linestyle=":", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / f"runtime_vs_permutations_{matching_name}.png", dpi=250)
    plt.close()


def plot_results(
        stats_list: List[DecompositionStats],
        n: int,
        bits: int,
        out_dir: Path,
        matching_method: str
):
    num_samples = len(stats_list)
    title_suffix = f"(N={n}, Samples={num_samples})"

    # Standard distribution and scalar plots
    plot_final_cycle_length(stats_list, out_dir, title_suffix)
    plot_final_num_permutations(stats_list, n, bits, out_dir, title_suffix)
    plot_runtime(stats_list, out_dir, title_suffix)
    plot_distribution_runtime(stats_list, out_dir, title_suffix)
    plot_cycle_length_distributions(stats_list, out_dir, title_suffix)
    plot_permutation_distributions(stats_list, out_dir, title_suffix)

    # Updated Pareto efficiency plots
    plot_runtime_vs_cycle_efficiency(stats_list, out_dir, matching_name=matching_method, metadata=title_suffix)
    plot_runtime_vs_permutation_efficiency(stats_list, out_dir, matching_name=matching_method, metadata=title_suffix)