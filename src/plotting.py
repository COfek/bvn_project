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
# Global Style Settings
# ============================================================
# Setting a clean style with a visible grid as requested
plt.style.use('default') 
mpl.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    
    # Grid Settings
    'axes.grid': True,
    'grid.alpha': 1.0,           # Fully opaque
    'grid.color': '#e0e0e0',     # Light gray grid
    'grid.linestyle': '-',       # Solid line
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,      # Grid behind plot elements
    'axes.facecolor': 'white',   # White background for contrast
    'figure.facecolor': 'white',
    
    # Edge colors
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
})

# Base colors for fixed methods
COLORS_BASE = {
    'BVN': '#1f77b4',     # Strong Blue
    'Split-Tree': '#d62728' # Strong Red
}

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
    # Sort nicely: try to sort by engine then base if possible
    def sort_key(k):
        parts = k.split('_')
        if len(parts) == 2:
            try:
                return (parts[0], int(parts[1]))
            except ValueError:
                return (parts[0], parts[1])
        return (k, 0)
    
    return sorted(list(keys), key=sort_key)


def _get_color_for_key(key: str, index: int, total: int):
    """
    Generate distinct colors for dynamic keys.
    Uses a colormap to separate different bases/methods.
    """
    # Use distinct colormaps. 
    # If few items, use 'tab10' qualitative map.
    # If many, use 'viridis' or 'plasma'.
    if total <= 10:
        cmap = mpl.colormaps['tab10']
        # Skip blue(0) and red(3) if possible since they are used by BVN/Split?
        # tab10: 0:blue, 1:orange, 2:green, 3:red, 4:purple, 5:brown, 6:pink, 7:gray, 8:olive, 9:cyan
        # We can shift index to avoid collision if desired, but simplified:
        # Base colors are manual, so keys get the cycle.
        # Let's shift by 1 to skip default blue which is close to BVN
        return cmap((index + 1) % 10) 
    else:
        cmap = mpl.colormaps['turbo']
        return cmap(index / max(1, total - 1))


# ============================================================
# Distribution Helpers
# ============================================================

def _plot_pdf_cdf_on_ax(ax, values, label: str, color=None):
    """Calculates and plots the PDF (KDE) and CDF on a given axis."""
    values = [v for v in values if v is not None]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return [], []

    # Handle constant data (no variance for KDE)
    if np.all(values == values[0]):
        line = ax.axvline(values[0], color=color or "tab:blue", lw=2, label=f"{label} (Const)")
        return [line], [f"{label} (Const)"]

    # Jitter for integer-heavy data to help KDE converge
    if np.all(np.mod(values, 1) == 0):
        values = values + np.random.normal(0, 0.1, size=values.shape)

    pdf_handle = None
    try:
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 500)
        pdf_values = kde(x_range)
        pdf_handle, = ax.plot(x_range, pdf_values, lw=2, label="PDF", color=color or "tab:blue")
        ax.fill_between(x_range, pdf_values, alpha=0.1, color=color or "tab:blue")
    except Exception:
        pass

    # Secondary Y-axis for CDF
    try:
        ax_cdf = ax.twinx()
        sorted_vals = np.sort(values)
        cdf_y = np.linspace(0, 1, len(sorted_vals))
        cdf_handle, = ax_cdf.plot(sorted_vals, cdf_y, "--", color="#555555", lw=1.5, label="CDF", alpha=0.7)
        ax_cdf.set_ylim(0, 1.05)
        # We don't want grid on the secondary axis distracting from the primary
        ax_cdf.grid(False) 
        return [pdf_handle, cdf_handle], ["PDF", "CDF"]
    except AttributeError:
        print(f"Warning: Could not create twinx on {type(ax)}")
        return [pdf_handle], ["PDF"]


# ============================================================
# Dynamic Grid Plotting
# ============================================================

def _plot_dynamic_grid(stats, out_dir, filename, title, baseline_map, radix_index):
    _prepare_plot_dir(out_dir)

    # Filter for active baselines
    active_baselines = {n: f for n, f in baseline_map.items() if any(getattr(s, f) is not None for s in stats)}
    keys = _get_all_keys(stats)

    total_plots = len(active_baselines) + len(keys)
    if total_plots == 0:
        return

    cols = 3
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes_list = axes.flatten()

    idx = 0
    # 1. Plot Core Baselines
    for name, field in active_baselines.items():
        vals = [getattr(s, field) for s in stats]
        c = COLORS_BASE.get(name, '#333333')
        h, l = _plot_pdf_cdf_on_ax(axes_list[idx], vals, name, color=c)
        axes_list[idx].set_title(f"{name} {title}")
        idx += 1

    # 2. Plot All Radix Bases
    for i, key in enumerate(keys):
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()} (B-{b})"
        else:
            label = f"Result ({key})"
        
        c = _get_color_for_key(key, i, len(keys))
            
        vals = [s.radix_multi_results[key][radix_index] for s in stats if key in s.radix_multi_results]
        h, l = _plot_pdf_cdf_on_ax(axes_list[idx], vals, label, color=c)
        axes_list[idx].set_title(f"{label} {title}")
        idx += 1

    # Hide unused axes
    for j in range(idx, len(axes_list)):
        axes_list[j].axis('off')

    plt.suptitle(f"Distribution Comparison: {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


# ============================================================
# Time-Series Plots
# ============================================================

def _plot_trend(stats, out_dir, filename, title, y_label, baseline_map, radix_index, hline=None):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)
    
    plt.figure(figsize=(12, 7))
    
    # Baselines
    for name, field in baseline_map.items():
        vals = [getattr(s, field) for s in stats]
        if all(v is None for v in vals): continue
        c = COLORS_BASE.get(name, '#000000')
        x_s, y_s = _smooth(xs, vals, window)
        plt.plot(x_s, y_s, label=name, linewidth=3.0, color=c)

    # Engines
    keys = _get_all_keys(stats)
    for i, key in enumerate(keys):
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()} (B-{b})"
        else:
            label = f"Result ({key})"
            
        c = _get_color_for_key(key, i, len(keys))

        vals = [s.radix_multi_results[key][radix_index] for s in stats if key in s.radix_multi_results]
        x_s, y_s = _smooth(xs, vals, window)
        plt.plot(x_s, y_s, label=label, linestyle="-", linewidth=2.0, alpha=0.9, color=c)

    if hline:
        plt.axhline(hline, color="#333333", linestyle="--", linewidth=1.5, label="Reference", alpha=0.7)

    plt.xlabel("Matrix Index")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_final_cycle_length(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    _plot_trend(stats, out_dir, "cycle_length_all_methods.png", 
                f"Cycle Length Trends (Smoothed) {title_suffix}", "Cycle Length",
                {"BVN": "cycle_length_bvn", "Split-Tree": "cycle_split"}, 1, hline=1.0)


def plot_final_num_permutations(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path, title_suffix: str = ""):
    _plot_trend(stats, out_dir, "permutation_count_all_methods.png", 
                f"Permutation Count Trends (Smoothed) {title_suffix}", "Number of Permutations",
                {"BVN": "num_permutations_bvn", "Split-Tree": "num_perm_split"}, 2, hline=n * n - 2 * n + 2)


def plot_runtime(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    _plot_trend(stats, out_dir, "runtime_comparison.png", 
                f"Runtime Trends (Smoothed) {title_suffix}", "Runtime (seconds)",
                {"BVN": "runtime_bvn", "Split-Tree": "runtime_split"}, 0)


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
    """
    _prepare_plot_dir(out_dir)

    plt.figure(figsize=(10, 8))

    # 1. Plot Baselines (BVN / Split-Tree)
    baselines = {
        "BVN": ("runtime_bvn", "cycle_length_bvn", COLORS_BASE['BVN'], "o"),
        "Split-Tree": ("runtime_split", "cycle_split", COLORS_BASE['Split-Tree'], "^")
    }

    for name, (rt_f, cyc_f, color, marker) in baselines.items():
        rts = [getattr(s, rt_f) for s in stats if getattr(s, rt_f) is not None]
        cycs = [getattr(s, cyc_f) for s in stats if getattr(s, cyc_f) is not None]

        if rts and cycs:
            med_rt = np.median(rts)
            med_cyc = np.median(cycs)
            plt.plot(med_cyc, med_rt, marker=marker, markersize=12,
                     label=name, color=color, alpha=0.9, linestyle='None')

    # 2. Plot Radix Bases
    keys = _get_all_keys(stats)
    if keys:
        for i, key in enumerate(keys):
            parts = str(key).split('_')
            if len(parts) == 2:
                eng, b = parts
                label = f"{eng.upper()} (B-{b})"
            else:
                label = f"Result ({key})"
            
            c = _get_color_for_key(key, i, len(keys))

            base_data = [s.radix_multi_results[key] for s in stats if key in s.radix_multi_results]
            if base_data:
                base_rts = [d[0] for d in base_data]
                base_cycs = [d[1] for d in base_data]
                med_rt = np.median(base_rts)
                med_cyc = np.median(base_cycs)
                plt.plot(med_cyc, med_rt, marker='v', markersize=11,
                            label=label, color=c, alpha=0.9, linestyle='None')

    plt.xlabel("Average Cycle Length")
    plt.ylabel("Average Runtime (Seconds)")
    plt.title(f"Efficiency Pareto (Matching: {matching_name.upper()}) {metadata}\nRuntime vs. Cycle Length")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / f"runtime_vs_cycle_{matching_name}.png", dpi=200)
    plt.close()


def plot_runtime_vs_permutation_efficiency(stats: List[DecompositionStats], out_dir: Path, matching_name: str, metadata: str = ""):
    """
    Plots Average Runtime against the Average Number of Permutations.
    """
    _prepare_plot_dir(out_dir)

    plt.figure(figsize=(10, 8))

    baselines = {
        "BVN": ("runtime_bvn", "num_permutations_bvn", COLORS_BASE['BVN'], "o"),
        "Split-Tree": ("runtime_split", "num_perm_split", COLORS_BASE['Split-Tree'], "^")
    }

    for name, (rt_f, perm_f, color, marker) in baselines.items():
        rts = [getattr(s, rt_f) for s in stats if getattr(s, rt_f) is not None]
        perms = [getattr(s, perm_f) for s in stats if getattr(s, perm_f) is not None]

        if rts and perms:
            med_rt = np.median(rts)
            med_perm = np.median(perms)
            plt.plot(med_perm, med_rt, marker=marker, markersize=12,
                     label=name, color=color, alpha=0.9, linestyle='None')

    keys = _get_all_keys(stats)
    if keys:
        for i, key in enumerate(keys):
            parts = str(key).split('_')
            if len(parts) == 2:
                eng, b = parts
                label = f"{eng.upper()} (B-{b})"
            else:
                label = f"Result ({key})"
            
            c = _get_color_for_key(key, i, len(keys))

            base_data = [s.radix_multi_results[key] for s in stats if key in s.radix_multi_results]
            if base_data:
                base_rts = [d[0] for d in base_data]
                base_perms = [d[2] for d in base_data] # Index 2 is permutations
                med_rt = np.median(base_rts)
                med_perms = np.median(base_perms)
                plt.plot(med_perms, med_rt, marker='v', markersize=11,
                         label=label, color=c, alpha=0.9, linestyle='None')

    plt.xlabel("Average Number of Permutations")
    plt.ylabel("Average Runtime (Seconds)")
    plt.title(f"Efficiency Pareto (Matching: {matching_name.upper()}) {metadata}\nRuntime vs. Permutations")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / f"runtime_vs_permutations_{matching_name}.png", dpi=200)
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

    plot_final_cycle_length(stats_list, out_dir, title_suffix)
    plot_final_num_permutations(stats_list, n, bits, out_dir, title_suffix)
    plot_runtime(stats_list, out_dir, title_suffix)
    plot_distribution_runtime(stats_list, out_dir, title_suffix)
    plot_cycle_length_distributions(stats_list, out_dir, title_suffix)
    plot_permutation_distributions(stats_list, out_dir, title_suffix)

    plot_runtime_vs_cycle_efficiency(stats_list, out_dir, matching_name=matching_method, metadata=title_suffix)
    plot_runtime_vs_permutation_efficiency(stats_list, out_dir, matching_name=matching_method, metadata=title_suffix)