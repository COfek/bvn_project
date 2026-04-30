from __future__ import annotations

import matplotlib as mpl

mpl.use('Agg')
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
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
    'grid.alpha': 0.5,           # Semi-transparent
    'grid.color': '#000000',     # Black grid for clarity
    'grid.linestyle': '--',      # Dashed line
    'grid.linewidth': 0.5,
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
}

# ============================================================
# Utilities
# ============================================================

def _prepare_plot_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def _bvn_legend_label(stats: List[DecompositionStats], default: str = "BVN") -> str:
    """
    Returns the legend label to use for the BVN baseline series.

    When stats include a recorded matching algorithm (new CSVs), the label
    becomes e.g. "WFA-BVN" - parallel to the radix labels like "WFA-B2".
    Falls back to plain "BVN" for legacy CSVs that predate the bvn_matching
    field, so old runs still re-plot cleanly.
    """
    for s in stats:
        m = getattr(s, "bvn_matching", None)
        if m:
            return f"{m.upper()}-{default}"
    return default


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


def _get_pareto_frontier(points):
    """
    Finds the Pareto frontier for minimizing both X and Y.
    Points: list of (x, y) tuples.
    Returns: list of (x, y) sorted by x.
    """
    # Sort by X (ascending), then Y (ascending)
    # If multiple points have same X, we only care about the one with min Y.
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    
    frontier = []
    min_y_so_far = float('inf')
    
    for x, y in sorted_points:
        if y < min_y_so_far:
            frontier.append((x, y))
            min_y_so_far = y
            
    return frontier


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

def _plot_dynamic_grid(stats, out_dir, filename, title, baseline_map, radix_index, k=1):
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
        if "cycle" in field:
            vals = []
            for s in stats:
                if getattr(s, field) is not None:
                     norm_k = s.cycle_length_bvn if s.cycle_length_bvn and s.cycle_length_bvn > 0 else k
                     vals.append(getattr(s, field) / norm_k)
        else:
            vals = [getattr(s, field) for s in stats]
            
        c = COLORS_BASE.get(name, '#333333')
        # Display the matching algo on the BVN baseline (e.g. "WFA-BVN")
        display_name = _bvn_legend_label(stats) if name == "BVN" else name
        h, l = _plot_pdf_cdf_on_ax(axes_list[idx], vals, display_name, color=c)
        axes_list[idx].set_title(f"{display_name} {title}")
        axes_list[idx].grid(True, which='both', linestyle='--', alpha=0.5)
        axes_list[idx].minorticks_on()
        idx += 1

    # 2. Plot All Radix Bases
    for i, key in enumerate(keys):
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()}-B{b}"
        else:
            label = f"Result ({key})"
        
        c = _get_color_for_key(key, i, len(keys))
            
        if radix_index == 1:
             vals = []
             for s in stats:
                 if key in s.radix_multi_results:
                     norm_k = s.cycle_length_bvn if s.cycle_length_bvn and s.cycle_length_bvn > 0 else k
                     vals.append(s.radix_multi_results[key][radix_index] / norm_k)
        else:
             vals = [s.radix_multi_results[key][radix_index] for s in stats if key in s.radix_multi_results]
             
        h, l = _plot_pdf_cdf_on_ax(axes_list[idx], vals, label, color=c)
        axes_list[idx].set_title(f"{label} {title}")
        axes_list[idx].grid(True, which='both', linestyle='--', alpha=0.5)
        # Ensure minor ticks are on for the grid to be fully 'clear' if zoom is large
        axes_list[idx].minorticks_on()
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

def _plot_trend(stats, out_dir, filename, title, y_label, baseline_map, radix_index, hline=None, k=1):
    _prepare_plot_dir(out_dir)
    xs = np.array([s.matrix_index for s in stats])
    window = max(5, len(stats) // 50)
    
    plt.figure(figsize=(12, 7))
    
    # Baselines
    for name, field in baseline_map.items():
        if "Cycle" in y_label:
             vals = []
             for s in stats:
                 if getattr(s, field) is not None:
                     norm_k = s.cycle_length_bvn if s.cycle_length_bvn and s.cycle_length_bvn > 0 else k
                     vals.append(getattr(s, field) / norm_k)
                 else:
                     vals.append(None)
        else:
             vals = [getattr(s, field) for s in stats]
             
        if all(v is None for v in vals): continue
        c = COLORS_BASE.get(name, '#000000')
        # Display the matching algo on the BVN baseline (e.g. "WFA-BVN")
        display_name = _bvn_legend_label(stats) if name == "BVN" else name
        x_s, y_s = _smooth(xs, vals, window)
        plt.plot(x_s, y_s, label=display_name, linewidth=3.0, color=c)

    # Engines
    keys = _get_all_keys(stats)
    for i, key in enumerate(keys):
        parts = str(key).split('_')
        if len(parts) == 2:
            eng, b = parts
            label = f"{eng.upper()}-B{b}"
        else:
            label = f"Result ({key})"
            
        c = _get_color_for_key(key, i, len(keys))

        if radix_index == 1:
            vals = []
            for s in stats:
                if key in s.radix_multi_results:
                     norm_k = s.cycle_length_bvn if s.cycle_length_bvn and s.cycle_length_bvn > 0 else k
                     vals.append(s.radix_multi_results[key][radix_index] / norm_k)
        else:
            vals = [s.radix_multi_results[key][radix_index] for s in stats if key in s.radix_multi_results]
            
        x_s, y_s = _smooth(xs, vals, window)
        plt.plot(x_s, y_s, label=label, linestyle="-", linewidth=2.0, alpha=0.9, color=c)

    if hline:
        plt.axhline(hline, color="#333333", linestyle="--", linewidth=1.5, label="Reference", alpha=0.7)

    plt.xlabel("Matrix Index")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_final_cycle_length(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = "", k: int = 1):
    _plot_trend(stats, out_dir, "cycle_length_all_methods.png", 
                f"Cycle Length Trends (Smoothed) {title_suffix}", "Cycle Length (Normalized)",
                {"BVN": "cycle_length_bvn"}, 1, hline=1.0, k=k)


def plot_final_num_permutations(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path, title_suffix: str = ""):
    _plot_trend(stats, out_dir, "permutation_count_all_methods.png", 
                f"Permutation Count Trends (Smoothed) {title_suffix}", "Number of Permutations",
                {"BVN": "num_permutations_bvn"}, 2, hline=None)


def plot_runtime(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    _plot_trend(stats, out_dir, "runtime_comparison.png", 
                f"Runtime Trends (Smoothed) {title_suffix}", "Runtime (seconds)",
                {"BVN": "runtime_bvn"}, 0)


# ============================================================
# Plot Orchestration
# ============================================================

def plot_distribution_runtime(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    b_map = {"BVN": "runtime_bvn"}
    _plot_dynamic_grid(stats, out_dir, "runtime_pdf_cdf_subplots.png", f"Runtime {title_suffix}", b_map, 0)


def plot_cycle_length_distributions(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = "", k: int = 1):
    bvn_map = {"BVN": "cycle_length_bvn"}
    _plot_dynamic_grid(stats, out_dir, "cycle_length_pdf_cdf.png", f"Cycle Length (Normalized) {title_suffix}", bvn_map, 1, k=k)


def plot_permutation_distributions(stats: List[DecompositionStats], out_dir: Path, title_suffix: str = ""):
    b_map = {"BVN": "num_permutations_bvn"}
    _plot_dynamic_grid(stats, out_dir, "permutation_pdf_cdf.png", f"Permutations {title_suffix}", b_map, 2)


def plot_runtime_vs_cycle_efficiency(stats: List[DecompositionStats], out_dir: Path, matching_name: str, metadata: str = "", k: int = 1):
    """
    Plots Average Runtime against the Average Cycle Length with Error Bars.
    Includes a Pareto Frontier connecting the non-dominated points.
    """
    _prepare_plot_dir(out_dir)

    plt.figure(figsize=(12, 8))
    
    all_points = [] # (x, y) for pareto

    # Add vertical line for Optimal Cycle = 1.0 (if relevant for the x-axis)
    # plt.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, label="Optimal Cycle", alpha=0.8, zorder=1)

    # 1. Plot Baselines (BVN). Label dynamically reflects which matching
    # algorithm BVN actually used - e.g. "WFA-BVN" - parallel to radix labels.
    bvn_label = _bvn_legend_label(stats)
    baselines = {
        bvn_label: ("runtime_bvn", "cycle_length_bvn", COLORS_BASE['BVN'], "o")
    }

    for name, (rt_f, cyc_f, color, marker) in baselines.items():
        rts = []
        cycs = []
        for s in stats:
             if getattr(s, rt_f) is not None and getattr(s, cyc_f) is not None:
                 base_k = s.cycle_length_bvn if s.cycle_length_bvn and s.cycle_length_bvn > 0 else k
                 rts.append(getattr(s, rt_f))
                 cycs.append(getattr(s, cyc_f) / base_k)

        if rts and cycs:
            rts = np.array(rts)
            cycs = np.array(cycs)
            mean_rt = np.mean(rts)
            std_rt = np.std(rts)
            mean_cyc = np.mean(cycs)
            std_cyc = np.std(cycs)
            
            plt.errorbar(mean_cyc, mean_rt, xerr=std_cyc, yerr=std_rt, 
                         fmt=marker, markersize=10, label=name, color=color, 
                         alpha=0.9, capsize=5, elinewidth=1.5, zorder=5)
            all_points.append((mean_cyc, mean_rt))

    # 2. Plot Radix Bases
    keys = _get_all_keys(stats)
    if keys:
        for i, key in enumerate(keys):
            parts = str(key).split('_')
            c = _get_color_for_key(key, i, len(keys))
            
            base_data = []  # stores (runtime, cycle_len, bvn_cycle, planes)
            for s in stats:
                if key in s.radix_multi_results:
                     res = s.radix_multi_results[key]
                     bvn = s.cycle_length_bvn if s.cycle_length_bvn and s.cycle_length_bvn > 0 else k
                     planes = res[3] if len(res) >= 4 else 0
                     base_data.append((res[0], res[1], bvn, planes))

            if base_data:
                base_rts = np.array([d[0] for d in base_data])
                base_cycs = np.array([d[1] / d[2] for d in base_data])
                base_planes = np.array([d[3] for d in base_data])

                mean_rt = np.mean(base_rts)
                std_rt = np.std(base_rts)
                mean_cyc = np.mean(base_cycs)
                std_cyc = np.std(base_cycs)

                all_points.append((mean_cyc, mean_rt))

                label = f"Result ({key})"
                marker_style = 'v'
                base_str = None

                if len(parts) == 2:
                    eng, b = parts
                    label = f"{eng.upper()}-B{b}"
                    base_str = b

                plt.errorbar(mean_cyc, mean_rt, xerr=std_cyc, yerr=std_rt,
                             fmt=marker_style, markersize=9, label=label, color=c,
                             alpha=0.9, capsize=5, elinewidth=1.5, zorder=5)

                # Annotate marker with "B{base}, {N}p" for accessibility (color
                # alone isn't enough for color-blind readers). Falls back to just
                # "B{base}" on legacy CSVs that have no plane count.
                annot_parts = []
                if base_str is not None:
                    annot_parts.append(f"B{base_str}")
                if base_planes.size > 0 and base_planes.max() > 0:
                    min_p = int(base_planes.min())
                    max_p = int(base_planes.max())
                    annot_parts.append(f"{min_p}p" if min_p == max_p else f"{min_p}-{max_p}p")
                if annot_parts:
                    plt.annotate(", ".join(annot_parts), (mean_cyc, mean_rt),
                                 xytext=(6, 6), textcoords='offset points',
                                 fontsize=8, color=c, alpha=0.95, zorder=6,
                                 fontweight='bold')

    # 3. Draw Pareto Frontier
    if all_points:
        frontier = _get_pareto_frontier(all_points)
        fx, fy = zip(*frontier)
        plt.plot(fx, fy, '-', color='black', alpha=0.6, linewidth=2, label="Pareto Frontier", zorder=4)

    plt.xlabel("Average Cycle Length (Normalized) (Lower is better)")
    plt.ylabel("Average Runtime (Seconds) (Lower is better)")
    plt.title(f"Efficiency Pareto (Matching: {matching_name.upper()}) {metadata}\nRuntime vs. Cycle Length")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    # plt.xscale('log') # Removed as requested
    # plt.yscale('log') # Removed as requested
    plt.savefig(out_dir / f"runtime_vs_cycle_{matching_name}.png", dpi=200)
    plt.close()


def plot_runtime_vs_permutation_efficiency(stats: List[DecompositionStats], out_dir: Path, matching_name: str, metadata: str = ""):
    """
    Plots Average Runtime against the Average Number of Permutations with Error Bars.
    Includes a Pareto Frontier connecting the non-dominated points.
    """
    _prepare_plot_dir(out_dir)

    plt.figure(figsize=(12, 8))
    
    all_points = []

    bvn_label = _bvn_legend_label(stats)
    baselines = {
        bvn_label: ("runtime_bvn", "num_permutations_bvn", COLORS_BASE['BVN'], "o")
    }

    for name, (rt_f, perm_f, color, marker) in baselines.items():
        rts = [getattr(s, rt_f) for s in stats if getattr(s, rt_f) is not None]
        perms = [getattr(s, perm_f) for s in stats if getattr(s, perm_f) is not None]

        if rts and perms:
            rts = np.array(rts)
            perms = np.array(perms)
            mean_rt = np.mean(rts)
            std_rt = np.std(rts)
            mean_perm = np.mean(perms)
            std_perm = np.std(perms)
            
            plt.errorbar(mean_perm, mean_rt, xerr=std_perm, yerr=std_rt,
                         fmt=marker, markersize=10, label=name, color=color, 
                         alpha=0.9, capsize=5, elinewidth=1.5, zorder=5)
            all_points.append((mean_perm, mean_rt))

    keys = _get_all_keys(stats)
    if keys:
        for i, key in enumerate(keys):
            parts = str(key).split('_')
            c = _get_color_for_key(key, i, len(keys))

            base_data = [s.radix_multi_results[key] for s in stats if key in s.radix_multi_results]
            if base_data:
                base_rts = np.array([d[0] for d in base_data])
                base_perms = np.array([d[2] for d in base_data])  # Index 2 is permutations
                # Plane count is index 3; tolerate legacy 3-tuples by defaulting to 0.
                base_planes = np.array([d[3] if len(d) >= 4 else 0 for d in base_data])

                mean_rt = np.mean(base_rts)
                std_rt = np.std(base_rts)
                mean_perms = np.mean(base_perms)
                std_perms = np.std(base_perms)

                all_points.append((mean_perms, mean_rt))

                label = f"Result ({key})"
                marker_style = 'v'
                base_str = None
                if len(parts) == 2:
                    eng, b = parts
                    label = f"{eng.upper()}-B{b}"
                    base_str = b

                plt.errorbar(mean_perms, mean_rt, xerr=std_perms, yerr=std_rt,
                             fmt=marker_style, markersize=9, label=label, color=c,
                             alpha=0.9, capsize=5, elinewidth=1.5, zorder=5)

                # Annotate marker with "B{base}, {N}p" for accessibility (color
                # alone isn't enough for color-blind readers). Falls back to just
                # "B{base}" on legacy CSVs that have no plane count.
                annot_parts = []
                if base_str is not None:
                    annot_parts.append(f"B{base_str}")
                if base_planes.size > 0 and base_planes.max() > 0:
                    min_p = int(base_planes.min())
                    max_p = int(base_planes.max())
                    annot_parts.append(f"{min_p}p" if min_p == max_p else f"{min_p}-{max_p}p")
                if annot_parts:
                    plt.annotate(", ".join(annot_parts), (mean_perms, mean_rt),
                                 xytext=(6, 6), textcoords='offset points',
                                 fontsize=8, color=c, alpha=0.95, zorder=6,
                                 fontweight='bold')

    # 3. Draw Pareto Frontier
    if all_points:
        frontier = _get_pareto_frontier(all_points)
        fx, fy = zip(*frontier)
        plt.plot(fx, fy, '-', color='black', alpha=0.6, linewidth=2, label="Pareto Frontier", zorder=4)

    plt.xlabel("Average Number of Permutations (Lower is better)")
    plt.ylabel("Average Runtime (Seconds) (Lower is better)")
    plt.title(f"Efficiency Pareto (Matching: {matching_name.upper()}) {metadata}\nRuntime vs. Permutations")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    # plt.xscale('log') # Removed as requested
    # plt.yscale('log') # Removed as requested
    plt.savefig(out_dir / f"runtime_vs_permutations_{matching_name}.png", dpi=200)
    plt.close()


def plot_results(
        stats_list: List[DecompositionStats],
        n: int,
        bits: int,
        out_dir: Path,
        matching_method: str,
        generator: str
):
    num_samples = len(stats_list)
    title_suffix = f"(N={n}, Samples={num_samples}, Gen={generator})"

    plot_final_cycle_length(stats_list, out_dir, title_suffix)
    plot_final_num_permutations(stats_list, n, bits, out_dir, title_suffix)
    plot_runtime(stats_list, out_dir, title_suffix)
    plot_distribution_runtime(stats_list, out_dir, title_suffix)
    plot_cycle_length_distributions(stats_list, out_dir, title_suffix)
    plot_permutation_distributions(stats_list, out_dir, title_suffix)

    plot_runtime_vs_cycle_efficiency(stats_list, out_dir, matching_name=matching_method, metadata=title_suffix, k=bits)
    plot_runtime_vs_permutation_efficiency(stats_list, out_dir, matching_name=matching_method, metadata=title_suffix)