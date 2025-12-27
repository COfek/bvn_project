from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import seaborn as sns

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

def _get_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Finds the non-dominated points (closest to 1.0 cycle and 0.0 runtime).
    """
    # Sort by Cycle Length primarily
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    sorted_pts = points[sorted_indices]

    frontier = [sorted_pts[0]]
    for i in range(1, len(sorted_pts)):
        if sorted_pts[i, 1] < frontier[-1][1]:
            frontier.append(sorted_pts[i])
    return np.array(frontier)


def plot_runtime_vs_cycle_efficiency(stats: List[DecompositionStats], out_dir: Path):
    # Renamed to clarify Framework vs Arbiter
    methods = {
        "BVN (Standard)": ("runtime_bvn", "cycle_length_bvn", "tab:blue", "o"),
        "Bitplane-WFA": ("runtime_maximal", "cycle_maximal", "tab:green", "D"),
        "Radix8-WFA": ("runtime_radix", "cycle_radix", "tab:purple", "X"),
        "SplitTree-WFA": ("runtime_split", "cycle_split", "tab:red", "P"),
    }

    plt.figure(figsize=(11, 8))
    frontier_points = []

    for name, (rt_f, cyc_f, color, marker) in methods.items():
        rts = [getattr(s, rt_f) for s in stats if getattr(s, rt_f) is not None]
        cycs = [getattr(s, cyc_f) for s in stats if getattr(s, cyc_f) is not None]

        if rts and cycs:
            m_rt, m_cyc = np.mean(rts), np.mean(cycs)
            plt.errorbar(m_cyc, m_rt, xerr=np.std(cycs), yerr=np.std(rts),
                         fmt=marker, markersize=10, capsize=5,
                         label=name, color=color, alpha=0.9)
            frontier_points.append([m_cyc, m_rt])

    # Draw the Pareto Frontier
    if len(frontier_points) > 1:
        pts = np.array(frontier_points)
        frontier = _get_pareto_frontier(pts)
        plt.plot(frontier[:, 0], frontier[:, 1], '--', color='black', alpha=0.4,
                 label="Pareto Frontier (Efficiency Limit)")

    plt.axvline(1.0, color="gold", linestyle="-", alpha=0.5, label="Optimal Cycle Boundary")
    plt.title("Framework Efficiency: Runtime vs. Cycle Length (Arbiter: WFA)", fontsize=14)
    plt.xlabel("Average Cycle Length (Higher Cycle = More Interference)", fontsize=11)
    plt.ylabel("Average Runtime (Seconds)", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.savefig(out_dir / "efficiency_pareto_wfa.png", dpi=220)
    plt.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def plot_comprehensive_3d_surface(csv_path: Path, out_dir: Path):
    """
    Reads 12,000+ results from CSV, groups them by (N, Density),
    and draws the tradeoff surface.
    """
    df = pd.read_csv(csv_path)
    _prepare_plot_dir(out_dir)

    # 1. Define algorithms to compare
    # Map: Label -> (Cycle Col, Perm Col, Runtime Col, Color)
    algorithms = {
        "BVN": ("cycle_length_bvn", "num_permutations_bvn", "runtime_bvn", "blue"),
        "Bitplane-WFA": ("cycle_maximal", "num_perm_maximal", "runtime_maximal", "green"),
        "Radix-8": ("cycle_radix", "num_perm_radix", "runtime_radix", "purple"),
        "Split-Tree": ("cycle_split", "num_perm_split", "runtime_split", "red"),
    }

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    for name, (cyc_c, perm_c, rt_c, color) in algorithms.items():
        # Clean data: Remove rows where this algorithm wasn't run
        data = df.dropna(subset=[cyc_c, perm_c, rt_c])

        # 2. Group by N and Density to find "Centroids"
        centroids = data.groupby(['n', 'density']).agg({
            cyc_c: 'mean',
            perm_c: 'mean',
            rt_c: 'mean'
        }).reset_index()

        # 3. Plot the Scatter Centroids (the clusters)
        # We use log scale for Runtime (Z) to make N=32 vs N=256 comparable
        z_vals = np.log10(centroids[rt_c] + 1e-9)
        ax.scatter(centroids[cyc_c], centroids[perm_c], z_vals,
                   color=color, s=100, label=name, edgecolors='black', linewidth=1.5)

        # 4. Generate Surface Mesh
        if len(centroids) > 3:
            # Create a grid across the span of cycle lengths and permutation counts
            xi = np.linspace(centroids[cyc_c].min(), centroids[cyc_c].max(), 20)
            yi = np.linspace(centroids[perm_c].min(), centroids[perm_c].max(), 20)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate the Z (runtime) surface
            zi = griddata((centroids[cyc_c], centroids[perm_c]), z_vals, (xi, yi), method='linear')

            # Plot the "Efficiency Sheet" for this algorithm
            ax.plot_surface(xi, yi, zi, color=color, alpha=0.2, shade=True)

    # 5. Styling
    ax.set_xlabel('Cycle Length (Accuracy)', labelpad=10)
    ax.set_ylabel('Num Permutations (Complexity)', labelpad=10)
    ax.set_zlabel('Log10(Runtime) (seconds)', labelpad=10)
    ax.set_title(f"Comprehensive Efficiency Surface (12,000 Matrices)\nClusters: N {{32..256}} x Density {{0.3..0.9}}",
                 fontsize=16)

    ax.view_init(elev=20, azim=-135)  # Optimal angle to see the 'climb' in runtime
    ax.legend(loc='upper left', fontsize=12)

    plt.savefig(out_dir / "comprehensive_3d_landscape.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_comprehensive_2d_analysis(csv_path: Path, out_dir: Path):
    """
    Generates a 3-subplot figure analyzing networking trade-offs:
    1. Reconfigurations vs. Runtime
    2. Completion Delay (Cycle) vs. Runtime
    3. Dimension (N) vs. Runtime
    """
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mapping of algorithms to their specific columns in the CSV
    algos = {
        "BVN": ("cycle_length_bvn", "num_permutations_bvn", "runtime_bvn"),
        "Bitplane-WFA": ("cycle_maximal", "num_perm_maximal", "runtime_maximal"),
        "Radix-8": ("cycle_radix", "num_perm_radix", "runtime_radix"),
        "Split-Tree": ("cycle_split", "num_perm_split", "runtime_split")
    }

    # Set up a 1x3 grid
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    sns.set_style("whitegrid")

    for name, (cyc_col, perm_col, rt_col) in algos.items():
        # Aggregate the 1,000 matrices per configuration into mean centroids
        summary = df.groupby(['n', 'density']).agg({
            cyc_col: 'mean',
            perm_col: 'mean',
            rt_col: 'mean'
        }).reset_index()

        # Panel 1: Reconfigurations (X) vs Runtime (Y)
        sns.lineplot(data=summary, x=perm_col, y=rt_col, ax=ax1,
                     label=name, marker='o', linewidth=2.5)

        # Panel 2: Completion Delay (X) vs Runtime (Y)
        sns.lineplot(data=summary, x=cyc_col, y=rt_col, ax=ax2,
                     label=name, marker='s', linewidth=2.5)

        # Panel 3: Dimension N (X) vs Runtime (Y)
        sns.lineplot(data=summary, x='n', y=rt_col, ax=ax3,
                     label=name, marker='D', linewidth=2.5)

    # Styling Panel 1: Switching Overhead
    ax1.set_title("Switch Reconfigurations vs. Runtime", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Avg. Number of Permutations (Reconfigurations)", fontsize=12)
    ax1.set_ylabel("Avg. Runtime (sec)", fontsize=12)

    # Styling Panel 2: Traffic Latency
    ax2.set_title("Traffic Completion Delay vs. Runtime", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Avg. Cycle Length (Completion Time Ratio)", fontsize=12)
    ax2.set_ylabel("Avg. Runtime (sec)", fontsize=12)
    # Mark the ideal completion ratio
    ax2.axvline(1.0, color='black', linestyle='--', alpha=0.6, label="Ideal (1.0)")

    # Styling Panel 3: Scalability
    ax3.set_xticks([32, 64, 128, 256])
    ax3.set_title("Matrix Dimension (N) vs. Runtime", fontsize=15, fontweight='bold')
    ax3.set_xlabel("Matrix Dimension (N)", fontsize=12)
    ax3.set_ylabel("Avg. Runtime (sec)", fontsize=12)

    plt.suptitle(
        f"Network Switch Efficiency Analysis (12,000 Matrices)\nLinear Scaling: N {{32, 64, 128, 256}} | Density {{0.3, 0.6, 0.9}}",
        fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = out_dir / "network_efficiency_final_layout.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_trajectories(csv_path: Path, out_dir: Path):
    """
    Creates a trajectory plot showing how the Accuracy vs. Runtime
    trade-off evolves as the matrix size (N) increases.
    """
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map algorithms to their data columns and colors
    algos = {
        "BVN": ("cycle_length_bvn", "runtime_bvn", "tab:blue"),
        "Bitplane-WFA": ("cycle_maximal", "runtime_maximal", "tab:orange"),
        "Radix-8": ("cycle_radix", "runtime_radix", "tab:purple"),
        "Split-Tree": ("cycle_split", "runtime_split", "tab:red")
    }

    plt.figure(figsize=(12, 8))

    for name, (cyc_col, rt_col, color) in algos.items():
        # Group by N to find the mean performance at each scale
        # (This averages out the density variations to show the pure N-scaling trend)
        trajectory = df.groupby('n').agg({cyc_col: 'mean', rt_col: 'mean'}).reset_index()

        # Plot the line connecting N=32 -> N=256
        plt.plot(trajectory[cyc_col], trajectory[rt_col], color=color,
                 label=name, marker='o', linewidth=3, markersize=8, alpha=0.9)

        # Label each point with its specific N size
        for i, row in trajectory.iterrows():
            plt.annotate(f"N={int(row['n'])}",
                         (row[cyc_col], row[rt_col]),
                         textcoords="offset points",
                         xytext=(5, 10),
                         ha='center',
                         fontsize=10,
                         fontweight='bold',
                         color=color)

    # Styling for clarity
    plt.axvline(1.0, color='black', linestyle='--', alpha=0.5, label="Ideal Delay (1.0)")
    plt.xlabel("Average Traffic Completion Delay (Cycle Length)", fontsize=12)
    plt.ylabel("Average Computation Runtime (Seconds)", fontsize=12)
    plt.title("Scaling Trajectories: Performance Stability from N=32 to N=256", fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Save the final result
    save_path = out_dir / "scaling_trajectories_accuracy_vs_cost.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_results(stats_list: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    _prepare_plot_dir(out_dir)
    plot_final_cycle_length(stats_list, out_dir)
    plot_final_num_permutations(stats_list, n, bits, out_dir)
    plot_runtime(stats_list, out_dir)
    plot_distribution_runtime(stats_list, out_dir)
    plot_cycle_length_distributions(stats_list, out_dir)
    plot_permutation_distributions(stats_list, out_dir)
    plot_runtime_vs_cycle_efficiency(stats_list, out_dir)
