from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path

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


def _plot_pdf_cdf_on_ax(
    ax,
    values,
    label: str,
    bins: int = 50,
):
    """
    Draw PDF and CDF of `values` on a single Axes.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    # --- PDF ---
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(centers, hist, linewidth=2, label=f"{label} PDF")

    # --- CDF ---
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, linestyle="--", linewidth=2, label=f"{label} CDF")

    ax.grid(True, alpha=0.3)

def plot_distribution_runtime(stats: List[DecompositionStats], out_dir: Path):
    """
    Runtime PDF + CDF for all methods OVERLAID.
    """
    _prepare_plot_dir(out_dir)

    methods = {
        "BVN": [s.runtime_bvn for s in stats],
        "Bitplane Max": [s.runtime_maximum for s in stats],
        "Bitplane WFA": [s.runtime_maximal for s in stats],
        "Split-Tree": [s.runtime_split for s in stats],
    }

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    for name, values in methods.items():
        _plot_pdf_cdf_on_ax(ax, values, label=name)

    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Density / CDF")
    ax.set_title("Runtime Distribution (PDF + CDF)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "runtime_pdf_cdf.png", dpi=220)
    plt.close()

def plot_cycle_length_distributions(stats: List[DecompositionStats], out_dir: Path):
    """
    Cycle length PDF + CDF in 2×2 subplots (one per method).
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
        _plot_pdf_cdf_on_ax(ax, values, label=name)
        ax.set_title(f"{name} Cycle Length")
        ax.set_xlabel("Cycle length")
        ax.set_ylabel("Density / CDF")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "cycle_length_pdf_cdf.png", dpi=220)
    plt.close()

def plot_permutation_distributions(stats: List[DecompositionStats], out_dir: Path):
    """
    Permutation count PDF + CDF in 2×2 subplots (one per method).
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
        _plot_pdf_cdf_on_ax(ax, values, label=name)
        ax.set_title(f"{name} Permutations")
        ax.set_xlabel("Number of permutations")
        ax.set_ylabel("Density / CDF")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "permutation_pdf_cdf.png", dpi=220)
    plt.close()

def plot_results(stats_list: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    """
    Final reporting:
      - Smoothed curves vs matrix index (comparative)
      - Distributional plots (CDF + PDF)
    """
    _prepare_plot_dir(out_dir)

    # Time-series
    plot_final_cycle_length(stats_list, out_dir=out_dir)
    plot_final_num_permutations(stats_list, n=n, bits=bits, out_dir=out_dir)
    plot_runtime(stats_list, out_dir=out_dir)

    # Distributions
    plot_distribution_runtime(stats_list, out_dir=out_dir)
    plot_cycle_length_distributions(stats_list, out_dir)
    plot_permutation_distributions(stats_list, out_dir)


