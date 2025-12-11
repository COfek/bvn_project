from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from pathlib import Path

from .utils.stats import DecompositionStats


def _prepare_plot_dir(out_dir: Path):
    """
    Ensure the output directory exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)


def _moving_average(values, window):
    """Simple moving average with valid mode."""
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values  # cannot smooth
    return np.convolve(values, np.ones(window) / window, mode='valid')


def _smooth(xs, ys, window):
    """Smooth both x and y so the curve stays aligned."""
    if len(ys) < window:
        return xs, ys
    ys_smooth = _moving_average(ys, window)
    xs_smooth = xs[window - 1:]
    return xs_smooth, ys_smooth


def plot_final_cycle_length(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    """
    Plot cycle length for:
    - BVN
    - Bitplane Maximum Matching
    - Bitplane Maximal (WFA)
    - Split-tree (new)

    With smoothing applied.
    """
    _prepare_plot_dir(out_dir)

    xs = np.array([s.matrix_index for s in stats])

    # Extract values
    cycle_bvn = [s.cycle_length_bvn for s in stats]
    cycle_max = [s.cycle_maximum for s in stats]
    cycle_wfa = [s.cycle_maximal for s in stats]
    cycle_split = [s.cycle_split for s in stats]

    # Determine smoothing window
    window = max(5, len(stats) // 50)

    # Smooth all curves
    xs_bvn, cycle_bvn = _smooth(xs, cycle_bvn, window)
    xs_max, cycle_max = _smooth(xs, cycle_max, window)
    xs_wfa, cycle_wfa = _smooth(xs, cycle_wfa, window)
    xs_split, cycle_split = _smooth(xs, cycle_split, window)

    # Upper bounds
    UB_bvn = 1.0

    plt.figure(figsize=(12, 7))

    plt.plot(xs_bvn, cycle_bvn, label="BVN Cycle Length", linewidth=2)
    plt.plot(xs_max, cycle_max, label="Bitplane Maximum Cycle Length", linewidth=2)
    plt.plot(xs_wfa, cycle_wfa, label="Bitplane Maximal (WFA) Cycle Length", linewidth=2)
    plt.plot(xs_split, cycle_split, label="Split-Tree Cycle Length", linewidth=2)

    # Draw bounds
    plt.axhline(UB_bvn, color="black", linestyle="--", label="Upper Bound = 1")

    plt.xlabel("Matrix Index")
    plt.ylabel("Cycle Length")
    plt.title("Cycle Length Comparison: BVN vs Bitplane Maximum vs Bitplane Maximal (WFA) vs Split-Tree")
    plt.grid(True)
    plt.legend()

    plt.savefig(out_dir / "cycle_length_all_methods.png", dpi=220)
    plt.close()


def plot_final_num_permutations(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    """
    Plot #permutations for:
    - BVN
    - Bitplane Maximum Matching
    - Bitplane Maximal (WFA)
    - Split-tree (new)

    With smoothing applied.
    """
    _prepare_plot_dir(out_dir)

    xs = np.array([s.matrix_index for s in stats])

    # Extract values
    perm_bvn = [s.num_permutations_bvn for s in stats]
    perm_max = [s.num_perm_maximum for s in stats]
    perm_wfa = [s.num_perm_maximal for s in stats]
    perm_split = [s.num_perm_split for s in stats]

    # Smoothing window
    window = max(5, len(stats) // 50)

    # Smooth all curves
    xs_bvn, perm_bvn = _smooth(xs, perm_bvn, window)
    xs_max, perm_max = _smooth(xs, perm_max, window)
    xs_wfa, perm_wfa = _smooth(xs, perm_wfa, window)
    xs_split, perm_split = _smooth(xs, perm_split, window)

    # Theoretical upper bounds
    UB_bvn = n * n - 2 * n + 2
    UB_bitplane = bits * n

    plt.figure(figsize=(12, 7))

    plt.plot(xs_bvn, perm_bvn, label="BVN Permutations", linewidth=2)
    plt.plot(xs_max, perm_max, label="Bitplane Maximum Permutations", linewidth=2)
    plt.plot(xs_wfa, perm_wfa, label="Bitplane Maximal (WFA) Permutations", linewidth=2)
    plt.plot(xs_split, perm_split, label="Split-Tree Permutations", linewidth=2)

    # Draw bounds
    plt.axhline(UB_bvn, color="orange", linestyle="--", label=f"BVN Upper Bound = {UB_bvn}")
    plt.axhline(UB_bitplane, color="purple", linestyle="--",
                label=f"Bitplane Upper Bound = bits * n = {UB_bitplane}")

    plt.xlabel("Matrix Index")
    plt.ylabel("Number of Permutations")
    plt.title("Permutation Count Comparison: BVN vs Bitplane Methods vs Split-Tree")
    plt.grid(True)
    plt.legend()

    plt.savefig(out_dir / "permutation_count_all_methods.png", dpi=220)
    plt.close()


def plot_runtime(stats: List[DecompositionStats], out_dir: Path):
    """
    Plot runtime curves with smoothing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = np.array([s.matrix_index for s in stats])

    rt_bvn  = [s.runtime_bvn for s in stats]
    rt_max  = [s.runtime_maximum for s in stats]
    rt_wfa  = [s.runtime_maximal for s in stats]
    rt_split = [s.runtime_split for s in stats]

    window = max(5, len(stats) // 50)

    xs_bvn, rt_bvn = _smooth(xs, rt_bvn, window)
    xs_max, rt_max = _smooth(xs, rt_max, window)
    xs_wfa, rt_wfa = _smooth(xs, rt_wfa, window)
    xs_split, rt_split = _smooth(xs, rt_split, window)

    plt.figure(figsize=(12, 7))
    plt.plot(xs_bvn, rt_bvn, label="BVN Runtime", linewidth=2)
    plt.plot(xs_max, rt_max, label="Bitplane Maximum Runtime", linewidth=2)
    plt.plot(xs_wfa, rt_wfa, label="Bitplane Maximal (WFA) Runtime", linewidth=2)
    plt.plot(xs_split, rt_split, label="Split-Tree Runtime", linewidth=2)

    plt.xlabel("Matrix Index")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison: BVN vs Maximum vs Maximal Matching vs Split-Tree")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir / "runtime_comparison.png", dpi=220)
    plt.close()


def plot_results(stats_list: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    """
    Higher-level entry point for final reporting: generates the two graphs described.

    Saves:
        - cycle_length_all_methods.png
        - permutation_count_all_methods.png
        - runtime_comparison.png
    """
    _prepare_plot_dir(out_dir)

    plot_final_cycle_length(stats_list, n=n, bits=bits, out_dir=out_dir)
    plot_final_num_permutations(stats_list, n=n, bits=bits, out_dir=out_dir)
    plot_runtime(stats_list, out_dir=out_dir)
