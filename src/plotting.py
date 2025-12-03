from __future__ import annotations

import os
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

from .utils.stats import DecompositionStats


def _prepare_plot_dir(out_dir: Path):
    """
    Ensure the output directory exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_final_cycle_length(stats: List[DecompositionStats], n: int, bits: int, out_dir: Path):
    """
    Plot cycle length for:
    - BVN
    - Bitplane Maximum Matching
    - Bitplane Maximal (WFA)

    Also draws theoretical upper bounds.
    """
    _prepare_plot_dir(out_dir)

    xs = [s.matrix_index for s in stats]

    # Extract values
    cycle_bvn = [s.cycle_length_bvn for s in stats]
    cycle_max = [s.cycle_maximum for s in stats]
    cycle_wfa = [s.cycle_maximal for s in stats]

    # Upper bounds
    UB_bvn = 1.0
    UB_maximum = 1.0
    UB_maximal = 1.0

    plt.figure(figsize=(12, 7))

    plt.plot(xs, cycle_bvn, label="BVN Cycle Length", linewidth=2)
    plt.plot(xs, cycle_max, label="Bitplane Maximum Cycle Length", linewidth=2)
    plt.plot(xs, cycle_wfa, label="Bitplane Maximal (WFA) Cycle Length", linewidth=2)

    # Draw bounds
    plt.axhline(UB_bvn, color="black", linestyle="--", label="Upper Bound = 1")

    plt.xlabel("Matrix Index")
    plt.ylabel("Cycle Length")
    plt.title("Cycle Length Comparison: BVN vs Bitplane Maximum vs Bitplane Maximal (WFA)")
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

    And draw upper bounds:
        - BVN: n^2 - 2n + 2
        - Bitplane methods: bits * n
    """
    _prepare_plot_dir(out_dir)

    xs = [s.matrix_index for s in stats]

    # Extract values
    perm_bvn = [s.num_permutations_bvn for s in stats]
    perm_max = [s.num_perm_maximum for s in stats]
    perm_wfa = [s.num_perm_maximal for s in stats]

    # Theoretical upper bounds
    UB_bvn = n * n - 2 * n + 2       # BVN upper bound
    UB_bitplane = bits * n           # bitplane bound

    plt.figure(figsize=(12, 7))

    plt.plot(xs, perm_bvn, label="BVN Permutations", linewidth=2)
    plt.plot(xs, perm_max, label="Bitplane Maximum Permutations", linewidth=2)
    plt.plot(xs, perm_wfa, label="Bitplane Maximal (WFA) Permutations", linewidth=2)

    # Draw bounds
    plt.axhline(UB_bvn, color="orange", linestyle="--", label=f"BVN Upper Bound = {UB_bvn}")
    plt.axhline(UB_bitplane, color="purple", linestyle="--",
                label=f"Bitplane Upper Bound = bits * n = {UB_bitplane}")

    plt.xlabel("Matrix Index")
    plt.ylabel("Number of Permutations")
    plt.title("Permutation Count Comparison: BVN vs Bitplane Methods")
    plt.grid(True)
    plt.legend()

    plt.savefig(out_dir / "permutation_count_all_methods.png", dpi=220)
    plt.close()


def plot_runtime(stats: List[DecompositionStats], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = [s.matrix_index for s in stats]

    rt_bvn  = [s.runtime_bvn for s in stats]
    rt_max  = [s.runtime_maximum for s in stats]
    rt_wfa  = [s.runtime_maximal for s in stats]

    plt.figure(figsize=(12, 7))
    plt.plot(xs, rt_bvn, label="BVN Runtime", linewidth=2)
    plt.plot(xs, rt_max, label="Bitplane Maximum Runtime", linewidth=2)
    plt.plot(xs, rt_wfa, label="Bitplane Maximal (WFA) Runtime", linewidth=2)

    plt.xlabel("Matrix Index")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison: BVN vs Maximum vs Maximal Matching")
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
    into out_dir.
    """
    _prepare_plot_dir(out_dir)

    plot_final_cycle_length(stats_list, n=n, bits=bits, out_dir=out_dir)
    plot_final_num_permutations(stats_list, n=n, bits=bits, out_dir=out_dir)
    plot_runtime(stats_list, out_dir=out_dir)


