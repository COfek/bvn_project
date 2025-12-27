from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import itertools
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import time

import numpy as np
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from .config import ExperimentConfig
from .utils.matrix_generator import random_sparse_doubly_stochastic, split_tree_friendly_matrix
from .algorithms.bvn import bvn_decomposition
from .algorithms.radix_decomposition import decompose_radix
from .algorithms.split_tree import split_tree_decomposition
from .utils.stats import DecompositionStats
from .utils.logging_utils import LOGGER

import os
TESTING = os.environ.get("PYTEST_RUNNING", "0") == "1"

# ------------------------------------------------------------
# Bitplane method selector
# ------------------------------------------------------------
def _select_bitplane_function(method: str):
    """
    Return a bit-plane decomposition function based on method.
    """
    if method == "maximum":
        from .algorithms.bitplane_maximum import (
            bitplane_decomposition_maximum as bit_fn,
        )
    elif method == "maximal":
        from .algorithms.bitplane_maximal import (
            bitplane_decomposition_maximal as bit_fn,
        )
    else:
        raise ValueError(f"Invalid bitplane_method '{method}'. Expected maximum/maximal.")

    return bit_fn


# ------------------------------------------------------------
# Compute bitplane for a single method
# ------------------------------------------------------------
def _compute_single_bitplane(
    matrix: np.ndarray,
    config: ExperimentConfig,
    method: str,
) -> Tuple[int, float]:
    """
    Run bit-plane decomposition for a single method (maximum or maximal).
    Returns (num_permutations, cycle_length).
    """
    bit_fn = _select_bitplane_function(method)

    components = bit_fn(
        matrix=matrix,
        bits=config.bitplane_bits,
        tol=config.bitplane_tol,
        max_workers=config.max_workers,
    )

    num_perm = len(components)
    cycle = float(sum(comp.weight for comp in components))

    return num_perm, cycle


# ------------------------------------------------------------
# Worker for one matrix
# ------------------------------------------------------------
def _compute_for_index(index: int, config: ExperimentConfig) -> DecompositionStats:
    """
    Worker: generates one matrix, applies BVN + bitplane + radix + split-tree.
    """
    # 1. Reproducible Matrix Generation
    rng_seed = config.random_seed + index if config.random_seed is not None else None
    rng = np.random.default_rng(rng_seed)

    matrix = random_sparse_doubly_stochastic(
        n=config.n,
        density=config.density,
        iters=config.sinkhorn_iters,
        eps=config.sinkhorn_eps,
        rng=rng,
    )

    # --- 2. BVN decomposition (Optimal Baseline) ---
    t0 = time.perf_counter()
    bvn_components = bvn_decomposition(matrix=matrix, tol=config.bvn_tol)
    runtime_bvn = time.perf_counter() - t0
    cycle_bvn = float(sum(comp.weight for comp in bvn_components))

    # Prepare slots for Bitplane results
    num_maximum = cycle_maximum = runtime_maximum = None
    num_maximal = cycle_maximal = runtime_maximal = None
    method = config.bitplane_method

    # --- 3. Bitplane decomposition ---
    if method in ["maximum", "both"]:
        t1 = time.perf_counter()
        num_maximum, cycle_maximum = _compute_single_bitplane(matrix, config, "maximum")
        runtime_maximum = time.perf_counter() - t1

    if method in ["maximal", "both"]:
        t2 = time.perf_counter()
        num_maximal, cycle_maximal = _compute_single_bitplane(matrix, config, "maximal")
        runtime_maximal = time.perf_counter() - t2

    # --- 4. NEW: Radix Decomposition (Base 8) ---
    t_radix_start = time.perf_counter()
    # Using base=8 to group bits into heavier, fewer planes
    radix_components = decompose_radix(
        matrix=matrix,
        base=8,
        precision_bits=config.bitplane_bits,
        tol=config.bitplane_tol,
        max_workers=config.max_workers
    )
    runtime_radix = time.perf_counter() - t_radix_start

    num_radix = len(radix_components)
    cycle_radix = float(sum(comp.weight for comp in radix_components))

    # --- 5. Split-tree decomposition ---
    t3 = time.perf_counter()
    components_split = split_tree_decomposition(
        matrix,
        sparsity_target=config.split_sparsity_target,
        max_depth=config.split_max_depth,
        p_schedule=config.split_p,
        split_method=config.split_method,
        cv_threshold=config.split_cv_threshold,
        min_matching_frac=config.split_min_matching_frac,
    )
    runtime_split = time.perf_counter() - t3

    num_split = len(components_split) if components_split else 0
    cycle_split = float(sum(comp.weight for comp in components_split)) if components_split else 0.0

    # --- 6. Return unified stats ---
    return DecompositionStats(
        matrix_index=index,
        num_permutations_bvn=len(bvn_components),
        cycle_length_bvn=cycle_bvn,
        runtime_bvn=runtime_bvn,
        num_perm_maximum=num_maximum,
        cycle_maximum=cycle_maximum,
        runtime_maximum=runtime_maximum,
        num_perm_maximal=num_maximal,
        cycle_maximal=cycle_maximal,
        runtime_maximal=runtime_maximal,
        num_perm_radix=num_radix,
        cycle_radix=cycle_radix,
        runtime_radix=runtime_radix,
        num_perm_split=num_split,
        cycle_split=cycle_split,
        runtime_split=runtime_split,
    )


# ------------------------------------------------------------
# Parallel experiment runner with progress bar + logging
# ------------------------------------------------------------
def run_experiment(config: ExperimentConfig) -> List[DecompositionStats]:
    """
    Parallel experiment runner with rich progress bar and logging.
    """
    LOGGER.info(
        f"[bold yellow]Starting Experiment[/bold yellow] | "
        f"n={config.n}, matrices={config.num_matrices}, "
        f"method={config.bitplane_method }, split={config.split_method}, "
    )

    results: List[DecompositionStats] = []

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task(
            f"Processing {config.num_matrices} matrices...",
            total=config.num_matrices
        )

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [
                executor.submit(_compute_for_index, idx, config)
                for idx in range(config.num_matrices)
            ]

            for future in as_completed(futures):
                try:
                    stats = future.result()
                    results.append(stats)
                except Exception as e:
                    LOGGER.error(f"[red]Worker thread error:[/red] {e}")

                progress.update(task, advance=1)

    results.sort(key=lambda s: s.matrix_index)

    LOGGER.info("[bold green]Experiment completed successfully.[/bold green]")
    return results


def run_comprehensive_study(base_config: ExperimentConfig, out_dir: Path):
    n_values = [32, 64]#, 128, 256]
    densities = [0.3, 0.6, 0.9]
    matrices_per_config = 5

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comprehensive_results.csv"

    # Header flag for CSV
    write_header = True

    for n, dens in itertools.product(n_values, densities):
        LOGGER.info(f"[bold cyan]BATCH START:[/bold cyan] N={n}, Density={dens}")

        batch_config = base_config
        batch_config.n = n
        batch_config.density = dens
        batch_config.num_matrices = matrices_per_config
        batch_config.split_max_depth = 1

        batch_results = []

        # Adding a progress bar for the internal batch
        with Progress("[progress.description]{task.description}", BarColumn(),
                      "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn()) as progress:
            task = progress.add_task(f"N={n} D={dens}", total=matrices_per_config)

            with ThreadPoolExecutor(max_workers=batch_config.max_workers) as executor:
                futures = [executor.submit(_compute_for_index, i, batch_config) for i in range(matrices_per_config)]
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        stat_dict = res.__dict__
                        stat_dict['n'] = n
                        stat_dict['density'] = dens
                        batch_results.append(stat_dict)
                    except Exception as e:
                        LOGGER.error(f"Error: {e}")
                    progress.update(task, advance=1)

        # Save current batch and CLEAR memory
        df = pd.DataFrame(batch_results)
        df.to_csv(csv_path, mode='a', index=False, header=write_header)
        write_header = False  # Only write header once

        del batch_results  # Free up RAM for the next N-size
        LOGGER.info(f"Batch N={n}, D={dens} saved to {csv_path}")

    return csv_path
