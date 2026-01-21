from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List, Tuple

import time

import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn, MofNCompleteColumn

from .config import ExperimentConfig
from .utils.matrix_generator import generate_scaled_doubly_stochastic_matrix
from .algorithms.bvn import bvn_decomposition
from .algorithms.radix_decomposition import decompose_radix
from .algorithms.split_tree import split_tree_decomposition
from .utils.stats import DecompositionStats
from .utils.logging_utils import LOGGER

from typing import List

import os
TESTING = os.environ.get("PYTEST_RUNNING", "0") == "1"


# ------------------------------------------------------------
# Worker for one matrix
# ------------------------------------------------------------
def _compute_for_index(index: int, config: ExperimentConfig) -> DecompositionStats:
    """
    Worker: generates one matrix, applies BVN + bitplane + specified Radix bases + split-tree.
    """
    # 1. Reproducible Matrix Generation
    rng_seed = config.random_seed + index if config.random_seed is not None else None
    rng = np.random.default_rng(rng_seed)

    matrix = generate_scaled_doubly_stochastic_matrix(
        n=config.n,
        k=(2**config.k)-1,
        rng=rng,
    )

    # --- 2. BVN decomposition (Optimal Baseline) ---
    t0 = time.perf_counter()
    bvn_components = bvn_decomposition(matrix=matrix)
    runtime_bvn = time.perf_counter() - t0
    cycle_length_bvn = float(sum(comp.weight for comp in bvn_components))

    # --- 3. Radix Decomposition (Iterating only specific bases) ---
    radix_multi_data = {}
    for base in config.radix_bases:
        t1 = time.perf_counter()
        radix_components = decompose_radix(
            matrix=matrix,
            base=base,
            matching_method=config.matching_method,
            max_workers=config.max_workers,
            step_strategy=getattr(config, 'radix_strategy', 'min')
        )
        radix_runtime = time.perf_counter() - t1

        c_len = float(sum(comp.weight for comp in radix_components))
        n_perm = len(radix_components)
        radix_multi_data[base] = (radix_runtime, c_len, n_perm)

    # --- 4. Split-tree decomposition ---
    num_split = cycle_split = runtime_split = None
    if not config.skip_split:
        t2 = time.perf_counter()
        components_split = split_tree_decomposition(
            matrix,
            sparsity_target=config.split_sparsity_target,
            max_depth=config.split_max_depth,
            p_schedule=config.split_p,
            split_method=config.split_method,
            cv_threshold=config.split_cv_threshold,
            min_matching_frac=config.split_min_matching_frac,
        )
        runtime_split = time.perf_counter() - t2
        num_split = len(components_split) if components_split else 0
        cycle_split = float(sum(comp.weight for comp in components_split)) if components_split else 0.0

    # --- 6. Return unified stats object ---
    return DecompositionStats(
        matrix_index=index,
        num_permutations_bvn=len(bvn_components),
        cycle_length_bvn=cycle_length_bvn,
        runtime_bvn=runtime_bvn,
        num_perm_split=num_split,
        cycle_split=cycle_split,
        runtime_split=runtime_split,
        radix_multi_results=radix_multi_data
    )


# ------------------------------------------------------------
# Parallel experiment runner with progress bar + logging
# ------------------------------------------------------------
def run_experiment(config: ExperimentConfig) -> List[DecompositionStats]:
    # 1. Create a dedicated console that forces terminal features
    # This often bypasses PyCharm's "dumb terminal" limitations
    console = Console(force_terminal=True)

    LOGGER.info(
        f"[bold yellow]Starting Experiment[/bold yellow] | "
        f"n={config.n}, matrices={config.num_matrices}, "
        f"parallel={config.is_parallel}"
    )

    results: List[DecompositionStats] = []

    # 2. Configure Progress with the explicit console
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,  # Link to our forced console
        transient=False,  # Keeps the bar visible after it finishes
        refresh_per_second=10  # Ensure it updates regularly
    )

    with progress:
        task = progress.add_task(
            "[cyan]Decomposing matrices...",
            total=config.num_matrices
        )

        if not config.is_parallel:
            for idx in range(config.num_matrices):
                try:
                    stats = _compute_for_index(idx, config)
                    results.append(stats)
                except Exception as e:
                    # Use progress.console.log instead of LOGGER inside this block
                    progress.console.log(f"[bold red]Error at index {idx}:[/bold red] {e}")

                progress.update(task, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(_compute_for_index, idx, config): idx
                    for idx in range(config.num_matrices)
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        progress.console.log(f"[bold red]Worker error:[/bold red] {e}")

                    progress.update(task, advance=1)

    results.sort(key=lambda s: s.matrix_index)
    console.print("[bold green]✔ Experiment completed successfully.[/bold green]")
    return results