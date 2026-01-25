from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .algorithms.bvn import bvn_decomposition
from .algorithms.radix_decomposition import decompose_radix
from .algorithms.split_tree import split_tree_decomposition
from .config import ExperimentConfig
from .utils.matrix_generator import generate_scaled_doubly_stochastic_matrix
from .utils.stats import DecompositionStats

logger = logging.getLogger(__name__)

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

    # Determine K from density if needed
    # Determine K from density if needed, or use fixed_k if provided
    if config.fixed_k is not None:
        effective_k = config.fixed_k
    elif config.density >= 0.999:
        effective_k = 5 * config.n
    else:
        # Density D ~= 1 - (1 - 1/N)^K
        # K ~= -N * ln(1-D)
        effective_k = int(-config.n * np.log(1.0 - config.density))
        effective_k = max(1, effective_k)

    matrix = generate_scaled_doubly_stochastic_matrix(
        n=config.n,
        k=effective_k,
        rng=rng,
    )

    # --- 2. BVN decomposition (Optimal Baseline) ---
    if config.engine == "all" or config.engine == "wfa_bvn":
        t0 = time.perf_counter()
        bvn_components = bvn_decomposition(matrix=matrix)
        runtime_bvn = time.perf_counter() - t0
        cycle_length_bvn = float(sum(comp.weight for comp in bvn_components))
        num_permutations_bvn = len(bvn_components)
    else:
        bvn_components = []
        runtime_bvn = None
        cycle_length_bvn = None
        num_permutations_bvn = 0

    # --- 3. Radix Decomposition ---
    radix_multi_data = {}
    
    if config.engine == "all":
        target_engines = ["wfa", "maximum", "heavy"]
    elif config.engine == "wfa_bvn":
        target_engines = ["wfa"]
    else:
        target_engines = [config.engine]

    for engine in target_engines:
        for base in config.radix_bases:
            t1 = time.perf_counter()
            radix_components = decompose_radix(
                matrix=matrix,
                base=base,
                matching_method=engine,
                max_workers=config.max_workers,
                step_strategy=getattr(config, 'radix_strategy', 'min')
            )
            radix_runtime = time.perf_counter() - t1

            c_len = float(sum(comp.weight for comp in radix_components))
            n_perm = len(radix_components)
            
            key = f"{engine}_{base}"
            radix_multi_data[key] = (radix_runtime, c_len, n_perm)

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

    return DecompositionStats(
        matrix_index=index,
        num_permutations_bvn=num_permutations_bvn,
        cycle_length_bvn=cycle_length_bvn,
        runtime_bvn=runtime_bvn,
        num_perm_split=num_split,
        cycle_split=cycle_split,
        runtime_split=runtime_split,
        radix_multi_results=radix_multi_data
    )


# ------------------------------------------------------------
# Parallel experiment runner
# ------------------------------------------------------------
def run_experiment(config: ExperimentConfig) -> List[DecompositionStats]:
    console = Console(force_terminal=True)

    logger.info(
        f"Starting Experiment | "
        f"n={config.n}, matrices={config.num_matrices}, "
        f"parallel={config.is_parallel}"
    )

    results: List[DecompositionStats] = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        refresh_per_second=10
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