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
from .config import ExperimentConfig
from src.utils.matrix_generator import (
    generate_scaled_doubly_stochastic_matrix,
    generate_binary_weighted_matrix,
    generate_weighted_sum_matrix,
    generate_sinkhorn_matrix
)
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

    if config.generator == "binary":
        matrix = generate_binary_weighted_matrix(n=config.n, rng=rng, bits=config.binary_bits)
        # Binary generator produces fixed K based on bits.
    elif config.generator == "weighted":
        matrix = generate_weighted_sum_matrix(
            n=config.n, 
            weights=config.weights, 
            sub_k=config.sub_k, 
           rng=rng
        )
    elif config.generator == "sinkhorn":
        matrix = generate_sinkhorn_matrix(
            n=config.n,
            k=effective_k,
            rng=rng,
            density=config.density
        )
    else:
        # Standard
        matrix = generate_scaled_doubly_stochastic_matrix(
            n=config.n,
            k=effective_k,
            rng=rng,
        )

    # --- 2. BVN decomposition (Optimal Baseline) ---
    # Determine BVN engine
    bvn_engine = "maximum"
    if config.engine == "wfa_bvn":
        bvn_engine = "wfa"
    elif config.engine == "heavy_bvn":
        bvn_engine = "heavy"
    elif config.engine == "heavy_static_bvn":
        bvn_engine = "heavy_static"
    
    if config.engine in ["all", "wfa_bvn", "shuffled", "heavy_bvn", "heavy_static_bvn", "maximum_bvn", "maximum"]:
        t0 = time.perf_counter()
        
        bvn_components = bvn_decomposition(matrix=matrix, matching_algorithm=bvn_engine)
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
        target_engines = ["wfa", "maximum", "heavy", "heavy_static"]
    elif config.engine == "wfa_bvn":
        target_engines = ["wfa"]
    elif config.engine == "heavy_bvn":
        target_engines = ["heavy"]
    elif config.engine == "heavy_static_bvn":
        target_engines = ["heavy_static"]
    elif config.engine == "maximum_bvn":
        target_engines = ["maximum"]
    else:
        target_engines = [config.engine]

    for engine in target_engines:
        if isinstance(config.radix_bases, list):
            for base in config.radix_bases:
                # Standard radix decomposition returns (components, simulated_max_plane_runtime)
                radix_components, simulated_max_runtime = decompose_radix(
                    matrix=matrix,
                    base=base,
                    matching_method=engine,
                    max_workers=config.max_workers,
                    step_strategy=getattr(config, 'radix_strategy', 'min')
                )
                
                # The runtime is now the simulated hardware parallel runtime (max plane time)
                radix_runtime = simulated_max_runtime

                c_len = float(sum(comp.weight for comp in radix_components))
                n_perm = len(radix_components)
            
                key = f"{engine}_{base}"
                radix_multi_data[key] = (radix_runtime, c_len, n_perm)

    return DecompositionStats(
        matrix_index=index,
        num_permutations_bvn=num_permutations_bvn,
        cycle_length_bvn=cycle_length_bvn,
        runtime_bvn=runtime_bvn,
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

    update_interval = max(1, config.num_matrices // 20)  # Log every 5%

    if not config.is_parallel:
        for idx in range(config.num_matrices):
            try:
                stats = _compute_for_index(idx, config)
                results.append(stats)
            except Exception as e:
                logger.error(f"Error at index {idx}: {e}")

            if (idx + 1) % update_interval == 0 or (idx + 1) == config.num_matrices:
                percent = ((idx + 1) / config.num_matrices) * 100
                logger.info(f"Progress: {idx + 1}/{config.num_matrices} matrices processed ({percent:.0f}%)")
    else:
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(_compute_for_index, idx, config): idx
                for idx in range(config.num_matrices)
            }
            completed_count = 0
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Worker error: {e}")

                completed_count += 1
                if completed_count % update_interval == 0 or completed_count == config.num_matrices:
                    percent = (completed_count / config.num_matrices) * 100
                    logger.info(f"Progress: {completed_count}/{config.num_matrices} matrices processed ({percent:.0f}%)")

    results.sort(key=lambda s: s.matrix_index)
    logger.info("Experiment completed successfully.")
    return results