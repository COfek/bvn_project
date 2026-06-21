from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import numpy as np
from rich.console import Console

from .algorithms.bvn import bvn_decomposition
from .algorithms.euler_splitting import euler_decomposition, decompose_euler_framework
from .algorithms.radix_decomposition import decompose_radix
from .config import ExperimentConfig
from src.utils.matrix_generator import generate_matrix
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

    matrix = generate_matrix(
        n=config.n,
        k=config.k,
        max_weight=config.max_weight,
        rng=rng,
        float_weights=config.float_weights,
        unit_weight=config.unit_weight,
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
    elif config.engine == "minimum_bvn":
        bvn_engine = "minimum"
    elif config.engine == "heavy_noaug_bvn":
        bvn_engine = "heavy_noaug"
    elif config.engine == "heavy_static_noaug_bvn":
        bvn_engine = "heavy_static_noaug"
    elif config.engine == "euler_bvn":
        # Baseline uses the SAME engine as the Euler leaves so that
        # "BvN on the original matrix" vs "BvN on each leaf" is a fair
        # comparison (Task 1: does splitting cut matching-extraction time?).
        bvn_engine = getattr(config, "euler_leaf_engine", "heavy_static")

    if config.engine in [
        "all", "wfa_bvn", "shuffled",
        "heavy_bvn", "heavy_static_bvn",
        "maximum_bvn", "maximum",
        "minimum_bvn",
        "heavy_noaug_bvn", "heavy_static_noaug_bvn",
        "euler_bvn",
    ]:
        t0 = time.perf_counter()
        bvn_components = bvn_decomposition(matrix=matrix, matching_algorithm=bvn_engine)
        runtime_bvn = time.perf_counter() - t0

        cycle_length_bvn = float(sum(comp.weight for comp in bvn_components))

        # Sanity check: BVN cycle length must equal the matrix scale factor S
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)
        S = float(row_sums[0])
        # if not np.allclose(row_sums, S, atol=1e-6):
        #     raise ValueError(f"Matrix index {index}: row sums vary "
        #                      f"(min={row_sums.min():.6g}, max={row_sums.max():.6g}).")
        # if not np.allclose(col_sums, S, atol=1e-6):
        #     raise ValueError(f"Matrix index {index}: col sums differ from row sums.")
        # if not np.isclose(cycle_length_bvn, S, atol=1e-6):
        #     raise ValueError(f"Matrix index {index}: BVN cycle length "
        #                      f"{cycle_length_bvn:.6f} != scale factor S={S:.6f}.")

        num_permutations_bvn = len(bvn_components)
        bvn_matching_used = bvn_engine
    else:
        bvn_components = []
        runtime_bvn = None
        cycle_length_bvn = None
        num_permutations_bvn = 0
        bvn_matching_used = None

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
    elif config.engine == "minimum_bvn":
        target_engines = ["minimum"]
    elif config.engine == "heavy_noaug_bvn":
        target_engines = ["heavy_noaug"]
    elif config.engine == "heavy_static_noaug_bvn":
        target_engines = ["heavy_static_noaug"]
    elif config.engine == "euler_bvn":
        # Radix runs with the same engine as the Euler leaves, over the
        # requested --radix-bases (pass an empty --radix-bases to skip Radix
        # and benchmark the Euler framework alone). This makes Task 2
        # (Euler leaves vs digit planes, same matching engine) a single run.
        target_engines = [getattr(config, "euler_leaf_engine", "heavy_static")]
    else:
        target_engines = [config.engine]

    radix_strategy = getattr(config, 'radix_strategy', 'min')
    strategies = ["min", "median", "max"] if radix_strategy == "all" else [radix_strategy]

    for engine in target_engines:
        if isinstance(config.radix_bases, list):
            for base in config.radix_bases:
                for strategy in strategies:
                    # Standard radix decomposition returns
                    # (components, simulated_max_plane_runtime, num_non_empty_planes)
                    radix_components, simulated_max_runtime, num_planes = decompose_radix(
                        matrix=matrix,
                        base=base,
                        matching_method=engine,
                        max_workers=config.max_workers,
                        step_strategy=strategy
                    )

                    # The runtime is now the simulated hardware parallel runtime (max plane time)
                    radix_runtime = simulated_max_runtime

                    c_len = float(sum(comp.weight for comp in radix_components))
                    n_perm = len(radix_components)

                    key = f"{engine}_{strategy}_{base}" if len(strategies) > 1 else f"{engine}_{base}"
                    radix_multi_data[key] = (radix_runtime, c_len, n_perm, num_planes)

    # --- Euler Splitting Framework (euler_bvn engine) ---
    if config.engine == "euler_bvn":
        euler_depths = getattr(config, 'euler_depths', [1])
        euler_matching = getattr(config, 'euler_leaf_engine', 'heavy_static')
        # Default split is "heuristic" (same-direction): by keeping all
        # M[i,j] copies of a pair on one side whenever possible it produces
        # SPARSER leaves than the plain/grouped alternating split, which is
        # what makes the per-leaf BvN cheaper. Other strategies remain
        # selectable via --euler-split-method for comparison runs.
        split_method = getattr(config, 'euler_split_method', 'heuristic')
        for depth in euler_depths:
            comps, split_rt, max_rt, n_leaves = decompose_euler_framework(
                matrix=matrix,
                matching_method=euler_matching,
                depth=depth,
                split_method=split_method,
                max_workers=config.max_workers,
            )
            c_len = float(sum(comp.weight for comp in comps))
            n_perm = len(comps)
            key = f"euler_{split_method}_depth{depth}"
            # 5-tuple: the trailing split_rt is the sequential time of the
            # splitting phase only; max_rt is the max single-leaf BvN time
            # (simulated parallel matching extraction).
            radix_multi_data[key] = (max_rt, c_len, n_perm, n_leaves, split_rt)

    return DecompositionStats(
        matrix_index=index,
        num_permutations_bvn=num_permutations_bvn,
        cycle_length_bvn=cycle_length_bvn,
        runtime_bvn=runtime_bvn,
        radix_multi_results=radix_multi_data,
        bvn_matching=bvn_matching_used,
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
                logger.error(f"Error at index {idx}: {e}", exc_info=True)

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
                    logger.error(f"Worker error: {e}", exc_info=True)

                completed_count += 1
                if completed_count % update_interval == 0 or completed_count == config.num_matrices:
                    percent = (completed_count / config.num_matrices) * 100
                    logger.info(f"Progress: {completed_count}/{config.num_matrices} matrices processed ({percent:.0f}%)")

    results.sort(key=lambda s: s.matrix_index)
    logger.info("Experiment completed successfully.")
    return results