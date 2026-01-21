from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ExperimentConfig:
    """
    Configuration for integer K-regular matrix generation and
    Radix/Split-Tree decomposition experiments.
    """

    # --- Matrix Generation Parameters ---
    n: int = 32  # Dimension of the square matrix (n x n)
    num_matrices: int = 1000  # Total number of matrices to process

    # K is the ground-truth sum for all rows and columns.
    # If K=127, it represents 7-bit precision.
    k: int = 8

    # --- Execution & Reproducibility ---
    random_seed: Optional[int] = 42

    # max_workers refers to the number of processes used inside
    # decompose_radix for parallel plane processing.
    max_workers: Optional[int] = None

    output_csv: Optional[str] = "results.csv"

    # --- Split-Tree Parameters ---
    split_sparsity_target: int = 3
    split_max_depth: int = 1
    split_p: float = 0.5
    split_cv_threshold: float = 0.15
    split_min_matching_frac: float = 0.8
    split_method: str = "pivot"  # pivot, random, or skip
    skip_split: bool = True

    # --- Radix Parameters ---
    # List of bases to test. Base 2 is mathematically identical to Bitplane.
    radix_bases: List[int] = field(default_factory=lambda: [2, 3, 8, 12, 16])

    # Strategy for selecting matching weight: "min", "max", or "median"
    radix_strategy: str = "min"

    # "heavy": heavy_node_matching_array,
    # "wfa": wavefront_matching_vectorized,
    # "maximum": maximum_matching_wrapper
    matching_method: str = "maximum"

    # --- Parallelism Control ---
    # Set to False to time individual matrices accurately using internal
    # ProcessPool parallelism for Radix planes.
    is_parallel: bool = False