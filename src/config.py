from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for integer K-regular matrix generation and
    Radix/Split-Tree decomposition experiments.
    """

    # --- Matrix Generation Parameters ---
    n: int = 128  # Dimension of the square matrix (n x n), updated default from README
    num_matrices: int = 10  # Total number of matrices, updated default from README (samples)
    # Details for the generated Matrix
    k: int = 10

    # --- Execution & Reproducibility ---
    random_seed: Optional[int] = 42

    # max_workers refers to the number of processes used inside
    # decompose_radix for parallel plane processing.
    max_workers: Optional[int] = None

    output_csv: Optional[str] = "results.csv"  # This will be derived from main.py args

    # --- Radix Parameters ---
    # List of bases to test.
    radix_bases: List[int] = field(default_factory=lambda: [2])

    # Strategy: "min", "max", "median"
    radix_strategy: str = "min"

    # --- Euler Splitting Framework Parameters ---
    # List of split depths to test (depth=d → 2^d leaf matrices).
    # depth=0 is plain BvN (no splitting); depth=1 gives 2 leaves, etc.
    euler_depths: List[int] = field(default_factory=lambda: [1])

    # Split strategy used by the Euler framework:
    # "heuristic" (same-direction, sparser leaves), "euler" (classic
    # alternating 2-colouring), "euler_grouped" (bulk O(nnz) variant),
    # or "greedy" (integral block assignment with Euler fallback).
    euler_split_method: str = "heuristic"

    # BvN matching engine used on the Euler leaves AND for the BvN baseline
    # when engine == "euler_bvn" (so baseline vs. leaves is a fair
    # comparison): "heavy", "heavy_static", "wfa", "maximum", "minimum".
    euler_leaf_engine: str = "heavy_static"

    engine: str = field(default="bvn", metadata={"help": "Decomposition engine: 'bvn', 'wfa', 'heavy', 'maximum', 'all', 'wfa_bvn'"})

    # --- Logging & Visualization ---
    plot: bool = True
    output_dir: str = "./outputs"

    # --- Parallelism Control ---
    is_parallel: bool = False
    
    # --- Matrix Weights ---
    max_weight: float = 15.0
    float_weights: bool = False
    # Scales the generated matrix by this factor after construction.
    # Use powers of the radix base (1, 16, 256, ...) to place the matrix at a
    # specific digit-plane significance level while keeping raw values in [0, max_weight].
    unit_weight: float = 1.0
