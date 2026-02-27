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
    density: float = 0.5  # Target density of the matrix


    # K is the ground-truth sum for all rows and columns.
    # If K=127, it represents 7-bit precision.
    # Note: If density is specified, k might be derived or ignored depending on generator logic
    k: int = 1028
    fixed_k: Optional[int] = 1028

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

    engine: str = field(default="bvn", metadata={"help": "Decomposition engine: 'bvn', 'wfa', 'heavy', 'maximum', 'all', 'wfa_bvn'"})

    # --- Logging & Visualization ---
    plot: bool = True
    output_dir: str = "./outputs"

    # --- Parallelism Control ---
    is_parallel: bool = False
    generator: str = "standard"
    binary_bits: int = 8
    
    # --- Weighted Generator ---
    weights: List[int] = field(default_factory=list)
    sub_k: List[int] = field(default_factory=list)
