from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for matrix generation, BVN decomposition, and bit-plane decomposition.
    """

    n: int = 32                            # Matrix size (n × n)
    num_matrices: int = 1000               # Number of matrices to generate
    density: float = 1.0                   # Sparsity level (1.0 = dense)
    sinkhorn_iters: int = 1000             # Sinkhorn normalization iterations
    sinkhorn_eps: float = 1e-12            # Epsilon for Sinkhorn stability
    bvn_tol: float = 1e-10                 # BVN decomposition tolerance
    bitplane_bits: int = 8                 # Bits for integer scaling
    bitplane_tol: float = 1e-9             # Tolerance for bit-plane (unused)
    random_seed: Optional[int] = 42        # Reproducibility seed
    max_workers: Optional[int] = None      # Max threads for parallel execution
    output_csv: Optional[str] = None       # Optional path to write results CSV
    bitplane_method: str = "skip"          # "maximum" or "maximal" or "both" or "skip"
    split_sparsity_target: int = 3         # Stop splitting when nnz ≤ 3
    split_max_depth: int = 1               # Max recursion depth
    split_p: float = 0.5                   # Probability used in binary split
    split_cv_threshold: float = 0.15       # Defined as the coefficient of variation (std / mean) over nonzero entries.
    split_min_matching_frac: float = 0.8   # Minimum fraction of rows/columns that must remain matchable at the current
                                           # value scale in order to allow a pivot split.
    split_method: str = "random"           # "pivot" or "random"
    skip_split = True