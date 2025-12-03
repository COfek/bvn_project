from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for matrix generation, BVN decomposition, and bit-plane decomposition.
    """

    n: int = 6                              # Matrix size (n × n)
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
    bitplane_method: str = "both"       # "maximum" or "maximal"

