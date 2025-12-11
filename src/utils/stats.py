from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DecompositionStats:
    """
    Unified statistics container for BVN and bit-plane decompositions.

    Works for all bitplane_method values:
      - "maximum"
      - "maximal"
      - "both"

    Fields not relevant to the selected method are left as None.
    """

    matrix_index: int

    # BVN stats
    num_permutations_bvn: int
    cycle_length_bvn: float
    runtime_bvn: float

    # Maximum matching bit-plane results
    num_perm_maximum: Optional[int] = None
    cycle_maximum: Optional[float] = None
    runtime_maximum: Optional[float] = None

    # Maximal matching (wavefront arbiter) results
    num_perm_maximal: Optional[int] = None
    cycle_maximal: Optional[float] = None
    runtime_maximal: Optional[float] = None

    # Split-tree decomposition results
    num_perm_split: Optional[int] = None
    cycle_split: Optional[float] = None
    runtime_split: Optional[float] = None

