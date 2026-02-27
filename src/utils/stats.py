from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class DecompositionStats:
    matrix_index: int

    # BVN stats
    num_permutations_bvn: int
    cycle_length_bvn: float
    runtime_bvn: float

    # Dictionary to store multiple bases: {key_str: (runtime, cycle, num_perms)}
    radix_multi_results: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)