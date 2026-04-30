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

    # Dictionary to store multiple bases: {key_str: (runtime, cycle, num_perms, num_planes)}
    # num_planes is the number of non-empty digit planes the radix decomposition used.
    # Older CSVs that lack the planes column will load with num_planes=0 (sentinel for "unknown").
    radix_multi_results: Dict[str, Tuple[float, float, int, int]] = field(default_factory=dict)

    # The matching algorithm used by the BVN baseline ("wfa", "heavy", "heavy_static",
    # "maximum"). None when BVN was not computed for this run, or for legacy CSVs that
    # predate this field. Plotting uses this to label the BVN legend entry as e.g.
    # "WFA-BVN" instead of plain "BVN", parallel to the radix labels like "WFA-B2".
    bvn_matching: Optional[str] = None