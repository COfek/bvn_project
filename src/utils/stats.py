from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class DecompositionStats:
    matrix_index: int

    # BVN stats
    num_permutations_bvn: int
    cycle_length_bvn: Optional[float]
    runtime_bvn: Optional[float]


    # Dictionary to store multiple bases:
    #   {key_str: (runtime, cycle, num_perms, num_planes[, split_time])}
    # num_planes is the number of non-empty digit planes (radix) or leaves
    # (Euler framework). The optional 5th element split_time is the
    # sequential splitting-phase time, populated only by Euler framework
    # keys ("euler_<split_method>_depth<d>"); for those keys, `runtime` is
    # the MAX single-leaf BvN time (simulated parallel extraction).
    # Older CSVs lacking the planes/split columns load with num_planes=0
    # and a 4-tuple respectively.
    radix_multi_results: Dict[str, Tuple[float, ...]] = field(default_factory=dict)

    # The matching algorithm used by the BVN baseline ("wfa", "heavy", "heavy_static",
    # "maximum"). None when BVN was not computed for this run, or for legacy CSVs that
    # predate this field. Plotting uses this to label the BVN legend entry as e.g.
    # "WFA-BVN" instead of plain "BVN", parallel to the radix labels like "WFA-B2".
    bvn_matching: Optional[str] = None