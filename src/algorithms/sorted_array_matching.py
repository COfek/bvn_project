import numpy as np
from typing import List, Tuple
import time



from numba import jit

@jit(nopython=True, nogil=True)
def _jit_greedy_match_loop(
    sorted_vals: np.ndarray,
    sorted_rows: np.ndarray,
    sorted_cols: np.ndarray,
    n: int
) -> List[Tuple[int, int]]:
    """
    Numba-optimized greedy matching loop.
    """
    row_occupied = np.zeros(n, dtype=np.int8)
    col_occupied = np.zeros(n, dtype=np.int8)
    matches = []

    # Iterate through the pre-sorted arrays
    for i in range(len(sorted_vals)):
        r = sorted_rows[i]
        c = sorted_cols[i]
        
        if row_occupied[r] == 0 and col_occupied[c] == 0:
            matches.append((int(r), int(c)))
            row_occupied[r] = 1
            col_occupied[c] = 1
            
            # Early exit if we have a full matching
            if len(matches) == n:
                break
                
    return matches


def sorted_array_matching(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    The "New Way" Matching Step (Accelerated):
    1. Only looks at active nodes (value > 0).
    2. Sorts them DESCENDING by value (Bigger Elements First).
    3. Iterates and selects non-conflicting nodes (No same row/col) using JIT.
    4. Returns the list of matches (r, c).
    """
    n = matrix.shape[0]

    # 1. Identify non-zero slots (the 'nodes')
    rows, cols = np.nonzero(matrix)
    if len(rows) == 0:
        return []
        
    vals = matrix[rows, cols]

    # 2. The Sort: Python's argsort is fast enough usually, or we can use numpy
    # We want descending sort, so we perform argsort on -vals
    # Note: Using stable sort might maintain tie-breaking behavior, but quicksort is faster
    sort_idx = np.argsort(-vals)  
    
    # 3. Apply sort order
    s_vals = vals[sort_idx]
    s_rows = rows[sort_idx]
    s_cols = cols[sort_idx]

    # 4. Call JIT Loop
    return _jit_greedy_match_loop(s_vals, s_rows, s_cols, n)


def run_full_decomposition(matrix: np.ndarray):
    """
    Full BVN Decomposition using the Sorted Array method.
    Includes the 'Pop' and 'Re-sort' logic.
    """
    residue = matrix.copy()
    n = residue.shape[0]
    permutations = []

    start_time = time.perf_counter()

    while np.max(residue) > 1e-9:
        # Get matching using the Sorted Array logic
        match_indices = sorted_array_matching(residue)

        if not match_indices:
            break

        # Determine the weight to subtract (min of the matching)
        w = min(residue[r, c] for r, c in match_indices)

        # Create the permutation mask and subtract weight
        perm_mask = np.zeros_like(matrix)
        for r, c in match_indices:
            residue[r, c] -= w
            perm_mask[r, c] = 1.0

        permutations.append((w, perm_mask))

        # Note: In a real simulation loop, the 'residue' now has
        # some elements that hit 0. The next call to sorted_array_matching
        # will effectively 'pop' them because np.nonzero(residue)
        # won't include them anymore.

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    return permutations, duration_ms

