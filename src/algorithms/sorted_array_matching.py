import numpy as np
from typing import List, Tuple
import time



def heavy_node_matching_array(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    The "New Way" Matching Step:
    1. Only looks at active nodes (value > 0).
    2. Sorts them DESCENDING by value (Bigger Elements First).
    3. Iterates and selects non-conflicting nodes (No same row/col).
    4. Returns the list of matches (r, c).
    """
    n = matrix.shape[0]

    # 1. Identify non-zero slots (the 'nodes')
    rows, cols = np.nonzero(matrix)
    vals = matrix[rows, cols]

    # 2. The Sort: This ensures we always try to pick the largest elements first
    # This is where we prioritize "zeroing out" the most mass
    nodes = sorted(zip(vals, rows, cols), key=lambda x: x[0], reverse=True)

    row_occupied = np.zeros(n, dtype=bool)
    col_occupied = np.zeros(n, dtype=bool)
    matches = []

    # 3. The Greedy Selection Loop
    for val, r, c in nodes:
        # Check for conflicts (edges in your Rook's Graph)
        if not row_occupied[r] and not col_occupied[c]:
            matches.append((int(r), int(c)))  # Return as (row, col)
            row_occupied[r] = True
            col_occupied[c] = True

            # If we reach N matches, we have a full permutation!
            if len(matches) == n:
                break

    return matches


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
        match_indices = heavy_node_matching_array(residue)

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
        # some elements that hit 0. The next call to heavy_node_matching_array
        # will effectively 'pop' them because np.nonzero(residue)
        # won't include them anymore.

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    return permutations, duration_ms

