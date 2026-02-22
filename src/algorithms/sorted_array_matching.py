import time
from typing import List, Tuple

import numpy as np
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
    
    This function iterates through all eligible edges (sorted by weight)
    and greedily selects edges that do not conflict with previously selected ones.
    Because it runs with `nogil=True`, multiple such matchings can occur in parallel.
    
    Args:
        sorted_vals: Array of edge weights, sorted descending.
        sorted_rows: Corresponding row indices.
        sorted_cols: Corresponding column indices.
        n: Matrix dimension (for tracking occupancy).
        
    Returns:
        List[Tuple[int, int]]: The greedy maximal matching.
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


@jit(nopython=True, nogil=True)
def _jit_decompose_sorted_static(matrix: np.ndarray, tol: float = 1e-9) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """
    JIT-compiled loop for 'heavy_static' (sorted array) decomposition.
    Optimized to extract non-zeros and sort them exactly ONCE at the start.
    """
    n = matrix.shape[0]
    weights = []
    all_matches = []

    # 1. Initial Extraction and Sort (Done exactly ONCE)
    rows, cols = np.nonzero(matrix > tol)
    if len(rows) == 0:
        return weights, all_matches

    vals = np.empty(len(rows), dtype=matrix.dtype)
    for k in range(len(rows)):
        vals[k] = matrix[rows[k], cols[k]]

    sort_idx = np.argsort(-vals)
    s_rows = rows[sort_idx]
    s_cols = cols[sort_idx]
    s_vals = vals[sort_idx]
    num_edges = len(s_vals)

    while True:
        # Greedily sweep the pre-sorted list to build a matching
        row_occupied = np.zeros(n, dtype=np.int8)
        col_occupied = np.zeros(n, dtype=np.int8)
        
        matches_r = []
        matches_c = []
        matches_idx = [] # Track which indices in s_vals we used

        for i in range(num_edges):
            if s_vals[i] > tol:
                r = s_rows[i]
                c = s_cols[i]
                
                if row_occupied[r] == 0 and col_occupied[c] == 0:
                    matches_r.append(r)
                    matches_c.append(c)
                    matches_idx.append(i)
                    row_occupied[r] = 1
                    col_occupied[c] = 1
                    
                    if len(matches_r) == n:
                        break
                        
        if len(matches_r) == 0:
            break

        # Find min weight (Lambda) for 'min' strategy
        min_w = np.inf
        for idx in matches_idx:
            if s_vals[idx] < min_w:
                min_w = s_vals[idx]
        
        if min_w <= tol:
            break

        # Format matches for return and subtract min_w
        formatted_matches = []
        for j in range(len(matches_r)):
            r = matches_r[j]
            c = matches_c[j]
            idx = matches_idx[j]
            
            # Apply subtraction
            new_val = max(0.0, matrix[r, c] - min_w)
            matrix[r, c] = new_val
            s_vals[idx] = new_val
            
            formatted_matches.append((int(r), int(c)))

        weights.append(min_w)
        all_matches.append(formatted_matches)

    return weights, all_matches


@jit(nopython=True, nogil=True)
def _jit_decompose_sorted_dynamic(matrix: np.ndarray, tol: float = 1e-9) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """
    JIT-compiled loop for 'heavy' (sorted array) decomposition.
    Resorts the non-zero edges dynamically at the start of every single iteration.
    """
    n = matrix.shape[0]
    weights = []
    all_matches = []

    while True:
        # 1. Identify non-zero slots dynamically
        rows, cols = np.nonzero(matrix > tol)
        if len(rows) == 0:
            break

        vals = np.empty(len(rows), dtype=matrix.dtype)
        for k in range(len(rows)):
            vals[k] = matrix[rows[k], cols[k]]

        # 2. Sort descending
        sort_idx = np.argsort(-vals)

        s_rows = rows[sort_idx]
        s_cols = cols[sort_idx]
        s_vals = vals[sort_idx]

        # 3. Match
        matches = _jit_greedy_match_loop(s_vals, s_rows, s_cols, n)

        if len(matches) == 0:
            break

        # 4. Find min weight
        min_w = np.inf
        for r, c in matches:
            val = matrix[r, c]
            if val < min_w:
                min_w = val

        if min_w <= tol:
            break

        # 5. Subtract and Format
        formatted_matches = []
        for r, c in matches:
            matrix[r, c] = max(0.0, matrix[r, c] - min_w)
            formatted_matches.append((int(r), int(c)))

        weights.append(min_w)
        all_matches.append(formatted_matches)

    return weights, all_matches
