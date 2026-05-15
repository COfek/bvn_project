import time
from typing import List, Tuple

import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def _jit_greedy_augmented_match(
    sorted_vals: np.ndarray,
    sorted_rows: np.ndarray,
    sorted_cols: np.ndarray,
    matrix: np.ndarray,
    n: int,
    tol: float,
) -> List[Tuple[int, int]]:
    """
    Maximum matching via greedy pass followed by BFS augmenting paths.

    Phase 1 – O(nnz) greedy sweep in descending-weight order.
    Phase 2 – BFS augmenting paths for any row still unmatched after Phase 1.

    By Birkhoff's theorem every doubly-stochastic matrix has a perfect matching,
    so Phase 2 will always find augmenting paths and return a full size-n matching
    as long as the residue matrix is non-zero.

    Returns a list of (row, col) pairs, one per matched row.
    """
    row_to_col = np.full(n, -1, dtype=np.int64)
    col_to_row = np.full(n, -1, dtype=np.int64)

    # ------------------------------------------------------------------
    # Phase 1: greedy sweep through pre-sorted edges
    # ------------------------------------------------------------------
    for i in range(len(sorted_vals)):
        r = int(sorted_rows[i])
        c = int(sorted_cols[i])
        if row_to_col[r] == -1 and col_to_row[c] == -1:
            row_to_col[r] = c
            col_to_row[c] = r

    # ------------------------------------------------------------------
    # Phase 2: BFS augmenting paths for any remaining unmatched rows
    # ------------------------------------------------------------------
    bfs_queue = np.empty(n, dtype=np.int64)
    visited_col = np.empty(n, dtype=np.int64)  # stores the row that visited each col

    for r in range(n):
        if row_to_col[r] != -1:
            continue  # already matched

        # Initialise BFS state
        for j in range(n):
            visited_col[j] = -1
        bfs_queue[0] = r
        q_front = 0
        q_back = 1
        found_col = -1

        while q_front < q_back:
            cur = bfs_queue[q_front]
            q_front += 1
            stop = False
            for c in range(n):
                if matrix[cur, c] > tol and visited_col[c] == -1:
                    visited_col[c] = cur
                    if col_to_row[c] == -1:
                        # Free column found — augmenting path ends here
                        found_col = c
                        stop = True
                        break
                    # Follow matched edge to the row that owns column c
                    bfs_queue[q_back] = col_to_row[c]
                    q_back += 1
            if stop:
                break

        if found_col != -1:
            # Flip the augmenting path from found_col back to the unmatched row r
            c = found_col
            while True:
                prev_r = int(visited_col[c])
                prev_c = int(row_to_col[prev_r])
                row_to_col[prev_r] = c
                col_to_row[c] = prev_r
                if prev_c == -1:
                    break  # reached the originally unmatched row
                c = prev_c

    # Build result list
    matches = []
    for r in range(n):
        if row_to_col[r] != -1:
            matches.append((int(r), int(row_to_col[r])))
    return matches


@jit(nopython=True, nogil=True)
def _jit_greedy_match_loop(
    sorted_vals: np.ndarray,
    sorted_rows: np.ndarray,
    sorted_cols: np.ndarray,
    n: int,
) -> List[Tuple[int, int]]:
    """
    Phase-1-only greedy loop (kept for reference / backward compatibility).
    Produces a *maximal* but not necessarily *maximum* matching.
    """
    row_occupied = np.zeros(n, dtype=np.int8)
    col_occupied = np.zeros(n, dtype=np.int8)
    matches = []
    for i in range(len(sorted_vals)):
        r = sorted_rows[i]
        c = sorted_cols[i]
        if row_occupied[r] == 0 and col_occupied[c] == 0:
            matches.append((int(r), int(c)))
            row_occupied[r] = 1
            col_occupied[c] = 1
            if len(matches) == n:
                break
    return matches


def sorted_array_matching(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Single matching step: sort all non-zero edges descending, then run
    greedy + augmenting paths to obtain a maximum (size-n) matching.
    """
    n = matrix.shape[0]
    rows, cols = np.nonzero(matrix)
    if len(rows) == 0:
        return []

    vals = matrix[rows, cols]
    sort_idx = np.argsort(-vals)
    s_vals = vals[sort_idx]
    s_rows = rows[sort_idx]
    s_cols = cols[sort_idx]

    return _jit_greedy_augmented_match(s_vals, s_rows, s_cols, matrix, n, 1e-9)


def run_full_decomposition(matrix: np.ndarray):
    """Full BvN decomposition using the Sorted Array method."""
    residue = matrix.copy()
    n = residue.shape[0]
    permutations = []
    start_time = time.perf_counter()

    while np.max(residue) > 1e-9:
        match_indices = sorted_array_matching(residue)
        if not match_indices:
            break

        w = min(residue[r, c] for r, c in match_indices)
        perm_mask = np.zeros_like(matrix)
        for r, c in match_indices:
            residue[r, c] -= w
            perm_mask[r, c] = 1.0
        permutations.append((w, perm_mask))

    end_time = time.perf_counter()
    return permutations, (end_time - start_time) * 1000


@jit(nopython=True, nogil=True)
def _jit_decompose_sorted_dynamic(
    matrix: np.ndarray, tol: float = 1e-9
) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """
    JIT-compiled BvN decomposition with dynamic sorted-array matching.

    At every iteration:
      1. Extract non-zero entries and sort them descending.
      2. Run greedy + BFS augmenting paths to get a full size-n matching.
      3. Peel the minimum weight λ from all matched entries.

    Re-sorting every iteration keeps match quality high (≈ Hungarian iterations)
    while each matching step is O(n² log n) instead of O(n³).
    """
    n = matrix.shape[0]
    weights = []
    all_matches = []

    while True:
        # 1. Extract and sort non-zero entries
        rows, cols = np.nonzero(matrix > tol)
        if len(rows) == 0:
            break

        vals = np.empty(len(rows), dtype=matrix.dtype)
        for k in range(len(rows)):
            vals[k] = matrix[rows[k], cols[k]]

        sort_idx = np.argsort(-vals)
        s_rows = rows[sort_idx]
        s_cols = cols[sort_idx]
        s_vals = vals[sort_idx]

        # 2. Maximum matching: greedy + augmenting paths
        matches = _jit_greedy_augmented_match(s_vals, s_rows, s_cols, matrix, n, tol)

        if len(matches) == 0:
            break

        # 3. Find minimum weight (λ) in the matching
        min_w = np.inf
        for r, c in matches:
            val = matrix[r, c]
            if val < min_w:
                min_w = val

        if min_w <= tol:
            break

        # 4. Subtract λ from all matched entries
        formatted_matches = []
        for r, c in matches:
            matrix[r, c] = max(0.0, matrix[r, c] - min_w)
            formatted_matches.append((int(r), int(c)))

        weights.append(min_w)
        all_matches.append(formatted_matches)

    return weights, all_matches


@jit(nopython=True, nogil=True)
def _jit_decompose_sorted_static(
    matrix: np.ndarray, tol: float = 1e-9
) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """
    JIT-compiled BvN decomposition with static sorted-array matching.

    Non-zero entries are extracted and sorted ONCE at the start; the sort order
    is reused across all iterations (values are updated, order is not refreshed).
    This saves the O(nnz log nnz) sort cost per iteration at the expense of
    potentially stale ordering as the matrix evolves.

    Each iteration still obtains a maximum matching via greedy + augmenting paths.
    """
    n = matrix.shape[0]
    weights = []
    all_matches = []

    # --- One-time extraction and sort ---
    rows, cols = np.nonzero(matrix > tol)
    if len(rows) == 0:
        return weights, all_matches

    vals = np.empty(len(rows), dtype=matrix.dtype)
    for k in range(len(rows)):
        vals[k] = matrix[rows[k], cols[k]]

    sort_idx = np.argsort(-vals)
    s_rows = rows[sort_idx]
    s_cols = cols[sort_idx]
    s_vals = vals[sort_idx]  # updated each iteration, but never re-sorted
    num_edges = len(s_vals)

    while True:
        # ----------------------------------------------------------
        # Phase 1: greedy sweep on the (possibly stale) sorted list
        # ----------------------------------------------------------
        row_to_col = np.full(n, -1, dtype=np.int64)
        col_to_row = np.full(n, -1, dtype=np.int64)

        for i in range(num_edges):
            if s_vals[i] > tol:
                r = int(s_rows[i])
                c = int(s_cols[i])
                if row_to_col[r] == -1 and col_to_row[c] == -1:
                    row_to_col[r] = c
                    col_to_row[c] = r

        # Check whether the greedy found anything at all
        total_greedy = 0
        for r in range(n):
            if row_to_col[r] != -1:
                total_greedy += 1
        if total_greedy == 0:
            break

        # ----------------------------------------------------------
        # Phase 2: BFS augmenting paths for unmatched rows
        # ----------------------------------------------------------
        bfs_queue = np.empty(n, dtype=np.int64)
        visited_col = np.empty(n, dtype=np.int64)

        for r in range(n):
            if row_to_col[r] != -1:
                continue

            for j in range(n):
                visited_col[j] = -1
            bfs_queue[0] = r
            q_front = 0
            q_back = 1
            found_col = -1

            while q_front < q_back:
                cur = bfs_queue[q_front]
                q_front += 1
                stop = False
                for c in range(n):
                    if matrix[cur, c] > tol and visited_col[c] == -1:
                        visited_col[c] = cur
                        if col_to_row[c] == -1:
                            found_col = c
                            stop = True
                            break
                        bfs_queue[q_back] = col_to_row[c]
                        q_back += 1
                if stop:
                    break

            if found_col != -1:
                c = found_col
                while True:
                    prev_r = int(visited_col[c])
                    prev_c = int(row_to_col[prev_r])
                    row_to_col[prev_r] = c
                    col_to_row[c] = prev_r
                    if prev_c == -1:
                        break
                    c = prev_c

        # ----------------------------------------------------------
        # Find min weight (λ) using actual matrix values for all matches
        # ----------------------------------------------------------
        min_w = np.inf
        for r in range(n):
            c = int(row_to_col[r])
            if c != -1:
                val = matrix[r, c]
                if val < min_w:
                    min_w = val

        if min_w <= tol:
            break

        # ----------------------------------------------------------
        # Subtract λ, update matrix and s_vals
        # ----------------------------------------------------------
        formatted_matches = []
        for r in range(n):
            c = int(row_to_col[r])
            if c != -1:
                matrix[r, c] = max(0.0, matrix[r, c] - min_w)
                formatted_matches.append((int(r), c))

        # Refresh s_vals from matrix so stale entries decay to zero
        for i in range(num_edges):
            s_vals[i] = matrix[s_rows[i], s_cols[i]]

        weights.append(min_w)
        all_matches.append(formatted_matches)

    return weights, all_matches


# ------------------------------------------------------------------
# No-augment variants (greedy Phase 1 only — partial / weak matchings)
# ------------------------------------------------------------------

def sorted_array_matching_noaug(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Single matching step: sort all non-zero edges descending, then run
    greedy-only (no augmenting paths). Returns a *maximal* but not
    necessarily *perfect* matching — similar in character to WFA.
    """
    n = matrix.shape[0]
    rows, cols = np.nonzero(matrix)
    if len(rows) == 0:
        return []

    vals = matrix[rows, cols]
    sort_idx = np.argsort(-vals)
    s_vals = vals[sort_idx]
    s_rows = rows[sort_idx]
    s_cols = cols[sort_idx]

    return _jit_greedy_match_loop(s_vals, s_rows, s_cols, n)


@jit(nopython=True, nogil=True)
def _jit_decompose_sorted_dynamic_noaug(
    matrix: np.ndarray, tol: float = 1e-9
) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """
    JIT-compiled BvN decomposition with dynamic sorted-array matching,
    **without** augmenting paths.  Each iteration re-sorts non-zero edges
    and runs a greedy-only sweep (Phase 1), producing partial matchings.
    Cycle length will exceed S (weak decomposition).
    """
    n = matrix.shape[0]
    weights = []
    all_matches = []

    while True:
        rows, cols = np.nonzero(matrix > tol)
        if len(rows) == 0:
            break

        vals = np.empty(len(rows), dtype=matrix.dtype)
        for k in range(len(rows)):
            vals[k] = matrix[rows[k], cols[k]]

        sort_idx = np.argsort(-vals)
        s_rows = rows[sort_idx]
        s_cols = cols[sort_idx]
        s_vals = vals[sort_idx]

        # Phase 1 only — greedy, no BFS augmentation
        matches = _jit_greedy_match_loop(s_vals, s_rows, s_cols, n)

        if len(matches) == 0:
            break

        min_w = np.inf
        for r, c in matches:
            val = matrix[r, c]
            if val < min_w:
                min_w = val

        if min_w <= tol:
            break

        formatted_matches = []
        for r, c in matches:
            matrix[r, c] = max(0.0, matrix[r, c] - min_w)
            formatted_matches.append((int(r), int(c)))

        weights.append(min_w)
        all_matches.append(formatted_matches)

    return weights, all_matches


@jit(nopython=True, nogil=True)
def _jit_decompose_sorted_static_noaug(
    matrix: np.ndarray, tol: float = 1e-9
) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """
    JIT-compiled BvN decomposition with static sorted-array matching,
    **without** augmenting paths.  Non-zero edges are sorted once; each
    iteration runs a greedy-only sweep over the (refreshed-value, fixed-order)
    edge list.  Produces partial matchings — weak decomposition.
    """
    n = matrix.shape[0]
    weights = []
    all_matches = []

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
        # Phase 1 only: greedy sweep on the (possibly stale) sorted list
        row_to_col = np.full(n, -1, dtype=np.int64)
        col_to_row = np.full(n, -1, dtype=np.int64)

        for i in range(num_edges):
            if s_vals[i] > tol:
                r = int(s_rows[i])
                c = int(s_cols[i])
                if row_to_col[r] == -1 and col_to_row[c] == -1:
                    row_to_col[r] = c
                    col_to_row[c] = r

        total_greedy = 0
        for r in range(n):
            if row_to_col[r] != -1:
                total_greedy += 1
        if total_greedy == 0:
            break

        min_w = np.inf
        for r in range(n):
            c = int(row_to_col[r])
            if c != -1:
                val = matrix[r, c]
                if val < min_w:
                    min_w = val

        if min_w <= tol:
            break

        formatted_matches = []
        for r in range(n):
            c = int(row_to_col[r])
            if c != -1:
                matrix[r, c] = max(0.0, matrix[r, c] - min_w)
                formatted_matches.append((int(r), c))

        for i in range(num_edges):
            s_vals[i] = matrix[s_rows[i], s_cols[i]]

        weights.append(min_w)
        all_matches.append(formatted_matches)

    return weights, all_matches
