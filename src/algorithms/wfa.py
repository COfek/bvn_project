from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

BoolMatrix = NDArray[np.bool_]


def wavefront_matching(mask: BoolMatrix) -> List[Tuple[int, int]]:
    """
    Wavefront Arbiter maximal matching (hardware-style diagonal sweep).
    Produces a GREEDY maximal matching (not maximum).
    """
    n = mask.shape[0]

    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(n, dtype=bool)
    matches: List[Tuple[int, int]] = []
    # Sweep diagonals k = i + j
    for k in range(2 * n - 1):
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                if row_free[i] and col_free[j] and mask[i, j]:
                    matches.append((i, j))
                    row_free[i] = False
                    col_free[j] = False
    return matches


from numba import jit

@jit(nopython=True, nogil=True)
def _jit_wfa_kernel(
    mask: np.ndarray, # Boolean or numeric mask
    n: int
) -> List[Tuple[int, int]]:
    """
    Numba-optimized Wavefront Arbiter Kernel.
    
    Iterates through the diagonals of the matrix (k = i + j) to find a maximal matching.
    This mimics the hardware wavefront propagation.
    
    Args:
        mask (np.ndarray): Boolean or integer mask where >0 indicates an edge.
        n (int): Matrix dimension.
        
    Returns:
        List[Tuple[int, int]]: A list of (row, col) indices representing the matching.
    """
    row_free = np.ones(n, dtype=np.int8)
    col_free = np.ones(n, dtype=np.int8)
    matches = []
    
    # Sweep diagonals k = i + j
    # Max k is 2*n - 2 (indices 0..n-1)
    for k in range(2 * n - 1):
        # Calculate range for i
        # i + j = k => j = k - i
        # 0 <= i < n AND 0 <= j < n
        # => 0 <= k - i < n => i <= k AND i > k - n
        
        start_i = max(0, k - n + 1)
        end_i = min(k + 1, n)
        
        # Iterate diagonal elements
        for i in range(start_i, end_i):
            j = k - i
            
            # Logic: If free and eligible
            # Note: Assuming mask[i, j] checks > 0 logic externally or implicit bool
            if row_free[i] == 1 and col_free[j] == 1:
                # Numba handles non-zero check on numeric types as Truthy usually,
                # but let's be explicit if possible. But inputs vary.
                if mask[i, j]:  # Works for bool and non-zero numbers
                    matches.append((i, j))
                    row_free[i] = 0
                    col_free[j] = 0
                    
    return matches


@jit(nopython=True, nogil=True)
def _jit_decompose_wfa(matrix: np.ndarray, n: int, tol: float):
    """
    Perform the entire iterative decomposition in Numba to release the GIL.
    
    This function implements the complete decomposition loop (matching -> weight calculation -> subtraction)
    entirely within compiled code. By releasing the GIL (`nogil=True`), multiple threads can execute 
    this function in parallel on different matrices (e.g. Radix planes or Split-Tree leaves), 
    achieving near-linear scaling on multi-core CPUs.

    Args:
        matrix (np.ndarray): The N x N float matrix to decompose. Modified in-place.
        n (int): Dimension of the matrix.
        tol (float): Tolerance for treating values as zero.

    Returns:
        Tuple[List[float], List[List[Tuple[int, int]]]]: 
            - A list of weights (lambda values).
            - A list of matching lists (list of (row, col) tuples).
    """
    # Clone matrix to avoid modifying the input if that's expected, 
    # but strictly speaking we modify 'x' in place in the python version.
    # To be safe and let caller handle copying, we work on 'matrix'.
    # If caller passes a copy, we are good.
    
    weights = []
    all_matches = []
    
    # We need a mutable working copy if we want to be pure, but usually we modify leaf.
    # Let's assume matrix is mutable and we can modify it.
    
    while True:
        # 1. Compute mask (implicitly or explicitly)
        # We invoke the kernel. The kernel expects a mask-like object.
        # We can pass the matrix and check > tol inside kernel, or compute mask here.
        # Computing boolean mask is fast in Numba.
        # Check if empty first
        
        has_elements = False
        for i in range(n):
            for j in range(n):
                if matrix[i, j] > tol:
                    has_elements = True
                    break
            if has_elements: break
        
        if not has_elements:
            break
            
        # Passing matrix directly to _jit_wfa_kernel which we modified to handle > 0 check
        # But wait, the kernel currently takes 'mask'. 
        # Let's pass a boolean mask to be consistent with previous kernel signature
        mask = matrix > tol
        matches = _jit_wfa_kernel(mask, n)
        
        if len(matches) == 0:
            break
            
        # 2. Find min weight (Lambda)
        lam = np.inf
        for k in range(len(matches)):
            i, j = matches[k]
            val = matrix[i, j]
            if val < lam:
                lam = val
                
        if lam <= tol:
            break
            
        # 3. Subtract
        for k in range(len(matches)):
            i, j = matches[k]
            matrix[i, j] -= lam
            if matrix[i, j] <= tol:
                matrix[i, j] = 0.0
                
        weights.append(lam)
        all_matches.append(matches)
        
    return weights, all_matches


def wavefront_matching_vectorized(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    JIT-Accelerated Wavefront Matching (Single Step).
    Kept for compatibility.
    """
    n = matrix.shape[0]
    # mask = matrix > 1e-12 # Caller usually passes boolean mask to this legacy func
    # But checking source, it's called with 'mask'.
    # _jit_wfa_kernel is robust.
    return _jit_wfa_kernel(matrix, n)

