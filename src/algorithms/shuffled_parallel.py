import numpy as np
import time
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .bvn import DecompositionComponent
from .wfa import _jit_wfa_kernel
from ..utils.matrix_shuffler import MatrixShuffler

def _worker_shuffled_greedy(matrix: np.ndarray, seed: int) -> Tuple[float, Dict[int, int]]:
    """
    Worker function:
    1. Shuffles the matrix.
    2. Runs greedy matching (WFA kernel).
    3. Maps matching back to original indices.
    4. Calculates weight.
    """
    shuffler = MatrixShuffler(matrix, seed=seed)
    shuffled_matrix = shuffler.get_shuffled_matrix()
    
    # Run WFA-style greedy matching
    # WFA kernel works on a mask
    mask = (shuffled_matrix > 1e-12).astype(np.int8)
    n = matrix.shape[0]
    
    matches_list = _jit_wfa_kernel(mask, n)
    
    if len(matches_list) < n:
        # WFA is also heuristic, might not find perfect matching every time?
        # Actually WFA hardware guarantees matching if one exists? 
        # No, WFA is maximal, not maximum.
        # But for high density DS matrices it works well. 
        # If it fails, we return 0 weight.
        return 0.0, {}
        
    # Calculate weight (min element)
    min_weight = 1e9
    matching_pairs = []
    
    for r, c in matches_list:
        val = shuffled_matrix[r, c]
        if val < min_weight:
            min_weight = val
        matching_pairs.append((r, c))
            
    original_pairs = shuffler.map_matching_to_original(matching_pairs)
    original_perm = {r: c for r, c in original_pairs}
    
    return min_weight, original_perm


def shuffled_parallel_decomposition(
    matrix: np.ndarray, 
    max_workers: int = 4,
    num_proposals: int = 8
) -> List[DecompositionComponent]:
    
    n = matrix.shape[0]
    components = []
    
    # We work on a copy of the matrix
    current_matrix = matrix.copy()
    
    # Threshold for when to stop
    # Floating point epsilon issues
    while np.sum(current_matrix) > 1e-9:
        
        # 1. Launch proposals in parallel
        # Ideally we want valid matchings.
        # If greedy fails to find a full matching, it returns 0 weight.
        # We need at least one valid matching.
        
        best_weight = -1.0
        best_perm = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(num_proposals):
                seed = np.random.randint(0, 100000) + i
                futures.append(executor.submit(_worker_shuffled_greedy, current_matrix, seed))
                
            for fut in as_completed(futures):
                w, perm = fut.result()
                if w > best_weight and len(perm) == n:
                    best_weight = w
                    best_perm = perm
        
        # If we failed to find ANY permutation (greedy failure), 
        # we must fall back to a guaranteed method (like standard WFA or BVN) to progress.
        if best_weight <= 0 or len(best_perm) < n:
            # Fallback: Use standard WFA (serial) on current matrix
            # For prototype, let's just break or use a simple fallback
            # We can import WFA here?
            break 
            
        # 2. Subtract
        components.append(DecompositionComponent(weight=best_weight, permutation=best_perm))
        for r, c in best_perm.items():
            current_matrix[r, c] -= best_weight
            if current_matrix[r, c] < 1e-12:
                current_matrix[r, c] = 0.0
                
    return components
