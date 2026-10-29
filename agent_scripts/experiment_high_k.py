
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.utils.matrix_generator import generate_scaled_doubly_stochastic_matrix
from src.algorithms.radix_decomposition import decompose_radix
from src.algorithms.bvn import bvn_decomposition

def run_high_k_experiment():
    n = 256
    # Aim for average entry ~200. K = 200 * 256 = 51200
    k = 51200 
    rng = np.random.default_rng(42)

    print(f"--- Generating High-K Matrix (N={n}, K={k}) ---")
    t0 = time.perf_counter()
    # Use permutation sum generator (integer)
    # Note: generating 51,000 permutations might be slow if done one by one.
    # The generator scales a float matrix and rounds, so it should be fast.
    matrix = generate_scaled_doubly_stochastic_matrix(n, k, rng).astype(np.int64)
    gen_time = time.perf_counter() - t0
    print(f"Generation took: {gen_time:.4f}s")

    # Statistics
    max_val = matrix.max()
    mean_val = matrix.mean()
    print(f"Stats: Max={max_val}, Mean={mean_val:.2f}")
    
    # Check digit depth for Base 16
    # 255 is FF (2 digits). 4095 is FFF (3 digits).
    # If max is ~200, most entries are < 256. 
    # value 200 in Hex is C8 (2 digits).
    # value 10 in Hex is A (1 digit).
    
    print("\n--- Radix Decomposition Tests ---")
    
    bases = [2, 16, 32]
    
    for base in bases:
        print(f"\n[Base {base}]")
        t_start = time.perf_counter()
        components = decompose_radix(
            matrix,
            base=base,
            max_workers=4,
            step_strategy="max", # Robust strategy
            matching_method="maximum"
        )
        t_end = time.perf_counter()
        
        num_planes = len(components) # approximation, actually it's list of components
        # To count actual unique planes, we'd need to inspect internals, 
        # but 'components' length is effectively number of matching steps.
        # Wait, decompose_radix returns list of (weight, match).
        
        # Calculate max theoretical planes
        if max_val > 0:
            theoretical_planes = int(np.floor(np.log(max_val) / np.log(base))) + 1
        else:
            theoretical_planes = 0
            
        print(f"  Runtime: {t_end - t_start:.4f}s")
        print(f"  Total Permutations: {len(components)}")
        print(f"  Theoretical Depth (Planes): {theoretical_planes}")
        
    print("\n--- BVN Baseline (Maximum Matching) ---")
    t_bvn = time.perf_counter()
    # For very large K, BVN might be slow if it peels 1 by 1.
    # optimize: "maximum" matching peels min_val of matching.
    bvn_comps = bvn_decomposition(matrix, matching_algorithm="maximum")
    print(f"  Runtime: {time.perf_counter() - t_bvn:.4f}s")
    print(f"  Components: {len(bvn_comps)}")

if __name__ == "__main__":
    run_high_k_experiment()
