
import time
import numpy as np
import sys
from src.utils.matrix_generator import generate_sinkhorn_matrix
from src.algorithms.radix_decomposition import decompose_radix

def debug_performance():
    n = 256
    k = 512
    precision = 8 # Irrelevant for integer but kept for signature
    
    print(f"--- Debugging Performance (N={n}, K={k}, Type=PermutationSum) ---")
    
    # 1. Test Generation Time
    print("1. Generating Permutation Sum Matrix...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    # Generate integer matrix directly
    from src.utils.matrix_generator import generate_scaled_doubly_stochastic_matrix
    matrix = generate_scaled_doubly_stochastic_matrix(n, k, rng).astype(np.float64)
    # Norm input false for integers
    normalize = False
    
    gen_time = time.perf_counter() - t0
    print(f"   Generation took: {gen_time:.4f} seconds")
    
    bases = [2, 4, 8, 16, 32]
    for base in bases:
        print(f"2. Decomposing (Radix Base {base}, Maximum Matching, Strategy=min)...")
        t1 = time.perf_counter()
        components = decompose_radix(
            matrix, 
            base=base, 
            matching_method="maximum", 
            step_strategy="min",
            float_precision_bits=precision, 
            normalize_input=normalize
        )
        dt = time.perf_counter() - t1
        # Since we can't easily count planes from outside without modifying library, 
        # let's infer it from max value of matrix for explanation
        max_val = np.max(matrix)
        est_planes = int(np.floor(np.log(max_val) / np.log(base))) + 1

        print(f"   Base {base} took: {dt:.4f} seconds | Components: {len(components)} | Est. Planes: {est_planes}")

    # 3. Test BVN Baseline (WFA on Scaled Matrix)
    from src.algorithms.bvn import bvn_decomposition
    # 3. Test BVN Baseline (WFA)
    from src.algorithms.bvn import bvn_decomposition
    print("3. Decomposing BVN Baseline (WFA)...")
    t2 = time.perf_counter()
    
    # No scaling for integer matrix
    scaled_matrix = matrix
    
    bvn_comps = bvn_decomposition(scaled_matrix, matching_algorithm="wfa")
    bvn_time = time.perf_counter() - t2
    print(f"   BVN Baseline (WFA) took: {bvn_time:.4f} seconds | Components: {len(bvn_comps)}")
    
    # 4. Test BVN Baseline (Maximum Matching)
    print("4. Decomposing BVN Baseline (Maximum Matching)...")
    t3 = time.perf_counter()
    bvn_max_comps = bvn_decomposition(scaled_matrix, matching_algorithm="maximum")
    bvn_max_time = time.perf_counter() - t3
    print(f"   BVN Baseline (Maximum) took: {bvn_max_time:.4f} seconds | Components: {len(bvn_max_comps)}")

if __name__ == "__main__":
    debug_performance()



