import numpy as np
import time
from src.utils.matrix_generator import generate_scaled_doubly_stochastic_matrix
from src.algorithms.bvn import bvn_decomposition, DecompositionComponent

def verify_method(name: str, matrix: np.ndarray, method: str):
    print(f"\n--- Testing {name} ({method}) ---", flush=True)
    start_time = time.perf_counter()
    try:
        components = bvn_decomposition(matrix, matching_algorithm=method)
    except Exception as e:
        print(f"Error during decomposition: {e}", flush=True)
        return

    end_time = time.perf_counter()
    
    duration = (end_time - start_time) * 1000
    print(f"Decomposition took: {duration:.2f} ms", flush=True)
    print(f"Number of components: {len(components)}", flush=True)
    
    # Verify reconstruction
    reconstructed = np.zeros_like(matrix)
    for comp in components:
        perm_matrix = comp.permutation
        # If permutation is a dict, convert to matrix (though bvn_decomposition now returns matrix)
        if isinstance(perm_matrix, dict):
             # Should be matrix in current implementation
             pass
        reconstructed += comp.weight * perm_matrix
        
    diff = np.abs(matrix - reconstructed)
    max_diff = np.max(diff)
    print(f"Max reconstruction error: {max_diff:.2e}", flush=True)
    
    if max_diff < 1e-9:
        print("[SUCCESS]: Matrix reconstructed correctly.", flush=True)
    else:
        print("[FAILURE]: Reconstruction error too high.", flush=True)

def main():
    N = 50
    K = 100
    rng = np.random.default_rng(42)
    
    print(f"Generating {N}x{N} K-Regular Matrix (K={K})...", flush=True)
    matrix = generate_scaled_doubly_stochastic_matrix(N, K, rng)
    print(f"Matrix generated. Sum: {np.sum(matrix)}", flush=True)
    
    # 1. Standard (Maximum Matching / Hungarian)
    verify_method("Standard BVN", matrix, "maximum")
    
    # 2. Greedy (Sorted Array Matching)
    verify_method("Greedy BVN", matrix, "heavy")
    
    # 3. Wavefront (WFA)
    verify_method("Wavefront BVN", matrix, "wfa")

if __name__ == "__main__":
    main()
