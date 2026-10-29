
import time
import numpy as np
from src.utils.matrix_generator import generate_scaled_doubly_stochastic_matrix
from src.algorithms.radix_decomposition import decompose_radix

def verify_scaling():
    n = 256
    k = 512
    scale_factor = 100 # Scale values from ~2 to ~200
    
    print(f"--- Scaling Investigation (N={n}, K={k}, Scale={scale_factor}) ---")
    
    rng = np.random.default_rng(42)
    # Generate integer matrix (Avg val ~2)
    matrix = generate_scaled_doubly_stochastic_matrix(n, k, rng).astype(np.float64)
    
    # Scaled Matrix (Avg val ~200)
    scaled_matrix = matrix * scale_factor
    print(f"Original Max: {matrix.max()}, Scaled Max: {scaled_matrix.max()}")

    # 1. Base 16 on Scaled Matrix
    print("\n1. Decomposing Scaled Matrix (Base 16)...")
    t0 = time.perf_counter()
    components = decompose_radix(
        scaled_matrix, 
        base=16, 
        matching_method="maximum", 
        step_strategy="max",
        normalize_input=False
    )
    dt = time.perf_counter() - t0
    
    # Estimate planes
    max_val = scaled_matrix.max()
    est_planes = int(np.floor(np.log(max_val) / np.log(16))) + 1
    
    print(f"   Time: {dt:.4f}s | Components: {len(components)}")
    print(f"   Est. Planes: {est_planes} (Parallelism enabled!)")
    
    # 2. Base 16 on Original Matrix (for comparison)
    print("\n2. Decomposing Original Matrix (Base 16)...")
    t1 = time.perf_counter()
    components_orig = decompose_radix(
        matrix, 
        base=16, 
        matching_method="maximum", 
        step_strategy="max", 
        normalize_input=False
    )
    dt_orig = time.perf_counter() - t1
    print(f"   Time: {dt_orig:.4f}s | Components: {len(components_orig)}")
    print(f"   Est. Planes: {int(np.floor(np.log(matrix.max()) / np.log(16))) + 1}")

if __name__ == "__main__":
    verify_scaling()
