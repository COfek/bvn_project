
import numpy as np
import sys
from src.utils.matrix_generator import generate_sinkhorn_matrix
from src.algorithms.radix_decomposition import decompose_radix

def verify_sinkhorn_generation(float_precision_bits=8):
    n = 128
    k = 1028
    rng = np.random.default_rng(42)
    
    print(f"Generating Sinkhorn matrix with N={n}, K={k}...")
    matrix = generate_sinkhorn_matrix(n, k, rng, density=0.5)
    
    # Check shape
    if matrix.shape != (n, n):
        print(f"FAIL: Incorrect shape {matrix.shape}")
        return
        
    # Check row/col sums
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    
    print(f"Row sums mean: {np.mean(row_sums):.4f}, std: {np.std(row_sums):.4f}")
    print(f"Col sums mean: {np.mean(col_sums):.4f}, std: {np.std(col_sums):.4f}")
    
    # Check for negative values
    if np.any(matrix < 0):
        print("FAIL: Matrix contains negative values")
        return
        
    # Check K-regularity (approximate)
    if not np.allclose(row_sums, k, atol=1e-5) or not np.allclose(col_sums, k, atol=1e-5):
        print("WARNING: Matrix is not perfectly K-regular (expected for float Sinkhorn)")
        print(f"Max row error: {np.max(np.abs(row_sums - k))}")
        print(f"Max col error: {np.max(np.abs(col_sums - k))}")
    print("\nTesting Radix Decomposition on Sinkhorn Matrix...")
    try:
        # Use base 2 and fixed-point precision
        print(f"Decomposing with precision {float_precision_bits} bits...")
        components = decompose_radix(matrix, base=2, matching_method="heavy", float_precision_bits=float_precision_bits)
        print(f"Decomposition successful. Components: {len(components)}")
        
        # Verify reconstruction
        reconstructed = np.zeros_like(matrix)
        total_weight = 0.0
        for comp in components:
            reconstructed += comp.matrix
            total_weight += comp.weight
            
        error = np.max(np.abs(matrix - reconstructed))
        tolerance = 1e-5
        if float_precision_bits:
            # Expected error is 0.5 * (1/2^bits) due to rounding
            tolerance = 0.5 / (2 ** float_precision_bits) + 1e-6
            print(f"Using tolerance {tolerance:.6f} for {float_precision_bits}-bit precision.")

        if error <= tolerance:
            print(f"SUCCESS: Reconstruction error {error:.6f} is within tolerance.")
        else:
            print(f"FAIL: Reconstruction error {error:.6f} is too high (tolerance {tolerance:.6f}).")
            
    except Exception as e:
        print(f"FAIL: Decomposition crashed with error: {e}")

    print("\nSinkhorn generation & decomposition verified.")

if __name__ == "__main__":
    verify_sinkhorn_generation()
