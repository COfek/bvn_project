import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BvN-Arbiter: High-Performance Matrix Decomposition Suite"
    )
    
    # Core Arguments from README
    parser.add_argument("--size", "-n", type=int, default=128, dest="n", 
                        help="Dimension of the N x N traffic matrix.")
    parser.add_argument("--density", "-d", type=float, default=0.5,
                        help="Probability (0.0 to 1.0) of a matrix entry being non-zero.")
    parser.add_argument("--fixed-k", type=int, default=None,
                        help="Force generate K-regular matrices with specific K (overrides density).")

    parser.add_argument("--engine", "-e", type=str, default="all", 
                        choices=["wfa", "max", "heavy", "all", "wfa_bvn", "shuffled", "heavy_bvn"],
                        help="Matching algorithm: wfa, max, heavy, all, wfa_bvn, or shuffled.")
    parser.add_argument("--samples", "-s", type=int, default=10, dest="samples",
                        help="Number of random matrices to test per engine.")
    parser.add_argument("--output", "-o", type=str, default="run",
                        help="Root directory path for saving generated plots and CSV logs.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enables detailed logging.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable automatic plot generation.")
    parser.add_argument("--generator", type=str, default="standard",
                        choices=["standard", "binary", "sinkhorn", "weighted"],
                        help="Matrix generation method: standard (sum of K perms), binary (weighted perms), sinkhorn, weighted (sum of weighted K-regular layers).")
    parser.add_argument("--binary-bits", type=int, default=8,
                        help="Number of bit-planes for binary generator (precision limit ~53).")
    
    # Weighted Generator Arguments
    parser.add_argument("--weights", type=int, nargs='*', default=[],
                        help="List of weights for weighted generator (e.g. 1 16).")
    parser.add_argument("--sub-k", type=int, nargs='*', default=[],
                        help="List of per-layer K values (or single value) for weighted generator.")

    # Advanced / Legacy Arguments
    parser.add_argument("--radix-bases", type=int, nargs='*', default=[2,4,8,16,32],
                        help="Radix bases to test (defaults to [2] i.e., bitplane).")

    parser.add_argument("--random-seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-workers", type=int, default=None, help="Workers for Radix planes.")
    
    # Split-tree arguments (Hidden/Advanced)
    parser.add_argument("--split-sparsity-target", type=int, default=3)
    parser.add_argument("--split-max-depth", type=int, default=1)
    parser.add_argument("--split-p", type=float, default=0.5)
    parser.add_argument("--split-cv-threshold", type=float, default=0.15)
    parser.add_argument("--split-min-matching-frac", type=float, default=0.8)
    parser.add_argument("--split-method", type=str, default="pivot", choices=["pivot", "random"])
    
    # Plotting utilities
    parser.add_argument("--plot-from-csv", type=str, default=None,
                        help="Path to results CSV file. If set, generates plots for this CSV and exits.")

    return parser.parse_args()
