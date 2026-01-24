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
    parser.add_argument("--engine", "-e", type=str, default="all", 
                        choices=["wfa", "max", "heavy", "all", "wfa_bvn"],
                        help="Matching algorithm: wfa, max, heavy, all, or wfa_bvn.")
    parser.add_argument("--samples", "-s", type=int, default=10, dest="samples",
                        help="Number of random matrices to test per engine.")
    parser.add_argument("--output", "-o", type=str, default="run",
                        help="Root directory path for saving generated plots and CSV logs.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enables detailed logging.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable automatic plot generation.")

    # Advanced / Legacy Arguments
    parser.add_argument("--radix-bases", type=int, nargs='+', default=[2],
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
