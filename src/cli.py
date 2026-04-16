import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BvN-Arbiter: High-Performance Matrix Decomposition Suite"
    )
    
    # Core Arguments from README
    parser.add_argument("--size", "-n", type=int, default=128, dest="n", 
                        help="Dimension of the N x N traffic matrix.")
    parser.add_argument("--k", "-k", type=int, default=10,
                        help="Number of permutations to sum for the generated matrix.")

    parser.add_argument("--engine", "-e", type=str, default="all", 
                        choices=["wfa", "max", "heavy", "heavy_static", "all", "wfa_bvn", "heavy_bvn", "heavy_static_bvn", "maximum_bvn"],
                        help="Matching algorithm: wfa, max, heavy, heavy_static, all, wfa_bvn, heavy_bvn, heavy_static_bvn, maximum_bvn.")
    parser.add_argument("--samples", "-s", type=int, default=10, dest="samples",
                        help="Number of random matrices to test per engine.")
    parser.add_argument("--output", "-o", type=str, default="run",
                        help="Root directory path for saving generated plots and CSV logs.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable automatic plot generation.")
    # Matrix Generation Weights
    parser.add_argument("--max-weight", type=float, default=1.0,
                        help="The maximum weight (W) to randomly sample from [0, W] when generating permutations.")
    parser.add_argument("--float-weights", action="store_true",
                        help="Generate float weights instead of integer weights.")

    # Advanced / Legacy Arguments
    parser.add_argument("--radix-bases", type=int, nargs='*', default=[2,4,8,16,32],
                        help="Radix bases to test (defaults to [2] i.e., bitplane).")

    parser.add_argument("--random-seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-workers", type=int, default=None, help="Workers for Radix planes.")
    
    # Plotting utilities
    parser.add_argument("--plot-from-csv", type=str, default=None,
                        help="Path to results CSV file. If set, generates plots for this CSV and exits.")

    return parser.parse_args()
