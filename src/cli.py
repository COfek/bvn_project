import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BvN-Arbiter: High-Performance Matrix Decomposition Suite"
    )
    
    # Core arguments
    parser.add_argument("--size", "-n", type=int, default=128, dest="n", 
                        help="Dimension of the N x N traffic matrix.")
    parser.add_argument("--k", "-k", type=int, default=10,
                        help="Number of permutations to sum for the generated matrix.")

    parser.add_argument("--engine", "-e", type=str, default="all",
                        choices=[
                            "wfa", "wwfa", "max", "heavy", "heavy_static", "all",
                            "wfa_bvn", "wwfa_bvn", "heavy_bvn", "heavy_static_bvn",
                            "maximum_bvn", "minimum_bvn",
                            "heavy_noaug_bvn", "heavy_static_noaug_bvn",
                            "euler_bvn",
                        ],
                        help=(
                            "Matching algorithm / compound mode (see README). "
                            "Diagnostic variants: minimum_bvn (min-weight Hungarian), "
                            "heavy_noaug_bvn / heavy_static_noaug_bvn (greedy without augmenting paths)."
                        ))
    parser.add_argument("--samples", "-s", type=int, default=10, dest="samples",
                        help="Number of random matrices to test per engine.")
    parser.add_argument("--output", "-o", type=str, default="run",
                        help="Root directory path for saving generated plots and CSV logs.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable automatic plot generation.")
    # Matrix Generation Weights
    parser.add_argument("--max-weight", type=float, default=15.0,
                        help="Maximum per-permutation weight sampled from [0, W]. "
                             "Keep at B-1 (e.g. 15 for base-16) so raw matrix values "
                             "stay in one digit and --unit-weight controls significance.")
    parser.add_argument("--unit-weight", type=float, default=1.0,
                        help="Scale factor applied to the matrix after generation. "
                             "Use powers of the radix base (1, 16, 256, ...) to run "
                             "the same digit-structure matrix at different significance levels.")
    parser.add_argument("--float-weights", action="store_true",
                        help="Generate float weights instead of integer weights.")

    # Splitting
    parser.add_argument("--radix-bases", type=int, nargs='*', default=None,
                        help="Radix bases to test. For radix engines the default is "
                             "[2, 4, 8, 16, 32]. For --engine euler_bvn, Radix does NOT "
                             "run unless this flag is given explicitly (e.g. "
                             "--radix-bases 2 4 8 16 to compare Euler leaves vs digit planes).")
    parser.add_argument("--euler-depths", type=int, nargs='*', default=[1],
                        help="Euler splitting depths to test (depth=d gives 2^d leaf matrices). "
                             "depth=0 is plain BvN with no splitting. "
                             "Defaults to [1] (one split -> 2 leaves).")
    parser.add_argument("--euler-colors", type=int, nargs='*', default=None,
                        help="Number of colors (leaves) for Euler splitting - must be powers of 2. "
                             "Converted to depths internally (colors=2^depth). "
                             "Overrides --euler-depths when given. "
                             "E.g. --euler-colors 2 4 8 is equivalent to --euler-depths 1 2 3.")
    parser.add_argument("--euler-split-method", type=str, default="heuristic_ruined",
                        choices=["heuristic", "heuristic_ruined", "heuristic_ruined_last", "heuristic_scan", "euler", "euler_grouped", "greedy"],
                        help="Split strategy for the Euler framework. 'heuristic' "
                             "(same-direction split, sparser leaves) is the default; "
                             "'euler' is the classic alternating 2-colouring.")
    parser.add_argument("--euler-leaf-engine", type=str, default="heavy_static",
                        choices=["heavy", "heavy_static", "wfa", "wwfa", "maximum", "minimum"],
                        help="BvN matching engine for the Euler leaves. Also used for "
                             "the BvN baseline and Radix planes when --engine euler_bvn, "
                             "so all three frameworks are compared with the same engine "
                             "(e.g. 'maximum' reproduces the Euler-vs-bitplane comparison).")

    parser.add_argument("--free-between-runs", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Free every earlier result (baseline, previous Radix base or "
                             "Euler depth) before each timed call. On by default; "
                             "--no-free-between-runs reproduces campaigns run before "
                             "2026-11-02.")
    parser.add_argument("--random-seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Process workers when matrices run in parallel. main.py runs them "
                             "one at a time, and Radix planes, Euler splits and leaves are "
                             "always timed alone, so this has no effect on the timings.")
    
    # Plotting utilities
    parser.add_argument("--plot-from-csv", type=str, default=None,
                        help="Path to results CSV file. If set, generates plots for this CSV and exits.")
    
    parser.add_argument("--step-strategy", type=str, default="min", choices=["min", "max", "median", "all"],
                        help="Step size of each peel (min = classic BvN). Applies to the BvN "
                             "baseline, the Radix planes and the Euler leaves; 'all' runs Radix "
                             "with each of the three and the baseline and Euler with min.")

    return parser.parse_args()
