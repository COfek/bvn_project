from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.config import ExperimentConfig
from src.runner import run_experiment
from src.plotting import plot_results

# NEW imports for run folders + logging
from src.utils.run_utils import create_run_folder, save_config, get_log_file_path
from src.utils.logging_utils import init_logger, RUN_LOG_FILE, print_banner, timed_section, LOGGER


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the BVN experiment.
    """
    parser = argparse.ArgumentParser(
        description="Run BVN and bit-plane decompositions on random doubly "
                    "stochastic matrices and plot statistics."
    )

    parser.add_argument("--n", type=int, default=32,
                        help="Matrix dimension n (default: 6).")

    parser.add_argument("--num-matrices", type=int, default=1000,
                        help="Number of matrices to generate (default: 1000).")

    parser.add_argument("--density", type=float, default=1.0,
                        help="Sparsity density in (0, 1] (default: 1.0).")

    parser.add_argument("--sinkhorn-iters", type=int, default=200,
                        help="Number of Sinkhorn iterations (default: 200).")

    parser.add_argument("--bitplane-bits", type=int, default=8,
                        help="Number of bits for bit-plane scaling (default: 8).")

    parser.add_argument("--bitplane-method", type=str, default="both",
                        choices=["maximum", "maximal", "both"],
                        help="Bitplane matching method: maximum, maximal (WFA), or both.")

    parser.add_argument("--random-seed", type=int, default=42,
                        help="Base random seed (default: 42).")

    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of worker threads (default: None = auto).")

    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional path to save results CSV (default: None).")

    parser.add_argument("--split-sparsity-target", type=int, default=3,
                        help="Stop splitting when nnz(X) <= this number (default: 3).")

    parser.add_argument("--split-max-depth", type=int, default=8,
                        help="Maximum recursion depth for split-tree (default: 8).")

    parser.add_argument("--split-p", type=float, default=0.5,
                        help="Probability p for random binary split (default: 0.5).")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Build an ExperimentConfig from parsed CLI arguments.
    """
    return ExperimentConfig(
        n=args.n,
        num_matrices=args.num_matrices,
        density=args.density,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_eps=1e-12,
        bvn_tol=1e-10,
        bitplane_bits=args.bitplane_bits,
        bitplane_tol=1e-9,
        bitplane_method=args.bitplane_method,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        output_csv=args.output_csv,
        split_sparsity_target=args.split_sparsity_target,
        split_max_depth=args.split_max_depth,
        split_p=args.split_p,
    )


def main() -> None:
    """
    Main entry point: parse arguments, run experiment, optionally save CSV, plot.
    """

    # ------------------------------
    # 1. Parse args & config
    # ------------------------------
    args = parse_args()
    config = build_config(args)

    # ------------------------------
    # 2. Prepare run folder
    # ------------------------------
    run_dir = create_run_folder()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 3. Initialize logging (run folder)
    # ------------------------------
    global RUN_LOG_FILE, LOGGER
    RUN_LOG_FILE = get_log_file_path(run_dir)
    LOGGER = init_logger()   # must be called AFTER RUN_LOG_FILE is set

    print_banner("BVN & Bitplane Experiment Started")
    LOGGER.info(f"Run directory: {run_dir}")

    # ------------------------------
    # 4. Save config.json
    # ------------------------------
    save_config(config, run_dir)

    # ------------------------------
    # 5. Run experiment
    # ------------------------------
    with timed_section("Running Decomposition Experiment"):
        stats_list = run_experiment(config)

    # ------------------------------
    # 6. Save CSV (optional, into run folder)
    # ------------------------------
    if config.output_csv is not None:
        csv_path = run_dir / config.output_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_stats_to_csv(stats_list, csv_path)
        LOGGER.info(f"Saved CSV: {csv_path}")

    # ------------------------------
    # 7. Plot results to run/plots/
    # ------------------------------
    with timed_section("Generating Plots"):
        plot_results(stats_list, n=config.n, bits=config.bitplane_bits, out_dir=plots_dir)

    print_banner("Experiment Complete")


def _write_stats_to_csv(stats_list, path: Path) -> None:
    """
    Write decomposition statistics to a CSV file.

    Supports unified DecompositionStats with:
        BVN stats
        Maximum bitplane stats
        Maximal (WFA) bitplane stats
    """
    import csv

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "matrix_index",
            "num_permutations_bvn",
            "cycle_length_bvn",
            "num_perm_maximum",
            "cycle_maximum",
            "num_perm_maximal",
            "cycle_maximal",
        ])

        for s in stats_list:
            writer.writerow([
                s.matrix_index,
                s.num_permutations_bvn,
                s.cycle_length_bvn,
                s.num_perm_maximum,
                s.cycle_maximum,
                s.num_perm_maximal,
                s.cycle_maximal,
            ])


if __name__ == "__main__":
    main()
