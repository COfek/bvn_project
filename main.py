from __future__ import annotations

import argparse
from pathlib import Path

# Updated imports: plot_comprehensive_2d_analysis replaces the 3D surface
from src.config import ExperimentConfig
from src.plotting import plot_results, plot_comprehensive_2d_analysis
from src.runner import run_experiment, run_comprehensive_study
from src.utils.logging_utils import init_logger, print_banner, timed_section
from src.utils.run_utils import create_run_folder, save_config, get_log_file_path

global RUN_LOG_FILE, LOGGER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Network Switch Decomposition: Scalability and Efficiency Study."
    )

    # --- Mode Toggle ---
    parser.add_argument("--single", action="store_true",
                        help="Run only a single configuration instead of the comprehensive sweep.")

    # --- Matrix Params ---
    parser.add_argument("--n", type=int, default=64, help="Matrix dimension for single-run mode.")
    parser.add_argument("--num-matrices", type=int, default=1000, help="Matrices per configuration.")
    parser.add_argument("--density", type=float, default=0.9, help="Density in (0, 1].")

    # --- Algorithm Params ---
    parser.add_argument("--bitplane-bits", type=int, default=16, help="Precision bits for Radix/Bitplane.")
    parser.add_argument("--bitplane-method", type=str, default="maximal", choices=["maximum", "maximal", "both"])
    parser.add_argument("--split-max-depth", type=int, default=1, help="Depth 1 for comparative fairness.")

    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        n=args.n,
        num_matrices=args.num_matrices,
        density=args.density,
        sinkhorn_iters=1000,
        sinkhorn_eps=1e-12,
        bvn_tol=1e-10,
        bitplane_bits=args.bitplane_bits,
        bitplane_tol=1e-9,
        bitplane_method=args.bitplane_method,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        split_max_depth=args.split_max_depth,
        split_sparsity_target=3,
        split_p=0.5,
        split_cv_threshold=0.15,
        split_min_matching_frac=0.8,
        split_method="random"
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    # Prepare Run Environment
    run_dir = create_run_folder()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global RUN_LOG_FILE, LOGGER
    RUN_LOG_FILE = get_log_file_path(run_dir)
    LOGGER = init_logger()

    # --- DEFAULT: Comprehensive Mode ---
    if not args.single:
        print_banner("COMPREHENSIVE NETWORK EFFICIENCY STUDY")
        LOGGER.info(f"Targeting 12,000 matrices (N=[32, 64, 128, 256], D=[0.3, 0.6, 0.9])")

        with timed_section("Batch Decomposition Execution"):
            # Generates comprehensive_results.csv
            csv_path = run_comprehensive_study(config, run_dir)

        with timed_section("3-Panel Performance Analysis"):
            # Generates the Reconfigurations, Completion Delay, and Scalability plots
            plot_comprehensive_2d_analysis(csv_path, plots_dir)

    # --- OVERRIDE: Single Configuration Mode ---
    else:
        print_banner(f"SINGLE RUN: N={config.n}, D={config.density}")
        save_config(config, run_dir)

        with timed_section("Execution"):
            stats_list = run_experiment(config)

        with timed_section("Generating Standard Distribution Plots"):
            plot_results(stats_list, n=config.n, bits=config.bitplane_bits, out_dir=plots_dir)

    print_banner("Experiment Successful - Data Saved")


if __name__ == "__main__":
    main()