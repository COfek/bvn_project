from __future__ import annotations

import argparse
import csv
import multiprocessing
from pathlib import Path

from src.config import ExperimentConfig
from src.plotting import plot_results
from src.runner import run_experiment
from src.utils.logging_utils import init_logger, print_banner, timed_section
from src.utils.run_utils import create_run_folder, save_config, get_log_file_path

# Global placeholders
RUN_LOG_FILE = None
LOGGER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BVN, Radix (incl. Bit-plane), and Split-tree decompositions."
    )
    parser.add_argument("--n", type=int, default=32, help="Matrix dimension.")
    parser.add_argument("--k", type=int, default=7,
                        help="The maximum number of bits which will be used to construct the matrix.")
    parser.add_argument("--num-matrices", type=int, default=1000, help="Number of matrices.")
    parser.add_argument("--random-seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-workers", type=int, default=None, help="Workers for Radix planes.")
    parser.add_argument("--output-csv", type=str, default="results.csv", help="CSV filename.")
    parser.add_argument("--split-sparsity-target", type=int, default=3)
    parser.add_argument("--split-max-depth", type=int, default=1)
    parser.add_argument("--split-p", type=float, default=0.5)
    parser.add_argument("--split-cv-threshold", type=float, default=0.15)
    parser.add_argument("--split-min-matching-frac", type=float, default=0.8)
    parser.add_argument("--split-method", type=str, default="pivot", choices=["pivot", "random"])
    parser.add_argument("--matching_method", type=str, default="maximum", choices=["heavy", "wfa", "maximum"])
    parser.add_argument("--radix-bases", type=int, nargs='+', default=[2, 3, 8, 12, 16])

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        n=args.n,
        k=args.k,
        num_matrices=args.num_matrices,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        output_csv=args.output_csv,
        split_sparsity_target=args.split_sparsity_target,
        split_max_depth=args.split_max_depth,
        split_p=args.split_p,
        split_cv_threshold=args.split_cv_threshold,
        split_min_matching_frac=args.split_min_matching_frac,
        split_method=args.split_method,
        radix_bases=args.radix_bases,
        matching_method=args.matching_method,
        is_parallel=False
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    # CHECK: Is this the main controller process or a spawned worker?
    is_main_process = multiprocessing.current_process().name == 'MainProcess'

    if is_main_process:
        # 1. SETUP: Only the main process creates folders and starts logging
        run_dir = create_run_folder()
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        global RUN_LOG_FILE, LOGGER
        RUN_LOG_FILE = get_log_file_path(run_dir)
        LOGGER = init_logger()

        print_banner("BVN, Radix, Split-Tree Experiment Started")
        LOGGER.info(f"Run directory: {run_dir} | K-Regular Sum: {config.k}")
        save_config(config, run_dir)
    else:
        # Workers don't initialize these variables
        run_dir = None
        plots_dir = None

    # 2. EXECUTION: This runs in the main process, delegating tasks to workers
    with timed_section("Running Decomposition Experiment"):
        stats_list = run_experiment(config)

    # 3. TEARDOWN: Only the main process writes final data and plots
    if is_main_process:
        if config.output_csv:
            csv_path = run_dir / config.output_csv
            _write_stats_to_csv(stats_list, csv_path)
            LOGGER.info(f"Saved CSV: {csv_path}")

        with timed_section("Generating Plots"):
            plot_results(stats_list, n=config.n, bits=config.k, out_dir=plots_dir)

        print_banner("Experiment Complete")


def _write_stats_to_csv(stats_list, path: Path) -> None:
    if not stats_list: return

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        all_bases = set()
        for s in stats_list:
            all_bases.update(s.radix_multi_results.keys())
        sorted_bases = sorted(list(all_bases))

        header = ["matrix_index", "bvn_perms", "bvn_cycle", "bvn_runtime"]
        for b in sorted_bases:
            label = "bitplane" if b == 2 else f"radix_{b}"
            header += [f"{label}_time", f"{label}_cycle", f"{label}_perms"]

        writer.writerow(header)

        for s in stats_list:
            row = [s.matrix_index, s.num_permutations_bvn, s.cycle_length_bvn, s.runtime_bvn]
            for b in sorted_bases:
                if b in s.radix_multi_results:
                    row += list(s.radix_multi_results[b])
                else:
                    row += [None, None, None]
            writer.writerow(row)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()