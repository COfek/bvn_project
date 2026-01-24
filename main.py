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


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    # Map 'max' to internal 'maximum'
    engine = args.engine
    if engine == "max":
        engine = "maximum"
        
    return ExperimentConfig(
        n=args.n,
        num_matrices=args.samples,
        density=args.density,
        engine=engine,
        radix_bases=args.radix_bases,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        output_dir=args.output,
        verbose=args.verbose,
        plot=not args.no_plot,
        
        # Split-tree
        split_sparsity_target=args.split_sparsity_target,
        split_max_depth=args.split_max_depth,
        split_p=args.split_p,
        split_cv_threshold=args.split_cv_threshold,
        split_min_matching_frac=args.split_min_matching_frac,
        split_method=args.split_method,
        is_parallel=False
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.plot_from_csv:
        # Special Mode: Just plot from existing CSV and exit
        csv_path = Path(args.plot_from_csv)
        if not csv_path.exists():
            print(f"Error: CSV not found at {csv_path}")
            return
            
        print(f"Regenerating plots from: {csv_path}")
        stats_list = _parse_stats_from_csv(csv_path)
        
        # Determine output dir (same folder as csv / plots)
        plots_dir = csv_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter first 5 samples
        plot_stats = stats_list[5:] if len(stats_list) > 5 else stats_list
        
        plot_results(
            plot_stats, 
            n=config.n, 
            bits=config.k, 
            out_dir=plots_dir, 
            matching_method=config.engine
        )
        print(f"Plots saved to: {plots_dir}")
        return

    # CHECK: Is this the main controller process or a spawned worker?
    is_main_process = multiprocessing.current_process().name == 'MainProcess'

    if is_main_process:
        # 1. SETUP: Create run folder inside outputs/runs/
        # Passing base_dir as config.output_dir/runs to match README structure if desired
        # Or just config.output_dir/runs
        # 1. SETUP: Create run folder inside output_dir directly
        base_runs = Path(config.output_dir)
        run_dir = create_run_folder(base_dir=str(base_runs))
        
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        global RUN_LOG_FILE, LOGGER
        RUN_LOG_FILE = get_log_file_path(run_dir)
        # Re-init logger to file
        LOGGER = init_logger() # logging_utils might need update to accept log_file, but assuming it picks up global or defaults

        print_banner("BvN-Arbiter Benchmark Started")
        LOGGER.info(f"Run directory: {run_dir} | Density: {config.density} | Engine: {config.engine}")
        save_config(config, run_dir)
    else:
        run_dir = None
        plots_dir = None

    # 2. EXECUTION
    with timed_section("Running Decomposition Experiment"):
        stats_list = run_experiment(config)

    # 3. TEARDOWN
    if is_main_process:
        csv_path = run_dir / "results_stats.csv"
        _write_stats_to_csv(stats_list, csv_path)
        LOGGER.info(f"Saved CSV: {csv_path}")

        if config.plot:
            with timed_section("Generating Plots"):
                # Exclude first 5 samples to remove JIT compilation spikes from plots
                plot_stats = stats_list[5:] if len(stats_list) > 5 else stats_list
                plot_results(plot_stats, n=config.n, bits=config.k, out_dir=plots_dir,
                             matching_method=config.engine)

        print_banner("Benchmark Complete")


def _write_stats_to_csv(stats_list, path: Path) -> None:
    if not stats_list: return

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Collect all unique keys from all stats
        all_keys = set()
        for s in stats_list:
            all_keys.update(s.radix_multi_results.keys())
        sorted_keys = sorted(list(all_keys))

        header = ["matrix_index", "bvn_perms", "bvn_cycle", "bvn_runtime"]
        for key in sorted_keys:
            # key is like "wfa_2" or "heavy_8"
            header += [f"{key}_time", f"{key}_cycle", f"{key}_perms"]

        writer.writerow(header)

        for s in stats_list:
            row = [s.matrix_index, s.num_permutations_bvn, s.cycle_length_bvn, s.runtime_bvn]
            for key in sorted_keys:
                if key in s.radix_multi_results:
                    row += list(s.radix_multi_results[key])
                else:
                    row += [None, None, None]
            writer.writerow(row)


def _parse_stats_from_csv(csv_path: Path) -> list:
    """Reconstructs stats objects from CSV for re-plotting."""
    # Local import to avoid circular dependency if placed at top in some designs,
    # but here stats is a utility.
    from src.utils.stats import DecompositionStats
    
    stats_list = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["matrix_index"])
            
            # BVN
            bvn_perms = row.get("bvn_perms")
            bvn_cycle = row.get("bvn_cycle")
            bvn_runtime = row.get("bvn_runtime")
            
            ds = DecompositionStats(
                matrix_index=idx,
                num_permutations_bvn=int(float(bvn_perms)) if bvn_perms and float(bvn_perms) > 0 else 0, # Note: stats usually expects int or None, but 0 is safe placeholder
                cycle_length_bvn=float(bvn_cycle) if bvn_cycle and bvn_cycle != "" and bvn_cycle != "None" else None,
                runtime_bvn=float(bvn_runtime) if bvn_runtime and bvn_runtime != "" and bvn_runtime != "None" else None
            )
            
            # Dynamic keys
            seen_engines = set()
            for col in row.keys():
                if col.endswith("_time"):
                    seen_engines.add(col[:-5])
            
            for key in seen_engines:
                t = row.get(f"{key}_time")
                c = row.get(f"{key}_cycle")
                p = row.get(f"{key}_perms")
                
                if t and c and p and t != "None":
                    ds.radix_multi_results[key] = (float(t), float(c), int(float(p)))
            
            stats_list.append(ds)
    return stats_list


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()