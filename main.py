from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

from src.cli import parse_args
from src.config import ExperimentConfig
from src.plotting import plot_results
from src.runner import run_experiment
from src.utils.analysis import write_euler_comparison_summary
from src.utils.io_utils import parse_stats_from_csv, write_stats_to_csv
from src.utils.logging_utils import configure_logging, print_banner, timed_section
from src.utils.run_utils import create_run_folder, get_log_file_path, save_config

logger = logging.getLogger(__name__)

def build_config(args) -> ExperimentConfig:
    # Map 'max' to internal 'maximum'
    engine = args.engine
    if engine == "max":
        engine = "maximum"

    # --radix-bases not given: radix engines keep their traditional default
    # sweep; euler_bvn runs NO radix unless bases are requested explicitly.
    radix_bases = args.radix_bases
    if radix_bases is None:
        radix_bases = [] if engine == "euler_bvn" else [2, 4, 8, 16, 32]

    # --euler-colors overrides --euler-depths when given.
    # Each color count must be a power of 2; depth = log2(colors).
    euler_depths = args.euler_depths
    if args.euler_colors is not None:
        bad = [c for c in args.euler_colors if c < 1 or (c & (c - 1)) != 0]
        if bad:
            raise ValueError(
                f"--euler-colors values must be powers of 2, got: {bad}"
            )
        euler_depths = [c.bit_length() - 1 for c in args.euler_colors]

    return ExperimentConfig(
        engine=engine,
        radix_bases=radix_bases,
        euler_depths=euler_depths,
        euler_split_method=args.euler_split_method,
        euler_leaf_engine=args.euler_leaf_engine,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        output_dir=args.output,
        plot=not args.no_plot,

        # Matrix generation
        n=args.n,
        num_matrices=args.samples,
        k=args.k,
        is_parallel=False,
        max_weight=args.max_weight,
        unit_weight=args.unit_weight,
        float_weights=args.float_weights,
        radix_strategy=args.step_strategy
    )

def main() -> None:
    args = parse_args()
    config = build_config(args)

    # CLI Mode: Plot from CSV
    if args.plot_from_csv:
        csv_path = Path(args.plot_from_csv)
        if not csv_path.exists():
            print(f"Error: CSV not found at {csv_path}")
            return
            
        configure_logging(level=logging.INFO) # Console only basically
        print(f"Regenerating plots from: {csv_path}")
        stats_list = parse_stats_from_csv(csv_path)
        
        # Determine output dir (same folder as csv / plots)
        plots_dir = csv_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter first 5 samples if we have enough
        plot_stats = stats_list[5:] if len(stats_list) > 10 else stats_list
        
        plot_results(
            plot_stats,
            n=config.n,
            bits=config.k,
            out_dir=plots_dir,
            matching_method=config.engine,
            generator="unified"
        )
        print(f"Plots saved to: {plots_dir}")

        summary = write_euler_comparison_summary(
            stats_list, csv_path.parent / "euler_comparison_summary.txt")
        if summary:
            print(summary)
        return

    # Standard Execution Mode
    is_main_process = multiprocessing.current_process().name == 'MainProcess'
    run_dir = None
    plots_dir = None

    if is_main_process:
        # 1. Setup Run Directory
        base_runs = Path(config.output_dir)
        run_dir = create_run_folder(base_dir=str(base_runs))
        
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 2. Configure Logging
        log_file = get_log_file_path(run_dir)
        configure_logging(log_file=log_file, level=logging.INFO)

        print_banner("BvN-Arbiter Benchmark Started")
        logger.info(f"Run directory: {run_dir} |  Engine: {config.engine} | Bases: {config.radix_bases}")
        save_config(config, run_dir)
    else:
        # Worker processes just inherit basic logging or configure silently if needed
        # (RichHandler in logging_utils already safe for workers)
        pass

    # 3. Execution
    try:
        with timed_section("Running Decomposition Experiment"):
            stats_list = run_experiment(config)
    except KeyboardInterrupt:
        if is_main_process:
            logger.warning("Interrupted by user.")
        return

    # 4. Teardown & Results
    if is_main_process:
        csv_path = run_dir / "results_stats.csv"
        write_stats_to_csv(stats_list, csv_path)
        logger.info(f"Saved CSV: {csv_path}")

        summary = write_euler_comparison_summary(
            stats_list, run_dir / "euler_comparison_summary.txt")
        if summary:
            print(summary)
            logger.info(f"Saved comparison summary: {run_dir / 'euler_comparison_summary.txt'}")

        if config.plot:
            with timed_section("Generating Plots"):
                # Exclude first 5 samples to remove JIT compilation spikes from plots
                plot_stats = stats_list[5:] if len(stats_list) > 10 else stats_list
                plot_results(plot_stats, n=config.n, bits=config.k, out_dir=plots_dir,
                            matching_method=config.engine, generator="unified")

        print_banner("Benchmark Complete!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()