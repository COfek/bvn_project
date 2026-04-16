from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

from src.cli import parse_args
from src.config import ExperimentConfig
from src.plotting import plot_results
from src.runner import run_experiment
from src.utils.io_utils import parse_stats_from_csv, write_stats_to_csv
from src.utils.logging_utils import configure_logging, print_banner, timed_section
from src.utils.run_utils import create_run_folder, get_log_file_path, save_config

logger = logging.getLogger(__name__)

def build_config(args) -> ExperimentConfig:
    # Map 'max' to internal 'maximum'
    engine = args.engine
    if engine == "max":
        engine = "maximum"
        
    return ExperimentConfig(
        engine=engine,
        radix_bases=args.radix_bases,
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
        float_weights=args.float_weights
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