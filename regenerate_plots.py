from pathlib import Path
import csv
import ast
from src.plotting import plot_results
from src.utils.stats import DecompositionStats

def regenerate(run_folder: str):
    folder = Path(run_folder)
    csv_path = folder / "results_stats.csv"
    plots_dir = folder / "plots"
    
    print(f"Loading data from {csv_path}...")
    
    stats_list = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Reconstruct DecompositionStats object
            radix_results = {}
            for key_col, val in row.items():
                if key_col.endswith("_time") and val:
                    key = key_col.replace("_time", "")
                    # keys: time, cycle, perms
                    # Ensure values exist and handle empty strings
                    try:
                        time_val = float(val)
                        cycle_val = float(row.get(f"{key}_cycle", 0))
                        perms_val = int(float(row.get(f"{key}_perms", 0))) # Handle 5.0 -> 5
                        radix_results[key] = (time_val, cycle_val, perms_val)
                    except ValueError:
                        continue
            
            try:
                # Helper to safe convert floats
                def safe_float(v): return float(v) if v else None
                def safe_int(v): return int(float(v)) if v else None

                stats = DecompositionStats(
                    matrix_index=int(row['matrix_index']),
                    num_permutations_bvn=safe_int(row.get('bvn_perms')),
                    cycle_length_bvn=safe_float(row.get('bvn_cycle')),
                    runtime_bvn=safe_float(row.get('bvn_runtime')),
                    num_perm_split=safe_int(row.get('split_perms')), 
                    cycle_split=safe_float(row.get('split_cycle')),
                    runtime_split=safe_float(row.get('split_runtime')),
                    radix_multi_results=radix_results
                )
                stats_list.append(stats)
            except Exception as e:
                print(f"Skipping row {row.get('matrix_index')}: {e}")
                
    print(f"Loaded {len(stats_list)} stats. Regenerating plots in {plots_dir}...")
    # N and Bits are for labels, defaults to N=64 are fine
    plot_results(stats_list, n=64, bits=100, out_dir=plots_dir, matching_method="all")
    print("Done!")

if __name__ == "__main__":
    regenerate("run/20260124_164225")
