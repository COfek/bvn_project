"""
Analyze Euler splitting framework results and compare against:
  - Plain BvN baselines (from final_runs/heavy_static_bvn, heavy_bvn, wfa_bvn, maximum_bvn)
  - Radix B=2 results embedded in those same runs

Usage:
    python agent_scripts/analyze_euler_results.py <euler_csv> [--baseline-dir final_runs]

Outputs a printed table comparing:
  - max_leaf_runtime (simulated parallel execution time)
  - cycle length C(D)
  - number of permutations N
across: BvN baseline, Radix B=2, euler_heuristic_depth{1..4}
  (NOTE: "euler"/"greedy" split methods were dropped from the runner --
   the framework's actual goal is SPARSE leaves (fewer non-zeros per leaf
   matrix) so the downstream per-leaf BvN decompositions, run in parallel,
   are cheap; "heuristic" (same-direction split) is what achieves that, by
   keeping each M[i,j] pair's copies on one side whenever possible. A
   latent crash bug in its JIT circuit-reconstruction (phantom-edge
   out-of-bounds write) has been fixed -- it now colours edges directly
   during traversal -- and is verified stable/lossless. This script now
   reports only that variant.)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarise(df: pd.DataFrame, col_time: str, col_cycle: str, col_perms: str) -> dict:
    return {
        "time_mean": df[col_time].mean() * 1000,      # ms
        "time_median": df[col_time].median() * 1000,
        "cycle_mean": df[col_cycle].mean(),
        "perms_mean": df[col_perms].mean(),
    }


# ---------------------------------------------------------------------------
# Collect baseline runs
# ---------------------------------------------------------------------------

BASELINE_ENGINES = {
    "Hungarian-BvN":   ("maximum_bvn", "bvn_runtime", "bvn_cycle", "bvn_perms"),
    "GW Dynamic-BvN":  ("heavy_bvn",   "bvn_runtime", "bvn_cycle", "bvn_perms"),
    "GW Static-BvN":   ("heavy_static_bvn", "bvn_runtime", "bvn_cycle", "bvn_perms"),
    "WFA-BvN":         ("wfa_bvn",     "bvn_runtime", "bvn_cycle", "bvn_perms"),
}

RADIX_B2_ENGINES = {
    "Hungarian-B2":    ("maximum_bvn", "maximum_2"),
    "GW Dynamic-B2":   ("heavy_bvn",   "heavy_2"),
    "GW Static-B2":    ("heavy_static_bvn", "heavy_static_2"),
    "WFA-B2":          ("wfa_bvn",     "wfa_2"),
}


def find_latest_run(base_dir: Path, engine_name: str) -> Path | None:
    """Find most-recently created results_stats.csv for a given engine name."""
    candidates = []
    for run_dir in sorted(base_dir.iterdir()):
        cfg_file = run_dir / "config.json"
        csv_file = run_dir / "results_stats.csv"
        if not csv_file.exists():
            continue
        if cfg_file.exists():
            import json
            cfg = json.loads(cfg_file.read_text())
            if cfg.get("engine") == engine_name:
                candidates.append((run_dir.name, csv_file))
        else:
            # Fallback: check log
            log = run_dir / "log.txt"
            if log.exists() and engine_name in log.read_text():
                candidates.append((run_dir.name, csv_file))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)  # latest first by timestamp
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("euler_csv", help="results_stats.csv from the euler_bvn run")
    ap.add_argument("--baseline-dir", default="final_runs",
                    help="Directory containing the baseline engine run folders")
    args = ap.parse_args()

    euler_csv = Path(args.euler_csv)
    if not euler_csv.exists():
        sys.exit(f"ERROR: {euler_csv} not found")

    base_dir = Path(args.baseline_dir)

    euler_df = load_csv(euler_csv)
    # Drop warmup rows (first 5)
    euler_df = euler_df.iloc[5:].reset_index(drop=True)

    print(f"\nEuler run: {euler_csv}  ({len(euler_df)} matrices after warmup skip)\n")

    rows = []

    # -----------------------------------------------------------------------
    # BvN baselines and Radix B=2
    # -----------------------------------------------------------------------
    for label, (engine_name, t_col, c_col, p_col) in BASELINE_ENGINES.items():
        csv = find_latest_run(base_dir, engine_name)
        if csv is None:
            print(f"  [warn] no run found for {engine_name}")
            continue
        df = load_csv(csv).iloc[5:].reset_index(drop=True)
        s = summarise(df, t_col, c_col, p_col)
        rows.append({
            "Method": label,
            "T_compute mean (ms)": f"{s['time_mean']:.1f}",
            "C(D) mean": f"{s['cycle_mean']:.0f}",
            "N mean": f"{s['perms_mean']:.0f}",
        })

    for label, (engine_name, radix_key) in RADIX_B2_ENGINES.items():
        csv = find_latest_run(base_dir, engine_name)
        if csv is None:
            continue
        df = load_csv(csv).iloc[5:].reset_index(drop=True)
        t_col = f"{radix_key}_time"
        c_col = f"{radix_key}_cycle"
        p_col = f"{radix_key}_perms"
        if t_col not in df.columns:
            print(f"  [warn] column {t_col} not in {csv}")
            continue
        s = summarise(df, t_col, c_col, p_col)
        rows.append({
            "Method": label,
            "T_compute mean (ms)": f"{s['time_mean']:.1f}",
            "C(D) mean": f"{s['cycle_mean']:.0f}",
            "N mean": f"{s['perms_mean']:.0f}",
        })

    # -----------------------------------------------------------------------
    # Euler framework results
    # -----------------------------------------------------------------------
    for depth in [1, 2, 3, 4]:
        for split in ["heuristic"]:
            key = f"euler_{split}_depth{depth}"
            t_col = f"{key}_time"
            c_col = f"{key}_cycle"
            p_col = f"{key}_perms"
            if t_col not in euler_df.columns:
                continue
            s = summarise(euler_df, t_col, c_col, p_col)
            rows.append({
                "Method": f"Euler-{split} depth={depth}",
                "T_compute mean (ms)": f"{s['time_mean']:.1f}",
                "C(D) mean": f"{s['cycle_mean']:.0f}",
                "N mean": f"{s['perms_mean']:.0f}",
            })

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    result_df = pd.DataFrame(rows)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)
    pd.set_option("display.colheader_justify", "left")
    print(result_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Sparsity analysis: greedy vs euler
    # -----------------------------------------------------------------------
    print("\n--- Euler-heuristic permutation count by depth ---")
    for depth in [1, 2, 3, 4]:
        gk = f"euler_heuristic_depth{depth}_perms"
        if gk not in euler_df.columns:
            continue
        g_mean = euler_df[gk].mean()
        print(f"  depth={depth}: Euler-heuristic N={g_mean:.1f}")

    # -----------------------------------------------------------------------
    # DCT-style comparison vs existing Radix B=2
    # -----------------------------------------------------------------------
    print("\n--- T_compute speedup of Euler framework vs GW Dynamic BvN baseline ---")
    gw_csv = find_latest_run(base_dir, "heavy_bvn")
    if gw_csv:
        gw_df = load_csv(gw_csv).iloc[5:].reset_index(drop=True)
        bvn_time = gw_df["bvn_runtime"].mean() * 1000
        radix_time = gw_df["heavy_2_time"].mean() * 1000 if "heavy_2_time" in gw_df.columns else None
        print(f"  GW Dynamic BvN:   {bvn_time:.1f} ms")
        if radix_time:
            print(f"  GW Dynamic B=2:   {radix_time:.1f} ms  ({bvn_time/radix_time:.1f}× speedup vs BvN)")
        for depth in [1, 2, 3, 4]:
            for split in ["heuristic"]:
                key = f"euler_{split}_depth{depth}_time"
                if key in euler_df.columns:
                    t = euler_df[key].mean() * 1000
                    print(f"  Euler-{split} d={depth}: {t:.1f} ms  ({bvn_time/t:.1f}× vs BvN)")


if __name__ == "__main__":
    main()
