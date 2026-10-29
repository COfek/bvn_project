"""
Task 1 — Matching-extraction time vs. split depth.

Question: does decomposing sparser/lighter leaf matrices reduce the time it
takes to extract the matchings, compared to running BvN on the original
matrix? Splitting cost is measured but reported SEPARATELY — the comparison
of interest is:

    baseline BvN time on the original matrix   (depth 0, no split)
        vs.
    MAX leaf BvN time after Euler splitting    (depth 1, 2, 3 → 2, 4, 8 leaves)

If max-leaf time at each extra depth is ~half the previous level, splitting
is "free parallelism" for the matching phase.

Split method: "heuristic" (same-direction split) only — plain euler is not
interesting here because it splits every entry ~50/50 in place, so leaves
have the same nnz as the parent (no sparsity gain).
Leaf matching engine: heavy_static (same as the runner uses).

Usage:
    python -m agent_scripts.task1_leaf_runtime            # dense config
    python -m agent_scripts.task1_leaf_runtime --sparse   # sparse config
    python -m agent_scripts.task1_leaf_runtime -m 20      # more samples
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.algorithms.bvn import bvn_decomposition
from src.algorithms.euler_splitting import decompose_euler_framework
from src.utils.matrix_generator import generate_matrix

LEAF_ENGINE = "heavy_static"
DEPTHS = [1, 2, 3]
SPLIT_METHODS = ["heuristic"]
WARMUP_SAMPLES = 2   # discarded — JIT compilation warm-up


def run(n: int, k: int, max_weight: int, num_matrices: int, seed: int, out_csv: Path):
    rows = []
    total = num_matrices + WARMUP_SAMPLES

    for idx in range(total):
        rng = np.random.default_rng(seed + idx)
        M = generate_matrix(n=n, k=k, max_weight=max_weight, rng=rng)
        S = int(M.sum(axis=1).max())
        is_warmup = idx < WARMUP_SAMPLES

        # ---- baseline: plain BvN on the original matrix (no split) ----
        t0 = time.perf_counter()
        bvn_decomposition(M.copy(), matching_algorithm=LEAF_ENGINE)
        t_baseline = time.perf_counter() - t0

        if not is_warmup:
            rows.append({
                "matrix": idx - WARMUP_SAMPLES, "S": S,
                "split_method": "none", "depth": 0, "n_leaves": 1,
                "split_time_ms": 0.0,
                "max_leaf_ms": t_baseline * 1e3,
                "ratio_vs_baseline": 1.0,
            })

        for split_method in SPLIT_METHODS:
            for depth in DEPTHS:
                _, split_t, max_leaf_t, n_leaves = decompose_euler_framework(
                    M, matching_method=LEAF_ENGINE,
                    depth=depth, split_method=split_method,
                )
                if is_warmup:
                    continue
                rows.append({
                    "matrix": idx - WARMUP_SAMPLES, "S": S,
                    "split_method": split_method, "depth": depth,
                    "n_leaves": n_leaves,
                    "split_time_ms": split_t * 1e3,
                    "max_leaf_ms": max_leaf_t * 1e3,
                    "ratio_vs_baseline": max_leaf_t / t_baseline if t_baseline > 0 else float("nan"),
                })

        tag = "warmup" if is_warmup else f"{idx - WARMUP_SAMPLES + 1}/{num_matrices}"
        print(f"  [{tag}] S={S} baseline={t_baseline*1e3:.1f}ms")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {out_csv}")

    # ---- aggregate summary table ----
    print(f"\n{'method':<10} {'depth':>5} {'leaves':>6} {'split ms':>9} "
          f"{'max-leaf ms':>11} {'vs baseline':>11} {'vs prev depth':>13}")
    print("-" * 72)

    def _avg(rs, key):
        vals = [r[key] for r in rs]
        return sum(vals) / len(vals)

    base_rows = [r for r in rows if r["depth"] == 0]
    base_ms = _avg(base_rows, "max_leaf_ms")
    print(f"{'baseline':<10} {0:>5} {1:>6} {0.0:>9.2f} {base_ms:>11.2f} {1.0:>11.3f} {'-':>13}")

    for split_method in SPLIT_METHODS:
        prev_ms = base_ms
        for depth in DEPTHS:
            rs = [r for r in rows
                  if r["split_method"] == split_method and r["depth"] == depth]
            if not rs:
                continue
            ml = _avg(rs, "max_leaf_ms")
            sp = _avg(rs, "split_time_ms")
            print(f"{split_method:<10} {depth:>5} {int(_avg(rs, 'n_leaves')):>6} "
                  f"{sp:>9.2f} {ml:>11.2f} {ml / base_ms:>11.3f} {ml / prev_ms:>13.3f}")
            prev_ms = ml


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", action="store_true",
                    help="sparse-traffic config (k=16) instead of dense (k=256)")
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("-m", "--num-matrices", type=int, default=10)
    ap.add_argument("--max-weight", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    k = 16 if args.sparse else 256
    scenario = "sparse" if args.sparse else "dense"
    out = Path("euler_exp") / f"task1_leaf_runtime_{scenario}.csv"

    print(f"Task 1: n={args.n}, k={k}, max_weight={args.max_weight}, "
          f"{args.num_matrices} matrices ({scenario}), leaf engine={LEAF_ENGINE}")
    run(args.n, k, args.max_weight, args.num_matrices, args.seed, out)
