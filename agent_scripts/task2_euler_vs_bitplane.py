"""
Task 2 — Parallel runtime: Euler leaves vs. bit-plane (Radix B=2) planes.

Both frameworks produce independent sub-matrices that can be decomposed in
parallel; simulated parallel runtime = MAX time over any single unit
(leaf or digit plane). Both use Hungarian ("maximum") matching so the
comparison isolates the *structure* of the sub-matrices:

    Euler leaves  — doubly stochastic, (S/2^d)-regular, dense-ish
    Bit planes    — 0/1 entries, sparse, NOT doubly stochastic

Reported per framework: number of parallel units, max unit time, and for
Euler also the (sequential) split time so the full cost picture is visible.

Usage:
    python -m agent_scripts.task2_euler_vs_bitplane            # dense config
    python -m agent_scripts.task2_euler_vs_bitplane --sparse   # sparse config
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.algorithms.euler_splitting import decompose_euler_framework
from src.algorithms.radix_decomposition import decompose_radix
from src.utils.matrix_generator import generate_matrix

MATCHING = "maximum"          # Hungarian, per professor's instruction
EULER_SPLIT = "heuristic"     # the split method the framework actually targets
DEPTHS = [1, 2, 3, 4]         # 2, 4, 8, 16 leaves
RADIX_BASES = [16, 8, 4, 2]   # ~2-3, ~3, ~5, ~9 planes — so unit counts can be
                              # matched against the Euler depths above
WARMUP_SAMPLES = 2


def run(n: int, k: int, max_weight: int, num_matrices: int, seed: int, out_csv: Path):
    rows = []
    total = num_matrices + WARMUP_SAMPLES

    for idx in range(total):
        rng = np.random.default_rng(seed + idx)
        M = generate_matrix(n=n, k=k, max_weight=max_weight, rng=rng)
        S = int(M.sum(axis=1).max())
        is_warmup = idx < WARMUP_SAMPLES

        # ---- Radix digit planes at several bases, Hungarian on each plane ----
        for base in RADIX_BASES:
            t0 = time.perf_counter()
            _, max_plane_rt, n_planes = decompose_radix(
                matrix=M, base=base, matching_method=MATCHING, step_strategy="min",
            )
            t_radix_total = time.perf_counter() - t0

            if is_warmup:
                continue
            rows.append({
                "matrix": idx - WARMUP_SAMPLES, "S": S,
                "framework": f"radix_b{base}", "depth": "",
                "n_units": n_planes,
                "split_time_ms": 0.0,
                "max_unit_ms": max_plane_rt * 1e3,
                "total_seq_ms": t_radix_total * 1e3,
            })

        # ---- Euler framework at several depths, Hungarian on each leaf ----
        for depth in DEPTHS:
            t0 = time.perf_counter()
            _, split_t, max_leaf_t, n_leaves = decompose_euler_framework(
                M, matching_method=MATCHING,
                depth=depth, split_method=EULER_SPLIT,
            )
            t_euler_total = time.perf_counter() - t0
            if is_warmup:
                continue
            rows.append({
                "matrix": idx - WARMUP_SAMPLES, "S": S,
                "framework": f"euler_{EULER_SPLIT}", "depth": depth,
                "n_units": n_leaves,
                "split_time_ms": split_t * 1e3,
                "max_unit_ms": max_leaf_t * 1e3,
                "total_seq_ms": t_euler_total * 1e3,
            })

        tag = "warmup" if is_warmup else f"{idx - WARMUP_SAMPLES + 1}/{num_matrices}"
        print(f"  [{tag}] S={S}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {out_csv}")

    # ---- aggregate summary ----
    print(f"\n{'framework':<16} {'depth':>5} {'units':>6} {'split ms':>9} "
          f"{'max-unit ms':>11} {'total seq ms':>12}")
    print("-" * 66)

    def _avg(rs, key):
        vals = [r[key] for r in rs]
        return sum(vals) / len(vals)

    for base in RADIX_BASES:
        radix_rows = [r for r in rows if r["framework"] == f"radix_b{base}"]
        if not radix_rows:
            continue
        print(f"{'radix_b' + str(base):<16} {'-':>5} {_avg(radix_rows, 'n_units'):>6.1f} "
              f"{0.0:>9.2f} {_avg(radix_rows, 'max_unit_ms'):>11.2f} "
              f"{_avg(radix_rows, 'total_seq_ms'):>12.2f}")

    for depth in DEPTHS:
        rs = [r for r in rows
              if r["framework"].startswith("euler") and r["depth"] == depth]
        if not rs:
            continue
        print(f"{'euler_' + EULER_SPLIT:<16} {depth:>5} {_avg(rs, 'n_units'):>6.1f} "
              f"{_avg(rs, 'split_time_ms'):>9.2f} {_avg(rs, 'max_unit_ms'):>11.2f} "
              f"{_avg(rs, 'total_seq_ms'):>12.2f}")


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
    out = Path("euler_exp") / f"task2_euler_vs_bitplane_{scenario}.csv"

    print(f"Task 2: n={args.n}, k={k}, max_weight={args.max_weight}, "
          f"{args.num_matrices} matrices ({scenario}), matching={MATCHING}")
    run(args.n, k, args.max_weight, args.num_matrices, args.seed, out)
