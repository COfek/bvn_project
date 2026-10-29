"""Every engine x framework on the MEASURED MoE traces, same grid as the TM campaign.

Workload: expert-parallel all-to-all matrices measured from three MoE models on two
datasets (Birkhoff-MoE, HotOptics'26; traces in paper_traces/).  The matrices are
used exactly as recorded -- no regularisation, no scaling, no rounding.

This mirrors agent_scripts/tm_workload_experiment.py configuration for configuration,
so the two workloads can be read side by side:
    for each of {Hungarian, WFA, GW Dynamic, GW Static}:
        BvN baseline | Radix B16, B8, B4, B2 | Euler d1..d4 (split-inclusive runtime)

It supersedes two earlier scripts:
  * moe_workload_experiment.py -- a synthetic MoE-*style* stand-in written before the
    real traces were available;
  * moe_raw_campaign.py -- the same traces but a narrower grid (no Euler).

Two differences from the TM campaign are forced by the data and are the point of
the section.

1. S is a bound, not an invariant.  A TM matrix is S-regular, so S is simply the
   common row sum.  These traces are not regular: row sums are per-rank token
   emissions, column sums are expert popularities.  We therefore report

       S = max_i max( sum_j D_ij , sum_j D_ji ),

   the cycle-length lower bound -- the busiest port must transmit that many units,
   so no schedule finishes sooner.  C/S >= 1 still holds and C/S = 1 is still
   optimal, so the column means the same thing as in the TM tables.

2. Fixed workload, repeated timing.  The TM campaign draws 10 random matrices per
   regime; here the 12 traces ARE the workload and cannot be resampled.  Since the
   matrices are small (n=8) and run in well under a millisecond, each configuration
   is timed REPEATS times and the median of the post-warm-up runs is reported.

Outputs:
  run/moe_workload/moe_results.csv     per-trace raw results
  run/moe_workload/moe_summary.csv     mean per (regime, engine, config)
"""
from __future__ import annotations
from _paths import REPO  # repository-relative paths

import os
import sys
import time


import numpy as np
import pandas as pd

from src.algorithms.bvn import bvn_decomposition
from src.algorithms.euler_splitting import decompose_euler_framework
from src.algorithms.radix_decomposition import decompose_radix
from src.utils.moe_trace import load_all

ROOT = str(REPO)
TRACES = os.path.join(ROOT, "paper_traces")
OUTDIR = os.path.join(ROOT, "run", "moe_workload")

ENGINES = [("Hungarian", "maximum"), ("WFA", "wfa"),
           ("GW Dynamic", "heavy"), ("GW Static", "heavy_static")]
BASES = [16, 8, 4, 2]
DEPTHS = [1, 2, 3, 4]
REPEATS = 9          # timed runs per configuration
WARMUP = 4           # leading runs discarded (Numba JIT + cache warm-up)


def _cycle(comps):
    return float(sum(c.weight for c in comps))


def _med(times):
    return float(np.median(times[WARMUP:]))


def lower_bound_S(M):
    """Cycle-length lower bound; equals the regular S when the matrix is regular."""
    return int(max(M.sum(axis=1).max(), M.sum(axis=0).max()))


def run_all(M, engine_key):
    """One dict per configuration, for a single matrix + engine."""
    out = []

    # --- BvN baseline ---
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        comps = bvn_decomposition(M.copy(), matching_algorithm=engine_key)
        ts.append((time.perf_counter() - t0) * 1000)
    rt = _med(ts)
    recon = sum(c.weight * c.permutation for c in comps)
    out.append(dict(config="BvN", N=len(comps), C=_cycle(comps), runtime_ms=rt,
                    split_ms=0.0, unit_ms=rt, units=1,
                    full_perms=sum(1 for c in comps if c.permutation.sum() == M.shape[0]),
                    recon_err=float(np.abs(recon - M).max())))

    # --- Radix (runtime = slowest plane, as in the TM campaign) ---
    for b in BASES:
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            comps, max_plane_s, n_planes = decompose_radix(
                M.copy(), base=b, matching_method=engine_key)
            ts.append(max_plane_s * 1000)
            _ = time.perf_counter() - t0
        rt = _med(ts)
        recon = sum(c.matrix for c in comps)
        out.append(dict(config=f"B{b}", N=len(comps), C=_cycle(comps), runtime_ms=rt,
                        split_ms=0.0, unit_ms=rt, units=n_planes,
                        full_perms=sum(1 for c in comps if (c.matrix > 0).sum() == M.shape[0]),
                        recon_err=float(np.abs(recon - M).max())))

    # --- Euler (runtime = split + slowest leaf, end-to-end) ---
    # NOTE: split_method="heuristic" is the same-direction split the paper measures.
    # The function's own default is "euler" (plain alternating) -- passing this
    # explicitly is required to match the runner and the published campaigns.
    for d in DEPTHS:
        ts, sp = [], []
        for _ in range(REPEATS):
            comps, split_s, max_leaf_s, n_leaves = decompose_euler_framework(
                M.copy(), matching_method=engine_key, depth=d, split_method="heuristic")
            ts.append((split_s + max_leaf_s) * 1000)
            sp.append(split_s * 1000)
        recon = sum(c.weight * c.permutation for c in comps)
        out.append(dict(config=f"Euler d{d}", N=len(comps), C=_cycle(comps),
                        runtime_ms=_med(ts), split_ms=_med(sp),
                        unit_ms=_med(ts) - _med(sp), units=n_leaves,
                        full_perms=sum(1 for c in comps if c.permutation.sum() == M.shape[0]),
                        recon_err=float(np.abs(recon - M).max())))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    traces = load_all(TRACES)
    rows = []

    for t in traces:
        M = t.matrix.astype(np.float64)
        S = lower_bound_S(M)
        dens = float((M > 0).mean())
        skew = float(M.sum(axis=0).max() / M.sum(axis=0).min())
        regime = "MoE mmlu" if "mmlu" in t.dataset else "MoE throughput"

        for eng_name, ek in ENGINES:
            for rec in run_all(M, ek):
                rec.update(regime=regime, knob=t.model, matrix=t.label, engine=eng_name,
                           S=S, density=dens, n=t.n, tokens=int(t.matrix.sum()),
                           layer=t.layer_id, dataset=t.dataset, col_skew=skew)
                rows.append(rec)
        print(f"[{regime}] {t.label}  S={S}  density={dens*100:.0f}%", flush=True)

    df = pd.DataFrame(rows)
    df["C_over_S"] = df["C"] / df["S"]
    df["eta"] = df["S"] / df["C"]
    df["frac_full"] = df["full_perms"] / df["N"]
    df.to_csv(os.path.join(OUTDIR, "moe_results.csv"), index=False)

    summary = (df.groupby(["regime", "engine", "config"], sort=False)
                 .agg(S=("S", "mean"), N=("N", "mean"), C=("C", "mean"),
                      C_over_S=("C_over_S", "mean"), eta=("eta", "mean"),
                      frac_full=("frac_full", "mean"), runtime_ms=("runtime_ms", "mean"),
                      split_ms=("split_ms", "mean"), unit_ms=("unit_ms", "mean"),
                      units=("units", "mean"), density=("density", "mean"))
                 .reset_index())
    summary.to_csv(os.path.join(OUTDIR, "moe_summary.csv"), index=False)

    worst = df.recon_err.max()
    print(f"\nmax reconstruction error over all {len(df)} runs: {worst:.3g}")

    order = ["BvN"] + [f"B{b}" for b in BASES] + [f"Euler d{d}" for d in DEPTHS]
    for regime in df.regime.unique():
        sub = summary[summary.regime == regime]
        print(f"\n================ {regime.upper()}  (n=8, 6 traces) "
              f"S={sub.S.iloc[0]:.0f} ================")
        print(f"{'engine':11s} {'config':10s} {'runtime(ms)':>12s} {'N':>8s} {'C':>9s} "
              f"{'C/S':>6s} {'eta':>6s} {'full':>6s}")
        for eng_name, _ in ENGINES:
            for cfg in order:
                r = sub[(sub.engine == eng_name) & (sub.config == cfg)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"{eng_name:11s} {cfg:10s} {r.runtime_ms:12.3f} {r.N:8.0f} "
                      f"{r.C:9.0f} {r.C_over_S:6.3f} {r.eta:6.3f} {r.frac_full:6.0%}")

    print(f"\nsaved -> {OUTDIR}\\moe_results.csv")
    print(f"saved -> {OUTDIR}\\moe_summary.csv")


if __name__ == "__main__":
    main()
