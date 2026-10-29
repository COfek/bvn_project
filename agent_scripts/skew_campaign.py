"""Skew-controlled campaign: the OLD generator (uniform-random weights) vs the NEW
generator (Griner-Avin traffic model), across the full skew range.

Motivation.  A first campaign compared TM at n_f=512 against the uniform benchmark and
found near-identical behaviour.  Diagnosis: at large n_f each matrix cell is the sum of
many flows, so the elephant/mice structure averages out and the TM matrix becomes
statistically almost identical to a uniform-random one (variation distance from the
uniform matrix 0.468 vs 0.474).  Griner & Avin document the same effect.  The skew only
survives at SMALL n_f.

This campaign therefore sweeps n_f and pairs every TM regime with an S-MATCHED uniform
control, so the only variable is the traffic skew.  For each matrix we also record the
variation distance from the uniform matrix, giving a principled skew axis.

Outputs:
  run/skew/skew_results.csv    per-matrix raw
  run/skew/skew_summary.csv    mean per (workload, n_f/k, engine, config)
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project")

import numpy as np
import pandas as pd

from src.algorithms.bvn import bvn_decomposition
from src.algorithms.radix_decomposition import decompose_radix
from src.algorithms.euler_splitting import decompose_euler_framework
from src.utils.griner_avin_tm import tm_integer
from src.utils.matrix_generator import generate_matrix

ROOT = r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project"
OUTDIR = os.path.join(ROOT, "run", "skew")

N = 256
N_MATRICES = 6
NF_SWEEP = [16, 32, 64, 128, 256, 512]
ENGINES = [("Hungarian", "maximum"), ("WFA", "wfa"),
           ("GW Dynamic", "heavy"), ("GW Static", "heavy_static")]
BASES = [16, 8, 4, 2]
DEPTHS = [1, 2, 3, 4]
MAXW = 64            # uniform generator weight cap; mean weight = MAXW/2


def var_dist(M):
    """Total-variation distance between M (normalised) and the uniform matrix."""
    S = M.sum(axis=1)[0]
    U = S / M.shape[0]
    return float(np.abs(M - U).sum() / (2.0 * M.sum()))


def _cycle(comps):
    return float(sum(c.weight for c in comps))


def run_all(M, engine_key):
    out = []
    t0 = time.perf_counter()
    comps = bvn_decomposition(M.copy(), matching_algorithm=engine_key)
    rt = (time.perf_counter() - t0) * 1000
    out.append(dict(config="BvN", N=len(comps), C=_cycle(comps), runtime_ms=rt,
                    split_ms=0.0, unit_ms=rt, units=1))
    for b in BASES:
        comps, mp_s, n_planes = decompose_radix(M.copy(), base=b, matching_method=engine_key)
        out.append(dict(config=f"B{b}", N=len(comps), C=_cycle(comps),
                        runtime_ms=float(mp_s) * 1000, split_ms=0.0,
                        unit_ms=float(mp_s) * 1000, units=n_planes))
    for d in DEPTHS:
        comps, sp_s, leaf_s, n_leaves = decompose_euler_framework(
            M.copy(), matching_method=engine_key, depth=d, split_method="heuristic")
        out.append(dict(config=f"Euler d{d}", N=len(comps), C=_cycle(comps),
                        runtime_ms=(sp_s + leaf_s) * 1000, split_ms=sp_s * 1000,
                        unit_ms=leaf_s * 1000, units=n_leaves))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Probe each TM regime once to learn its S, then derive the S-matched uniform k.
    probe = np.random.default_rng(7)
    regimes = []
    for nf in NF_SWEEP:
        S_tm = int(tm_integer(N, nf, probe, w_small=7).sum(axis=1)[0])
        k = max(1, int(round(S_tm / (MAXW / 2))))     # uniform: E[S] = k * MAXW/2
        regimes.append(("TM (new)", nf, "tm", nf, S_tm))
        regimes.append(("Uniform (old)", nf, "uniform", k, S_tm))
    print("regime plan:")
    for w, nf, kind, knob, S in regimes:
        print(f"  {w:14s} n_f={nf:4d}  kind={kind:8s} knob={knob:4d}  S_target~{S}", flush=True)

    rows = []
    for workload, nf, kind, knob, _S in regimes:
        rng = np.random.default_rng(42)
        mk = (lambda r: tm_integer(N, knob, r, w_small=7)) if kind == "tm" \
            else (lambda r: generate_matrix(N, k=knob, max_weight=MAXW, rng=r))
        for _, ek in ENGINES:            # warm-up (JIT), discarded
            run_all(mk(rng), ek)
        for m in range(N_MATRICES):
            M = mk(rng)
            S = int(M.sum(axis=1)[0]); dens = float((M > 0).mean()); vd = var_dist(M)
            for eng, ek in ENGINES:
                for rec in run_all(M, ek):
                    rec.update(workload=workload, nf=nf, knob=knob, matrix=m,
                               engine=eng, S=S, density=dens, var_dist=vd)
                    rows.append(rec)
            print(f"[{workload} n_f={nf}] {m+1}/{N_MATRICES} S={S} dens={dens*100:.0f}% "
                  f"vd={vd:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df["C_over_S"] = df["C"] / df["S"]
    df.to_csv(os.path.join(OUTDIR, "skew_results.csv"), index=False)
    summ = (df.groupby(["workload", "nf", "engine", "config"], sort=False)
              .agg(S=("S", "mean"), density=("density", "mean"), var_dist=("var_dist", "mean"),
                   N=("N", "mean"), C=("C", "mean"), C_over_S=("C_over_S", "mean"),
                   runtime_ms=("runtime_ms", "mean"), split_ms=("split_ms", "mean"),
                   unit_ms=("unit_ms", "mean"), units=("units", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(OUTDIR, "skew_summary.csv"), index=False)
    print(f"\nsaved -> {OUTDIR}\\skew_summary.csv")
    print("SKEW CAMPAIGN COMPLETE")


if __name__ == "__main__":
    main()
