"""Radix with a SCALE knob: Radix(B, c).

Motivation.  The cycle length of a Radix decomposition is governed not by the base B
alone but by how tightly the entry range fits the radix representation.  With D =
ceil(log_B(max+1)) planes the representable range is B^D, and when B^D overshoots the
actual maximum entry the decomposition wastes a plane and inflates the cycle (measured:
base 16 on a matrix with max=354 overshoots 11.6x and loses to base 8).

Fix.  Scale the matrix by an integer c BEFORE decomposing, chosen so that c*max lands
just below B^D.  This is legitimate preprocessing, not a measurement trick: if
sum_i lambda_i P_i = c*M then sum_i (lambda_i/c) P_i = M, so dividing every weight by c
yields a valid decomposition of the ORIGINAL matrix with the SAME permutation count and
the SAME cycle ratio C/S.  Hold times become multiples of 1/c, which is physically fine.

For every base we try c=1 (baseline) plus the tight scales for the natural plane count
and one plane more, and we verify reconstruction of the original matrix in every case.

Outputs:
  run/radix_scale/scale_results.csv    per (regime, matrix, engine, base, scale)
  run/radix_scale/scale_summary.csv    means
"""
from __future__ import annotations
from _paths import REPO  # repository-relative paths
import os, sys, time

import numpy as np
import pandas as pd

from src.algorithms.radix_decomposition import decompose_radix
from src.utils.griner_avin_tm import tm_integer
from src.utils.matrix_generator import generate_matrix

ROOT = str(REPO)
OUTDIR = os.path.join(ROOT, "run", "radix_scale")

N = 256
N_MATRICES = 4
BASES = [16, 8, 4, 2]
ENGINES = [("Hungarian", "maximum"), ("WFA", "wfa"),
           ("GW Dynamic", "heavy"), ("GW Static", "heavy_static")]
REGIMES = [("TM dense", "tm", 512), ("Uniform", "uniform", 256)]


def planes_for(B, mx):
    D = 1
    while B ** D <= mx:
        D += 1
    return D


def scale_candidates(B, mx):
    """c=1, plus the tight scale for the natural plane count D0 and for D0+1."""
    D0 = planes_for(B, mx)
    cands = {1: D0}
    for D in (D0, D0 + 1):
        c = (B ** D - 1) // mx          # largest c with c*mx <= B^D - 1
        if c >= 2:
            cands[c] = planes_for(B, c * mx)
    return sorted(cands.items())        # [(c, D), ...]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for regime, kind, knob in REGIMES:
        rng = np.random.default_rng(42)
        mk = (lambda r: tm_integer(N, knob, r, w_small=7)) if kind == "tm" \
            else (lambda r: generate_matrix(N, k=knob, max_weight=64, rng=r))
        # warm-up (JIT)
        Mw = mk(rng)
        for _, ek in ENGINES:
            decompose_radix(Mw.copy(), base=2, matching_method=ek)
        print(f"[{regime}] warm-up done", flush=True)

        for m in range(N_MATRICES):
            M0 = mk(rng)
            S0 = int(M0.sum(axis=1)[0]); mx0 = int(M0.max())
            for eng, ek in ENGINES:
                for B in BASES:
                    for c, D in scale_candidates(B, mx0):
                        M = M0 * c
                        S = int(M.sum(axis=1)[0]); mx = int(M.max())
                        t0 = time.perf_counter()
                        comps, plane_s, npl = decompose_radix(
                            M.copy(), base=B, matching_method=ek)
                        _wall = time.perf_counter() - t0
                        C = float(sum(x.weight for x in comps))
                        # verify: dividing weights by c reconstructs the ORIGINAL matrix
                        R = np.zeros_like(M0)
                        for x in comps:
                            R += (x.weight / c) * x.matrix
                        err = float(np.abs(R - M0).max())
                        rows.append(dict(
                            regime=regime, matrix=m, engine=eng, base=B, scale=c,
                            planes=D, max_entry=mx, overshoot=(B ** D) / mx,
                            S_scaled=S, N=len(comps), C_scaled=C, C_over_S=C / S,
                            C_original=C / c, runtime_ms=plane_s * 1000,
                            recon_err=err))
            print(f"[{regime}] matrix {m+1}/{N_MATRICES}  S0={S0} max0={mx0}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "scale_results.csv"), index=False)
    bad = df[df.recon_err > 1e-6]
    print(f"\nreconstruction check: {len(df)-len(bad)}/{len(df)} exact"
          f"{'  !! ' + str(len(bad)) + ' FAILED' if len(bad) else '  (all valid)'}")

    summ = (df.groupby(["regime", "engine", "base", "scale"], sort=False)
              .agg(planes=("planes", "mean"), overshoot=("overshoot", "mean"),
                   N=("N", "mean"), C_over_S=("C_over_S", "mean"),
                   runtime_ms=("runtime_ms", "mean"), recon_err=("recon_err", "max"))
              .reset_index())
    summ.to_csv(os.path.join(OUTDIR, "scale_summary.csv"), index=False)

    for regime, _, _ in REGIMES:
        print(f"\n================= {regime} =================")
        for eng, _ in ENGINES:
            s = summ[(summ.regime == regime) & (summ.engine == eng)]
            print(f"\n  {eng}")
            print(f"    {'base':>4} {'scale':>6} {'planes':>7} {'over':>6} {'C/S':>7} "
                  f"{'N':>7} {'T(ms)':>8}")
            for B in BASES:
                sb = s[s.base == B].sort_values("scale")
                base_cs = sb[sb.scale == 1].C_over_S.iloc[0]
                for _, r in sb.iterrows():
                    tag = ""
                    if r.scale > 1:
                        tag = " <-- BEST" if r.C_over_S == sb.C_over_S.min() and r.C_over_S < base_cs else ""
                    print(f"    {int(r.base):>4} {int(r.scale):>6} {int(r.planes):>7} "
                          f"{r.overshoot:>5.1f}x {r.C_over_S:>7.3f} {r.N:>7.0f} "
                          f"{r.runtime_ms:>8.1f}{tag}")
    print("\nSCALE CAMPAIGN COMPLETE")


if __name__ == "__main__":
    main()
