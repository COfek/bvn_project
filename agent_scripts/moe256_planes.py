"""The digit planes of the n = 256 MoE matrices, and GW Static's repair on them (Chen K18).

Regenerates the 36 matrices of moe256_campaign.py (same seeds) and prints, per trace family and
for B = 16 and B = 8, per digit plane k (weight B^k): the mean range of the row sums, the largest
row or column sum S*, and the term B^k S* of the Radix cycle bound (Proposition prop:radixbound),
then the bound itself over S. This explains which base keeps the cycle short on which family.

Then, for GW Static, it times the slowest plane with and without the Augment repair
(heavy_static against heavy_static_noaug), to show where its B = 8 time goes: on an irregular
plane most extractions have no perfect matching, and the repair searches in vain. Timed on an
otherwise idle machine, one run per plane after a warm-up.
Run with BVN_CAMPAIGN=2026_11 BVN_RUN_ROOT=run/rerun_2026_11e.
"""
from __future__ import annotations
from _paths import REPO
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(REPO))
import moe256_campaign as mc
from src.utils.moe_trace import load_all, upsample
from src.algorithms.radix_decomposition import extract_digit_planes, _plane_plan

traces = load_all(mc.mt.TRACES)
mats = []
for i, t in enumerate(traces):
    fam = "MoE mmlu" if "mmlu" in t.dataset else "MoE throughput"
    for s in range(mc.SAMPLES):
        mats.append((fam, mc.matrix_for(i, s, t)))

rows = []
for fam, M in mats:
    S = max(M.sum(0).max(), M.sum(1).max())
    for B in (16, 8):
        bound = 0.0
        for k, (w, P) in enumerate(extract_digit_planes(M, B)):
            r, c = P.sum(1), P.sum(0)
            sst = max(r.max(), c.max())
            bound += w * sst
            rows.append(dict(family=fam, B=B, plane=int(w), row_lo=r.min(), row_hi=r.max(),
                             S_star=sst, term=w * sst, S=S, entry_mean=M.mean(),
                             entry_max=M.max()))
        rows.append(dict(family=fam, B=B, plane=-1, term=bound, S=S))
df = pd.DataFrame(rows)
for fam in ("MoE throughput", "MoE mmlu"):
    d = df[df.family == fam]
    one = d[(d.B == 16) & (d.plane == -1)]            # one row per matrix
    p0 = d[(d.B == 16) & (d.plane == 1)]
    print(f"=== {fam}: {len(one)} matrices, mean S {one.S.mean():.0f}, mean entry "
          f"{p0.entry_mean.mean():.1f}, largest entry {p0.entry_max.max():.0f}")
    for B in (16, 8):
        g = d[(d.B == B) & (d.plane > 0)].groupby("plane")
        e = g[["row_lo", "row_hi", "S_star", "term"]].mean()
        e["n"] = g.size()
        print(f"  B = {B}")
        for w, r in e.iterrows():
            print(f"    plane x{w:<3d} row sums {r.row_lo:7.1f} - {r.row_hi:7.1f}   S* {r.S_star:7.1f}"
                  f"   B^k S* {r.term:8.0f}   (present in {int(r.n)} matrices)")
        b = d[(d.B == B) & (d.plane == -1)]
        print(f"    bound sum_k B^k S* / S = {(b.term / b.S).mean():.3f}")

# GW Static's slowest plane, with and without the Augment repair
_plane_plan(extract_digit_planes(mats[0][1], 16)[0][1], 0, "heavy_static")       # JIT warm-up
_plane_plan(extract_digit_planes(mats[0][1], 16)[0][1], 0, "heavy_static_noaug")
out = []
for fam, M in mats:
    for B in (16, 8):
        for eng in ("heavy_static", "heavy_static_noaug"):
            worst = 0.0
            for w, P in extract_digit_planes(M, B):
                t0 = time.perf_counter()
                _plane_plan(P, 0, eng)
                worst = max(worst, time.perf_counter() - t0)
            out.append(dict(family=fam, B=B, engine=eng, slowest_ms=1000 * worst))
o = pd.DataFrame(out).groupby(["family", "B", "engine"]).slowest_ms.mean().unstack().round(0)
print("\nGW Static, slowest plane [ms], with and without Augment (mean over 18 matrices):")
print(o.to_string())
