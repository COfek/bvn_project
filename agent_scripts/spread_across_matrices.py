"""Spread across the matrices behind Figures 3 and 4 (Chen K10).

Each point of Figures 3 (Radix) and 4 (Euler) is a mean over the 995 matrices; the figures shade
one standard deviation across those matrices around each line. This prints, for the text:
  - the largest standard deviation of N and of T_parallel, as a percentage of the mean, over
    every setup of each figure;
  - for GW Static under Euler, on how many matrices N at each depth is below N unsplit.
Run with BVN_CAMPAIGN=2026_11 BVN_RUN_ROOT=run/rerun_2026_11e, as for the figures.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_data import radix_frame, euler_frame  # noqa: E402

ENGINES = {"Hungarian": ("max", "maximum"), "WWFA": ("wwfa", "wwfa"),
           "GW Dynamic": ("heavy", "heavy"), "GW Static": ("heavy_static", "heavy_static")}
BASES, DEPTHS = (16, 8, 4, 2), (1, 2, 3, 4)


def rel_sd(x):
    return x.std() / x.mean() * 100


rows = []   # (figure, engine, point, metric, sd as % of mean)
for eng, (rkey, pre) in ENGINES.items():
    d = radix_frame(rkey)
    pts = [("unsplit", d["bvn_runtime"], d["bvn_perms"])] + [
        (f"B={B}", d[f"{pre}_{B}_time"], d[f"{pre}_{B}_perms"]) for B in BASES]
    for p, T, N in pts:
        rows += [("Radix", eng, p, "T", rel_sd(T)), ("Radix", eng, p, "N", rel_sd(N))]
    e = euler_frame(pre)
    pts = [("unsplit", e["bvn_runtime"], e["bvn_perms"])] + [
        (f"depth {k}", e[f"euler_heuristic_ruined_depth{k}_time"]
         + e[f"euler_heuristic_ruined_depth{k}_split"], e[f"euler_heuristic_ruined_depth{k}_perms"])
        for k in DEPTHS]
    for p, T, N in pts:
        rows += [("Euler", eng, p, "T", rel_sd(T)), ("Euler", eng, p, "N", rel_sd(N))]
    if eng == "GW Static":
        n = len(e)
        for k in DEPTHS:
            below = int((e[f"euler_heuristic_ruined_depth{k}_perms"] < e["bvn_perms"]).sum())
            print(f"GW Static, Euler depth {k}: N below unsplit on {below} of {n} matrices")

for fig in ("Radix", "Euler"):
    for m in ("N", "T"):
        r = max((x for x in rows if x[0] == fig and x[3] == m), key=lambda x: x[4])
        print(f"{fig}: largest sd of {m} = {r[4]:.1f}% of the mean ({r[1]}, {r[2]})")
both_N = max((x for x in rows if x[3] == "N"), key=lambda x: x[4])
print(f"both figures: largest sd of N = {both_N[4]:.1f}% ({both_N[0]}, {both_N[1]}, {both_N[2]})")
