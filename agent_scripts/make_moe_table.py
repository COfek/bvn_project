"""Summary table for section 5.8 (measured MoE traffic).

Mirrors the layout of tm_tab_comparison.tex: engine x config rows, metric column
groups, one column wide.  The two column groups are the two trace families
(throughput bench vs MMLU) rather than two generators, and each cell is a mean
over the six traces in that family.

Configs follow tm_tab_comparison (BvN / B2 / Euler d4) so the two workloads can be
read against each other, plus the best radix base for each family -- which is not
the same base for both.  The full grid is in run/moe_workload/.

NOTE: cells average traces whose S differs (e.g. 19,613 and 22,188 for the two
Mixtral-8x22B throughput layers), so C is a mean over matrices of different sizes
and C/S here is a ratio of means.  Per-model, per-layer figures are in
run/moe_workload/moe_results.csv.
"""
from __future__ import annotations
from _paths import PAPER  # repository-relative paths

import os

import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from radix_extraction import correct_summary, correct_scale_results  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = str(PAPER / "moe_tab_summary.tex")

ENGINES = ["Hungarian", "GW Dynamic", "GW Static", "WFA"]
REG = [("MoE throughput", "Thr."), ("MoE mmlu", "MMLU")]
RADIX = ["B16", "B8", "B4", "B2"]   # NB: not startswith("B") -- that matches "BvN"

s = correct_summary(pd.read_csv(os.path.join(ROOT, "run", "moe_workload", "moe_summary.csv")), "moe")  # charge Radix's digit extraction


def num(x, dec=0):
    """Format with the thousands separator the paper's tables use."""
    return f"{x:,.{dec}f}".replace(",", "{,}")


best_base = {}
for reg, _ in REG:
    sub = s[(s.regime == reg) & (s.config.isin(RADIX))]
    best_base[reg] = sub.groupby("config").C_over_S.mean().idxmin()
print("best radix base per regime:", best_base)

rows = []
for eng in ENGINES:
    cfgs = ["BvN", None, "B2", "Euler d4"]
    block = []
    for cfg in cfgs:
        cells = []
        label = cfg
        for reg, _ in REG:
            c = best_base[reg] if cfg is None else cfg
            r = s[(s.regime == reg) & (s.engine == eng) & (s.config == c)].iloc[0]
            cells.append((r.runtime_ms, r.N, r.C_over_S))
        if cfg is None:
            names = {best_base[r] for r, _ in REG}
            label = best_base[REG[0][0]] if len(names) == 1 else \
                f"{best_base[REG[0][0]]}/{best_base[REG[1][0]]}"
        block.append((label, cells))
    rows.append((eng, block))

L = [r"\begin{tabular}{ll rr rr rr}", r"\toprule",
     r"& & \multicolumn{2}{c}{$\Tc$ (ms)} & \multicolumn{2}{c}{$N$}"
     r" & \multicolumn{2}{c}{$C/S$} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"\textbf{Engine} & \textbf{Config} & Thr. & MMLU & Thr. & MMLU & Thr. & MMLU \\",
     r"\midrule"]

for i, (eng, block) in enumerate(rows):
    if i:
        L.append(r"\midrule")
    L.append(r"\multirow{4}{*}{%s}" % eng)
    for label, cells in block:
        (t0, n0, c0), (t1, n1, c1) = cells
        L.append(f"  & {label} & {num(t0,2)} & {num(t1,2)} & {num(n0)} & {num(n1)}"
                 f" & {c0:.3f} & {c1:.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("wrote", OUT)
