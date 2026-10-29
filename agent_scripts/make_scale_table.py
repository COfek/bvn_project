"""LaTeX table for the Radix scaling result (per-matrix best scale, then averaged)."""
import os
import pandas as pd

ROOT = r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project"
PAPER = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper"
d = pd.read_csv(os.path.join(ROOT, "run", "radix_scale", "scale_results.csv"))

BASES = [16, 8, 4, 2]
ENG = "Hungarian"
L = [r"\begin{tabular}{l r rr r rr rr}", r"\toprule",
     r"& & \multicolumn{2}{c}{$C/S$} & & \multicolumn{2}{c}{$N$}"
     r" & \multicolumn{2}{c}{$T_{compute}$ (ms)} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r"Workload & Base & unscaled & scaled & $\Delta$ & unscaled & scaled"
     r" & unscaled & scaled \\", r"\midrule"]

for wi, reg in enumerate(["TM dense", "Uniform"]):
    s = d[(d.regime == reg) & (d.engine == ENG)]
    best = s.loc[s.groupby(["matrix", "base"]).C_over_S.idxmin()]
    un = s[s.scale == 1]
    for bi, B in enumerate(BASES):
        u = un[un.base == B]
        t = best[best.base == B]
        ucs, tcs = u.C_over_S.mean(), t.C_over_S.mean()
        delta = (tcs - ucs) / ucs * 100
        name = reg if bi == 0 else ""
        dstr = f"{delta:+.1f}\\%" if abs(delta) > 0.05 else "---"
        L.append(f"{name} & {B} & {ucs:.3f} & {tcs:.3f} & {dstr} & "
                 f"{u.N.mean():.0f} & {t.N.mean():.0f} & "
                 f"{u.runtime_ms.mean():.0f} & {t.runtime_ms.mean():.0f} " + r"\\")
    L.append(r"\midrule" if wi == 0 else r"\bottomrule")
L.append(r"\end{tabular}")
open(os.path.join(PAPER, "scale_tab.tex"), "w", encoding="utf-8").write("\n".join(L))
print("wrote scale_tab.tex")

# console echo
for reg in ["TM dense", "Uniform"]:
    s = d[(d.regime == reg) & (d.engine == ENG)]
    best = s.loc[s.groupby(["matrix", "base"]).C_over_S.idxmin()]
    un = s[s.scale == 1]
    print(f"\n{reg}:")
    for B in BASES:
        u, t = un[un.base == B], best[best.base == B]
        print(f"  B{B:<3} C/S {u.C_over_S.mean():.4f} -> {t.C_over_S.mean():.4f}   "
              f"N {u.N.mean():.0f} -> {t.N.mean():.0f}   "
              f"T {u.runtime_ms.mean():.0f} -> {t.runtime_ms.mean():.0f} ms")
