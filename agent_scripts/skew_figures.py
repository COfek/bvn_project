"""Figures + LaTeX tables for the skew-controlled comparison of the two generators.

Reads  run/skew/skew_summary.csv   (from skew_campaign.py)
Writes paper/plots_pdf/skew_*.pdf|png  and  paper/skew_tab_*.tex
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8})

ROOT = r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project"
PLOTS = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper\plots_pdf"
PAPER = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper"
SUMF = os.path.join(ROOT, "run", "skew", "skew_summary.csv")

ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WFA", "#CC79A7", "D")]
NEW, OLD = "TM (new)", "Uniform (old)"
BASES = [16, 8, 4, 2]
DEPTHS = [1, 2, 3, 4]
RADIX_ORDER = ["BvN"] + [f"B{b}" for b in BASES]
EULER_ORDER = ["BvN"] + [f"Euler d{d}" for d in DEPTHS]

df = pd.read_csv(SUMF)
NFS = sorted(df.nf.unique())


def pick(workload, nf, engine):
    return df[(df.workload == workload) & (df.nf == nf) & (df.engine == engine)]\
        .set_index("config")


def props(workload, nf):
    r = df[(df.workload == workload) & (df.nf == nf)].iloc[0]
    return dict(S=r.S, density=r.density, var_dist=r.var_dist)


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS, f"{name}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def _fmt(x, d=0):
    return f"{x:,.{d}f}".replace(",", "{,}")


# ---------------------------------------------------------------- FIG A
def fig_divergence():
    """For three representative configurations, plot each metric against the skew
    axis, with the new generator solid and the S-matched old generator dashed."""
    cfgs = ["BvN", "B2", "Euler d4"]
    metrics = [("C_over_S", "Cycle ratio $C/S$"),
               ("N", "Permutation count $N$"),
               ("runtime_ms", "$T_{compute}$ [ms]")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    vd = [props(NEW, nf)["var_dist"] for nf in NFS]
    for r, (mkey, mlab) in enumerate(metrics):
        for c, cfg in enumerate(cfgs):
            ax = axes[r, c]
            for name, col, mk in ENG:
                yn = [pick(NEW, nf, name).loc[cfg, mkey] for nf in NFS]
                yo = [pick(OLD, nf, name).loc[cfg, mkey] for nf in NFS]
                ax.plot(NFS, yn, "-", marker=mk, color=col, lw=1.6, ms=6,
                        markeredgecolor="white", markeredgewidth=.6, label=name)
                ax.plot(NFS, yo, "--", marker=mk, color=col, lw=1.3, ms=5, alpha=.5,
                        markerfacecolor="none")
            ax.set_xscale("log", base=2)
            ax.set_xticks(NFS); ax.set_xticklabels([str(x) for x in NFS])
            if mkey in ("N", "runtime_ms"):
                ax.set_yscale("log")
            if mkey == "C_over_S":
                ax.axhline(1.0, color="gray", ls=":", lw=1.2)
            ax.grid(True, which="both", ls=":", lw=.4, alpha=.6)
            if r == 0:
                ax.set_title(f"{cfg}", fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(mlab)
            if r == 2:
                ax.set_xlabel("$n_f$  (fewer flows $\\Rightarrow$ more skew $\\rightarrow$)")
    # secondary annotation: skew level
    for c in range(3):
        ax = axes[0, c]
        for x, v in zip(NFS, vd):
            ax.annotate(f"{v:.2f}", (x, ax.get_ylim()[1]), fontsize=6.5, color="gray",
                        ha="center", va="bottom", annotation_clip=False)
    h = [Line2D([], [], color=c, marker=m, ls="-", label=n) for n, c, m in ENG]
    h += [Line2D([], [], color="k", ls="-", label="TM (new generator)"),
          Line2D([], [], color="k", ls="--", alpha=.5, label="Uniform (old), $S$-matched")]
    fig.legend(handles=h, ncol=6, loc="lower center", fontsize=9.5, framealpha=.95,
               bbox_to_anchor=(.5, -.02))
    fig.suptitle("Skew-controlled comparison of the two generators\n"
                 "each TM regime is paired with an $S$-matched uniform control; "
                 "grey numbers = variation distance from the uniform matrix",
                 fontsize=13.5, y=1.005)
    fig.tight_layout(rect=[0, .04, 1, .99])
    save(fig, "skew_divergence")


# ---------------------------------------------------------------- FIG B
def fig_workload_stats():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (k, lab) in zip(axes, [("var_dist", "Variation distance from uniform matrix"),
                                   ("density", "Density (fraction non-zero)"),
                                   ("S", "Scale $S$")]):
        for wl, ls, mk, al in [(NEW, "-", "o", 1.0), (OLD, "--", "x", .65)]:
            y = [props(wl, nf)[k] for nf in NFS]
            ax.plot(NFS, y, ls, marker=mk, lw=1.6, ms=7, alpha=al,
                    color="#0072B2" if wl == NEW else "#D55E00", label=wl)
        ax.set_xscale("log", base=2); ax.set_xticks(NFS)
        ax.set_xticklabels([str(x) for x in NFS])
        if k == "S":
            ax.set_yscale("log")
        ax.set_xlabel("$n_f$"); ax.set_ylabel(lab)
        ax.grid(True, which="both", ls=":", lw=.4, alpha=.6)
        ax.legend(fontsize=9, framealpha=.95)
    fig.suptitle("Properties of the generated matrices: the two generators converge as $n_f$ grows",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    save(fig, "skew_workload_stats")


# ---------------------------------------------------------------- TABLES
def tab_framework(workload, nf, order, fname, unit_col="Runtime (ms)"):
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"\textbf{Matching} & \textbf{Method} & \textbf{" + unit_col +
         r"} & \textbf{Cycle Length} & \textbf{Permutations} \\", r"\midrule"]
    for i, (name, _, _) in enumerate(ENG):
        d = pick(workload, nf, name).reindex(order).dropna()
        b_rt, b_c, b_n = d.runtime_ms.min(), d.C.min(), d.N.min()
        L.append(r"\multirow{%d}{*}{%s}" % (len(d), name))
        for cfg, r in d.iterrows():
            lbl = ("BvN baseline" if cfg == "BvN" else
                   (f"Radix $B={cfg[1:]}$" if cfg.startswith("B") else
                    cfg.replace("Euler d", "depth ")))
            rt, cc, nn = _fmt(r.runtime_ms, 1), _fmt(r.C), _fmt(r.N)
            if abs(r.runtime_ms - b_rt) < 1e-9: rt = r"\textbf{%s}" % rt
            if abs(r.C - b_c) < 1e-9: cc = r"\textbf{%s}" % cc
            if abs(r.N - b_n) < 1e-9: nn = r"\textbf{%s}" % nn
            L.append(f"  & {lbl} & {rt} & {cc} & {nn} " + r"\\")
        L.append(r"\midrule" if i < len(ENG) - 1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(PAPER, fname), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", fname)


def tab_workload_props(fname="skew_tab_props.tex"):
    L = [r"\begin{tabular}{r rrr rrr}", r"\toprule",
         r"& \multicolumn{3}{c}{\textbf{TM (new generator)}} &"
         r" \multicolumn{3}{c}{\textbf{Uniform (old), $S$-matched}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"$n_f$ & $S$ & density & var.\ dist. & $S$ & density & var.\ dist. \\", r"\midrule"]
    for nf in NFS:
        a, b = props(NEW, nf), props(OLD, nf)
        L.append(f"{nf} & {_fmt(a['S'])} & {a['density']*100:.0f}\\% & {a['var_dist']:.3f} & "
                 f"{_fmt(b['S'])} & {b['density']*100:.0f}\\% & {b['var_dist']:.3f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, fname), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", fname)


def tab_sweep(fname="skew_tab_sweep.tex"):
    """Head-to-head at every skew level for the three headline configurations."""
    L = [r"\begin{tabular}{r l rr rr rr}", r"\toprule",
         r"& & \multicolumn{2}{c}{$C/S$} & \multicolumn{2}{c}{$N$}"
         r" & \multicolumn{2}{c}{$T_{compute}$ (ms)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"$n_f$ & Config & New & Old & New & Old & New & Old \\", r"\midrule"]
    for i, nf in enumerate(NFS):
        for j, cfg in enumerate(["BvN", "B2", "Euler d4"]):
            a = pick(NEW, nf, "Hungarian").loc[cfg]
            b = pick(OLD, nf, "Hungarian").loc[cfg]
            w = pick(NEW, nf, "WFA").loc[cfg]
            wo = pick(OLD, nf, "WFA").loc[cfg]
            head = f"{nf}" if j == 0 else ""
            L.append(f"{head} & {cfg} & {a.C_over_S:.3f} & {b.C_over_S:.3f} & "
                     f"{_fmt(a.N)} & {_fmt(b.N)} & {_fmt(a.runtime_ms,1)} & "
                     f"{_fmt(b.runtime_ms,1)} " + r"\\")
        L.append(r"\midrule" if i < len(NFS) - 1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(PAPER, fname), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", fname)


def tab_wfa_skew(fname="skew_tab_wfa.tex"):
    """WFA cycle inflation is the metric most sensitive to skew."""
    L = [r"\begin{tabular}{r rr rr}", r"\toprule",
         r"& \multicolumn{2}{c}{WFA BvN $C/S$} & \multicolumn{2}{c}{Radix $B{=}2$ $C/S$ (Hungarian)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"$n_f$ & New (TM) & Old (unif.) & New (TM) & Old (unif.) \\", r"\midrule"]
    for nf in NFS:
        w_n = pick(NEW, nf, "WFA").loc["BvN", "C_over_S"]
        w_o = pick(OLD, nf, "WFA").loc["BvN", "C_over_S"]
        r_n = pick(NEW, nf, "Hungarian").loc["B2", "C_over_S"]
        r_o = pick(OLD, nf, "Hungarian").loc["B2", "C_over_S"]
        L.append(f"{nf} & {w_n:.3f} & {w_o:.3f} & {r_n:.3f} & {r_o:.3f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, fname), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", fname)


if __name__ == "__main__":
    print("n_f levels:", NFS)
    fig_workload_stats()
    fig_divergence()
    tab_workload_props()
    tab_sweep()
    tab_wfa_skew()
    # reference regimes: most-skewed and least-skewed
    lo, hi = NFS[0], NFS[-1]
    tab_framework(OLD, hi, RADIX_ORDER, "skew_tab_old_radix.tex")
    tab_framework(OLD, hi, EULER_ORDER, "skew_tab_old_euler.tex", "Total (ms)")
    tab_framework(NEW, hi, RADIX_ORDER, "skew_tab_new_radix_dense.tex")
    tab_framework(NEW, hi, EULER_ORDER, "skew_tab_new_euler_dense.tex", "Total (ms)")
    tab_framework(NEW, lo, RADIX_ORDER, "skew_tab_new_radix_skewed.tex")
    tab_framework(NEW, lo, EULER_ORDER, "skew_tab_new_euler_skewed.tex", "Total (ms)")
    print("\n--- headline: does skew change the conclusions? ---")
    for nf in NFS:
        a, b = props(NEW, nf), props(OLD, nf)
        wn = pick(NEW, nf, "WFA").loc["BvN", "C_over_S"]
        wo = pick(OLD, nf, "WFA").loc["BvN", "C_over_S"]
        hn = pick(NEW, nf, "Hungarian").loc["Euler d4", "C_over_S"]
        print(f"  n_f={nf:4d}  vd(new)={a['var_dist']:.3f} vd(old)={b['var_dist']:.3f} | "
              f"WFA BvN C/S new={wn:.3f} old={wo:.3f} | Hung Euler d4 C/S new={hn:.3f}")
