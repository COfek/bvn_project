"""Build the full paper-style artifact set for the Griner-Avin TM workload,
mirroring every figure/table of the conference draft, plus a head-to-head
comparison against the uniform (thesis) generator.

Reads   run/tm_workload/tm_summary.csv   (produced by tm_workload_experiment.py)
Writes  paper/plots_pdf/tm_*.pdf|png     and paper/tm_tab_*.tex
"""
from __future__ import annotations
from _paths import PAPER, PLOTS, REPO  # repository-relative paths
import os
import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from radix_extraction import correct_summary, correct_scale_results  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8})

ROOT = str(REPO)
PLOTS = str(PLOTS)
PAPER = str(PAPER)
SUM = os.path.join(ROOT, "run", "tm_workload", "tm_summary.csv")

ENG = [("Hungarian", "#0072B2", "o"), ("GW Dynamic", "#D55E00", "s"),
       ("GW Static", "#009E73", "^"), ("WFA", "#CC79A7", "D")]
BASES = [16, 8, 4, 2]
DEPTHS = [1, 2, 3, 4]
RADIX_ORDER = ["BvN"] + [f"B{b}" for b in BASES]
EULER_ORDER = ["BvN"] + [f"Euler d{d}" for d in DEPTHS]
PRIMARY = "TM dense"
BASELINE = "Uniform (thesis)"

def _tfmt(v, _p):
    if v >= 1: return f"{v:g} ms"
    if v >= 1e-3: return f"{v*1e3:g} µs"
    return f"{v*1e6:g} ns"
TFMT = FuncFormatter(_tfmt)

df = correct_summary(pd.read_csv(SUM), "tm")   # charge Radix's digit extraction


def sub(regime, engine):
    d = df[(df.regime == regime) & (df.engine == engine)].set_index("config")
    return d


def S_of(regime):
    return float(df[df.regime == regime].S.iloc[0])


def dens_of(regime):
    return float(df[df.regime == regime].density.mean())


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS, f"{name}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# =====================================================================
# FIG 1 — cross-engine Radix trade-off (runtime vs cycle | vs permutations)
# =====================================================================
def fig_radix(regime):
    S = S_of(regime)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for name, col, mk in ENG:
        d = sub(regime, name).reindex(RADIX_ORDER).dropna()
        for ax, xk in ((a1, "C"), (a2, "N")):
            ax.plot(d[xk], d.runtime_ms, "-", color=col, lw=1.5, alpha=.85)
            ax.plot(d[xk], d.runtime_ms, mk, color=col, ms=8, markeredgecolor="white",
                    markeredgewidth=.7, label=name)
            for cfg, r in d.iterrows():
                ax.annotate(cfg, (r[xk], r.runtime_ms), fontsize=7.5, color=col,
                            xytext=(4, 3), textcoords="offset points")
    a1.axvline(S, color="gray", ls="--", lw=1.2, label=f"$C=S={S:.0f}$")
    a1.set_xlabel("Decomposition cycle length  C  (lower is better)")
    a2.set_xscale("log"); a2.set_xlabel("Permutation count  N  (log, lower is better)")
    for ax, t in ((a1, "(a) runtime vs. cycle length"), (a2, "(b) runtime vs. permutation count")):
        ax.set_yscale("log"); ax.set_ylabel("Parallel runtime [ms] (log)")
        ax.set_title(t); ax.grid(True, which="both", ls=":", lw=.4, alpha=.6)
        ax.legend(title="Matching engine", fontsize=9, framealpha=.95)
    fig.suptitle(f"Cross-engine Radix trade-off — Griner--Avin TM workload ({regime}, "
                 f"n=256, S={S:.0f}, density={dens_of(regime)*100:.0f}%)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "tm_tradeoff_radix")


# =====================================================================
# FIG 2 — Euler trade-off (runtime vs N | runtime vs cycle)
# =====================================================================
def fig_euler(regime):
    S = S_of(regime)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for name, col, mk in ENG:
        d = sub(regime, name).reindex(EULER_ORDER).dropna()
        lab = [c.replace("Euler ", "") for c in d.index]
        a1.plot(d.N, d.runtime_ms, "-", color=col, lw=1.5, alpha=.85)
        a1.plot(d.N, d.runtime_ms, mk, color=col, ms=8, markeredgecolor="white",
                markeredgewidth=.7, label=name)
        a2.plot(d.C, d.runtime_ms, "-", color=col, lw=1.5, alpha=.85)
        a2.plot(d.C, d.runtime_ms, mk, color=col, ms=8, markeredgecolor="white",
                markeredgewidth=.7, label=name)
        for l, (_, r) in zip(lab, d.iterrows()):
            a1.annotate(l, (r.N, r.runtime_ms), fontsize=7.5, color=col,
                        xytext=(4, 3), textcoords="offset points")
            a2.annotate(l, (r.C, r.runtime_ms), fontsize=7.5, color=col,
                        xytext=(4, 3), textcoords="offset points")
    a1.set_xscale("log"); a1.set_xlabel("Permutation count  N  (log)")
    a1.set_title("(a) Euler trade-off (TOTAL runtime = split + max-leaf)")
    a2.axvline(S, color="gray", ls="--", lw=1.2, label=f"$C=S={S:.0f}$ (optimal)")
    a2.set_xlabel("Cycle length  C")
    a2.set_title("(b) Runtime vs cycle length\n(strong engines drop straight down at $C=S$)")
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.set_ylabel("TOTAL compute time [ms] (split + max-leaf, log)")
        ax.grid(True, which="both", ls=":", lw=.4, alpha=.6)
        ax.legend(title="Leaf engine", fontsize=9, framealpha=.95)
    fig.suptitle(f"Euler splitting — TM workload ({regime}, n=256, S={S:.0f})",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "tm_euler_tradeoff")


# =====================================================================
# FIG 3 — design space (N, C, T_compute) projections
# =====================================================================
def fig_design(regime):
    proj = [("N (configs)", "C (cycle)", "N", "C", "log", None),
            ("N (configs)", "$T_{compute}$ (ms, total)", "N", "runtime_ms", "log", "log"),
            ("C (cycle)", "$T_{compute}$ (ms, total)", "C", "runtime_ms", None, "log")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (xl, yl, xk, yk, xs, ys) in zip(axes, proj):
        for name, col, mk in ENG:
            r = sub(regime, name).reindex(RADIX_ORDER).dropna()
            e = sub(regime, name).reindex(EULER_ORDER).dropna()
            ax.plot(r[xk], r[yk], "-o", color=col, lw=1.4, ms=6,
                    markeredgecolor="white", markeredgewidth=.6)
            ax.plot(e[xk], e[yk], "--^", color=col, lw=1.4, ms=6,
                    markeredgecolor="white", markeredgewidth=.6)
            ax.scatter([r.loc["BvN", xk]], [r.loc["BvN", yk]], color=col, marker="*",
                       s=240, edgecolor="k", lw=.6, zorder=6)
        if xs: ax.set_xscale(xs)
        if ys: ax.set_yscale(ys)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(True, which="both", ls=":", lw=.4, alpha=.6)
    leg = [Line2D([], [], color=c, marker="*", ls="", ms=12, label=n) for n, c, _ in ENG]
    leg += [Line2D([], [], color="gray", ls="-", marker="o", label="Radix (B16→B2)"),
            Line2D([], [], color="gray", ls="--", marker="^", label="Euler (d1→d4)"),
            Line2D([], [], color="gray", marker="*", ls="", ms=12, label="BvN baseline")]
    axes[0].legend(handles=leg, fontsize=8, loc="upper left", framealpha=.95)
    fig.suptitle(f"Design space $(N, C, T_{{compute}})$ — TM workload ({regime}, n=256)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "tm_design_space")


# =====================================================================
# DCT helpers
# =====================================================================
def configs_of(regime):
    """label -> (N, C, T_compute_ms) for every engine x framework."""
    cfg = {}
    for name, _, _ in ENG:
        d = sub(regime, name)
        for c in RADIX_ORDER + EULER_ORDER[1:]:
            if c in d.index:
                r = d.loc[c]
                cfg[f"{name} {c}"] = (r.N, r.C, r.runtime_ms)
    return cfg


CMAP = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
        "#D55E00", "#CC79A7", "#999999", "#000000", "#882255"]


# =====================================================================
# FIG 4 — winner map (T_config x ratio r)
# =====================================================================
def fig_winner(regime):
    cfg = configs_of(regime)
    labels = list(cfg); NCT = np.array([cfg[l] for l in labels])
    tcfg = np.logspace(-3, 0, 240)
    rg = np.logspace(np.log10(5), 3, 240)
    TC, R = np.meshgrid(tcfg, rg)
    allD = np.stack([NCT[i, 2] + TC * (NCT[i, 0] + NCT[i, 1] * R) for i in range(len(labels))])
    win = allD.argmin(axis=0)
    ws = sorted(set(win.ravel().tolist()))
    remap = {w: i for i, w in enumerate(ws)}
    print(f"  [winner map {regime}] " +
          ", ".join(f"{labels[w]} {(win==w).mean()*100:.0f}%" for w in ws))
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.pcolormesh(TC, R, np.vectorize(remap.get)(win),
                  cmap=ListedColormap(CMAP[:len(ws)]),
                  norm=BoundaryNorm(np.arange(-.5, len(ws)+.5), len(ws)), shading="auto")
    ax.axhline(10, color="k", ls="--", lw=1.3, label="$r=10$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(TFMT)
    ax.set_xlabel("$T_{config}$"); ax.set_ylabel("ratio  $r = T_{unit}/T_{config}$")
    ax.legend(handles=[Patch(color=CMAP[remap[w]], label=labels[w]) for w in ws] +
              [Line2D([], [], color="k", ls="--", lw=1.3, label="$r=10$")],
              title="Winning configuration", fontsize=8.5, title_fontsize=9,
              loc="upper left", framealpha=.95)
    ax.set_title(f"DCT winner map — TM workload ({regime})\n"
                 f"$T_{{config}}$: 1$\\mu$s–1ms, $r\\geq5$, n=256, S={S_of(regime):.0f}",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "tm_winner_map")


# =====================================================================
# FIG 5 — sensitivity to the computation budget
# =====================================================================
def fig_sensitivity(regime):
    cfg = configs_of(regime)
    labels = list(cfg); NCT = np.array([cfg[l] for l in labels])
    tcfg = np.logspace(-3, 0, 220)
    rg = np.logspace(np.log10(5), 3, 220)
    TC, R = np.meshgrid(tcfg, rg)
    SC = [(1.0, r"$T_{compute}$ (full)"), (.25, r"$T_{compute}/4$"),
          (.125, r"$T_{compute}/8$"), (1/16, r"$T_{compute}/16$"),
          (1/32, r"$T_{compute}/32$"), (1/64, r"$T_{compute}/64$"),
          (1/128, r"$T_{compute}/128$"), (0.0, r"$T_{compute}=0$ (prescheduled)")]
    wins = [np.stack([NCT[i, 2]*s + TC*(NCT[i, 0] + NCT[i, 1]*R)
                      for i in range(len(labels))]).argmin(axis=0) for s, _ in SC]
    ws = sorted(set(np.concatenate([w.ravel() for w in wins]).tolist()))
    remap = {w: i for i, w in enumerate(ws)}
    bvn_i = labels.index("Hungarian BvN")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    for ax, w, (s, t) in zip(axes.ravel(), wins, SC):
        ax.pcolormesh(TC, R, np.vectorize(remap.get)(w),
                      cmap=ListedColormap(CMAP[:len(ws)]),
                      norm=BoundaryNorm(np.arange(-.5, len(ws)+.5), len(ws)), shading="auto")
        ax.axhline(10, color="k", ls="--", lw=1.1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(TFMT)
        p = (w == bvn_i).mean()*100
        ax.set_title(f"{t}\nHungarian BvN wins {p:.0f}%", fontsize=10.5,
                     color="#0072B2" if p > .5 else "black",
                     fontweight="bold" if p > .5 else "normal")
        print(f"  [{t}] " + ", ".join(f"{labels[x]} {(w==x).mean()*100:.0f}%"
                                      for x in sorted(set(w.ravel().tolist()))))
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("ratio  $r=T_{unit}/T_{config}$")
    for ax in axes[1]:
        ax.set_xlabel("$T_{config}$")
    h = [Patch(color=CMAP[remap[w]], label=labels[w]) for w in ws] + \
        [Line2D([], [], color="k", ls="--", lw=1.1, label="$r=10$")]
    fig.legend(handles=h, title="Winning configuration", ncol=min(len(h), 5),
               loc="lower center", fontsize=9, title_fontsize=10,
               framealpha=.95, bbox_to_anchor=(.5, -.02))
    fig.suptitle("Winner map vs the computation-time budget — TM workload "
                 f"({regime}, n=256)\n$T_{{config}}$: 1$\\mu$s–1ms, $r\\geq5$",
                 fontsize=13, y=1.005)
    fig.tight_layout(rect=[0, .05, 1, .99])
    save(fig, "tm_tcompute_sensitivity")


# =====================================================================
# FIG 6 — TM vs Uniform head-to-head
# =====================================================================
def fig_compare(a=PRIMARY, b=BASELINE):
    """Agreement plot: uniform baseline on x, realistic TM on y, with the y=x
    diagonal.  Points on the diagonal mean the realistic workload reproduces the
    synthetic conclusion for that (engine, configuration)."""
    order = RADIX_ORDER + EULER_ORDER[1:]
    fmk = {"BvN": "*", "B16": "o", "B8": "o", "B4": "o", "B2": "o",
           "Euler d1": "^", "Euler d2": "^", "Euler d3": "^", "Euler d4": "^"}
    Sa, Sb = S_of(a), S_of(b)
    metrics = [("runtime_ms", "$T_{compute}$ [ms]", True, 1.0),
               ("N", "Permutation count $N$", True, 1.0),
               ("C_over_S", "Cycle ratio  $C/S$", False, 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, (mk_key, lab, logscale, _) in zip(axes, metrics):
        xs, ys = [], []
        for name, col, _ in ENG:
            da, db = sub(a, name), sub(b, name)
            for cfg in order:
                if cfg not in da.index or cfg not in db.index:
                    continue
                xv = db.loc[cfg, "C"] / Sb if mk_key == "C_over_S" else db.loc[cfg, mk_key]
                yv = da.loc[cfg, "C"] / Sa if mk_key == "C_over_S" else da.loc[cfg, mk_key]
                ax.plot(xv, yv, fmk[cfg], color=col, ms=11 if cfg == "BvN" else 7,
                        markeredgecolor="white", markeredgewidth=.6, alpha=.9)
                xs.append(xv); ys.append(yv)
        lo, hi = min(xs + ys), max(xs + ys)
        pad = (hi / lo) ** 0.06 if logscale else (hi - lo) * 0.06
        if logscale:
            lo, hi = lo / pad, hi * pad
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, zorder=0, label="$y=x$ (identical)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Uniform baseline — {lab}")
        ax.set_ylabel(f"TM (realistic) — {lab}")
        ax.set_title(lab, fontsize=11)
        ax.grid(True, which="both", ls=":", lw=.4, alpha=.6)
        ax.legend(fontsize=8, loc="upper left", framealpha=.95)
    h = [Line2D([], [], color=c, marker="s", ls="", label=n) for n, c, _ in ENG]
    h += [Line2D([], [], color="gray", marker="*", ls="", ms=12, label="BvN"),
          Line2D([], [], color="gray", marker="o", ls="", label="Radix"),
          Line2D([], [], color="gray", marker="^", ls="", label="Euler")]
    fig.legend(handles=h, ncol=7, loc="lower center", fontsize=9,
               framealpha=.95, bbox_to_anchor=(.5, -.06))
    fig.suptitle("Does the realistic workload change the conclusions?  "
                 f"TM (S={Sa:.0f}) vs. uniform-random baseline (S={Sb:.0f}), n=256\n"
                 "each point is one engine $\\times$ configuration; "
                 "on the diagonal $=$ same behaviour on both workloads",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "tm_vs_uniform")


# =====================================================================
# LaTeX tables
# =====================================================================
def _fmt(x, d=0):
    return f"{x:,.{d}f}".replace(",", "{,}")


def tab_framework(regime, order, fname, unit_col="Runtime (ms)"):
    """Engine x {BvN + framework configs} table."""
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"\textbf{Matching} & \textbf{Method} & \textbf{" + unit_col +
         r"} & \textbf{Cycle Length} & \textbf{Permutations} \\", r"\midrule"]
    for i, (name, _, _) in enumerate(ENG):
        d = sub(regime, name).reindex(order).dropna()
        best_rt, best_c, best_n = d.runtime_ms.min(), d.C.min(), d.N.min()
        L.append(r"\multirow{%d}{*}{%s}" % (len(d), name))
        for cfg, r in d.iterrows():
            lbl = "BvN baseline" if cfg == "BvN" else (
                f"Radix $B={cfg[1:]}$" if cfg.startswith("B") else
                cfg.replace("Euler d", "depth "))
            rt = _fmt(r.runtime_ms, 1); cc = _fmt(r.C); nn = _fmt(r.N)
            if abs(r.runtime_ms - best_rt) < 1e-9: rt = r"\textbf{%s}" % rt
            if abs(r.C - best_c) < 1e-9: cc = r"\textbf{%s}" % cc
            if abs(r.N - best_n) < 1e-9: nn = r"\textbf{%s}" % nn
            L.append(f"  & {lbl} & {rt} & {cc} & {nn} " + r"\\")
        L.append(r"\midrule" if i < len(ENG)-1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(PAPER, fname), "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {fname}")


def tab_euler_vs_radix(regime, fname="tm_tab_euler_vs_radix.tex"):
    d = sub(regime, "Hungarian")
    S = S_of(regime)
    rows = []
    for dep in DEPTHS:
        e = d.loc[f"Euler d{dep}"]
        rows.append(("Euler depth %d" % dep, e.units, e.unit_ms, e.split_ms,
                     e.runtime_ms, e.C / S, e.N))
        # closest radix by unit count
        cand = [(abs(d.loc[f"B{b}"].units - e.units), b) for b in BASES]
        cand.sort()
        b = cand[0][1]; r = d.loc[f"B{b}"]
        rows.append((f"Radix $B={b}$", r.units, r.unit_ms, 0.0,
                     r.runtime_ms, r.C / S, r.N))
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"\textbf{Config} & \textbf{Units} & \textbf{Max-unit (ms)} & "
         r"\textbf{Split (ms)} & \textbf{Total (ms)} & $C/S$ & \textbf{Perms} \\",
         r"\midrule"]
    for i, (lbl, u, mu, sp, tot, cs, n) in enumerate(rows):
        spx = r"$\sim$0" if sp < 1e-9 else _fmt(sp, 1)
        L.append(f"{lbl} & {u:.0f} & {_fmt(mu,1)} & {spx} & {_fmt(tot,1)} & "
                 f"{cs:.3f} & {_fmt(n)} " + r"\\")
        if i % 2 == 1 and i < len(rows)-1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, fname), "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {fname}")


def tab_compare(a=PRIMARY, b=BASELINE, fname="tm_tab_comparison.tex"):
    L = [r"\begin{tabular}{ll rr rr rr}", r"\toprule",
         r"& & \multicolumn{2}{c}{$T_{compute}$ (ms)} & \multicolumn{2}{c}{$N$}"
         r" & \multicolumn{2}{c}{$C/S$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"\textbf{Engine} & \textbf{Config} & TM & Unif. & TM & Unif. & TM & Unif. \\",
         r"\midrule"]
    key = ["BvN", "B2", "Euler d4"]
    for i, (name, _, _) in enumerate(ENG):
        da, db = sub(a, name), sub(b, name)
        L.append(r"\multirow{3}{*}{%s}" % name)
        for c in key:
            ra, rb = da.loc[c], db.loc[c]
            lbl = c if c != "Euler d4" else "Euler d4"
            L.append(f"  & {lbl} & {_fmt(ra.runtime_ms,1)} & {_fmt(rb.runtime_ms,1)} & "
                     f"{_fmt(ra.N)} & {_fmt(rb.N)} & {ra.C/S_of(a):.3f} & "
                     f"{rb.C/S_of(b):.3f} " + r"\\")
        L.append(r"\midrule" if i < len(ENG)-1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(PAPER, fname), "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {fname}")


if __name__ == "__main__":
    print("regimes:", list(df.regime.unique()))
    fig_radix(PRIMARY)
    fig_euler(PRIMARY)
    fig_design(PRIMARY)
    fig_winner(PRIMARY)
    fig_sensitivity(PRIMARY)
    fig_compare()
    tab_framework(PRIMARY, RADIX_ORDER, "tm_tab_radix.tex")
    tab_framework(PRIMARY, EULER_ORDER, "tm_tab_euler.tex", unit_col="Total (ms)")
    tab_euler_vs_radix(PRIMARY)
    tab_compare()
    # console comparison
    print("\n=== TM dense vs Uniform (Hungarian) ===")
    for c in RADIX_ORDER + EULER_ORDER[1:]:
        ra, rb = sub(PRIMARY, "Hungarian").loc[c], sub(BASELINE, "Hungarian").loc[c]
        print(f"  {c:10s} TM: {ra.runtime_ms:7.1f}ms N={ra.N:6.0f} C/S={ra.C/S_of(PRIMARY):.3f}"
              f"   | Unif: {rb.runtime_ms:7.1f}ms N={rb.N:6.0f} C/S={rb.C/S_of(BASELINE):.3f}")
