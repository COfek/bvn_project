"""Combined three-way trade-off figure: all engines x all bases on one log-log plot.

x = cycle length (log), y = runtime (log), bubble size = permutation count,
colour = matching engine, marker shape = decomposition (BvN / Radix base).

Reusable: point SOURCES at either the canonical final_runs folders or the
fresh rerun_2026 folders. Regenerate after the rerun for clean Hungarian data.
"""
from __future__ import annotations
import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project"

# name -> (results_stats.csv path, column-prefix, colour)
def canonical():
    base = f"{ROOT}/final_runs"
    return {
        "Hungarian":  (f"{base}/20260522_155907/results_stats.csv", "maximum",      "#4C72B0"),
        "GW Dynamic": (f"{base}/20260522_131035/results_stats.csv", "heavy",        "#55A868"),
        "GW Static":  (f"{base}/20260522_121704/results_stats.csv", "heavy_static", "#C44E52"),
        "WFA":        (f"{base}/20260522_144633/results_stats.csv", "wfa",          "#8172B3"),
    }

def rerun():
    import glob
    base = f"{ROOT}/run/rerun_2026"
    def latest(eng):
        hits = sorted(glob.glob(f"{base}/{eng}/*/results_stats.csv"))
        return hits[-1] if hits else None
    return {
        "Hungarian":  (latest("max"),          "maximum",      "#4C72B0"),
        "GW Dynamic": (latest("heavy"),         "heavy",        "#55A868"),
        "GW Static":  (latest("heavy_static"),  "heavy_static", "#C44E52"),
        "WFA":        (latest("wfa"),            "wfa",          "#8172B3"),
    }

BASES = ["2", "4", "8", "16"]
SHAPE = {"BvN": "o", "16": "s", "8": "^", "4": "D", "2": "v"}

def collect(csv, pre):
    df = pd.read_csv(csv).iloc[5:]  # drop JIT warm-up
    rows = [("BvN", df["bvn_cycle"].mean(), df["bvn_runtime"].mean(), df["bvn_perms"].mean())]
    for b in BASES:
        rows.append((b, df[f"{pre}_{b}_cycle"].mean(),
                        df[f"{pre}_{b}_time"].mean(),
                        df[f"{pre}_{b}_perms"].mean()))
    return rows

def main(use_rerun: bool):
    sources = rerun() if use_rerun else canonical()
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for name, (csv, pre, col) in sources.items():
        if csv is None:
            print(f"skip {name}: no csv"); continue
        rows = collect(csv, pre)
        ax.plot([r[1] for r in rows], [r[2] for r in rows], "-", color=col, alpha=0.35, zorder=1)
        for lbl, c, r, p in rows:
            ax.scatter(c, r, s=40 + p / 12, marker=SHAPE[lbl], color=col,
                       edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Decomposition cycle length  (log scale, lower is better)")
    ax.set_ylabel("Parallel runtime, seconds  (log scale, lower is better)")
    ax.set_title("Three-way trade-off across all engines and decomposition strategies\n"
                 "(N=256, k=256; bubble area $\\propto$ permutation count)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    engine_leg = [Line2D([], [], marker="o", color=col, ls="-", label=n, markeredgecolor="k")
                  for n, (_, _, col) in sources.items()]
    shape_leg = [Line2D([], [], marker=SHAPE[k], color="gray", ls="", markeredgecolor="k",
                        label=("BvN baseline" if k == "BvN" else f"Radix B={k}"))
                 for k in ["BvN", "16", "8", "4", "2"]]
    size_leg = [Line2D([], [], marker="o", color="lightgray", ls="", markeredgecolor="k",
                       markersize=np.sqrt(40 + p / 12) / 1.3, label=f"{p:,} perms")
                for p in (500, 2000, 8000)]
    l1 = ax.legend(handles=engine_leg, title="Matching engine", loc="upper right", fontsize=9, frameon=True)
    l2 = ax.legend(handles=shape_leg, title="Decomposition", loc="lower left", fontsize=8, frameon=True)
    ax.add_artist(l1)
    ax.add_artist(ax.legend(handles=size_leg, title="Bubble size", loc="upper left", fontsize=8,
                            frameon=True, labelspacing=1.4, borderpad=1.0))
    ax.add_artist(l2)

    plt.tight_layout()
    tag = "rerun" if use_rerun else "canonical"
    out = f"{ROOT}/../paper/plots_pdf/combined_tradeoff_{tag}"
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print("saved", out + ".pdf")

if __name__ == "__main__":
    main(use_rerun="--rerun" in sys.argv)
