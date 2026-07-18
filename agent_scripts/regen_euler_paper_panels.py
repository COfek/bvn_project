"""Regenerate the paper's Euler figure panels with END-TO-END runtime
(total = split + max-leaf extraction) instead of max-leaf only.

Overwrites, per engine folder (paper reads these paths directly):
  plots_pdf/euler_<eng>_k256/runtime_comparison.pdf
  plots_pdf/euler_<eng>_k256/runtime_vs_permutations_euler_bvn.pdf
  plots_pdf/euler_<eng>_k16/runtime_comparison.pdf
"""
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8})
OUT = "C:/Users/ofekc/Desktop/Msc/Thesis/paper/plots_pdf"
# engine -> (data prefix, plots_pdf folder token, display name)
ENG = [("maximum", "maximum", "Hungarian"), ("wfa", "wfa", "WFA"),
       ("heavy", "heavy", "GW Dynamic"), ("heavy_static", "heavystatic", "GW Static")]
COLS = {"BvN": "#000000", "depth 1": "#0072B2", "depth 2": "#D55E00", "depth 3": "#009E73"}

def load(pre, scen):
    path = sorted(glob.glob(f"euler_exp/eng_{pre}_{scen}/*/results_stats.csv"))[-1]
    return pd.read_csv(path).iloc[5:].reset_index(drop=True)

def totals(df):
    """Per-sample end-to-end runtime (ms) per config."""
    out = {"BvN": df["bvn_runtime"] * 1000}
    for d in [1, 2, 3]:
        out[f"depth {d}"] = (df[f"euler_heuristic_depth{d}_time"]
                             + df[f"euler_heuristic_depth{d}_split"]) * 1000
    return out

def perms(df):
    out = {"BvN": df["bvn_perms"]}
    for d in [1, 2, 3]:
        out[f"depth {d}"] = df[f"euler_heuristic_depth{d}_perms"]
    return out

for pre, tok, name in ENG:
    for scen, k in [("dense", "k256"), ("sparse", "k16")]:
        df = load(pre, scen)
        tot = totals(df)
        # ---- runtime_comparison: smoothed per-matrix end-to-end runtime ----
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for lbl, series in tot.items():
            ax.plot(series.rolling(9, min_periods=1).mean(), color=COLS[lbl], lw=1.6, label=lbl)
        ax.set_yscale("log")
        ax.set_xlabel("Matrix index")
        ax.set_ylabel("End-to-end runtime [ms]  (split + max-leaf, log)")
        ax.set_title(f"{name} — end-to-end runtime, {scen} (n=256)", fontsize=11)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        ax.legend(fontsize=9, framealpha=0.95)
        fig.tight_layout()
        fig.savefig(f"{OUT}/euler_{tok}_{k}/runtime_comparison.pdf", bbox_inches="tight")
        plt.close(fig)
        # ---- runtime_vs_permutations (dense only, used in paper) ----
        if scen == "dense":
            pm = perms(df)
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            xs = [pm[l].mean() for l in tot]
            ys = [tot[l].mean() for l in tot]
            ax.plot(xs, ys, "-", color="gray", lw=1.0, alpha=0.6, zorder=1)
            for lbl in tot:
                ax.errorbar(pm[lbl].mean(), tot[lbl].mean(),
                            xerr=pm[lbl].std(), yerr=tot[lbl].std(),
                            fmt="o", ms=8, color=COLS[lbl], capsize=3,
                            markeredgecolor="white", markeredgewidth=0.7, label=lbl, zorder=3)
                ax.annotate(lbl, (pm[lbl].mean(), tot[lbl].mean()), fontsize=8,
                            xytext=(6, 4), textcoords="offset points", color=COLS[lbl])
            ax.set_yscale("log")
            ax.set_xlabel("Permutation count  N")
            ax.set_ylabel("End-to-end runtime [ms]  (split + max-leaf, log)")
            ax.set_title(f"{name} — runtime vs. permutations, dense (n=256, k=256)", fontsize=11)
            ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
            fig.tight_layout()
            fig.savefig(f"{OUT}/euler_{tok}_{k}/runtime_vs_permutations_euler_bvn.pdf", bbox_inches="tight")
            plt.close(fig)
        print(f"regenerated euler_{tok}_{k}")
