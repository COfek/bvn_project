"""
DCT vs T_config plot  (log-scale x-axis, realistic T_unit).

For each (matching, framework) combination we use mean values from the paper
experiments (dense scenario: k=256, n=256, max_weight=64, ~1000 matrices):

  DCT = T_calc + N * T_config + C * T_unit

T_unit = 0.01 ms (10 us) — representative of a 100 Gbps link with 1 Mbit
demand unit, and consistent with optical-circuit-switch literature.

T_config is swept on a log scale from 1 ns to 1 ms, covering the full range
of practical optical switch reconfiguration hardware.

Key result (crossover analysis):
  - WFA, GW Dynamic, GW Static: B=2 ALWAYS better (lower intercept AND slope)
  - Hungarian: B=2 better for T_config < 2.96 ms  (off right edge of plot)
  => In the entire realistic hardware range, all Radix B=2 variants
     outperform their BvN counterparts.

Output: saved to the paper plots directory as dct_vs_tconfig.png
"""

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ------------------------------------------------------------------
# FRESH means computed from the data (dense k=256).
#   BvN + Radix B=2: run/rerun_2026 (1000 samples, Hungarian bug fixed).
#   Euler: euler_exp/eng_*_dense_d4 with TOTAL T_calc = max-leaf + split.
# ------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
def _rerun(e): return sorted((_ROOT / "run/rerun_2026" / e).glob("*/results_stats.csv"))[-1]
def _euler(p): return sorted((_ROOT / "euler_exp" / f"eng_{p}_dense_d4").glob("*/results_stats.csv"))[-1]

def _build_configs():
    eng = [("Hungarian", "max", "maximum"), ("WFA", "wfa", "wfa"),
           ("GW Dyn", "heavy", "heavy"), ("GW Stat", "heavy_static", "heavy_static")]
    cfgs = []
    for nm, re, pre in eng:
        dr = pd.read_csv(_rerun(re)).iloc[5:]
        cfgs.append((f"{nm}-BvN", dr["bvn_runtime"].mean()*1000, round(dr["bvn_perms"].mean()), round(dr["bvn_cycle"].mean())))
    for nm, re, pre in eng:
        dr = pd.read_csv(_rerun(re)).iloc[5:]
        cfgs.append((f"{nm}-B2", dr[f"{pre}_2_time"].mean()*1000, round(dr[f"{pre}_2_perms"].mean()), round(dr[f"{pre}_2_cycle"].mean())))
    # Euler representatives: Hungarian d4 and GW Static d3 (TOTAL = time + split)
    for nm, pre, d in [("Hungarian-Euler-d4", "maximum", 4), ("GW Stat-Euler-d3", "heavy_static", 3)]:
        de = pd.read_csv(_euler(pre)).iloc[5:]
        tcalc = (de[f"euler_heuristic_depth{d}_time"].mean() + de[f"euler_heuristic_depth{d}_split"].mean())*1000
        cfgs.append((nm, tcalc, round(de[f"euler_heuristic_depth{d}_perms"].mean()), round(de[f"euler_heuristic_depth{d}_cycle"].mean())))
    return cfgs

# ------------------------------------------------------------------
# Measured mean values from paper (dense scenario k=256)
# T_calc in ms | N = permutation count | C = decomposition cycle length
# ------------------------------------------------------------------
#  label              T_calc_ms    N       C
CONFIGS = _build_configs()

# Colours: BvN baselines solid, Radix-B2 dashed; grouped by matching
STYLE = {
    "Hungarian-BvN":  dict(color="#1f77b4", ls="-",  lw=2.2, label="Hungarian BvN"),
    "WFA-BvN":        dict(color="#ff7f0e", ls="-",  lw=2.2, label="WFA BvN"),
    "GW Dyn-BvN":     dict(color="#2ca02c", ls="-",  lw=2.2, label="GW Dynamic BvN"),
    "GW Stat-BvN":    dict(color="#d62728", ls="-",  lw=2.2, label="GW Static BvN"),
    "Hungarian-B2":   dict(color="#1f77b4", ls="--", lw=1.8, label="Hungarian B=2"),
    "WFA-B2":         dict(color="#ff7f0e", ls="--", lw=1.8, label="WFA B=2"),
    "GW Dyn-B2":      dict(color="#2ca02c", ls="--", lw=1.8, label="GW Dynamic B=2"),
    "GW Stat-B2":     dict(color="#d62728", ls="--", lw=1.8, label="GW Static B=2"),
    "Hungarian-Euler-d4": dict(color="#1f77b4", ls="-.", lw=1.8, label="Hungarian Euler d=4"),
    "GW Stat-Euler-d3":   dict(color="#d62728", ls="-.", lw=1.8, label="GW Static Euler d=3"),
}

T_UNIT_MS = 0.01   # 10 us — realistic optical-switch link rate
# Log sweep: 1 ns (1e-6 ms) to 1 ms
t_cfg = np.logspace(-6, 0, 600)   # ms

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

for label, t_calc, n_perms, cycle in CONFIGS:
    dct = t_calc + n_perms * t_cfg + cycle * T_UNIT_MS
    s = STYLE[label]
    ax.plot(t_cfg, dct, color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])

ax.set_xscale("log")
ax.set_yscale("log")


def _ms_label(val, _pos):
    if val >= 1:
        return f"{val:.0f} ms"
    elif val >= 1e-3:
        return f"{val * 1e3:.0f} μs"
    else:
        return f"{val * 1e6:.0f} ns"


ax.xaxis.set_major_formatter(FuncFormatter(_ms_label))
ax.yaxis.set_major_formatter(FuncFormatter(_ms_label))
ax.set_xlabel(r"$T_{\mathrm{config}}$  [log scale]", fontsize=13)
ax.set_ylabel("DCT  [log scale]", fontsize=13)
ax.set_title(
    r"Demand Completion Time vs.\ Reconfiguration Delay  (log-log scale)"
    "\n"
    r"($n{=}256$, $k{=}256$, $T_{\mathrm{unit}}{=}0.01\,\mathrm{ms}$,"
    r" optical switch: $1\,\mathrm{ns}$--$1\,\mathrm{ms}$)",
    fontsize=11,
)
ax.grid(True, which="both", ls="--", alpha=0.35)

# Legend: two columns, BvN solid / Radix dashed / Euler dash-dot
ax.legend(fontsize=9, ncol=2, loc="upper left",
          title="solid = BvN baseline   dashed = Radix B=2   dash-dot = Euler",
          title_fontsize=8)

ax.set_xlim(1e-6, 1)

plt.tight_layout()

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
out_dir = Path(__file__).parent.parent.parent / "paper" / "plots_pdf"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "dct_vs_tconfig.pdf"
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")

# ------------------------------------------------------------------
# Zoomed companion: same T_config sweep, but LINEAR y-axis capped at
# DCT = 1000 ms. The slow BvN baselines (1200-1900 ms intercepts) leave
# the frame; the competitive bottom cluster (Radix B=2 + Euler) separates
# clearly, making the Euler-vs-B2 ordering and crossovers readable.
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
for label, t_calc, n_perms, cycle in CONFIGS:
    dct = t_calc + n_perms * t_cfg + cycle * T_UNIT_MS
    s = STYLE[label]
    ax.plot(t_cfg, dct, color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(_ms_label))
ax.set_xlabel(r"$T_{\mathrm{config}}$  [log scale]", fontsize=13)
ax.set_ylabel("DCT  [ms, linear]", fontsize=13)
ax.set_title(
    r"Demand Completion Time vs.\ Reconfiguration Delay --- competitive-region zoom"
    "\n"
    r"($n{=}256$, $k{=}256$, $T_{\mathrm{unit}}{=}0.01\,\mathrm{ms}$,"
    r" DCT axis: $0$--$1000\,\mathrm{ms}$, linear)",
    fontsize=11,
)
ax.grid(True, which="both", ls="--", alpha=0.35)
ax.legend(fontsize=9, ncol=2, loc="upper left",
          title="solid = BvN baseline   dashed = Radix B=2   dash-dot = Euler",
          title_fontsize=8)
ax.set_xlim(1e-6, 1)
ax.set_ylim(0, 1000)

plt.tight_layout()
out_path_zoom = out_dir / "dct_vs_tconfig_zoom.pdf"
plt.savefig(out_path_zoom)
plt.close()
print(f"Saved: {out_path_zoom}")
