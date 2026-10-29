"""Option-1 experiment: run the existing engines + Euler framework on REALISTIC
(MoE-style, non-regular) demand matrices, and compare against the regular baseline.

Motivation (Birkhoff-MoE, HotOptics'26): real MoE all-to-all matrices are sparse,
skewed and NOT doubly stochastic. Our framework assumes K-regular doubly-stochastic
input. This script quantifies what actually happens when we feed the engines a
non-regular matrix:
  - the matching engines still produce a valid (lossless) decomposition, but WEAK: C > S*
  - Euler splitting's (S/2)-regular guarantee BREAKS (halves are no longer regular),
    because an Euler tour needs every vertex to have even degree.

Outputs a comparison table to stdout and a figure to paper/plots_pdf/moe_workload.pdf.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, r"C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms.bvn import bvn_decomposition
from src.algorithms.euler_splitting import decompose_euler_framework, euler_split_once
from src.utils.matrix_generator import generate_matrix

OUT = r"C:\Users\ofekc\Desktop\Msc\Thesis\paper\plots_pdf"
N = 64
ENGINES = [("Hungarian", "maximum"), ("WFA", "wfa"),
           ("GW Dynamic", "heavy"), ("GW Static", "heavy_static")]


def moe_matrix(n, tokens_per_rank, skew, rng):
    """MoE-style all-to-all: each source rank sends `tokens_per_rank` tokens to
    destination ranks drawn from a skewed (power-law) expert-popularity law.
    Row sums are equal (= tokens_per_rank); column sums are skewed -> the matrix
    is NON-doubly-stochastic, like a real MoE dispatch matrix."""
    pop = 1.0 / (np.arange(1, n + 1) ** skew)      # power-law popularity
    pop = pop[rng.permutation(n)]                   # randomise which ranks are hot
    pop /= pop.sum()
    M = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        dests = rng.choice(n, size=tokens_per_rank, p=pop)
        idx, cnt = np.unique(dests, return_counts=True)
        M[i, idx] += cnt
    return M.astype(np.float64)


def props(M):
    rs, cs = M.sum(1), M.sum(0)
    S_star = int(max(rs.max(), cs.max()))          # conservation lower bound on C
    ds = np.allclose(rs, rs[0]) and np.allclose(cs, cs[0])
    return dict(S_star=S_star, row_lo=int(rs.min()), row_hi=int(rs.max()),
                col_lo=int(cs.min()), col_hi=int(cs.max()),
                nnz=float((M > 0).mean()), doubly_stochastic=ds)


def run_engine(M, algo):
    t0 = time.perf_counter()
    comps = bvn_decomposition(M.copy(), matching_algorithm=algo)
    dt = (time.perf_counter() - t0) * 1000
    N_ = len(comps)
    C = sum(c.weight for c in comps)
    # reconstruction check
    R = np.zeros_like(M)
    for c in comps:
        R += c.weight * c.permutation
    recon_err = float(np.abs(R - M).max())
    return N_, float(C), dt, recon_err


def euler_regularity(M, split_method="heuristic"):
    """Split once and report whether the two halves are (S/2)-regular."""
    red, blue = euler_split_once(M, split_method=split_method)
    lossless = np.array_equal(red + blue, np.round(M).astype(np.int64))
    def spread(H):
        rs, cs = H.sum(1), H.sum(0)
        return int(rs.max() - rs.min()), int(cs.max() - cs.min())
    return lossless, spread(red), spread(blue)


def main():
    rng = np.random.default_rng(0)

    # --- workloads ---
    # Regular baseline (thesis generator): sum of k weighted permutations, exactly DS.
    reg = generate_matrix(N, k=N, max_weight=16, rng=rng)     # S ~ 64*8 = 512
    # MoE-style: equal row sums, skewed columns -> non-DS. Match scale (~512).
    moe = moe_matrix(N, tokens_per_rank=512, skew=1.1, rng=rng)

    workloads = [("Regular (baseline)", reg), ("MoE-style (skewed)", moe)]

    for name, M in workloads:
        p = props(M)
        print(f"\n=== {name} (n={N}) ===")
        print(f"  doubly-stochastic: {p['doubly_stochastic']}   "
              f"row-sum [{p['row_lo']},{p['row_hi']}]   col-sum [{p['col_lo']},{p['col_hi']}]   "
              f"S*={p['S_star']}   density={p['nnz']*100:.0f}%")
        print(f"  {'engine':12s} {'N':>7s} {'C':>8s} {'C/S*':>6s} {'runtime(ms)':>12s} {'recon_err':>10s}")
        for eng_name, algo in ENGINES:
            N_, C, dt, err = run_engine(M, algo)
            print(f"  {eng_name:12s} {N_:7d} {C:8.0f} {C/p['S_star']:6.2f} {dt:12.1f} {err:10.2e}")

        # Euler regularity test
        lossless, red_sp, blue_sp = euler_regularity(M, "heuristic")
        print(f"  Euler split (heuristic): lossless={lossless}  "
              f"red row/col-sum spread={red_sp}  blue row/col-sum spread={blue_sp}")
        if red_sp == (0, 0) and blue_sp == (0, 0):
            print(f"    -> halves ARE (S/2)-regular: Euler guarantee HOLDS.")
        else:
            print(f"    -> halves are NOT regular (spread != 0): Euler guarantee BREAKS.")
        # Euler-framework depth-2 decomposition cycle vs S*
        comps, split_t, max_leaf, nleaves = decompose_euler_framework(
            M, matching_method="maximum", depth=2, split_method="heuristic")
        C_e = sum(c.weight for c in comps)
        print(f"  Euler d2 (Hungarian leaves): N={len(comps)}  C={C_e:.0f}  "
              f"C/S*={C_e/p['S_star']:.2f}  leaves={nleaves}")

    # --- figure: visualise the two matrices + column-sum skew ---
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for col, (name, M) in enumerate(workloads):
        im = ax[0, col].imshow(M, cmap="viridis")
        ax[0, col].set_title(f"{name}\ndemand matrix (n={N})", fontsize=11)
        ax[0, col].set_xlabel("destination rank"); ax[0, col].set_ylabel("source rank")
        fig.colorbar(im, ax=ax[0, col], fraction=0.046)
        cs = M.sum(0); rs = M.sum(1)
        ax[1, col].bar(np.arange(N), np.sort(cs)[::-1], color="#0072B2", label="col sums (recv)")
        ax[1, col].axhline(rs.mean(), color="#D55E00", ls="--", lw=1.5, label="row sum (send)")
        ax[1, col].set_title("port load (sorted)", fontsize=11)
        ax[1, col].set_xlabel("rank (sorted by load)"); ax[1, col].set_ylabel("total demand")
        ax[1, col].legend(fontsize=9)
    fig.suptitle("Regular baseline vs. MoE-style demand: the MoE matrix has balanced sends "
                 "but skewed receives\n(non-doubly-stochastic -> Euler's regularity assumption "
                 "does not hold)", fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/moe_workload.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/moe_workload.png", dpi=140, bbox_inches="tight")
    print(f"\nsaved figure -> {OUT}/moe_workload.pdf")


if __name__ == "__main__":
    main()
