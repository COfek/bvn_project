"""
Brute-force enumeration of the COMPLETE decomposition space of a small
doubly-stochastic / K-regular integer traffic matrix into switch configurations
(permutation / matching matrices) with positive-integer coefficients.

Goal: ground the thesis claim that the decomposition space contains MANY
distinct points with DIFFERENT DCT and a NON-TRIVIAL (beatable) optimum.

----------------------------------------------------------------------
Model.
A crossbar switch configuration is a MATCHING (a partial permutation: at most
one cell per row and per column). A schedule decomposes the demand matrix D
into  D = sum_k  a_k * P_k  with a_k positive integers and P_k matchings.
  N = number of configurations (= number of reconfigurations / config switches)
  C = sum_k a_k = cycle length = total number of time slots.

KEY FACT (verified below): if we restrict configurations to FULL permutation
matrices, then because each row of a K-regular matrix sums to K, every complete
decomposition has  sum a_k == K  identically -> C is constant == S. The C axis
only becomes non-trivial once partial matchings are allowed (a real switch can
idle ports). Strong (perfect-matching / BvN) decompositions then sit exactly at
C = S; wasteful ones use more, shorter configurations and reach C > S. This is
precisely the N-vs-C tradeoff the thesis studies.

DCT model:  DCT = T_calc + N * T_config + C * T_unit
  T_unit  = 0.01 ms (10 us)
  T_calc  = 0   (comparing schedules of the SAME matrix; offline compute ignored)
  T_config swept on a log grid 1 ns .. 1 ms.

Output: paper/plots_pdf/decomposition_space_demo.pdf
"""

import itertools
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. The matrix: a small K-regular doubly-stochastic integer matrix.
#    D = sum of K=3 permutation matrices: every row and column sums to K=3.
#    Chosen (by search, see scan_matrix_choice.py) so that the space is rich:
#    it admits TWO distinct strong (perfect-matching / 3-edge-coloring)
#    decompositions, and the matching-based configuration space is large but
#    fully brute-force enumerable, with a real N-vs-C tradeoff.
# ------------------------------------------------------------------
D = np.array([[1, 1, 0, 1],
              [1, 1, 1, 0],
              [1, 1, 1, 0],
              [0, 0, 1, 2]], dtype=int)
n = D.shape[0]
row_sums = D.sum(1)
col_sums = D.sum(0)
S = int(max(row_sums.max(), col_sums.max()))   # max line sum
K = S                                           # K-regular: all line sums == K

# Confirm K-regularity and that D is a sum of K permutation matrices.
assert (row_sums == K).all() and (col_sums == K).all(), "not K-regular"

print("=" * 72)
print(f"Matrix D  (n={n}, K={K}); every row & column sums to K={K}:")
print(D)
print(f"S (max line sum) = {S};  K-regular doubly-stochastic integer matrix.")
print("=" * 72)


# ------------------------------------------------------------------
# Enumerators for matchings and perfect matchings.
# ------------------------------------------------------------------
def all_matchings(M):
    """All NON-EMPTY matchings (partial permutations) using only cells >0.
    Returns list of tuple-sorted edge sets."""
    nn = M.shape[0]
    cells = [(i, j) for i in range(nn) for j in range(nn) if M[i, j] > 0]
    res = []
    def rec(start, usedr, usedc, cur):
        if cur:
            res.append(tuple(cur))
        for k in range(start, len(cells)):
            i, j = cells[k]
            if i not in usedr and j not in usedc:
                usedr.add(i); usedc.add(j); cur.append((i, j))
                rec(k + 1, usedr, usedc, cur)
                cur.pop(); usedr.discard(i); usedc.discard(j)
    rec(0, set(), set(), [])
    return res


def perfect_matchings(M):
    """All perfect matchings (full permutations) in the support of M."""
    nn = M.shape[0]
    out = []
    def rec(row, used, cur):
        if row == nn:
            out.append(tuple(cur)); return
        for c in range(nn):
            if not used[c] and M[row, c] > 0:
                used[c] = True; cur.append((row, c))
                rec(row + 1, used, cur)
                cur.pop(); used[c] = False
    rec(0, [False] * nn, [])
    return out


def mat_of(edges, nn):
    M = np.zeros((nn, nn), dtype=int)
    for i, j in edges:
        M[i, j] = 1
    return M


def min_edge(M, edges):
    return min(M[i, j] for i, j in edges)


# Cross-checks: weighted permanent and number of support perfect matchings.
def permanent_int(M):
    nn = M.shape[0]
    tot = 0
    for p in itertools.permutations(range(nn)):
        prod = 1
        for i in range(nn):
            prod *= M[i, p[i]]
            if prod == 0:
                break
        tot += prod
    return tot


pm_support = len(perfect_matchings(D))
print(f"# perfect matchings in support of D = {pm_support}")
print(f"weighted permanent perm(D)          = {permanent_int(D)}")


# ------------------------------------------------------------------
# 2. BRUTE-FORCE: enumerate ALL distinct complete decompositions of D into
#    configurations (matchings) with positive integer coefficients.
#    Memoized on the residual matrix; final dedup by canonical multiset.
# ------------------------------------------------------------------
def enumerate_decompositions(M, cache):
    key = M.tobytes()
    if key in cache:
        return cache[key]
    if not M.any():
        cache[key] = {frozenset()}
        return cache[key]
    res = set()
    for edges in all_matchings(M):
        Mm = mat_of(edges, M.shape[0])
        m = min_edge(M, edges)
        ekey = tuple(sorted(edges))
        for a in range(1, m + 1):
            for s in enumerate_decompositions(M - a * Mm, cache):
                d = dict(s)
                d[ekey] = d.get(ekey, 0) + a
                res.add(frozenset(d.items()))
    cache[key] = res
    return res


cache = {}
all_decomps = [dict(s) for s in enumerate_decompositions(D, cache)]
total = len(all_decomps)
print("=" * 72)
print(f"TOTAL distinct complete decompositions (matchings allowed) = {total}")
print("=" * 72)

# Verify each reconstructs D.
for d in all_decomps:
    R = sum(c * mat_of(e, n) for e, c in d.items())
    assert np.array_equal(R, D), "bad decomposition"
print("All decompositions verified to reconstruct D.")


# ------------------------------------------------------------------
# 2b. The PERMUTATION-ONLY space (full permutations only): demonstrates C==S.
# ------------------------------------------------------------------
def enumerate_perm_decompositions(M, cache):
    key = M.tobytes()
    if key in cache:
        return cache[key]
    if not M.any():
        cache[key] = {frozenset()}
        return cache[key]
    res = set()
    for edges in perfect_matchings(M):
        Mm = mat_of(edges, M.shape[0])
        m = min_edge(M, edges)
        ekey = tuple(sorted(edges))
        for a in range(1, m + 1):
            for s in enumerate_perm_decompositions(M - a * Mm, cache):
                d = dict(s)
                d[ekey] = d.get(ekey, 0) + a
                res.add(frozenset(d.items()))
    cache[key] = res
    return res


perm_cache = {}
perm_decomps = [dict(s) for s in enumerate_perm_decompositions(D, perm_cache)]
perm_Cs = [sum(d.values()) for d in perm_decomps]
print(f"Permutation-ONLY decompositions: {len(perm_decomps)}; "
      f"all C == S={S}: {all(c == S for c in perm_Cs)} "
      f"(C set = {sorted(set(perm_Cs))})")


# ------------------------------------------------------------------
# 3. Combinatorial cross-check on the STRONG (perfect-matching, coeff-1)
#    decompositions = proper K-edge-colorings of the K-regular bipartite
#    multigraph of D (unordered = / K!).
# ------------------------------------------------------------------
def count_ordered_K_pm_partitions(M, k):
    if k == 0:
        return 1 if not M.any() else 0
    cnt = 0
    for edges in perfect_matchings(M):
        if min_edge(M, edges) >= 1:
            cnt += count_ordered_K_pm_partitions(M - mat_of(edges, M.shape[0]), k - 1)
    return cnt


ordered_colorings = count_ordered_K_pm_partitions(D, K)
unordered_strong_formula = ordered_colorings / math.factorial(K)
print(f"proper {K}-edge-colorings (ordered K-PM partitions) = {ordered_colorings}")
print(f"  => unordered strong decompositions = {ordered_colorings}/{K}! = "
      f"{unordered_strong_formula}")

# Strong decompositions in the FULL enumeration: all coeffs 1, all configs are
# perfect matchings, N == K.
def is_perfect(edges):
    return len(edges) == n


strong_in_enum = [d for d in all_decomps
                  if all(c == 1 for c in d.values())
                  and all(is_perfect(e) for e in d.keys())
                  and len(d) == K]
print(f"strong (coeff-1, perfect, N=K) decompositions in enumeration = "
      f"{len(strong_in_enum)}")
formula_matches = (len(strong_in_enum) == round(unordered_strong_formula)
                   and unordered_strong_formula == int(unordered_strong_formula))
print(f"FORMULA MATCHES brute force on strong decompositions: {formula_matches}")


# ------------------------------------------------------------------
# 4. (N, C) for every decomposition.
# ------------------------------------------------------------------
records = []
for d in all_decomps:
    N = len(d)
    C = sum(d.values())
    strong = (all(c == 1 for c in d.values())
              and all(is_perfect(e) for e in d.keys()) and C == S)
    records.append(dict(N=N, C=C, strong=strong, decomp=d))

Ns = [r["N"] for r in records]
Cs = [r["C"] for r in records]
print("=" * 72)
print(f"N range (# configurations)         : {min(Ns)} .. {max(Ns)}")
print(f"C range (cycle length = sum coeffs): {min(Cs)} .. {max(Cs)}")
print(f"all C >= S={S}: {all(c >= S for c in Cs)}")
strong_Cs = [r['C'] for r in records if r['strong']]
print(f"strong decompositions: count={len(strong_Cs)}, all C==S={S}: "
      f"{all(c == S for c in strong_Cs)}")


# ------------------------------------------------------------------
# 5. DCT model & sweep.
# ------------------------------------------------------------------
T_UNIT_MS = 0.01            # 10 us
T_CALC_MS = 0.0
t_grid = np.logspace(-6, 0, 600)    # 1 ns .. 1 ms (in ms)


def dct(N, C, t):
    return T_CALC_MS + N * t + C * T_UNIT_MS


# (a) many distinct DCT at a representative T_config.
T_REP = 1e-3                # 1 us
dct_rep = np.array([dct(r["N"], r["C"], T_REP) for r in records])
distinct_dct_rep = len(set(np.round(dct_rep, 12)))
print("=" * 72)
print(f"At T_config = {T_REP} ms ({T_REP*1e6:.0f} ns): {distinct_dct_rep} "
      f"distinct DCT values among {total} decompositions.")


# Reference picks.
# PLAIN STRONG BvN: a strong perfect-matching decomposition, C = S, N = K.
strong_recs = [r for r in records if r["strong"]]
bvn_pick = min(strong_recs, key=lambda r: (r["C"], r["N"]))

# RADIX BASE-2: greedy, each step take a matching and subtract the largest
# power-of-two coefficient that fits its min support entry. Deterministic
# (lexicographically smallest perfect matching preferred, else smallest
# matching) so it is a single well-defined schedule inside our space.
def radix_base2_decomp(M):
    decomp = {}
    R = M.copy()
    while R.any():
        pms = perfect_matchings(R)
        if pms:
            edges = sorted(pms)[0]
        else:
            edges = sorted(all_matchings(R))[0]
        m = min_edge(R, edges)
        a = 1 << (int(m).bit_length() - 1)     # largest power of two <= m
        R = R - a * mat_of(edges, M.shape[0])
        ekey = tuple(sorted(edges))
        decomp[ekey] = decomp.get(ekey, 0) + a
    return decomp


radix_decomp = radix_base2_decomp(D)
assert np.array_equal(sum(c * mat_of(e, n) for e, c in radix_decomp.items()), D)
radix_N = len(radix_decomp)
radix_C = sum(radix_decomp.values())


def find_record(decomp):
    target = frozenset(decomp.items())
    for r in records:
        if frozenset(r["decomp"].items()) == target:
            return r
    return None


radix_rec = find_record(radix_decomp)
print(f"Plain strong BvN pick : N={bvn_pick['N']} C={bvn_pick['C']}")
print(f"Radix base-2 pick     : N={radix_N} C={radix_C} "
      f"(in enumeration: {radix_rec is not None})")


# (b) argmin changes as T_config grows.
argmin_by_t = []
for t in t_grid:
    vals = np.array([dct(r["N"], r["C"], t) for r in records])
    argmin_by_t.append(int(vals.argmin()))

seen = []
for idx in argmin_by_t:
    if idx not in seen:
        seen.append(idx)

print("=" * 72)
print(f"# DISTINCT argmin decompositions across the T_config sweep: {len(seen)}")
for idx in seen:
    r = records[idx]
    ts = [t for t, a in zip(t_grid, argmin_by_t) if a == idx]
    tag = []
    if idx == records.index(bvn_pick):
        tag.append("=BvN")
    if radix_rec is not None and idx == records.index(radix_rec):
        tag.append("=radix")
    print(f"  opt N={r['N']} C={r['C']} strong={r['strong']:d} {' '.join(tag):8s} "
          f"for T_config in [{min(ts):.3e}, {max(ts):.3e}] ms")

optimum_changes = len(seen) > 1

# (c) interior winner?
bvn_idx = records.index(bvn_pick)
radix_idx = records.index(radix_rec) if radix_rec is not None else -1
interior_ranges = []
for idx in seen:
    if idx != bvn_idx and idx != radix_idx:
        ts = [t for t, a in zip(t_grid, argmin_by_t) if a == idx]
        interior_ranges.append((records[idx], min(ts), max(ts)))
interior_wins = len(interior_ranges) > 0
print(f"INTERIOR (non-BvN, non-radix) decomposition wins for some T_config: "
      f"{interior_wins}")
for r, lo, hi in interior_ranges:
    print(f"  interior winner N={r['N']} C={r['C']} on [{lo:.3e}, {hi:.3e}] ms")

minN_rec = min(records, key=lambda r: (r["N"], r["C"]))
print(f"Min-N extreme (fewest configs): N={minN_rec['N']} C={minN_rec['C']}")


# ------------------------------------------------------------------
# 6. Figure: (N, C) scatter colored by DCT at T_REP, special points marked.
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "mathtext.fontset": "cm",
})

fig, ax = plt.subplots(figsize=(7.4, 5.6))

Narr = np.array(Ns, float)
Carr = np.array(Cs, float)
# jitter so coincident (N,C) points with different decomps are visible
rs = np.random.default_rng(0)
jx = rs.uniform(-0.13, 0.13, size=len(Narr))
jy = rs.uniform(-0.13, 0.13, size=len(Carr))

sc = ax.scatter(Narr + jx, Carr + jy, c=dct_rep, cmap="viridis", s=58,
                edgecolor="k", linewidth=0.35, zorder=3, alpha=0.9)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(f"DCT at $T_{{config}}={T_REP*1e6:.0f}$ ns  (ms)")


def mark(rec, marker, color, label):
    ax.scatter([rec["N"]], [rec["C"]], marker=marker, s=300,
               facecolor="none", edgecolor=color, linewidth=2.6,
               zorder=6, label=label)


mark(bvn_pick, "s", "#D55E00", f"Plain strong BvN  (N={bvn_pick['N']}, C={bvn_pick['C']})")
if radix_rec is not None:
    mark(radix_rec, "D", "#0072B2", f"Radix base-2  (N={radix_N}, C={radix_C})")

# Sweep optima
opt_colors = ["#CC79A7", "#009E73", "#E69F00", "#56B4E9"]
for k, idx in enumerate(seen):
    r = records[idx]
    if idx in (bvn_idx, radix_idx):
        continue
    ax.scatter([r["N"]], [r["C"]], marker="*", s=420, facecolor="none",
               edgecolor=opt_colors[k % len(opt_colors)], linewidth=2.6,
               zorder=7, label=f"Sweep-opt  (N={r['N']}, C={r['C']})")

ax.axhline(S, color="grey", ls=":", lw=1.0, zorder=1)
ax.text(max(Ns), S, f" $C=S={S}$ (strong / BvN)", va="bottom", ha="right",
        color="grey", fontsize=9)

ax.set_xlabel("$N$  (number of configurations / reconfigurations)")
ax.set_ylabel("$C$  (cycle length $=\\sum$ coefficients $=$ time slots)")
ax.set_title(
    f"Decomposition space of a {n}$\\times${n} $K{{=}}{K}$ doubly-stochastic matrix\n"
    f"{total} distinct decompositions   |   "
    f"$T_{{unit}}{{=}}{T_UNIT_MS*1000:.0f}\\,\\mu$s,  "
    f"$T_{{config}}{{=}}{T_REP*1e6:.0f}$ ns")

handles, labels = ax.get_legend_handles_labels()
uniq = {}
for h, l in zip(handles, labels):
    uniq.setdefault(l, h)
ax.legend(uniq.values(), uniq.keys(), loc="upper left", framealpha=0.95,
          fontsize=9)
ax.grid(True, alpha=0.25, zorder=0)
ax.set_xticks(range(min(Ns), max(Ns) + 1))
ax.set_yticks(range(min(Cs), max(Cs) + 1))
fig.tight_layout()

out = Path("C:/Users/ofekc/Desktop/Msc/Thesis/paper/plots_pdf/decomposition_space_demo.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("=" * 72)
print(f"Saved figure to {out}")


# ------------------------------------------------------------------
# Compact summary
# ------------------------------------------------------------------
print("\n=== SUMMARY ===")
print(f"matrix n={n} K={K} entries={D.tolist()}")
print(f"total_decompositions={total}")
print(f"perm_only_decomps={len(perm_decomps)} (all C==S={S})")
print(f"ordered_K_colorings={ordered_colorings} unordered_strong={unordered_strong_formula} "
      f"strong_in_enum={len(strong_in_enum)} formula_matches={formula_matches}")
print(f"N_range={min(Ns)}..{max(Ns)} C_range={min(Cs)}..{max(Cs)}")
print(f"distinct_DCT_at_T_REP={distinct_dct_rep}")
print(f"optimum_changes={optimum_changes} n_distinct_optima={len(seen)}")
print(f"interior_wins={interior_wins}")
print(f"bvn N={bvn_pick['N']} C={bvn_pick['C']} | radix N={radix_N} C={radix_C} "
      f"| minN N={minN_rec['N']} C={minN_rec['C']}")
