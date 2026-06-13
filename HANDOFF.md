# BvN-Arbiter — Session Handoff

You're picking up an MSc thesis project mid-stream. This file gets you to
context-parity with the previous Claude Code sessions in one read. **Branch:
`euler_splitting`** (not `master` — all the recent work is here).

---

## 1. What this project is

A Python framework that decomposes **scaled doubly stochastic** demand matrices
(rows and columns all sum to a common scale factor S) into weighted permutation
matrices, for **reconfigurable datacenter network / optical circuit switch
scheduling** research. Each permutation = one switch configuration; its weight =
how long that configuration is held.

It benchmarks four bipartite matching engines across **three** decomposition
frameworks (classical BvN, Radix / digit-plane, and **Euler splitting**),
measuring a three-way tradeoff:

- **Permutation count (N)** — number of switch reconfigurations
- **Cycle length (C = ∑ϕᵢ)** — total transmission duration (optimum is S)
- **Algorithmic runtime (T_calc)** — wall-clock decomposition time

These feed the real cost function the project cares about — **Demand Completion
Time**:

  DCT = T_calc + N·T_config + C·T_unit

where T_config is the hardware reconfiguration delay ("latency tax") and T_unit
is per-unit transmission time. Different (matching, framework) combos win on
different (T_config, T_unit) hardware profiles.

---

## 2. The accompanying paper (NOT in git)

There is an LaTeX thesis paper at `C:\Users\ofekc\Desktop\Msc\Thesis\paper\file.tex`
(sibling of this repo). **It is deliberately NOT version-controlled and lives
only on the original machine** — so a session on another station won't have it.
This handoff captures its state so you have the context regardless.

Paper structure (as of last session): Abstract → §1 Introduction (motivated by
Avin & Schmid, *Revolutionizing Datacenter Networks via Reconfigurable
Topologies*, CACM 2025 — Chen Avin is the advisor) → §2 Background → §3 Strategies
→ §4 DCT → §5 Matrix Generation → §6 Sub-optimal Strategies → §7 Radix Framework
→ §8 **Euler Splitting Framework** (new) → §9 Results (incl. §9.8 Euler all-engines
results, §9.9 Euler-vs-Radix) → §10 Summary (incl. §10.6 DCT vs T_config).

**Paper status:** body is complete and consistent (both frameworks × dense/sparse
× 4 engines, all figures 4-panel, Table-2-consistent). Front matter (abstract +
intro) added. **The one remaining gap: a Conclusion / Limitations / Future Work
section** — the paper currently ends abruptly at the DCT analysis → bibliography.
If asked to "finish the paper," that's the missing piece. Limitations to state:
simulated (not measured) parallelism; Euler runs use 100 samples vs radix's 1000;
single synthetic traffic model; no real optical-switch validation.

---

## 3. Layout

```
bvn_project/
├── main.py                       # entry point (→ cli → runner → plotting + analysis)
├── src/
│   ├── cli.py                    # argparse (all flags incl. --euler-*)
│   ├── config.py                 # ExperimentConfig dataclass
│   ├── runner.py                 # per-matrix worker + experiment driver
│   ├── plotting.py               # matplotlib (_key_label handles euler keys)
│   ├── algorithms/
│   │   ├── bvn.py                # classical BvN
│   │   ├── radix_decomposition.py# digit-plane decomposition
│   │   ├── euler_splitting.py    # EULER FRAMEWORK (the main recent work)
│   │   ├── wfa.py                # Wavefront Arbiter (Numba JIT)
│   │   └── sorted_array_matching.py # heavy / heavy_static (Numba JIT, BFS augment)
│   └── utils/
│       ├── matrix_generator.py   # K-regular generator (sum of k weighted perms)
│       ├── analysis.py           # post-run euler_comparison_summary.txt
│       ├── io_utils.py, stats.py, run_utils.py, logging_utils.py
├── tests/                        # pytest (962 tests incl. paper-scale)
├── agent_scripts/                # GITIGNORED scratch (task1/2/3, DCT plot script)
├── final_runs/                   # canonical Table-2 batch lives here (see §6)
└── .venv/                        # Windows venv (Scripts/python.exe)
```

**Engines:** `wfa`, `maximum` (scipy Hungarian), `heavy` (= GW Dynamic, greedy
re-sorted every iter), `heavy_static` (= GW Static, sort-once). The first three
are "strong" (BFS augmenting paths → perfect matchings → C = S); WFA is "weak"
(maximal-not-perfect → C ≥ S). Also: `minimum` (min-weight Hungarian) and
`*_noaug` (greedy without augmentation) — diagnostics only.

---

## 4. The Euler splitting framework (the core recent work)

`euler_splitting.py`. Splits an S-regular matrix into two (S/2)-regular halves
via an Euler-tour 2-colouring of its bipartite multigraph; depth d → 2^d
doubly-stochastic leaves, each BvN-decomposed independently (parallelism
simulated as max-leaf time). **Every leaf stays doubly stochastic, so C = S
exactly at any depth** for strong engines — the key differentiator vs Radix
(whose digit planes inflate C) and Split-Tree (dropped).

Split methods (`euler_split_once`, selectable via `--euler-split-method`):
- **`heuristic`** (default) — same-direction traversal keeps each entry's copies
  in one half → **sparser leaves** → faster extraction. THE one we use.
  Implemented with O(1)-amortized tier-stack selection (`_euler_split_heuristic_fast_jit`),
  ~30× faster than the original O(n)-scan (`heuristic_scan`, kept as reference).
- `euler` — plain alternating 2-colouring (splits every entry ~50/50, no sparsity).
- `euler_grouped` — O(nnz) bulk version of plain euler.
- `greedy` — whole-block assignment, falls back to euler.

`decompose_euler_framework(matrix, matching_method, depth, split_method)` returns
`(components, split_time, max_leaf_runtime, n_leaves)`. The splits within each
tree level run concurrently on threads (`ThreadPoolExecutor`; the JIT kernels are
nogil), so `split_time` is parallel wall-clock.

**Run it via the pipeline:** `-e euler_bvn` with `--euler-depths`,
`--euler-split-method`, `--euler-leaf-engine`. Under `euler_bvn`, the BvN
baseline, Radix planes, and Euler leaves all use `--euler-leaf-engine` (fair
comparison). Empty `--radix-bases` → no Radix (Euler-only).

---

## 5. Key findings (so the next session doesn't re-derive them)

These came from the professor's three tasks:

1. **Does splitting reduce matching-extraction time?** YES — each depth halves
   max-leaf time (~0.5 per level). First dense split is better-than-half (0.43×)
   because the heuristic also makes leaves sparser. (Paper §9.8, Table 4/5.)
2. **Euler leaves vs Radix digit planes (matched parallelism, Hungarian):**
   Euler wins at low parallelism (≤3 units), Radix at 4-9 units; Euler alone
   past 9. Radix pays 8-54% cycle inflation; Euler holds C=S. (§9.9, Table 7.)
3. **Split bottleneck:** was `_choose_next_jit` O(n) re-scan per unit edge
   (O(n²S) total). Fixed with O(1) tier-stacks + parallel levels → split is now
   <4% of extraction time it saves. (§8.3.)

**Strong/weak dichotomy:** the three strong engines hold C = S at every depth;
**WFA breaks it** (each weak leaf inflates independently: 11873→12903 dense). This
is a genuine finding — Euler's C=S guarantee is conditional on a strong engine.

**Headline DCT result:** Hungarian Euler depth-4 is the **lowest-DCT config in
the study for T_config ≲ 80µs** (the optical-switch regime) — its C·T_unit
optimum (81.7ms vs 113.5ms for Radix B=2) outweighs its modestly higher T_calc.
Euler complements Radix: cycle-optimal on fast hardware.

**Engine ranking on Euler (dense extraction):** Hungarian fastest (~1.5× ahead of
GW Static at every depth — opposite of the Radix story where GW Static B=2 wins,
because Euler leaves stay dense so iteration count dominates).

---

## 6. Canonical experiment data — IMPORTANT for figure consistency

The paper's **Table 2 (dense summary)** was produced by the batch
`final_runs/20260522_*` — one run per engine, verified exact-match to the table:
- `20260522_121704` → heavy_static_bvn (GW Static)
- `20260522_131035` → heavy_bvn (GW Dynamic)
- `20260522_144633` → wfa_bvn (WFA)
- `20260522_155907` → maximum_bvn (Hungarian)

Config: n=256, k=256, max_weight=64, 1000 matrices, radix-bases [2,4,8,16],
generator "unified". The paper's radix figure panels are sourced from these.
**If you regenerate radix figures, use this batch** so figures stay consistent
with Table 2.

Euler experiment runs live under `euler_exp/` (GITIGNORED — won't transfer to
another station; re-run via the pipeline if needed). The `eng_{maximum,wfa,heavy}_*`
and `final_task*` folders there feed the Euler tables/figures (100 samples each).

---

## 7. How to run experiments (reproduces the paper)

```powershell
# Setup (fresh station) — see README for full version
python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt

# Task 1 (Euler-only, halving): dense + sparse
python main.py -e euler_bvn -n 256 -k 256 --max-weight 64 -s 105 --euler-depths 1 2 3 -o euler_exp/t1_dense
python main.py -e euler_bvn -n 256 -k 16  --max-weight 64 -s 105 --euler-depths 1 2 3 -o euler_exp/t1_sparse

# Task 2 / all-engines (set --euler-leaf-engine to maximum|wfa|heavy|heavy_static)
python main.py -e euler_bvn -n 256 -k 256 --max-weight 64 -s 105 --euler-leaf-engine maximum \
    --euler-depths 1 2 3 4 --radix-bases 2 4 8 16 -o euler_exp/t2_dense
```

Each run writes `results_stats.csv`, plots, and (for euler_bvn)
`euler_comparison_summary.txt` to the timestamped run folder.

---

## 8. Gotchas (read before long runs)

### ⚠️ Long runs die if the machine sleeps
Background runs >~30 min were **killed twice** by the machine sleeping/closing.
Before any 1000-sample or multi-engine run, disable sleep
(`powercfg /change standby-timeout-ac 0` or Settings → Power → Never). A dead run
leaves a partial folder with NO `results_stats.csv` (CSV is written only at 100%).

### Venv is Windows-only
`.venv/Scripts/python.exe` with Windows numpy wheels. Run from PowerShell. On a
new station, recreate the venv (`python -m venv .venv ; pip install -r requirements.txt`).

### Numba JIT warm-up
First samples have inflated timings. Every run drops the first 5 samples. Use
`-s` ≥ ~100 for clean means; the paper uses 100 (euler) / 995 (radix).

### `agent_scripts/` and `*.csv` are gitignored
The standalone task scripts and the DCT-figure generator
(`agent_scripts/plot_dct_vs_tconfig.py`) do NOT transfer via git. The main
pipeline (`main.py`) reproduces every experiment, so this is fine for running —
but the DCT *figure* generator would need to be recreated if you re-make that plot.

### Matrix generation is scaled doubly stochastic, not unit
Row/col sums = S = sum of k weights, not 1. So `bvn_cycle` in CSVs equals S.
Plots show "Normalized Cycle Length" = C / bvn_cycle (BvN = 1.0; Radix > 1).

### Figure placement in the paper
Use `\begin{figure}[htbp]`, NOT `[H]`, for anything taller than ~⅓ page — `[H]`
forces exact placement and creates large white-space gaps when a tall figure
can't share a page. (Already fixed throughout, but keep in mind for new figures.)

### Alon's algorithm is NOT used
The Euler split uses Hierholzer + alternating colouring. Alon (2003) was removed
from the paper because we never implemented his specific algorithm. Cole-Hopcroft
(the foundational result) stays.

---

## 9. Suggested first move in a new session

State your intent up front. Likely next tasks:
- **Finish the paper**: write the Conclusion / Limitations / Future Work section
  (§2 above) — but note the paper isn't in git, so this only works on the
  original machine.
- **More experiments**: bump Euler runs to 1000 samples to match radix (expensive,
  ~hours; disable sleep first).
- **Code**: everything is committed on `euler_splitting`. `git log --oneline -10`
  for recent history.

Diagnostics (PowerShell, project root):
```powershell
git status ; git log --oneline -10
.\.venv\Scripts\python.exe -m pytest tests/ -q -k "not visual"
```
