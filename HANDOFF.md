# BvN-Arbiter — Session Handoff

You're picking up an MSc thesis project mid-stream. This file gets you to
context-parity with the previous Claude Code sessions in one read. **Branch:
`euler_splitting`** (not `master` — all work is here).

---

## 1. What this project is

A Python framework that decomposes **scaled doubly stochastic** demand matrices
(rows and columns all sum to a common scale factor S) into weighted permutation
matrices, for **reconfigurable datacenter network / optical circuit switch
scheduling** research. Each permutation = one switch configuration; its weight =
how long that configuration is held.

Four bipartite matching engines × three decomposition frameworks × three step
strategies, measured on a three-way trade-off:

- **Permutation count (N)** — number of switch reconfigurations
- **Cycle length (C = Σλᵢ)** — total transmission slots (optimum = S)
- **Runtime (T_calc)** — wall-clock decomposition time

All priced by **Demand Completion Time**:  `DCT = T_calc + N·T_config + C·T_unit`.

**Engines:** Hungarian (`maximum`, scipy), WFA (`wfa`, maximal/weak),
GW Dynamic (`heavy`, greedy re-sorted per iter), GW Static (`heavy_static`,
sorted once). Strong engines (all but WFA) achieve C=S exactly on regular input
(with min-step).

**Frameworks:** classical BvN; **Radix** (digit planes base B — NOT doubly
stochastic, cycle inflates); **Euler splitting** (Euler-tour 2-colouring →
2^depth leaves, each (S/2^d)-regular → C=S preserved; same-direction heuristic
gives sparser leaves; split is real cost: ~17–70 ms, grows with depth).

**Step strategies (NEW, fully plumbed):** per peel, λ = **min** (classic, only
one preserving C=S) / **median** / **max** of the matched entries, subtraction
clipped at 0 (overdraw ⇒ C>S). Kernels already supported it; this session
plumbed it through `bvn_decomposition`, Euler leaves, and the runner
(`--step-strategy min|median|max|all`; `all` applies to Radix only).

## 2. Where things live

| What | Where |
|---|---|
| **Paper (active)** | `../paper/test.tex` → `test.pdf` (42 pp). NOT in this git repo (Overleaf-synced folder). `file.tex` there is CORRUPT — ignore. |
| **Self-contained paper bundle** | `../final_paper/` (test.tex + plots_pdf/, compiles standalone) |
| **Meeting package** | `../meeting_doc.pdf` (16 pp: Part A story/chapters, Part B 12 exhibits, Part C theory). Generator: `agent_scripts/make_meeting_doc.py` → `../paper/meeting_doc.tex` |
| **Figure board** | `../figure_board.pdf` (9 exhibits; `agent_scripts/make_figure_board.py`) |
| **All figures** | `../paper/plots_pdf/` (paper `\includegraphics` root) |
| **Dense radix+BvN canonical** | `run/rerun_2026/{max,heavy,heavy_static,wfa}` (1000 samples, seed 42; Hungarian BvN bug FIXED here: C=S=8211) |
| **Step-strategy radix** | `run/step_strategy_2026/{...}` (105 samples, bases 2-16 × min/median/max) |
| **Euler min-strategy (d1–d4)** | `euler_exp/eng_{maximum,heavy,heavy_static,wfa}_dense_d4` (+ older `_dense`/`_sparse` d1–d3 runs) |
| **Euler median/max** | `euler_exp/strat_{median,max}_{engine}` |
| **TTFP (time-to-first-perm)** | `run/ttfp/ttfp_results.csv` (`agent_scripts/ttfp_experiment.py`) |
| **Old canonical (pre-bugfix)** | `final_runs/20260522_*` — Hungarian one had 16 impossible C<S matrices; superseded by rerun_2026 |

## 3. Conventions (breaking these invalidates comparisons)

- Matrices: `n=256, k=256, max_weight=64, seed 42+index` (dense); sparse = k=16.
  Same seed ⇒ identical matrices across runs (verified per-matrix).
- Drop first 5 rows of every results CSV (JIT warm-up): `df.iloc[5:]`.
- Runtime totals: BvN = `bvn_runtime`; Radix = max-plane time (digit extraction
  ≈0); **Euler = `depth_time` + `depth_split`** (split is NOT free — every
  figure/analysis must use the split-inclusive total).
- Radix-family runs leave `bvn_*` columns NaN except the `max` engine (runner
  quirk); BvN reference comes from the euler_bvn runs (same matrices).
- Cross-run S mismatch: 1000-sample runs S̄=8211, 105-sample runs S̄=8174. Never
  mix BvN-vs-Euler comparisons across the two families (caused a winner-map
  artifact once; fixed by sourcing BvN from the same run as Euler).

## 4. Key verified results (all audited, `agent_scripts/verify_strategy_runs.py` = ALL PASS)

1. **Strong engines: C = S exactly** (BvN & every Euler depth, min-step); WFA weak (C>S).
2. **Radix trades C↑ for N↓**; **Euler trades N↑ for T_calc↓ at C=S** (GW Static
   exception: λ̄≈1 saturated ⇒ N flat). Orthogonal trajectories in (N,C,T_calc).
3. **DCT winner maps** (fresh, split-inclusive): in the sensible regime
   (r=T_unit/T_config ≥ 5, T_config 1µs–1ms) **Euler depth wins ~97%**, choice =
   which depth (by T_config); Radix B2 only a small corner. **T_calc=0 ⇒ plain
   Hungarian BvN wins 100%** (frameworks exist only to buy compute time).
4. **TTFP reverses the ranking**: first config in ~1 ms for BvN/Radix (WFA BvN:
   0.11 ms) vs 20–41 ms for Euler (split is a hard barrier). Pipelining insight.
5. **Step strategies** (full grid in meeting doc Exhibit 11): Hungarian BvN+max
   → N=175 (λ̄≈61, near the W_max=64 floor; ⌈C/W_max⌉=166 ⇒ within 5% of optimal
   N for its cycle). Median ≈ free lunch for Hungarian B8. WFA+max catastrophic
   (C≈5S). B2 rows identical across strategies (0/1 planes — built-in sanity).
6. GW Dyn vs GW Static: same mean cycle by averaging; per-matrix differs (±192).
   Digit planes NOT doubly stochastic ⇒ only the conservation LOWER bound
   C ≥ Σ B^d·S*(Δ_d) is guaranteed (measured gap ≈0.5%).

## 5. Theory framing (meeting doc Part C)

Provable: strong ⟺ C=S; N ≥ max(⌈S/W_max⌉, max row-nnz); DCT linear ⇒ optimum
is a lower-convex-hull vertex, closed-form break-even; counting the space is
#P-hard (48,840 decompositions for a 4×4 K=3 toy, formula verified).
Conjectures: frontier collapse (B4–B16 always dominated); Radix cycle law
(lower bound + near-tightness); C_WFA ≤ 2S (min-step only!); min-N NP-hard gap;
depth-scaling/saturation law; optimal-depth bound.

## 6. Code changes this session (committed)

- `src/algorithms/bvn.py`: `step_strategy` param; strategy-aware λ in the
  Hungarian path **over positive entries only** (bugfix: median over zeros
  stalled/truncated decompositions); clipped subtraction; kernels now called
  with explicit `strategy_int` (fixed latent positional-arg bug).
- `src/algorithms/euler_splitting.py`: `step_strategy` through
  `decompose_euler_framework` → `_leaf_bvn`.
- `src/runner.py`: passes single strategies to BvN baseline + Euler ('all' → min there).
- `agent_scripts/`: generators (`make_meeting_doc.py`, `make_figure_board.py`,
  `dct_parameter_space.py`, `winner_map_tcalc_sensitivity.py`,
  `build_full_strategy_table.py`, `plot_thesis_candidates.py`,
  `regen_euler_paper_panels.py`, `regenerate_total_runtime.py`), experiments
  (`ttfp_experiment.py`, `decomposition_space_demo.py`), detached launchers
  (`run_*_detached.ps1`), audit (`verify_strategy_runs.py`).

## 7. Operational gotchas (learned the hard way)

- **Long runs MUST be detached**: `Start-Process pwsh -File agent_scripts\run_X_detached.ps1`
  — background Bash dies with the Claude session (killed two overnight batches).
  Runs write a `*_DONE.marker`. Disable sleep first: `powercfg /change standby-timeout-ac 0`.
- Python heredocs through Bash mangle backslashes — use Write/Edit tools for
  scripts, or run `.py` files.
- `main.py --help` crashes (cp1252 vs unicode arrow); actual runs fine. Use
  `./.venv/Scripts/python.exe` always.
- `agent_scripts/` is gitignored — `git add -f` for keepers.
- LaTeX: compile with `pdflatex -interaction=nonstopmode test.tex` in `../paper`;
  meeting doc reads `strategy_full_table.tex` (generated) from that dir.

## 8. Open threads

1. **Paper §10.7 prose** still cites the old wide-plane winner shares ("4 of 32,
   ~67 µs") — the figure now shows the r≥5 regime (Euler 97%/depth-banded).
   Rewrite that paragraph when user approves.
2. **Strategy-aware winner map**: configs like GW-Static-BvN-max (29 ms, N=175)
   would likely claim territory — not yet drawn.
3. **Sparse (k=16) strategy grid** not run (dense only, per user scope).
4. **TTFP + step-strategy exhibits** are in the meeting doc; NOT yet in the
   paper (user explicitly: show first, then decide).
5. Thesis chapters skeleton (Ch3 Problem Definition & Model, Ch4 Solution
   Framework merged) agreed in meeting doc Part A — actual restructure of
   test.tex not started.
6. `tab:step_strategy` in the paper still cites the OLD unverifiable runs —
   regenerated data now exists in `run/step_strategy_2026`; update the table
   from it when touching the paper.

## 9. The advisor loop

Lecturer (Hebrew emails) asked for: high-level story+chapters, ≤10 exhibits
each answering one question with one number, theoretical claims. All delivered
in `meeting_doc.pdf`. His last meeting points (first-permutation results,
T_calc=0/half maps, r≥5 realistic axes, chapter merges, step-strategy
trade-off) — ALL addressed; see Exhibits 10–12.
