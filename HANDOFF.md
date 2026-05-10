# BvN-Arbiter — Session Handoff

You're picking up an MSc thesis project mid-stream. This file gets you to context-parity with the previous session in one read.

---

## 1. What this project is

A Python framework that decomposes **scaled doubly stochastic** demand matrices (rows and columns all sum to a common scale factor S) into weighted permutation matrices, for crossbar-switch scheduling research. It benchmarks several bipartite matching algorithms across two decomposition frameworks (classical BvN and Radix / digit-plane), measuring a three-way tradeoff:

- **Permutation count (N)** — number of switch reconfigurations
- **Cycle length (∑ϕᵢ)** — total transmission duration
- **Algorithmic runtime (T_calc)** — wall-clock decomposition time

These feed the real cost function the project actually cares about — **Demand Completion Time**:

$$DCT = T_{calc} + (N \times T_{config}) + \left(\sum_{i=1}^{N} \phi_i \times T_{unit}\right)$$

where T_config is hardware reconfiguration delay and T_unit is per-unit transmission time. Different (matching, framework) combos win on different (T_config, T_unit) hardware profiles. The current README focuses on the three input metrics; the DCT framing lives in `project background.pdf` (uploaded but not yet folded into the README).

---

## 2. Layout

```
bvn_project/
├── main.py                       # entry point
├── src/
│   ├── cli.py                    # argparse
│   ├── config.py                 # ExperimentConfig dataclass
│   ├── runner.py                 # per-matrix worker + experiment driver
│   ├── plotting.py               # matplotlib (heavy file, ~620 lines)
│   ├── algorithms/
│   │   ├── bvn.py                # classical BvN decomposition
│   │   ├── radix_decomposition.py# digit-plane decomposition
│   │   ├── wfa.py                # Wavefront Arbiter (Numba JIT)
│   │   └── sorted_array_matching.py  # heavy / heavy_static (Numba JIT)
│   └── utils/
│       ├── matrix_generator.py   # K-regular generator
│       ├── io_utils.py           # CSV read/write
│       ├── stats.py              # DecompositionStats dataclass
│       ├── run_utils.py          # run-folder + config persistence
│       └── logging_utils.py
├── tests/                        # pytest correctness tests
├── agent_scripts/                # scratch sweeps, analysis, debug
├── run/                          # timestamped per-run outputs
├── summary/                      # headline 16-experiment grid
└── .venv/                        # Windows venv (Scripts/python.exe)
```

Engines: `wfa`, `maximum` (scipy Hungarian), `heavy` (greedy sorted-edge, dynamic re-sort), `heavy_static` (greedy sorted-edge, sort-once). Compound modes `wfa_bvn`, `heavy_bvn`, `heavy_static_bvn`, `maximum_bvn`, `all` run BvN and Radix on the same matrices for direct comparison. See README for the full CLI table.

---

## 3. What was just done in the last session

### A. README rewritten to match real code
- Directory tree, CLI flags, all 4 matching engines + compound modes, new "Decomposition Strategies" section, fixed broken code fences. Don't believe any older description that mentions `heavy_point.py` (file was renamed to `sorted_array_matching.py`) or a `--density` flag (it's `--k`).

### B. Plane count threaded through to plots
- `decompose_radix` now returns `(components, runtime, num_non_empty_planes)`.
- Runner stores a 4-tuple `(runtime, cycle, perms, planes)` in `radix_multi_results`.
- CSV gains `{key}_planes` columns. Reader tolerates older CSVs missing the column (defaults to 0 sentinel).

### C. BVN matching algorithm now recorded
- `DecompositionStats.bvn_matching: Optional[str]` field added.
- Runner sets it to whatever matching BVN actually used (`wfa`, `heavy`, `heavy_static`, or `maximum`).
- CSV gains `bvn_matching` column. Reader tolerates older CSVs missing it.

### D. Plot legend convention is now consistent: `<matching>-<framework>`
- Radix labels: `WFA-B2`, `HEAVY-B4`, `MAX-B16`, etc. (was `WFA (B-2)` etc.)
- BVN baseline: `WFA-BVN`, `MAX-BVN`, etc. via a new `_bvn_legend_label(stats)` helper.
- Helper falls back to plain `BVN` when `bvn_matching` is missing (legacy CSVs).
- Applied uniformly across the two efficiency plots, the time-series `_plot_trend`, and the distribution `_plot_dynamic_grid`.

### E. Per-point annotations on efficiency plots (color-blind safety)
- Each radix point on `runtime_vs_cycle_*` and `runtime_vs_permutations_*` is annotated with `B<base>, <N>p` (e.g. `B2, 8p`) in the marker color, bold, fontsize 8.
- Falls back to `B<base>` if plane data is missing, or `B<base>, <min>-<max>p` if plane count varies across samples.

### F. WFA Trend line removed from both efficiency plots
- It was only wired up for WFA (no equivalent for other engines) and looped on itself when the base→performance relationship wasn't monotonic. The Pareto Frontier remains as the sole overlay.

### G. Tests updated
- `tests/test_simulation_correctness.py`: 3 call sites updated to unpack the new 3-tuple from `decompose_radix`.

### H. Typo fix (user-applied)
- `src/algorithms/__init__,py` → `__init__.py` (the comma was preventing the package from being a clean module).

---

## 4. Open threads / things in flight

### Sanity check for BVN cycle length
A snippet was drafted but not yet inserted. It verifies for each generated matrix that:
1. All rows sum to the same value S.
2. All columns sum to S.
3. `cycle_length_bvn ≈ S` (within 1e-6).

The intended placement is in `src/runner.py` inside `_compute_for_index`, immediately after the BVN block (after `bvn_matching_used = bvn_engine`). Hard-fail with `raise ValueError`. Skip the cycle check on the `else` branch where BVN doesn't run.

```python
row_sums = matrix.sum(axis=1)
col_sums = matrix.sum(axis=0)
S = float(row_sums[0])
if not np.allclose(row_sums, S, atol=1e-6):
    raise ValueError(f"Matrix index {index}: row sums vary "
                     f"(min={row_sums.min():.6g}, max={row_sums.max():.6g}).")
if not np.allclose(col_sums, S, atol=1e-6):
    raise ValueError(f"Matrix index {index}: col sums differ from row sums.")
if not np.isclose(cycle_length_bvn, S, atol=1e-6):
    raise ValueError(f"Matrix index {index}: BVN cycle length "
                     f"{cycle_length_bvn:.6f} != scale factor S={S:.6f}.")
```

### Possible follow-ups (not committed to)
- **Split-Tree decomposition framework** — referenced in `project background.pdf` but not yet implemented in `src/algorithms/`. Heuristics it'd need: Sparsity Target, Max Depth, CV Threshold, Minimum Matching Fraction.
- **DCT-aware comparison plot** — a fourth efficiency plot that plots actual DCT under user-specified (T_config, T_unit) hardware profiles, since that's the real metric. The three current efficiency plots (runtime, perms, cycle) are inputs to DCT, not DCT itself.
- **README**: add the DCT formulation and scaled-doubly-stochastic framing to the "Mathematical Context" section. Currently it says "doubly stochastic" but the input is really *scaled* (rows/cols sum to S, not 1). The user already understands this distinction (came up explicitly in the last session); README just doesn't reflect it yet.

---

## 5. Things to know (gotchas)

### Venv is Windows-only
`.venv/Scripts/python.exe` with Windows-built numpy wheels (`_delvewheel` shims). Don't try to run it from a Linux shell. Pytest and the actual `main.py` invocations need to be run from PowerShell:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
.venv\Scripts\python.exe main.py -n 64 -k 32 -e wfa_bvn -s 30 --radix-bases 2 4 8 16
```

### Files use CRLF line endings
Most files are CRLF (`\r\n`). Some recent rewrites are LF — that's fine for Python but watch for spurious git diffs if your editor auto-converts.

### `agent_scripts/` is scratch
Many of the scripts there assign `decompose_radix(...)` to a single variable (don't unpack the tuple). They were already in that broken state before any recent refactor; they're not part of the main flow. Don't bother fixing unless you actually want to run them.

### Numba JIT warm-up
First few samples in any run get inflated timings from JIT compilation. The plotting code already drops the first 5 samples (`stats_list[5:]` in `main.py`). When debugging, prefer `--samples 30+` so the warm-up samples aren't a big fraction of the data.

### Matrix generation is scaled doubly stochastic, not unit
The generator (`generate_matrix`) sums `k` random permutations with random integer or float weights. Row/col sums are S = sum of those k weights — *not* 1. So `cycle_length_bvn` in the CSV equals S, not 1. The plots show "Average Cycle Length (Normalized)" because each cycle is divided by `cycle_length_bvn` before plotting (BVN divided by BVN = 1.0; Radix shows up as a ratio above 1). This was a point of confusion in the last session — don't get tripped up by it.

### Hardcoded `5e-1` warmup drop
The "drop first 5 samples" logic is hardcoded in two places (`main.py` and `--plot-from-csv` mode). If `--samples < 10`, all samples are kept. This is fine but worth knowing if a small-sample run produces weirdly different plots than a large-sample one.

---

## 6. Suggested first move in the new session

State your intent up front. If you're picking up the BVN sanity-check work, say so and reference section 4 above. If you're starting something new, give a one-liner about what.

Useful diagnostic commands (PowerShell, project root):

```powershell
git status
git log --oneline -10
.venv\Scripts\python.exe -m pytest tests/ -v
```
