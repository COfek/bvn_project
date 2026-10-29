---
name: campaign-data-provenance
description: "Which campaign backs which paper number, which are superseded, and the measurement facts a results claim must respect"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26bcde8f-a14e-4d6d-9736-b16700cca933
  modified: 2026-08-29T13:44:08.346Z
---

Established by audit on 2026-08-29. Any number in the paper must trace to one of these.

## Canonical campaigns

- **Radix and the BvN baseline**: `run/rerun_2026/{max,heavy,heavy_static,wfa}/<timestamp>/`
  — 1000 matrices, first 5 discarded, `n=256 k=256 max_weight=64 seed=42`. ~57 min per engine.
- **Euler**: `euler_exp/eng_{maximum,heavy,heavy_static,wfa}_dense_d4/<timestamp>/`
  — **105 matrices**, depths 1–4.
- **Superseded, never cite**: `final_runs/20260522_*` (its Hungarian run had 16 matrices with
  impossible `C < S`), `euler_exp/eng_*_dense/20260612_*`, `euler_exp/final_task2_dense/`.

## The facts that trap people

- **Euler is a 10x smaller sample than Radix** (100 vs 995 after warm-up). No caption says so.
- **The generator seeds matrix i as `seed + i`**, so the Euler campaign's matrices *are* the
  Radix campaign's first 105. Restricting the Radix side to that window makes an
  Euler-vs-Radix comparison exact — same matrices, same `S = 8174.39`. Comparing the full
  Radix window instead silently sets `S = 8211` against `S = 8174`.
- **Three different baselines exist across the paper's tables**: 8211 (Radix table),
  8174 (Euler tables), 8148 (`run/tm_workload`, which used only 10 matrices and discarded 1,
  not 5).
- **Campaign means are precise but not accurate.** Within a run the mean over 995 matrices has
  ~0.15% standard error; three same-seed campaigns of identical work gave 674 / 694 / 717 ms,
  a **6% spread**. That variation is common to every matrix in a run, so averaging inside a run
  cannot see it. Only repeating the whole campaign measures it. Any claim resting on a gap
  smaller than ~6% of the value is inside the noise.
- **`T_compute` is a computed maximum**, not a measured parallel run: every campaign ran with
  `is_parallel: false`, and Radix's charge is the slowest plane only — digit extraction is
  *not* added, despite what Section 6.1 says.
- **`\omega` (the generator's per-permutation weight bound) is 64, but entries of `D` reach
  ~300**, because summed permutations collide. See [[paper-notation-conventions]]; the paper's
  entry bound `W_max` was removed from the model because of this.

## Tooling written for this

- `agent_scripts/make_paper_tables.py` builds `tab_results_summary.tex`,
  `tab_euler_all_engines.tex` and `tab_euler_vs_radix.tex` from the canonical campaigns, each
  with a provenance header. **Before this, those three tables had no generator at all** and
  were hand-assembled from four directories. Never hand-edit them; rerun the script.
- `agent_scripts/run_variance_campaign.ps1` runs three full sweeps and records the measurement
  environment (`environment.json`). Scheduled once for 2026-08-30 02:30 as the Windows task
  **"BvN variance campaign"**; ~13.5 h, output in `run/variance_2026/sweep{1,2,3}/`.

## CLI gotcha

`--engine` takes **`max`** for Hungarian; `--euler-leaf-engine` takes **`maximum`**. Using the
wrong one exits 2. Interpreter is `bvn_project/.venv/Scripts/python.exe` (Python 3.13,
numpy 2.3.5, scipy 1.17.0, numba 0.63.1). Machine: 24 logical CPUs, AMD64, Windows 11.

See [[paper-working-files]] and [[paper-writing-principles]].
