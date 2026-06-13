# BvN-Arbiter: High-Performance Matrix Decomposition Suite

A modular Python framework for decomposing K-regular / doubly-stochastic traffic matrices into weighted permutation matrices for crossbar switch scheduling. The suite benchmarks several bipartite matching engines and **three** decomposition frameworks — classical BvN, Radix / digit-plane, and **Euler splitting** — against a **three-way tradeoff** between **Permutation Count**, **Cycle Length**, and **Algorithmic Runtime**.

---

## Table of Contents
* [Setup](#setup)
* [Quick Start](#quick-start)
* [Overview](#overview)
* [The Tradeoff Model](#the-tradeoff-model)
* [Directory Structure](#directory-structure)
* [Decomposition Strategies](#decomposition-strategies)
* [Matching Engines](#matching-engines)
* [CLI Configuration](#cli-configuration)
* [Outputs & Plots](#outputs--plots)
* [Tests](#tests)
* [Mathematical Context](#mathematical-context)

---

## Setup

Requires **Python 3.11+**. From a fresh clone:

```bash
# 1. Clone and enter the project
git clone <repo-url> bvn_project
cd bvn_project

# 2. Create and activate a virtual environment
python -m venv .venv

#    Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
#    macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the install (runs the fast test subset)
pytest -q -k "not visual"
```

> **First-run note:** the matching kernels are JIT-compiled with Numba on first
> use, so the first matrix of any run is slow. Every experiment already discards
> the first 5 samples to exclude this warm-up.

If you don't use the venv wrapper, substitute your interpreter for `python`
(e.g. `./.venv/Scripts/python.exe` on Windows, `./.venv/bin/python` on Unix).

---

## Quick Start

```bash
# Compare all four matching engines (BvN + Radix) on dense traffic
python main.py -n 256 -k 256 --max-weight 64 -e all -s 100

# Euler splitting framework: BvN baseline vs Euler leaves, depths 1-3
python main.py -n 256 -k 256 --max-weight 64 -e euler_bvn --euler-depths 1 2 3 -s 100

# Euler vs Radix at matched parallelism, Hungarian leaves
python main.py -n 256 -k 256 --max-weight 64 -e euler_bvn \
    --euler-leaf-engine maximum --euler-depths 1 2 3 4 --radix-bases 2 4 8 16 -s 100
```

Each run writes a timestamped folder under `--output` (default `run/`) with the
config, log, `results_stats.csv`, and plots. See [Outputs & Plots](#outputs--plots).

---

## Overview
In high-speed networking, a crossbar switch must schedule traffic from $N$ inputs to $N$ outputs. This scheduling is modeled as a Birkhoff–von Neumann (BvN) decomposition of a traffic matrix into a weighted sum of permutation matrices. This suite evaluates how different matching engines and decomposition strategies impact scheduling efficiency, focusing on the balance between computational overhead and schedule quality.

Each input matrix is generated as a sum of $k$ random permutations, each scaled by a random weight in $[0, W]$ (integer or float). The framework then runs a **baseline BvN decomposition** and/or a **Radix (digit-plane) decomposition** on each matrix and records runtime, cycle length, and permutation count for every configuration.

---

## The Tradeoff Model
The suite evaluates engines based on three competing parameters:
1. **Permutation Count** — total unique configurations required (minimizes reconfiguration overhead).
2. **Cycle Length** — total weight / duration of the schedule (maximizes throughput).
3. **Runtime** — CPU time required to compute the decomposition (critical for real-time scheduling). For Radix, runtime is the **maximum** plane runtime, simulating ideal hardware parallelism across digit planes.

---

## Directory Structure
```text
bvn_project/
├── main.py                              # Entry point
├── src/
│   ├── cli.py                           # argparse CLI
│   ├── config.py                        # ExperimentConfig dataclass
│   ├── runner.py                        # Per-matrix worker + experiment driver
│   ├── plotting.py                      # Matplotlib visualization
│   ├── algorithms/
│   │   ├── bvn.py                       # Classical BvN decomposition
│   │   ├── radix_decomposition.py       # Radix / digit-plane decomposition
│   │   ├── euler_splitting.py           # Euler splitting framework (JIT)
│   │   ├── wfa.py                       # Wavefront Arbiter (JIT)
│   │   └── sorted_array_matching.py     # Greedy sorted-edge matching (JIT)
│   └── utils/
│       ├── matrix_generator.py          # K-regular matrix generation
│       ├── io_utils.py                  # CSV read/write helpers
│       ├── analysis.py                  # Post-run Euler comparison summary
│       ├── run_utils.py                 # Run-folder + config persistence
│       ├── logging_utils.py             # Rich logging + timed sections
│       └── stats.py                     # DecompositionStats dataclass
├── tests/                               # pytest correctness tests
├── agent_scripts/                       # Sweep drivers, analysis, debug helpers
├── run/                                 # Timestamped per-run outputs (created at runtime)
├── summary/                             # Headline experiment grid (engine × generator)
├── logs/                                # Historical run logs
├── requirements.txt
└── README.md
```

---

## Decomposition Strategies

The runner can produce several decompositions per matrix and compare them side-by-side.

| Strategy | Description |
| :--- | :--- |
| **Classical BvN** | Iteratively peel a permutation out of the matrix, subtract its minimum weight $\lambda$, and repeat until the matrix is zero. Implemented in `src/algorithms/bvn.py`. |
| **Radix (Digit-Plane)** | Split the matrix into digit planes in some integer base $b$ (e.g. $b \in \{2, 4, 8, 16, 32\}$), decompose each plane independently with a chosen matching engine, and combine. Reported runtime is the **max** plane runtime, simulating ideal hardware parallelism. Implemented in `src/algorithms/radix_decomposition.py`. |
| **Euler Splitting** | Recursively split the $S$-regular matrix into two $(S/2)$-regular halves via an Euler-tour 2-colouring of its bipartite multigraph; depth $d$ yields $2^d$ doubly-stochastic leaves, each decomposed independently. Unlike Radix, every leaf stays doubly stochastic, so the cycle length stays at the optimum $C = S$. Reported runtime is the **max** leaf runtime; the split phase is timed separately. Implemented in `src/algorithms/euler_splitting.py`. |

The compound engine modes (`wfa_bvn`, `heavy_bvn`, `heavy_static_bvn`, `maximum_bvn`) run BvN and Radix on the same matrix using the same matching algorithm, enabling apples-to-apples comparison. The `euler_bvn` mode additionally runs the Euler splitting framework (see [CLI Configuration](#cli-configuration)).

---

## Matching Engines

All four engines are heavily optimized with Numba JIT (`nopython=True, nogil=True`).

| Engine (`-e`) | Strategy | Complexity | Notes |
| :--- | :--- | :--- | :--- |
| `wfa` | Wavefront diagonal sweep | $O(N^2)$ per match | Hardware-style maximal matching, lowest latency. |
| `maximum` | Hungarian (`scipy.optimize.linear_sum_assignment`) | $O(N^3)$ | True maximum-weight matching, minimizes permutation count. |
| `heavy` | Greedy sorted-edge matching, **resorted every iteration** | $O(L \log L)$ per iter | Targets cycle length / throughput. |
| `heavy_static` | Greedy sorted-edge matching, **sorted once at startup** | $O(L \log L)$ once | Faster variant of `heavy`; trades some optimality for speed. |

There are also two `maximum` variants used for diagnostics: `minimum` (minimum-weight Hungarian) and the `*_noaug` greedy engines (greedy **without** BFS augmenting paths, producing weak / partial matchings with cycle length > S).

### Compound modes
| Engine flag | What it runs |
| :--- | :--- |
| `wfa_bvn` | BvN with `wfa` + Radix with `wfa` |
| `heavy_bvn` | BvN with `heavy` + Radix with `heavy` |
| `heavy_static_bvn` | BvN with `heavy_static` + Radix with `heavy_static` |
| `maximum_bvn` | BvN with `maximum` + Radix with `maximum` |
| `minimum_bvn` | BvN + Radix with minimum-weight Hungarian |
| `heavy_noaug_bvn`, `heavy_static_noaug_bvn` | Greedy without augmenting paths (weak decomposition, diagnostic) |
| `euler_bvn` | BvN baseline + **Euler splitting** (+ optional Radix); leaf engine set by `--euler-leaf-engine` |
| `all` | BvN with `maximum` + Radix with all four engines |

---

## CLI Configuration

```bash
python main.py --size 256 --k 256 --engine all --samples 100
```

### CLI Arguments

| Flag | Shorthand | Type | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `--size` | `-n` | `int` | Dimension $N$ of the $N \times N$ traffic matrix. | `128` |
| `--k` | `-k` | `int` | Number of permutations summed to build the matrix. | `10` |
| `--engine` | `-e` | `str` | Matching engine or compound mode. See tables above. | `all` |
| `--samples` | `-s` | `int` | Number of random matrices to test. | `10` |
| `--output` | `-o` | `str` | Root directory for run folders. | `run` |
| `--no-plot` | — | flag | Disable automatic plot generation. | off |
| `--max-weight` | — | `float` | Max weight $W$ when sampling permutation weights from $[0, W]$. | `15.0` |
| `--unit-weight` | — | `float` | Scale factor applied to the matrix after generation (use powers of the radix base to shift digit-plane significance). | `1.0` |
| `--float-weights` | — | flag | Sample real-valued weights instead of integers. | off |
| `--radix-bases` | — | `int*` | List of radix bases to test. Not given → `[2 4 8 16 32]` for radix engines, **none** for `euler_bvn` (pass explicitly to add Radix to an Euler run). | (per engine) |
| `--step-strategy` | — | `str` | Radix step-size: `min` / `max` / `median` / `all`. | `min` |
| `--random-seed` | — | `int` | Base seed (each sample uses `seed + index`). | `42` |
| `--max-workers` | — | `int` | Worker count for parallel plane/leaf processing. | `None` |
| `--plot-from-csv` | — | `str` | Path to an existing `results_stats.csv`; regenerates plots only and exits. | `None` |

#### Euler-splitting flags (used with `-e euler_bvn`)

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--euler-depths` | `int*` | Split depths to test (depth $d$ → $2^d$ leaves; depth 0 = plain BvN). | `[1]` |
| `--euler-split-method` | `str` | `heuristic` (same-direction, sparser leaves), `euler` (plain 2-colouring), `euler_grouped` (O(nnz) bulk), `greedy`. | `heuristic` |
| `--euler-leaf-engine` | `str` | Matching engine for the leaves **and** the BvN baseline/Radix planes in this run: `heavy`, `heavy_static`, `wfa`, `maximum`, `minimum`. | `heavy_static` |

### Execution Examples

```bash
# Compare all engines on a 256x256 dense matrix, 100 samples
python main.py -n 256 -k 256 --max-weight 64 -e all -s 100

# WFA + BvN comparison with multiple radix bases
python main.py -n 256 --max-weight 64 -e wfa_bvn -s 1000 --radix-bases 2 4 8 16

# Euler splitting only (no Radix): BvN baseline vs depths 1-3, GW Static leaves
python main.py -n 256 -k 256 --max-weight 64 -e euler_bvn --euler-depths 1 2 3 -s 100

# Euler vs Radix at matched parallelism, Hungarian on every unit
python main.py -n 256 -k 256 --max-weight 64 -e euler_bvn \
    --euler-leaf-engine maximum --euler-depths 1 2 3 4 --radix-bases 2 4 8 16 -s 100

# Sparse-traffic scenario (k=16)
python main.py -n 256 -k 16 --max-weight 64 -e euler_bvn --euler-depths 1 2 3 -s 100

# Regenerate plots from an existing run
python main.py --plot-from-csv run/20260227_113213/results_stats.csv
```

For `euler_bvn` runs, a human-readable `euler_comparison_summary.txt` is also written to the run folder (extraction-time-vs-baseline ratios and the Euler-vs-Radix table).

---

## Outputs & Plots

Each invocation creates a timestamped folder under the `--output` directory containing the run's config, log, CSV, and plots.

### Output Structure
```text
run/
└── 20260227_113213/
    ├── config.json              # Snapshot of ExperimentConfig
    ├── log.txt                  # Per-run log (also mirrored to logs/)
    ├── results_stats.csv        # One row per matrix; runtime/cycle/perms for BvN + every engine_base combo
    └── plots/
        ├── cycle_length_all_methods.png
        ├── cycle_length_pdf_cdf.png
        ├── permutation_count_all_methods.png
        ├── permutation_pdf_cdf.png
        ├── runtime_comparison.png
        ├── runtime_pdf_cdf_subplots.png
        ├── runtime_vs_cycle_<engine>.png
        └── runtime_vs_permutations_<engine>.png
```

### Plot Categories

**Core performance metrics** — cycle length, permutation count, and runtime tracked across the matrix index, with the first 5 samples dropped to avoid Numba JIT warm-up spikes.

**Statistical distributions** — PDF/CDF subplots for runtime, cycle length, and permutation count to reveal stability across traffic patterns.

**Pareto / tradeoff plots** — `runtime_vs_cycle_*` and `runtime_vs_permutations_*` highlight the engine's position in the three-way tradeoff space.

---

## Tests

Run the correctness suite with:

```bash
pytest -q -k "not visual"     # fast: skips the heatmap/graph rendering tests
pytest                         # full suite incl. visual output to tests/visual_output/
```

The tests live in `tests/` and verify:
- Generated matrices are K-regular (equal row & column sums).
- BvN, Radix, and Euler-splitting decompositions exactly reconstruct the original matrix.
- Euler splits are lossless and produce doubly-stochastic $(S/2)$-regular halves.
- Cycle-length invariants hold ($C = S$ for strong engines, $C \ge S$ for WFA).
- Permutation count and cycle length metrics are computed correctly.

A GitHub Actions workflow (`.github/workflows/python-tests.yml`) runs pytest on push.

---

## Mathematical Context

A doubly-stochastic / K-regular non-negative matrix $M$ admits a Birkhoff–von Neumann decomposition:

$$M = \sum_{i=1}^{r} \lambda_i P_i, \qquad \lambda_i > 0, \quad \sum_i \lambda_i = K$$

where each $P_i$ is a permutation matrix. Different matching engines produce different sequences of $P_i$ and different weights $\lambda_i$, leading to the three-way tradeoff this suite measures.

The **Radix** strategy generalizes the classical bitplane approach: the integer matrix is decomposed digit-by-digit in base $b$, each digit plane is independently decomposed via the chosen matching engine, and the planes are summed back. Float matrices are quantized to a fixed precision (16 bits) before digit extraction.
