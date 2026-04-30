# BvN-Arbiter: High-Performance Matrix Decomposition Suite

A modular Python framework for decomposing K-regular / doubly-stochastic traffic matrices into weighted permutation matrices for crossbar switch scheduling. The suite benchmarks several bipartite matching engines and two decomposition strategies (classical BvN and Radix / digit-plane) against a **three-way tradeoff** between **Permutation Count**, **Cycle Length**, and **Algorithmic Runtime**.

---

## Table of Contents
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
│   │   ├── wfa.py                       # Wavefront Arbiter (JIT)
│   │   └── sorted_array_matching.py     # Greedy sorted-edge matching (JIT)
│   └── utils/
│       ├── matrix_generator.py          # K-regular matrix generation
│       ├── io_utils.py                  # CSV read/write helpers
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

The runner can produce two decompositions per matrix and compare them side-by-side.

| Strategy | Description |
| :--- | :--- |
| **Classical BvN** | Iteratively peel a permutation out of the matrix, subtract its minimum weight $\lambda$, and repeat until the matrix is zero. Implemented in `src/algorithms/bvn.py`. |
| **Radix (Digit-Plane)** | Split the matrix into digit planes in some integer base $b$ (e.g. $b \in \{2, 4, 8, 16, 32\}$), decompose each plane independently with a chosen matching engine, and combine. Reported runtime is the **max** plane runtime, simulating ideal hardware parallelism. Implemented in `src/algorithms/radix_decomposition.py`. |

The compound engine modes (`wfa_bvn`, `heavy_bvn`, `heavy_static_bvn`, `maximum_bvn`) run BvN and Radix on the same matrix using the same matching algorithm, enabling apples-to-apples comparison.

---

## Matching Engines

All four engines are heavily optimized with Numba JIT (`nopython=True, nogil=True`).

| Engine (`-e`) | Strategy | Complexity | Notes |
| :--- | :--- | :--- | :--- |
| `wfa` | Wavefront diagonal sweep | $O(N^2)$ per match | Hardware-style maximal matching, lowest latency. |
| `maximum` | Hungarian (`scipy.optimize.linear_sum_assignment`) | $O(N^3)$ | True maximum-weight matching, minimizes permutation count. |
| `heavy` | Greedy sorted-edge matching, **resorted every iteration** | $O(L \log L)$ per iter | Targets cycle length / throughput. |
| `heavy_static` | Greedy sorted-edge matching, **sorted once at startup** | $O(L \log L)$ once | Faster variant of `heavy`; trades some optimality for speed. |

### Compound modes
| Engine flag | What it runs |
| :--- | :--- |
| `wfa_bvn` | BvN with `wfa` + Radix with `wfa` |
| `heavy_bvn` | BvN with `heavy` + Radix with `heavy` |
| `heavy_static_bvn` | BvN with `heavy_static` + Radix with `heavy_static` |
| `maximum_bvn` | BvN with `maximum` + Radix with `maximum` |
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
| `--max-weight` | — | `float` | Max weight $W$ when sampling permutation weights from $[0, W]$. | `1.0` |
| `--float-weights` | — | flag | Sample real-valued weights instead of integers. | off |
| `--radix-bases` | — | `int*` | List of radix bases to test. | `[2 4 8 16 32]` |
| `--random-seed` | — | `int` | Base seed (each sample uses `seed + index`). | `42` |
| `--max-workers` | — | `int` | Worker count for parallel plane processing (currently sequential for HW simulation). | `None` |
| `--plot-from-csv` | — | `str` | Path to an existing `results_stats.csv`; regenerates plots only and exits. | `None` |

### Execution Examples

```bash
# Compare all engines on a 256x256 matrix, 100 samples
python main.py -n 256 -k 256 -e all -s 100

# WFA + BvN comparison with multiple radix bases
python main.py -n 256 -e wfa_bvn -s 1000 --radix-bases 2 4 8 16

# Float-weight (Sinkhorn-style) experiment
python main.py -n 128 -k 100 -e heavy_bvn --float-weights --max-weight 1.0 -s 100

# Regenerate plots from an existing run
python main.py --plot-from-csv run/20260227_113213/results_stats.csv
```

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
pytest
```

The tests live in `tests/` and verify:
- Generated matrices are K-regular (equal row & column sums).
- BvN and Radix decompositions exactly reconstruct the original matrix.
- Permutation count and cycle length metrics are computed correctly.

A GitHub Actions workflow (`.github/workflows/python-tests.yml`) runs pytest on push.

---

## Mathematical Context

A doubly-stochastic / K-regular non-negative matrix $M$ admits a Birkhoff–von Neumann decomposition:

$$M = \sum_{i=1}^{r} \lambda_i P_i, \qquad \lambda_i > 0, \quad \sum_i \lambda_i = K$$

where each $P_i$ is a permutation matrix. Different matching engines produce different sequences of $P_i$ and different weights $\lambda_i$, leading to the three-way tradeoff this suite measures.

The **Radix** strategy generalizes the classical bitplane approach: the integer matrix is decomposed digit-by-digit in base $b$, each digit plane is independently decomposed via the chosen matching engine, and the planes are summed back. Float matrices are quantized to a fixed precision (16 bits) before digit extraction.
