# BvN-Arbiter: High-Performance Matrix Decomposition Suite

A modular Python framework for decomposing traffic demand matrices into optimal crossbar switch permutations. This project benchmarks multiple bipartite matching strategies to explore the **three-way tradeoff** between **Permutation Count**, **Cycle Length**, and **Algorithmic Runtime**.



## 📋 Table of Contents
* [Overview](#overview)
* [The Tradeoff Model](#the-tradeoff-model)
* [Directory Structure](#directory-structure)
* [Matching Engines](#matching-engines)
* [CLI Configuration](#cli-configuration)
* [Outputs & Plots](#outputs--plots)
* [Results](#results)
* [Mathematical Context](#mathematical-context)

---

## 📖 Overview
In high-speed networking, a crossbar switch must schedule traffic from $N$ inputs to $N$ outputs. This scheduling is modeled as a Birkhoff-von Neumann (BvN) decomposition of a traffic matrix. This suite evaluates how different matching engines impact scheduling efficiency, specifically focusing on the balance between computational overhead and schedule quality.

---

## ⚖️ The Tradeoff Model
The suite evaluates engines based on three competing parameters:
1.  **Permutation Count:** Total unique configurations required (minimizes reconfiguration overhead).
2.  **Cycle Length:** The total weight/duration of the schedule (maximizes throughput).
3.  **Runtime:** CPU time required to compute the decomposition (critical for real-time scheduling).

---

## 📂 Directory Structure
```text
bvn_project/
├── src/
│   ├── algorithms/
│   │   ├── wfa.py                   # Vectorized Wavefront implementation
│   │   ├── sorted_array_matching.py # Sorted Array Matching implementation
│   │   ├── bvn.py                   # BVN decomposition
│   │   ├── radix_decomposition.py   # Radix decomposition
│   │   └── heavy_point.py           # Weighted Conflict Greedy implementation
│   ├── utils/
│   │   ├── matrix_generator.py      # Random & Doubly Stochastic matrix generators
│   │   ├── plotting.py              # Visualization & Plotting tools
│   │   ├── run_utils.py             # Run utils
│   │   ├── stats.py                 # Statistics utils
│   │   └── logging_utils.py         # Logging utils
├── outputs/
│   ├── plots/                       # Generated performance graphs
│   └── logs/                        # CSV/JSON raw benchmark data
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation

## ⚙️ Matching Engines

The suite evaluates three distinct strategies, each representing a different priority within the tradeoff space. Users can select a specific engine or run all simultaneously for comparative analysis.

| Engine | Strategy | Complexity | Optimization Goal |
| :--- | :--- | :--- | :--- |
| **Vectorized WFA** | Geometric Wavefront | $O(N)$ | **Runtime** (Lowest latency) |
| **Max Matching** | Cardinality Search | $O(N^3)$ | **Permutation Count** |
| **Heavy-Point** | Weighted Greedy | $O(L \log L)$ | **Cycle Length** (Throughput) |



### Engine Descriptions:

* **Vectorized Wavefront (WFA):** Optimized for speed, this engine uses a diagonal-based approach to find a maximal matching. It is ideal for systems where the time to compute a schedule is the primary bottleneck.
* **Maximum Bipartite Matching:** Utilizes cardinality-based algorithms to ensure the highest number of edges are selected in each permutation. This effectively reduces the total number of permutations but at a higher computational cost.
* **Heavy-Point Matching:** A custom greedy approach that prioritizes high-weight nodes in the conflict graph. By sinking the most "mass" in each cycle, it targets the optimal Cycle Length, ensuring high throughput despite a moderate runtime.


## ⌨️ CLI Configuration

The suite is fully configurable via the command line to allow for automated batch testing and research across various matrix profiles. 

```bash
python src/main.py --size 512 --density 0.7 --engine all --samples 100

### CLI Arguments

| Flag | Shorthand | Type | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `--size` | `-n` | `int` | The dimension of the $N \times N$ traffic matrix. | `128` |
| `--density` | `-d` | `float` | Probability (0.0 to 1.0) of a matrix entry being non-zero. | `0.5` |
| `--engine` | `-e` | `str` | Matching algorithm: `wfa`, `max`, `heavy`, or `all`. | `all` |
| `--samples` | `-s` | `int` | Number of random matrices to test per engine for averaging. | `10` |
| `--output` | `-o` | `path` | Root directory path for saving generated plots and CSV logs. | `./outputs` |
| `--verbose` | `-v` | `flag` | Enables per-permutation logging for detailed step analysis. | `False` |
| `--plot` | `-p` | `flag` | Automatically generate and save plots after the benchmark. | `True` |

---

### Execution Examples

```bash
# Compare all engines on a 512x512 matrix with high density
python src/main.py -n 512 -d 0.8 -e all

# Stress test the Heavy-Point engine with 100 samples
python src/main.py -n 256 -e heavy -s 100 --verbose

# Run a quick WFA test and save results to a custom folder
python src/main.py -n 1024 -e wfa -s 1 -o ./benchmarks/wfa_test

## 📊 Outputs & Plots

The suite provides a comprehensive visualization library to analyze the behavior and efficiency of different matching algorithms. All plots are automatically saved to the `outputs/` directory for each benchmark run.

### 📈 Core Performance Metrics
These plots track performance across the test set (matrix index):
* **Final Cycle Length:** Tracks total schedule weight as a function of the matrix index.
* **Final Permutation Count:** Visualizes the number of permutations required per matrix sample.
* **Runtime Analysis:** Plots the execution time (ms) for each matrix index to identify computational variance.

### 📊 Statistical Distributions
To understand algorithmic stability across different traffic patterns:
* **Runtime Distribution:** A histogram showing the frequency of execution times.
* **Cycle Length Distribution:** Analysis of how the total weight is distributed across the sample set.
* **Permutation Distribution:** Frequency analysis of the number of permutations required.

### ⚖️ Pareto Efficiency & Tradeoffs
The following plots are crucial for finding the optimal balance for specific system constraints:
* **Runtime vs. Cycle Efficiency:** Specifically highlights the tradeoff between computational cost and throughput optimization.
* **Runtime vs. Permutation Efficiency:** Evaluates how much CPU time is required to achieve a reduction in total reconfiguration overhead.

---

### 📁 Output Structure
The system generates unique, timestamped subdirectories for every execution:

```text
bvn_project/
├── runs/
│   └── run_20260124_1030/       # Timestamped run folder
│       ├── plots/               # Folder containing all generated visualizations
│       ├── results_stats.csv    # Raw statistical data for every matrix in the run
│       └── run_config.json      # The specific CLI parameters used for this run
└── logs/
    └── run_20260124_1030.log    # Timestamped execution, status, and error logs