## Detailed Per-Simulation Base Breakdowns

The tables below detail the performance of the baseline BVN vs. Radix bases (2, 4, 8, 16) across the 16 simulations. **Avg Runtime** is the average time to process a single matrix. **Avg Permutations** and **Avg Cycle Length** average exactly how many discrete matrices the output was factored into, and the resulting cycle bounds for the specific method.

### heavy_bvn_binary_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.00s | 8.0 | 255.0 |
| **Base 2** | 0.01s | 8.0 | 255.0 |
| **Base 4** | 0.00s | 8.0 | 255.0 |
| **Base 8** | 0.00s | 8.0 | 255.0 |
| **Base 16** | 0.00s | 8.0 | 255.0 |

<br>

### heavy_bvn_sinkhorn_100_128
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 2.03s | 8186.0 | 92.1 |
| **Base 2** | 0.09s | 723.0 | 144.2 |
| **Base 4** | 0.19s | 953.7 | 115.1 |
| **Base 8** | 0.30s | 1333.6 | 134.1 |
| **Base 16** | 0.41s | 1680.3 | 95.8 |

<br>

### heavy_bvn_standard_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.19s | 232.8 | 233.0 |
| **Base 2** | 0.05s | 189.8 | 260.1 |
| **Base 4** | 0.11s | 217.4 | 234.4 |
| **Base 8** | 0.11s | 216.2 | 216.6 |
| **Base 16** | 0.11s | 216.2 | 216.3 |

<br>

### heavy_bvn_weighted_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.08s | 180.3 | 185.2 |
| **Base 2** | 0.01s | 33.2 | 317.8 |
| **Base 4** | 0.00s | 29.6 | 251.9 |
| **Base 8** | 0.00s | 29.6 | 251.9 |
| **Base 16** | 0.00s | 29.6 | 251.7 |

<br>

### heavy_static_bvn_binary_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.00s | 50.2 | 281.0 |
| **Base 2** | 0.00s | 8.0 | 255.0 |
| **Base 4** | 0.00s | 8.0 | 255.0 |
| **Base 8** | 0.00s | 10.6 | 260.6 |
| **Base 16** | 0.00s | 15.5 | 281.9 |

<br>

### heavy_static_bvn_sinkhorn_100_128
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.41s | 8186.0 | 95.9 |
| **Base 2** | 0.08s | 733.3 | 146.0 |
| **Base 4** | 0.11s | 965.9 | 113.7 |
| **Base 8** | 0.15s | 1442.0 | 134.8 |
| **Base 16** | 0.22s | 2062.6 | 98.3 |

<br>

### heavy_static_bvn_standard_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.02s | 210.0 | 210.0 |
| **Base 2** | 0.02s | 195.4 | 267.8 |
| **Base 4** | 0.04s | 213.9 | 230.8 |
| **Base 8** | 0.04s | 210.1 | 210.4 |
| **Base 16** | 0.04s | 210.0 | 210.0 |

<br>

### heavy_static_bvn_weighted_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.01s | 173.4 | 242.8 |
| **Base 2** | 0.01s | 34.3 | 326.8 |
| **Base 4** | 0.00s | 30.5 | 259.0 |
| **Base 8** | 0.00s | 30.5 | 259.0 |
| **Base 16** | 0.00s | 30.5 | 258.8 |

<br>

### maximum_bvn_binary_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.00s | 8.0 | 255.0 |
| **Base 2** | 0.01s | 8.0 | 255.0 |
| **Base 4** | 0.00s | 8.0 | 255.0 |
| **Base 8** | 0.00s | 8.0 | 255.0 |
| **Base 16** | 0.00s | 8.0 | 255.0 |

<br>

### maximum_bvn_sinkhorn_100_128
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.46s | 910.2 | 88.0 |
| **Base 2** | 0.12s | 706.7 | 142.1 |
| **Base 4** | 0.13s | 812.3 | 110.6 |
| **Base 8** | 0.18s | 1100.2 | 133.7 |
| **Base 16** | 0.23s | 1340.1 | 93.6 |

<br>

### maximum_bvn_standard_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.11s | 144.8 | 174.8 |
| **Base 2** | 0.09s | 171.9 | 237.9 |
| **Base 4** | 0.12s | 158.3 | 202.4 |
| **Base 8** | 0.11s | 147.0 | 177.3 |
| **Base 16** | 0.11s | 147.0 | 177.0 |

<br>

### maximum_bvn_weighted_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.01s | 20.0 | 170.0 |
| **Base 2** | 0.01s | 26.5 | 261.1 |
| **Base 4** | 0.01s | 20.0 | 170.2 |
| **Base 8** | 0.01s | 20.0 | 170.2 |
| **Base 16** | 0.01s | 20.0 | 170.0 |

<br>

### wfa_bvn_binary_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.08s | 353.8 | 377.4 |
| **Base 2** | 0.01s | 8.0 | 255.0 |
| **Base 4** | 0.00s | 16.0 | 340.0 |
| **Base 8** | 0.01s | 24.4 | 347.9 |
| **Base 16** | 0.01s | 44.3 | 376.7 |

<br>

### wfa_bvn_sinkhorn_100_128
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.48s | 8182.7 | 129.3 |
| **Base 2** | 0.08s | 776.2 | 151.4 |
| **Base 4** | 0.12s | 1095.3 | 136.0 |
| **Base 8** | 0.17s | 1649.8 | 148.2 |
| **Base 16** | 0.27s | 2479.3 | 130.6 |

<br>

### wfa_bvn_standard_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.03s | 257.2 | 257.9 |
| **Base 2** | 0.03s | 210.4 | 287.2 |
| **Base 4** | 0.05s | 255.9 | 273.6 |
| **Base 8** | 0.06s | 257.2 | 258.2 |
| **Base 16** | 0.06s | 257.2 | 257.9 |

<br>

### wfa_bvn_weighted_1000_256
| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |
| :--- | :--- | :--- | :--- |
| **BVN** | 0.04s | 252.1 | 272.7 |
| **Base 2** | 0.01s | 35.9 | 341.1 |
| **Base 4** | 0.01s | 32.5 | 276.7 |
| **Base 8** | 0.01s | 32.5 | 276.6 |
| **Base 16** | 0.01s | 32.5 | 276.4 |

<br>
