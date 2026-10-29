"""Traffic model TM(t_l, n_f, c_l, n) of Griner & Avin.

Source: C. Griner and C. Avin, "Integrating topology and traffic engineering to
maximize throughput in reconfigurable networks", Computer Networks 274 (2026) 111803.
Reference implementation released by the authors at
https://github.com/KhenGR/Integrated-Topology-and-Traffic-Engineering-for-Reconfigurable-Datacenter-Networks
(src/traficGen.py).  `traffic_generator` and `random_permutation_matrix_no_dig`
below are the authors' code, reproduced verbatim for fidelity.

Model.  A demand matrix is the sum of n_f random *derangement* permutation
matrices (no self-traffic on the diagonal).  A fraction t_l of the flows are
"large" and jointly carry a load c_l; the remaining flows are "small" and carry
(1 - c_l).  Each flow weight is drawn from a Gaussian around its class mean with
standard deviation sigma * mean (the paper uses sigma = 0.01, i.e. 1%).  Because
a sum of weighted permutation matrices contributes exactly one entry per row and
per column, the result is doubly stochastic by construction.

Paper parameter values: t_l = 0.2, c_l = 0.7, sigma = 0.01, n = 64;
n_f swept from ~64 (sparse) to ~3000 (dense).

Integer adaptation (`tm_integer`).
--------------------------------
The authors' generator returns a *float* matrix normalised to row sums 1.  Our
decomposition framework requires *integer* matrices: the Euler split builds a
bipartite multigraph with M[i, j] parallel edges, so entries must be integral,
and the split needs exact regularity.  Rounding the float matrix entrywise would
destroy regularity.  Instead we quantise the *flow weights* rather than the
matrix: the sum of integer-weighted permutation matrices is automatically an
exactly S-regular integer matrix with S = sum of the integer weights.  The
large/small load structure of the model is preserved exactly.
"""
from __future__ import annotations

import numpy as np
from numpy import random
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Authors' reference implementation (verbatim, from src/traficGen.py)
# ---------------------------------------------------------------------------

def traffic_generator(nl, ns, cl, n, sigma):
    """
    This is the traffic model used in the paper, used to generate traffic
    :param nl: number of large flows
    :param ns: number of small flows
    :param cl: ratio of the load of large flows (cs is the small flow load)
    :param n: size of the rectangular traffic matrix
    :param sigma: variance parameter for the model
    :return: an nxn double stochastic matrix with nl+ns flows
    """
    cs = 1 - cl
    large_ratio = cl / nl

    if nl > 0 and cl > 0:
        large_flows = np.random.normal(large_ratio, sigma * large_ratio, int(nl))
    else:
        large_flows = np.zeros(int(nl))
    if ns > 0 and cs > 0:
        small_flows = np.random.normal(cs / ns, sigma * (cs / ns), int(ns))
    else:
        small_flows = np.zeros(ns)
    large_flows_matrix = np.zeros((n, n))
    for i in range(int(nl)):
        large_flows_matrix += large_flows[i] * random_permutation_matrix_no_dig(n)
    small_flows_matrix = np.zeros((n, n))
    for i in range(int(ns)):
        small_flows_matrix += small_flows[i] * random_permutation_matrix_no_dig(n)
    new_matrix = small_flows_matrix + large_flows_matrix
    new_matrix = n * new_matrix / np.sum(new_matrix)
    return new_matrix


def random_permutation_matrix_no_dig(size_of_perm):
    """
    Generate a random permutation matrix with no elements on the diagonal
    :param size_of_perm: size of matrix
    :return: an size_of_perm by size_of_perm matrix
    """
    arr = random.permutation(size_of_perm)
    # Generate a random permutation of integers from 0 to n-1
    while np.any(arr == np.arange(size_of_perm)):
        arr = random.permutation(size_of_perm)

    # Create an n x n matrix filled with zeros
    permutation_matrix = np.zeros((size_of_perm, size_of_perm), dtype=int)

    # Place ones according to the permutation
    for i in range(size_of_perm):
        permutation_matrix[i, arr[i]] = 1

    return permutation_matrix


# ---------------------------------------------------------------------------
# Integer-quantised variant (same model, exact S-regular integer output)
# ---------------------------------------------------------------------------

def _derangement(n: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Random derangement (no fixed point), same semantics as the authors'
    `random_permutation_matrix_no_dig` but with an explicit rng for seeding."""
    arr = rng.permutation(n)
    while np.any(arr == np.arange(n)):
        arr = rng.permutation(n)
    return arr


def tm_integer(
    n: int,
    n_f: int,
    rng: np.random.Generator,
    t_l: float = 0.2,
    c_l: float = 0.7,
    sigma: float = 0.01,
    w_small: int = 7,
) -> NDArray[np.float64]:
    """TM(t_l, n_f, c_l, n) with integer-quantised flow weights.

    Returns an exactly S-regular non-negative integer matrix (as float64, the
    dtype the decomposition engines expect), where S = sum of the integer flow
    weights.

    Args:
        n:        matrix dimension (number of ToRs / ports).
        n_f:      total number of flows per node (density knob; the paper sweeps
                  ~64 sparse .. ~3000 dense).
        rng:      seeded numpy Generator.
        t_l:      fraction of flows that are "large"   (paper: 0.2).
        c_l:      share of total load carried by large flows (paper: 0.7).
        sigma:    Gaussian weight noise as a fraction of the class mean (paper: 0.01).
        w_small:  integer weight assigned to the mean *small* flow.  This sets the
                  quantisation scale; the large-flow weight follows from the model
                  as w_small * (c_l/n_l) / (c_s/n_s).  With the paper's t_l=0.2,
                  c_l=0.7 the ratio is ~9.15, so w_small=7 gives large flows a
                  weight of ~64 -- matching the W_max=64 of our own benchmark.
    """
    n_l = int(np.ceil(t_l * n_f))
    n_s = int(n_f - n_l)
    c_s = 1.0 - c_l

    mean_large = c_l / n_l if n_l > 0 else 0.0
    mean_small = c_s / n_s if n_s > 0 else 0.0

    large_w = rng.normal(mean_large, sigma * mean_large, n_l) if n_l > 0 and c_l > 0 else np.zeros(n_l)
    small_w = rng.normal(mean_small, sigma * mean_small, n_s) if n_s > 0 and c_s > 0 else np.zeros(n_s)

    # Quantise the WEIGHTS (not the matrix): scale so the mean small flow -> w_small.
    scale = (w_small / mean_small) if mean_small > 0 else (w_small / mean_large)
    wi_large = np.maximum(np.round(large_w * scale), 1).astype(np.int64)
    wi_small = np.maximum(np.round(small_w * scale), 1).astype(np.int64)

    M = np.zeros((n, n), dtype=np.int64)
    for w in np.concatenate([wi_large, wi_small]):
        p = _derangement(n, rng)
        M[np.arange(n), p] += w

    return M.astype(np.float64)
