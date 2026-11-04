"""Regression tests for the code review of 2026-11-04 (docs/code_review_2026-11-04.md).

Each test pins one bug that was found and fixed, or one behaviour the paper relies on
that had no test.
"""
import hashlib

import numpy as np
import pytest

from src.algorithms.bvn import bvn_decomposition, bvn_decomposition_timed
from src.algorithms.euler_splitting import decompose_euler_framework, euler_split_once
from src.algorithms.radix_decomposition import decompose_radix, extract_digit_planes
from src.config import ExperimentConfig
from src.runner import ExperimentIncomplete, _compute_for_index, run_experiment
from src.utils.matrix_generator import generate_matrix

ENGINES = ["maximum", "heavy", "heavy_static", "wfa"]


def _rebuild(planes, shape):
    out = np.zeros(shape)
    for w, p in planes:
        out += w * p
    return out


# --- Radix digit extraction (review B: log-based digit count) ---------------------

@pytest.mark.parametrize("base", [2, 3, 4, 8, 10, 16])
def test_float_digit_planes_keep_the_largest_entries(base):
    """Float mode scales the largest entry to 2**16; ceil(16 / log2 b) planes dropped it."""
    M = generate_matrix(12, 6, 5, np.random.default_rng(5), float_weights=True)
    rebuilt = _rebuild(extract_digit_planes(M, base), M.shape)
    np.testing.assert_allclose(rebuilt, M, atol=M.max() / 2 ** 16)


@pytest.mark.parametrize("value,base", [(1000, 10), (243, 3), (59049, 3), (10 ** 6, 10),
                                         (256, 2), (4096, 16), (729, 9)])
def test_integer_maximum_at_a_power_of_the_base(value, base):
    """floor(log(b**e) / log(b)) can be e - 1, which dropped the top digit."""
    M = np.eye(4) * value
    np.testing.assert_array_equal(_rebuild(extract_digit_planes(M, base), M.shape), M)
    comps, _, _ = decompose_radix(M, base=base, matching_method="heavy_static")
    np.testing.assert_allclose(sum(c.matrix for c in comps), M)


def test_radix_rejects_base_below_two():
    with pytest.raises(ValueError):
        extract_digit_planes(np.eye(3), 1)


# --- no silent fallbacks (review B) -----------------------------------------------

def test_unknown_names_raise():
    M = generate_matrix(6, 3, 4, np.random.default_rng(0))
    with pytest.raises(ValueError):
        bvn_decomposition(M, matching_algorithm="max")
    with pytest.raises(ValueError):
        bvn_decomposition(M, step_strategy="mean")
    with pytest.raises(ValueError):
        decompose_radix(M, base=2, matching_method="max")
    with pytest.raises(ValueError):
        euler_split_once(M, split_method="heurstic")
    with pytest.raises(ValueError):
        decompose_euler_framework(M, split_method="heurstic")


def test_euler_rejects_float_and_irregular_input():
    M = generate_matrix(6, 3, 4, np.random.default_rng(0))
    with pytest.raises(ValueError):
        decompose_euler_framework(M + 0.5, depth=1)
    bad = M.copy()
    bad[0, 0] += 1
    with pytest.raises(ValueError):
        decompose_euler_framework(bad, depth=1)


def test_failed_matrices_fail_the_run():
    """A matrix that raises used to be dropped, and the run still looked complete."""
    cfg = ExperimentConfig(n=8, k=4, max_weight=4.0, num_matrices=3, engine="euler_bvn",
                           radix_bases=[], euler_depths=[1], float_weights=True)
    with pytest.raises(ExperimentIncomplete) as info:
        run_experiment(cfg)
    assert info.value.failed == [0, 1, 2]


# --- timing changes must not change any result (review A) -------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_timed_bvn_returns_the_same_decomposition(engine):
    M = generate_matrix(24, 12, 9, np.random.default_rng(3))
    plain = bvn_decomposition(M, matching_algorithm=engine)
    timed, seconds = bvn_decomposition_timed(M, matching_algorithm=engine)
    assert seconds > 0 and len(plain) == len(timed)
    for a, b in zip(plain, timed):
        assert a.weight == b.weight
        np.testing.assert_array_equal(a.permutation, b.permutation)
    np.testing.assert_allclose(sum(c.weight * c.permutation for c in timed), M)


@pytest.mark.parametrize("engine", ENGINES)
def test_paper_euler_configuration_is_exact(engine):
    """The paper's own setup: heuristic split, depths 1-4; leaves regular, C = S for exact engines."""
    M = generate_matrix(32, 32, 16, np.random.default_rng(42))
    S = M.sum(axis=1)[0]
    for depth in (1, 2, 3, 4):
        comps, split_s, leaf_s, n_leaves = decompose_euler_framework(
            M, matching_method=engine, depth=depth, split_method="heuristic")
        np.testing.assert_allclose(sum(c.weight * c.permutation for c in comps), M)
        assert split_s > 0 and leaf_s > 0 and n_leaves <= 2 ** depth
        if engine != "wfa":
            assert abs(sum(c.weight for c in comps) - S) < 1e-9


def test_heuristic_split_is_sparser_than_the_plain_split():
    """The claim behind sparse Euler splitting: fewer non-zeros in the component matrices."""
    sparser = 0
    for seed in range(10):
        M = generate_matrix(32, 32, 16, np.random.default_rng(seed)).astype(np.int64)
        plain = sum(np.count_nonzero(h) for h in euler_split_once(M, "euler"))
        heur = sum(np.count_nonzero(h) for h in euler_split_once(M, "heuristic"))
        sparser += heur < plain
    assert sparser == 10


def test_free_between_runs_changes_no_euler_result():
    kw = dict(n=24, k=24, num_matrices=1, engine="euler_bvn", radix_bases=[2, 4],
              euler_depths=[1, 2, 3], euler_leaf_engine="heavy_static", max_weight=8.0)
    a = _compute_for_index(0, ExperimentConfig(free_between_runs=False, **kw))
    b = _compute_for_index(0, ExperimentConfig(free_between_runs=True, **kw))
    euler_keys = [k for k in a.radix_multi_results if k.startswith("euler_")]
    assert len(euler_keys) == 3
    for key in a.radix_multi_results:
        assert a.radix_multi_results[key][1:4] == b.radix_multi_results[key][1:4], key


# --- the campaign matrices (data provenance) ---------------------------------------

def test_generator_reproduces_the_campaign_matrices():
    """Matrix i of every paper campaign is generate_matrix(256, 256, 64, seed 42 + i); the
    published S of matrices 0-2 are 7790, 8743 and 8178."""
    pins = {0: (7790.0, "8f9b018d9ba2d23e"), 1: (8743.0, "cdbae00eb2e82d34"),
            2: (8178.0, "b7f778825c47f5a0")}
    for i, (S, digest) in pins.items():
        M = generate_matrix(256, 256, 64.0, np.random.default_rng(42 + i))
        assert M.sum(axis=1)[0] == S
        h = hashlib.sha256(np.ascontiguousarray(M, dtype=np.float64).tobytes()).hexdigest()
        assert h[:16] == digest
