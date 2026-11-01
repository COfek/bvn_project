"""--free-between-runs must change no N or cycle length, only what memory the timed
calls run beside."""
from src.config import ExperimentConfig
from src.runner import _compute_for_index


def _stats(free):
    cfg = ExperimentConfig(n=24, k=24, num_matrices=2, engine="euler_bvn", radix_bases=[2, 4],
                           euler_depths=[1, 2, 3], euler_leaf_engine="heavy",
                           euler_split_method="heuristic_ruined", max_weight=8.0,
                           free_between_runs=free)
    return [_compute_for_index(i, cfg) for i in range(2)]


def test_free_between_runs_changes_no_result():
    for a, b in zip(_stats(False), _stats(True)):
        assert a.num_permutations_bvn == b.num_permutations_bvn
        assert a.cycle_length_bvn == b.cycle_length_bvn
        assert set(a.radix_multi_results) == set(b.radix_multi_results)
        for key, va in a.radix_multi_results.items():
            vb = b.radix_multi_results[key]
            assert va[1] == vb[1] and va[2] == vb[2], key   # cycle length and N
