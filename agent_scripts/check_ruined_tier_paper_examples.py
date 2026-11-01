"""Would the paper's three worked Euler examples change under the four-tier split
(split_method="heuristic_ruined")? Rebuilds exactly the matrices the figure scripts use:
  fig:eulermaps    fig_euler_heatmap_paper.py    _make_matrix(8, 4, 5, max_weight=8)
  fig:eulergraph   fig_euler_bipartite_paper.py  _make_matrix(6, 3, 0, max_weight=6)
  fig:splitting(d) fig_splitting_paper.py        generate_matrix(6, k=6, max_weight=4, rng 40)
and prints, for both rules, the non-zeros of each half and the pairs kept whole or split.

Usage: python agent_scripts/check_ruined_tier_paper_examples.py
"""
import numpy as np

from _paths import REPO  # noqa: F401  (puts the repository on sys.path)
from src.algorithms.euler_splitting import euler_split_once
from src.utils.matrix_generator import generate_matrix
from tests.test_euler_splitting import _make_matrix


def report(name, M):
    out = {}
    for m in ("heuristic", "heuristic_ruined"):
        r, b = (np.asarray(x).astype(int) for x in euler_split_once(M.astype(float), split_method=m))
        out[m] = (r, b)
        whole = int(np.sum((r > 0) ^ (b > 0)))
        split = int(np.sum((r > 0) & (b > 0)))
        print(f"  {name:34s} {m:17s} non-zeros red/blue = {np.count_nonzero(r)}/{np.count_nonzero(b)}  "
              f"pairs whole/split = {whole}/{split}")
    same = all(np.array_equal(out["heuristic"][k], out["heuristic_ruined"][k]) for k in (0, 1))
    print(f"  {'':34s} -> {'identical split' if same else 'DIFFERENT split'}")
    return same


ok = [report("fig:eulermaps (n=8, seed 5)", _make_matrix(8, 4, 5, max_weight=8).astype(np.int64)),
      report("fig:eulergraph (n=6, seed 0)", _make_matrix(6, 3, 0, max_weight=6).astype(np.int64)),
      report("fig:splitting (d) (n=6, rng 40)",
             generate_matrix(6, k=6, max_weight=4.0, rng=np.random.default_rng(40)).astype(np.int64))]
print("all three examples split identically under both rules" if all(ok)
      else "at least one example differs: its figure and caption would change")
