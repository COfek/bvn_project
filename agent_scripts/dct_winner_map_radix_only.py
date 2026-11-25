"""Figure 5's DCT winner map without the Euler setups: unsplit and Radix only (Chen K7).

A sanity check for Chen, not a paper figure (Ofek, 2026-11-26). It runs dct_parameter_space.py
unchanged except that no Euler depth competes, and copies the single-column winner map to
Thesis/for_chen/dct_winner_map_radix_only.{pdf,png}. GW Static unsplit and GW Static B16 have
nearly the same grey in winner_colors, so B16 is drawn in another colour here.
Run with BVN_CAMPAIGN=2026_11 BVN_RUN_ROOT=run/rerun_2026_11e, as for Figure 5.
"""
import os, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "for_chen"
tmp = Path(tempfile.mkdtemp())
(tmp / "plots_pdf").mkdir()
os.environ["BVN_PAPER_DIR"] = str(tmp)   # keep the paper's own figures untouched
sys.path.insert(0, str(HERE))

import winner_colors   # noqa: E402
_orig = winner_colors.color_of
winner_colors.color_of = lambda lab: "#2a9d8f" if lab == "GW Static B16" else _orig(lab)

p = HERE / "dct_parameter_space.py"
src = p.read_text(encoding="utf-8")
assert src.count("EULER_DEPTHS = [1, 2, 3, 4]") == 1
src = src.replace("EULER_DEPTHS = [1, 2, 3, 4]", "EULER_DEPTHS = []")
try:
    exec(compile(src, str(p), "exec"), {"__file__": str(p), "__name__": "__main__"})
except Exception as e:   # the later figures of that script assume Euler; the map is saved first
    print("stopped after the winner map:", type(e).__name__, e)
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    shutil.copyfile(tmp / "plots_pdf" / f"dct_winner_map.{ext}", OUT / f"dct_winner_map_radix_only.{ext}")
print("saved", OUT / "dct_winner_map_radix_only.pdf")
