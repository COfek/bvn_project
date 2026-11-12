"""Merge the blocks of run_euler_1000.ps1 into one results_stats.csv per sweep and engine.

Block a ran --random-seed 42+a, so its local matrix i is global matrix a+i. Its first WARM rows
are warm-up and are dropped, except in block 0, whose rows 0..WARM-1 are kept: they are the
campaign's own warm-up matrices 0-4, which every reader (variance_data, make_paper_tables)
discards itself, exactly as for the single-run campaigns. Disturbed or failed blocks
(block_<a>_disturbed_*, block_<a>_failed_*) are never read.

Writes <root>/sweep<s>/euler/<engine>/merged/results_stats.csv with global matrix indices
0..TOTAL-1, and refuses if any index is missing or doubled.

Usage: .venv/Scripts/python.exe agent_scripts/merge_euler_blocks.py [root] [total] [warm]
"""
from __future__ import annotations
from _paths import REPO  # repository-relative paths

import glob
import os
import re
import sys

import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "run/euler_1000_2026_11"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
WARM = int(sys.argv[3]) if len(sys.argv) > 3 else 5

base = os.path.join(str(REPO), ROOT)
bad = 0
for eng_dir in sorted(glob.glob(os.path.join(base, "sweep*", "euler", "*"))):
    if not os.path.isdir(eng_dir):
        continue
    parts = []
    for blk in sorted(os.listdir(eng_dir)):
        m = re.fullmatch(r"block_(\d+)", blk)
        if not m:
            continue
        a = int(m.group(1))
        csvs = sorted(glob.glob(os.path.join(eng_dir, blk, "*", "results_stats.csv")))
        if not csvs:
            continue
        d = pd.read_csv(csvs[-1])
        if a > 0:
            d = d[d.matrix_index >= WARM]
        d = d.assign(matrix_index=d.matrix_index + a)
        parts.append(d)
    rel = os.path.relpath(eng_dir, str(REPO)).replace(os.sep, "/")
    if not parts:
        print(f"{rel}: no blocks yet")
        continue
    df = pd.concat(parts).sort_values("matrix_index").reset_index(drop=True)
    idx = df.matrix_index.tolist()
    if idx != list(range(TOTAL)):
        missing = sorted(set(range(TOTAL)) - set(idx))
        doubled = len(idx) - len(set(idx))
        print(f"{rel}: INCOMPLETE ({len(set(idx))} of {TOTAL} matrices, "
              f"{len(missing)} missing, {doubled} doubled), not written")
        bad += 1
        continue
    out = os.path.join(eng_dir, "merged")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "results_stats.csv"), index=False)
    print(f"{rel}: merged {TOTAL} matrices -> {rel}/merged/results_stats.csv")
sys.exit(1 if bad else 0)
