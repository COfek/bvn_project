"""Repository-relative paths for the agent scripts. No script may hard-code a machine path.

REPO   the root of this repository (the folder above agent_scripts/), wherever it is cloned.
PAPER  the paper's working folder, where generated tables and figures are written. By default
       a folder named `paper` next to this repository (the layout on the author's machine);
       set the environment variable BVN_PAPER_DIR to use another folder, for example a clone
       of the paper repository: `set BVN_PAPER_DIR=..\\Ofek-Cohen-BvN-Thesis`.
PLOTS  PAPER/plots_pdf, the figures the paper includes.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("BVN_PAPER_DIR", REPO.parent / "paper")).resolve()
PLOTS = PAPER / "plots_pdf"

# Scripts import the framework as `src....`; make that work from any working directory.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
