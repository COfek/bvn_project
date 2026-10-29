"""MoE expert-parallel all-to-all traces -> S-regular integer demand matrices.

Source of the traces: the Birkhoff-MoE measurements (StygianNet / Purdue),
https://stygianet.cs.purdue.edu/papers/birkhoff-moe-hotoptics26.pdf.

Trace format
------------
One CSV per (model, dataset, pass, layer).  Columns:

    model, dataset, pass_idx, layer_id, source_rank, dest_rank_0 ... dest_rank_{n-1}

Each row is one *source* GPU rank; `dest_rank_j` is the number of tokens that
rank routes to rank j during that layer's expert all-to-all.  So the file already
IS the demand matrix -- `M[i, j]` = tokens from rank i to rank j -- once the
metadata columns are dropped and the rows are sorted by `source_rank`.  With 8
ranks the matrices are 8x8 and fully dense (every rank sends to every rank).

Why a raw trace cannot be fed to the decomposition directly
-----------------------------------------------------------
Our framework decomposes *S-regular* integer matrices: every row and every column
must sum to the same S.  The traces are integer (token counts), but they are not
regular, and in general cannot be:

  * Row sums are the tokens each rank *emits*.  Under top-k routing with a fixed
    batch these are often equal (Mixtral 8x22B: every row sums to 16384), but not
    always -- a rank that finishes its microbatch early emits fewer tokens.
  * Column sums are the tokens each rank *receives*, i.e. the popularity of the
    experts it hosts.  Expert popularity is exactly what is skewed in MoE, so
    column sums are never equal in practice.

A demand matrix is made regular by adding *slack*: padding entries that occupy
circuit time without carrying real traffic.  This is the physically honest model
-- the switch really does sit idle on those ports -- and it is the standard
completion used when scheduling non-uniform demand on a crossbar.

`regularize` adds the minimum possible slack.  With S = max(max row sum, max
column sum), the total padding is exactly n*S - sum(M), which is forced: no
S-regular matrix dominating M can add less.  The padding is distributed greedily,
pairing rows that are short of S with columns that are short of S, which adds at
most 2n-1 fill entries.

The padding fraction is reported by `regularize` and is worth quoting alongside
any result: on these traces it ranges from ~4% (Mixtral 8x22B, throughput bench,
which is nearly row-regular already) to ~50% (DeepSeek, MMLU, where one rank
emits 29x fewer tokens than another).  A large padding fraction means the
decomposition is scheduling mostly idle time, which is a finding about the
workload, not an artefact of our method.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["MoETrace", "load_trace", "load_all", "regularize", "coarsen"]


@dataclass(frozen=True)
class MoETrace:
    """One measured expert all-to-all matrix, plus the metadata that identifies it."""

    matrix: NDArray[np.int64]  # (n, n), matrix[i, j] = tokens from rank i to rank j
    model: str
    dataset: str
    pass_idx: int
    layer_id: int
    path: str

    @property
    def n(self) -> int:
        return self.matrix.shape[0]

    @property
    def label(self) -> str:
        """Short human-readable name, e.g. 'Mixtral-8x22B / mmlu / L5'."""
        model = self.model.replace("-v0.1", "").replace("-Instruct", "")
        data = "mmlu" if "mmlu" in self.dataset else "throughput"
        return f"{model} / {data} / L{self.layer_id}"


def load_trace(path: str) -> MoETrace:
    """Read one trace CSV into an (n, n) integer demand matrix."""
    df = pd.read_csv(path)
    dest = [c for c in df.columns if re.fullmatch(r"dest_rank_\d+", c)]
    dest.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    df = df.sort_values("source_rank")

    matrix = df[dest].to_numpy(dtype=np.int64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{os.path.basename(path)}: {matrix.shape[0]} source ranks but "
            f"{matrix.shape[1]} destination columns -- expected a square matrix"
        )
    if (matrix < 0).any():
        raise ValueError(f"{os.path.basename(path)}: negative token counts")

    first = df.iloc[0]
    return MoETrace(
        matrix=matrix,
        model=str(first["model"]),
        dataset=str(first["dataset"]),
        pass_idx=int(first["pass_idx"]),
        layer_id=int(first["layer_id"]),
        path=path,
    )


def load_all(directory: str) -> list[MoETrace]:
    """Load every trace CSV in `directory`, sorted by filename."""
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no CSV traces found in {directory}")
    return [load_trace(p) for p in paths]


def regularize(matrix: NDArray[np.int64]) -> tuple[NDArray[np.int64], int, float]:
    """Pad `matrix` with slack until every row and column sums to the same S.

    Returns (padded, S, padding_fraction), where padding_fraction is the share of
    the padded matrix's mass that is slack rather than measured traffic.  S is the
    smallest value for which an S-regular completion exists, so the padding is
    minimal.
    """
    matrix = np.asarray(matrix, dtype=np.int64)
    n = matrix.shape[0]
    S = int(max(matrix.sum(axis=1).max(), matrix.sum(axis=0).max()))

    padded = matrix.copy()
    row_deficit = S - padded.sum(axis=1)
    col_deficit = S - padded.sum(axis=0)

    # Total row deficit equals total column deficit (both are n*S - sum(M)), so
    # greedily matching any short row against any short column always terminates.
    i = j = 0
    while i < n and j < n:
        if row_deficit[i] == 0:
            i += 1
            continue
        if col_deficit[j] == 0:
            j += 1
            continue
        fill = min(row_deficit[i], col_deficit[j])
        padded[i, j] += fill
        row_deficit[i] -= fill
        col_deficit[j] -= fill

    assert (padded.sum(axis=1) == S).all() and (padded.sum(axis=0) == S).all()
    assert (padded >= matrix).all()

    slack = int(padded.sum() - matrix.sum())
    return padded, S, slack / float(padded.sum())


def coarsen(matrix: NDArray[np.int64], S_target: int) -> tuple[NDArray[np.int64], int]:
    """Rescale an S-regular matrix down to approximately `S_target`, staying regular.

    Token counts run to tens of thousands, which makes S -- and therefore the
    cycle length -- far larger than the switch models we compare against.  Dividing
    entrywise would break regularity, so we divide, floor, and then re-pad the
    rounding loss with `regularize`.  Returns (coarsened, S_actual); S_actual is
    close to but not exactly S_target because of the re-padding.

    Use this only to put a trace on a comparable scale; it perturbs the demand and
    should not be used when reporting the trace's own numbers.
    """
    matrix = np.asarray(matrix, dtype=np.int64)
    S = int(matrix.sum(axis=1).max())
    if S_target >= S:
        return matrix, S
    scaled = (matrix * S_target) // S
    padded, S_actual, _ = regularize(scaled)
    return padded, S_actual
