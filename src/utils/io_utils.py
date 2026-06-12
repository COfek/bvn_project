from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.utils.stats import DecompositionStats


def write_stats_to_csv(stats_list: List[DecompositionStats], path: Path) -> None:
    """Writes the list of decomposition stats to a CSV file."""
    if not stats_list:
        return

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Collect all unique keys from all stats
        all_keys = set()
        for s in stats_list:
            all_keys.update(s.radix_multi_results.keys())
        sorted_keys = sorted(list(all_keys))

        header = ["matrix_index", "bvn_perms", "bvn_cycle", "bvn_runtime", "bvn_matching"]
        for key in sorted_keys:
            # key is like "wfa_2", "heavy_8" or "euler_heuristic_depth2".
            # "_split" is the sequential split-phase time; only Euler
            # framework keys populate it (None for radix/plain keys).
            header += [f"{key}_time", f"{key}_cycle", f"{key}_perms",
                       f"{key}_planes", f"{key}_split"]

        writer.writerow(header)

        for s in stats_list:
            row = [s.matrix_index, s.num_permutations_bvn, s.cycle_length_bvn, s.runtime_bvn,
                   s.bvn_matching]
            for key in sorted_keys:
                if key in s.radix_multi_results:
                    res = s.radix_multi_results[key]
                    # Pad to 5 fields (older tuples lack planes and/or split)
                    row += list(res) + [None] * (5 - len(res))
                else:
                    row += [None, None, None, None, None]
            writer.writerow(row)


def parse_stats_from_csv(csv_path: Path) -> List[DecompositionStats]:
    """Reconstructs stats objects from CSV for re-plotting."""
    stats_list = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["matrix_index"])
            
            # BVN
            bvn_perms = row.get("bvn_perms")
            bvn_cycle = row.get("bvn_cycle")
            bvn_runtime = row.get("bvn_runtime")
            
            # Optional column added later; older CSVs won't have it.
            bvn_matching_raw = row.get("bvn_matching")
            bvn_matching = (
                bvn_matching_raw
                if bvn_matching_raw and bvn_matching_raw not in ("", "None")
                else None
            )

            ds = DecompositionStats(
                matrix_index=idx,
                num_permutations_bvn=int(float(bvn_perms)) if bvn_perms and float(bvn_perms) > 0 else 0,
                cycle_length_bvn=float(bvn_cycle) if bvn_cycle and bvn_cycle != "" and bvn_cycle != "None" else None,
                runtime_bvn=float(bvn_runtime) if bvn_runtime and bvn_runtime != "" and bvn_runtime != "None" else None,
                bvn_matching=bvn_matching,
            )
            
            # Dynamic keys
            seen_engines = set()
            # row.keys() might fail if header is not parsed correctly, but DictReader handles it
            for col in row.keys(): 
                if col and col.endswith("_time"):
                    seen_engines.add(col[:-5])
            
            for key in seen_engines:
                t = row.get(f"{key}_time")
                c = row.get(f"{key}_cycle")
                p = row.get(f"{key}_perms")
                pl = row.get(f"{key}_planes")  # Optional; older CSVs won't have this column
                sp = row.get(f"{key}_split")   # Optional; only Euler keys / newer CSVs

                if t and c and p and t != "None":
                    if pl and pl != "None" and pl != "":
                        planes = int(float(pl))
                    else:
                        planes = 0  # sentinel: plane count unknown for legacy CSVs
                    if sp and sp != "None" and sp != "":
                        ds.radix_multi_results[key] = (
                            float(t), float(c), int(float(p)), planes, float(sp))
                    else:
                        ds.radix_multi_results[key] = (
                            float(t), float(c), int(float(p)), planes)
            
            stats_list.append(ds)
    return stats_list
