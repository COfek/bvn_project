from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Optional

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

        header = ["matrix_index", "bvn_perms", "bvn_cycle", "bvn_runtime"]
        for key in sorted_keys:
            # key is like "wfa_2" or "heavy_8"
            header += [f"{key}_time", f"{key}_cycle", f"{key}_perms"]

        writer.writerow(header)

        for s in stats_list:
            row = [s.matrix_index, s.num_permutations_bvn, s.cycle_length_bvn, s.runtime_bvn]
            for key in sorted_keys:
                if key in s.radix_multi_results:
                    row += list(s.radix_multi_results[key])
                else:
                    row += [None, None, None]
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
            
            ds = DecompositionStats(
                matrix_index=idx,
                num_permutations_bvn=int(float(bvn_perms)) if bvn_perms and float(bvn_perms) > 0 else 0,
                cycle_length_bvn=float(bvn_cycle) if bvn_cycle and bvn_cycle != "" and bvn_cycle != "None" else None,
                runtime_bvn=float(bvn_runtime) if bvn_runtime and bvn_runtime != "" and bvn_runtime != "None" else None
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
                
                if t and c and p and t != "None":
                    ds.radix_multi_results[key] = (float(t), float(c), int(float(p)))
            
            stats_list.append(ds)
    return stats_list
