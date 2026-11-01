"""Peak memory of each overnight process, from euler_exp/ruined_tier/memtrace_overnight.csv.

The trace was sampled every 120 s, from 03:22 to the end of the run, by the loop that
agent_scripts/sample_memory.ps1 records (private bytes and working set of each python
process, and the machine's free physical memory, all in MiB; the machine has 31.2 GiB). A
peak here is the highest sample, so a lower bound on the true peak. Each
process is matched to its run by its start time: the campaigns start when the previous one
ends (progress.log), the first at the end of the paired run 1 and the paired run 2 after
the last campaign. Part 1 (the paired run 1) ended before the trace began.

Usage: python agent_scripts/summarise_memtrace.py
"""
import pandas as pd

from _paths import REPO  # repository-relative paths

DIR = REPO / "euler_exp" / "ruined_tier"


def main():
    t = pd.read_csv(DIR / "memtrace_overnight.csv", parse_dates=["time", "start"])
    ends = [l.split(" ", 1) for l in (DIR / "progress.log").read_text().splitlines() if " done" in l]
    starts = {}                     # start time -> run label, in order
    prev = pd.Timestamp(pd.read_json(DIR / "environment.json", typ="series")["end"])
    for ts, label in ends:
        starts[prev] = label.replace(" done", "")
        prev = pd.Timestamp(ts)
    starts[prev] = "paired run 2 (rep2)"
    keys = sorted(starts)

    def label(st):
        best = min(keys, key=lambda k: abs((st - k).total_seconds()))
        return starts[best] if abs((st - best).total_seconds()) < 60 else "unmatched"

    t["run"] = t.start.map(label)
    g = t.groupby("run", sort=False).agg(samples=("time", "size"), peak_private_mb=("priv_mb", "max"),
                                         peak_working_set_mb=("ws_mb", "max"), min_free_mb=("free_mb", "min"))
    for c in ("peak_private_mb", "peak_working_set_mb", "min_free_mb"):
        g[c.replace("_mb", "_gib")] = (g[c] / 1024).round(2)
    g.to_csv(DIR / "memtrace_summary.csv")
    print(g.to_string())


if __name__ == "__main__":
    main()
