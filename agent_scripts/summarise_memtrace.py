"""Peak memory of each overnight process, from euler_exp/ruined_tier/memtrace_overnight.csv.

The trace was sampled every 120 s, from 03:20 to the end of the run, with
  Get-Process python | Where-Object PrivateMemorySize64 -gt 50MB
(private bytes and working set in MB, and the machine's free physical memory). Each
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
    g.to_csv(DIR / "memtrace_summary.csv")
    print(g.to_string())


if __name__ == "__main__":
    main()
