import csv
import sys
from pathlib import Path
from collections import defaultdict

def summarize(csv_path):
    print(f"Reading {csv_path}")
    data = defaultdict(list)
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    data[key].append(float(val))
                except ValueError:
                    pass
                    
    print("\n--- Summary Statistics (Mean) ---")
    keys = sorted(data.keys())
    for key in keys:
        if "count" in key or "num" in key or "time" in key or "cycle" in key:
            values = data[key]
            if values:
                avg = sum(values) / len(values)
                print(f"{key}: {avg:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        summarize(sys.argv[1])
    else:
        print("Usage: python summarize.py <csv_file>")
