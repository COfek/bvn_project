import csv
import statistics
from pathlib import Path
import sys

def summarize_csv(file_path, k=1):
    print(f"Reading results from: {file_path}")
    
    data = {} # key -> list of values
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fields = reader.fieldnames
            
            for field in fields:
                if field != "matrix_index":
                    data[field] = []
            
            count = 0
            for row in reader:
                count += 1
                for field in fields:
                    if field != "matrix_index" and row[field]:
                        try:
                            val = float(row[field])
                            data[field].append(val)
                        except ValueError:
                            pass
                            
            print(f"Processed {count} samples.\n")
            
            # Group by method
            methods = set()
            for field in fields:
                if field.endswith("_time"):
                    method = field[:-5] # remove _time
                    methods.add(method)
                elif field.endswith("_runtime"):
                    method = field[:-8] # remove _runtime
                    methods.add(method)
            
            # Print table
            print(f"{'Method':<20} | {'Runtime (s)':<15} | {'Permutations':<15} | {'Cycle Length':<15}")
            print("-" * 75)
            
            # Sort methods: BVN first, then Split, then Radix (sorted by base)
            sorted_methods = []
            radix_methods = []
            
            if "bvn" in methods: sorted_methods.append("bvn")
            if "split" in methods: sorted_methods.append("split")
            
            for m in methods:
                if m not in ["bvn", "split"]:
                    radix_methods.append(m)
            
            # Sort radix by base if possible (e.g. maximum_2)
            def radix_sort_key(s):
                try:
                    return int(s.split('_')[-1])
                except:
                    return 999
            radix_methods.sort(key=radix_sort_key)
            
            sorted_methods.extend(radix_methods)
            
            for method in sorted_methods:
                if method == "split": continue 
                
                # Try multiple key variants
                runtime_keys = [f"{method}_time", f"{method}_runtime", f"runtime_{method}"]
                perm_keys = [f"{method}_perms", f"{method}_permutations", f"permutations_{method}", f"num_permutations_{method}"]
                if method == "bvn": perm_keys.append("bvn_perms")
                cycle_keys = [f"{method}_cycle", f"{method}_cycle_length", f"cycle_length_{method}"]
                
                runtimes = []
                for key in runtime_keys:
                    if key in data: runtimes = data[key]; break
                
                perms = []
                for key in perm_keys:
                    if key in data: perms = data[key]; break
                    
                cycles = []
                for key in cycle_keys:
                    if key in data: cycles = data[key]; break
                
                if not runtimes: continue
                
                avg_runtime = statistics.mean(runtimes)
                avg_perm = statistics.mean(perms)
                avg_cycle = statistics.mean(cycles) / k
                
                print(f"{method:<20} | {avg_runtime:<15.4f} | {avg_perm:<15.1f} | {avg_cycle:<15.1f}")

    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", default="run/20260215_101407/results_stats.csv")
    parser.add_argument("--k", type=int, default=1, help="Normalization factor for cycle length")
    args = parser.parse_args()
    
    summarize_csv(args.file_path, k=args.k)
