import pandas as pd
import sys

def analyze_csv(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Analysis for {filepath}")
    print("-" * 40)
    
    bases = [2, 4, 8, 16]
    
    # BVN Baseline
    if 'bvn_perms' in df.columns:
        print(f"BVN Mean Perms: {df['bvn_perms'].mean():.2f}")
        print(f"BVN Mean Runtime: {df['bvn_runtime'].mean():.4f}s")
        if 'bvn_cycle' in df.columns:
            print(f"BVN Mean Cycle: {df['bvn_cycle'].mean():.2f}")
    print("-" * 40)

    for base in bases:
        perm_col = f'wfa_{base}_perms'
        time_col = f'wfa_{base}_time'
        cyc_col = f'wfa_{base}_cycle'
        
        if perm_col in df.columns:
            avg_perms = df[perm_col].mean()
            avg_time = df[time_col].mean()
            
            print(f"Base {base}:")
            print(f"  Avg Perms: {avg_perms:.2f}")
            print(f"  Avg Time:  {avg_time:.4f}s")
            
            if cyc_col in df.columns:
                avg_cycle = df[cyc_col].mean()
                print(f"  Avg Cycle Sum: {avg_cycle:.2f}")
                # Cycle length is usually total weight sum? 
                # If cycle_len ~ 15 for Base 16, and avg perms ~ 20...
                # It means per-perm weight is < 1? No.
                # Let's see the numbers.
        else:
            print(f"Base {base} data not found.")
            
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_csv(sys.argv[1])
    else:
        print("Please provide csv path")
