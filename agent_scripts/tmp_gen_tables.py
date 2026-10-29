import pandas as pd
import glob
from pathlib import Path

def main():
    runs = glob.glob("summary/*/*")
    output = []
    
    output.append("## Detailed Per-Simulation Base Breakdowns\n")
    output.append("The tables below detail the performance of the baseline BVN vs. Radix bases (2, 4, 8, 16) across the 16 simulations. **Avg Runtime** is the average time to process a single matrix. **Avg Permutations** and **Avg Cycle Length** average exactly how many discrete matrices the output was factored into, and the resulting cycle bounds for the specific method.\n")
    
    runs = sorted(runs)
    
    for r in runs:
        rp = Path(r)
        csv_path = rp / "results_stats.csv"
        if not csv_path.exists(): continue
        df = pd.read_csv(csv_path)
        
        parent_name = rp.parent.name
        
        output.append(f"### {parent_name}")
        output.append("| Method | Avg Runtime (s) | Avg Permutations | Avg Cycle Length |")
        output.append("| :--- | :--- | :--- | :--- |")
        
        bvn_rt = df['bvn_runtime'].mean() if 'bvn_runtime' in df.columns else 0
        bvn_p = df['bvn_perms'].mean() if 'bvn_perms' in df.columns else 0
        bvn_c = df['bvn_cycle'].mean() if 'bvn_cycle' in df.columns else 0
        output.append(f"| **BVN** | {bvn_rt:.2f}s | {bvn_p:.1f} | {bvn_c:.1f} |")
        
        engine_prefix = None
        for col in df.columns:
            if col.endswith('_16_time'):
                engine_prefix = col.replace('_16_time', '')
                break
                
        if engine_prefix:
            for base in [2, 4, 8, 16]:
                time_col = f"{engine_prefix}_{base}_time"
                cycle_col = f"{engine_prefix}_{base}_cycle"
                perm_col = f"{engine_prefix}_{base}_perms"
                
                if time_col in df.columns:
                    rt = df[time_col].mean()
                    c = df[cycle_col].mean()
                    p = df[perm_col].mean()
                    output.append(f"| **Base {base}** | {rt:.2f}s | {p:.1f} | {c:.1f} |")
            
        output.append("\n<br>\n")

    with open("base_breakdowns.md", "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    main()
