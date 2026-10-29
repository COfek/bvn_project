import subprocess
import sys

def main():
    engines = ["wfa_bvn", "heavy_bvn", "heavy_static_bvn", "maximum_bvn"]
    generators = ["standard", "binary", "sinkhorn", "weighted"]
    
    python_exe = ".venv\\Scripts\\python.exe"
    
    for engine in engines:
        for generator in generators:
            if generator == "sinkhorn":
                size = 128
                samples = 100
            else:
                size = 256
                samples = 1000
                
            out_folder = f"summary/{engine}_{generator}_{samples}_{size}"
            
            cmd = [
                python_exe, "main.py",
                "--engine", engine,
                "--generator", generator,
                "--size", str(size),
                "--samples", str(samples),
                "--radix-bases", "2", "4", "8", "16",
                "--output", out_folder
            ]
            
            # Weighted generator expects weights, so applying the ones used in previous tests.
            if generator == "weighted":
                cmd.extend(["--weights", "1", "16", "--sub-k", "10"])
                
            print(f"\n{'='*60}")
            print(f"Starting Simulation: Engine={engine}, Generator={generator}")
            print(f"Command: {' '.join(cmd)}")
            print(f"{'='*60}\n")
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running simulation: {e}")
                sys.exit(1)
                
    print("\nAll simulations completed successfully.")

if __name__ == "__main__":
    main()
