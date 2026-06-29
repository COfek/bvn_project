# Runs the missing GW Static (heavy_static) Euler experiments (dense + sparse),
# matching the params of the existing eng_* Euler runs. Detached from any session.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_euler_static_detached.ps1' -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project'
$py = '.\.venv\Scripts\python.exe'

# dense (k=256)
& $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
    --euler-leaf-engine heavy_static --euler-depths 1 2 3 --euler-split-method heuristic `
    --random-seed 42 --no-plot -o 'euler_exp/eng_heavy_static_dense' `
    *>> 'euler_exp/detached_static_dense.log'

# sparse (k=16)
& $py main.py -n 256 -k 16 --max-weight 64 -s 105 --engine euler_bvn `
    --euler-leaf-engine heavy_static --euler-depths 1 2 3 --euler-split-method heuristic `
    --random-seed 42 --no-plot -o 'euler_exp/eng_heavy_static_sparse' `
    *>> 'euler_exp/detached_static_sparse.log'

New-Item -ItemType File 'euler_exp/STATIC_DONE.marker' -Force | Out-Null
