# Runs Euler depth 4 (dense) for all four leaf engines, matching the existing
# eng_*_dense Euler params. Detached so it survives session teardown.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_euler_d4_detached.ps1' -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project'
$py = '.\.venv\Scripts\python.exe'
foreach ($eng in @('maximum','heavy','heavy_static','wfa')) {
    & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
        --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method heuristic `
        --random-seed 42 --no-plot -o "euler_exp/eng_${eng}_dense_d4" `
        *>> "euler_exp/detached_${eng}_d4.log"
}
New-Item -ItemType File 'euler_exp/D4_DONE.marker' -Force | Out-Null
