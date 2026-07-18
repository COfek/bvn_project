# BvN + Euler (depths 1-4) with median/max step strategies, all four leaf engines.
# (min strategy already covered by euler_exp/eng_*_dense_d4.) Detached.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_euler_strategies_detached.ps1' -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project'
$py = '.\.venv\Scripts\python.exe'
foreach ($strat in @('median','max')) {
    foreach ($eng in @('maximum','heavy','heavy_static','wfa')) {
        & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
            --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method heuristic `
            --step-strategy $strat --random-seed 42 --no-plot `
            -o "euler_exp/strat_${strat}_${eng}" `
            *>> "euler_exp/detached_strat_${strat}_${eng}.log"
    }
}
New-Item -ItemType File 'euler_exp/STRAT_DONE.marker' -Force | Out-Null
