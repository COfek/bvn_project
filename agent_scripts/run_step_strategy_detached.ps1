# Step-strategy experiment: min/median/max step size across Radix bases 2-16,
# all four engines, 105 matrices dense, seed 42. Detached (survives session).
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_step_strategy_detached.ps1' -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project'
$py = '.\.venv\Scripts\python.exe'
foreach ($eng in @('max','heavy','heavy_static','wfa')) {
    & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine $eng `
        --radix-bases 2 4 8 16 --step-strategy all `
        --random-seed 42 --no-plot -o "run/step_strategy_2026/$eng" `
        *>> "run/step_strategy_2026/detached_$eng.log"
}
New-Item -ItemType File 'run/step_strategy_2026/STEP_DONE.marker' -Force | Out-Null
