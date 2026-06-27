# Runs the remaining rerun engines sequentially, detached from any Claude session.
# Launch with:  Start-Process pwsh -ArgumentList '-File','agent_scripts\run_rerun_detached.ps1' -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project'
$py = '.\.venv\Scripts\python.exe'
$engines = @('heavy','heavy_static','wfa')
foreach ($eng in $engines) {
    & $py main.py -n 256 -k 256 --max-weight 64 -s 1000 --engine $eng `
        --random-seed 42 --no-plot --radix-bases 2 4 8 16 -o "run/rerun_2026/$eng" `
        *>> "run/rerun_2026/detached_$eng.log"
}
New-Item -ItemType File 'run/rerun_2026/ALL_DONE.marker' -Force | Out-Null
