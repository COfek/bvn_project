# Fills in the BvN baseline that the variance campaign missed.
#
# The runner computes the baseline only for engines in a gate list in
# src/runner.py: "maximum" is in it, plain "heavy", "heavy_static" and "wfa" are
# not, but their "_bvn" variants are. The variance campaign used the plain names,
# copied from run_rerun_detached.ps1, so Hungarian got its baseline (--engine max
# maps to "maximum") and the other three did not.
#
# This runs baseline only, no Radix sweep: passing --radix-bases with no values
# gives an empty list, so each engine is one decomposition per matrix. That is
# minutes per engine rather than the ~50 the full sweep costs.
#
# It waits for the variance campaign to finish first, so the two never contend
# for the machine: wall-clock timings are the measurement here.
#
# Launch: pwsh -NoProfile -File agent_scripts\run_baselines_after_sweeps.ps1

$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project'
$py   = 'C:\Users\ofekc\Desktop\Msc\Thesis\bvn_project\.venv\Scripts\python.exe'
$root = 'run/variance_2026'
$log  = "$root/baselines.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}

# --- wait for the sweeps, but not forever -----------------------------------
$deadline = (Get-Date).AddHours(6)
Say "waiting for $root/ALL_DONE.marker before starting"
while (-not (Test-Path "$root/ALL_DONE.marker")) {
    if ((Get-Date) -gt $deadline) {
        Say "ABORT: sweeps did not finish within 6 hours; not starting, to avoid contending"
        exit 1
    }
    Start-Sleep -Seconds 60
}
Say "sweeps finished; starting baseline runs"

# engine name -> the variant that also computes the BvN baseline
$engines = @{ 'heavy' = 'heavy_bvn'; 'heavy_static' = 'heavy_static_bvn'; 'wfa' = 'wfa_bvn' }

$t0 = Get-Date
foreach ($sweep in 1..3) {
    foreach ($name in $engines.Keys) {
        $variant = $engines[$name]
        $s = Get-Date
        Say "sweep $sweep  baseline  $name  (--engine $variant)  starting"
        & $py main.py -n 256 -k 256 --max-weight 64 -s 1000 --engine $variant `
            --random-seed 42 --no-plot --radix-bases `
            -o "$root/sweep$sweep/baseline/$name" `
            *>> "$root/sweep${sweep}_baseline_${name}.log"
        Say ("sweep $sweep  baseline  $name  done in {0:N1} min" -f ((Get-Date) - $s).TotalMinutes)
    }
}
Say ("=== baselines complete in {0:N2} hours ===" -f ((Get-Date) - $t0).TotalHours)
New-Item -ItemType File "$root/BASELINES_DONE.marker" -Force | Out-Null
