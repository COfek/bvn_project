# Re-runs sweep 3 of the variance campaign on a quiet machine.
#
# The original sweep 3 ran 08:53-12:22 while the machine was in use, and it shows:
# it is 10 to 23 percent slower than the mean of sweeps 1 and 2 on the greedy
# engines, and the sweep took 209 minutes against 192 and 191. Hungarian barely
# moved, which fits contention rather than a real change: its solver is compute
# bound and holds a core, while the greedy scans lose cache and boost headroom to
# a competing process.
#
# Sweeps 1 and 2 ran overnight untouched and agree to 1 to 3 percent. This run
# replaces sweep 3 so all three are comparable.
#
# The old sweep 3 is moved aside rather than deleted, so the contaminated data
# stays available as evidence of what interference looks like.
#
# Launch: pwsh -NoProfile -File agent_scripts\run_sweep3_again.ps1

$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot   # the repository root, wherever it is cloned
Set-Location $Repo
$py   = (Join-Path $Repo '.venv\Scripts\python.exe')
$root = 'run/variance_2026'
$log  = "$root/sweep3_rerun.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}

function LoadNow {
    try {
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        return "cpu_load=${cpu}% processes=$((Get-Process).Count)"
    } catch { return "load unavailable" }
}

# the baseline job may still be running; never overlap, the measurement is wall clock
$deadline = (Get-Date).AddHours(3)
while (Test-Path "$root/BASELINES_RUNNING.marker") {
    if ((Get-Date) -gt $deadline) { Say "ABORT: baselines still running after 3 h"; exit 1 }
    Start-Sleep -Seconds 60
}

if (Test-Path "$root/sweep3") {
    Move-Item "$root/sweep3" "$root/sweep3_contaminated" -Force
    Say "old sweep 3 moved to sweep3_contaminated, kept as evidence"
}

Say "=== sweep 3 re-run starting === $(LoadNow)"
$t0 = Get-Date

foreach ($eng in @('max', 'heavy', 'heavy_static', 'wfa')) {
    $s = Get-Date
    Say "radix  $eng  starting"
    & $py main.py -n 256 -k 256 --max-weight 64 -s 1000 --engine $eng `
        --random-seed 42 --no-plot --radix-bases 2 4 8 16 `
        -o "$root/sweep3/radix/$eng" *>> "$root/sweep3_radix_${eng}_rerun.log"
    Say ("radix  $eng  done in {0:N1} min   {1}" -f ((Get-Date) - $s).TotalMinutes, (LoadNow))
}

foreach ($eng in @('maximum', 'heavy', 'heavy_static', 'wfa')) {
    $s = Get-Date
    Say "euler  $eng  starting"
    & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
        --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method heuristic `
        --random-seed 42 --no-plot -o "$root/sweep3/euler/$eng" `
        *>> "$root/sweep3_euler_${eng}_rerun.log"
    Say ("euler  $eng  done in {0:N1} min   {1}" -f ((Get-Date) - $s).TotalMinutes, (LoadNow))
}

# the three baselines the plain engine names do not compute
foreach ($pair in @(@('heavy','heavy_bvn'), @('heavy_static','heavy_static_bvn'), @('wfa','wfa_bvn'))) {
    $s = Get-Date
    Say "baseline  $($pair[0])  starting"
    & $py main.py -n 256 -k 256 --max-weight 64 -s 1000 --engine $pair[1] `
        --random-seed 42 --no-plot --radix-bases `
        -o "$root/sweep3/baseline/$($pair[0])" *>> "$root/sweep3_baseline_$($pair[0])_rerun.log"
    Say ("baseline  $($pair[0])  done in {0:N1} min" -f ((Get-Date) - $s).TotalMinutes)
}

Say ("=== sweep 3 re-run complete in {0:N2} hours ===" -f ((Get-Date) - $t0).TotalHours)
New-Item -ItemType File "$root/SWEEP3_RERUN_DONE.marker" -Force | Out-Null
