# Three full sweeps of the paper's campaigns, to measure campaign-to-campaign spread.
#
# Every number in the paper is one campaign mean. Within a campaign the mean over
# 995 matrices is precise to about 0.15%, but three same-seed campaigns of identical
# work have differed by 6%: that variation is common to every matrix in a run, so
# averaging inside a run cannot see it. Only repeating the whole campaign can.
#
# Each sweep repeats the exact commands that produced run/rerun_2026 and
# euler_exp/eng_*_dense_d4, same seed and same sample sizes, so the three sweeps are
# directly comparable to each other and to the published numbers.
#
# Output: run/variance_2026/sweep{1,2,3}/...  and a per-sweep environment record.
# Launch:  pwsh -NoProfile -File agent_scripts\run_variance_campaign.ps1

$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot   # the repository root, wherever it is cloned
Set-Location $Repo
# relative to this script, so the task does not depend on the working directory Task Scheduler gives it
$py   = (Join-Path $Repo '.venv\Scripts\python.exe')
$root = 'run/variance_2026'
New-Item -ItemType Directory -Force -Path $root | Out-Null

$master = "$root/campaign.log"
function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $master -Value $line
}

# --- the measurement environment, which no previous campaign recorded ----------
& $py -c @"
import json, platform, sys, os
info = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'processor': platform.processor(), 'machine': platform.machine(),
        'cpu_count': os.cpu_count()}
for m in ('numpy', 'scipy', 'numba', 'pandas'):
    try:
        info[m] = __import__(m).__version__
    except Exception as e:
        info[m] = f'unavailable ({e})'
print(json.dumps(info, indent=2))
"@ > "$root/environment.json" 2>&1
Say "environment recorded -> $root/environment.json"

# --- load at each boundary, so contention shows up in the record --------------
function LoadNow {
    try {
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        $procs = (Get-Process).Count
        return "cpu_load=${cpu}% processes=${procs}"
    } catch { return "load unavailable" }
}

# --engine takes 'max' for Hungarian; 'maximum' is only valid for --euler-leaf-engine
$radixEngines = @('max', 'heavy', 'heavy_static', 'wfa')
$eulerEngines = @('maximum', 'heavy', 'heavy_static', 'wfa')

Say "=== variance campaign starting: 3 sweeps, 4 Radix + 4 Euler engines each ==="
Say (LoadNow)
$t0 = Get-Date

foreach ($sweep in 1..3) {
    Say "--- sweep $sweep of 3 begins --- $(LoadNow)"
    $sw = Get-Date

    foreach ($eng in $radixEngines) {
        $s = Get-Date
        Say "sweep $sweep  radix  $eng  starting"
        & $py main.py -n 256 -k 256 --max-weight 64 -s 1000 --engine $eng `
            --random-seed 42 --no-plot --radix-bases 2 4 8 16 `
            -o "$root/sweep$sweep/radix/$eng" `
            *>> "$root/sweep${sweep}_radix_${eng}.log"
        Say ("sweep $sweep  radix  $eng  done in {0:N1} min   {1}" -f `
             ((Get-Date) - $s).TotalMinutes, (LoadNow))
    }

    foreach ($eng in $eulerEngines) {
        $s = Get-Date
        Say "sweep $sweep  euler  $eng  starting"
        & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
            --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method heuristic `
            --random-seed 42 --no-plot -o "$root/sweep$sweep/euler/$eng" `
            *>> "$root/sweep${sweep}_euler_${eng}.log"
        Say ("sweep $sweep  euler  $eng  done in {0:N1} min   {1}" -f `
             ((Get-Date) - $s).TotalMinutes, (LoadNow))
    }

    Say ("--- sweep $sweep complete in {0:N1} min ---" -f ((Get-Date) - $sw).TotalMinutes)
    New-Item -ItemType File "$root/SWEEP${sweep}_DONE.marker" -Force | Out-Null
}

Say ("=== all three sweeps complete in {0:N2} hours ===" -f ((Get-Date) - $t0).TotalHours)
New-Item -ItemType File "$root/ALL_DONE.marker" -Force | Out-Null
