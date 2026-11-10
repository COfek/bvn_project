# Every campaign behind the paper's times, rerun under one timing method (2026-11-05).
#
# Code review 2026-11-04 (docs/code_review_2026-11-04.md, A1-A3), fixed in aadef99:
#   * the clock stops before the dense permutation matrices are built (it used to include
#     them: about half the recorded time for GW Static and WFA);
#   * the Euler split is timed one split at a time, a level charged its slowest split
#     (it used to be a threaded wall clock);
#   * every earlier result is freed before each timed call, and Radix holds only matchings
#     while it times later planes.
# N and C do not change (checked on campaign matrices against run/variance_2026).
#
# Same commands as run_variance_campaign.ps1, run_baselines_after_sweeps.ps1 and
# run_euler_freed_campaign.ps1: n=256 k=256 max_weight=64 seed 42; Radix and baselines on
# 1000 matrices, Euler on 105 at depths 1-4 with the three-tier heuristic; three sweeps.
# Then the workload campaigns (TM, MoE trace, scaled Radix) once each.
# Radix's digit extraction is now inside its recorded time: do NOT add
# radix_extraction.py's correction to these results.
#
# Output: run/rerun_2026_11/sweep{1,2,3}/{radix,baseline,euler}/<engine>/,
#         run/rerun_2026_11/{tm_workload,moe_workload,radix_scale}/,
#         campaign.log, environment.json, memtrace.csv, FAILED.txt (if any), DONE.marker.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_full_rerun_2026_11.ps1' -WindowStyle Hidden
# Smoke test: -StartAt now -RadixSamples 7 -EulerSamples 7 -Root run/_smoke -SkipWorkloads
# Resume after a pause: add -Resume. Every step that campaign.log already records as
# "done ... exit 0" is skipped; a step stopped part-way must first be moved out of $Root
# (to $Root/_aborted/), since it is rerun from the start.
param([string]$StartAt = '02:00', [int]$RadixSamples = 1000, [int]$EulerSamples = 105,
      [string]$Root = 'run/rerun_2026_11', [switch]$SkipWorkloads, [switch]$Resume)
$ErrorActionPreference = 'Continue'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001u) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$py   = (Join-Path $Repo '.venv\Scripts\python.exe')
$root = $Root
New-Item -ItemType Directory -Force -Path $root | Out-Null
$master = "$root/campaign.log"
$failed = @()

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $master -Value $line
}
function LoadNow {
    try {
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        return "cpu_load=${cpu}% processes=$((Get-Process).Count)"
    } catch { return "load unavailable" }
}
function Step($label, $log, [scriptblock]$cmd) {
    if ($Resume -and (Test-Path $master) -and
        (Select-String -Path $master -Pattern ("\] " + [regex]::Escape($label) + "  done in .* exit 0$") -Quiet)) {
        Say "$label  already done, skipped (resume)"
        return
    }
    Say "$label  starting  $(LoadNow)"
    $t = Get-Date
    & $cmd *>> $log
    $code = $LASTEXITCODE
    Say ("$label  done in {0:N1} min  exit $code" -f ((Get-Date) - $t).TotalMinutes)
    if ($code -ne 0) { $script:failed += "$label (exit $code, log $log)" }
}

$start = if ($StartAt -eq 'now') { Get-Date } else { [datetime]::Today.Add([timespan]::Parse($StartAt)) }
if ($start -lt (Get-Date).AddMinutes(-1)) { $start = $start.AddDays(1) }
Say "waiting until $($start.ToString('s'))"
while ((Get-Date) -lt $start) { Start-Sleep -Seconds 30 }

$envFile = if ($Resume) { "$root/environment_resume_$(Get-Date -Format 'yyyyMMdd_HHmm').json" } else { "$root/environment.json" }
& $py -c @"
import json, platform, sys, os, subprocess
git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
info = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'processor': platform.processor(), 'machine': platform.machine(),
        'cpu_count': os.cpu_count(), 'commit': git('rev-parse', '--short', 'HEAD'),
        'dirty_files': [l for l in git('status', '--porcelain', '--', 'src', 'main.py', 'agent_scripts').splitlines()]}
for m in ('numpy', 'scipy', 'numba', 'pandas'):
    try:
        info[m] = __import__(m).__version__
    except Exception as e:
        info[m] = f'unavailable ({e})'
print(json.dumps(info, indent=2))
"@ > $envFile 2>&1
Say "environment recorded -> $envFile"

$sampler = Start-Job -ScriptBlock {
    param($dir)
    $log = Join-Path $dir 'memtrace.csv'
    if (-not (Test-Path $log)) { 'time,pid,start,cpu_s,ws_mb,priv_mb,free_mb' | Set-Content $log }
    while (-not (Test-Path (Join-Path $dir 'FINISHED.marker'))) {
        $f = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
        Get-Process python -ErrorAction SilentlyContinue |
            Where-Object { $_.PrivateMemorySize64 -gt 50MB } |
            ForEach-Object { '{0},{1},{2},{3},{4},{5},{6}' -f (Get-Date -Format s), $_.Id,
                $_.StartTime.ToString('s'), [int]$_.CPU, [int]($_.WorkingSet64 / 1MB),
                [int]($_.PrivateMemorySize64 / 1MB), $f } | Add-Content $log
        Start-Sleep -Seconds 120
    }
} -ArgumentList (Resolve-Path $root).Path

# --engine takes 'max' for Hungarian; 'maximum' is only valid for --euler-leaf-engine
$radixEngines = @('max', 'heavy', 'heavy_static', 'wfa')
$baselineEngines = @('heavy', 'heavy_static', 'wfa')      # Hungarian's baseline comes with its Radix run
$eulerEngines = @('maximum', 'heavy', 'heavy_static', 'wfa')
$t0 = Get-Date
foreach ($sweep in 1..3) {
    Say "--- sweep $sweep of 3 ---"
    foreach ($eng in $radixEngines) {
        Step "sweep $sweep radix $eng" "$root/sweep${sweep}_radix_${eng}.log" {
            & $py main.py -n 256 -k 256 --max-weight 64 -s $RadixSamples --engine $eng `
                --random-seed 42 --no-plot --radix-bases 2 4 8 16 -o "$root/sweep$sweep/radix/$eng" }
    }
    foreach ($eng in $baselineEngines) {
        Step "sweep $sweep baseline $eng" "$root/sweep${sweep}_baseline_${eng}.log" {
            & $py main.py -n 256 -k 256 --max-weight 64 -s $RadixSamples --engine "${eng}_bvn" `
                --random-seed 42 --no-plot --radix-bases -o "$root/sweep$sweep/baseline/$eng" }
    }
    foreach ($eng in $eulerEngines) {
        Step "sweep $sweep euler $eng" "$root/sweep${sweep}_euler_${eng}.log" {
            & $py main.py -n 256 -k 256 --max-weight 64 -s $EulerSamples --engine euler_bvn `
                --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method heuristic `
                --random-seed 42 --no-plot -o "$root/sweep$sweep/euler/$eng" }
    }
}

if (-not $SkipWorkloads) {
$env:BVN_RUN_ROOT = $root
Step "workload tm" "$root/workload_tm.log" { & $py agent_scripts/tm_workload_experiment.py }
Step "workload moe" "$root/workload_moe.log" { & $py agent_scripts/moe_trace_experiment.py }
Step "workload radix_scale" "$root/workload_radix_scale.log" { & $py agent_scripts/radix_scale_campaign.py }
}

Say ("=== all campaigns finished in {0:N2} hours ===" -f ((Get-Date) - $t0).TotalHours)
if ($failed.Count) {
    $failed | Set-Content "$root/FAILED.txt"
    Say "FAILED: $($failed.Count) step(s), see FAILED.txt"
} else {
    New-Item -ItemType File "$root/DONE.marker" -Force | Out-Null
}
New-Item -ItemType File "$root/FINISHED.marker" -Force | Out-Null
Wait-Job $sampler -Timeout 180 | Out-Null
