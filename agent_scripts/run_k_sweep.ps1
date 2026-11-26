# K17 (Chen, 2026-11-23 meeting): "we can check with higher and lower k". The main campaign uses
# k = 256 permutations per matrix; this one repeats it at k = 64, 128, 512 and 1024, everything else
# unchanged: n=256 max_weight=64 seed 42, Radix (bases 2 4 8 16) with the Hungarian baseline, the
# unsplit baselines of the other engines, and Euler (heuristic_ruined, depths 1-4), for the four
# engines of the paper (Hungarian, GW Dynamic, GW Static, WWFA); 105 matrices per k (5 warm-up),
# one sweep. k runs from small to large, so the cheap values finish first.
#
# THE COMPUTER MUST BE FREE FOR THE WHOLE RUN. Ofek, 2026-11-26: "do it at 3 am".
#
# Output: run/k_sweep_2026_11/k{K}/{radix,baseline,euler}/<engine>/,
#         campaign.log, environment.json, memtrace.csv, FAILED.txt (if any), DONE.marker.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_k_sweep.ps1' -WindowStyle Hidden
# Smoke test: -StartAt now -Samples 7 -Ks 64 -Root run/_smoke_k
# Resume after a pause: add -Resume (steps campaign.log records as "done ... exit 0" are skipped).
param([string]$StartAt = '03:00', [int]$Samples = 105, [int[]]$Ks = @(64, 128, 512, 1024),
      [string]$Root = 'run/k_sweep_2026_11', [switch]$Resume)
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

function RadixStep($k, $eng) {
    Step "k $k radix $eng" "$root/k${k}_radix_${eng}.log" {
        & $py main.py -n 256 -k $k --max-weight 64 -s $Samples --engine $eng `
            --random-seed 42 --no-plot --radix-bases 2 4 8 16 -o "$root/k$k/radix/$eng" }
}
function BaselineStep($k, $eng) {
    Step "k $k baseline $eng" "$root/k${k}_baseline_${eng}.log" {
        & $py main.py -n 256 -k $k --max-weight 64 -s $Samples --engine "${eng}_bvn" `
            --random-seed 42 --no-plot --radix-bases -o "$root/k$k/baseline/$eng" }
}
function EulerStep($k, $eng) {
    Step "k $k euler $eng" "$root/k${k}_euler_${eng}.log" {
        & $py main.py -n 256 -k $k --max-weight 64 -s $Samples --engine euler_bvn `
            --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method heuristic_ruined `
            --random-seed 42 --no-plot -o "$root/k$k/euler/$eng" }
}

# --engine takes 'max' for Hungarian, whose baseline comes with its Radix run;
# --euler-leaf-engine takes 'maximum'
$t0 = Get-Date
foreach ($k in $Ks) {
    Say "--- k = $k ---"
    foreach ($eng in @('max', 'heavy', 'heavy_static', 'wwfa')) { RadixStep $k $eng }
    foreach ($eng in @('heavy', 'heavy_static', 'wwfa')) { BaselineStep $k $eng }
    foreach ($eng in @('maximum', 'heavy', 'heavy_static', 'wwfa')) { EulerStep $k $eng }
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
