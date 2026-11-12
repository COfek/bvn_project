# Fourth part of the 2026-11 rerun (2026-11-12): WWFA added to the engine lists of the
# workload campaigns (tm_workload_experiment.py, moe_trace_experiment.py,
# radix_scale_campaign.py), so Figure 3 and Tables 5-6 can show it. All five engines run
# again in the same process, so every line of Figure 3 comes from one run.
#
# THE COMPUTER MUST BE FREE FOR THE WHOLE RUN. Start it only once Ofek has confirmed that.
#
# Output: run/rerun_2026_11d/{tm_workload,moe_workload,radix_scale}/, campaign.log,
#         environment.json, memtrace.csv, FAILED.txt (if any), DONE.marker.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_workloads_2026_11d.ps1' -WindowStyle Hidden
param([string]$Root = 'run/rerun_2026_11d', [string]$StartAt = '', [switch]$Resume)
$ErrorActionPreference = 'Continue'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001u) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
if ($StartAt) {
    $when = [datetime]::Parse($StartAt)
    while ((Get-Date) -lt $when) { Start-Sleep -Seconds 30 }
}
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

$t0 = Get-Date
$env:BVN_RUN_ROOT = $root
Step "workload tm" "$root/workload_tm.log" { & $py agent_scripts/tm_workload_experiment.py }
Step "workload moe" "$root/workload_moe.log" { & $py agent_scripts/moe_trace_experiment.py }
Step "workload radix_scale" "$root/workload_radix_scale.log" { & $py agent_scripts/radix_scale_campaign.py }

Say ("=== all campaigns finished in {0:N2} hours ===" -f ((Get-Date) - $t0).TotalHours)
if ($failed.Count) {
    $failed | Set-Content "$root/FAILED.txt"
    Say "FAILED: $($failed.Count) step(s), see FAILED.txt"
} else {
    New-Item -ItemType File "$root/DONE.marker" -Force | Out-Null
}
New-Item -ItemType File "$root/FINISHED.marker" -Force | Out-Null
Wait-Job $sampler -Timeout 180 | Out-Null
