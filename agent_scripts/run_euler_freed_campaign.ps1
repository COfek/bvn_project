# The paper's Euler campaigns, rerun with every earlier result freed before each timed call
# (R12, 2026-11-02). The runner frees by default since d9ea165; the flag is passed anyway so
# the command records it.
#
# Replaces the Euler half of run/variance_2026: those campaigns held the baseline's and the
# previous depth's results while timing, which inflated GW Static's and WFA's Euler times
# (euler_exp/ruined_tier/REPORT.md, section 4). Same commands otherwise: n=256 k=256
# max_weight=64, 105 matrices from seed 42, depths 1-4, the three-tier heuristic, four
# leaf engines, three sweeps. N and C are deterministic and must match variance_2026.
#
# Output: run/variance_freed_2026/sweep{1,2,3}/euler/<engine>/, campaign.log,
# environment.json, memtrace.csv, DONE.marker.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_euler_freed_campaign.ps1' -WindowStyle Hidden

$ErrorActionPreference = 'Continue'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$py   = (Join-Path $Repo '.venv\Scripts\python.exe')
$root = 'run/variance_freed_2026'
New-Item -ItemType Directory -Force -Path $root | Out-Null

$master = "$root/campaign.log"
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

& $py -c @"
import json, platform, sys, os, subprocess
info = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'processor': platform.processor(), 'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
        'commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                 capture_output=True, text=True).stdout.strip()}
for m in ('numpy', 'scipy', 'numba', 'pandas'):
    try:
        info[m] = __import__(m).__version__
    except Exception as e:
        info[m] = f'unavailable ({e})'
print(json.dumps(info, indent=2))
"@ > "$root/environment.json" 2>&1
Say "environment recorded -> $root/environment.json"

# memory sampler, in the background, until DONE.marker
$sampler = Start-Job -ScriptBlock {
    param($dir)
    $log = Join-Path $dir 'memtrace.csv'
    'time,pid,start,cpu_s,ws_mb,priv_mb,free_mb' | Set-Content $log
    while (-not (Test-Path (Join-Path $dir 'DONE.marker'))) {
        $f = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
        Get-Process python -ErrorAction SilentlyContinue |
            Where-Object { $_.PrivateMemorySize64 -gt 50MB } |
            ForEach-Object { '{0},{1},{2},{3},{4},{5},{6}' -f (Get-Date -Format s), $_.Id,
                $_.StartTime.ToString('s'), [int]$_.CPU, [int]($_.WorkingSet64 / 1MB),
                [int]($_.PrivateMemorySize64 / 1MB), $f } | Add-Content $log
        Start-Sleep -Seconds 120
    }
} -ArgumentList (Resolve-Path $root).Path

$engines = @('maximum', 'heavy', 'heavy_static', 'wfa')
Say "=== freed Euler campaign starting: 3 sweeps x 4 engines ==="
foreach ($sweep in 1, 2, 3) {
    foreach ($eng in $engines) {
        Say "sweep $sweep  euler  $eng  starting  $(LoadNow)"
        $t = Get-Date
        & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
            --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method heuristic `
            --free-between-runs --random-seed 42 --no-plot -o "$root/sweep$sweep/euler/$eng" `
            *>> "$root/sweep${sweep}_euler_${eng}.log"
        Say ("sweep $sweep  euler  $eng  done in {0:N1} min  exit $LASTEXITCODE" -f ((Get-Date) - $t).TotalMinutes)
    }
}
Say "=== done ==="
New-Item -ItemType File "$root/DONE.marker" -Force | Out-Null
Wait-Job $sampler -Timeout 180 | Out-Null
