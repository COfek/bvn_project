# Times the three Euler split rules as full campaigns with every earlier result freed before
# each timed call (--free-between-runs), on the paper's Euler workload (n=256 k=256
# max_weight=64, 105 matrices from seed 42, depths 1-4, the four leaf engines):
#   heuristic              three tiers: consistent, empty, opposite (the paper's)
#   heuristic_ruined       four tiers: consistent, empty, non consistent, opposite
#   heuristic_ruined_last  non-consistent pairs taken last at their lock side (b_last)
# Two sweeps, the rule order rotated between them, so the run-to-run spread is measured.
# 24 campaigns, about 5 h. Waits until the start time (default 01:00) so the machine is
# free until then; keeps it from idle-sleeping while it waits and runs; samples memory.
# Output: euler_exp/tier_rules_freed/sweep{1,2}/eng_<engine>_<rule>/, progress.log,
# memtrace.csv, DONE.marker.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_tier_rules_freed_detached.ps1' -WindowStyle Hidden
param([string]$StartAt = '01:00')
$ErrorActionPreference = 'Continue'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001u) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$py = '.\.venv\Scripts\python.exe'
$out = 'euler_exp/tier_rules_freed'
New-Item -ItemType Directory $out -Force | Out-Null

$start = [datetime]::Today.Add([timespan]::Parse($StartAt))
if ($start -lt (Get-Date).AddMinutes(-1)) { $start = $start.AddDays(1) }
Add-Content "$out/progress.log" "$(Get-Date -Format s) waiting until $($start.ToString('s'))"
while ((Get-Date) -lt $start) { Start-Sleep -Seconds 30 }

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
} -ArgumentList (Resolve-Path $out).Path

$rules = @('heuristic', 'heuristic_ruined', 'heuristic_ruined_last')
$engines = @('maximum', 'heavy', 'heavy_static', 'wfa')
foreach ($s in 1, 2) {
    $order = if ($s -eq 1) { $rules } else { @($rules[2], $rules[0], $rules[1]) }
    foreach ($eng in $engines) {
        foreach ($m in $order) {
            & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
                --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method $m `
                --free-between-runs --random-seed 42 --no-plot `
                -o "$out/sweep$s/eng_${eng}_$m" *>> "$out/campaign_sweep${s}_${eng}_$m.log"
            Add-Content "$out/progress.log" "$(Get-Date -Format s) sweep $s $eng $m done"
        }
    }
}
New-Item -ItemType File "$out/DONE.marker" -Force | Out-Null
Wait-Job $sampler -Timeout 180 | Out-Null
