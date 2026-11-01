# Samples the memory of every python process (private bytes above 50 MiB) and the machine's
# free physical memory every 120 s, appending to euler_exp/ruined_tier/memtrace_overnight.csv,
# until euler_exp/ruined_tier/DONE.marker exists. This is the sampler behind that file on
# 2026-11-01 (it ran there as a Git Bash loop calling the same PowerShell command, from 03:22).
# Columns: time, pid, process start, CPU seconds, working set and private bytes in MiB
# (PowerShell's MB = 2^20 bytes), free physical memory in MiB
# (Win32_OperatingSystem.FreePhysicalMemory, KiB / 1024). The machine has 31.2 GiB
# (Win32_ComputerSystem.TotalPhysicalMemory = 33,449,762,816 bytes).
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\sample_memory.ps1' -WindowStyle Hidden
$Repo = Split-Path -Parent $PSScriptRoot
$out = Join-Path $Repo 'euler_exp/ruined_tier'
$log = Join-Path $out 'memtrace_overnight.csv'
if (-not (Test-Path $log)) { 'time,pid,start,cpu_s,ws_mb,priv_mb,free_mb' | Set-Content $log }
while (-not (Test-Path (Join-Path $out 'DONE.marker'))) {
    $f = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
    Get-Process python -ErrorAction SilentlyContinue |
        Where-Object { $_.PrivateMemorySize64 -gt 50MB } |
        ForEach-Object {
            '{0},{1},{2},{3},{4},{5},{6}' -f (Get-Date -Format s), $_.Id, $_.StartTime.ToString('s'),
                [int]$_.CPU, [int]($_.WorkingSet64 / 1MB), [int]($_.PrivateMemorySize64 / 1MB), $f
        } | Add-Content $log
    Start-Sleep -Seconds 120
}
