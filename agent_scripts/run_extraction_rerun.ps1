# Rerun the Radix digit-extraction measurement on an idle machine.
#
# radix_extraction.py was first run on 2026-10-04 while the machine was in use
# (best of five runs per matrix). This reruns it for all four campaigns at night.
# The daytime measurements are kept in run/radix_extraction/daytime_2026-10-04/
# so the two can be compared before the tables and figures are regenerated;
# regenerating them is a separate, reviewed step.
#
# Launch:  pwsh -NoProfile -File agent_scripts\run_extraction_rerun.ps1

$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot   # the repository root, wherever it is cloned
Set-Location $Repo
$py   = (Join-Path $Repo '.venv\Scripts\python.exe')
$root = 'run/radix_extraction'
$keep = "$root/daytime_2026-10-04"
$log  = "$root/rerun.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
}
function LoadNow {
    try {
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        return "cpu_load=${cpu}% processes=$((Get-Process).Count)"
    } catch { return "load unavailable" }
}

if (-not (Test-Path $keep)) {
    New-Item -ItemType Directory -Force -Path $keep | Out-Null
    Copy-Item "$root/*.csv" $keep
    Say "daytime measurements kept in $keep"
}

Say "rerun starting  $(LoadNow)"
& $py agent_scripts/radix_extraction.py variance tm radix_scale moe *>> $log
Say "rerun finished  $(LoadNow)"
New-Item -ItemType File "$root/RERUN_DONE.marker" -Force | Out-Null
