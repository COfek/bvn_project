# Runs the four-tier ablation (ablate_ruined_tier.py) once the overnight comparison
# (run_euler_ruined_tier_detached.ps1) has written euler_exp/ruined_tier/DONE.marker, so the
# two never share the machine. The ablation times nothing, so it may use 16 processes.
# Keeps the machine from idle-sleeping while it waits and runs.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_ablation_after_ruined_tier.ps1' -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$py = '.\.venv\Scripts\python.exe'
$out = 'euler_exp/ruined_tier'
while (-not (Test-Path "$out/DONE.marker")) { Start-Sleep -Seconds 20 }
Start-Sleep -Seconds 30
$env:PYTHONIOENCODING = 'utf-8'
& $py agent_scripts/ablate_ruined_tier.py 105 16 1 *>> "$out/ablation.log"
New-Item -ItemType File "$out/ABLATION_DONE.marker" -Force | Out-Null
