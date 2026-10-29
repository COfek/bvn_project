# Ablation of the same-direction heuristic in Euler splitting.
#
# The paper uses --euler-split-method heuristic (same-direction split, sparser
# leaves) and never measured it against the classic alternating 2-colouring
# (--euler-split-method euler). This runs both, for the Hungarian and GW Dynamic
# leaf engines, with exactly the arguments of the Euler campaign in
# run_variance_campaign.ps1 (n=256, k=256, W_max=64, 105 matrices, seed 42,
# depths 1-4), back to back on an idle machine.
#
# The order alternates methods (H, E, E, H) so a slow drift in machine speed
# during the night falls on both methods equally rather than on one.
#
# Output: run/ablation_heuristic_2026/{heuristic,euler}/{maximum,heavy}/...
# Launch:  pwsh -NoProfile -File agent_scripts\run_heuristic_ablation.ps1

$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot   # the repository root, wherever it is cloned
Set-Location $Repo
$py   = (Join-Path $Repo '.venv\Scripts\python.exe')
$root = 'run/ablation_heuristic_2026'
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
import json, platform, sys, os
info = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'processor': platform.processor(), 'cpu_count': os.cpu_count()}
for m in ('numpy', 'scipy', 'numba', 'pandas'):
    try: info[m] = __import__(m).__version__
    except Exception as e: info[m] = f'unavailable ({e})'
print(json.dumps(info, indent=2))
"@ > "$root/environment.json" 2>&1

$runs = @(
    @('heuristic', 'maximum'), @('euler', 'maximum'),
    @('euler', 'heavy'),       @('heuristic', 'heavy')
)
Say "=== heuristic ablation starting: $($runs.Count) runs === $(LoadNow)"
$t0 = Get-Date
foreach ($r in $runs) {
    $method, $eng = $r
    $s = Get-Date
    Say "$method  $eng  starting  $(LoadNow)"
    & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
        --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method $method `
        --random-seed 42 --no-plot -o "$root/$method/$eng" `
        *>> "$root/${method}_${eng}.log"
    Say ("$method  $eng  done in {0:N1} min   {1}" -f ((Get-Date) - $s).TotalMinutes, (LoadNow))
}
Say ("=== ablation complete in {0:N1} min ===" -f ((Get-Date) - $t0).TotalMinutes)
New-Item -ItemType File "$root/ALL_DONE.marker" -Force | Out-Null
