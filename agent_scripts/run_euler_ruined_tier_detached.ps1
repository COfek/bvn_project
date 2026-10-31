# Compares the Euler split heuristics "heuristic" (three tiers, the paper's) and
# "heuristic_ruined" (a fourth tier for already-ruined pairs) on the canonical Euler
# campaign's matrices (n=256 k=256 max_weight=64, 105 matrices from seed 42).
#   1. the paired comparison: both methods back to back on every matrix (~85 min)
#   2. both methods as full campaigns, same parameters as eng_*_dense_d4 (~2 h)
#   3. the paired comparison again, to see whether its timing differences repeat (~85 min)
# Detached so it survives session teardown.
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_euler_ruined_tier_detached.ps1' -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot   # the repository root, wherever it is cloned
Set-Location $Repo
$py = '.\.venv\Scripts\python.exe'
$out = 'euler_exp/ruined_tier'
New-Item -ItemType Directory $out -Force | Out-Null

& $py agent_scripts/euler_ruined_tier_compare.py 105 *>> "$out/paired.log"

$engines = @('maximum','heavy','heavy_static','wfa')
for ($e = 0; $e -lt $engines.Count; $e++) {
    $eng = $engines[$e]
    $methods = if ($e % 2 -eq 0) { @('heuristic','heuristic_ruined') } else { @('heuristic_ruined','heuristic') }
    foreach ($m in $methods) {
        & $py main.py -n 256 -k 256 --max-weight 64 -s 105 --engine euler_bvn `
            --euler-leaf-engine $eng --euler-depths 1 2 3 4 --euler-split-method $m `
            --random-seed 42 --no-plot -o "$out/eng_${eng}_dense_d4_$m" `
            *>> "$out/campaign_${eng}_$m.log"
        Add-Content "$out/progress.log" "$(Get-Date -Format s) campaign $eng $m done"
    }
}

& $py agent_scripts/euler_ruined_tier_compare.py 105 _rep2 *>> "$out/paired_rep2.log"
New-Item -ItemType File "$out/DONE.marker" -Force | Out-Null
