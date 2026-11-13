# Euler campaigns on all 1000 matrices (2026-11), so that the Euler tables rest on the same 995
# matrices as the Radix ones. Same command as run_rerun_2026_11b.ps1 / run_wwfa_campaign.ps1
# (n=256, k=256, max weight 64, depths 1-4, heuristic_ruined, seed 42), three sweeps, engines
# maximum, heavy, heavy_static, wwfa.
#
# It runs only while the computer is free, over as many nights as it takes:
#   * Work is cut into blocks of 50 matrices. Matrix i is always seed 42 + i, so a block that
#     starts at matrix a runs --random-seed 42+a; it first decomposes 5 extra matrices that are
#     thrown away (the warm-up the paper discards), so every kept matrix is timed warm.
#     Block a keeps matrices a+5 .. a+54; together the blocks keep 5 .. 999, as the Radix campaign.
#   * A block starts only between 03:00 and 08:45, and only when other programs have used less
#     than 3% of the CPU over the last minute. Otherwise the launcher waits and checks again.
#   * While a block runs, the CPU used by every other program is sampled every 15 s. If its mean
#     over the block exceeds 3% of the machine, or one sample exceeds 15%, the block is marked
#     disturbed: its folder is renamed *_disturbed_<time> (kept as evidence, never merged) and the
#     block is run again later. Keyboard and mouse input are not checked (Ofek, 2026-11-12).
#   * Finished blocks are listed in blocks_done.txt, so a relaunch resumes where it stopped.
#
# Output: run/euler_1000_2026_11/sweep{1,2,3}/euler/<engine>/block_<a>/<stamp>/results_stats.csv,
#         blocks_done.txt, campaign.log, environment_*.json, DONE.marker.
# Merge afterwards: .venv/Scripts/python.exe agent_scripts/merge_euler_blocks.py
# Launch: Start-Process pwsh -ArgumentList '-NoProfile','-File','agent_scripts\run_euler_1000.ps1' -WindowStyle Hidden
param([string]$Root = 'run/euler_1000_2026_11', [int]$Block = 50, [int]$Warm = 5,
      [int]$Total = 1000, [string]$From = '03:00', [string]$Until = '08:45',
      [double]$StartMax = 3.0, [double]$MeanMax = 3.0, [double]$PeakMax = 15.0,
      [int[]]$Sweeps = @(1, 2, 3), [string[]]$Engines = @('maximum', 'heavy', 'heavy_static', 'wwfa'))
$ErrorActionPreference = 'Continue'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001u) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$py = (Join-Path $Repo '.venv\Scripts\python.exe')
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$master = "$Root/campaign.log"
$doneFile = "$Root/blocks_done.txt"
if (-not (Test-Path $doneFile)) { New-Item -ItemType File $doneFile | Out-Null }
$cores = [Environment]::ProcessorCount

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $master -Value $line
}

function CpuTimes {
    $t = @{}
    foreach ($p in Get-Process) {
        try { if ($p.Id -ne 0) { $t[$p.Id] = $p.TotalProcessorTime.TotalSeconds } } catch {}
    }
    return $t
}

function OthersPercent($before, $after, $seconds, $exclude) {
    # CPU used by every process except $exclude, as a share of the whole machine
    $sum = 0.0
    foreach ($id in $after.Keys) {
        if ($exclude -contains $id) { continue }
        $b = if ($before.ContainsKey($id)) { $before[$id] } else { 0.0 }
        $d = $after[$id] - $b
        if ($d -gt 0) { $sum += $d }
    }
    return 100.0 * $sum / ($seconds * $cores)
}

function ProcessTree($root) {
    # $root and every process descended from it. The venv's python.exe is a launcher that
    # starts the real interpreter as a child, so the run must be excluded as a whole tree.
    $kids = @{}
    foreach ($w in Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId) {
        $pp = [int]$w.ParentProcessId
        if (-not $kids.ContainsKey($pp)) { $kids[$pp] = @() }
        $kids[$pp] += [int]$w.ProcessId
    }
    $tree = @($root); $i = 0
    while ($i -lt $tree.Count) {
        $c = $tree[$i]; $i++
        if ($kids.ContainsKey($c)) { foreach ($k in $kids[$c]) { if ($tree -notcontains $k) { $tree += $k } } }
    }
    return $tree
}

function InWindow {
    # <Root>/window_overrides.txt, read at every check, may replace the window for one date:
    # lines "yyyy-MM-dd HH:mm HH:mm" (date, first start, last start); "#" starts a comment.
    $now = Get-Date
    $f, $u = $From, $Until
    $ov = "$Root/window_overrides.txt"
    if (Test-Path $ov) {
        foreach ($l in Get-Content $ov) {
            $w = ($l -replace '#.*$', '').Trim() -split '\s+'
            if ($w.Count -eq 3 -and $w[0] -eq $now.ToString('yyyy-MM-dd')) { $f, $u = $w[1], $w[2] }
        }
    }
    return ($now.TimeOfDay -ge [timespan]::Parse($f)) -and ($now.TimeOfDay -le [timespan]::Parse($u))
}

function WaitUntilFree {
    while ($true) {
        if (-not (InWindow)) { Start-Sleep -Seconds 300; continue }
        $a = CpuTimes; Start-Sleep -Seconds 60; $b = CpuTimes
        $o = OthersPercent $a $b 60 @($PID)
        if ($o -lt $StartMax) { return $o }
        Say ("waiting: other programs at {0:N1}% of the CPU" -f $o)
        Start-Sleep -Seconds 240
    }
}

function RecordEnvironment {
    $envFile = "$Root/environment_$(Get-Date -Format 'yyyyMMdd_HHmm').json"
    & $py -c @"
import json, platform, sys, os, subprocess
git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
info = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'processor': platform.processor(), 'cpu_count': os.cpu_count(),
        'commit': git('rev-parse', '--short', 'HEAD'),
        'dirty_files': [l for l in git('status', '--porcelain', '--', 'src', 'main.py').splitlines()]}
for m in ('numpy', 'scipy', 'numba', 'pandas'):
    try:
        info[m] = __import__(m).__version__
    except Exception as e:
        info[m] = f'unavailable ({e})'
print(json.dumps(info, indent=2))
"@ > $envFile 2>&1
}

# "-Engines maximum,wwfa" from another shell arrives as one string: split it
$engines = @($Engines | ForEach-Object { $_ -split ',' } | Where-Object { $_ } | ForEach-Object { $_.Trim() })
$fails = @{}
$jobs = foreach ($s in $Sweeps) { foreach ($e in $engines) { for ($a = 0; $a -lt ($Total - $Warm); $a += $Block) {
    [pscustomobject]@{ Sweep = $s; Engine = $e; Start = $a } } } }
Say "launcher started, $($jobs.Count) blocks in all, $((Get-Content $doneFile).Count) already done"
$lastEnvDay = ''

foreach ($j in $jobs) {
    $key = "$($j.Sweep) $($j.Engine) $($j.Start)"
    while (-not (Select-String -Path $doneFile -Pattern "^$([regex]::Escape($key))$" -Quiet)) {
        $free = WaitUntilFree
        $day = Get-Date -Format 'yyyyMMdd'
        if ($day -ne $lastEnvDay) { RecordEnvironment; $lastEnvDay = $day }
        $n = [Math]::Min($Block + $Warm, $Total - $j.Start)
        $out = "$Root/sweep$($j.Sweep)/euler/$($j.Engine)/block_$($j.Start)"
        if (Test-Path $out) { Remove-Item -Recurse -Force $out }   # a block cut short last time
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        $log = "$out.log"
        Say ("block {0} starting ({1} matrices from seed {2}), other programs {3:N1}%" -f $key, $n, (42 + $j.Start), $free)
        $argList = @('main.py', '-n', '256', '-k', '256', '--max-weight', '64', '-s', "$n",
                     '--engine', 'euler_bvn', '--euler-leaf-engine', $j.Engine,
                     '--euler-depths', '1', '2', '3', '4', '--euler-split-method', 'heuristic_ruined',
                     '--random-seed', "$(42 + $j.Start)", '--no-plot', '-o', $out)
        $t0 = Get-Date
        $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
                -RedirectStandardOutput $log -RedirectStandardError "$out.err.log"
        $samples = @(); $prev = CpuTimes; $tp = Get-Date
        while (-not $p.HasExited) {
            Start-Sleep -Seconds 15
            $now = CpuTimes; $tn = Get-Date
            $samples += OthersPercent $prev $now (($tn - $tp).TotalSeconds) (@($PID) + (ProcessTree $p.Id))
            $prev = $now; $tp = $tn
        }
        $p.WaitForExit()
        $mean = if ($samples.Count) { ($samples | Measure-Object -Average).Average } else { 0 }
        $peak = if ($samples.Count) { ($samples | Measure-Object -Maximum).Maximum } else { 0 }
        $mins = ((Get-Date) - $t0).TotalMinutes
        $csv = Get-ChildItem -Path $out -Recurse -Filter results_stats.csv -ErrorAction SilentlyContinue
        if ($p.ExitCode -ne 0 -or -not $csv) {
            $fails[$key] = 1 + $(if ($fails.ContainsKey($key)) { $fails[$key] } else { 0 })
            Move-Item $out "$out`_failed_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
            if ($fails[$key] -ge 3) {
                Say ("block {0} FAILED 3 times (exit {1}), skipped; see {2}.err.log" -f $key, $p.ExitCode, $out)
                break
            }
            Say ("block {0} FAILED (exit {1}) after {2:N1} min, will retry" -f $key, $p.ExitCode, $mins)
            Start-Sleep -Seconds 60
        } elseif ($mean -gt $MeanMax -or $peak -gt $PeakMax) {
            Say ("block {0} DISTURBED (other programs mean {1:N1}%, peak {2:N1}%) after {3:N1} min, discarded" -f $key, $mean, $peak, $mins)
            Move-Item $out "$out`_disturbed_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        } else {
            Add-Content -Path $doneFile -Value $key
            Say ("block {0} done in {1:N1} min (other programs mean {2:N1}%, peak {3:N1}%)" -f $key, $mins, $mean, $peak)
        }
    }
}
Say "=== all blocks done ==="
New-Item -ItemType File "$Root/DONE.marker" -Force | Out-Null
[Win32.Power]::SetThreadExecutionState(0x80000000u) | Out-Null
