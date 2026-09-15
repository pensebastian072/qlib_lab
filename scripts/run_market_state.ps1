# Market State worker: rebuild journal\exports\market_state.xlsx from the qlib store
# and the three history parquets. DESCRIPTIVE ONLY - measures state, forecasts nothing.
# ASCII-only (PS 5.1). Called by _run_market_state.vbs from Task Scheduler MarketStateDaily.
#
# Structure and every defensive trick below are lifted from run_preopen.ps1 - see this
# repo's CLAUDE.md for why each exists (Norton file locks, uv trampoline grandchildren,
# block-buffered stdout, and a C: spinner that turns a 2 min run into 55 min).
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
Set-Location $repo
$env:PYTHONPATH = $repo

$runs = Join-Path $repo "journal\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$log = Join-Path $runs ("market_state_" + (Get-Date -Format "yyyyMMdd") + ".log")

# Norton transiently locks freshly-written files on this box, and the child holds this
# same file open through cmd's >> redirect, so a bare Out-File loses the handle race.
function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format s), $msg
    for ($i = 0; $i -lt 10; $i++) {
        try {
            [IO.File]::AppendAllText($log, $line + "`r`n", (New-Object Text.UTF8Encoding($false)))
            return
        } catch {
            Start-Sleep -Milliseconds 300
        }
    }
    Write-Output $line
}

# Breaks a HANG, not a slow day. A healthy run is ~1-3 min; 60 leaves room for the
# spinner's bad-disk behaviour without letting a wedged run sit until the next reboot
# while Task Scheduler still reports LastTaskResult 0.
$TIMEOUT_MIN = 60
$OVERRUN_MIN = 15

$xlsx = Join-Path $repo "journal\exports\market_state.xlsx"
$before = $null
if (Test-Path $xlsx) { $before = (Get-Item $xlsx).LastWriteTimeUtc }

$started = Get-Date
Write-Log "market state build start"

# cmd so >> / 2>&1 are native (avoids PS stderr wrapping into ErrorRecords).
# python -u so progress lands as it happens instead of at process exit.
$inner = '""{0}" -u "{1}\scripts\build_market_state.py" >> "{2}" 2>&1"' -f $py, $repo, $log
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $inner" -PassThru -WindowStyle Hidden

$exit = 0
if (-not $proc.WaitForExit($TIMEOUT_MIN * 60 * 1000)) {
    # Kill the TREE: the venv python is a uv trampoline whose real interpreter is a
    # grandchild, so killing cmd alone orphans it.
    & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null
    Write-Log ("TIMEOUT after {0} min - killed process tree (pid {1})" -f $TIMEOUT_MIN, $proc.Id)
    $exit = 1
} else {
    # A bounded WaitForExit can return before ExitCode is populated on a -PassThru
    # object; the parameterless call flushes it.
    $proc.WaitForExit()
    if ($proc.ExitCode -eq $null) { $exit = 3 } else { $exit = $proc.ExitCode }
}

$elapsed = [int]((Get-Date) - $started).TotalMinutes
if ($exit -eq 0 -and $elapsed -ge $OVERRUN_MIN) {
    Write-Log ("WARNING: ran {0} min (overrun bar {1}); check C: disk latency" -f $elapsed, $OVERRUN_MIN)
}

# Exit code must mean "is there a fresh workbook", not "did a process end". A run that
# exits 0 without rewriting the file leaves yesterday's state on screen looking current.
if ($exit -eq 0) {
    $fresh = $false
    if (Test-Path $xlsx) {
        $after = (Get-Item $xlsx).LastWriteTimeUtc
        $fresh = ($before -eq $null) -or ($after -gt $before)
    }
    if (-not $fresh) {
        # Excel holds an exclusive lock on an open workbook, so a run while the user
        # has it open cannot replace it and writes market_state.pending.xlsx instead.
        # That is a completed run with a degraded delivery, not a failed one -- warn,
        # do not fail the task, or every day the file is left open reports red.
        $pending = Join-Path $repo "journal\exports\market_state.pending.xlsx"
        $pendingFresh = $false
        if (Test-Path $pending) {
            $pendingFresh = ((Get-Item $pending).LastWriteTimeUtc -gt $started.ToUniversalTime())
        }
        if ($pendingFresh) {
            Write-Log "WARNING: market_state.xlsx is locked (open in Excel?) - wrote market_state.pending.xlsx; close the workbook and re-run to refresh in place"
        } else {
            Write-Log "ERROR: exited 0 but market_state.xlsx did not refresh"
            $exit = 2
        }
    }
}

Write-Log ("market state build done (exit {0}, {1} min)" -f $exit, $elapsed)
exit $exit
