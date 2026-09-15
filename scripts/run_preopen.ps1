# Pre-open worker: publish the vol-desk SIGNAL flag before 09:30 (the options_desk
# repo reads it and sends the alert). Does NOT refetch/retrain -- reads the .bin store as of the prior
# session close (exactly what a pre-open ticket wants). ASCII-only (PS 5.1).
# Called by _run_preopen.vbs (hidden) from Task Scheduler PreOpenVolDesk.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
Set-Location $repo
$env:PYTHONPATH = $repo

$runs = Join-Path $repo "journal\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$log = Join-Path $runs ("vol_desk_preopen_" + (Get-Date -Format "yyyyMMdd") + ".log")

# The child writes this same file through cmd's >> redirect, and Norton transiently
# locks freshly-written files on this box, so a bare Out-File loses the handle race
# and throws IOException. That silently dropped the final "done" line on the first
# real run of this script -- the exact line the exit-code work exists to produce.
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

# Hang-breaker. On 2026-08-11 C: (the SATA spinner) went to 0% idle with a 17-25 deep
# queue and 0.16-1.2 s/read; this run took 55 min instead of the usual 20 and landed its
# flag at 09:51, long after the 09:05 desk read and the 09:15 catch-up. Nothing failed --
# and that was the problem: cmd /c blocks forever, so a truly stuck run would sit until
# the next reboot while the task still reported LastTaskResult 0. 90 min is deliberately
# far above a bad-disk-day run (55 min): this breaks a HANG, it does not enforce the
# pre-open deadline. OVERRUN_MIN only warns.
$TIMEOUT_MIN = 90
$OVERRUN_MIN = 20

$flag = Join-Path $repo "journal\flags\vol_desk_state.json"
$flagBefore = $null
if (Test-Path $flag) { $flagBefore = (Get-Item $flag).LastWriteTimeUtc }

$started = Get-Date
Write-Log "pre-open vol desk start"

# Flag only -- the options_desk repo (OptionsDeskDaily ~09:05) owns the Telegram alert
# now, reading this flag. cmd so >> / 2>&1 are native (avoids PS stderr wrapping).
# python -u: stdout through >> is block-buffered, so a buffered run shows ONLY this
# start line for its whole life and you cannot tell "alive and slow" from "dead" by
# reading the log. -u makes the progress lines land as they happen.
$inner = '""{0}" -u -m qlib_lab.vol_desk >> "{1}" 2>&1"' -f $py, $log
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $inner" -PassThru -WindowStyle Hidden

$exit = 0
if (-not $proc.WaitForExit($TIMEOUT_MIN * 60 * 1000)) {
    # Kill the TREE, not just cmd.exe: the venv python is a uv trampoline shim that
    # spawns the real interpreter as a child, so killing the parent alone orphans it.
    & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null
    Write-Log ("TIMEOUT after {0} min - killed process tree (pid {1})" -f $TIMEOUT_MIN, $proc.Id)
    $exit = 1
} else {
    # The bounded WaitForExit can return before ExitCode is populated on a -PassThru
    # object; the parameterless call flushes it. Treat a still-null code as failure
    # rather than silently reporting success.
    $proc.WaitForExit()
    if ($proc.ExitCode -eq $null) { $exit = 3 } else { $exit = $proc.ExitCode }
}

$elapsed = [int]((Get-Date) - $started).TotalMinutes
if ($exit -eq 0 -and $elapsed -ge $OVERRUN_MIN) {
    Write-Log ("WARNING: ran {0} min (overrun bar {1}); check C: disk latency - the 09:05 desk read may have used a STALE flag" -f $elapsed, $OVERRUN_MIN)
}

# The exit code must mean "is there a fresh flag", not "did a process end". A run that
# exits 0 without republishing leaves the desk reading yesterday, which is exactly the
# failure this file is meant to make visible in Task Scheduler.
if ($exit -eq 0) {
    $fresh = $false
    if (Test-Path $flag) {
        $after = (Get-Item $flag).LastWriteTimeUtc
        $fresh = ($flagBefore -eq $null) -or ($after -gt $flagBefore)
    }
    if (-not $fresh) {
        Write-Log "ERROR: exited 0 but vol_desk_state.json did not refresh"
        $exit = 2
    }
}

Write-Log ("pre-open vol desk done (exit {0}, {1} min)" -f $exit, $elapsed)
exit $exit
