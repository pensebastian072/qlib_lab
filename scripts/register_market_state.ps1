# Register the MarketStateDaily Task Scheduler job -- rebuilds the Market State
# workbook (journal\exports\market_state.xlsx). Runs as the current user at normal
# (Limited) run level -- no admin, no UAC. ASCII-only. Runs wscript ->
# _run_market_state.vbs (hidden).
#
# Default 11:45 local: AFTER QlibLabDaily (11:15) so the flags and parquets it reads
# are already fresh, and well clear of the 06:00-09:30 window where eight tasks
# contend for the C: spinner.
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_market_state.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_market_state.ps1 -At 12:30
#   powershell -ExecutionPolicy Bypass -File scripts\register_market_state.ps1 -Unregister
param(
    [string]$At = "11:45",
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "MarketStateDaily"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "unregistered $taskName"
    return
}

$vbs = Join-Path $repo "scripts\_run_market_state.vbs"
if (-not (Test-Path $vbs)) { throw "missing $vbs" }

$action   = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
# Weekdays only -- the underlying tape does not move on Sat/Sun.
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 70)

# No -Principal: runs as the registering user, Limited run level, only when
# logged on -- no elevation required.
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
Write-Output "registered '$taskName' weekdays at $At"
Write-Output "run now to test: Start-ScheduledTask -TaskName $taskName"
Write-Output "log: journal\runs\market_state_<YYYYMMDD>.log"
Write-Output "output: journal\exports\market_state.xlsx"
Write-Output "exit codes: 0 ok | 1 timeout | 2 exited 0 but workbook did not refresh | 3 unknown"
