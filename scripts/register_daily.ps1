# Register the daily QlibLabDaily Task Scheduler job. Runs as the current user
# at normal (Limited) run level -- no admin, no UAC. ASCII-only. Runs wscript ->
# _run_daily.vbs (hidden).
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily.ps1 -At 09:30
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily.ps1 -Unregister
param(
    [string]$At = "09:30",          # local time; box is on 08:00-20:00
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "QlibLabDaily"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "unregistered $taskName"
    return
}

$vbs = Join-Path $repo "scripts\_run_daily.vbs"
if (-not (Test-Path $vbs)) { throw "missing $vbs" }

$action   = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
$trigger  = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 60)

# No -Principal: runs as the registering user, Limited run level, only when
# logged on -- no elevation required.
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
Write-Output "registered '$taskName' daily at $At"
Write-Output "run now to test: Start-ScheduledTask -TaskName $taskName"
Write-Output "log: journal\runs\qlib_daily_<YYYYMMDD>.log"
