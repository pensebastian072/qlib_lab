# Register the PreOpenVolDesk Task Scheduler job -- publishes + Telegram-pushes
# the vol-desk ticket before the 09:30 open. Runs as the current user at normal
# (Limited) run level -- no admin, no UAC. ASCII-only. Runs wscript ->
# _run_preopen.vbs (hidden). Default 08:55 local (box is on 08:00-20:00).
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_preopen.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_preopen.ps1 -At 08:45
#   powershell -ExecutionPolicy Bypass -File scripts\register_preopen.ps1 -Unregister
param(
    [string]$At = "08:55",
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "PreOpenVolDesk"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "unregistered $taskName"
    return
}

$vbs = Join-Path $repo "scripts\_run_preopen.vbs"
if (-not (Test-Path $vbs)) { throw "missing $vbs" }

$action   = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
# Weekdays only -- no market open on Sat/Sun.
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

# No -Principal: runs as the registering user, Limited run level, only when
# logged on -- no elevation required.
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
Write-Output "registered '$taskName' weekdays at $At"
Write-Output "run now to test: Start-ScheduledTask -TaskName $taskName"
Write-Output "log: journal\runs\vol_desk_preopen_<YYYYMMDD>.log"
Write-Output "NOTE: needs secrets\telegram.json {bot_token,chat_id} for the push."
