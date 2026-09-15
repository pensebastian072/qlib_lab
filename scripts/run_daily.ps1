# Daily worker: refresh OHLCV + rebuild .bin store, train + gate + publish the
# qlib shadow flag, snapshot ETF flows. ASCII-only (PowerShell 5.1 cp1252).
# Called by _run_daily.vbs (hidden) from Task Scheduler QlibLabDaily.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
# The task launches from system32 -- put the package on the path so `-m` resolves.
Set-Location $repo
$env:PYTHONPATH = $repo

$runs = Join-Path $repo "journal\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$log = Join-Path $runs ("qlib_daily_" + (Get-Date -Format "yyyyMMdd") + ".log")

"[{0}] run start" -f (Get-Date -Format s) | Out-File -FilePath $log -Append -Encoding utf8
# Run through cmd so >> / 2>&1 are native (avoids PowerShell wrapping python's
# stderr in NativeCommandError records).
& cmd.exe /c "`"$py`" -m qlib_lab.data_fetch >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m qlib_lab.cot >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m qlib_lab.fred_liquidity >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m qlib_lab.funding >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m qlib_lab.pipeline >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m qlib_lab.etf_flows >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m qlib_lab.vol_desk >> `"$log`" 2>&1"
"[{0}] run done" -f (Get-Date -Format s) | Out-File -FilePath $log -Append -Encoding utf8
