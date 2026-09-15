' Launch the daily qlib run HIDDEN (window style 0) via python.exe through the
' PowerShell worker. Deliberately NOT pythonw.exe -- Norton 360 blocks creating
' pythonw.exe in a venv Scripts dir on this box (see norton-blocks-pythonw-venv).
Set sh = CreateObject("WScript.Shell")
dir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
ps1 = dir & "run_daily.ps1"
sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & ps1 & """", 0, False
