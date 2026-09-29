@echo off
REM One-step install for Qlib Lab (Windows). Double-click me.
REM Creates .venv, installs dependencies, then starts the app and opens your browser.
setlocal EnableExtensions
set "PYTHONUTF8=1"
cd /d "%~dp0"
title Qlib Lab - install
echo.
echo  Installing Qlib Lab ...
echo  (Dashboard only. For the full research stack run:  install.bat --full)
echo.
set "REQ=requirements-ui.txt"
if /i "%~1"=="--full" set "REQ=requirements.txt"

if exist ".venv\Scripts\python.exe" goto :deps
where uv >nul 2>nul
if not errorlevel 1 (
  uv venv --python 3.11 .venv
  if errorlevel 1 goto :fail
  goto :deps
)
call :findpy
if errorlevel 1 goto :nopy
%PY% -m venv .venv
if errorlevel 1 goto :fail

:deps
where uv >nul 2>nul
if not errorlevel 1 (
  uv pip install --python .venv\Scripts\python.exe -r "%REQ%"
) else (
  .venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -r "%REQ%"
)
if errorlevel 1 goto :fail
if exist ".env.example" if not exist ".env" copy /y ".env.example" ".env" >nul

echo.
echo  Installed. Next time just double-click start.bat
echo.
call "%~dp0start.bat"
exit /b 0

:findpy
py -3.11 -c "import sys" >nul 2>nul && (set "PY=py -3.11" & exit /b 0)
py -3 -c "import sys" >nul 2>nul && (set "PY=py -3" & exit /b 0)
python -c "import sys" >nul 2>nul && (set "PY=python" & exit /b 0)
exit /b 1

:nopy
echo  Python 3.11 or newer was not found.
echo  Install it from https://www.python.org/downloads/  (tick "Add python.exe to PATH"),
echo  then double-click install.bat again.
if not defined NO_BROWSER pause
exit /b 1

:fail
echo.
echo  Install failed - see the messages above.
if not defined NO_BROWSER pause
exit /b 1
