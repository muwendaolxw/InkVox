@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python was not found. Install Python or fix its PATH.
    pause
    exit /b 1
  )
  python "%~dp0launch_inkvox.py"
) else (
  py "%~dp0launch_inkvox.py"
)
if errorlevel 1 (
  echo [ERROR] InkVox could not start. See the message above.
  pause
)
