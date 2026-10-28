@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
  exit /b
)
echo Please run setup_environment.cmd first, then launch.cmd.
pause
exit /b 1
