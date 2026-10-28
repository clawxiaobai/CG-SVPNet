@echo off
setlocal
cd /d "%~dp0"
echo Creates a project-local .venv. Existing global Python packages are not changed.
python --version
if errorlevel 1 goto fail
python -c "import sys; assert sys.version_info[:2] == (3, 13), 'Please use Python 3.13 x64 for the tested dependency lock.'"
if errorlevel 1 goto fail
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto fail
echo Installation complete. Double-click launch.cmd.
pause
exit /b 0
:fail
echo Setup failed. Read the error above. Python 3.13 x64 with Tk is required.
pause
exit /b 1
