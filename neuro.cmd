@echo off
setlocal
cd /d "%~dp0"
if /I "%~1"=="tools" (
  echo LIST GREP READ WRITE MKDIR RUN SESSIONS
  exit /b 0
)
set "PY="
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" set "PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if defined PY (
  "%PY%" -u mesh.py code %*
  exit /b %errorlevel%
)
where py >nul 2>&1
if %errorlevel%==0 (
  py -3 -u mesh.py code %*
  exit /b %errorlevel%
)
python -u mesh.py code %*
