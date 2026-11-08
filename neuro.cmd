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
set "AGENT=%~dp0..\neuro-agents\agents.py"
if not exist "%AGENT%" set "AGENT=%~dp0mesh.py"
if defined PY goto usepy
where py >nul 2>&1
if %errorlevel%==0 goto usepylauncher
python -u "%AGENT%" code %*
exit /b %errorlevel%
:usepy
"%PY%" -u "%AGENT%" code %*
exit /b %errorlevel%
:usepylauncher
py -3 -u "%AGENT%" code %*
exit /b %errorlevel%
