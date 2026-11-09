@echo off
setlocal
cd /d "%~dp0"
if /I "%~1"=="tools" (
  echo LIST GREP READ WRITE MKDIR RUN SESSIONS
  exit /b 0
)
set "PY="
call :pyfile "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if errorlevel 1 if not errorlevel 2 goto waitpy
if not defined PY call :pyfile "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if errorlevel 1 if not errorlevel 2 goto waitpy
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
:waitpy
echo Python could not be read. This launch waits.
exit /b 1
:pyfile
set "PYCAND=%~1"
powershell -NoProfile -Command "try { $i = Get-Item -LiteralPath $env:PYCAND -Force -ErrorAction Stop; if ($i.PSIsContainer) { exit 2 }; exit 0 } catch [System.Management.Automation.ItemNotFoundException] { exit 2 } catch { exit 1 }"
if errorlevel 2 exit /b 2
if errorlevel 1 exit /b 1
set "PY=%~1"
exit /b 0
