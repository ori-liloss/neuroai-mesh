@echo off
cd /d "%~dp0"
set "PY="
call :pyfile "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if errorlevel 1 if not errorlevel 2 goto waitpy
if not defined PY call :pyfile "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if errorlevel 1 if not errorlevel 2 goto waitpy
if not defined PY set "PY=python"
"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(0.4); r=s.connect_ex(('127.0.0.1',8780)); s.close(); sys.exit(0 if r==0 else 1)"
if not errorlevel 1 (
  echo The page is already open on port 8780.
  exit /b 0
)
echo Leave this window open.
"%PY%" -u mesh.py serve
if errorlevel 1 pause
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
